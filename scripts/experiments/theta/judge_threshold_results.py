import argparse
import ast
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from instructions.load_instructions import load_category, resolve_items


def _parse_runs(results_txt: Path) -> List[Dict[str, Any]]:
    blocks = [block.strip() for block in results_txt.read_text().split("---\n") if block.strip()]
    rows: List[Dict[str, Any]] = []
    for block in blocks:
        record: Dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("theta="):
                parts = line.split()
                for part in parts:
                    key, value = part.split("=", 1)
                    if key == "theta":
                        record[key] = float(value)
                    elif key == "trial":
                        record[key] = int(value)
                    elif key == "success":
                        record[key] = int(value)
                    else:
                        record[key] = value
            elif line.startswith("instruction: "):
                record["instruction"] = line[len("instruction: "):]
            elif line.startswith("urp_action: "):
                record["urp_action"] = line[len("urp_action: "):]
            elif line.startswith("planner_steps: "):
                record["planner_steps"] = ast.literal_eval(line[len("planner_steps: "):])
            elif line.startswith("parsed_actions: "):
                record["parsed_actions"] = ast.literal_eval(line[len("parsed_actions: "):])
        rows.append(record)
    return rows


def _match_sequences(pred: List[str], gold_sequences: List[List[str]], order_matters: bool) -> bool:
    pred_clean = [step for step in pred if step]
    for seq in gold_sequences:
        seq_clean = [step for step in seq if step]
        if order_matters:
            if pred_clean == seq_clean:
                return True
        else:
            if set(pred_clean) == set(seq_clean):
                return True
    return False


def _match_rules(pred: List[str], rules: Dict[str, List[str]]) -> bool:
    pred_map: Dict[str, set[str]] = {}
    for act in pred:
        if not act.startswith("robot.pick_and_place("):
            continue
        inner = act[len("robot.pick_and_place("):-1]
        try:
            obj, dest = [s.strip() for s in inner.split(",")]
        except Exception:
            continue
        pred_map.setdefault(dest, set()).add(obj)
    for dest, objs in rules.items():
        if dest not in pred_map:
            return False
        for obj in objs:
            if obj not in pred_map[dest]:
                return False
    return True


def _score_with_gold(item: Dict[str, Any], pred_actions: List[str], policy_meta: Dict[str, Any]) -> Optional[int]:
    gold = item.get("gold")
    if not gold:
        return None
    order_matters = policy_meta.get("order_matters", True)
    if "sequences" in gold:
        return 1 if _match_sequences(pred_actions, gold["sequences"], order_matters) else 0
    if "rules" in gold:
        return 1 if _match_rules(pred_actions, gold["rules"]) else 0
    return None


def _is_canonical_plan(actions: List[str]) -> bool:
    if not actions:
        return False
    return all(act == "done()" or act.startswith("robot.pick_and_place(") for act in actions)


def _judge_batch(
    theta: float,
    category: str,
    order_matters: bool,
    batch_rows: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Dict[str, Any]]:
    from scripts.modules.semantic_cache import get_openai_client, log_openai_call

    system = (
        "You are a strict evaluator for robot task plans.\n"
        "You must compare predicted plans against gold policies.\n"
        "Return score 1 only if the predicted plan satisfies at least one gold policy.\n"
        "If order_matters is false, treat action order as irrelevant.\n"
        "Do not give partial credit.\n"
        "Return JSON only."
    )
    payload = {
        "category": category,
        "theta": theta,
        "order_matters": order_matters,
        "items": [
            {
                "id": row["id"],
                "instruction": row["instruction"],
                "gold": row["gold"],
                "predicted": row["predicted"],
            }
            for row in batch_rows
        ],
        "response_schema": {
            "results": [
                {
                    "id": "item id",
                    "score": "0 or 1",
                    "reason_short": "very short rationale",
                }
            ]
        },
    }

    client = get_openai_client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    log_openai_call("judge_threshold_results", f"{category}:{theta}", time.monotonic() - start)
    content = response.choices[0].message.content.strip()
    data = json.loads(content)
    results = {}
    for item in data.get("results", []):
        item_id = item.get("id")
        if item_id:
            results[item_id] = {
                "score": 1 if int(item.get("score", 0)) == 1 else 0,
                "reason_short": item.get("reason_short", ""),
            }
    return results


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_txt")
    parser.add_argument("--category", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    results_txt = Path(args.results_txt)
    out_base = Path(args.out) if args.out else results_txt.with_name(results_txt.stem + "_judged")

    category = load_category(args.category)
    items = {item["id"]: item for item in resolve_items(category)}
    policy_meta = category.get("policy", {})
    order_matters = policy_meta.get("order_matters", True)

    parsed_rows = _parse_runs(results_txt)
    _log(
        f"[judge] loaded {len(parsed_rows)} run entries from {results_txt} "
        f"for category={args.category}",
        verbose,
    )
    item_rows: List[Dict[str, Any]] = []
    pending_by_theta: Dict[float, List[Dict[str, Any]]] = {}
    deterministic_counts: Dict[float, int] = {}

    for row in parsed_rows:
        item = items[row["id"]]
        predicted = row.get("planner_steps") or row.get("parsed_actions") or []
        deterministic_score = None
        judge_mode = "llm_batch"
        reason_short = ""

        if _is_canonical_plan(predicted):
            deterministic_score = _score_with_gold(item, predicted, policy_meta)
            if deterministic_score is not None:
                judge_mode = "deterministic"
                reason_short = "Matches gold policy exactly." if deterministic_score == 1 else "Canonical plan does not match gold policy."

        base = {
            "theta": row["theta"],
            "category": row["category"],
            "id": row["id"],
            "instruction": row["instruction"],
            "predicted": predicted,
            "gold": item.get("gold"),
            "judge_mode": judge_mode,
            "score": deterministic_score,
            "reason_short": reason_short,
        }
        item_rows.append(base)
        if deterministic_score is None:
            pending_by_theta.setdefault(row["theta"], []).append(base)
        else:
            deterministic_counts[row["theta"]] = deterministic_counts.get(row["theta"], 0) + 1

    for theta in sorted({row["theta"] for row in item_rows}):
        total = sum(1 for row in item_rows if row["theta"] == theta)
        deterministic = deterministic_counts.get(theta, 0)
        pending = len(pending_by_theta.get(theta, []))
        _log(
            f"[judge] theta={theta}: total_items={total} "
            f"deterministic={deterministic} llm_batch_pending={pending}",
            verbose,
        )

    for theta, pending_rows in pending_by_theta.items():
        for start in range(0, len(pending_rows), args.batch_size):
            batch = pending_rows[start:start + args.batch_size]
            batch_ids = ",".join(row["id"] for row in batch)
            _log(
                f"[judge] theta={theta}: sending batch size={len(batch)} ids={batch_ids}",
                verbose,
            )
            judged = _judge_batch(theta, args.category, order_matters, batch, args.model)
            for row in batch:
                verdict = judged.get(row["id"], {"score": 0, "reason_short": "Missing evaluator verdict."})
                row["score"] = verdict["score"]
                row["reason_short"] = verdict["reason_short"]
                _log(
                    f"[judge] theta={theta} id={row['id']} mode=llm_batch "
                    f"score={row['score']} reason={row['reason_short']}",
                    verbose,
                )

    grouped: Dict[float, List[Dict[str, Any]]] = {}
    for row in item_rows:
        grouped.setdefault(row["theta"], []).append(row)

    aggregate = []
    for theta, rows in sorted(grouped.items()):
        scores = [int(row["score"]) for row in rows if row["score"] is not None]
        success_rate = sum(scores) / len(scores) if scores else 0.0
        aggregate.append(
            {
                "theta": theta,
                "category": args.category,
                "num_items": len(rows),
                "success_rate": success_rate,
                "num_deterministic": sum(1 for row in rows if row["judge_mode"] == "deterministic"),
                "num_llm_batch": sum(1 for row in rows if row["judge_mode"] == "llm_batch"),
            }
        )
        _log(
            f"[judge] theta={theta}: success_rate={success_rate:.4f} "
            f"num_items={len(rows)} deterministic={sum(1 for row in rows if row['judge_mode'] == 'deterministic')} "
            f"llm_batch={sum(1 for row in rows if row['judge_mode'] == 'llm_batch')}",
            verbose,
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    with open(out_base.with_suffix(".json"), "w") as f:
        json.dump({"aggregate": aggregate, "items": item_rows}, f, indent=2)

    with open(out_base.with_suffix(".csv"), "w", newline="") as f:
        fieldnames = list(aggregate[0].keys()) if aggregate else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate:
            writer.writerow(row)

    _log(f"[judge] saved judged results to {out_base.with_suffix('.json')}", True)
    _log(f"[judge] saved aggregate CSV to {out_base.with_suffix('.csv')}", True)


if __name__ == "__main__":
    main()
