import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from instructions.load_instructions import load_category, resolve_items
from scripts.experiments.baselines.progprompt_baseline import run_progprompt_item


DEFAULT_CATEGORIES = ["explicit_unambiguous", "explicit_ambiguous"]


def _aggregate_category_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    successes = sum(int(r["success"]) for r in rows)
    parseable = sum(1 for r in rows if r["parse_status"] in {"ok", "done_only"})
    avg_actions = sum(r["num_actions"] for r in rows) / count if count else 0.0
    avg_latency = sum(r["latency_seconds"] for r in rows) / count if count else 0.0
    done_rate = sum(1 for r in rows if r["termination_reason"] == "done") / count if count else 0.0
    assertion_share = sum(1 for r in rows if r["assertion_count"] > 0) / count if count else 0.0
    avg_assertions = sum(r["assertion_count"] for r in rows) / count if count else 0.0
    return {
        "count": count,
        "success_rate": successes / count if count else 0.0,
        "parseable_rate": parseable / count if count else 0.0,
        "avg_actions": avg_actions,
        "avg_latency_seconds": avg_latency,
        "done_rate": done_rate,
        "plans_with_assertions_rate": assertion_share,
        "avg_assertions_per_plan": avg_assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--out", default="scripts/experiments/baselines/results/progprompt_baseline")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    category_summaries: List[Dict[str, Any]] = []
    trace_records: List[Dict[str, Any]] = []

    for category in args.categories:
        cat = load_category(category)
        items = resolve_items(cat)
        policy_meta = cat.get("policy", {})
        category_rows: List[Dict[str, Any]] = []

        for item in items:
            item_for_run = {**item, "category": category}
            for trial in range(1, args.num_trials + 1):
                print(
                    f"[progress] category={category} id={item['id']} "
                    f"trial={trial}/{args.num_trials}"
                )
                result = run_progprompt_item(
                    item_for_run,
                    model=args.model,
                    max_steps=args.max_steps,
                    max_retries=args.max_retries,
                    verbose=args.verbose,
                    policy_meta=policy_meta,
                )

                row = {
                    "category": category,
                    "id": item["id"],
                    "trial": trial,
                    "instruction": item["instruction"],
                    "success": int(result["success"]),
                    "parse_status": result["parse_status"],
                    "termination_reason": result["termination_reason"],
                    "num_actions": result["num_actions"],
                    "num_invalid_actions": result["num_invalid_actions"],
                    "latency_seconds": result["latency_seconds"],
                    "assertion_count": result["assertion_count"],
                    "unsupported_assertions": result["unsupported_assertions"],
                }
                category_rows.append(row)
                all_rows.append(row)

                if args.save_traces:
                    trace_records.append(
                        {
                            "category": category,
                            "id": item["id"],
                            "trial": trial,
                            "instruction": item["instruction"],
                            "pred_actions": result["pred_actions"],
                            "parse_status": result["parse_status"],
                            "termination_reason": result["termination_reason"],
                            "raw_program": result["raw_program"],
                            "step_records": result["step_records"],
                            "assertion_count": result["assertion_count"],
                            "unsupported_assertions": result["unsupported_assertions"],
                        }
                    )

        summary = _aggregate_category_rows(category_rows)
        category_summaries.append({"category": category, **summary})

    overall = _aggregate_category_rows(all_rows)

    with open(out_base.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "categories": category_summaries,
                "overall": overall,
                "items": all_rows,
                "model": args.model,
                "max_steps": args.max_steps,
                "max_retries": args.max_retries,
                "num_trials": args.num_trials,
            },
            f,
            indent=2,
        )

    with open(out_base.with_suffix(".csv"), "w", newline="") as f:
        fieldnames = [
            "category",
            "count",
            "success_rate",
            "parseable_rate",
            "avg_actions",
            "avg_latency_seconds",
            "done_rate",
            "plans_with_assertions_rate",
            "avg_assertions_per_plan",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in category_summaries:
            writer.writerow(row)
        writer.writerow({"category": "overall", **overall})

    if args.save_traces:
        with open(out_base.with_suffix(".jsonl"), "w") as f:
            for record in trace_records:
                f.write(json.dumps(record) + "\n")

    print("Overall success rate:", overall["success_rate"])
    print("Overall parseable rate:", overall["parseable_rate"])
    print("Plans with assertions rate:", overall["plans_with_assertions_rate"])


if __name__ == "__main__":
    main()
