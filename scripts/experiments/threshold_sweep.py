import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List

import openai
try:
    import numpy as np
except Exception:
    np = None

from instructions.load_instructions import load_category, resolve_items
from scripts.modules.ope import OPE
from scripts.modules.ope_mat import OPE_mat
from scripts.modules.urp import URP


DEFAULT_THETAS = [0.65, 0.70, 0.75, 0.80, 0.85]
DEFAULT_CATEGORIES = [
    "explicit_unambiguous",
    "explicit_ambiguous",
    "implicit",
    "risk_aware",
    "materials",
    "toxicity",
]


def llm_judge(instruction: str, action: str, model: str = "gpt-4o-mini") -> int:
    """
    LLM-as-judge: returns 1 if action satisfies instruction, else 0.
    """
    system = (
        "You are a strict evaluator for robot task execution. "
        "Given a user instruction and a proposed action description, "
        "respond ONLY with JSON: {\"score\": 0 or 1}. "
        "Return 1 only if the action clearly satisfies the instruction."
    )
    user = f"Instruction: {instruction}\nAction: {action}\n"
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return 1 if int(data.get("score", 0)) == 1 else 0
    except Exception:
        # Fallback: accept '1' or '0'
        if "1" in content and "0" not in content:
            return 1
        return 0


def _extract_actions(action_text: str) -> List[str]:
    lines = [ln.strip() for ln in action_text.splitlines() if ln.strip()]
    actions = []
    for ln in lines:
        if ln.startswith("robot.pick_and_place") or ln == "done()":
            actions.append(ln)
    if not actions and action_text:
        actions = [action_text.strip()]
    return actions


def _match_sequences(pred: List[str], gold_sequences: List[List[str]], order_matters: bool) -> bool:
    for seq in gold_sequences:
        if order_matters:
            if pred == seq:
                return True
        else:
            if set(pred) == set(seq):
                return True
    return False


def _match_rules(pred: List[str], rules: Dict[str, List[str]]) -> bool:
    # Build predicted mapping: dest -> set(objects)
    pred_map = {}
    for act in pred:
        if not act.startswith("robot.pick_and_place("):
            continue
        inner = act[len("robot.pick_and_place("):-1]
        try:
            obj, dest = [s.strip() for s in inner.split(",")]
        except Exception:
            continue
        pred_map.setdefault(dest, set()).add(obj)
    # All required objects must appear in correct dest
    for dest, objs in rules.items():
        if dest not in pred_map:
            return False
        for o in objs:
            if o not in pred_map[dest]:
                return False
    return True


def score_with_gold(item: Dict[str, Any], pred_actions: List[str], policy_meta: Dict[str, Any]) -> Optional[int]:
    gold = item.get("gold")
    if not gold:
        return None
    order_matters = policy_meta.get("order_matters", True)
    if "sequences" in gold:
        if _match_sequences(pred_actions, gold["sequences"], order_matters):
            return 1
        return 0
    if "rules" in gold:
        return 1 if _match_rules(pred_actions, gold["rules"]) else 0
    return None


def run_item(item: Dict[str, Any], theta: float, category: str, judge_model: str, policy_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run OPE + URP and judge the generated action.
    Returns: dict with success and stats.
    """
    found_objects = item["objects"]
    stats: Dict[str, Any] = {}

    # Choose OPE variant (materials tasks use OPE_mat, others use OPE)
    if category == "materials":
        objects_info = OPE_mat(found_objects, found_objects, theta=theta, stats=stats, llm_temperature=0)
    else:
        objects_info = OPE(found_objects, found_objects, theta=theta, stats=stats, llm_temperature=0)

    # URP to get an action-like structured response
    action = URP(
        item["instruction"],
        found_objects,
        objects_info,
        use_OPE=True,
        rel_objects=found_objects,
        theta=theta,
        stats=stats,
        llm_temperature=0,
    )

    pred_actions = _extract_actions(action)
    gold_score = score_with_gold(item, pred_actions, policy_meta)
    if gold_score is None:
        score = llm_judge(item["instruction"], action, model=judge_model)
    else:
        score = gold_score

    return {
        "success": score,
        "stats": stats,
    }


def aggregate_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, float]:
    ope_total = sum(s.get("ope_relations_total", 0) for s in stats_list)
    ope_kept = sum(s.get("ope_relations_kept", 0) for s in stats_list)
    ope_objects = sum(s.get("ope_objects", 0) for s in stats_list)

    urp_total = sum(s.get("urp_relations_total", 0) for s in stats_list)
    urp_kept = sum(s.get("urp_relations_kept", 0) for s in stats_list)
    urp_keywords = sum(s.get("urp_keywords", 0) for s in stats_list)

    avg_relations_per_object = (ope_kept / ope_objects) if ope_objects else 0.0
    avg_relations_per_object_before = (ope_total / ope_objects) if ope_objects else 0.0

    avg_relations_per_keyword = (urp_kept / urp_keywords) if urp_keywords else 0.0
    avg_relations_per_keyword_before = (urp_total / urp_keywords) if urp_keywords else 0.0

    return {
        "relations_total": ope_total + urp_total,
        "relations_kept": ope_kept + urp_kept,
        "avg_relations_per_object": avg_relations_per_object,
        "avg_relations_per_object_before": avg_relations_per_object_before,
        "avg_relations_per_keyword": avg_relations_per_keyword,
        "avg_relations_per_keyword_before": avg_relations_per_keyword_before,
        "ope_relations_total": ope_total,
        "ope_relations_kept": ope_kept,
        "urp_relations_total": urp_total,
        "urp_relations_kept": urp_kept,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-list", nargs="+", type=float, default=DEFAULT_THETAS)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--out", default="results/threshold_sweep")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    if np is not None:
        np.random.seed(0)
    if args.cache_only:
        os.environ["CONCEPTNET_CACHE_ONLY"] = "1"

    results = []

    for theta in args.theta_list:
        for category in args.categories:
            cat = load_category(category)
            items = resolve_items(cat)
            policy_meta = cat.get("policy", {})

            successes = 0
            stats_list = []
            for item in items:
                out = run_item(item, theta=theta, category=category, judge_model=args.judge_model, policy_meta=policy_meta)
                successes += out["success"]
                stats_list.append(out["stats"])

            success_rate = successes / len(items) if items else 0.0
            stats = aggregate_stats(stats_list)

            results.append({
                "theta": theta,
                "category": category,
                "success_rate": success_rate,
                **stats,
            })

    # Overall
    overall = {}
    for theta in args.theta_list:
        subset = [r for r in results if r["theta"] == theta]
        if not subset:
            continue
        overall_success = sum(r["success_rate"] for r in subset) / len(subset)
        avg_rel_obj = sum(r["avg_relations_per_object"] for r in subset) / len(subset)
        avg_rel_kw = sum(r["avg_relations_per_keyword"] for r in subset) / len(subset)
        overall[theta] = {
            "overall_success_rate": overall_success,
            "avg_relations_per_object": avg_rel_obj,
            "avg_relations_per_keyword": avg_rel_kw,
        }

    # Save JSON/CSV
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with open(out_base.with_suffix(".json"), "w") as f:
        json.dump({"results": results, "overall": overall, "seed": 0}, f, indent=2)

    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Print best theta and trade-off
    best_theta = max(overall.keys(), key=lambda t: overall[t]["overall_success_rate"])
    best = overall[best_theta]

    print("Best theta:", best_theta)
    print("Overall success rate:", best["overall_success_rate"])
    print(
        "Trade-off: higher theta keeps fewer relations (lower avg relations per object/keyword), "
        "which can reduce noise but may hurt recall; lower theta keeps more relations, "
        "which can improve recall but may add noise."
    )


if __name__ == "__main__":
    main()
