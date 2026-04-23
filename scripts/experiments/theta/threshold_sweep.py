import argparse
import csv
import io
import json
import os
import random
import re
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any, List, Optional

import openai
try:
    import numpy as np
except Exception:
    np = None

from instructions.load_instructions import load_category, resolve_items
from scripts.modules.semantic_cache import flush_all_caches, get_openai_client, log_openai_call
from scripts.modules.ope import OPE
from scripts.modules.ope_mat import OPE_mat
from scripts.modules.ope_score_par import OPE_score_par
from scripts.modules.pl_direct import DIRECT
from scripts.modules.pl_iter import ITER
from scripts.modules.urp import URP
from scripts.modules.urp_risk import URP_risk


DEFAULT_THETAS = [0.65, 0.70, 0.75, 0.80, 0.85]
DEFAULT_CATEGORIES = [
    "explicit_unambiguous",
    "explicit_ambiguous",
    "implicit",
    "risk_aware",
    "materials",
    "toxicity",
]

DESTINATION_ALIASES = {
    "user": ["in front of the user", "front of the user", "to the user", "user"],
    "trash can": ["trash can", "trash", "bin"],
    "table": ["on the table", "to the table", "table"],
    "counter": ["far counter", "on the counter", "to the counter", "counter"],
    "fridge": ["in the fridge", "to the fridge", "fridge", "refrigerator"],
    "ceramic bowl": ["in the ceramic bowl", "to the ceramic bowl", "in the bowl", "to the bowl", "ceramic bowl", "bowl"],
    "cup": ["in the cup", "to the cup", "cup"],
    "ceramic mug": ["in the ceramic mug", "to the ceramic mug", "ceramic mug", "mug"],
    "microwave oven": ["in the microwave oven", "to the microwave oven", "microwave oven", "microwave"],
    "child": ["to the child", "child", "kid", "son"],
    "trivet": ["on the trivet", "to the trivet", "trivet"],
    "big block": ["on the big block", "to the big block", "big block"],
    "medium block": ["on the medium block", "to the medium block", "medium block"],
    "block": ["on the block", "to the block", "block"],
    "brick": ["on the brick", "to the brick", "brick"],
    "wooden box": ["on the wooden box", "to the wooden box", "wooden box"],
    "plastic container": ["in the plastic container", "to the plastic container", "plastic container"],
    "toxic bin": ["to the toxic bin", "in the toxic bin", "toxic bin", "special container", "safety container"],
    "standard bin": ["to the standard bin", "in the standard bin", "standard bin", "standard container"],
    "secure venomous bin": ["to the secure venomous bin", "in the secure venomous bin", "secure venomous bin", "safe area"],
    "public area bin": ["to the public area bin", "in the public area bin", "public area bin"],
    "verification bin": ["to the verification bin", "in the verification bin", "verification bin", "testing area"],
    "hazardous bin": ["to the hazardous bin", "in the hazardous bin", "hazardous bin", "safety containers"],
    "non-hazardous bin": ["to the non-hazardous bin", "in the non-hazardous bin", "non-hazardous bin", "standard containers"],
}

PLANNER_DESTINATIONS = {
    "user",
    "table",
    "counter",
    "trash can",
    "fridge",
    "ceramic bowl",
    "cup",
    "plate",
    "dishwasher",
    "handwashing",
    "glass bin",
    "plastic bin",
    "paper bin",
    "mixed bin",
    "wax bin",
    "standard bin",
    "green hellebore bin",
    "public area",
    "venomous animals room",
    "hazardous chemical bin",
    "non-hazardous chemical bin",
    "ceramic mug",
    "microwave oven",
    "child",
    "trivet",
    "toxic bin",
    "secure venomous bin",
    "public area bin",
    "verification bin",
    "hazardous bin",
    "non-hazardous bin",
}

OBJECT_ALIASES = {
    "chips": ["multigrain chips", "bag of chips", "chips"],
    "coke": ["coca cola", "coca-cola", "coke can", "coke"],
    "7up": ["7up can", "7up"],
    "tea": ["cup of tea", "tea"],
}


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
    client = get_openai_client()
    start = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    log_openai_call("llm_judge", instruction, time.monotonic() - start)
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return 1 if int(data.get("score", 0)) == 1 else 0
    except Exception:
        # Fallback: accept '1' or '0'
        if "1" in content and "0" not in content:
            return 1
        return 0


def _normalize_phrase(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_candidates(name: str) -> List[str]:
    aliases = OBJECT_ALIASES.get(name, [])
    return sorted({name, *aliases}, key=len, reverse=True)


def _find_destination(segment: str, known_objects: List[str]) -> Optional[str]:
    normalized = _normalize_phrase(segment)
    candidates = {}
    for obj in known_objects:
        if obj in DESTINATION_ALIASES:
            candidates[obj] = DESTINATION_ALIASES[obj]
    for dest, aliases in DESTINATION_ALIASES.items():
        if dest not in candidates and dest in known_objects:
            candidates[dest] = aliases
    for dest, aliases in candidates.items():
        for alias in sorted(set(aliases), key=len, reverse=True):
            if alias in normalized:
                return dest
    return None


def _find_objects(segment: str, known_objects: List[str], destination: Optional[str]) -> List[str]:
    normalized = _normalize_phrase(segment)
    if destination:
        for alias in DESTINATION_ALIASES.get(destination, []):
            normalized = normalized.replace(alias, " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
    found = []
    for obj in sorted(known_objects, key=len, reverse=True):
        if destination and obj == destination:
            continue
        for alias in _alias_candidates(obj):
            if re.search(rf"\\b{re.escape(alias)}\\b", normalized):
                found.append(obj)
                normalized = re.sub(rf"\\b{re.escape(alias)}\\b", " ", normalized, count=1)
                break
    return found


def _canonicalize_nl_actions(action_text: str, item: Dict[str, Any]) -> List[str]:
    known_objects = item.get("objects") or []
    if not action_text.strip():
        return []

    segments = []
    for block in action_text.splitlines():
        for part in re.split(r"(?<=[.!?])\s+|\bthen\b|\bfinally\b", block):
            part = part.strip(" -")
            if part:
                segments.append(part)

    actions = []
    for segment in segments:
        lowered = segment.lower()
        if "pick" not in lowered and "place" not in lowered and "move" not in lowered:
            continue
        destination = _find_destination(segment, known_objects)
        objects = _find_objects(segment, known_objects, destination)
        if not destination or not objects:
            continue
        for obj in objects:
            actions.append(f"robot.pick_and_place({obj}, {destination})")

    if actions and "done()" not in actions:
        actions.append("done()")
    return actions


def _extract_actions(action_text: str, item: Dict[str, Any]) -> List[str]:
    lines = [ln.strip() for ln in action_text.splitlines() if ln.strip()]
    actions = []
    for ln in lines:
        if ln.startswith("robot.pick_and_place") or ln == "done()":
            actions.append(ln)
    if actions:
        if "done()" not in actions:
            actions.append("done()")
        return actions
    actions = _canonicalize_nl_actions(action_text, item)
    if actions:
        return actions
    if action_text:
        return [action_text.strip()]
    return actions


def _gold_destinations(item: Dict[str, Any]) -> set[str]:
    gold = item.get("gold") or {}
    destinations = set((gold.get("rules") or {}).keys())
    for seq in gold.get("sequences", []):
        for act in seq:
            if not isinstance(act, str) or not act.startswith("robot.pick_and_place("):
                continue
            inner = act[len("robot.pick_and_place("):-1]
            try:
                _, dest = [s.strip() for s in inner.split(",", 1)]
            except Exception:
                continue
            destinations.add(dest)
    return destinations


def _planner_targets(found_objects: List[str], item: Dict[str, Any]) -> Dict[str, str]:
    gold_destinations = _gold_destinations(item)
    place_targets = {}
    for obj in found_objects:
        if obj in PLANNER_DESTINATIONS or obj in gold_destinations:
            place_targets[obj] = obj
    return place_targets


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


def _run_item_impl(
    item: Dict[str, Any],
    theta: float,
    category: str,
    judge_model: str,
    policy_meta: Dict[str, Any],
    planner: str,
    planner_model: str,
) -> Dict[str, Any]:
    """
    Run OPE + URP and judge the generated action.
    Returns: dict with success and stats.
    """
    found_objects = item["objects"]
    stats: Dict[str, Any] = {}

    # Choose OPE/URP variant by category
    if category == "materials":
        objects_info = OPE_mat(found_objects, found_objects, theta=theta, stats=stats, llm_temperature=0)
        urp_fn = URP
        use_ope_in_urp = True
    elif category == "risk_aware":
        objects_info = OPE_score_par(found_objects, found_objects, item["instruction"], theta=theta, stats=stats, llm_temperature=0)
        urp_fn = URP_risk
        use_ope_in_urp = True
    else:
        objects_info = OPE(found_objects, found_objects, theta=theta, stats=stats, llm_temperature=0)
        urp_fn = URP
        use_ope_in_urp = True

    # URP to get a structured high-level query for the planner
    urp_action = urp_fn(
        item["instruction"],
        found_objects,
        objects_info,
        use_OPE=use_ope_in_urp,
        rel_objects=found_objects,
        theta=theta,
        stats=stats,
        llm_temperature=0,
    )

    place_targets = _planner_targets(found_objects, item)
    planner_steps = []
    if place_targets:
        if planner == "iter":
            planner_steps = ITER(
                found_objects=found_objects,
                PICK_TARGETS=found_objects,
                PLACE_TARGETS=place_targets,
                query=urp_action,
                model=planner_model,
                limit_num_options=max(3, len(found_objects)),
                verbose=False,
                top_logprobs=5,
            )
        elif planner == "direct":
            planner_steps = DIRECT(
                found_objects=found_objects,
                PICK_TARGETS=found_objects,
                PLACE_TARGETS=place_targets,
                query=urp_action,
                model=planner_model,
            )

    pred_actions = [step for step in planner_steps if step]
    if not pred_actions:
        pred_actions = _extract_actions(urp_action, item)
    if pred_actions and "done()" not in pred_actions:
        pred_actions.append("done()")
    gold_score = score_with_gold(item, pred_actions, policy_meta)
    if gold_score is None:
        score = llm_judge(item["instruction"], "\n".join(pred_actions), model=judge_model)
    else:
        score = gold_score

    return {
        "success": score,
        "stats": stats,
        "action": urp_action,
        "pred_actions": pred_actions,
        "planner_steps": planner_steps,
    }


def run_item(
    item: Dict[str, Any],
    theta: float,
    category: str,
    judge_model: str,
    policy_meta: Dict[str, Any],
    planner: str,
    planner_model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        return _run_item_impl(item, theta, category, judge_model, policy_meta, planner, planner_model)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return _run_item_impl(item, theta, category, judge_model, policy_meta, planner, planner_model)


def aggregate_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, float]:
    ope_total = sum(s.get("ope_relations_total", 0) for s in stats_list)
    ope_kept = sum(s.get("ope_relations_kept", 0) for s in stats_list)
    ope_objects = sum(s.get("ope_objects", 0) for s in stats_list)
    ope_zero_relation_objects = sum(s.get("ope_zero_relation_objects", 0) for s in stats_list)

    urp_total = sum(s.get("urp_relations_total", 0) for s in stats_list)
    urp_kept = sum(s.get("urp_relations_kept", 0) for s in stats_list)
    urp_keywords = sum(s.get("urp_keywords", 0) for s in stats_list)
    urp_zero_relation_keywords = sum(s.get("urp_zero_relation_keywords", 0) for s in stats_list)

    avg_relations_per_object = (ope_kept / ope_objects) if ope_objects else 0.0
    avg_relations_per_object_before = (ope_total / ope_objects) if ope_objects else 0.0
    zero_relation_object_ratio = (ope_zero_relation_objects / ope_objects) if ope_objects else 0.0

    avg_relations_per_keyword = (urp_kept / urp_keywords) if urp_keywords else 0.0
    avg_relations_per_keyword_before = (urp_total / urp_keywords) if urp_keywords else 0.0
    zero_relation_keyword_ratio = (urp_zero_relation_keywords / urp_keywords) if urp_keywords else 0.0

    zero_relation_queries = ope_zero_relation_objects + urp_zero_relation_keywords
    total_relation_queries = ope_objects + urp_keywords
    zero_relation_query_ratio = (zero_relation_queries / total_relation_queries) if total_relation_queries else 0.0
    zero_relation_object_names = sorted(
        {
            name
            for stats in stats_list
            for name in stats.get("ope_zero_relation_object_names", [])
        }
    )
    zero_relation_keyword_names = sorted(
        {
            name
            for stats in stats_list
            for name in stats.get("urp_zero_relation_keyword_names", [])
        }
    )
    zero_relation_terms = sorted(
        {f"object:{name}" for name in zero_relation_object_names}
        | {f"keyword:{name}" for name in zero_relation_keyword_names}
    )

    return {
        "relations_total": ope_total + urp_total,
        "relations_kept": ope_kept + urp_kept,
        "avg_relations_per_object": avg_relations_per_object,
        "avg_relations_per_object_before": avg_relations_per_object_before,
        "avg_relations_per_keyword": avg_relations_per_keyword,
        "avg_relations_per_keyword_before": avg_relations_per_keyword_before,
        "ope_objects": ope_objects,
        "ope_zero_relation_objects": ope_zero_relation_objects,
        "ope_zero_relation_object_ratio": zero_relation_object_ratio,
        "ope_relations_total": ope_total,
        "ope_relations_kept": ope_kept,
        "urp_keywords": urp_keywords,
        "urp_zero_relation_keywords": urp_zero_relation_keywords,
        "urp_zero_relation_keyword_ratio": zero_relation_keyword_ratio,
        "urp_relations_total": urp_total,
        "urp_relations_kept": urp_kept,
        "zero_relation_queries": zero_relation_queries,
        "total_relation_queries": total_relation_queries,
        "zero_relation_query_ratio": zero_relation_query_ratio,
        "ope_zero_relation_object_names": "; ".join(zero_relation_object_names),
        "urp_zero_relation_keyword_names": "; ".join(zero_relation_keyword_names),
        "zero_relation_terms": "; ".join(zero_relation_terms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-list", nargs="+", type=float, default=DEFAULT_THETAS)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--out", default="scripts/experiments/theta/results/threshold_sweep")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--planner", choices=["direct", "iter", "none"], default="direct")
    parser.add_argument("--planner-model", default="gpt-4o-mini")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--save-policies", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    if np is not None:
        np.random.seed(0)
    if args.cache_only:
        os.environ["CONCEPTNET_CACHE_ONLY"] = "1"

    results = []
    policy_log_lines = []

    for theta in args.theta_list:
        for category in args.categories:
            cat = load_category(category)
            items = resolve_items(cat)
            policy_meta = cat.get("policy", {})

            successes = 0
            stats_list = []
            for item in items:
                for trial in range(args.num_trials):
                    print(
                        f"[progress] theta={theta} category={category} id={item['id']} "
                        f"trial={trial + 1}/{args.num_trials}"
                    )
                    out = run_item(
                        item,
                        theta=theta,
                        category=category,
                        judge_model=args.judge_model,
                        policy_meta=policy_meta,
                        planner=args.planner,
                        planner_model=args.planner_model,
                        verbose=args.verbose,
                    )
                    successes += out["success"]
                    stats_list.append(out["stats"])
                    if args.save_policies:
                        policy_log_lines.append(
                            f"theta={theta} category={category} id={item['id']} trial={trial} success={out['success']}\n"
                        )
                        policy_log_lines.append(f"instruction: {item['instruction']}\n")
                        policy_log_lines.append(f"urp_action: {out.get('action','')}\n")
                        policy_log_lines.append(f"planner_steps: {json.dumps(out.get('planner_steps', []))}\n")
                        policy_log_lines.append(f"parsed_actions: {json.dumps(out.get('pred_actions', []))}\n")
                        policy_log_lines.append("---\n")

            denom = len(items) * args.num_trials if items else 0.0
            success_rate = successes / denom if denom else 0.0
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
        json.dump({"results": results, "overall": overall, "seed": 0, "num_trials": args.num_trials}, f, indent=2)

    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    if args.save_policies:
        txt_path = out_base.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.writelines(policy_log_lines)
    if args.plot:
        try:
            from scripts.experiments.theta.plot_threshold_results import main as plot_main
        except Exception:
            plot_main = None
        if plot_main is not None:
            import sys
            previous_argv = sys.argv
            try:
                sys.argv = ["plot_threshold_results.py", str(out_base.with_suffix(".json"))]
                plot_main()
            finally:
                sys.argv = previous_argv
    flush_all_caches()

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
