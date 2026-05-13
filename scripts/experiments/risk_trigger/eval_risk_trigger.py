import argparse
import csv
import io
import json
import os
import random
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None

from instructions.load_instructions import load_category, resolve_items
from scripts.experiments.theta.threshold_sweep import (
    _extract_actions,
    _planner_targets,
    score_with_gold,
)
from scripts.modules.ope import OPE
from scripts.modules.pipeline_config import get_category_pipeline
from scripts.modules.pl_direct import DIRECT
from scripts.modules.risk_trigger import EXCLUDED_AUDIT_CATEGORIES, assess_risk_trigger
from scripts.modules.semantic_cache import flush_all_caches
from scripts.modules.urp import URP


DEFAULT_CATEGORIES = [
    "explicit_unambiguous",
    "explicit_ambiguous",
    "implicit",
    "risk_aware",
]


def _run_pipeline(
    item: Dict[str, Any],
    category: str,
    theta: float,
    planner_model: str,
    policy_meta: Dict[str, Any],
    pipeline: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    def impl() -> Dict[str, Any]:
        found_objects = item["objects"]
        stats: Dict[str, Any] = {}
        objects_info = OPE(
            found_objects,
            found_objects,
            theta=theta,
            stats=stats,
            llm_temperature=0,
            mode=pipeline["ope_mode"],
            pipeline_config=pipeline,
            user_request=item["instruction"],
        )
        urp_action = URP(
            item["instruction"],
            found_objects,
            objects_info,
            use_OPE=True,
            rel_objects=found_objects,
            theta=theta,
            stats=stats,
            llm_temperature=0,
            mode=pipeline["urp_mode"],
            pipeline_config=pipeline,
        )

        planner_steps = []
        place_targets = _planner_targets(found_objects, item)
        if place_targets:
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

        return {
            "category": category,
            "objects_info": objects_info,
            "urp_action": urp_action,
            "planner_steps": planner_steps,
            "pred_actions": pred_actions,
            "gold_success": score_with_gold(item, pred_actions, policy_meta),
            "stats": stats,
        }

    if verbose:
        return impl()
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return impl()


def _compare_triggered_pipeline(
    item: Dict[str, Any],
    category: str,
    theta: float,
    planner_model: str,
    policy_meta: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    standard_pipeline = get_category_pipeline(category)
    risk_pipeline = get_category_pipeline("risk_aware")
    standard = _run_pipeline(item, category, theta, planner_model, policy_meta, standard_pipeline, verbose=verbose)
    risk = _run_pipeline(item, "risk_aware", theta, planner_model, policy_meta, risk_pipeline, verbose=verbose)

    if standard["pred_actions"] == risk["pred_actions"]:
        difference = "same_actions"
    elif standard["gold_success"] != risk["gold_success"]:
        difference = "different_gold_score"
    else:
        difference = "different_actions_same_gold_score"

    return {
        "standard": standard,
        "risk_triggered": risk,
        "qualitative_difference": difference,
    }


def _summarize(rows: List[Dict[str, Any]], excluded_categories: List[str]) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        category = row["category"]
        bucket = by_category.setdefault(category, {"count": 0, "triggered": 0, "correct": 0})
        bucket["count"] += 1
        if row["risk_index_required"]:
            bucket["triggered"] += 1
        if row.get("trigger_correct"):
            bucket["correct"] += 1

    for bucket in by_category.values():
        count = bucket["count"]
        bucket["trigger_rate"] = bucket["triggered"] / count if count else 0.0
        bucket["accuracy"] = bucket["correct"] / count if count else 0.0

    risk_aware = by_category.get("risk_aware", {"count": 0, "triggered": 0})
    risk_aware_recall = risk_aware["triggered"] / risk_aware["count"] if risk_aware["count"] else None

    nonrisk_rows = [
        row
        for row in rows
        if row["category"] in {"explicit_unambiguous", "explicit_ambiguous", "implicit"}
    ]
    nonrisk_triggered = sum(1 for row in nonrisk_rows if row["risk_index_required"])
    nonrisk_false_positive_rate = nonrisk_triggered / len(nonrisk_rows) if nonrisk_rows else None

    explicit_rows = [
        row for row in rows if row["category"] in {"explicit_unambiguous", "explicit_ambiguous"}
    ]
    explicit_triggered = sum(1 for row in explicit_rows if row["risk_index_required"])
    explicit_false_positive_rate = explicit_triggered / len(explicit_rows) if explicit_rows else None

    overall_accuracy = (
        sum(1 for row in rows if row.get("trigger_correct")) / len(rows)
        if rows
        else None
    )

    implicit_triggered = [
        {
            "id": row["id"],
            "instruction": row["instruction"],
            "confidence": row["confidence"],
            "risk_cues": row["risk_cues"],
            "risk_reasons": row["risk_reasons"],
        }
        for row in rows
        if row["category"] == "implicit" and row["risk_index_required"]
    ]

    return {
        "by_category": by_category,
        "overall_trigger_accuracy": overall_accuracy,
        "risk_aware_recall": risk_aware_recall,
        "explicit_false_positive_rate": explicit_false_positive_rate,
        "nonrisk_false_positive_rate": nonrisk_false_positive_rate,
        "implicit_triggered_items": implicit_triggered,
        "excluded_categories": excluded_categories,
    }


def _write_outputs(
    out_base: Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    traces: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    serializable_rows = []
    for row in rows:
        serializable_rows.append({
            **row,
            "risk_cues": row["risk_cues"],
            "risk_reasons": row["risk_reasons"],
        })

    out_base.with_suffix(".json").write_text(json.dumps({
        "model": args.model,
        "theta": args.theta,
        "categories": args.categories,
        "run_triggered_pipeline": args.run_triggered_pipeline,
        "summary": summary,
        "items": serializable_rows,
    }, indent=2))

    csv_fields = [
        "category",
        "id",
        "instruction",
        "risk_index_required",
        "expected_risk_index_required",
        "trigger_correct",
        "confidence",
        "recommended_mode",
        "source",
        "risk_cues",
        "risk_reasons",
    ]
    with out_base.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "category": row["category"],
                "id": row["id"],
                "instruction": row["instruction"],
                "risk_index_required": row["risk_index_required"],
                "expected_risk_index_required": row["expected_risk_index_required"],
                "trigger_correct": row["trigger_correct"],
                "confidence": row["confidence"],
                "recommended_mode": row["recommended_mode"],
                "source": row["source"],
                "risk_cues": "; ".join(row["risk_cues"]),
                "risk_reasons": "; ".join(row["risk_reasons"]),
            })

    if args.save_traces:
        with out_base.with_suffix(".jsonl").open("w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--theta", type=float, default=0.75)
    parser.add_argument("--planner-model", default="gpt-4o-mini")
    parser.add_argument("--out", default="scripts/experiments/risk_trigger/results/risk_trigger_audit")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--run-triggered-pipeline", action="store_true")
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Disable category overrides and hide the category label from the trigger LLM for classifier evaluation.",
    )
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    if np is not None:
        np.random.seed(0)

    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    excluded_categories = []

    for category in args.categories:
        if category in EXCLUDED_AUDIT_CATEGORIES:
            print(f"[skip] category={category} excluded from first risk-trigger audit")
            excluded_categories.append(category)
            continue

        cat = load_category(category)
        items = resolve_items(cat)
        if args.limit_per_category is not None:
            items = items[: args.limit_per_category]
        policy_meta = cat.get("policy", {})

        for item in items:
            print(f"[risk-trigger] category={category} id={item['id']}")
            start = time.monotonic()
            trigger = assess_risk_trigger(
                user_instruction=item["instruction"],
                detected_objects=item["objects"],
                task_category=None if args.llm_only else category,
                model=args.model,
                use_category_overrides=not args.llm_only,
            )
            latency = time.monotonic() - start
            expected_risk_index_required = category == "risk_aware"
            row = {
                "category": category,
                "id": item["id"],
                "instruction": item["instruction"],
                "objects": item["objects"],
                "risk_index_required": trigger["risk_index_required"],
                "expected_risk_index_required": expected_risk_index_required,
                "trigger_correct": trigger["risk_index_required"] == expected_risk_index_required,
                "confidence": trigger["confidence"],
                "risk_reasons": trigger["risk_reasons"],
                "risk_cues": trigger["risk_cues"],
                "recommended_mode": trigger["recommended_mode"],
                "source": trigger["source"],
                "latency_seconds": latency,
            }
            rows.append(row)

            trace: Dict[str, Any] = {
                "item": item,
                "trigger": trigger,
                "latency_seconds": latency,
            }

            if args.run_triggered_pipeline and trigger["risk_index_required"]:
                trace["pipeline_comparison"] = _compare_triggered_pipeline(
                    item,
                    category,
                    args.theta,
                    args.planner_model,
                    policy_meta,
                    verbose=args.verbose,
                )
            traces.append(trace)

    summary = _summarize(rows, excluded_categories)
    _write_outputs(Path(args.out), rows, summary, traces, args)
    flush_all_caches()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
