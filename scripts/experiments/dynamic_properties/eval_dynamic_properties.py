import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import numpy as np
except Exception:
    np = None

from scripts.modules.dynamic_properties import (
    assess_dynamic_property_need,
    induce_task_properties,
    normalize_property_name,
)
from scripts.modules.pipeline_config import STANDARD_OPE_PROPERTIES
from scripts.modules.semantic_cache import flush_all_caches


BASE_PROPERTIES = STANDARD_OPE_PROPERTIES

PROPERTY_ALIASES = {
    "microwave-safe": {"microwave-safe", "microwave safe", "microwave-compatible", "microwave compatible"},
    "heat-resistant": {"heat-resistant", "heat resistant", "heat-proof", "heat proof", "heatproof"},
    "non-metallic": {"non-metallic", "non metallic", "not metal", "non-metal"},
    "container": {"container", "container type", "food container", "holds food", "hold food", "sealed", "containment"},
    "non-flammable": {"non-flammable", "non flammable", "not flammable", "fire-resistant", "fire resistant"},
    "stable": {"stable", "stability"},
    "waterproof": {"waterproof", "water-resistant", "water resistant"},
    "protective": {"protective", "protection", "protects", "water protection"},
    "food-safe": {"food-safe", "food safe", "safe for food", "food contact safe"},
    "clean": {"clean", "hygienic", "sanitary", "easy to clean"},
    "non-toxic": {"non-toxic", "non toxic", "safe", "not poisonous"},
    "contaminating": {"contaminating", "contamination", "contaminant", "spill risk"},
    "chemically reactive": {"chemically reactive", "reactive", "chemical interaction", "chemical compatibility", "chemical separation"},
}


CASES = [
    {
        "id": "dyn_01",
        "label": "microwave compatibility",
        "instruction": "Heat my food in the microwave.",
        "objects": ["aluminum tray", "glass container", "soup bowl", "microwave oven"],
        "expected_required": True,
        "expected_properties": ["microwave-safe", "heat-resistant", "non-metallic", "container"],
    },
    {
        "id": "dyn_02",
        "label": "hot object placement",
        "instruction": "Put the hot pan on something safe.",
        "objects": ["hot pan", "plastic plate", "wooden trivet", "paper towel", "table"],
        "expected_required": True,
        "expected_properties": ["heat-resistant", "non-flammable", "stable"],
    },
    {
        "id": "dyn_03",
        "label": "water protection",
        "instruction": "Move the item that can protect the phone from water to the user.",
        "objects": ["phone", "paper bag", "plastic pouch", "cloth napkin", "user"],
        "expected_required": True,
        "expected_properties": ["waterproof", "protective"],
    },
    {
        "id": "dyn_04",
        "label": "food-safe container",
        "instruction": "Put the snack in a safe container for food.",
        "objects": ["chips", "chemical beaker", "ceramic bowl", "dusty box", "user"],
        "expected_required": True,
        "expected_properties": ["food-safe", "clean", "non-toxic", "container"],
    },
    {
        "id": "dyn_05",
        "label": "chemical food separation",
        "instruction": "Move the cleaning chemical away from food items.",
        "objects": ["bleach bottle", "apple", "bread", "counter", "hazardous bin"],
        "expected_required": True,
        "expected_properties": ["contaminating", "chemically reactive", "food-safe"],
    },
    {
        "id": "dyn_06",
        "label": "ordinary object request",
        "instruction": "Bring me the apple.",
        "objects": ["apple", "book", "user"],
        "expected_required": False,
        "expected_properties": [],
    },
    {
        "id": "dyn_07",
        "label": "ordinary placement",
        "instruction": "Move the red block to the table.",
        "objects": ["red block", "blue block", "table"],
        "expected_required": False,
        "expected_properties": [],
    },
    {
        "id": "dyn_08",
        "label": "drink retrieval covered by base",
        "instruction": "Bring me something to drink.",
        "objects": ["water bottle", "coke", "cup", "user"],
        "expected_required": False,
        "expected_properties": [],
    },
    {
        "id": "dyn_09",
        "label": "snack retrieval",
        "instruction": "Bring me a snack.",
        "objects": ["chips", "apple", "book", "user"],
        "expected_required": False,
        "expected_properties": [],
    },
]


def _alias_group(name: str) -> str:
    normalized = normalize_property_name(name)
    for canonical, aliases in PROPERTY_ALIASES.items():
        if normalized == canonical or normalized in {normalize_property_name(alias) for alias in aliases}:
            return canonical
    return normalized.replace("-", " ")


def _groups(properties: List[str]) -> Set[str]:
    return {_alias_group(prop) for prop in properties}


def _score_properties(expected: List[str], base: List[str], induced: List[str]) -> Dict[str, Any]:
    expected_groups = _groups(expected)
    base_groups = _groups(base)
    induced_groups = _groups(induced)
    merged_groups = base_groups | induced_groups

    covered = sorted(expected_groups & merged_groups)
    missing = sorted(expected_groups - merged_groups)
    dynamic_expected = expected_groups - base_groups
    dynamic_covered = sorted(dynamic_expected & induced_groups)
    dynamic_missing = sorted(dynamic_expected - induced_groups)

    return {
        "property_recall": (len(covered) / len(expected_groups)) if expected_groups else None,
        "dynamic_property_recall": (len(dynamic_covered) / len(dynamic_expected)) if dynamic_expected else None,
        "covered_expected_properties": covered,
        "missing_expected_properties": missing,
        "dynamic_covered_expected_properties": dynamic_covered,
        "dynamic_missing_expected_properties": dynamic_missing,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    positives = [row for row in rows if row["expected_required"]]
    negatives = [row for row in rows if not row["expected_required"]]
    correct = sum(1 for row in rows if row["trigger_correct"])
    positive_triggered = sum(1 for row in positives if row["dynamic_property_induction_required"])
    negative_not_triggered = sum(1 for row in negatives if not row["dynamic_property_induction_required"])
    positive_property_recalls = [
        row["property_recall"]
        for row in positives
        if row["property_recall"] is not None
    ]
    positive_dynamic_recalls = [
        row["dynamic_property_recall"]
        for row in positives
        if row["dynamic_property_recall"] is not None
    ]

    return {
        "count": len(rows),
        "trigger_accuracy": correct / len(rows) if rows else 0.0,
        "positive_trigger_recall": positive_triggered / len(positives) if positives else None,
        "negative_specificity": negative_not_triggered / len(negatives) if negatives else None,
        "avg_property_recall_positive": (
            sum(positive_property_recalls) / len(positive_property_recalls)
            if positive_property_recalls
            else None
        ),
        "avg_dynamic_property_recall_positive": (
            sum(positive_dynamic_recalls) / len(positive_dynamic_recalls)
            if positive_dynamic_recalls
            else None
        ),
    }


def _write_outputs(out_base: Path, rows: List[Dict[str, Any]], traces: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    summary = _summarize(rows)
    out_base.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": args.model,
                "max_properties": args.max_properties,
                "summary": summary,
                "items": rows,
            },
            indent=2,
        )
    )

    fieldnames = [
        "id",
        "label",
        "expected_required",
        "dynamic_property_induction_required",
        "trigger_correct",
        "confidence",
        "recommended_mode",
        "property_recall",
        "dynamic_property_recall",
        "expected_properties",
        "induced_properties",
        "covered_expected_properties",
        "missing_expected_properties",
        "missing_property_reasons",
    ]
    with out_base.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "; ".join(row[key]) if isinstance(row.get(key), list) else row.get(key)
                for key in fieldnames
            })

    if args.save_traces:
        with out_base.with_suffix(".jsonl").open("w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Dynamic Property Induction on curated missing-property cases.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-properties", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--out", default="scripts/experiments/dynamic_properties/results/dynamic_property_audit")
    args = parser.parse_args()

    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    if np is not None:
        np.random.seed(0)

    cases = CASES[: args.limit] if args.limit is not None else CASES
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for case in cases:
        print(f"[dynamic-property] id={case['id']} label={case['label']}")
        start = time.monotonic()
        assessment = assess_dynamic_property_need(
            user_instruction=case["instruction"],
            detected_objects=case["objects"],
            base_properties=BASE_PROPERTIES,
            task_category=case["label"],
            model=args.model,
        )
        assessment_latency = time.monotonic() - start

        induction = {
            "dynamic_properties": [],
            "merged_properties": BASE_PROPERTIES,
            "rejected_properties": [],
            "source": "not_run",
        }
        if assessment["dynamic_property_induction_required"]:
            start = time.monotonic()
            induction = induce_task_properties(
                user_instruction=case["instruction"],
                detected_objects=case["objects"],
                base_properties=BASE_PROPERTIES,
                task_category=case["label"],
                max_properties=args.max_properties,
                model=args.model,
            )
            induction_latency = time.monotonic() - start
        else:
            induction_latency = 0.0

        induced_names = [item["name"] for item in induction["dynamic_properties"]]
        property_scores = _score_properties(case["expected_properties"], BASE_PROPERTIES, induced_names)
        trigger_correct = assessment["dynamic_property_induction_required"] == case["expected_required"]

        row = {
            "id": case["id"],
            "label": case["label"],
            "instruction": case["instruction"],
            "objects": case["objects"],
            "expected_required": case["expected_required"],
            "expected_properties": case["expected_properties"],
            "dynamic_property_induction_required": assessment["dynamic_property_induction_required"],
            "trigger_correct": trigger_correct,
            "confidence": assessment["confidence"],
            "recommended_mode": assessment["recommended_mode"],
            "missing_property_reasons": assessment["missing_property_reasons"],
            "expected_property_types": assessment["expected_property_types"],
            "induced_properties": induced_names,
            "rejected_properties": induction["rejected_properties"],
            "assessment_latency_seconds": assessment_latency,
            "induction_latency_seconds": induction_latency,
            **property_scores,
        }
        rows.append(row)
        traces.append({"case": case, "assessment": assessment, "induction": induction, "row": row})

    _write_outputs(Path(args.out), rows, traces, args)
    flush_all_caches()


if __name__ == "__main__":
    main()
