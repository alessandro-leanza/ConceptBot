import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.experiments.baselines.react_baseline import (
    _planner_destinations,
    category_guidance,
    llm_judge,
    score_with_gold,
)
from scripts.modules.semantic_cache import get_openai_client, log_openai_call


LLM_PLANNER_SEED_EXAMPLES = [
    {
        "id": "seed_01",
        "instruction": "Give the coke to the user and throw the chips away.",
        "objects": ["coke", "chips", "user", "trash can"],
        "destinations": ["user", "trash can"],
        "plan": [
            "robot.pick_and_place(coke, user)",
            "robot.pick_and_place(chips, trash can)",
            "done()",
        ],
    },
    {
        "id": "seed_02",
        "instruction": "Put the sponge on the table.",
        "objects": ["sponge", "table", "user"],
        "destinations": ["table", "user"],
        "plan": [
            "robot.pick_and_place(sponge, table)",
            "done()",
        ],
    },
]


def _list_text(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_similarity(a: str, b: str) -> float:
    a_counts = Counter(_tokenize(a))
    b_counts = Counter(_tokenize(b))
    if not a_counts or not b_counts:
        return 0.0
    overlap = sum(min(a_counts[token], b_counts[token]) for token in a_counts)
    denom = (sum(a_counts.values()) * sum(b_counts.values())) ** 0.5
    return overlap / denom if denom else 0.0


def _first_gold_plan(item: Dict[str, Any]) -> List[str]:
    gold = item.get("gold") or {}
    sequences = gold.get("sequences") or []
    if sequences:
        return list(sequences[0])

    rules = gold.get("rules") or {}
    plan: List[str] = []
    for dest, objects in rules.items():
        for obj in objects:
            plan.append(f"robot.pick_and_place({obj}, {dest})")
    if plan:
        plan.append("done()")
    return plan


def build_fewshot_pool(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for item in items:
        plan = _first_gold_plan(item)
        if not plan:
            continue
        destinations = _planner_destinations(item)
        examples.append(
            {
                "id": item.get("id", ""),
                "instruction": item.get("instruction", ""),
                "objects": list(item.get("objects") or []),
                "destinations": destinations,
                "plan": plan,
            }
        )
    return examples


def _select_fewshot_examples(
    item: Dict[str, Any],
    fewshot_examples: Optional[List[Dict[str, Any]]],
    num_examples: int,
) -> List[Dict[str, Any]]:
    pool = list(LLM_PLANNER_SEED_EXAMPLES)
    if fewshot_examples:
        current_id = item.get("id")
        pool.extend([ex for ex in fewshot_examples if ex.get("id") != current_id])

    scored = [
        (_lexical_similarity(item.get("instruction", ""), ex.get("instruction", "")), idx, ex)
        for idx, ex in enumerate(pool)
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))

    selected: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _score, _idx, example in scored:
        example_id = example.get("id") or example.get("instruction", "")
        if example_id in seen_ids:
            continue
        seen_ids.add(example_id)
        selected.append(example)
        if len(selected) >= num_examples:
            break
    return selected


def _format_plan_lines(plan: Sequence[str]) -> str:
    return "\n".join(plan)


def _format_example(example: Dict[str, Any]) -> str:
    return (
        f"Task description: {example['instruction']}\n"
        "Completed plans:\n"
        f"Visible objects are {_list_text(example.get('objects') or [])}\n"
        f"Valid destinations are {_list_text(example.get('destinations') or [])}\n"
        "Next Plans:\n"
        f"{_format_plan_lines(example.get('plan') or [])}\n"
    )


def _build_system_prompt(max_steps: int) -> str:
    return (
        "You are an embodied high-level planner using an LLM-Planner-style prompt.\n"
        "Create the next high-level plan for a household pick-and-place task.\n"
        "Allowed executable actions:\n"
        "- robot.pick_and_place(object, destination)\n"
        "- done()\n"
        "Rules:\n"
        "- Use only objects listed under Visible objects.\n"
        "- Use only destinations listed under Valid destinations.\n"
        "- Move one object per action.\n"
        "- Do not invent objects, destinations, wrappers, or explanations.\n"
        f"- Use at most {max_steps} robot.pick_and_place actions.\n"
        "- End with done().\n"
        "Return only the Next Plans lines, one action per line.\n"
    )


def _build_user_prompt(
    item: Dict[str, Any],
    destinations: List[str],
    examples: List[Dict[str, Any]],
    diagnostics: str = "",
) -> str:
    found_objects = item.get("objects") or []
    example_text = "\n".join(_format_example(example) for example in examples)
    diagnostic_text = f"\nPrevious validation feedback:\n{diagnostics}\n" if diagnostics else ""
    return (
        "Create a high-level plan for completing a household task using the allowed "
        "actions and visible objects.\n\n"
        f"{example_text}\n"
        f"Task description: {item['instruction']}\n"
        "Completed plans:\n"
        f"Visible objects are {_list_text(found_objects)}\n"
        f"Valid destinations are {_list_text(destinations)}\n"
        f"{category_guidance(item, destinations)}"
        f"{diagnostic_text}"
        "Next Plans:\n"
    )


def _normalize_plan_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        return ""
    stripped = stripped.strip("`").strip()
    stripped = re.sub(r"^(Next Plans|Plan|Program|Actions)\s*:\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^\d+[\).\s-]+", "", stripped)
    stripped = stripped.strip("- ").strip()
    if stripped.endswith("."):
        stripped = stripped[:-1].strip()
    return stripped


def _parse_pick_and_place(line: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"robot\.pick_and_place\((.+),\s*(.+)\)", line)
    if not match:
        return None
    obj = match.group(1).strip()
    dest = match.group(2).strip()
    if not obj or not dest:
        return None
    return obj, dest


def _execute_plan(
    plan_text: str,
    found_objects: List[str],
    destinations: List[str],
    max_steps: int,
) -> Dict[str, Any]:
    accepted_actions: List[str] = []
    moved_objects: set[str] = set()
    step_records: List[Dict[str, Any]] = []
    invalid_actions = 0
    parse_status = "no_valid_action"
    termination_reason = "program_end"

    for idx, raw_line in enumerate(plan_text.splitlines(), start=1):
        line = _normalize_plan_line(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            continue

        if line == "done()":
            termination_reason = "done"
            step_records.append(
                {
                    "line_number": idx,
                    "raw_line": raw_line,
                    "normalized_line": line,
                    "kind": "done",
                    "accepted": True,
                    "observation": "plan finished",
                }
            )
            if not accepted_actions:
                parse_status = "done_only"
            elif parse_status == "no_valid_action":
                parse_status = "ok"
            break

        parsed = _parse_pick_and_place(line)
        if parsed is None:
            invalid_actions += 1
            step_records.append(
                {
                    "line_number": idx,
                    "raw_line": raw_line,
                    "normalized_line": line,
                    "kind": "invalid",
                    "accepted": False,
                    "observation": "invalid statement",
                }
            )
            continue

        obj, dest = parsed
        if len(accepted_actions) >= max_steps:
            termination_reason = "max_steps"
            step_records.append(
                {
                    "line_number": idx,
                    "raw_line": raw_line,
                    "normalized_line": line,
                    "kind": "action",
                    "accepted": False,
                    "observation": "max executable actions reached",
                }
            )
            break

        if obj not in found_objects:
            invalid_actions += 1
            accepted = False
            observation = f"invalid object '{obj}'"
        elif dest not in destinations:
            invalid_actions += 1
            accepted = False
            observation = f"invalid destination '{dest}'"
        elif line in accepted_actions:
            invalid_actions += 1
            accepted = False
            observation = "duplicate action"
        elif obj in moved_objects:
            invalid_actions += 1
            accepted = False
            observation = f"object '{obj}' already moved"
        else:
            accepted_actions.append(line)
            moved_objects.add(obj)
            accepted = True
            observation = "accepted action"
            parse_status = "ok"

        step_records.append(
            {
                "line_number": idx,
                "raw_line": raw_line,
                "normalized_line": line,
                "kind": "action",
                "accepted": accepted,
                "observation": observation,
            }
        )

    pred_actions = list(accepted_actions)
    if termination_reason == "done":
        pred_actions.append("done()")
    elif pred_actions and pred_actions[-1] != "done()":
        pred_actions.append("done()")

    return {
        "pred_actions": pred_actions,
        "parse_status": parse_status,
        "termination_reason": termination_reason,
        "num_invalid_actions": invalid_actions,
        "step_records": step_records,
    }


def _diagnostics_from_execution(execution: Dict[str, Any]) -> str:
    issues = []
    for record in execution["step_records"]:
        if not record.get("accepted", False):
            issues.append(f"- {record['normalized_line']}: {record['observation']}")
    if not issues:
        return ""
    return "Fix these invalid Next Plans lines:\n" + "\n".join(issues[:8])


def run_llm_planner_item(
    item: Dict[str, Any],
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
    max_retries: int = 2,
    verbose: bool = False,
    policy_meta: Optional[Dict[str, Any]] = None,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
    num_examples: int = 3,
) -> Dict[str, Any]:
    found_objects = item.get("objects") or []
    destinations = _planner_destinations(item)
    selected_examples = _select_fewshot_examples(item, fewshot_examples, num_examples)

    client = get_openai_client()
    system_prompt = _build_system_prompt(max_steps)
    total_start = time.monotonic()

    best_execution: Optional[Dict[str, Any]] = None
    best_plan = ""
    diagnostics = ""
    attempts_used = 0

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1
        user_prompt = _build_user_prompt(item, destinations, selected_examples, diagnostics=diagnostics)
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        log_openai_call("llm_planner_baseline", item["instruction"], time.monotonic() - start)
        plan_text = response.choices[0].message.content.strip()
        execution = _execute_plan(plan_text, found_objects, destinations, max_steps)

        if verbose:
            print(f"[attempt {attempt + 1}]")
            print(plan_text)

        if best_execution is None:
            best_execution = execution
            best_plan = plan_text
        else:
            best_score = (
                len(best_execution["pred_actions"]),
                1 if best_execution["termination_reason"] == "done" else 0,
                -best_execution["num_invalid_actions"],
            )
            new_score = (
                len(execution["pred_actions"]),
                1 if execution["termination_reason"] == "done" else 0,
                -execution["num_invalid_actions"],
            )
            if new_score > best_score:
                best_execution = execution
                best_plan = plan_text

        if execution["parse_status"] in {"ok", "done_only"} and execution["termination_reason"] == "done":
            break
        diagnostics = _diagnostics_from_execution(execution)

    assert best_execution is not None

    gold_score = score_with_gold(item, best_execution["pred_actions"], policy_meta or {})
    if gold_score is None:
        success = llm_judge(item["instruction"], "\n".join(best_execution["pred_actions"]), model=model)
    else:
        success = gold_score

    return {
        "raw_plan": best_plan,
        "pred_actions": best_execution["pred_actions"],
        "success": success,
        "parse_status": best_execution["parse_status"],
        "termination_reason": best_execution["termination_reason"],
        "num_actions": len([a for a in best_execution["pred_actions"] if a != "done()"]),
        "num_invalid_actions": best_execution["num_invalid_actions"],
        "latency_seconds": round(time.monotonic() - total_start, 3),
        "step_records": best_execution["step_records"],
        "destinations": destinations,
        "fewshot_example_ids": [ex.get("id", "") for ex in selected_examples],
        "num_attempts": attempts_used,
    }
