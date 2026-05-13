import re
import time
from typing import Any, Dict, List, Optional, Tuple

from scripts.experiments.baselines.react_baseline import (
    _planner_destinations,
    category_guidance,
    llm_judge,
    score_with_gold,
)
from scripts.modules.semantic_cache import get_openai_client, log_openai_call


PROGPROMPT_FEWSHOT = """Example:
Scene objects: coke, chips, user, trash can
Valid destinations: user, trash can
Instruction: Give the coke to the user and throw the chips away.
Program:
# goal: satisfy the instruction with concise robot actions
assert(object_exists("coke"))
assert(destination_exists("user"))
robot.pick_and_place(coke, user)
assert(object_exists("chips"))
assert(destination_exists("trash can"))
robot.pick_and_place(chips, trash can)
done()
"""


SUPPORTED_ASSERTIONS = {
    "object_exists",
    "destination_exists",
    "not_moved",
}


def _list_text(values: List[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _build_system_prompt(max_steps: int) -> str:
    return (
        "You are a robot planner writing a short program.\n"
        "Write a concise program that solves the task.\n"
        "Allowed executable statements:\n"
        "- robot.pick_and_place(object, destination)\n"
        "- done()\n"
        "Allowed helper statements:\n"
        "- assert(object_exists(\"name\"))\n"
        "- assert(destination_exists(\"name\"))\n"
        "- assert(not_moved(\"name\"))\n"
        "- comments starting with #\n"
        "Rules:\n"
        "- Use one statement per line.\n"
        "- Use only scene objects listed in the prompt.\n"
        "- Use only valid destinations listed in the prompt.\n"
        "- Move one object per action.\n"
        "- Do not invent objects or destinations.\n"
        f"- Use at most {max_steps} robot.pick_and_place actions before done().\n"
        "- End the program with done().\n"
        "- Do not output explanations outside the program.\n"
    )


def _build_user_prompt(
    item: Dict[str, Any],
    destinations: List[str],
    max_steps: int,
    diagnostics: str = "",
) -> str:
    found_objects = item.get("objects") or []
    extra = f"\nPrevious validation feedback:\n{diagnostics}\n" if diagnostics else ""
    return (
        f"{PROGPROMPT_FEWSHOT}\n"
        f"Scene objects: {_list_text(found_objects)}\n"
        f"Valid destinations: {_list_text(destinations)}\n"
        f"{category_guidance(item, destinations)}"
        f"Instruction: {item['instruction']}\n"
        f"Maximum executable actions: {max_steps}\n"
        f"{extra}\n"
        "Program:\n"
    )


def _normalize_program_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    stripped = re.sub(r"^(Program|Plan|Explanation)\s*:\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^\d+[\).\s-]+", "", stripped)
    return stripped.strip()


def _parse_assertion(line: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r'assert\((\w+)\("([^"]+)"\)\)', line)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _parse_pick_and_place(line: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"robot\.pick_and_place\((.+),\s*(.+)\)", line)
    if not match:
        return None
    obj = match.group(1).strip()
    dest = match.group(2).strip()
    if not obj or not dest:
        return None
    return obj, dest


def _evaluate_assertion(
    assertion_name: str,
    argument: str,
    found_objects: List[str],
    destinations: List[str],
    moved_objects: set[str],
) -> Tuple[str, bool]:
    if assertion_name == "object_exists":
        passed = argument in found_objects
        return (f"assert object_exists('{argument}') -> {passed}", passed)
    if assertion_name == "destination_exists":
        passed = argument in destinations
        return (f"assert destination_exists('{argument}') -> {passed}", passed)
    if assertion_name == "not_moved":
        passed = argument not in moved_objects
        return (f"assert not_moved('{argument}') -> {passed}", passed)
    return (f"unsupported assertion '{assertion_name}' ignored", True)


def _execute_program(
    program_text: str,
    found_objects: List[str],
    destinations: List[str],
    max_steps: int,
) -> Dict[str, Any]:
    raw_lines = program_text.splitlines()
    accepted_actions: List[str] = []
    moved_objects: set[str] = set()
    step_records: List[Dict[str, Any]] = []
    invalid_actions = 0
    assertion_count = 0
    unsupported_assertions = 0
    termination_reason = "program_end"
    parse_status = "no_valid_action"

    for idx, raw_line in enumerate(raw_lines, start=1):
        line = _normalize_program_line(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            step_records.append(
                {
                    "line_number": idx,
                    "raw_line": raw_line,
                    "normalized_line": line,
                    "kind": "comment",
                    "accepted": True,
                    "observation": "comment ignored",
                }
            )
            continue

        if line.startswith("assert("):
            assertion = _parse_assertion(line)
            if assertion is None:
                invalid_actions += 1
                step_records.append(
                    {
                        "line_number": idx,
                        "raw_line": raw_line,
                        "normalized_line": line,
                        "kind": "assert",
                        "accepted": False,
                        "observation": "invalid assert syntax",
                    }
                )
                continue
            assertion_name, argument = assertion
            assertion_count += 1
            observation, passed = _evaluate_assertion(
                assertion_name,
                argument,
                found_objects,
                destinations,
                moved_objects,
            )
            if assertion_name not in SUPPORTED_ASSERTIONS:
                unsupported_assertions += 1
            step_records.append(
                {
                    "line_number": idx,
                    "raw_line": raw_line,
                    "normalized_line": line,
                    "kind": "assert",
                    "accepted": passed,
                    "observation": observation,
                }
            )
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
                    "observation": "program finished",
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
            observation = f"invalid object '{obj}'"
            accepted = False
        elif dest not in destinations:
            invalid_actions += 1
            observation = f"invalid destination '{dest}'"
            accepted = False
        elif line in accepted_actions:
            invalid_actions += 1
            observation = "duplicate action"
            accepted = False
        elif obj in moved_objects:
            invalid_actions += 1
            observation = f"object '{obj}' already moved"
            accepted = False
        else:
            accepted_actions.append(line)
            moved_objects.add(obj)
            observation = "accepted action"
            accepted = True
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
        "assertion_count": assertion_count,
        "unsupported_assertions": unsupported_assertions,
        "step_records": step_records,
    }


def _diagnostics_from_execution(execution: Dict[str, Any]) -> str:
    issues = []
    for record in execution["step_records"]:
        if not record.get("accepted", False):
            issues.append(f"- {record['normalized_line']}: {record['observation']}")
    if not issues:
        return ""
    return "Fix these issues in the next program:\n" + "\n".join(issues[:8])


def run_progprompt_item(
    item: Dict[str, Any],
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
    max_retries: int = 2,
    verbose: bool = False,
    policy_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    found_objects = item.get("objects") or []
    destinations = _planner_destinations(item)
    client = get_openai_client()
    system_prompt = _build_system_prompt(max_steps)
    total_start = time.monotonic()

    best_execution: Optional[Dict[str, Any]] = None
    best_program = ""
    diagnostics = ""
    attempts_used = 0

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1
        user_prompt = _build_user_prompt(item, destinations, max_steps, diagnostics=diagnostics)
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        log_openai_call("progprompt_baseline", item["instruction"], time.monotonic() - start)
        program_text = response.choices[0].message.content.strip()
        execution = _execute_program(program_text, found_objects, destinations, max_steps)

        if verbose:
            print(f"[attempt {attempt + 1}]")
            print(program_text)

        if best_execution is None:
            best_execution = execution
            best_program = program_text
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
                best_program = program_text

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
        "raw_program": best_program,
        "pred_actions": best_execution["pred_actions"],
        "success": success,
        "parse_status": best_execution["parse_status"],
        "termination_reason": best_execution["termination_reason"],
        "num_actions": len([a for a in best_execution["pred_actions"] if a != "done()"]),
        "num_invalid_actions": best_execution["num_invalid_actions"],
        "latency_seconds": round(time.monotonic() - total_start, 3),
        "step_records": best_execution["step_records"],
        "destinations": destinations,
        "assertion_count": best_execution["assertion_count"],
        "unsupported_assertions": best_execution["unsupported_assertions"],
        "num_attempts": attempts_used,
    }
