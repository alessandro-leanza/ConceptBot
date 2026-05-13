import re
import time
from typing import Any, Dict, List, Optional, Tuple

from scripts.modules.semantic_cache import get_openai_client, log_openai_call


REACT_FEWSHOT = """Example:
Scene objects: coke, chips, user, trash can
Valid destinations: user, trash can
Instruction: Give the coke to the user and throw the chips away.
Thought 1: I should move the coke to the user first.
Action 1: robot.pick_and_place(coke, user)
Observation 1: Accepted action robot.pick_and_place(coke, user).
Thought 2: I should throw the chips in the trash can.
Action 2: robot.pick_and_place(chips, trash can)
Observation 2: Accepted action robot.pick_and_place(chips, trash can).
Thought 3: The requested actions are complete.
Action 3: done()
Observation 3: Plan finished.
"""

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


def _list_text(values: List[str]) -> str:
    return ", ".join(values) if values else "(none)"


def category_guidance(item: Dict[str, Any], destinations: List[str]) -> str:
    category = item.get("category")
    destination_text = _list_text(destinations)
    if category == "materials":
        return (
            "Category-specific guidance:\n"
            "- This is a material-sorting task.\n"
            f"- Treat these destinations as containers, not objects to move: {destination_text}.\n"
            "- Move only sortable items; do not move destination bins or containers.\n"
            "- Infer material from the object name using ordinary commonsense.\n"
            "- Use plastic bin for plastic items, glass bin for glass items, paper bin for paper/cardboard items, and wax bin for wax items when those destinations are available.\n"
            "- Use mixed bin for multi-material, coated, composite, or doubtful items when that destination is available.\n"
        )
    if category == "toxicity":
        return (
            "Category-specific guidance:\n"
            "- This is a toxicity or hazard-sorting task.\n"
            f"- Treat these destinations as containers or areas, not objects to move: {destination_text}.\n"
            "- Move only plants, animals, or substances; do not move destination bins or areas.\n"
            "- Classify each movable item by commonsense toxicity, poisonousness, venom, or chemical hazard.\n"
            "- Use toxic, hazardous, safety, or secure venomous destinations for clearly toxic, poisonous, venomous, or hazardous items.\n"
            "- Use standard, non-hazardous, or public destinations for clearly safe/non-toxic items.\n"
            "- Use mixed or verification destinations for uncertain or doubtful items when those destinations are available.\n"
        )
    return ""


def _planner_destinations(item: Dict[str, Any]) -> List[str]:
    found_objects = item.get("objects") or []
    gold_destinations = _gold_destinations(item)
    destinations = []
    for obj in found_objects:
        if obj in PLANNER_DESTINATIONS or obj in gold_destinations:
            destinations.append(obj)
    for dest in sorted(gold_destinations):
        if dest not in destinations:
            destinations.append(dest)
    return destinations


def _gold_destinations(item: Dict[str, Any]) -> set[str]:
    gold = item.get("gold") or {}
    destinations = set((gold.get("rules") or {}).keys())
    for seq in gold.get("sequences", []):
        for act in seq:
            if not isinstance(act, str) or not act.startswith("robot.pick_and_place("):
                continue
            inner = act[len("robot.pick_and_place("):-1]
            try:
                _obj, dest = [s.strip() for s in inner.split(",")]
            except Exception:
                continue
            destinations.add(dest)
    return destinations


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


def score_with_gold(item: Dict[str, Any], pred_actions: List[str], policy_meta: Dict[str, Any]) -> Optional[int]:
    gold = item.get("gold")
    if not gold:
        return None
    order_matters = policy_meta.get("order_matters", True)
    if "sequences" in gold:
        return 1 if _match_sequences(pred_actions, gold["sequences"], order_matters) else 0
    if "rules" in gold:
        return 1 if _match_rules(pred_actions, gold["rules"]) else 0
    return None


def llm_judge(instruction: str, action: str, model: str = "gpt-4o-mini") -> int:
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
        response_format={"type": "json_object"},
    )
    log_openai_call("react_baseline_judge", instruction, time.monotonic() - start)
    content = resp.choices[0].message.content.strip()
    try:
        import json

        data = json.loads(content)
        return 1 if int(data.get("score", 0)) == 1 else 0
    except Exception:
        return 0


def _build_system_prompt(max_steps: int) -> str:
    return (
        "You are a robot planner using ReAct-style reasoning.\n"
        "You must solve the task with interleaved Thought, Action, Observation steps.\n"
        "You are allowed only two actions:\n"
        "- robot.pick_and_place(object, destination)\n"
        "- done()\n"
        "Rules:\n"
        "- Use only objects listed in the scene.\n"
        "- Use only valid destinations listed in the prompt.\n"
        "- Move one object per action.\n"
        "- Do not invent objects or destinations.\n"
        "- Observations are provided by the environment and may tell you an action is invalid.\n"
        f"- Finish within at most {max_steps} action steps.\n"
        "Return exactly two lines for the next step:\n"
        "Thought N: <brief reasoning>\n"
        "Action N: <robot.pick_and_place(object, destination) or done()>\n"
        "Do not output Observation lines yourself.\n"
        "Do not output JSON.\n"
    )


def _build_user_prompt(
    item: Dict[str, Any],
    destinations: List[str],
    step_idx: int,
    transcript: List[str],
    max_steps: int,
) -> str:
    found_objects = item.get("objects") or []
    history = "\n".join(transcript).strip()
    if not history:
        history = "Observation 0: Scene initialized."

    return (
        f"{REACT_FEWSHOT}\n"
        f"Scene objects: {_list_text(found_objects)}\n"
        f"Valid destinations: {_list_text(destinations)}\n"
        f"{category_guidance(item, destinations)}"
        f"Instruction: {item['instruction']}\n"
        f"Maximum actions: {max_steps}\n\n"
        f"Current trajectory:\n{history}\n\n"
        f"Continue with step {step_idx}.\n"
        f"Return only:\nThought {step_idx}: ...\nAction {step_idx}: ...\n"
    )


def _extract_step(response_text: str, step_idx: int) -> Tuple[str, Optional[str]]:
    thought_match = re.search(
        rf"Thought\s*{step_idx}\s*:\s*(.+?)(?:\n|$)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        rf"Action\s*{step_idx}\s*:\s*(.+?)(?:\n|$)",
        response_text,
        flags=re.IGNORECASE,
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else None
    return thought, action


def _parse_pick_and_place(action: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"robot\.pick_and_place\((.+),\s*(.+)\)", action)
    if not match:
        return None
    obj = match.group(1).strip()
    dest = match.group(2).strip()
    if not obj or not dest:
        return None
    return obj, dest


def _observe_action(
    action: Optional[str],
    found_objects: List[str],
    destinations: List[str],
    accepted_actions: List[str],
    moved_objects: set[str],
) -> Tuple[str, bool, bool]:
    if not action:
        return "Invalid action: missing Action line.", False, False

    if action == "done()":
        return "Plan finished.", True, False

    parsed = _parse_pick_and_place(action)
    if parsed is None:
        return (
            "Invalid action: expected robot.pick_and_place(object, destination) or done().",
            False,
            False,
        )

    obj, dest = parsed
    if obj not in found_objects:
        return f"Invalid action: object '{obj}' is not in the scene.", False, False
    if dest not in destinations:
        return f"Invalid action: destination '{dest}' is not valid for this scene.", False, False
    if action in accepted_actions:
        return f"Invalid action: {action} was already executed.", False, False
    if obj in moved_objects:
        return f"Invalid action: object '{obj}' was already moved.", False, False

    accepted_actions.append(action)
    moved_objects.add(obj)
    return f"Accepted action {action}.", False, True


def run_react_item(
    item: Dict[str, Any],
    model: str = "gpt-4o-mini",
    max_steps: int = 6,
    max_retries: int = 2,
    verbose: bool = False,
    policy_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    found_objects = item.get("objects") or []
    destinations = _planner_destinations(item)
    transcript = ["Observation 0: Scene initialized."]
    accepted_actions: List[str] = []
    moved_objects: set[str] = set()
    invalid_actions = 0
    termination_reason = "max_steps"
    parse_status = "no_valid_action"
    raw_steps: List[Dict[str, str]] = []

    client = get_openai_client()
    system_prompt = _build_system_prompt(max_steps)
    total_start = time.monotonic()

    for step_idx in range(1, max_steps + 1):
        user_prompt = _build_user_prompt(item, destinations, step_idx, transcript, max_steps)

        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        log_openai_call("react_baseline", item["instruction"], time.monotonic() - start)
        content = response.choices[0].message.content.strip()

        thought, action = _extract_step(content, step_idx)
        observation, is_done, accepted = _observe_action(
            action,
            found_objects,
            destinations,
            accepted_actions,
            moved_objects,
        )

        raw_steps.append(
            {
                "step": str(step_idx),
                "thought": thought,
                "action": action or "",
                "observation": observation,
                "raw_model_output": content,
            }
        )
        transcript.extend(
            [
                f"Thought {step_idx}: {thought}".rstrip(),
                f"Action {step_idx}: {action or ''}".rstrip(),
                f"Observation {step_idx}: {observation}",
            ]
        )

        if verbose:
            print(transcript[-3])
            print(transcript[-2])
            print(transcript[-1])

        if accepted:
            parse_status = "ok"
            invalid_actions = 0
        else:
            invalid_actions += 1
            if action == "done()" and not accepted_actions:
                parse_status = "done_only"
            elif accepted_actions:
                parse_status = "ok"
            else:
                parse_status = "no_valid_action"

        if is_done:
            termination_reason = "done"
            break
        if invalid_actions > max_retries:
            termination_reason = "max_retries"
            break
    else:
        termination_reason = "max_steps"

    pred_actions = list(accepted_actions)
    if termination_reason == "done":
        pred_actions.append("done()")
    elif pred_actions and pred_actions[-1] != "done()":
        pred_actions.append("done()")

    gold_score = score_with_gold(item, pred_actions, policy_meta or {})
    if gold_score is None:
        success = llm_judge(item["instruction"], "\n".join(pred_actions), model=model)
    else:
        success = gold_score

    raw_trace = "\n".join(transcript)

    return {
        "raw_trace": raw_trace,
        "pred_actions": pred_actions,
        "success": success,
        "parse_status": parse_status,
        "termination_reason": termination_reason,
        "num_actions": len([a for a in pred_actions if a != "done()"]),
        "num_invalid_actions": invalid_actions,
        "destinations": destinations,
        "step_records": raw_steps,
        "latency_seconds": round(time.monotonic() - total_start, 3),
    }
