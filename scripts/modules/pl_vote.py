import re
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.modules.semantic_cache import get_openai_client, log_openai_call


FEWSHOT_EXAMPLES = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one on the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
""".strip()


DESTINATION_ONLY_NAMES = {
    "user",
    "child",
    "table",
    "counter",
    "trash can",
    "fridge",
    "dishwasher",
    "handwashing",
    "microwave oven",
    "plastic bin",
    "glass bin",
    "paper bin",
    "mixed bin",
    "wax bin",
    "toxic bin",
    "standard bin",
    "secure venomous bin",
    "public area bin",
    "verification bin",
    "hazardous bin",
    "non-hazardous bin",
}


def _list_text(values: Iterable[str]) -> str:
    return ", ".join(sorted(str(value) for value in values))


def make_options(
    pick_targets: Iterable[str],
    place_targets: Iterable[str],
    termination_string: str = "done()",
) -> List[str]:
    options = [
        f"robot.pick_and_place({pick}, {place})"
        for pick in pick_targets
        for place in place_targets
        if pick != place
    ]
    options.append(termination_string)
    return options


def _parse_pick(action: str) -> Optional[str]:
    if not action.startswith("robot.pick_and_place("):
        return None
    inner = action[len("robot.pick_and_place("):-1]
    try:
        pick, _ = [part.strip() for part in inner.split(",", 1)]
    except ValueError:
        return None
    return pick


def _requested_actions_from_query(query: str, allowed_actions: Sequence[str]) -> List[str]:
    ordered = []
    for action in allowed_actions:
        if action != "done()" and action in query:
            ordered.append((query.index(action), action))
    return [action for _, action in sorted(ordered)]


def _extract_action(text: str, allowed_actions: Sequence[str]) -> Optional[str]:
    content = text.strip().strip("`")
    if not content:
        return None

    allowed = set(allowed_actions)
    for line in content.splitlines():
        candidate = line.strip().strip("`")
        if candidate in allowed:
            return candidate

    for action in allowed_actions:
        if action in content:
            return action

    match = re.search(r"robot\.pick_and_place\([^()]+,\s*[^()]+\)|done\(\)", content)
    if match and match.group(0) in allowed:
        return match.group(0)
    return None


def _build_system_prompt(max_tasks: int) -> str:
    return (
        "You are a robot task planner.\n"
        "At each step, choose exactly one next action from the available actions.\n"
        "The robot can only perform pick-and-place actions and done().\n"
        "Return exactly one action string and no other text.\n"
        "Rules:\n"
        "- Use only actions listed between BEGIN AVAILABLE ACTIONS and END AVAILABLE ACTIONS.\n"
        "- Use done() only when the user request is fully satisfied.\n"
        "- Do not invent objects or destinations.\n"
        "- Do not explain your reasoning.\n"
        "- Do not combine multiple actions in one response.\n"
        "- Do not move an object that has already been moved in the executed action history.\n"
        "- Do not move bins, containers, areas, appliances, or people; they are destinations only.\n"
        "- If the user request contains explicit robot.pick_and_place actions that are listed as available actions, choose the first one that is not already in the executed history.\n"
        "- When all requested pick-and-place instructions have been executed, return done().\n"
        "- Do not add extra clean-up, sorting, or placement actions that are not requested.\n"
        f"- The full plan may contain at most {max_tasks} pick-and-place actions before done()."
    )


def _build_user_prompt(
    found_objects: Sequence[str],
    pick_targets: Sequence[str],
    place_targets: Sequence[str],
    query: str,
    available_actions: Sequence[str],
    history: Sequence[str],
) -> str:
    history_text = "\n".join(history) if history else "none"
    return (
        f"BEGIN EXAMPLES\n{FEWSHOT_EXAMPLES}\nEND EXAMPLES\n\n"
        f"User request:\n{query}\n\n"
        f"Detected objects: {_list_text(found_objects)}\n"
        f"Pickable objects: {_list_text(pick_targets)}\n"
        f"Valid destinations: {_list_text(place_targets)}\n\n"
        f"Executed action history:\n{history_text}\n\n"
        "BEGIN AVAILABLE ACTIONS\n"
        + "\n".join(available_actions)
        + "\nEND AVAILABLE ACTIONS\n\n"
        "Next action:"
    )


def _vote_next_action(
    found_objects: Sequence[str],
    pick_targets: Sequence[str],
    place_targets: Sequence[str],
    query: str,
    available_actions: Sequence[str],
    history: Sequence[str],
    model: str,
    voting_samples: int,
    temperature: float,
    max_tasks: int,
) -> Tuple[Optional[str], Dict[str, object]]:
    client = get_openai_client()
    system_prompt = _build_system_prompt(max_tasks)
    user_prompt = _build_user_prompt(
        found_objects,
        pick_targets,
        place_targets,
        query,
        available_actions,
        history,
    )

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        n=voting_samples,
        max_tokens=80,
    )
    log_openai_call("planner_vote", query, time.monotonic() - start)

    raw_outputs = [choice.message.content.strip() for choice in response.choices]
    parsed_actions = [_extract_action(output, available_actions) for output in raw_outputs]
    valid_actions = [action for action in parsed_actions if action is not None]
    counts = Counter(valid_actions)
    selected = None
    if counts:
        selected = sorted(counts.items(), key=lambda item: (-item[1], available_actions.index(item[0])))[0][0]

    scores = {action: count / voting_samples for action, count in sorted(counts.items())}
    return selected, {
        "raw_outputs": raw_outputs,
        "parsed_actions": parsed_actions,
        "scores": scores,
        "selected_action": selected,
        "num_valid_votes": len(valid_actions),
        "num_samples": voting_samples,
        "temperature": temperature,
    }


def VOTE_WITH_TRACE(
    found_objects: List[str],
    PICK_TARGETS,
    PLACE_TARGETS,
    query: str,
    model: str = "gpt-4o-mini",
    max_tasks: int = 12,
    voting_samples: int = 5,
    temperature: float = 0,
) -> Dict[str, object]:
    pick_targets = [pick for pick in PICK_TARGETS if pick not in DESTINATION_ONLY_NAMES]
    if not pick_targets:
        pick_targets = list(PICK_TARGETS)
    place_targets = list(PLACE_TARGETS)
    all_options = make_options(pick_targets, place_targets)
    requested_actions = _requested_actions_from_query(query, all_options)
    steps: List[str] = []
    traces: List[Dict[str, object]] = []
    moved_objects: set[str] = set()
    selected = ""

    while selected != "done()" and len([s for s in steps if s != "done()"]) < max_tasks:
        if requested_actions:
            available_actions = [
                option
                for option in requested_actions
                if option not in steps and _parse_pick(option) not in moved_objects
            ] + ["done()"]
        else:
            available_actions = [
                option
                for option in all_options
                if option == "done()" or _parse_pick(option) not in moved_objects
            ]
        selected, trace = _vote_next_action(
            found_objects=found_objects,
            pick_targets=pick_targets,
            place_targets=place_targets,
            query=query,
            available_actions=available_actions,
            history=steps,
            model=model,
            voting_samples=voting_samples,
            temperature=temperature,
            max_tasks=max_tasks,
        )
        traces.append(trace)
        if not selected:
            break
        steps.append(selected)
        moved_pick = _parse_pick(selected)
        if moved_pick:
            moved_objects.add(moved_pick)

    if steps and steps[-1] != "done()":
        steps.append("done()")
    if not steps:
        steps = ["done()"]

    return {
        "steps": steps,
        "step_traces": traces,
        "voting_samples": voting_samples,
        "temperature": temperature,
        "model": model,
    }


def VOTE(
    found_objects: List[str],
    PICK_TARGETS,
    PLACE_TARGETS,
    query: str,
    model: str = "gpt-4o-mini",
    max_tasks: int = 12,
    voting_samples: int = 5,
    temperature: float = 0,
) -> List[str]:
    return VOTE_WITH_TRACE(
        found_objects=found_objects,
        PICK_TARGETS=PICK_TARGETS,
        PLACE_TARGETS=PLACE_TARGETS,
        query=query,
        model=model,
        max_tasks=max_tasks,
        voting_samples=voting_samples,
        temperature=temperature,
    )["steps"]
