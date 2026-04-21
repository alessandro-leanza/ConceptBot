import re
import time
import json
from typing import Iterable, List

from scripts.modules.semantic_cache import get_openai_client, log_openai_call


FEWSHOT_EXAMPLES = """
Example 1
Scene objects: chips, coke, apple, user, trash can
Instruction: Bring me something to drink.
Output:
{"steps":["robot.pick_and_place(coke, user)","done()"]}

Example 2
Scene objects: chips, apple, water bottle, user, trash can
Instruction: Throw away the snack and bring me something healthy.
Output:
{"steps":["robot.pick_and_place(chips, trash can)","robot.pick_and_place(apple, user)","done()"]}

Example 3
Scene objects: sponge, table, user, trash can
Instruction: I spilled a drink on the table. Help me clean it.
Output:
{"steps":["robot.pick_and_place(sponge, user)","done()"]}
""".strip()


def _list_text(values: Iterable[str]) -> str:
    return ", ".join(sorted(values))


def _extract_actions(text: str, allowed_actions: set[str]) -> List[str]:
    try:
        payload = json.loads(text)
        raw_steps = payload.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []
    except Exception:
        raw_steps = []

    actions: List[str] = []
    for raw_line in raw_steps:
        line = str(raw_line).strip().strip("`")
        if not line:
            continue
        if line == "done()":
            actions.append(line)
            break
        if re.fullmatch(r"robot\.pick_and_place\([^()]+,\s*[^()]+\)", line):
            if line in allowed_actions:
                actions.append(line)
    if actions and actions[-1] != "done()":
        actions.append("done()")
    return actions


def DIRECT(
    found_objects: List[str],
    PICK_TARGETS,
    PLACE_TARGETS,
    query: str,
    model: str = "gpt-4o-mini",
    max_tasks: int = 12,
) -> List[str]:
    allowed_actions = {
        f"robot.pick_and_place({pick}, {place})"
        for pick in PICK_TARGETS
        for place in PLACE_TARGETS
    }

    system_message = (
        "You are a robot task planner.\n"
        "The robot can only perform pick-and-place actions.\n"
        "You must return JSON only.\n"
        "Return exactly one JSON object with a key named 'steps'.\n"
        "The value of 'steps' must be a JSON array of strings.\n"
        "Each string must be either:\n"
        "- robot.pick_and_place(object, destination)\n"
        "- done()\n"
        "Rules:\n"
        "- Use only objects present in the scene.\n"
        "- Use only destinations explicitly listed as valid destinations.\n"
        "- Do not invent objects or destinations.\n"
        "- Do not explain your reasoning.\n"
        "- Do not output natural-language instructions.\n"
        "- Do not combine multiple objects in one step.\n"
        "- Each pick-and-place action must move exactly one object to exactly one destination.\n"
        "- If two objects must go to the same destination, output two separate robot.pick_and_place steps.\n"
        "- Always terminate the plan with done().\n"
        "- If you cannot find any valid action, return {\"steps\":[\"done()\"]}.\n"
        "- If multiple plans are valid, choose one concise valid plan.\n"
        f"- Use at most {max_tasks} pick-and-place actions before done()."
    )

    user_message = (
        f"{FEWSHOT_EXAMPLES}\n\n"
        f"Scene objects: {_list_text(found_objects)}\n"
        f"Pickable objects: {_list_text(PICK_TARGETS)}\n"
        f"Valid destinations: {_list_text(PLACE_TARGETS)}\n"
        "Return JSON only.\n"
        f"Instruction: {query}\n"
        "Output:"
    )

    client = get_openai_client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    log_openai_call("planner_direct", query, time.monotonic() - start)

    content = response.choices[0].message.content.strip()
    return _extract_actions(content, allowed_actions)
