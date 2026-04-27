import json
import re
import time
from typing import Any, Dict, List, Optional


NEAR_DUPLICATE_ALIASES = {
    "breakable": "fragile",
    "easily breakable": "fragile",
    "shatterable": "fragile",
    "liquid holding": "hold liquid",
    "holds liquid": "hold liquid",
    "contains liquid": "hold liquid",
    "toxic": "poisonous",
    "poison": "poisonous",
    "non toxic": "safe",
    "non-toxic": "safe",
    "heat proof": "heat-resistant",
    "heatproof": "heat-resistant",
    "heat resistant": "heat-resistant",
    "microwave safe": "microwave-safe",
    "microwave compatible": "microwave-safe",
    "not metal": "non-metallic",
    "non metallic": "non-metallic",
}


def normalize_property_name(name: str) -> str:
    """
    Normalize a short property label for matching and prompt use.

    Existing OPE properties include spaces (for example, "hold liquid"), while
    compatibility properties are often clearer with hyphens (for example,
    "microwave-safe"). This function keeps meaningful hyphens and normalizes
    surrounding whitespace without changing base properties passed by callers.
    """
    text = str(name).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _dedupe_key(name: str) -> str:
    normalized = normalize_property_name(name)
    alias = NEAR_DUPLICATE_ALIASES.get(normalized, normalized)
    return alias.replace("-", " ")


def _coerce_dynamic_name(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("name")
    return None


def merge_properties(
    base_properties: Optional[List[str]],
    dynamic_properties: Optional[List[Any]],
    max_dynamic_properties: int = 6,
) -> List[str]:
    """
    Merge fixed OPE properties with induced task properties.

    Base properties are preserved in their original order and spelling.
    Dynamic properties are normalized, deduplicated against base properties and
    simple aliases, capped, and appended. Base properties are never removed.
    """
    merged = list(base_properties or [])
    seen = {_dedupe_key(prop) for prop in merged}
    added = 0

    for item in dynamic_properties or []:
        raw_name = _coerce_dynamic_name(item)
        if not raw_name:
            continue
        normalized = normalize_property_name(raw_name)
        if not normalized:
            continue

        key = _dedupe_key(normalized)
        if key in seen:
            continue

        merged.append(normalized)
        seen.add(key)
        added += 1
        if added >= max_dynamic_properties:
            break

    return merged


def _fallback_result(
    base_properties: Optional[List[str]],
    source: str,
    rejected_properties: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "dynamic_properties": [],
        "merged_properties": list(base_properties or []),
        "rejected_properties": rejected_properties or [],
        "source": source,
    }


def _validate_dynamic_properties(raw_items: Any, max_properties: int) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    validated = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = normalize_property_name(item.get("name", ""))
        if not name:
            continue
        key = _dedupe_key(name)
        if key in seen:
            continue
        seen.add(key)
        try:
            priority = int(item.get("priority", len(validated) + 1))
        except Exception:
            priority = len(validated) + 1

        validated.append(
            {
                "name": name,
                "description": str(item.get("description", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "expected_value_type": str(item.get("expected_value_type", "yes_no")).strip() or "yes_no",
                "priority": priority,
            }
        )
        if len(validated) >= max_properties:
            break

    validated.sort(key=lambda item: item.get("priority", 999))
    return validated[:max_properties]


def _validate_rejected_properties(raw_items: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_items, list):
        return []

    rejected = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = normalize_property_name(item.get("name", ""))
        reason = str(item.get("reason", "")).strip()
        if name:
            rejected.append({"name": name, "reason": reason})
    return rejected


def induce_task_properties(
    user_instruction: str,
    detected_objects: List[str],
    base_properties: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    max_properties: int = 6,
    model: str = "gpt-4o-mini",
    llm_temperature: float = 0,
) -> Dict[str, Any]:
    """
    Induce a small set of task-relevant properties for optional OPE expansion.

    Returns a validated dictionary with dynamic properties, merged properties,
    rejected properties, and a source marker. Any API or JSON failure returns a
    safe fallback with no dynamic properties.
    """
    base_properties = list(base_properties or [])
    max_properties = max(0, int(max_properties))
    if max_properties == 0:
        return _fallback_result(base_properties, source="disabled")

    system_prompt = (
        "You are helping a robot planning system decide which object properties should be checked before executing "
        "a pick-and-place task.\n\n"
        "The robot can only pick objects and place them at destinations. Propose a small set of task-relevant object "
        "properties that would help decide which detected objects are suitable, safe, or risky for the user's instruction.\n\n"
        "Rules:\n"
        "- Return JSON only.\n"
        "- Do not include properties already covered by base_properties.\n"
        "- Prefer concise property names usable as commonsense retrieval targets.\n"
        "- Use properties that can plausibly be inferred from object names and commonsense relations.\n"
        "- Do not invent object-specific facts.\n"
        "- Do not propose more than max_properties dynamic properties.\n"
        "- Prefer safety, feasibility, compatibility, material, containment, and task-success properties.\n"
        "- Avoid vague properties such as useful, appropriate, good, or relevant.\n"
        "- Do not replace base properties; only add missing task-relevant properties."
    )
    user_payload = {
        "user_instruction": user_instruction,
        "detected_objects": list(detected_objects),
        "task_category": task_category,
        "base_properties": base_properties,
        "max_properties": max_properties,
        "output_schema": {
            "dynamic_properties": [
                {
                    "name": "short property name",
                    "description": "one sentence explaining what the property means",
                    "reason": "one sentence explaining why it matters for this task",
                    "expected_value_type": "yes_no",
                    "priority": 1,
                }
            ],
            "rejected_properties": [
                {
                    "name": "short property name",
                    "reason": "why it was not needed",
                }
            ],
        },
    }

    try:
        from scripts.modules.semantic_cache import get_openai_client, log_openai_call

        client = get_openai_client()
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            temperature=llm_temperature,
            response_format={"type": "json_object"},
        )
        log_openai_call("dynamic_properties", user_instruction, time.monotonic() - start)
        payload = json.loads(response.choices[0].message.content)
    except Exception as exc:
        return _fallback_result(
            base_properties,
            source="fallback",
            rejected_properties=[{"name": "dynamic_property_induction", "reason": f"{type(exc).__name__}: {exc}"}],
        )

    dynamic_items = _validate_dynamic_properties(payload.get("dynamic_properties"), max_properties)
    rejected_items = _validate_rejected_properties(payload.get("rejected_properties"))
    merged = merge_properties(base_properties, dynamic_items, max_dynamic_properties=max_properties)
    dynamic_names = set(merged[len(base_properties):])
    dynamic_items = [item for item in dynamic_items if item["name"] in dynamic_names]

    return {
        "dynamic_properties": dynamic_items,
        "merged_properties": merged,
        "rejected_properties": rejected_items,
        "source": "llm",
    }
