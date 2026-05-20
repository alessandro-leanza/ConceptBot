import json
import time
from typing import Any, Dict, List, Optional

from scripts.modules.pipeline_config import STANDARD_OPE_PROPERTIES
from scripts.modules.semantic_cache import get_openai_client, log_openai_call


CONFIDENCE_LEVELS = {"low", "medium", "high"}
EXCLUDED_AUDIT_CATEGORIES = {"materials", "toxicity"}
RISK_CORE_TERMS = {
    "appliance",
    "break",
    "breakage",
    "breaking",
    "burn",
    "chemical",
    "child",
    "damage",
    "dishwasher",
    "electric",
    "electrical",
    "fire",
    "fragile",
    "handwashing",
    "hazardous",
    "heat",
    "hot",
    "knife",
    "material compatibility",
    "microwave",
    "poison",
    "poisonous",
    "scissors",
    "sharp",
    "stack",
    "stacking",
    "toxic",
    "unstable",
}


def _as_clean_string_list(value: Any, limit: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def _fallback_result(reason: str, source: str = "fallback") -> Dict[str, Any]:
    return {
        "risk_index_required": False,
        "confidence": "low",
        "risk_reasons": [reason] if reason else [],
        "risk_cues": [],
        "recommended_mode": "standard",
        "source": source,
    }


def _category_override(task_category: Optional[str]) -> Optional[Dict[str, Any]]:
    if task_category == "risk_aware":
        return {
            "risk_index_required": True,
            "confidence": "high",
            "risk_reasons": [
                "The instruction belongs to the risk_aware category, so Risk Index evaluation is required by policy."
            ],
            "risk_cues": ["category:risk_aware"],
            "recommended_mode": "risk",
            "source": "category_override",
        }
    if task_category in EXCLUDED_AUDIT_CATEGORIES:
        return {
            "risk_index_required": False,
            "confidence": "high",
            "risk_reasons": [
                f"The {task_category} category is excluded from the first implicit-risk audit because it already uses task-specific OPE/URP prompts."
            ],
            "risk_cues": [f"category:{task_category}"],
            "recommended_mode": "standard",
            "source": "category_override",
        }
    return None


def _build_messages(
    user_instruction: str,
    detected_objects: List[str],
    task_category: Optional[str],
    dynamic_properties: Optional[List[str]],
) -> List[Dict[str, str]]:
    system_prompt = """
You are auditing a robot pick-and-place planning pipeline.

Decide whether this task should activate a numerical, interaction-aware Risk Index module.
The Risk Index is more expensive than standard object-property extraction and should be used only when the task requires explicit hazard reasoning, object interaction risk, or safety trade-offs that would change the robot's pick-and-place decision.

The key question is not "could any hazard exist?" but "does the robot need enhanced risk reasoning to choose the correct object, destination, or object-object interaction?"

Activate the Risk Index when the instruction or scene implies risk-sensitive selection among alternatives, such as heat handling, microwave/material compatibility, sharp tools near vulnerable recipients, fragile object placement, breakage-sensitive appliance use, unstable stacking, chemicals, toxicity, fire, electrical risk, or unsafe object interactions.
Treat explicit language about avoiding damage, avoiding breakage, safety, unsafe placement, or risk-sensitive assignment to destinations as strong evidence for activation.
Prioritize risks that are intrinsic to the manipulated objects, their destinations, or their interactions. A hazard that is only part of the background situation should not activate the Risk Index if the robot can solve the request by selecting an ordinary non-hazardous object and no risky alternative must be compared.
The Risk Index is not a general hazard detector. Activate it only if assigning numerical danger scores to candidate objects or object-object pairs would plausibly change the final action plan.
Use two conditions for activation: (1) the final plan depends on a risk-sensitive choice among candidate objects, destinations, or interactions; and (2) choosing incorrectly could create or worsen physical harm, object damage, breakage, or unsafe interaction. If the robot is only selecting a benign support object and the risky condition is not created or worsened by the pick-and-place action, return risk_index_required=false.

Do NOT activate the Risk Index merely because a generic property such as "dangerous" could be checked by standard OPE.
Do NOT activate it for ordinary explicit pick-and-place, ordinary ambiguity resolution, or harmless object selection.
Do NOT activate it when the task can be resolved by ordinary semantic selection and no risk-sensitive alternative must be compared.
Do NOT infer hidden preparation steps, contamination, perishability, environmental hazards, or health risk unless they are directly implied and affect the pick-and-place choice.
If standard OPE properties such as fragile, hold liquid, safe, stable, or poisonous are sufficient to resolve the task, return risk_index_required=false.

Return JSON only with this schema:
{
  "risk_index_required": true or false,
  "confidence": "low" | "medium" | "high",
  "risk_reasons": ["short reason"],
  "risk_cues": ["short cue"],
  "recommended_mode": "risk" | "standard"
}
""".strip()

    payload = {
        "user_instruction": user_instruction,
        "detected_objects": detected_objects,
        "task_category": task_category,
        "base_ope_properties": STANDARD_OPE_PROPERTIES,
        "dynamic_properties": dynamic_properties or [],
        "positive_examples": [
            {
                "instruction": "Heat food using an appliance.",
                "objects": ["metal container", "heat-safe container", "food bowl", "appliance"],
                "risk_index_required": True,
                "reason": "Heating can require material compatibility and unsafe interaction reasoning.",
            },
            {
                "instruction": "Give a sharp tool to a vulnerable recipient.",
                "objects": ["sharp tool", "safe tool", "recipient"],
                "risk_index_required": True,
                "reason": "Sharp tools and vulnerable recipients imply an explicit safety trade-off.",
            },
            {
                "instruction": "Build the safest stack.",
                "objects": ["fragile item", "heavy item", "stable container"],
                "risk_index_required": True,
                "reason": "Stacking fragile or unstable objects requires interaction-aware risk reasoning.",
            },
        ],
        "negative_examples": [
            {
                "instruction": "Bring me a named food item.",
                "objects": ["food item", "book", "user"],
                "risk_index_required": False,
                "reason": "Ordinary pick-and-place can be handled by standard OPE.",
            },
            {
                "instruction": "Move a colored block to a surface.",
                "objects": ["colored block", "other block", "surface"],
                "risk_index_required": False,
                "reason": "No implicit hazard or interaction-specific risk is present.",
            },
            {
                "instruction": "Sort objects by material.",
                "objects": ["glass object", "plastic object", "paper object"],
                "risk_index_required": False,
                "reason": "Material sorting is better handled by the material-specific OPE prompt.",
            },
        ],
    }
    user_prompt = json.dumps(payload, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _contains_core_risk_signal(*values: Any) -> bool:
    text_parts = []
    for value in values:
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
        elif value is not None:
            text_parts.append(str(value))
    text = " ".join(text_parts).lower()
    return any(term in text for term in RISK_CORE_TERMS)


def _validate_llm_result(data: Dict[str, Any], user_instruction: str, detected_objects: List[str]) -> Dict[str, Any]:
    required = bool(data.get("risk_index_required", False))
    confidence = str(data.get("confidence", "low")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "low"
    risk_reasons = _as_clean_string_list(data.get("risk_reasons"))
    risk_cues = _as_clean_string_list(data.get("risk_cues"))

    if required and confidence != "high":
        has_core_signal = _contains_core_risk_signal(user_instruction, detected_objects, risk_reasons, risk_cues)
        if not has_core_signal:
            required = False
            confidence = "medium"
            risk_reasons = [
                "The LLM suggested risk activation, but no core risk signal was present for interaction-aware Risk Index routing."
            ]
            risk_cues = []

    recommended_mode = str(data.get("recommended_mode", "")).strip().lower()
    if recommended_mode not in {"risk", "standard"}:
        recommended_mode = "risk" if required else "standard"
    if required:
        recommended_mode = "risk"
    else:
        recommended_mode = "standard"

    return {
        "risk_index_required": required,
        "confidence": confidence,
        "risk_reasons": risk_reasons,
        "risk_cues": risk_cues,
        "recommended_mode": recommended_mode,
        "source": "llm",
    }


def assess_risk_trigger(
    user_instruction: str,
    detected_objects: List[str],
    task_category: Optional[str] = None,
    dynamic_properties: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    use_category_overrides: bool = True,
) -> Dict[str, Any]:
    """
    Decide whether the interaction-aware Risk Index should be activated.

    This helper is intentionally conservative and disabled from the main
    ConceptBot pipeline by default. It is used by the standalone audit script
    to study implicit risk activation for the revision response.
    """
    if use_category_overrides:
        override = _category_override(task_category)
        if override is not None:
            return override

    try:
        client = get_openai_client()
        messages = _build_messages(user_instruction, detected_objects, task_category, dynamic_properties)
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        log_openai_call("risk_trigger", user_instruction, time.monotonic() - start)
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        if not isinstance(data, dict):
            return _fallback_result("Risk trigger returned non-object JSON.")
        return _validate_llm_result(data, user_instruction, detected_objects)
    except Exception as exc:
        return _fallback_result(f"Risk trigger failed safely: {exc}")
