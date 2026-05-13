STANDARD_OPE_PROPERTIES = [
    "dangerous",
    "fragile",
    "deformable",
    "hold liquid",
    "safe",
    "stable",
    "poisonous",
]

TOXICITY_OPE_PROPERTIES = [
    "dangerous",
    "safe",
    "poisonous",
    "toxic",
    "venomous",
    "hazardous",
]

MATERIAL_OPE_PROPERTIES = [
    "metal",
    "aluminum",
    "plastic",
    "glass",
    "wood",
    "ceramic",
    "fabric",
    "wax",
    "paper",
    "cardboard",
    "mixed material",
]

RISK_OPE_PROPERTIES = ["dangerous"]


CATEGORY_PIPELINE = {
    "explicit_unambiguous": {
        "ope_mode": "standard",
        "urp_mode": "standard",
        "ope_properties": STANDARD_OPE_PROPERTIES,
        "ope_prompt_type": "standard_binary",
        "urp_prompt_type": "standard_properties",
        "ope_cache_kind": "ope_standard",
        "urp_cache_prefix": "urp",
        "dynamic_property_induction": False,
    },
    "explicit_ambiguous": {
        "ope_mode": "standard",
        "urp_mode": "standard",
        "ope_properties": STANDARD_OPE_PROPERTIES,
        "ope_prompt_type": "standard_binary",
        "urp_prompt_type": "standard_properties",
        "ope_cache_kind": "ope_standard",
        "urp_cache_prefix": "urp",
        "dynamic_property_induction": False,
    },
    "implicit": {
        "ope_mode": "standard",
        "urp_mode": "implicit",
        "ope_properties": STANDARD_OPE_PROPERTIES,
        "ope_prompt_type": "standard_binary",
        "urp_prompt_type": "implicit_needs",
        "ope_cache_kind": "ope_standard",
        "urp_cache_prefix": "urp_implicit",
        "dynamic_property_induction": False,
    },
    "toxicity": {
        "ope_mode": "toxicity",
        "urp_mode": "toxicity",
        "ope_properties": TOXICITY_OPE_PROPERTIES,
        "ope_prompt_type": "toxicity_binary",
        "urp_prompt_type": "toxicity_sorting",
        "ope_cache_kind": "ope_toxicity",
        "urp_cache_prefix": "urp_toxicity",
        "dynamic_property_induction": False,
        "property_hints": ["toxic", "non-toxic", "poisonous", "venomous", "hazardous"],
    },
    "materials": {
        "ope_mode": "materials",
        "urp_mode": "materials",
        "ope_properties": MATERIAL_OPE_PROPERTIES,
        "ope_prompt_type": "materials_list",
        "urp_prompt_type": "material_sorting",
        "ope_cache_kind": "ope_materials",
        "urp_cache_prefix": "urp_materials",
        "dynamic_property_induction": False,
        "property_hints": ["paper", "cardboard", "wax", "plastic", "glass", "metal", "aluminum"],
    },
    "risk_aware": {
        "ope_mode": "risk",
        "urp_mode": "risk",
        "ope_properties": RISK_OPE_PROPERTIES,
        "ope_prompt_type": "risk_score_and_interactions",
        "urp_prompt_type": "risk_aware",
        "ope_cache_kind": "ope_risk",
        "urp_cache_prefix": "urp_risk",
        "dynamic_property_induction": False,
    },
}


MODE_DEFAULTS = {
    "standard": CATEGORY_PIPELINE["explicit_unambiguous"],
    "toxicity": CATEGORY_PIPELINE["toxicity"],
    "materials": CATEGORY_PIPELINE["materials"],
    "risk": CATEGORY_PIPELINE["risk_aware"],
}


def get_category_pipeline(category: str) -> dict:
    if category not in CATEGORY_PIPELINE:
        raise KeyError(f"Unknown instruction category: {category}")
    return dict(CATEGORY_PIPELINE[category])


def get_mode_pipeline(mode: str) -> dict:
    if mode not in MODE_DEFAULTS:
        raise KeyError(f"Unknown pipeline mode: {mode}")
    return dict(MODE_DEFAULTS[mode])
