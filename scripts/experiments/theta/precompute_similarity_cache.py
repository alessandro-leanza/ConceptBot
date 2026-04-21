import argparse

from instructions.load_instructions import load_category, resolve_items
from scripts.modules.conceptnet_backend import get_conceptnet_relations
from scripts.modules.semantic_cache import (
    flush_all_caches,
    get_cached_keywords,
    get_cached_ope_similarities,
    get_cached_urp_object_keyword_similarities,
    get_cached_urp_request_similarities,
    precompute_all_similarity_caches,
)


DEFAULT_CATEGORIES = [
    "explicit_unambiguous",
    "explicit_ambiguous",
    "implicit",
    "risk_aware",
    "materials",
    "toxicity",
]

OPE_STANDARD_TARGETS = ["dangerous", "fragile", "deformable", "hold liquid", "safe", "stable", "poisonous"]
OPE_MATERIAL_TARGETS = ["metal", "plastic", "glass", "wood", "ceramic", "fabric", "wax"]
OPE_RISK_TARGETS = ["dangerous"]
OPE_RELATIONS = sorted(["MadeOf", "UsedFor", "IsA", "HasProperty", "CapableOf", "PartOf", "RelatedTo"])
URP_RELATIONS = sorted(["IsA", "UsedFor", "HasProperty", "CapableOf", "MannerOf"])


def precompute_item(category: str, item: dict) -> None:
    objects = item["objects"]
    instruction = item["instruction"]
    keywords = get_cached_keywords(instruction, model="gpt-4o-mini", llm_temperature=0)

    for obj in objects:
        obj_relations = get_conceptnet_relations(obj, relations=OPE_RELATIONS)
        if category == "materials":
            get_cached_ope_similarities(obj, obj_relations, OPE_MATERIAL_TARGETS, kind="ope_materials")
        elif category == "risk_aware":
            get_cached_ope_similarities(obj, obj_relations, OPE_RISK_TARGETS, kind="ope_risk")
        else:
            get_cached_ope_similarities(obj, obj_relations, OPE_STANDARD_TARGETS, kind="ope_standard")
            if category == "toxicity":
                get_cached_ope_similarities(obj, obj_relations, OPE_STANDARD_TARGETS, kind="ope_standard")
        urp_obj_relations = get_conceptnet_relations(obj, relations=URP_RELATIONS)
        get_cached_urp_object_keyword_similarities(
            instruction=instruction,
            query=obj,
            relations=urp_obj_relations,
            keywords=keywords,
            kind="urp_object_to_keywords" if category != "risk_aware" else "urp_risk_object_to_keywords",
        )

    for keyword in keywords:
        keyword_relations = get_conceptnet_relations(keyword, relations=URP_RELATIONS)
        get_cached_urp_request_similarities(
            instruction=instruction,
            query=keyword,
            relations=keyword_relations,
            kind="urp_keyword_to_request" if category != "risk_aware" else "urp_risk_keyword_to_request",
        )
        get_cached_urp_request_similarities(
            instruction=instruction,
            query=keyword,
            relations=keyword_relations,
            kind="urp_keyword_relations_to_request" if category != "risk_aware" else "urp_risk_keyword_relations_to_request",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = parser.parse_args()

    total_items = 0
    for category in args.categories:
        cat = load_category(category)
        items = resolve_items(cat)
        for item in items:
            print(f"[precompute] category={category} id={item['id']}")
            precompute_item(category, item)
            total_items += 1

    paths = precompute_all_similarity_caches()
    flush_all_caches()
    print("Precompute completed")
    print(f"Items processed: {total_items}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
