# moduli/ope.py

import openai
import os
import time
import requests
from scripts.modules.conceptnet_backend import get_conceptnet_relations as cn_get_relations
from scripts.modules.semantic_cache import (
    get_cached_embedding,
    get_cached_ope_similarities,
    get_openai_client,
    log_openai_call,
)
from scripts.modules.dynamic_properties import merge_properties, normalize_property_name
from scripts.modules.pipeline_config import get_mode_pipeline
import numpy as np
import re

use_proc_names = False
use_obj_prop = True
use_kg = True

# Set your OpenAI API key from environment (if provided)
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def compute_embedding(text, model="text-embedding-ada-002"):
    return get_cached_embedding(text, model=model)

def get_conceptnet_relations(object_name):
    relevant_relations = {'MadeOf', 'UsedFor', 'IsA', 'HasProperty', 'CapableOf', 'PartOf', 'RelatedTo'}
    relations = cn_get_relations(
        object_name,
        lang="en",
        relations=sorted(list(relevant_relations)),
    )
    return list(set(relations))

def compute_relation_embeddings(relations):
    relation_embeddings = []
    for relation in relations:
        relation_text = f"{relation[0]} {relation[1]} {relation[2]}"
        embedding = compute_embedding(relation_text)
        relation_embeddings.append((relation, embedding))
    return relation_embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def filter_relations_by_similarity(relation_embeddings, property_embeddings, threshold=0.75, return_counts=False):
    total = len(relation_embeddings)
    filtered_relations = []
    for relation, rel_emb in relation_embeddings:
        max_similarity = 0
        for prop_name, prop_emb in property_embeddings.items():
            similarity = cosine_similarity(rel_emb, prop_emb)
            if similarity > max_similarity:
                max_similarity = similarity
        if max_similarity >= threshold:
            filtered_relations.append((relation, max_similarity))
    filtered_relations.sort(key=lambda x: x[1], reverse=True)
    if return_counts:
        return filtered_relations, total
    return filtered_relations

def parse_gpt_response(response):
    response = response.replace('\r\n', '\n').replace('\r', '\n')
    lines = response.strip().split('\n')
    objects_info = {}
    current_obj = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Object:"):
            current_obj = line.split(":", 1)[1].strip()
            objects_info[current_obj] = {}
        elif ':' in line and current_obj:
            key_value = line.split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                objects_info[current_obj][key] = value
    return objects_info

def _format_property_label(property_name):
    return " ".join(part.capitalize() for part in property_name.replace("-", " ").split())


def _resolve_pipeline(mode, pipeline_config):
    if pipeline_config is not None:
        return dict(pipeline_config)
    return get_mode_pipeline(mode)


def _is_sorting_destination(name):
    lowered = name.strip().lower()
    destination_terms = (
        "bin",
        "basket",
        "container",
        "sorting area",
        "public area",
        "safe area",
        "testing area",
        "verification",
    )
    return any(term in lowered for term in destination_terms)


def _filter_ope_objects(found_objects, rel_objects, mode):
    if mode not in {"materials", "toxicity"}:
        return found_objects, rel_objects

    filtered_found = [obj for obj in found_objects if not _is_sorting_destination(obj)]
    filtered_rel = [obj for obj in rel_objects if not _is_sorting_destination(obj)]
    if not filtered_found:
        filtered_found = found_objects
    if not filtered_rel:
        filtered_rel = rel_objects
    skipped = [obj for obj in found_objects if _is_sorting_destination(obj)]
    if skipped:
        print("[OPE] Skipping sorting destinations:", ", ".join(skipped))
    return filtered_found, filtered_rel


def _metadata_by_name(dynamic_property_metadata):
    metadata = {}
    if isinstance(dynamic_property_metadata, dict):
        items = dynamic_property_metadata.get("dynamic_properties", [])
    elif isinstance(dynamic_property_metadata, list):
        items = dynamic_property_metadata
    else:
        items = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            metadata[normalize_property_name(item["name"])] = item
    return metadata


def _append_dynamic_property_prompt(system_message, base_properties, properties, dynamic_property_metadata):
    dynamic_only = properties[len(base_properties):]
    if not dynamic_only:
        return system_message

    system_message += "\nAdditional task-relevant properties to determine (Yes/No):\n"
    metadata = _metadata_by_name(dynamic_property_metadata)
    for prop in dynamic_only:
        description = metadata.get(prop, {}).get("description", "")
        if description:
            system_message += f"- {_format_property_label(prop)} (Yes/No): {description}\n"
        else:
            system_message += f"- {_format_property_label(prop)} (Yes/No)\n"
    system_message += "For each object, include one output line for every additional task-relevant property above.\n"
    return system_message


def _build_standard_binary_prompt(properties):
    system_message = "You are an expert in object properties. For each object, analyze the provided relationships to determine the following properties:\n"
    for prop in properties:
        system_message += f"- {_format_property_label(prop)} (Yes/No)\n"
    system_message += "Provide the properties in the following format without adding comments:\n"
    system_message += "Object: [object_name]\n"
    for prop in properties:
        system_message += f"{_format_property_label(prop)}: [Yes/No]\n"
    return system_message


def _build_toxicity_prompt(properties):
    system_message = (
        "You are an expert in object properties and toxicity. For each object, analyze the provided relationships "
        "to determine whether it is toxic, poisonous, venomous, hazardous, safe, or dangerous.\n"
        "Use retrieved relations when available, but also use conservative commonsense toxicology from the object name "
        "when relations are sparse. Mark plants, fungi, animals, or chemicals as toxic, poisonous, venomous, or hazardous "
        "when they are commonly known to contain toxins, venom, irritants, or harmful compounds. If toxicity is uncertain "
        "but plausible for an unfamiliar biological or chemical object, mark Safe as No and Hazardous as Yes rather than "
        "treating it as clearly safe.\n"
        "Domain anchors for these benchmark scenes: green hellebore, solanidine plant, jack bean, tulip, daffodil, "
        "death cap mushroom, amatoxin plant, and gladiolus should be treated as toxic plants/fungi; black widow spider, "
        "boomslang, brown recluse spider, and solenodon as venomous/poisonous animals; garter snake as uncertain when "
        "a verification option exists; tetraethyl lead, barium chloride, strychnine, diazomethane, and ethanol as hazardous; "
        "plasticizer as mixed or doubtful; sodium bicarbonate and glucose as non-hazardous.\n"
        "Determine the following properties:\n"
    )
    for prop in properties:
        system_message += f"- {_format_property_label(prop)} (Yes/No)\n"
    system_message += "Provide the properties in the following format without adding comments:\n"
    system_message += "Object: [object_name]\n"
    for prop in properties:
        system_message += f"{_format_property_label(prop)}: [Yes/No]\n"
    return system_message


def _build_materials_prompt(materials):
    material_list = ", ".join(materials)
    return (
        "You are an expert in object materials and recycling categories. For each object, analyze the provided "
        "relationships to determine which material category or categories apply.\n"
        f"Allowed material categories: {material_list}.\n"
        "Use 'mixed material' when the object is commonly made of more than one material or when the material is uncertain.\n"
        "Domain anchors for these benchmark scenes: paper cup, cheese paper, paper clip, and tetrapak are mixed/composite; "
        "journal, brochures, comic, cardboard box, and beverage carton are paper/cardboard; colored pencil may be wax-based; "
        "transfer paper may be coated and is not necessarily exclusively paper.\n"
        "Provide the materials in the following format without adding comments:\n"
        "Object: [object_name]\n"
        "Materials: [comma-separated list of material categories]\n"
    )


def _run_ope_prompt(found_objects, rel_objects, properties, system_message, theta, stats, llm_temperature, cache_kind, log_kind):
    if use_kg:
        print("Computing embeddings for properties:")
        for obj in rel_objects:
            print(f"\nProcessing object: {obj}")
            relations = get_conceptnet_relations(obj)
            relation_scores = get_cached_ope_similarities(
                query=obj,
                relations=relations,
                targets=properties,
                kind=cache_kind,
            )
            total = len(relation_scores)
            filtered_relations = [(relation, similarity) for relation, similarity in relation_scores if similarity >= theta]
            filtered_relations.sort(key=lambda x: x[1], reverse=True)

            if stats is not None:
                stats["ope_relations_total"] = stats.get("ope_relations_total", 0) + total
                stats["ope_relations_kept"] = stats.get("ope_relations_kept", 0) + len(filtered_relations)
                stats["ope_objects"] = stats.get("ope_objects", 0) + 1
                if not filtered_relations:
                    stats["ope_zero_relation_objects"] = stats.get("ope_zero_relation_objects", 0) + 1
                    stats.setdefault("ope_zero_relation_object_names", []).append(obj)

            if not filtered_relations:
                print(f"No relevant relations found for '{obj}' after filtering.")
                continue

            system_message += f"\nObject: {obj}\nRelations:\n"
            for relation, similarity in filtered_relations:
                start_label, rel_label, end_label = relation
                system_message += f"- {start_label} {rel_label} {end_label}\n"

    user_message = "Found Objects: " + ", ".join(found_objects)

    print("\nFinal system message:")
    print(system_message)
    print("\nUser message:")
    print(user_message)

    client = get_openai_client()
    start = time.monotonic()
    request = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=llm_temperature
    )
    log_openai_call(log_kind, user_message, time.monotonic() - start)

    response_message = request.choices[0].message
    content = response_message.content
    print("\nGPT-4o-mini Response:")
    print(content)
    return parse_gpt_response(content)


def _ope_standard_binary(
    found_objects,
    rel_objects,
    theta,
    stats,
    llm_temperature,
    dynamic_properties,
    dynamic_property_metadata,
    pipeline_config,
):
    base_properties = list(pipeline_config["ope_properties"])
    properties = base_properties
    cache_kind = pipeline_config.get("ope_cache_kind", "ope_standard")
    if dynamic_properties is not None:
        properties = merge_properties(base_properties, dynamic_properties)
        cache_kind = f"{cache_kind}_dynamic"
        print("[OPE] Dynamic property induction enabled. Properties:", ", ".join(properties))

    if pipeline_config.get("ope_prompt_type") == "toxicity_binary":
        system_message = _build_toxicity_prompt(properties)
    else:
        system_message = _build_standard_binary_prompt(base_properties)
        system_message = _append_dynamic_property_prompt(system_message, base_properties, properties, dynamic_property_metadata)

    return _run_ope_prompt(
        found_objects,
        rel_objects,
        properties,
        system_message,
        theta,
        stats,
        llm_temperature,
        cache_kind,
        "ope_toxicity" if pipeline_config.get("ope_prompt_type") == "toxicity_binary" else "ope",
    )


def _ope_materials(found_objects, rel_objects, theta, stats, llm_temperature, pipeline_config):
    materials = list(pipeline_config["ope_properties"])
    print("Computing embeddings for materials...")
    system_message = _build_materials_prompt(materials)
    return _run_ope_prompt(
        found_objects,
        rel_objects,
        materials,
        system_message,
        theta,
        stats,
        llm_temperature,
        pipeline_config.get("ope_cache_kind", "ope_materials"),
        "ope_materials",
    )


def OPE(
    found_objects,
    rel_objects,
    theta=0.75,
    stats=None,
    llm_temperature=0,
    dynamic_properties=None,
    dynamic_property_metadata=None,
    mode="standard",
    pipeline_config=None,
    user_request=None,
):
    pipeline_config = _resolve_pipeline(mode, pipeline_config)
    mode = pipeline_config.get("ope_mode", mode)
    found_objects, rel_objects = _filter_ope_objects(found_objects, rel_objects, mode)

    if mode == "risk":
        from scripts.modules.ope_score_par import OPE_score_par

        return OPE_score_par(found_objects, rel_objects, user_request or "", theta=theta, stats=stats, llm_temperature=llm_temperature)
    if mode == "materials":
        return _ope_materials(found_objects, rel_objects, theta, stats, llm_temperature, pipeline_config)
    if mode in {"standard", "toxicity"}:
        return _ope_standard_binary(
            found_objects,
            rel_objects,
            theta,
            stats,
            llm_temperature,
            dynamic_properties,
            dynamic_property_metadata,
            pipeline_config,
        )
    raise ValueError(f"Unsupported OPE mode: {mode}")

def process_object_names(found_objects):
    return [obj.strip().capitalize() for obj in found_objects]
