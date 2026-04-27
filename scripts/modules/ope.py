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


def OPE(
    found_objects,
    rel_objects,
    theta=0.75,
    stats=None,
    llm_temperature=0,
    dynamic_properties=None,
    dynamic_property_metadata=None,
):
    base_properties = ['dangerous', 'fragile', 'deformable', "hold liquid", 'safe', 'stable', 'poisonous']
    properties = base_properties
    if dynamic_properties is not None:
        properties = merge_properties(base_properties, dynamic_properties)
        print("[OPE] Dynamic property induction enabled. Properties:", ", ".join(properties))
    property_embeddings = {}

    system_message = (
        "You are an expert in object properties. For each object, analyze the provided relationships to determine the following properties:\n"
        "- Dangerous (Yes/No)\n"
        "- Fragile (Yes/No)\n"
        "- Deformable (Yes/No)\n"
        "- Hold Liquid (Yes/No)\n"
        "- Safe (Yes/No)\n"
        "- Stable (Yes/No)\n"
        "- Poisonous (Yes/No)\n"
        "Provide the properties in the following format without adding comments:\n"
        "Object: [object_name]\n"
        "Dangerous: [Yes/No]\n"
        "Fragile: [Yes/No]\n"
        "Hold Liquid: [Yes/No]\n"
        "Deformable: [Yes/No]\n"
        "Safe: [Yes/No]\n"
        "Stable: [Yes/No]\n"
        "Poisonous: [Yes/No]\n"
    )

    dynamic_only = properties[len(base_properties):]
    if dynamic_only:
        system_message += "\nAdditional task-relevant properties to determine (Yes/No):\n"
        metadata_by_name = {}
        if isinstance(dynamic_property_metadata, dict):
            for item in dynamic_property_metadata.get("dynamic_properties", []):
                if isinstance(item, dict) and item.get("name"):
                    metadata_by_name[normalize_property_name(item["name"])] = item
        elif isinstance(dynamic_property_metadata, list):
            for item in dynamic_property_metadata:
                if isinstance(item, dict) and item.get("name"):
                    metadata_by_name[normalize_property_name(item["name"])] = item

        for prop in dynamic_only:
            description = metadata_by_name.get(prop, {}).get("description", "")
            if description:
                system_message += f"- {_format_property_label(prop)} (Yes/No): {description}\n"
            else:
                system_message += f"- {_format_property_label(prop)} (Yes/No)\n"
        system_message += "For each object, include one output line for every additional task-relevant property above.\n"

    if use_kg:
        print("Computing embeddings for properties:")
        for obj in rel_objects:
            print(f"\nProcessing object: {obj}")
            relations = get_conceptnet_relations(obj)
            relation_scores = get_cached_ope_similarities(
                query=obj,
                relations=relations,
                targets=properties,
                kind="ope_standard_dynamic" if dynamic_properties is not None else "ope_standard",
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
    log_openai_call("ope", user_message, time.monotonic() - start)

    response_message = request.choices[0].message
    content = response_message.content
    print("\nGPT-4o-mini Response:")
    print(content)
    objects_info = parse_gpt_response(content)
    return objects_info

def process_object_names(found_objects):
    return [obj.strip().capitalize() for obj in found_objects]
