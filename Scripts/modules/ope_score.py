# moduli/ope.py

import openai
from openai import OpenAI
import requests
import numpy as np
import re

use_proc_names = False
use_obj_prop = True
use_kg = False

# Set your OpenAI API key
openai.api_key = ''

def compute_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding
    return np.array(embedding)

def get_conceptnet_relations(object_name):
    object_name = object_name.lower().replace(' ', '_')
    url = f'http://api.conceptnet.io/c/en/{object_name}?limit=100'
    response = requests.get(url).json()

    if 'error' in response:
        print(f"Error: {response['error']['details']}")
        return []

    relations = []
    relevant_relations = {'MadeOf', 'UsedFor', 'IsA', 'HasProperty', 'CapableOf', 'PartOf', 'RelatedTo'}

    for edge in response.get('edges', []):
        rel_label = edge['rel']['label']
        if rel_label in relevant_relations:
            start = edge['start']['label']
            end = edge['end']['label']
            relations.append((start, rel_label, end))

    # Rimuove duplicati
    relations = list(set(relations))
    return relations

def compute_relation_embeddings(relations):
    relation_embeddings = []
    for relation in relations:
        relation_text = f"{relation[0]} {relation[1]} {relation[2]}"
        embedding = compute_embedding(relation_text)
        relation_embeddings.append((relation, embedding))
    return relation_embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def filter_relations_by_similarity(relation_embeddings, property_embeddings, threshold=0.75):
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
                try:
                    value = int(value.strip())
                    if 1 <= value <= 3:
                        objects_info[current_obj][key] = value
                    else:
                        print(f"Invalid score '{value}' for property '{key}' in object '{current_obj}'.")
                except ValueError:
                    print(f"Invalid format for property '{key}' in object '{current_obj}'.")

    return objects_info

def OPE_score(found_objects, rel_objects):
    properties = ["fragile", "toxic", "dangerous", "stable", 'deformable']
    property_embeddings = {}
    system_message = (
        "You are an expert in object properties. For each object, analyze the provided relationships to determine the following properties:\n"
        "- Fragile (Score 1-3)\n"
        "- Stable (Score 1-3)\n"
        "- Deformable (Score 1-3)\n"
        "- Toxic (Score 1-3)\n"
        "- Dangerous (Score 1-3)\n\n"
        "Provide the properties in the following format without adding comments:\n"
        "Object: [object_name]\n"
        "Fragile: [Score 1-3]\n"
        "Stable: [Score 1-3]\n"
        "Deformable: [Score 1-3]\n"
        "Toxic: [Score 1-3]\n"
        "Dangerous: [Score 1-3]\n"
    )

    if use_kg:
        print("Computing embeddings for properties:")
        for prop in properties:
            embedding = compute_embedding(prop)
            property_embeddings[prop] = embedding
       
        for obj in rel_objects:
            print(f"\nProcessing object: {obj}")
            obj_embedding = compute_embedding(obj)

            relations = get_conceptnet_relations(obj)
            
            relation_embeddings = compute_relation_embeddings(relations)
            
            filtered_relations = filter_relations_by_similarity(relation_embeddings, property_embeddings)
            
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

    client = OpenAI()
    request = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    response_message = request.choices[0].message
    content = response_message.content
    print("\nGPT-4o-mini Response:")
    print(content)
    objects_info = parse_gpt_response(content)
    return objects_info

def process_object_names(found_objects):
    return [obj.strip().capitalize() for obj in found_objects]
