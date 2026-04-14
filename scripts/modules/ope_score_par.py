# moduli/ope.py

import openai
import os
from openai import OpenAI
import requests
from scripts.modules.conceptnet_backend import get_conceptnet_relations as cn_get_relations
import numpy as np
import re
import wikipediaapi
import wikipedia
# import spacy
from typing import List

from openie import StanfordOpenIE
# Carica modello NLP
# nlp = spacy.load("en_core_web_sm")

from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

use_proc_names = False
use_obj_prop = True
use_kg = False
use_wiki = False

# Set your OpenAI API key from environment (if provided)
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def compute_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding
    return np.array(embedding)

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

def compute_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def filter_relations_by_similarity(relation_embeddings, property_embeddings, threshold=0.75, return_counts=False):
    total = len(relation_embeddings)
    filtered_relations = []
    for relation, rel_emb in relation_embeddings:
        max_similarity = 0
        for prop_name, prop_emb in property_embeddings.items():
            similarity = compute_cosine_similarity(rel_emb, prop_emb)
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
                try:
                    value = int(value.strip())
                    if 1 <= value <= 3: 
                        objects_info[current_obj][key] = value
                    else:
                        print(f"Invalid score '{value}' for property '{key}' in object '{current_obj}'.")
                except ValueError:
                    print(f"Invalid format for property '{key}' in object '{current_obj}'.")

    return objects_info

def extract_openie_triples(text: str):
    """
    Extracts triples (subject, relation, object) from a text using Stanford OpenIE.

    Args:
        text (str): Input text to be parsed.

    Returns:
        list: A list of triples (subject, relation, object).
    """
    triples = []
    with StanfordOpenIE() as client:
        extracted = client.annotate(text)
        triples = [(triple['subject'], triple['relation'], triple['object']) for triple in extracted]
    return triples

def extract_wikipedia_triples(text: str, property_embeddings: dict) -> List[tuple]:
    """
    It extracts triples from Wikipedia text and filters them using similarity cosines.
    """
    with StanfordOpenIE() as client:
        triples = client.annotate(text)
    
    print(f"Extracted {len(triples)} triples from text.")

    # Converte le triple in formato (soggetto, relazione, oggetto)
    formatted_triples = [(triple['subject'], triple['relation'], triple['object']) for triple in triples]

    # Filtra le triple basandosi sulla cosine similarity
    filtered_triples = filter_triples_by_similarity(formatted_triples, property_embeddings)
    
    return filtered_triples


def filter_triples_by_similarity(relation_triples, property_embeddings, deduplicated_embeddings=None, 
                                 threshold=0.75, use_embedding_filter=False, use_keyword_filter=True, keywords=None,
                                 return_counts=False):
    """
    Filters triples based on cosine similarity or a set of keywords.
    Args:
        relation_triples: List of triples (subject, relation, object).
        property_embeddings: Dictionary with embeddings of properties.
        deduplicated_embeddings: List of embeddings corresponding to deduplicated triples (optional).
        threshold: Threshold of similarity cosines for filtering.
        use_embedding_filter: If True, uses cosine similarity based filtering.
        use_keyword_filter: If True, uses keyword-based filtering.
        keywords: List of keywords for word-based filtering.
    Returns:
        filtered_triples: List of filtered triples.

    """
    filtered_triples = []
    total = len(relation_triples)
    keywords = keywords or [
        "danger", "dangers", "dangerous", "hazard", "hazards", "hazardous", 
        "risk", "risks", "risky", "injury", "injuries", "harm", "harms", 
        "harming", "harmful", "unsafe", "unsafety", "toxic", "toxicity", 
        "poisonous", "poison", "poisons", "flammable", "flammability", 
        "explosive", "explosives", "explode", "explosion", 
        "radioactive", "radioactivity", "corrosive", "corrosion", 
        "fatal", "fatality", "fatalities", "lethal", "lethality", 
        "unstable", "instability", "volatile", "volatility", 
        "contaminated", "contamination", "deadly",
        "break", "breaking", "broken", "shatter", "shattered", "fracture", "fractured", 
        "crack", "cracked", "snap", "snapped", "damage", "damaging", "damaged", 
        "destroy", "destroyed", "destruction", "weaken", "weakened", 
        "melt", "melting", "melted", "warp", "warped", "deform", "deformed", 
        "bend", "bent", "collapse", "collapsed", "tear", "torn", 
        "split", "splitting", "rupture", "ruptured", 
        "deteriorate", "deteriorated", "deterioration",
        
        "safe", "safety", "secure", "security", "stability", "stable", 
        "protected", "protection", "insulated", "non-toxic", 
        "heat-resistant", "shockproof", "childproof", "durable", "reliable", 
        "reinforced", "shatterproof",
        
        "appropriate", "suitable", "intended", "designed", "recommended", 
        "approved", "validated", "certified",
        
        "fragile", "fragility", "delicate", "brittle", 
        "unstable", "unstability", "wobbly", "weak", "vulnerable",
        
        "resistant", "resistance", "heatproof", "waterproof", 
        "fireproof", "scratch-resistant", "impact-resistant", 
       
        "compatible", "incompatible", "adaptable", "customizable", "modular"
    ]


    # First filtering based on keywords
    if use_keyword_filter:
        for relation in relation_triples:
            if len(relation) == 3:  
                triple_text = f"{relation[0]} {relation[1]} {relation[2]}"
                if any(keyword in triple_text.lower() for keyword in keywords):
                    filtered_triples.append((relation, 1.0))

    # Second filtering based on embeddings
    if use_embedding_filter:
        dangerous_embedding = property_embeddings.get("dangerous")
        if dangerous_embedding is None:
            raise ValueError("Dangerous embedding not found in property embeddings.")

        if deduplicated_embeddings is None:
            raise ValueError("Embeddings for deduplicated triples must be provided for embedding filtering.")

        refined_triples = []
        for i, (relation, _) in enumerate(filtered_triples):
            triple_text = f"{relation[0]} {relation[1]} {relation[2]}"
            triple_embedding = deduplicated_embeddings[i]

            similarity = compute_cosine_similarity(triple_embedding, dangerous_embedding)

            print(f"[Cosine Similarity] Relation: '{triple_text}' - Similarity: {similarity:.2f}")

            if similarity >= threshold:
                refined_triples.append((relation, similarity))

        print(f"[Embedding Filter] Found {len(refined_triples)} relations after embedding filtering.")

        refined_triples.sort(key=lambda x: x[1], reverse=True)
        if return_counts:
            return refined_triples, total
        return refined_triples

    if return_counts:
        return filtered_triples, total
    return filtered_triples


def filter_triples(triples: List[str], keywords: List[str]) -> List[str]:
    """
    Filters triples based on keywords.
    """
    filtered = []
    for triple in triples:
        if any(keyword in triple.lower() for keyword in keywords):
            filtered.append(triple)
    return filtered


def fetch_wikipedia_content(label):
    """
    Retrieves the content of the Wikipedia page given a label.
    In case of ambiguity, it allows the user to manually choose the correct page.
    """
    wikipedia.set_lang("en")
    wikipedia.headers = {
        "User-Agent": "" #### Insert your User-Agent
    }

    try:
        print(f"Searching for: {label}")
        page = wikipedia.page(label, auto_suggest=False)
        print(f"Found page: {page.title}")
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for '{label}'. Multiple options are available:")
        for i, option in enumerate(e.options, start=1):
            print(f"{i}. {option}")
        while True:
            try:
                selected_index = int(input("Select the correct page (enter number): ")) - 1
                if 0 <= selected_index < len(e.options):
                    selected_option = e.options[selected_index]
                    print(f"You selected: {selected_option}")
                    content = wikipedia.page(selected_option).content
                    return content
                else:
                    print("Invalid selection. Please enter a number corresponding to the options.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as new_error:
                print(f"Error fetching the selected page: {new_error}")
                return None
    except wikipedia.exceptions.PageError:
        print(f"Page for '{label}' does NOT exist.")
        return None
    except Exception as e:
        print(f"Unexpected error for '{label}': {e}")
        return None


def extract_wikipedia_triples(text: str, property_embeddings: dict) -> List[tuple]:
    """
    It extracts triples from Wikipedia text and filters them using keywords and similarity thingies.
    """
    with StanfordOpenIE() as client:
        triples = client.annotate(text)

    print(f"Extracted {len(triples)} triples from text.")

    formatted_triples = [(triple['subject'], triple['relation'], triple['object']) for triple in triples]

    keyword_filtered_triples = filter_triples_by_similarity(
        formatted_triples, property_embeddings, use_keyword_filter=True, use_embedding_filter=False
    )

    embedding_filtered_triples = filter_triples_by_similarity(
        keyword_filtered_triples, property_embeddings, use_keyword_filter=False, use_embedding_filter=True
    )

    return embedding_filtered_triples


def cluster_and_deduplicate_relations(relations, model="text-embedding-ada-002", epsilon=0.05, min_samples=1):
    """
    Group and deduplicate semantically similar relationships using DBSCAN.

    Args:
        relations (list): List of relations to process.
        model (str): OpenAI model to be used to compute embeddings.
        epsilon (float): Maximum distance to consider two relations in the same cluster.
        min_samples (int): Minimum number of relationships for a cluster.

    Returns:
        tuple: List of representative (deduplicated) relations and their respective embeddings.
    """
    if not relations:
        print("No relations provided for clustering.")
        return [], []

    print(f"Computing embeddings for {len(relations)} relations...")

    embeddings = [compute_embedding(relation, model=model) for relation in relations]

    if len(embeddings) == 0:
        print("No embeddings were computed. Returning an empty list.")
        return [], []

    embeddings = np.array(embeddings)

    if embeddings.ndim != 2:
        print("Embeddings are not in 2D format. Skipping clustering.")
        return [], []

    embeddings = normalize(embeddings)

    similarity_matrix = cosine_similarity(embeddings, embeddings)

    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)

    clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
    cluster_labels = clustering.fit_predict(distance_matrix)

    clustered_relations = {}
    clustered_embeddings = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clustered_relations:
            clustered_relations[label] = relations[idx] 
            clustered_embeddings[label] = embeddings[idx]

    deduplicated_relations = list(clustered_relations.values())
    deduplicated_embeddings = list(clustered_embeddings.values())

    print(f"Clustering completed: {len(deduplicated_relations)} unique relations identified.")
    print("\nDeduplicated Relations:")
    for idx, relation in enumerate(deduplicated_relations, start=1):
        print(f"{idx}. {relation}")

    return deduplicated_relations, deduplicated_embeddings


def process_object(object_name, property_embeddings, theta=0.75, stats=None, use_embedding_filter=False):
    """
    Process a single object: extract relationships from ConceptNet and Wikipedia, apply filtering.
    """
    system_message = f"\nObject: {object_name}\nRelations:\n"

    # Relazioni da ConceptNet
    relations_conceptnet = get_conceptnet_relations(object_name) if use_kg else []
    filtered_relations_conceptnet = []
    if relations_conceptnet:
        relation_embeddings = compute_relation_embeddings(relations_conceptnet)
        filtered_relations_conceptnet, total = filter_relations_by_similarity(
            relation_embeddings, property_embeddings, threshold=theta, return_counts=True
        )
        if stats is not None:
            stats["ope_relations_total"] = stats.get("ope_relations_total", 0) + total
            stats["ope_relations_kept"] = stats.get("ope_relations_kept", 0) + len(filtered_relations_conceptnet)
            stats["ope_objects"] = stats.get("ope_objects", 0) + 1
        if len(filtered_relations_conceptnet) == 0:
            print(f"No relations from ConceptNet above threshold for '{object_name}'")

    else:
        print(f"No relations from ConceptNet for '{object_name}'")

    # Relazioni da Wikipedia
    relations_wikipedia = []
    if use_wiki:
        print(f"Fetching Wikipedia content for '{object_name}'...")
        wiki_content = fetch_wikipedia_content(object_name)
        if wiki_content:
            extracted_triples = extract_openie_triples(wiki_content)
            
            keyword_filtered_triples, total = filter_triples_by_similarity(
                extracted_triples, property_embeddings,
                use_keyword_filter=True, use_embedding_filter=False, threshold=theta, return_counts=True
            )
            if stats is not None:
                stats["ope_relations_total"] = stats.get("ope_relations_total", 0) + total
                stats["ope_relations_kept"] = stats.get("ope_relations_kept", 0) + len(keyword_filtered_triples)
                stats["ope_objects"] = stats.get("ope_objects", 0) + 1

            deduplicated_triples, deduplicated_embeddings = cluster_and_deduplicate_relations(
                [f"{triple[0]} {triple[1]} {triple[2]}" for triple, _ in keyword_filtered_triples]
            )

            for idx, relation in enumerate(deduplicated_triples, start=1):
                system_message += f"{idx}. {relation}\n"

            if use_embedding_filter:
                relations_wikipedia, total = filter_triples_by_similarity(
                    deduplicated_triples, property_embeddings,
                    use_keyword_filter=False, use_embedding_filter=True, threshold=theta, return_counts=True
                )
                if stats is not None:
                    stats["ope_relations_total"] = stats.get("ope_relations_total", 0) + total
                    stats["ope_relations_kept"] = stats.get("ope_relations_kept", 0) + len(relations_wikipedia)
                    stats["ope_objects"] = stats.get("ope_objects", 0) + 1

    filtered_relations = filtered_relations_conceptnet + relations_wikipedia

    if filtered_relations:
        system_message += "\n[Final Relations]\n"
        for relation, similarity in filtered_relations:
            start_label, rel_label, end_label = relation
            system_message += f"- {start_label} {rel_label} {end_label}\n"

    return system_message



def OPE_score_par(found_objects, rel_objects, user_request, theta=0.75, stats=None, llm_temperature=0):
    properties = ["dangerous"]
    property_embeddings = {}

    if use_kg:
        print("Computing embeddings for 'dangerous' property...")
        for prop in properties:
            embedding = compute_embedding(prop)
            property_embeddings[prop] = embedding

    print("Processing objects in parallel...")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda obj: process_object(obj, property_embeddings, theta=theta, stats=stats), 
            rel_objects
        ))

    system_message = (
        "You are an expert in object properties. For each object, analyze the provided relationships to determine the following property:\n"
        "- Dangerous: Assign a score from 1 to 5 for the object's danger level in isolation based on the following criteria:\n"
        "  1: Not dangerous. The object is completely safe under all circumstances and poses no risk of damage or harm.\n"
        "  2: Low danger. The object has minimal risk in normal conditions but could be slightly harmful or damaged in rare situations.\n"
        "  3: Moderate danger. The object can cause harm or become damaged if mishandled or used improperly in some scenarios.\n"
        "  4: High danger. The object poses a significant risk of harm or damage even in normal conditions, requiring careful handling.\n"
        "  5: Extremely dangerous. The object is highly risky or fragile, and its use or presence poses severe danger in almost all situations.\n"
        "Additionally, analyze how each object interacts with other objects in the scene. For each object, determine if it becomes more dangerous or susceptible to damage when associated with other objects. Assign a score from 1 to 5 for the level of danger CAUSED BY THE INTERACTION, based on the following criteria:\n"
        "  1: No added danger. The combination is completely safe, with no risk of harm or damage.\n"
        "  2: Low additional danger. The combination presents minimal risk of harm or damage, which could occur in rare cases.\n"
        "  3: Moderate additional danger. The combination could result in harm or damage under improper use or specific conditions.\n"
        "  4: High additional danger. The combination poses significant risk of harm or damage even in normal conditions, requiring caution.\n"
        "  5: Extremely dangerous. The combination is highly unsafe, with severe risk of harm or damage in almost all situations.\n"
        "Do not overestimate the risk unless there is clear evidence of a significant hazard. Remember that everyday objects are often safe when used correctly, and scores should reflect realistic risks, not hypothetical or exaggerated scenarios." 
        "Provide the properties in the following format without adding comments:\n"
        "Object: [object_name]\n"
        "Dangerous: [Score 1-5]\n"
        "DangerousWith: [list of ONLY other object names that add danger, e.g., 'object_name (score)', or empty list if no additional danger exists with other objects in the scene]\n"
        "EXAMPLES:\n"
        "##########\n"
        "Object: glass bowl\n"
        "Dangerous: 1\n"
        "DangerousWith: [hot water (2)]\n"
        "Explanation: The glass bowl in isolation is not dangerous (score 1). However, when combined with hot water, it can shatter if not heat-resistant, posing a moderate danger (score 2).\n"
        "##########\n"
        "Object: fragile vase\n"
        "Dangerous: 1\n"
        "DangerousWith: [heavy objects (4)]\n"
        "Explanation: The vase has low inherent danger (score 1), but placing heavy objects on it can cause it to break, leading to sharp fragments (score 4).\n"
    )


    for result in results:
        system_message += result

    user_message = (
        f"User Request: {user_request}\n"
        "Found Objects in Scene: " + ", ".join(found_objects) +
        "\nAnalyze each object individually and determine which objects increase its danger level (only among the objects found and taking into account the user request)."
    )

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
        ],
        temperature=llm_temperature
    )

    response_message = request.choices[0].message
    content = response_message.content
    print("\nGPT-4o-mini Response:")
    print(content)
    objects_info = parse_gpt_response_with_context(content)
    return objects_info




def parse_gpt_response_with_context(response):
    """
    Parsing of GPT response to obtain score and list of objects that increase hazard.
    """
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
            objects_info[current_obj] = {"score": None, "dangerous_with": []}
        elif line.startswith("Dangerous:") and current_obj:
            try:
                score = int(line.split(":", 1)[1].strip())
                objects_info[current_obj]["score"] = score
            except ValueError:
                print(f"Invalid score format for object '{current_obj}'.")
        elif line.startswith("DangerousWith:") and current_obj:
            dangerous_with = line.split(":", 1)[1].strip()
            if dangerous_with.startswith("[") and dangerous_with.endswith("]"):
                dangerous_with = dangerous_with[1:-1].split(",")
                objects_info[current_obj]["dangerous_with"] = [item.strip() for item in dangerous_with if item.strip()]
    return objects_info

def process_object_names(found_objects):
    return [obj.strip().capitalize() for obj in found_objects]
