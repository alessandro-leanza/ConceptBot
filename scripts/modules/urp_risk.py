# moduli/urp.py

import openai
import time
#import spacy
import requests
from scripts.modules.conceptnet_backend import get_conceptnet_relations as cn_get_relations
from scripts.modules.semantic_cache import (
    get_cached_embedding,
    get_cached_keywords,
    get_openai_client,
    get_cached_urp_object_keyword_similarities,
    get_cached_urp_request_similarities,
    log_openai_call,
)
import numpy as np


# Load the spaCy language model
#nlp = spacy.load("en_core_web_sm")

# Configuration flags
use_request_processing = True
use_KG_query_objects = True
use_KG_query_request= True
use_example = True
use_obj_query = True



#def extract_keywords_from_message(message):
#    doc = nlp(message)
#    keywords = []
#    for token in doc:
#        if token.dep_ in ['nsubj', 'ROOT', 'dobj', 'iobj', 'pobj', 'attr', 'advcl', 'agent', 'oprd', 'xcomp', 'ccomp']:
#            keywords.append(token.text)
#    return keywords


def extract_keywords_llm(text, llm_temperature=0):
    keywords_list = get_cached_keywords(text, model="gpt-4o-mini", llm_temperature=llm_temperature)
    print('Keywords: ', keywords_list)
    return keywords_list


def compute_embedding(text, model="text-embedding-ada-002"):
    return get_cached_embedding(text, model=model)


def get_conceptnet_relations(keyword):
    #relevant_relations = {'RelatedTo', 'IsA', 'UsedFor', 'HasProperty', 'CapableOf', 'MannerOf'}
    relevant_relations = {'IsA', 'UsedFor', 'HasProperty', 'CapableOf', 'MannerOf'}
    relations = cn_get_relations(
        keyword,
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


def filter_relations_by_similarity(relation_embeddings, reference_embedding, threshold=0.75, return_counts=False):
    total = len(relation_embeddings)
    filtered_relations = []
    for relation, rel_emb in relation_embeddings:
        similarity = cosine_similarity(rel_emb, reference_embedding)
        if similarity >= threshold:
            filtered_relations.append((relation, similarity))
    filtered_relations.sort(key=lambda x: x[1], reverse=True)
    print(filtered_relations)
    if return_counts:
        return filtered_relations, total
    return filtered_relations

def filter_relations_by_similarity_obj_key(relation_embeddings, property_embeddings, threshold=0.75, return_counts=False):
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
    # Ordina le relazioni per punteggio di similarità
    filtered_relations.sort(key=lambda x: x[1], reverse=True)
    if return_counts:
        return filtered_relations, total
    return filtered_relations

def list_to_string(found_obj):
    result_string = ', '.join(found_obj)
    return result_string

def URP_risk(user_message, found_objects, objects_info, use_OPE, rel_objects, theta=0.75, stats=None, llm_temperature=0):
    string_found_objects = list_to_string(found_objects)

    if use_OPE:
        obj_info_str = str(objects_info)
        system_env = "\nOBJECTS in the scene and risk-evaluations (Carefully analyzes these properties):\n " + obj_info_str + "\n The risk score ranges from 1 to 5, the higher the score the greater the risk. Instead, within DangerousWith[] is told how dangerous the interaction with other elements is"
    else:
        system_env = "\nOBJECTS in the scene: " + string_found_objects

    system_goal = ""
    system_examples = " "

    if use_OPE:
        system_goal = """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations. "
        "If the user requests an item, and you are in doubt between several items, bring back the items that seem most correct to you but specify very clearly that only one (or two or the exact number as the case may be) is needed anyway. "
        "Try to understand what the user's needs are and what you can do to meet them. Try to understand what the user is giving relevance to.\n\n"
        
        "IMPORTANT: You are provided with a set of objects and their associated risk evaluations. Use this information carefully to ensure both SAFETY and the SUCCESSFUL execution of the task:\n"
        "- Each object has a 'Dangerous' score ranging from 1 to 5:\n"
        "  - 1: The object is completely safe under all circumstances.\n"
        "  - 2: The object poses minimal risk but could be slightly harmful or damaged in rare cases.\n"
        "  - 3: The object can cause harm or become damaged if mishandled or used improperly in some scenarios.\n"
        "  - 4: The object poses significant risk or fragility, even in normal conditions.\n"
        "  - 5: The object is extremely dangerous or fragile, posing a severe risk in almost all scenarios.\n\n"
        "- Objects may also have a 'DangerousWith' list, which identifies specific objects in the scene that increase the danger level of the analyzed object when combined. The list includes the interacting object name and an interaction danger score (1 to 5):\n"
        "  - 1: The combination is completely safe.\n"
        "  - 2: The combination presents minimal additional risk.\n"
        "  - 3: The combination could result in harm or damage.\n"
        "  - 4: The combination poses a significant additional risk.\n"
        "  - 5: The combination is highly unsafe and must be avoided.\n\n"
        "USAGE GUIDELINES:\n"
        "- Avoid using objects with high 'DangerousWith' interaction scores ( (4) or (5) ), to consider the others items with lower interaction risk, unless absolutely necessary. Consider the 'DangerousWith' list when multiple objects are involved: the important thing is to meet the user's true need, even with alternative objects.\n"
        
        "Provide your response in two sections. First, provide a 'Reasoning:' section where you explain your thought process for solving the request, using the properties of the objects that have been given to you previously and explaining why one item is better than another to meet the demand. Second, provide an 'Answer:' section where you give the robot instructions to execute the task."
        """
    else:
        system_goal = """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations. "
        "If the user asks for something, consider the 'user' position to give something to the user. For complex requests, try to break down the solution into smaller tasks.\n\n"

        "Provide your response in two sections. First, provide a 'Reasoning:' section where you explain your thought process for solving the request, explaining why one item is better than another to meet the demand. Second, provide an 'Answer:' section where you give the robot instructions to execute the task."
        """


    if use_example:
        system_examples ="""
            EXAMPLES:
            ########
            Input: 'I want' / 'Bring me' an apple.
            Output: Pick the apple and place it in front of the user.
            ########
            Input: Organize the blue items together.
            Output: Pick the blue blocks and place them in the same bowl (if there is a bowl in the scene) or position (angle or middle).
            ########
            Input: Create a tower of blocks.
            Output: Pick the blocks and place them on top of each other to form the highest possible stack. There is no need to change the position of the first block.
            ########
            Input: Sort items by color.
            Output: Pick the items and put them in the bowls of the same color. If there are no bowls, then put them in different corners according to color.
            ########
            Input: Group the red blocks.
            Output: Pick the red blocks and place them together in a single position or bowl.
            ########
            ########
            Input: Place the conductive items in a safe location.
            Output: Pick the conductive items and place them in a safe, non-conductive area to prevent electrical hazards.
            ########
            """
 

    system_message = system_goal + system_examples + system_env

    if use_KG_query_request:
        keywords = extract_keywords_llm(user_message, llm_temperature=llm_temperature)
        conceptnet_relations = []
        for keyword in keywords:
            relations = get_conceptnet_relations(keyword)
            relation_scores = get_cached_urp_request_similarities(
                instruction=user_message,
                query=keyword,
                relations=relations,
                kind="urp_risk_keyword_to_request",
            )
            total = len(relation_scores)
            filtered_relations_kw = [(relation, similarity) for relation, similarity in relation_scores if similarity >= theta]
            filtered_relations_kw.sort(key=lambda x: x[1], reverse=True)
            conceptnet_relations.extend(filtered_relations_kw)
            if stats is not None:
                stats["urp_relations_total"] = stats.get("urp_relations_total", 0) + total
                stats["urp_relations_kept"] = stats.get("urp_relations_kept", 0) + len(filtered_relations_kw)
                stats["urp_keywords"] = stats.get("urp_keywords", 0) + 1
                if not filtered_relations_kw:
                    stats["urp_zero_relation_keywords"] = stats.get("urp_zero_relation_keywords", 0) + 1
                    stats.setdefault("urp_zero_relation_keyword_names", []).append(keyword)

        filtered_relations = conceptnet_relations
        system_message += "\nRelevant Relations from ConceptNet:\n"
        for relation, similarity in filtered_relations:
            system_message += f"- {relation[0]} {relation[1]} {relation[2]}\n"

    if use_KG_query_objects:
        keywords = extract_keywords_llm(user_message, llm_temperature=llm_temperature)

        conceptnet_relations_obj = []
        for obj in rel_objects:
            rel_obj = get_conceptnet_relations(obj)
            relation_scores = get_cached_urp_object_keyword_similarities(
                instruction=user_message,
                query=obj,
                relations=rel_obj,
                keywords=keywords,
                kind="urp_risk_object_to_keywords",
            )
            conceptnet_relations_obj.extend([(relation, similarity) for relation, similarity in relation_scores if similarity >= theta])
            if stats is not None:
                stats["urp_relations_total"] = stats.get("urp_relations_total", 0) + len(relation_scores)
                stats["urp_relations_kept"] = stats.get("urp_relations_kept", 0) + len(
                    [(relation, similarity) for relation, similarity in relation_scores if similarity >= theta]
                )

        filtered_relations_obj = conceptnet_relations_obj
        filtered_relations_obj.sort(key=lambda x: x[1], reverse=True)

        system_message += "\n\nRelevant Object Relations from ConceptNet (Object-Keyword Similarity):\n"
        for relation, similarity in filtered_relations_obj:
            system_message += f"- {relation[0]} {relation[1]} {relation[2]} (Similarity: {similarity:.2f})\n"

        conceptnet_relations_key = []
        for keyword in keywords:
            rel_key = get_conceptnet_relations(keyword)
            relation_scores = get_cached_urp_request_similarities(
                instruction=user_message,
                query=keyword,
                relations=rel_key,
                kind="urp_risk_keyword_relations_to_request",
            )
            conceptnet_relations_key.extend([(relation, similarity) for relation, similarity in relation_scores if similarity >= theta])

        filtered_relations_key = conceptnet_relations_key

        system_message += "\nRelevant Keyword Relations from ConceptNet (Keyword-User Request Similarity):\n"
        for relation, similarity in filtered_relations_key:
            system_message += f"- {relation[0]} {relation[1]} {relation[2]} (Similarity: {similarity:.2f})\n"


    print('System Message:\n')
    print(system_message)

    if use_request_processing:
        client = get_openai_client()
        start = time.monotonic()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=llm_temperature
        )
        log_openai_call("urp_risk", user_message, time.monotonic() - start)
        response_message = response.choices[0].message.content

        print('\nResponse Message:')
        print(response_message)

        reasoning = response_message.split("Reasoning:")[1].split("Answer:")[0].strip()
        answer = response_message.split("Answer:")[1].strip()

        print(f"\nReasoning:\n{reasoning}\n")
        print(f"Answer:\n{answer}\n")

    return answer

