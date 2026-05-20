# moduli/urp.py

import openai
import os
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
from scripts.modules.pipeline_config import get_mode_pipeline
import numpy as np


# Load the spaCy language model
#nlp = spacy.load("en_core_web_sm")

# Configuration flags
use_request_processing = True
use_KG_query_objects = True
use_KG_query_request= True
use_example = True
use_obj_query = True


def _verbose_prompts():
    return os.getenv("CONCEPTBOT_VERBOSE_PROMPTS", "0") == "1"



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

    filtered_relations.sort(key=lambda x: x[1], reverse=True)
    if return_counts:
        return filtered_relations, total
    return filtered_relations

def list_to_string(found_obj):
    result_string = ', '.join(found_obj)
    return result_string


def _resolve_pipeline(mode, pipeline_config):
    if pipeline_config is not None:
        return dict(pipeline_config)
    return get_mode_pipeline(mode)


def _cache_kind(prefix, suffix):
    return f"{prefix}_{suffix}" if prefix != "urp" else f"urp_{suffix}"


def _standard_examples():
    if not use_example:
        return " "
    return """
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


def _implicit_examples():
    if not use_example:
        return " "
    return """
            IMPLICIT REQUEST EXAMPLES:
            ########
            Input: The user states a need, such as being thirsty, hungry, tired, or wanting to host people.
            Output: Infer the useful object categories needed to satisfy that need, then choose suitable objects available in the scene.
            ########
            Input: The user describes a problem, such as a spill, dirty object, or messy surface.
            Output: Bring the appropriate tool or object that helps solve the problem, rather than moving the problematic object itself.
            ########
            Input: The user asks for everything in a category.
            Output: Select all available objects that match that category, not just one representative.
            ########
            Input: The user asks to prepare, make, or serve something.
            Output: Gather all necessary available ingredients or serving items implied by that request.
            ########
            Input: The user asks to refresh, refrigerate, preserve, or store items.
            Output: Put all available perishable items in the refrigerator if a refrigerator is available.
            ########
            """


def _build_system_env(mode, found_objects, objects_info, use_OPE):
    string_found_objects = list_to_string(found_objects)

    if not use_OPE:
        return "\nOBJECTS in the scene: " + string_found_objects

    obj_info_str = str(objects_info)
    if mode == "risk":
        return (
            "\nOBJECTS in the scene and risk-evaluations (Carefully analyze these properties):\n "
            + obj_info_str
            + "\nThe risk score ranges from 1 to 5, the higher the score the greater the risk. "
            + "DangerousWith[] describes how dangerous the interaction with other elements is."
        )
    if mode == "materials":
        return "\nOBJECTS in the scene and extracted material information:\n " + obj_info_str
    if mode == "toxicity":
        return "\nOBJECTS in the scene and extracted toxicity-related properties:\n " + obj_info_str
    return "\nOBJECTS in the scene and extracted properties:\n " + obj_info_str + "\nProperty values are usually Yes/No unless otherwise specified."


def _category_task_rules(mode):
    if mode == "materials":
        return """

        MATERIAL-SORTING RULES:
        - Treat bins, baskets, containers, and sorting areas as destinations only; never ask the robot to pick them up.
        - Use exact destination names available in the scene; do not invent missing bins or categories.
        - Move every non-destination object whose inferred material maps to an available matching destination.
        - Use a mixed or general container for composite, coated, multi-material, or uncertain objects when such a destination is available.
        - When the request asks for exclusive membership in a material category, include only objects with clear evidence for that material.
        """
    if mode == "toxicity":
        return """

        TOXICITY-SORTING RULES:
        - Treat bins, containers, and areas as destinations only; never ask the robot to pick them up.
        - Use exact destination names available in the scene; do not invent missing safety, toxic, hazardous, or non-hazardous destinations.
        - Place objects inferred as toxic, poisonous, venomous, hazardous, or unsafe into the requested hazardous/safety destination when available.
        - Place clearly non-toxic or safe objects into the requested standard or non-hazardous destination when available.
        - If an object is uncertain and a mixed, testing, or verification destination exists, use that destination rather than treating the object as clearly safe.
        """
    if mode == "risk":
        return """

        RISK-AWARE TASK RULES:
        - Treat people, appliances, bins, and areas as destinations only; never ask the robot to pick them up.
        - If a request can be satisfied by choosing a safer suitable alternative, choose that alternative and do not move unsafe alternatives.
        - For heat, appliances, stacking, sharp objects, fragile objects, chemicals, or object-object interactions, use the provided risk scores to avoid unsafe pairings and destinations.
        - When ordering matters for safety, move protective or supporting objects before placing risky or fragile objects on or near them.
        - Preserve the user's goal, but prefer plans that reduce physical harm, object damage, breakage, or unsafe interaction.
        """
    if mode == "implicit":
        return """

        IMPLICIT TASK RULES:
        - For spills, dirty items, or dirty surfaces, bring an available cleaning tool to the user unless a destination is explicitly requested.
        - For hospitality or serving requests, choose a concise set of suitable food/drink items and place them on the requested serving destination when available.
        - For health-related sorting, use commonsense nutritional categories and preserve both positive and negative groups when both are requested.
        - For preparation requests, gather the available ingredients or serving items that are necessary for the implied task.
        - For requests involving drinks, include available beverages and appropriate drinking vessels only when the request asks for all or everything drinkable.
        """
    return """

        STANDARD TASK RULES:
        - For spills or dirty items, bring the sponge or cleaning tool to the user unless the user explicitly names another destination. Do not place the cleaning tool on the table or spill location.
        - If the user asks for an item to be brought or given and no destination is specified, use the user as destination.
        """


def _build_system_goal(mode, use_OPE):
    if mode == "materials":
        return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rewrite the user's material-sorting request into robot-executable pick-and-place instructions. "
        "The robot can ONLY perform pick and place operations. "
        "You are given objects in the scene and their extracted material information. Use the material information to decide which objects belong in each destination bin. "
        "Place each object into the bin matching its material category when a matching bin exists. If an object is made of multiple materials, use the mixed bin when available. If the material is uncertain, use the mixed bin when available. "
        "Treat bins, baskets, containers, and sorting areas as destinations, not as objects to be moved; never output a pick-and-place instruction for a bin, basket, container, or sorting area itself. "
        "Do not invent objects or destinations. Do not ask the robot to inspect, cut, wash, or otherwise manipulate objects beyond pick-and-place.\n\n"
        "Provide your response in two sections. First, provide a 'Reasoning:' section where you briefly explain which objects match which materials. Second, provide an 'Answer:' section where you give concise robot instructions."
        """
    if mode == "toxicity":
        return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rewrite the user's toxicity-sorting request into robot-executable pick-and-place instructions. "
        "The robot can ONLY perform pick and place operations. "
        "You are given objects in the scene and their extracted toxicity-related properties, such as poisonous, toxic, venomous, hazardous, safe, or non-toxic. "
        "Place toxic, poisonous, venomous, or hazardous objects in the designated toxic, hazardous, venomous, safety, or verification container requested by the user. Place clearly non-toxic or safe objects in the standard, public, or non-hazardous container. "
        "If the instruction provides a mixed, verification, or testing container and toxicity is uncertain, use that container instead of treating the object as clearly safe. "
        "Treat bins, baskets, containers, and sorting areas as destinations, not as objects to be moved; never output a pick-and-place instruction for a bin, basket, container, or sorting area itself. "
        "Be conservative with safety-critical objects. Do not invent objects or destinations.\n\n"
        "Provide your response in two sections. First, provide a 'Reasoning:' section where you briefly explain which objects are toxic or non-toxic. Second, provide an 'Answer:' section where you give concise robot instructions."
        """
    if mode == "risk":
        if use_OPE:
            return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations. "
        "If the user requests a single item by category, choose exactly one best matching object unless the user explicitly asks for all, every, or multiple objects. "
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
        return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations. "
        "If the user asks for something, consider the 'user' position to give something to the user. For complex requests, try to break down the solution into smaller tasks.\n\n"
        "Provide your response in two sections. First, provide a 'Reasoning:' section where you explain your thought process for solving the request, explaining why one item is better than another to meet the demand. Second, provide an 'Answer:' section where you give the robot instructions to execute the task."
        """
    if mode in ("standard", "implicit"):
        if mode == "implicit":
            return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to infer the user's unstated practical need and rewrite it into robot-executable pick-and-place instructions. "
        "The robot can ONLY perform pick and place operations. "
        "Use the available objects and their extracted properties to infer which objects satisfy the hidden need. "
        "Do not invent objects, destinations, tools, or actions outside pick-and-place.\n\n"

        "IMPLICIT REQUEST RULES:\n"
        "- If the instruction expresses a need or state, infer the useful object category that satisfies it, then choose available objects from that category.\n"
        "- If the instruction expresses multiple needs or states, choose one suitable object for each need unless the user explicitly asks for all or everything.\n"
        "- If the instruction describes a problem, choose the available tool or object that helps solve the problem; do not move the problematic object unless moving it is itself the solution.\n"
        "- If the problem requires a cleaning or repair tool, bring the tool to the user unless the user explicitly asks to place the tool at another valid destination.\n"
        "- If the instruction asks for a single object by category, choose exactly one representative object.\n"
        "- Words such as 'a', 'an', or 'something' indicate a single representative object unless the instruction also says all, every, or everything.\n"
        "- If the instruction asks for all or everything in a category, select every available object that belongs to that category and use the user's location if it is a delivery request.\n"
        "- If the instruction contrasts positive and negative groups, preserve both parts of the request: move the negative group to the requested disposal/storage destination and move the positive group to the requested destination.\n"
        "- For health-related sorting, sugary snacks and soft drinks are usually unhealthy; fruits, vegetables, water, tea, and plain nutrition or energy bars may be treated as healthier unless described otherwise.\n"
        "- If the instruction names a prepared food, sauce, or recipe, infer the standard ingredients and gather all necessary available ingredients rather than selecting only one ingredient. Include main, fat, liquid, acidic, or seasoning components when they are available and relevant.\n"
        "- If ingredients are being gathered and no destination is specified, deliver them to the user or to a suitable bowl if one is available; do not use an arbitrary surface unless serving is requested.\n"
        "- If the instruction implies preparation or serving, gather all necessary available ingredients or serving items implied by the request.\n"
        "- If the instruction implies refrigeration, freshness, refreshment, or preservation, store every available perishable item in the refrigerator when it is a valid destination. Perishable items include eggs, dairy/fat ingredients, fresh produce, citrus, and food ingredients that can spoil or benefit from cold storage.\n"
        "- If the instruction implies hospitality or welcoming guests, choose a concise serving set, usually one snack or food item and one drink item, and place them on a serving surface rather than bringing them to the user. Do not move every possible refreshment unless explicitly requested.\n"
        "- If the instruction asks for one drinkable item, choose exactly one suitable beverage.\n"
        "- If the instruction asks for all or everything to drink, include all available beverages, unfamiliar branded beverage names that may be drinks, water, soda-like drinks, tea-like drinks, and appropriate drinking vessels such as cups or mugs. Exclude condiment bottles or non-drink containers.\n"
        "- For spills, dirty items, or dirty surfaces, bring an available cleaning tool to the user rather than placing the tool on the spill, table, dirty item, or another work surface.\n"
        "- Preserve every requested action in compound instructions.\n"
        "- Use only destinations that are present in the scene or clearly implied by the user request.\n\n"

        "OUTPUT RULES:\n"
        "- Provide one explicit pick-and-place instruction per object.\n"
        "- Do not combine multiple objects into a single instruction if they will be moved separately.\n"
        "- Provide your response in two sections: 'Reasoning:' and 'Answer:'."
        """
        return """
        "You are an expert on the Franka Emika Panda robot. Rewrite the user's request into robot-executable pick-and-place instructions. "
        "The robot can ONLY perform pick-and-place actions with the objects and destinations available in the scene. "
        "Use the extracted object properties and ConceptNet relations as evidence, but do not invent objects, destinations, or capabilities.\n\n"

        "GENERAL RULES:\n"
        "1. Preserve all explicit objects, actions, constraints, and destinations in the user request.\n"
        "2. If the request is implicit or describes a problem, infer the practical need and choose available objects that satisfy or help solve that need.\n"
        "3. For singular requests choose one best matching object; for all, every, or everything requests choose all matching available objects.\n"
        "4. If no destination is specified for a delivery request or for a useful object/tool, use the user as destination; otherwise use only destinations present in the scene or clearly implied by the request.\n"
        "5. Prefer the object whose name, common use, or ConceptNet relations most directly match the requested category; when relations are missing, use extracted properties conservatively rather than ignoring the object.\n"
        "6. Prefer safe, stable, and task-suitable objects; avoid dangerous, poisonous, or unsuitable objects unless the user explicitly requests them.\n\n"

        "Provide your response in two sections: 'Reasoning:' and 'Answer:'. In the Answer, write one explicit pick-and-place instruction for each object that must be moved."
        """
    if use_OPE:
        return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations."
        "Consider the most stable and safe solution. "
        "If the user requests an item, and you are in doubt between several items, bring back the items that seem most correct to you but specify very clearly that only one (or two or the exact number as the case may be) is needed anyway. "
        "Try to understand what the user's needs are and what you can do to meet them. Try to understand what the user is giving relevance to.\n\n"
        
        "IMPORTANT: You are provided with a set of objects and their associated properties. Use this information carefully to ensure both SAFETY and the SUCCESSFUL execution of the task:\n"
        "GENERAL EXECUTION RULES:\n"
        "- Preserve every object explicitly requested by the user unless the request asks for a category and only one representative object is needed.\n"
        "- Preserve every requested action in compound instructions, especially when different verbs or destinations are used.\n"
        "- A delivery clause must not replace or remove a disposal, placement, sorting, or moving clause from the same instruction.\n"
        "- If the user says 'bring me', 'give me', or 'I want' and no other destination is explicitly specified, place the selected object or objects in front of the user.\n"
        "- If the request asks for a useful tool or object but does not name a valid destination, bring the useful object to the user.\n"
        "- For singular category requests such as 'a', 'an', or 'something', choose one representative object, not all possible alternatives.\n"
        "- For requests with exclusions such as 'not X', do not select X and choose one suitable alternative unless multiple alternatives are explicitly requested.\n"
        "- Do not select objects from a different semantic category merely because they are generally safe, edible, or useful.\n"
        "- When a category is ambiguous, prefer the object whose name or common use most directly matches the requested category over a broader adjacent category.\n"
        "- If multiple requested objects share the same destination, state each object separately and keep the same destination for all of them.\n"
        "- Use only destinations that are present in the scene or clearly implied by the user request. Avoid relational destinations such as 'next to another object' unless such a place is explicitly available.\n"

        "Provide your response in two sections. First, provide a 'Reasoning:' section where you explain your thought process for solving the request, using the properties of the objects that have been given to you previously and explaining why one item is better than another to meet the demand. Second, provide an 'Answer:' section where you give the robot instructions to execute the task."
        """
    return """
        "You are an expert on the Franka Emika Panda robot. Your goal is to rework the user's high-level request into a request that the robot can understand and fulfill. "
        "Consider that the robot can ONLY perform pick and place operations."
        "Consider the most stable and SAFE solution to not damage the elements and the user, even if it means considering alternative objects to the request. "
        "If the user asks for something, consider the 'user' position to give something to the user. For complex requests, try to break down the solution into smaller tasks. "
        "If the user requests a single item by category, choose exactly one best matching object unless the user explicitly asks for all, every, or multiple objects. "
        "Try to understand what the user's needs are and what you can do to meet them. Try to understand what the user is giving relevance to.\n\n"
        "GENERAL EXECUTION RULES:\n"
        "- Preserve every object explicitly requested by the user unless the request asks for a category and only one representative object is needed.\n"
        "- Preserve every requested action in compound instructions, especially when different verbs or destinations are used.\n"
        "- A delivery clause must not replace or remove a disposal, placement, sorting, or moving clause from the same instruction.\n"
        "- If the user says 'bring me', 'give me', or 'I want' and no other destination is explicitly specified, place the selected object or objects in front of the user.\n"
        "- If the request asks for a useful tool or object but does not name a valid destination, bring the useful object to the user.\n"
        "- For singular category requests such as 'a', 'an', or 'something', choose one representative object, not all possible alternatives.\n"
        "- For requests with exclusions such as 'not X', do not select X and choose one suitable alternative unless multiple alternatives are explicitly requested.\n"
        "- Do not select objects from a different semantic category merely because they are generally safe, edible, or useful.\n"
        "- When a category is ambiguous, prefer the object whose name or common use most directly matches the requested category over a broader adjacent category.\n"
        "- If multiple requested objects share the same destination, state each object separately and keep the same destination for all of them.\n"
        "- Use only destinations that are present in the scene or clearly implied by the user request. Avoid relational destinations such as 'next to another object' unless such a place is explicitly available.\n"

        "Provide your response in two sections. First, provide a 'Reasoning:' section where you explain your thought process for solving the request, explaining why one item is better than another to meet the demand. Second, provide an 'Answer:' section where you give the robot instructions to execute the task."
        """


def URP(
    user_message,
    found_objects,
    objects_info,
    use_OPE,
    rel_objects,
    theta=0.75,
    stats=None,
    llm_temperature=0,
    mode="standard",
    pipeline_config=None,
):
    pipeline_config = _resolve_pipeline(mode, pipeline_config)
    mode = pipeline_config.get("urp_mode", mode)
    cache_prefix = pipeline_config.get("urp_cache_prefix", "urp")

    system_goal = _build_system_goal(mode, use_OPE) + _category_task_rules(mode)
    system_examples = _standard_examples()
    if mode == "implicit":
        system_examples += _implicit_examples()
    system_env = _build_system_env(mode, found_objects, objects_info, use_OPE)
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
                kind=_cache_kind(cache_prefix, "keyword_to_request"),
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
                kind=_cache_kind(cache_prefix, "object_to_keywords"),
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
                kind=_cache_kind(cache_prefix, "keyword_relations_to_request"),
            )
            conceptnet_relations_key.extend([(relation, similarity) for relation, similarity in relation_scores if similarity >= theta])

        filtered_relations_key = conceptnet_relations_key

        system_message += "\nRelevant Keyword Relations from ConceptNet (Keyword-User Request Similarity):\n"
        for relation, similarity in filtered_relations_key:
            system_message += f"- {relation[0]} {relation[1]} {relation[2]} (Similarity: {similarity:.2f})\n"


    if _verbose_prompts():
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
        log_openai_call(cache_prefix, user_message, time.monotonic() - start)
        response_message = response.choices[0].message.content

        if _verbose_prompts():
            print('\nResponse Message:')
            print(response_message)

        reasoning = response_message.split("Reasoning:")[1].split("Answer:")[0].strip()
        answer = response_message.split("Answer:")[1].strip()

        if _verbose_prompts():
            print(f"\nReasoning:\n{reasoning}\n")
            print(f"Answer:\n{answer}\n")

    return answer

