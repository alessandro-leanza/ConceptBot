import openai
import numpy as np
import tiktoken 
from heapq import nlargest
import matplotlib.pyplot as plt

def gpt3_scoring(query, call_law_start, commands_string, options, model="gpt-4o-mini", top_logprobs=5, verbose=False):
    messages = [
        {"role": "system", "content": call_law_start + '\n' + commands_string},
        {"role": "user", "content": 'USER INPUT:\n' + query + 'NEXT STEP IS:'}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=128,
        logprobs=True,
        top_logprobs=top_logprobs
    )
    content = response.choices[0].message.content.strip()
    verbose and print('Model response:', content)
    logprobs_data = response.choices[0].logprobs
    tokens = [token_logprob.token for token_logprob in logprobs_data.content]

    encoding = tiktoken.encoding_for_model(model)
    top_logprobs_list = []
    for token_logprob in logprobs_data.content:
        top_logprob_dict = {top_logprob.token: top_logprob.logprob for top_logprob in token_logprob.top_logprobs}
        top_logprobs_list.append(top_logprob_dict)

    scores = {}
    for option in options:
        success = True
        total_logprob = 0
        option_tokens = encoding.encode(option)
        for i, token_id in enumerate(option_tokens):
            if i >= len(top_logprobs_list):
                success = False
                break
            token_str = encoding.decode([token_id])
            top_logprob = top_logprobs_list[i]
            if token_str in top_logprob:
                total_logprob += top_logprob[token_str]
            else:
                total_logprob += -100
        if success:
            scores[option] = total_logprob

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    for i, (option, score) in enumerate(sorted_scores):
        if verbose:
            print(f"{i+1}. {option} (logprob: {score})")
        if i >= 9:
            break

    return scores, response

def affordance_scoring(PLACE_TARGETS, options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl")
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 1 
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores

def make_options(pick_targets, place_targets, termination_string="done()"):
    options = []
    for pick in pick_targets:
        for place in place_targets:
            option = f"robot.pick_and_place({pick}, {place})"
            options.append(option)
    options.append(termination_string)
    return options

def normalize_scores(scores):
    min_score = min(scores.values())
    max_distance_from_zero = abs(min_score)
    normed_scores = {key: (max_distance_from_zero - abs(score)) / max_distance_from_zero for key, score in scores.items()}
    return normed_scores

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description

gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

...

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""
use_environment_description = True

def SAYCAN_prop(found_objects, PICK_TARGETS, PLACE_TARGETS, query, objects_info, max_tasks=10):
    steps_text = []
    combined_scores = {}

    termination_string = 'done()'
    call_law_start = (
        "After \\\\'USER INPUT\\\\' there will be a request to satisfy and there may already be actions trying to satisfy it. "
        "You must choose from the available options which one you think is next given the previous ones. Return only the option you have chosen, without adding or removing anything else! "
        "It is important that the option be chosen from the set of possible options defined within \\\\'BEGIN OPTIONS AVAILABLES\\\\' through \\\\'END OPTIONS\\\\'. DO NOT USE OTHER OPTIONS THAT ARE NOT PRESENT HERE! "
        "You have available also a scene description with the objects observed pickable in the scene, this description is defined within \\\\'BEGIN SCENE DESCRIPTION\\\\' through \\\\'END SCENE DESCRIPTION\\\\'. "
        "In addition to objects there are also properties, consider in policy generation these properties to achieve a feasible, stable and safe policy."
        "IMPORTANT: Answer \\'done()\\' when you think the request has been satisfied."
    )

    options_begin = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
    commands_string = "\n".join(options_begin)
    commands_string = 'BEGIN OPTIONS AVAILABLES:\n' + commands_string + '\nEND OPTIONS.\n'

    scene_description = build_scene_description(found_objects)
    if objects_info:
        properties_description = "\n".join(
            [f"{obj}: {props}" for obj, props in objects_info.items()]
        )
        scene_description += "\nProperties:\n" + properties_description

    env_description = 'BEGIN SCENE DESCRIPTION:\n' + scene_description + '\nEND SCENE DESCRIPTION.\n'

    gpt3_prompt = gpt3_context + call_law_start
    gpt3_prompt += "\n" + env_description

    all_llm_scores = []
    all_affordance_scores = []
    all_combined_scores = []
    affordance_scores = affordance_scoring(PLACE_TARGETS, options_begin, found_objects, block_name="box", bowl_name="circle", verbose=False)
    num_tasks = 0
    selected_task = ""
    steps_text = []

    while not selected_task == termination_string:
        num_tasks += 1
        if num_tasks > max_tasks:
            break

        options_begin = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
        if selected_task in options_begin:
            options_begin.remove(selected_task)

        llm_scores, res = gpt3_scoring(query, gpt3_prompt, commands_string, options_begin, model="gpt-4o-mini", top_logprobs=5, verbose=True)

        normalized_llm_scores = normalize_scores(llm_scores)
        combined_scores = {}
        for option, llm_score in normalized_llm_scores.items():
            if option in affordance_scores:
                affordance_score = affordance_scores[option]
                combined_scores[option] = llm_score * affordance_score
        if combined_scores:
            selected_task = max(combined_scores, key=combined_scores.get)
            print("Selecting:", selected_task)

        steps_text.append(selected_task)
        query += '\n' + selected_task
        all_llm_scores.append(llm_scores)
        all_affordance_scores.append(affordance_scores)
        all_combined_scores.append(combined_scores)

    print('**** Solution ****')
    print(env_description)
    print('# ' + query)
    for i, step in enumerate(steps_text):
        if step == '' or step == termination_string:
            break
        print('Step ' + str(i) + ': ' + step)

    return steps_text
