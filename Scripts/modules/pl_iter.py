import openai
import numpy as np
import tiktoken 
from heapq import nlargest
import matplotlib.pyplot as plt


LLM_CACHE = {}

call_law_start = "After \\'USER INPUT\\' there will be a request to satisfy and there may already be actions trying to satisfy it. You must choose from the available options which one you think is next given the previous ones. Return only the option you have chosen, without adding or removing anything else! It is important that the option be chosen from the set of possible options defined within \\'BEGIN OPTIONS AVAILABLES\\' through \\'END OPTIONS\\'. DO NOT USE OTHER OPTIONS THAT ARE NOT PRESENT HERE!"

def gpt3_call(model="gpt-4o-mini", call_law='', query='', temperature=0, max_tokens=1000, logprobs=True, echo=False, top_logprobs=2):
    id = (model, call_law, query, max_tokens, temperature, logprobs, echo)

    LLM_CACHE = {} 

    if id in LLM_CACHE:
        print('Cache hit, returning')
        return LLM_CACHE[id], LLM_CACHE[id], LLM_CACHE[id]

    response = openai.chat.completions.create(
        model=model,
        messages = [
            {"role": "system", "content": call_law},
            {"role": "user", "content": 'USER INPUT:\n' + query + 'NEXT STEP IS:'}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs
    )

    response_message = response.choices[0].message
    content = response_message.content

    # Convert the answer to a dictionary if necessary and save it in the cache
    LLM_CACHE[id] = response if isinstance(response, dict) else response.dict()
    return LLM_CACHE[id], response, content

def gpt3_scoring(query, call_law_start, commands_string, options, model="gpt-4o-mini", limit_num_options=10, option_start="\n", verbose=False, print_tokens=False, top_logprobs=2):
    verbose and print("Scoring", len(options), "options")

    all_responses = []
    scores = {}
    i=1
    num_call=0
    num_err=0
    isError = False 
    chosen_option_error = ''

    query_error = 'Choose an option that is available! Keep in mind also the scene description!\n' + commands_string

    while i <= limit_num_options:
      num_call += 1 

      call_law = call_law_start + '\n' + commands_string

      LLM_CACHE = {} 

      if isError==False:
        response, log_response, content = gpt3_call(model=model, call_law=call_law, query=query, max_tokens=100, temperature=0, logprobs=True, top_logprobs = top_logprobs)
        num_err = 0
      else:
        response, log_response, content = gpt3_call(model=model, call_law=call_law, query=query_error, max_tokens=100, temperature=0, logprobs=True, top_logprobs = top_logprobs)

      all_responses.append(response)
      print('\nRESPONSE: ', content)

      chosen_option = content.strip()
      if chosen_option in options:
        
        isError = False
        chosen_option_error = ''

        i += 1
        print("Option matched:", chosen_option)
        options.remove(chosen_option)
        print('\n New set of options: ', options)
        commands_string = "\n".join(options)
        commands_string = 'BEGIN OPTIONS AVAILABLES:\n' + commands_string + '\nEND OPTIONS.\n'

        if isinstance(log_response, dict):
            logprobs_content = log_response['choices'][0]['logprobs']['content']
        else:
            logprobs_content = log_response.choices[0].logprobs.content

        total_logprob = 0
        for token_logprob in logprobs_content:
            if isinstance(token_logprob, dict):
                logprob = token_logprob['logprob']

            else:
                logprob = token_logprob.logprob
            total_logprob += logprob

        scores[chosen_option] = total_logprob

        ########

      else:
        isError = True
        if chosen_option not in chosen_option_error:
          chosen_option_error += chosen_option + '\n'
        query_error = query + '.\n Which is the next step? \nPlease note that\n' + chosen_option_error + 'are all WRONG answers and it is a big mistake if you respond with one of the above options. Respond with another possible right option for fullfill the request.' + 'IMPORTANT: If you think that the request is been already satisfied respond \'done()\'!.'
        print("Not present")
        num_err +=1
        print('\nError number: ', num_err)
        if num_err > 2:
          isError = False

      if num_call > 6:
        i = limit_num_options + 1



    query = query + chosen_option + '\n'

    for i, (option, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        if verbose:
            print(score, "\t", option)
        if i >= 9:
            break

    return scores, all_responses

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description

def affordance_scoring(PLACE_TARGETS, options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl")
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 1 
      continue
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


def step_to_nlp(step):
  step = step.replace("robot.pick_and_place(", "")
  step = step.replace(")", "")
  pick, place = step.split(", ")
  return "Pick the " + pick + " and place it on the " + place + "."

def normalize_scores(scores):
    # Trova il punteggio massimo e minimo
    min_score = min(scores.values())
    max_score = max(scores.values())
    max_distance_from_zero = abs(min_score)

    normed_scores = {}
    for key, score in scores.items():
        normed_scores[key] = (max_distance_from_zero - abs(score)) / max_distance_from_zero

    return normed_scores


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

def ITER(found_objects, PICK_TARGETS, PLACE_TARGETS, query, model="gpt-4o-mini", limit_num_options=3, verbose=True, top_logprobs = 5):
    max_tasks = 10
    query_input = query
    options_begin = make_options(PICK_TARGETS, PLACE_TARGETS)
   
    commands_string = "\n".join(options_begin)
    commands_string = 'BEGIN OPTIONS AVAILABLES:\n' + commands_string + '\nEND OPTIONS.\n'

    scene_description = build_scene_description(found_objects)
    env_description = scene_description
    env_description = 'BEGIN SCENE DESCRIPTION:\n' + env_description + '\nEND SCENE DESCRIPTION.\n'

    call_law_start = "After \\\\'USER INPUT\\\\' there will be a request to satisfy and there may already be actions trying to satisfy it. You must choose from the available options which one you think is next given the previous ones. Return only the option you have chosen, without adding or removing anything else! It is important that the option be chosen from the set of possible options defined within \\\\'BEGIN OPTIONS AVAILABLES\\\\' through \\\\'END OPTIONS\\\\'. DO NOT USE OTHER OPTIONS THAT ARE NOT PRESENT HERE! You have available also a scene description with the objects observed pickable in the scene, this description is defined within \\\\'BEGIN SCENE DESCRIPTION\\\\' through \\\\'END SCENE DESCRIPTION\\\\'. IMPORTANT: Answer \\'done()\\' when you think the request has been satisfied."

    
    gpt3_prompt = gpt3_context + call_law_start
    if use_environment_description:
        gpt3_prompt += "\n" + env_description

    print('GPT3_PROMPT: \n',gpt3_prompt, '\n')

    termination_string = 'done()'

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

        LLM_CACHE={} 
        llm_scores, res = gpt3_scoring(query, gpt3_prompt, commands_string, options_begin, model="gpt-4o-mini", top_logprobs=5, verbose=True)
        normalized_llm_scores = normalize_scores(llm_scores)
        print('NORMALIZED LLM SCORES:', normalized_llm_scores)
        combined_scores = {}
        for option, llm_score in normalized_llm_scores.items():
            print('OPTION: ', option)
            print('LLMSCORE: ', llm_score)
            if option in affordance_scores:
                affordance_score = affordance_scores[option]
                print('AFF SCORE: ', affordance_score)          
                combined_scores[option] = llm_score * affordance_score
            if combined_scores:
                selected_task = max(combined_scores, key=combined_scores.get)
                print("Selecting:", selected_task)

        steps_text.append(selected_task)
        print(num_tasks, "Selecting: ", selected_task)
        query += '\n' + selected_task
        all_llm_scores.append(llm_scores)
        all_affordance_scores.append(affordance_scores)
        all_combined_scores.append(combined_scores)

    print('**** Solution ****')
    print(env_description)
    print('# ' + query_input)
    for i, step in enumerate(steps_text):
        if step == '' or step == termination_string:
            break
        print('Step ' + str(i) + ': ' + step)
        nlp_step = step_to_nlp(step)

    return steps_text
