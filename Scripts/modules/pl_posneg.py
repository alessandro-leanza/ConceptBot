import openai

def ask_gpt_for_top_5_options(options, query, gpt3_context, found_objects, model="gpt-4o-mini"):
    system_message = """
    You are responsible for selecting the next steps from a list of predefined options.
    Your task is to pick exactly 5 of the best options from the AVAILABLE OPTIONS that will help achieve the user's goal.
    The robot can perform only pick and place operation. For example 'robot.pick_and_place(cube, table)' means pick the cube and place it on the table; robot.pick_and_place(tool, user) means give the tool to the user.
    ATTENTION: The option 'done()' should ONLY be selected if all user instructions have been fully completed.
    
    You are only allowed to select options from the provided list, and you must provide reasoning before selecting the options.

    The structure of your response must be:
    Reasoning: <Your reasoning here>
    Answer: <semicolon-separated list of 5 options directly from the available options list>.

    It is important to choose exactly 5 options. Each option must be separated by a semicolon (;).

    Some Examples:
    1-
    USER INPUT:
    Place the red block on the blue block, then place the yellow block on the green bowl.

    --> In this case the first thing to do is to pick the red block and place it on the blue block, so you have to look between the available options which ones are the most similar to this case.

    2-
    USER INPUT:
    Place the red block on the blue block, then place the yellow block on the green bowl.
    robot.pick_and_place(red block, blue block)

    --> In this formulation of the user input the “robot.pick_and_place(red block, blue block)” option has already been executed, so your focus must be on the part of the query that has not yet been executed and therefore on the yellow block and the blue bowl.

    3-
    USER INPUT:
    Place the red block on the blue block, then place the yellow block on the green bowl.
    robot.pick_and_place(red block, blue block)
    robot.pick_and_place(yellow block, green bowl)

    --> In this formulation of the user input it seems that the user's request has finally been fulfilled, so we can think of “done()”

    Your task is to pick exactly 5 of the best options from the AVAILABLE OPTIONS that will help achieve the user's goal on the IMMEDIATE NEXT STEP. Note that there is just one object for each name you find in the options.
    For example: 'robot.pick_and_place(green block, red block)' and 'robot.pick_and_place(green block, blue block)', the 'green block' is the same for both the options. Take this in mind!
    """ 


    #user_message = 'USER INPUT:\n' + query + "\nTHE IMMEDIATE NEXT STEP:" + '\nBEGIN OPTIONS AVAILABLES:\n' + "\n".join(options) + '\nEND OPTIONS.\n'
    user_message = 'USER INPUT:\n' + query + "\nFOUND OBJECTS IN THE SCENE: " + str(found_objects) + "\nTHE IMMEDIATE NEXT STEP:" 
    print("System Message:", system_message)
    print("User Message:", user_message)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=300,
        n=1 
    )

    content = response.choices[0].message.content.strip()
    print(content)

    reasoning = content.split("Reasoning: ")[1].split("\n")[0]
    top_5_options = content.split("Answer:")[1].strip().split("; ")

    
    if len(top_5_options) < 1:
        top_5_options.extend(ask_gpt_for_more_options(options, query, 5 - len(top_5_options), found_objects))

    
    elif len(top_5_options) > 5:
        top_5_options = top_5_options[:5]

    return top_5_options, reasoning


def ask_gpt_for_more_options(options, query, num_missing, found_objects, model="gpt-4o-mini"):
    system_message = f"""
    You need to return {num_missing} more options. You can only choose options from the provided list.
    Provide the options as a semicolon-separated list.
    """

    # user_message = 'USER INPUT:\n' + query + '\nBEGIN OPTIONS AVAILABLES:\n' + "\n".join(options) + '\nEND OPTIONS.\n'
    user_message = 'USER INPUT:\n' + query + "\nFOUND OBJECTS IN THE SCENE: " + str(found_objects) + "\nTHE IMMEDIATE NEXT STEP:" 

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=100,
        n=1
    )

    content = response.choices[0].message.content.strip()
    missing_options = content.split("Answer:")[1].strip().split("; ")

    return missing_options


# def filter_invalid_options(top_5_options, available_options):
    
    # valid_options = [option for option in top_5_options if option in available_options]

    # if len(valid_options) < len(top_5_options):
    #     print(f"Attenzione: Sono state rimosse {len(top_5_options) - len(valid_options)} opzioni non valide.")

    # return valid_options


def positive_prompting(top_5_options, query, model="gpt-4o-mini", num_samples=5):
    system_message = """
    You are a robot responsible for identifying the most relevant actions from a list of 5 available options.
    Your goal is to choose the options that are most likely to help achieve the user's objective in pick and place operation. For example 'robot.pick_and_place(cube, table)' means pick the cube and place it on the table; robot.pick_and_place(tool, user) means give the tool to the user.
    Please try to satisfy the user's request, not to think of additional next steps, but these are not made explicit in the request.
    ATTENTION: If the task is not yet fully completed, 'done()' should NOT be chosen.

    Please consider the task described by the user, and return the options that are most beneficial for completing this task.

    Example 1:
    USER INPUT:
    Place the red block on the blue block.

    AVAILABLE OPTIONS:
    - robot.pick_and_place(red block, blue block)
    - robot.pick_and_place(blue block, table)
    - robot.pick_and_place(red block, table)
    - done()

    Reasoning: The user wants to place the red block on the blue block, so the best action is to pick and place the red block on the blue block.
    Answer: robot.pick_and_place(red block, blue block)

    Example 2:
    USER INPUT:
    Place the red block on the blue block.
    robot.pick_and_place(red block, blue block)

    AVAILABLE OPTIONS:
    - robot.pick_and_place(red block, blue block)
    - robot.pick_and_place(blue block, table)
    - robot.pick_and_place(red block, table)
    - done()

    Reasoning: The user wants to place the red block on the blue block and immediately below it has written “robot.pick_and_place(red block, blue block)” which means it has already performed that action. Then the user's request is to be considered fulfilled. It means that I can terminate the operations with “done()”
    Answer: done()

    The structure of your response must be:
    Reasoning: <Explain why these actions are the most relevant>
    Answer: <semicolon-separated list of the best options from the available list>
    '\nBEGIN OPTIONS AVAILABLES:\n'""" + "\n".join(top_5_options) + """ '\nEND OPTIONS.\n'
    """


    user_message = 'USER INPUT:\n' + query + 'NEXT STEP IS:'

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=150,
        n=num_samples 
    )

    option_frequency = {option: 0 for option in top_5_options}

    for choice in response.choices:
        content = choice.message.content.strip()
        print(content)
        selected_options = content.split("Answer:")[1].strip().split("; ")
        for option in selected_options:
            if option in option_frequency:
                option_frequency[option] += 1

    total_responses = num_samples
    positive_scores = {option: option_frequency[option] / total_responses for option in option_frequency}

    return positive_scores


def negative_prompting(top_5_options, query, model="gpt-4o-mini", num_samples=5):
    system_message = """
    You are a robot responsible for identifying the LEAST relevant actions from a list of 5 available options.
    Your goal is to select the actions that DO NOT contribute to the user's objective for the IMMEDIATE NEXT STEP, or that may even hinder the achievement of the user's goal.
    ATTENTION: 'done()' should be selected if the task is NOT fully completed.

    The structure of your response must be:
    Reasoning: <Explain why these actions are less relevant or counterproductive>
    Answer: <semicolon-separated list of the least useful options from the available list>

    Example
    USER INPUT:
    Place the red block on the blue block.

    AVAILABLE OPTIONS:
    - robot.pick_and_place(red block, blue block)
    - robot.pick_and_place(blue block, table)
    - robot.pick_and_place(red block, table)
    - done()

    Reasoning: The user wants to place the red block on the blue block, so the best action is clear and is to pick and place the red block on the blue block. So the others are the least relevant actions.
    Answer: robot.pick_and_place(blue block, table), robot.pick_and_place(red block, table), done()

    Make sure your answer consists only of the options provided.
    '\nBEGIN OPTIONS AVAILABLES:\n'""" + "\n".join(top_5_options) + """ '\nEND OPTIONS.\n'
    """


    user_message = 'USER INPUT:\n' + query + 'NEXT STEP IS:'

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=150,
        n=num_samples 
    )

    option_frequency = {option: 0 for option in top_5_options}

    for choice in response.choices:
        content = choice.message.content.strip()
        print(content)
        selected_options = content.split("Answer:")[1].strip().split("; ")
        for option in selected_options:
            if option in option_frequency:
                option_frequency[option] += 1

    total_responses = num_samples
    negative_scores = {option: option_frequency[option] / total_responses for option in option_frequency}

    return negative_scores


def combine_prompting_scores(positive_scores, negative_scores):
    combined_scores = {}

    for option in positive_scores:
        combined_scores[option] = positive_scores[option] - negative_scores.get(option, 0)

    return combined_scores 

def run_prompting(gpt3_context, query, options):
   
    top_5_options, reasoning = ask_gpt_for_top_5_options(gpt3_context, options, query)

    # top_5_options = filter_invalid_options(top_5_options, options)


    print("Top 5 Options:")
    print(top_5_options)
    print("Reasoning:")
    print(reasoning)

    positive_scores = positive_prompting(top_5_options, query)
    print("Positive Prompting Scores:")
    print(positive_scores)

    negative_scores = negative_prompting(top_5_options, query)
    print("Negative Prompting Scores:")
    print(negative_scores)

    combined_scores = combine_prompting_scores(positive_scores, negative_scores)
    print("Combined Scoring Results:")
    print(combined_scores)

    return combined_scores, positive_scores, negative_scores

def make_options(PICK_TARGETS, PLACE_TARGETS, pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
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


def normalize_scores(combined_scores):
    normalized_scores = {}
    for option, score in combined_scores.items():
        normalized_score = (score + 1) / 2 
        normalized_scores[option] = normalized_score

    return normalized_scores


def get_rpn_score_for_object(object_name, detected_objects, rpn_scores):
    if object_name in detected_objects:
        index = detected_objects.index(object_name)
        return rpn_scores[index]
    return 0.0 


def combine_llm_and_affordance_scores(llm_scores, affordance_scores):
    combined_scores = {}
    for option in llm_scores:
        llm_score = llm_scores[option]
        affordance_score = affordance_scores.get(option, 1.0)
        combined_score = llm_score * affordance_score
        combined_scores[option] = combined_score

    return combined_scores


def get_bounding_box_for_object(object_name, detected_objects, rescaled_detection_boxes):
    """
    Returns the bounding box associated with an object if found in detected_objects.

    :param object_name: Name of the object for which the bounding box is to be obtained.
    :param detected_objects: List of detected objects.
    :param rescaled_detection_boxes: List of bounding boxes associated with detected objects.
    :return: Bounding box of the object (None if not found).
    """
    if object_name in detected_objects:
        index = detected_objects.index(object_name)
        return rescaled_detection_boxes[index]
    return None 

def calculate_affordance_score_with_properties(llm_options, objects_info):
    affordance_property_scores = {}

    for option in llm_options:
        pick_object = option.split("(")[1].split(",")[0].strip()

        if pick_object in objects_info:
            properties = objects_info[pick_object]
            affordance_score = 1.0

            if properties["fragile"] == "Yes":
                affordance_score -= 0.1

            if properties["dangerous"] == "Yes":
                affordance_score -= 0.1 

            if properties["deformable_material"] == "Yes":
                affordance_score -= 0.1 

            if properties["magnetic_or_conductive"] == "Yes":
                affordance_score -= 0.05 
                
            if properties["hold_liquid"] == "Yes":
                affordance_score -= 0.05 

            affordance_score = max(0, affordance_score)

        else:
            affordance_score = 1.0 

        affordance_property_scores[option] = affordance_score

    return affordance_property_scores


def calculate_affordance_score_with_rpn_and_bbox(llm_options, detected_objects, rpn_scores, rescaled_detection_boxes, max_gripper_size, use_rpn=True, use_bbox=True):
    affordance_rpn_scores = {}
    affordance_bbox_scores = {}

    for option in llm_options:
        pick_object = option.split("(")[1].split(",")[0].strip()
        place_object = option.split(",")[1].replace(")", "").strip()

        if use_rpn:
            pick_rpn_score = get_rpn_score_for_object(pick_object, detected_objects, rpn_scores)
            place_rpn_score = get_rpn_score_for_object(place_object, detected_objects, rpn_scores)
            affordance_rpn_score = (pick_rpn_score + place_rpn_score) / 2
        else:
            affordance_rpn_score = 1.0 

        affordance_rpn_scores[option] = affordance_rpn_score

        if use_bbox:
            bounding_box = get_bounding_box_for_object(pick_object, detected_objects, rescaled_detection_boxes)
            if bounding_box is not None:
                xmin, ymin, xmax, ymax = bounding_box
                width = xmax - xmin
                height = ymax - ymin
                max_side = max(width, height)

                if max_side > max_gripper_size:
                    penalty = (max_side - max_gripper_size) / max_gripper_size
                    affordance_bbox_score = 1 - penalty
                else:
                    affordance_bbox_score = 1.0  
            else:
                affordance_bbox_score = 1.0 
        else:
            affordance_bbox_score = 1.0 

        affordance_bbox_scores[option] = affordance_bbox_score

    return affordance_rpn_scores, affordance_bbox_scores

def calculate_affordance_score_with_all_factors(llm_options, detected_objects, max_gripper_size, object_properties, use_rpn=True, use_bbox=True, use_properties=True):
    affordance_rpn_scores = {}
    affordance_bbox_scores = {}
    affordance_property_scores = {}
    affordance_scores = {}

    property_penalties = {
        "fragile": 0.07,              
        "dangerous": 0.1,           
        "deformable_material": 0.05, 
        "magnetic_or_conductive": 0.05,  
        "hold_liquid": 0.05   
    }

    for option in llm_options:

        if option == "done()":
            affordance_scores[option] = 1.0
            continue

        pick_object = option.split("(")[1].split(",")[0].strip()
        place_object = option.split(",")[1].replace(")", "").strip()
        pick_object = option.split("(")[1].split(",")[0].strip()
        place_object = option.split(",")[1].replace(")", "").strip()

        if use_rpn:
            print('')
            #ViLD 
           #pick_rpn_score = get_rpn_score_for_object(pick_object, detected_objects, rpn_scores)
           #place_rpn_score = get_rpn_score_for_object(place_object, detected_objects, rpn_scores)
           #affordance_rpn_score = (pick_rpn_score + place_rpn_score) / 2
        else:
            affordance_rpn_score = 1.0  

        affordance_rpn_scores[option] = affordance_rpn_score

        if use_bbox:
            print('')
            #ViLD
            # bounding_box = get_bounding_box_for_object(pick_object, detected_objects, rescaled_detection_boxes)
            # if bounding_box is not None:
            #     xmin, ymin, xmax, ymax = bounding_box
            #     width = xmax - xmin
            #     height = ymax - ymin
            #     max_side = max(width, height)

            #     if max_side > max_gripper_size:
            #         penalty = (max_side - max_gripper_size) / max_gripper_size
            #     
            #         affordance_bbox_score = 1.0 - penalty
            #     else:
            #         affordance_bbox_score = 1.0
            # else:
            #     affordance_bbox_score = 1.0  
        else:
            affordance_bbox_score = 1.0 

        affordance_bbox_scores[option] = affordance_bbox_score

        if use_properties:
            object_property = object_properties.get(pick_object, {})
            affordance_property_score = 1.0
            if object_property:
                if object_property.get("fragile", "No") == "Yes":
                    affordance_property_score -= property_penalties["fragile"]
                if object_property.get("dangerous", "No") == "Yes":
                    affordance_property_score -= property_penalties["dangerous"]
                if object_property.get("deformable_material", "No") == "Yes":
                    affordance_property_score -= property_penalties["deformable_material"]
                if object_property.get("magnetic_or_conductive", "No") == "Yes":
                    affordance_property_score -= property_penalties["magnetic_or_conductive"]
                if object_property.get("hold_liquid", "No") == "Yes":
                    affordance_property_score -= property_penalties["hold_liquid"]
            else:
                affordance_property_score = 1.0 
        else:
            affordance_property_score = 1.0

        affordance_property_scores[option] = affordance_property_score

        affordance_scores[option] = affordance_rpn_score * affordance_bbox_score * affordance_property_score

    return affordance_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores


def run_prompting_with_affordance(gpt3_context, query, options, detected_objects, max_gripper_size, object_properties, use_rpn=True, use_bbox=True, use_properties=True):
    
    top_5_options, reasoning = ask_gpt_for_top_5_options(options, query, gpt3_context, detected_objects)
    # top_5_options = filter_invalid_options(top_5_options, options)


    positive_scores = positive_prompting(top_5_options, query)
   
    negative_scores = negative_prompting(top_5_options, query)

    combined_llm_scores = combine_prompting_scores(positive_scores, negative_scores)

    normalized_llm_scores = normalize_scores(combined_llm_scores)

    affordance_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores = calculate_affordance_score_with_all_factors(
        top_5_options, detected_objects, max_gripper_size, object_properties, use_rpn, use_bbox, use_properties)

    final_combined_scores = combine_llm_and_affordance_scores(normalized_llm_scores, affordance_scores)

    print("Final Combined Scoring Results:")
    print(final_combined_scores)

    return final_combined_scores, affordance_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores, normalized_llm_scores


def print_combined_scores_with_details(llm_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores, final_combined_scores):
    print("\nLLM Scores (Normalized):")
    for option, score in llm_scores.items():
        print(f"{option}: {score:.2f}")

    print("\nAffordance Scores (RPN):")
    for option, score in affordance_rpn_scores.items():
        print(f"{option}: {score:.2f}")

    print("\nAffordance Scores (Bounding Box):")
    for option, score in affordance_bbox_scores.items():
        print(f"{option}: {score:.2f}")

    print("\nAffordance Scores (Properties):")
    for option, score in affordance_property_scores.items():
        print(f"{option}: {score:.2f}")

    print("\nFinal Combined Scores:")
    for option, score in final_combined_scores.items():
        print(f"{option}: {score:.2f}")


# ViLD
max_gripper_size = 46.5  #pixel


def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description


def POSNEG(user_query, max_tasks, PICK_TARGETS, PLACE_TARGETS, urp_query, found_objects, rpn_scores, rescaled_detection_boxes, max_gripper_size, objects_info, use_rpn, use_bbox, use_properties):

    policy = []

    termination_string = "done()"

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

    objects = [blue block, green bowl, red block, yellow bowl, green block]
    # stack the blocks.
    robot.pick_and_place(green block, blue block)
    robot.pick_and_place(red block, green block)
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

    use_environment_description = False

    gpt3_context_lines = gpt3_context.split("\n")
    gpt3_context_lines_keep = []
    for gpt3_context_line in gpt3_context_lines:
        if "objects =" in gpt3_context_line and not use_environment_description:
            continue
        gpt3_context_lines_keep.append(gpt3_context_line)

    gpt3_context = "\n".join(gpt3_context_lines_keep)
    gpt3_context = 'BEGIN EXAMPLES:' + gpt3_context + 'END EXAMPLES. \n'

    options_begin = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
    if use_environment_description:
        commands_string = "\n".join(options_begin)
        commands_string = 'BEGIN OPTIONS AVAILABLES:\n' + commands_string + '\nEND OPTIONS.\n'
        print(commands_string)
    else:
        commands_string = ''

    scene_description = build_scene_description(found_objects)
    env_description = 'BEGIN SCENE DESCRIPTION:\n' + scene_description + '\nEND SCENE DESCRIPTION.\n'

    commands_string = commands_string + env_description

    all_llm_scores = []
    all_affordance_scores = []
    all_combined_scores = []
    steps_text = []
    selected_task = ""
    num_tasks = 0

    while not selected_task == termination_string:
        num_tasks += 1
        if num_tasks > max_tasks:
            break

        options_begin = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)

        if selected_task in options_begin:
            options_begin.remove(selected_task)

        final_combined_scores, affordance_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores, normalized_llm_scores = run_prompting_with_affordance(gpt3_context, 
            urp_query, options_begin, found_objects, max_gripper_size, objects_info, use_rpn, use_bbox, use_properties)

        print_combined_scores_with_details(normalized_llm_scores, affordance_rpn_scores, affordance_bbox_scores, affordance_property_scores, final_combined_scores)

        if final_combined_scores:
            selected_task = max(final_combined_scores, key=final_combined_scores.get)
            print("Selecting:", selected_task)

        if selected_task == "done()":
            print("Task completed, stopping execution.")
            break

        steps_text.append(selected_task)
        urp_query += '\n' + selected_task
        print("\n", urp_query, "\n")

        all_llm_scores.append(normalized_llm_scores)
        all_affordance_scores.append(affordance_scores)
        all_combined_scores.append(final_combined_scores)



    print('**** Solution ****')
    print(env_description)
    print('# ' + user_query)
    for i, step in enumerate(steps_text):
        if step == '' or step == termination_string:
            break
        print('Step ' + str(i) + ': ' + step)
        
        policy.append(step)

    return policy
