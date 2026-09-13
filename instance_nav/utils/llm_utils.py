# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
import openai
import httpx
import re
from typing import List, Tuple
from instance_nav.utils.color_dict import iscc_nbs_colors
def parse_object_goal_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    question = f"""
    I: go to the kitchen and then go to the toilet. A: kitchen, toilet
    I: go to the chair and then go to another chair. A: chair, chair
    I: navigate to the green sofa and turn right and find several chairs, finally go to the painting. A: green sofa, chairs, painting
    I: approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen. A: window, television, oven, kitchen
    I: walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed. A: plant, table, bedroom, bed
    I: go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room. A: stairs, room, book shelf, table, next room
    I: Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances. A: table, cushion, column, appliances.
    I: Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table. A: chair, table
    I: {language_instr}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"].strip()
    print("landmarks: ", result)
    return [x.strip() for x in result.split(",")]

def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai
    # openai.base_url = "https://api.gptsapi.net/v1"
    # print("openai base_url: ", openai.base_url)
    openai_key = os.environ["DASHSCOPE_API_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', http_client=httpx.Client(trust_env=False))
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": "go to the kitchen and then go to the toilet",
            },
            {
                "role": "assistant",
                "content": "kitchen, toilet"
            },
            {
                "role": "user",
                "content": "go to the chair and then go to another chair"
            },
            {
                "role": "assistant",
                "content": "chair, chair"
            },
            {
                "role": "user",
                "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
            },
            {
                "role": "assistant",
                "content": "green sofa, chairs, painting"
            },
            {
                "role": "user",
                "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
            },
            {
                "role": "assistant",
                "content": "window, television, oven, kitchen"
            },
            {
                "role": "user",
                "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
            },
            {
                "role": "assistant",
                "content": "plant, table, bedroom, bed"
            },
            {
                "role": "user",
                "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
            },
            {
                "role": "assistant",
                "content": "stairs, room, book shelf, table, next room"
            },
            {
                "role": "user",
                "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"
            },
            {
                "role": "assistant",
                "content": "table, cushion, column, appliances"
            },
            {
                "role": "user",
                "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
            },
            {
                "role": "assistant",
                "content": "chair, table"
            },
            {
                "role": "user",
                "content": language_instr
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    print(f"GPT analyse result: {text}")
    return [x.strip() for x in text.split(",")]

def parse_color_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks with colors, returning two lists:
    - List of object names
    - List of lists of RGB values (each object can have one or two colors)

    The LLM converts ISCC-NBS color names to RGB values based on ISCC-NBS rules.

    Example input: "First approach the grayish yellowish-brown stairs, then find a nearby light bluish-gray sofa and go there, next come to a very pale purplish-blue picture before finally navigate to a olive-black sink."
    Example output:
    (
        ["stairs", "sofa", "picture", "sink"],
        [[[150, 130, 98]], [[176, 191, 195]], [[194, 203, 222]], [[86, 86, 45]]]
    )
    """
    # Initialize OpenAI client
    openai_key = os.environ["DASHSCOPE_API_KEY"]
    client = openai.OpenAI(
        api_key=openai_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        http_client=httpx.Client(trust_env=False)
    )

    # Updated prompt instructing LLM to return objects with RGB values for ISCC-NBS colors
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in ISCC-NBS color nomenclature. For each object in the instruction, extract the object name and convert any ISCC-NBS color names to their corresponding RGB values ([R,G,B]). Return the result in the format: 'object:[R,G,B]' for one color, or 'object:[R1,G1,B1],[R2,G2,B2]' for two colors. If no color is specified, use 'object:none'. Separate multiple objects with commas. Ensure RGB values are integers between 0 and 255."
            },
            {
                "role": "user",
                "content": "go to the kitchen and then go to the toilet",
            },
            {
                "role": "assistant",
                "content": "kitchen:none, toilet:none"
            },
            {
                "role": "user",
                "content": "go to the green sofa and then go to the red chair",
            },
            {
                "role": "assistant",
                "content": "sofa:[0,128,0], chair:[255,0,0]"
            },
            {
                "role": "user",
                "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting",
            },
            {
                "role": "assistant",
                "content": "sofa:[0,128,0], chairs:none, painting:none"
            },
            {
                "role": "user",
                "content": "approach the blue window in front, turn right and go to the black television, and finally go by the oven in the kitchen",
            },
            {
                "role": "assistant",
                "content": "window:[0,0,255], television:[0,0,0], oven:none, kitchen:none"
            },
            {
                "role": "user",
                "content": "First approach the grayish yellowish-brown stairs, then find a nearby light bluish-gray sofa and go there, next come to a very pale purplish-blue picture before finally navigate to a olive-black sink.",
            },
            {
                "role": "assistant",
                "content": "stairs:[150,130,98], sofa:[176,191,195], picture:[194,203,222], sink:[86,86,45]"
            },
            {
                "role": "user",
                "content": language_instr
            }
        ],
        max_tokens=300,
    )

    # Parse LLM response
    text = response.choices[0].message.content
    print(f"GPT analyse result: {text}")

    # Initialize output lists
    objects = []
    colors_rgb = []

    # Regex to match 'object:[R,G,B]' or 'object:[R1,G1,B1],[R2,G2,B2]' or 'object:none'
    pattern = re.compile(r'(\w+?)(?::\[(\d+),(\d+),(\d+)\](?:,\[(\d+),(\d+),(\d+)\])?|:(none))')

    # Find all matches in the response
    matches = pattern.findall(text)
    print(f"Parsed matches: {matches}")  # Debug print to verify parsing

    for match in matches:
        obj = match[0]
        objects.append(obj)

        if match[7] == 'none':  # Handle 'none' case
            colors_rgb.append([])
        else:
            # Extract RGB values (one or two colors)
            rgb_list = []
            # First color
            rgb_list.append([int(match[1]), int(match[2]), int(match[3])])
            # Second color, if present
            if match[4] and match[5] and match[6]:
                rgb_list.append([int(match[4]), int(match[5]), int(match[6])])
            colors_rgb.append(rgb_list)

    return objects, colors_rgb

def parse_color_object_goal_instruction_v2(objects_info: List[dict], classes_list: List[str]) -> Tuple[List[str], List[List[List[int]]]]:
    """
    Parse objects_info into a series of landmarks with colors, returning two lists:
    - List of object names that exist in classes_list (or their most relevant matches)
    - List of lists of RGB values for each object (derived from color_name using LLM)

    Uses LLM to verify if object names are in classes_list or find the most relevant category,
    and to convert ISCC-NBS color names to RGB values.

    Example input: objects_info = [
        {"name": "cushion", "color_name": "bluish-gray cushion"},
        {"name": "chair", "color_name": "light bluish-gray chair"},
        {"name": "counter", "color_name": "dark grayish and olive-green counter"},
        {"name": "picture", "color_name": "very pale purplish-blue picture"}
    ]
    Example output:
    (
        ["cushion", "chair", "counter", "picture"],
        [[[135, 140, 155]], [[176, 191, 195]], [[48, 46, 42], [109, 113, 115]], [[194, 203, 222]]]
    )
    """
        # Initialize OpenAI client
    openai_key = os.environ["DASHSCOPE_API_KEY"]
    client = openai.OpenAI(
        api_key=openai_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        http_client=httpx.Client(trust_env=False)
    )

    # Initialize output lists
    object_categories = []
    colors_rgb = []

    # Prepare classes list for LLM query
    classes_list_str = ",".join(classes_list)

    for obj_info in objects_info:
        obj_name = obj_info["name"]
        color_name = obj_info["color_name"]

        # Use LLM to verify object name and convert color_name to RGB
        response = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in ISCC-NBS color nomenclature and object categorization. For the given object and color description, extract the object name and convert any ISCC-NBS color names to their corresponding RGB values ([R,G,B]). Return the result in the format: 'object:[R,G,B]' for one color, or 'object:[R1,G1,B1],[R2,G2,B2]' for two colors. If the object is not in the provided list, return the most relevant category from the list. Ensure RGB values are integers between 0 and 255."
                },
                {
                    "role": "user",
                    "content": "What is television most relevant to among tv_monitor,plant,chair, and convert dark gray to RGB"
                },
                {
                    "role": "assistant",
                    "content": "tv_monitor:[64,64,64]"
                },
                {
                    "role": "user",
                    "content": "What is drawer most relevant to among tv_monitor,chest_of_drawers,chair, and convert light blue to RGB"
                },
                {
                    "role": "assistant",
                    "content": "chest_of_drawers:[135,206,235]"
                },
                {
                    "role": "user",
                    "content": "What is sofa most relevant to among couch,bed,chair, and convert green to RGB"
                },
                {
                    "role": "assistant",
                    "content": "couch:[0,128,0]"
                },
                {
                    "role": "user",
                    "content": "What is painting most relevant to among picture,window,door, and convert very pale purplish-blue to RGB"
                },
                {
                    "role": "assistant",
                    "content": "picture:[194,203,222]"
                },
                {
                    "role": "user",
                    "content": f"What is {obj_name} most relevant to among {classes_list_str}, and convert {color_name} to RGB"
                }
            ],
            max_tokens=300,
            extra_body={"enable_thinking": False}
        )

        # Parse LLM response
        text = response.choices[0].message.content.strip()
        print(f"LLM response for {obj_name}: {text}")

        # Regex to match 'object:[R,G,B]' or 'object:[R1,G1,B1],[R2,G2,B2]'
        pattern = re.compile(r'(\w+):(?:\[(\d+),(\d+),(\d+)\](?:,\[(\d+),(\d+),(\d+)\])?)$')
        match = pattern.match(text)

        if match:
            matched_category = match.group(1)
            rgb_list = []

            # Extract first color
            try:
                rgb_list.append([int(match.group(2)), int(match.group(3)), int(match.group(4))])
            except (TypeError, ValueError) as e:
                print(f"Error: Invalid RGB values in LLM response for {obj_name}: {text}. Skipping.")
                continue

            # Extract second color if present
            if match.group(5) and match.group(6) and match.group(7):
                try:
                    rgb_list.append([int(match.group(5)), int(match.group(6)), int(match.group(7))])
                except (TypeError, ValueError) as e:
                    print(f"Error: Invalid second RGB values in LLM response for {obj_name}: {text}. Skipping.")
                    continue

            if matched_category in classes_list:
                object_categories.append(matched_category)
                colors_rgb.append(rgb_list)
            else:
                print(f"Warning: LLM returned {matched_category} which is not in classes_list for {obj_name}. Skipping.")
        else:
            print(f"Error: Invalid LLM response format for {obj_name}: {text}. Expected format 'object:[R,G,B]' or 'object:[R1,G1,B1],[R2,G2,B2]'. Skipping.")

    return object_categories, colors_rgb

def parse_color_object_goal_instruction_v2_1(objects_info: List[dict], classes_list: List[str]) -> Tuple[List[str], List[List[List[int]]]]:
    # Initialize OpenAI client
    openai_key = os.environ["DASHSCOPE_API_KEY"]
    client = openai.OpenAI(
        api_key=openai_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        http_client=httpx.Client(trust_env=False)
    )

    # Initialize output lists
    object_categories = []
    colors_text = []

    # Prepare classes list for LLM query
    classes_list_str = ",".join(classes_list)

    for obj_info in objects_info:
        obj_name = obj_info["name"]
        color_name = obj_info["color_name"]
        # Note: absolute_floor is no longer used

        # Use LLM to verify object name and extract ISCC-NBS color names
        response = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in ISCC-NBS color nomenclature and object categorization. For the given object and color description, extract the object name and identify any ISCC-NBS color names present in the description. Return the result in the format: 'object:color1' for one color, or 'object:color1,color2' for two colors. If multiple colors are present (e.g., 'dark grayish and olive-green'), return them as a comma-separated list. If the object is not in the provided list, return the most relevant category from the list. Ensure color names are valid ISCC-NBS terms."
                },
                {
                    "role": "user",
                    "content": "What is television most relevant to among tv_monitor,plant,chair, and identify ISCC-NBS color in dark-gray"
                },
                {
                    "role": "assistant",
                    "content": "tv_monitor:dark-gray"
                },
                {
                    "role": "user",
                    "content": "What is drawer most relevant to among tv_monitor,chest_of_drawers,chair, and identify ISCC-NBS color in light-blue"
                },
                {
                    "role": "assistant",
                    "content": "chest_of_drawers:light-blue"
                },
                {
                    "role": "user",
                    "content": "What is cushion most relevant to among counter,table,chair,cushion on the sofa, and identify ISCC-NBS colors in dark grayish olive-green cushion"
                },
                {
                    "role": "assistant",
                    "content": "cushion on the sofa:dark grayish olive-green"
                },
                {
                    "role": "user",
                    "content": "What is sink most relevant to among sink on the counter,counter,table,chair, and identify ISCC-NBS colors in brownish-black and grayish-green sink"
                },
                {
                    "role": "assistant",
                    "content": "sink on the counter:brownish-black,grayish-green"
                },
                {
                    "role": "user",
                    "content": "What is sofa most relevant to among shelving,counter,table,chair,sofa,door,window , and identify ISCC-NBS colors very pale-blue and pale purplish-blue sofa"
                },
                {
                    "role": "assistant",
                    "content": "sofa:very pale-blue,pale purplish-blue"
                },
                {
                    "role": "user",
                    "content": "What is shelving most relevant to among shelving,counter,table,chair, and identify ISCC-NBS colors dark grayish reddish-brown shelving"
                },
                {
                    "role": "assistant",
                    "content": "shelving:dark grayish reddish-brown"
                },
                {
                    "role": "user",
                    "content": f"What is {obj_name} most relevant to among {classes_list_str}, and identify ISCC-NBS color in {color_name}"
                }
            ],
            max_tokens=300,
            extra_body={"enable_thinking": False}
        )

        # Parse LLM response
        text = response.choices[0].message.content.strip()
        print(f"LLM response for {obj_name}: {text}")

        # Regex to match 'object:color1' or 'object:color1,color2'
        pattern = re.compile(r'^([\w\s]+?):([a-zA-Z\s\-]+)(?:,([a-zA-Z\s\-]+))?$')
        match = pattern.match(text)

        if match:
            matched_category = match.group(1).strip()
            color_list = []

            # Extract first color
            color1 = match.group(2).strip()
            if color1 in iscc_nbs_colors:
                color_list.append(list(iscc_nbs_colors[color1]))
            else:
                print(f"Warning: Color '{color1}' for {obj_name} not found in iscc_nbs_colors. Skipping object.")
                continue

            # Extract second color if present
            if match.group(3):
                color2 = match.group(3).strip()
                if color2 in iscc_nbs_colors:
                    color_list.append(list(iscc_nbs_colors[color2]))
                else:
                    print(f"Warning: Color '{color2}' for {obj_name} not found in iscc_nbs_colors. Using only first color.")

            # Only keep valid categories
            if matched_category in classes_list:
                object_categories.append(matched_category)
                colors_text.append(color_list)
            else:
                print(f"Warning: LLM returned {matched_category} which is not in classes_list for {obj_name}. Skipping.")
        else:
            print(f"Error: Invalid LLM response format for {obj_name}: {text}. Expected format 'object:color1' or 'object:color1,color2'. Skipping.")

    return object_categories, colors_text


def parse_color_object_goal_instruction_v3(objects_info: List[dict], classes_list: List[str]) -> Tuple[List[str], List[List[List[int]]]]:
    # Initialize OpenAI client
    openai_key = os.environ["DASHSCOPE_API_KEY"]
    client = openai.OpenAI(
        api_key=openai_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        http_client=httpx.Client(trust_env=False)
    )

    # Initialize output lists
    object_categories = []
    colors_text = []
    absolute_floors = []
    # Prepare classes list for LLM query
    classes_list_str = ",".join(classes_list)

    for obj_info in objects_info:
        obj_name = obj_info["name"]
        color_name = obj_info["color_name"]
        absolute_floor = int(obj_info["absolute_floor"][1:])
        # Use LLM to verify object name and extract ISCC-NBS color names
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in ISCC-NBS color nomenclature and object categorization. For the given object and color description, extract the object name and identify any ISCC-NBS color names present in the description. Return the result in the format: 'object:color1' for one color, or 'object:color1,color2' for two colors. If multiple colors are present (e.g., 'dark grayish and olive-green'), return them as a comma-separated list. If the object is not in the provided list, return the most relevant category from the list. Ensure color names are valid ISCC-NBS terms."
                },
                {
                    "role": "user",
                    "content": "What is television most relevant to among tv_monitor,plant,chair, and identify ISCC-NBS color in dark-gray"
                },
                {
                    "role": "assistant",
                    "content": "tv_monitor:dark-gray"
                },
                {
                    "role": "user",
                    "content": "What is drawer most relevant to among tv_monitor,chest_of_drawers,chair, and identify ISCC-NBS color in light-blue"
                },
                {
                    "role": "assistant",
                    "content": "chest_of_drawers:light-blue"
                },
                {
                    "role": "user",
                    "content": "What is cushion most relevant to among counter,table,chair,cushion on the sofa, and identify ISCC-NBS colors in dark grayish olive-green cushion"
                },
                {
                    "role": "assistant",
                    "content": "cushion on the sofa:dark grayish olive-green"
                },
                {
                    "role": "user",
                    "content": "What is sink most relevant to among sink on the counter,counter,table,chair, and identify ISCC-NBS colors in brownish-black and grayish-green sink"
                },
                {
                    "role": "assistant",
                    "content": "sink on the counter:brownish-black,grayish-green"
                },
                {
                    "role": "user",
                    "content": "What is shelving most relevant to among shelving,counter,table,chair, and identify ISCC-NBS colors dark grayish reddish-brown shelving"
                },
                {
                    "role": "assistant",
                    "content": "shelving:dark grayish reddish-brown"
                },
                {
                    "role": "user",
                    "content": f"What is {obj_name} most relevant to among {classes_list_str}, and identify ISCC-NBS color in {color_name}"
                }
            ],
            max_tokens=300,
            extra_body={"enable_thinking": False}
        )

        # Parse LLM response
        text = response.choices[0].message.content.strip()
        print(f"LLM response for {obj_name}: {text}")

        # Regex to match 'object:color1' or 'object:color1,color2'
        pattern = re.compile(r'^([\w\s]+?):([a-zA-Z\s\-]+)(?:,([a-zA-Z\s\-]+))?$')
        match = pattern.match(text)

        if match:
            matched_category = match.group(1).strip()
            color_list = []

            # Extract first color
            color1 = match.group(2).strip()
            if color1 in iscc_nbs_colors:
                color_list.append(list(iscc_nbs_colors[color1]))
            else:
                print(f"Warning: Color '{color1}' for {obj_name} not found in iscc_nbs_colors. Skipping object.")
                continue

            # Extract second color if present
            if match.group(3):
                color2 = match.group(3).strip()
                if color2 in iscc_nbs_colors:
                    color_list.append(list(iscc_nbs_colors[color2]))  # ✅ 转换成RGB
                else:
                    print(f"Warning: Color '{color2}' for {obj_name} not found in iscc_nbs_colors. Using only first color.")

            # Only keep valid categories
            if matched_category in classes_list:
                object_categories.append(matched_category)
                colors_text.append(color_list)
                absolute_floors.append(absolute_floor)
            else:
                print(f"Warning: LLM returned {matched_category} which is not in classes_list for {obj_name}. Skipping.")
        else:
            print(f"Error: Invalid LLM response format for {obj_name}: {text}. Expected format 'object:color1' or 'object:color1,color2'. Skipping.")

    return object_categories, colors_text, absolute_floors


def parse_spatial_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        question = f"""
# move a bit to the right of the refrigerator.
robot.move_to_right('refrigerator')
###
# move in between the couch and bookshelf.
robot.move_in_between('couch', 'bookshelf')
###
# move in the middle of the stairs and oven.
robot.move_in_between('stairs', 'oven')
###
# face the toilet.
robot.face('toilet')
###
# move to the south side of the table
robot.move_south('table')
###
# move to the west of the chair
robot.move_west('chair')
###
# move to the east of the chair
robot.move_east('chair')
###
# turn left 60 degrees
robot.turn(-60)
###
# turn right 20 degrees
robot.turn(20)
###
# find any chairs in the environment
robot.move_to_object('chair')
###
# come to the table
robot.move_to_object('table')
###
# with the television on your left
robot.with_object_on_left('television')
###
# with the toilet to your right, turn left 30 degrees
robot.with_object_on_right('toilet')
robot.turn(-30)
###
# with the toilet to your left, turn right 30 degrees
robot.with_object_on_left('toilet')
robot.turn(30)
###
# with the television behind you
robot.face('television')
robot.turn(180)
###
# go to the south of the table in front of you, with the table on your left, turn right 45 degrees
robot.move_south('table')
robot.with_object_on_left('table')
robot.turn(45)
###
# move forward for 3 meters
robot.move_forward(3)
###
# move right 2 meters
robot.turn(90)
robot.move_forward(2)
###
# move left 3 meters
robot.turn(-90)
robot.move_forward(3)
###
# move back and forth between the chair and the table 3 times
pos1 = robot.get_pos('chair')
pos2 = robot.get_pos('table')
for i in range(3):
    robot.move_to(pos1)
    robot.move_to(pos2)
###
# go to the television
robot.move_to_object('television')
###
# monitor the chair from 4 view points
contour = robot.get_contour('chair')
np.random.shuffle(contour)
for pos in contour[:4]:
    robot.move_to(pos)
    robot.face('chair')
###
# navigate to 3 meters right of the table
robot.move_to_right('table')
robot.face('table')
robot.turn(180)
robot.move_forward(3)
###
# turn west
robot.turn_absolute(-90)
###
# turn east
robot.turn_absolute(90)
###
# turn south
robot.turn_absolute(180)
###
# turn north
robot.turn_absolute(0)
###
# turn east and then turn left 90 degrees
robot.turn_absolute(90)
robot.turn(-90)
###
# move 3 meters north of the chair
robot.move_north('chair')
robot.turn_absolute(0)
robot.move_forward(3)
###
# move 1 meters south of the chair
robot.move_south('chair')
robot.turn_absolute(180)
robot.move_forward(3)
###
# move 3 meters west
robot.turn_absolute(-90)
robot.move_forward(3)
# {lang}
    """
        print("lang: ", lang)
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=question,
            max_tokens=400,
            temperature=0.0,
            stop="###",
        )
        result = response["choices"][0]["text"].strip()
        if result:
            results += result + "\n"
        print(result)
    print("generated code")
    print(results)
    return results

def parse_spatial_instruction(language_instr):
    import openai

    openai_key = os.environ["DASHSCOPE_API_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        client = openai.OpenAI(api_key=openai_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', http_client=httpx.Client(trust_env=False))
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": "move a bit to the right of the refrigerator"},
                {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
                {"role": "user", "content": "move in between the couch and bookshelf"},
                {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
                {"role": "user", "content": "move in the middle of the stairs and oven"},
                {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
                {"role": "user", "content": "face the toilet"},
                {"role": "assistant", "content": "robot.face('toilet')"},
                {"role": "user", "content": "move to the south side of the table"},
                {"role": "assistant", "content": "robot.move_south('table')"},
                {"role": "user", "content": "move to the west of the chair"},
                {"role": "assistant", "content": "robot.move_west('chair')"},
                {"role": "user", "content": "move to the east of the chair"},
                {"role": "assistant", "content": "robot.move_east('chair')"},
                {"role": "user", "content": "turn left 60 degrees"},
                {"role": "assistant", "content": "robot.turn(-60)"},
                {"role": "user", "content": "turn right 20 degrees"},
                {"role": "assistant", "content": "robot.turn(20)"},
                {"role": "user", "content": "find any chairs in the environment"},
                {"role": "assistant", "content": "robot.move_to_object('chair')"},
                {"role": "user", "content": "come to the table"},
                {"role": "assistant", "content": "robot.move_to_object('table')"},
                {"role": "user", "content": "with the television on your left"},
                {"role": "assistant", "content": "robot.with_object_on_left('television')"},
                {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
                {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
                {"role": "user", "content": "with the television behind you"},
                {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
                {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
                {"role": "assistant", "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
                {"role": "user", "content": "move forward for 3 meters"},
                {"role": "assistant", "content": "robot.move_forward(3)"},
                {"role": "user", "content": "move right 2 meters"},
                {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
                {"role": "user", "content": "move left 3 meters"},
                {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
                {"role": "assistant", "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
                {"role": "user", "content": "go to the television"},
                {"role": "assistant", "content": "robot.move_to_object('television')"},
                {"role": "user", "content": "monitor the chair from 4 view points"},
                {"role": "assistant", "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
                {"role": "user", "content": "navigate to 3 meters right of the table"},
                {"role": "assistant", "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
                {"role": "user", "content": "turn west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)"},
                {"role": "user", "content": "turn east"},
                {"role": "assistant", "content": "robot.turn_absolute(90)"},
                {"role": "user", "content": "turn south"},
                {"role": "assistant", "content": "robot.turn_absolute(180)"},
                {"role": "user", "content": "turn north"},
                {"role": "assistant", "content": "robot.turn_absolute(0)"},
                {"role": "user", "content": "turn east and then turn left 90 degrees"},
                {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
                {"role": "user", "content": "move 3 meters north of the chair"},
                {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 1 meters south of the chair"},
                {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 3 meters west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": lang},
            ],
            max_tokens=300,
        )
        text = response.choices[0].message.content
        if text:
            results += text + "\n"
    return text
if __name__ == '__main__':
    #text = parse_spatial_instruction("go to the sofa, turn right and move in between the table and the chair, and then move back and forth to the keyboard and the screen twice")
    text, color = parse_color_object_goal_instruction("Turn around and find a grayish-blue and dark bluish-gray chair, go to a very light-blue table in the front and then walk to the moderate-brown counter, finally move to the purplish-gray sofa.")
    print(text)
    print(color)
