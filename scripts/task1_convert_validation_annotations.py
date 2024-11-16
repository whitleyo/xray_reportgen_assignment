import os
import numpy as np
import pandas as pd
import json
import torch
from transformers import pipeline
from huggingface_hub import notebook_login, login
import re
import datetime

## Setup Directories

data_dir = '../data'
image_dir = os.path.join(data_dir, 'images')
token_dir = '../../github_tokens'



## Do Login

token_file = os.path.join(token_dir, 'hugging_face_owhitley_token_nov_2024.txt')
with open(token_file, 'r') as file:
    lines = [line.rstrip('\n') for line in file.readlines()]
    token_use = lines[0]

login(token_use)

## Load json

anno_json_fname = os.path.join(data_dir, 'annotation_quiz_all.json')
with open(anno_json_fname, "r") as file:
    data = json.load(file)

## Setup llama model for converting text to json reports

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

## Setup json format and function to extract json from LLM response

json_format_dict={
    'lung': 'summarize any findings relevant to lung bere, leaving empty string if none',
    'heart': 'summarize any findings relevant to heart here, leaving empty string if none',
    'bone': 'summarize any findings relevant to bone here, leaving empty string if none',
    'mediastinal': 'summarize any findings relevant to mediastinal here, leaving empty string if none',
    'others': 'summarize any other findings here that are not relevant to lung, heart, bone or mediastinal, leaving empty string if none'
}
json_format_str=json.dumps(json_format_dict)


def extract_json_from_response(input_text):
    """
    Function to extract json output from LLM prompt response
    """    
    try:
        # strip out ```json and ``` as this is a common output.
        # this is a pretty hacky way of dealing with it but if it works, great.
        input_text = re.sub('```json', '', input_text)
        input_text = re.sub('```', '', input_text)
        # see if it's even json format
        json_data = json.loads(input_text)
        try:
            # Ensure that all required keys are present and are of right datatype
            assert type(json_data['lung']) is str
            assert type(json_data['heart']) is str
            assert type(json_data['bone']) is str
            assert type(json_data['mediastinal']) is str
            assert type(json_data['others']) is str
        except:
            raise ValueError('invalid json keys or values')
        return json_data
    except json.JSONDecodeError as e:
        print(f"Invalid JSON string: {e}")
        print(input_text)
        raise ValueError('Invalid JSON String')
            

## loop through the validation set, modifying entries 
n_items_val = len(data['val'])
max_tries=10
for i in range(n_items_val):
    print("Running llama for validation set item {} of {}".format(str(i+1), str(n_items_val)))
    val_i = data['val'][i]
    # we pop out original response
    original_report_i = val_i.pop('original_report')
    user_prompt = "Given the following report in parentheses ({}), create a json report with tissue relevant information separated by key as in the following format: {}.".format(original_report_i, json_format_str)
    system_prompt = "You are a chatbot that given a string of text returns a summary report in the requested json format. Do NOT return code or markdown, and strictly output json format as follows:  ({}). Only use valid json.".format(json_format_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512,
    )
    # extract json
    content_result = outputs[0]['generated_text'][2]['content']
    max_tries = 100
    for j in range(max_tries + 1):
        try:
            json_out_i = extract_json_from_response(content_result)
            print('succeeded after {} tries'.format(str(j + 1)))
            val_i['report'] = json_out_i
            break
        except:
            if j == max_tries:
                raise ValueError('Reached Max Tries')
                print(content_result)
            user_prompt = "Given the following report in parentheses ({}), and the following incorrectly formatted json report ({}) return a correctly formatted json report, as in the follwing format: {}. return only valid json.".format(original_report_i, content_result, json_format_str)
            system_prompt = "You are a chatbot that given a string of text returns a summary report in the requested json format. Do NOT return code or markdown, and strictly follow the json format in parentheses ({}). Only use valid json".format(json_format_str)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=512,
            )
            # extract json
            content_result = outputs[0]['generated_text'][2]['content']

            
## Save Output

output_dir = os.path.join(results, 'task1_convert_validation_annotations')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_fname = os.path.join(output_dir, 'annotation_quiz_all_modified.json')
with open(output_fname, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
    
