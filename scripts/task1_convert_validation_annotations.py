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
    'lung': 'summary',
    'heart': 'summary',
    'bone': 'summary',
    'mediastinal': 'summary',
    'others': 'summary'
}
json_format_str=json.dumps(json_format_dict)

## S

def extract_json_from_response(input_text):
    """
    Function to extract json output from LLM prompt response
    """    
    # strip out ```json and ``` as this is a common output.
    # this is a pretty hacky way of dealing with it but if it works, great.
    input_text = re.sub('```json', '', input_text)
    input_text = re.sub('```', '', input_text)
    # see if it's even json format
    try:
        json_data = json.loads(input_text)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON string: {e}")
        print(input_text)
        return 'invalid json'
    
    try:
        # Ensure that all required keys are present and are of right datatype
        assert type(json_data['lung']) is str
        assert type(json_data['heart']) is str
        assert type(json_data['bone']) is str
        assert type(json_data['mediastinal']) is str
        assert type(json_data['others']) is str
    except:
        return 'invalid keys or values'
    return json_data

system_prompt = """
You are a researcher tasked with summarizing doctor reports (in input).
Report any findings if any for relevant tissues indicated by keys in json output.
If there are no relevant findings for a given tissue, do not report anything.
Do not confuse results present in one tissue as pertaining to another tissue.

### Example 1 ###
report: 
'The cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size. The
lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural
effusion. There are mild degenerative endplate changes in the thoracic spine. There are no acute
bony findings.'
output: 
{{
'lung': 'Lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. Pulmonary vasculature are within normal limits in size.',
'heart': 'Cardiac silhouette within normal limits in size.',
'mediastinal': 'Mediastinal contours within normal limits in size.',
'bone': 'Mild degenerative endplate changes in the thoracic spine. No acute bony findings.',
'others': ''
}}

### Example2 ###
report:
'Bony structures are intact. Cardiac contours are within normal limits. The lungs are clear. Mediastinal contours appear to be within normal limits.'
output:
{{
'lung': 'The lungs are clear',
'heart': 'Cardiac contours are within normal limits',
'mediastinal': 'Mediastinal contours within normal limits in size.',
'bone': 'Bony structures are intact',
'others': ''
}}
### Example 3 ###
report:
'The heart is abnormal structurally. Mediastinal contours are outside normal limits, which is of concern. The XXXX appears intact'
output:
{{
'lung': '',
'heart': 'Heart has abnormal structure.',
'mediastinal': 'Mediastinal contours are outside normal limits.',
'bone': '',
'others': 'The XXXX appears intact'
}}
### Example 4 ###
report:
'The lung is abnormal structurally, with large contusions. Mediastinal contours are outside normal limits, which is of concern. The XXXX appears intact'
output:
{{
'lung': 'Lung has abnormal structure with large contusions',
'heart': '',
'mediastinal': 'Mediastinal contours are outside normal limits.',
'bone': '',
'others': 'The XXXX appears intact'
}}
### Example 5 ###
report:
'The spine and the ribs appear to be malformed. Heart broadly appears normal, suprisingly.'
output:
{{
'lung': '',
'heart': 'Heart appears normal',
'mediastinal': '',
'bone': 'The spine and the ribs appear to be malformed',
'others': ''
}}

Return all output in the following json format:

{}

""".format(json_format_str)
            
## loop through the validation set, modifying entries 
n_items_val = len(data['val'])
max_tries=10
for i in range(n_items_val):
    print('####################################################')
    print("Running llama for validation set item {} of {}".format(str(i+1), str(n_items_val)))
    val_i = data['val'][i]
    # we pop out original response
    original_report_i = val_i.pop('original_report')
    user_prompt = "Summarize following report in parentheses ({}).".format(original_report_i)
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
        print("original report: {}".format(original_report_i))
        try:
            try:
                json_out_i = extract_json_from_response(content_result)
                print("json extraction result: {}".format(json_out_i))
                assert type(json_out_i) is dict
            except:
                if json_out_i is str:
                    if json_out_i == 'invalid json':
                        try:
                            print('attempting appending of curly brace to end.')
                            # try correcting for the most common error, which is forgetting an end bracket.
                            # this is hacky but should deal with majority of errors
                            content_result_mod = content_result + '}'
                            json_out_i = extract_json_from_response(content_result_mod)
                        except:
                            # raise exception
                            raise ValueError('content result not fixed by adding end bracket, submit modified query')
                    else:
                        raise ValueError('content result could not have valid json extracted, reason: {}'.format(json_out_i))
                else:
                    # just set json_out_i to invalid json.
                    json_out_i = 'invalid json'
                    raise ValueError('unknown output from extract_json_from_response. suggest debugging that function.')
                    
            print('succeeded after {} tries'.format(str(j + 1)))
            val_i['report'] = json_out_i
            break
        except:
            if j == max_tries:
                raise ValueError('Reached Max Tries')
                print(content_result)
            user_prompt = "Given the  following report in parentheses ({0}), I received the following incorrect response: '{1}'.  Reason: '{2}'. For the aforementioned report, return a correctly formatted json output that summarizes results in the report, as in the following format: {3}. DO NOT RETURN CODE or instructions. Output must start with {{ and end with }}. return only valid json.".format(original_report_i, content_result, json_out_i, json_format_str)
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

output_dir = os.path.join('../results', 'task1_convert_validation_annotations')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_fname = os.path.join(output_dir, 'annotation_quiz_all_modified.json')
with open(output_fname, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    
    
