import json
from datetime import datetime
from functools import partial
import os
import sys
import gc
from datetime import datetime


import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import pipeline
from peft import LoraConfig
from peft import get_peft_model
from qwen_vl_utils import process_vision_info

sys.path.append('../src')

from data_utils import *

## General Settings
top_image_dir = '../data/images'
json_fpath = '../results/task1_convert_validation_annotations/annotation_quiz_all_modified.json'
peft_model_dir = '../results/task2_qwen2_vl2b_train/peft_model_epoch_1'
output_dir = '../results/task2_qwen2_vl2b_label_val_test'

## Setup output directory
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

## Model Setup
print('### Model Setup ###')
print(datetime.now())
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="cuda"
)
model.load_adapter(peft_model_dir)
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
## Setup Validation and Test Datasets
ds_val = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='val', inference_mode=False, img_size=224)
ds_test = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='train', inference_mode=False, img_size=224)
print('number of validation examples: {}'.format(len(ds_val)))
print('number of test examples: {}'.format(len(ds_test)))


## Run Generation

def generate(image, model):
    ## Would really be better to just modify qwenl2_helpers.py but don't have time to rerun
    output_json_format = """
     {
         \"lung\": \"summary of lung related findings, empty if no findings\",
         \"heart\": \"summary of heart related findings, empty if no findings\",
         \"bone\": \"summary of bone related findings, empty if no findings\",
         \"mediastinal\": \"summary of mediastinal related findings, empty if no findings\",
         \"others\": \"summary of any findings not related to lung, heart, bone, or mediastinal\"
     }
     """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Summarize the input image(s) in the following json format {}".format(output_json_format)},
            ],
        },
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    output_text = output_text[0]
    output_text = output_text.strip('<|im_end|>')
    return output_text

# setup output json
output_json = {
    "val": [],
    "test": []
}

def convert_2_json(gen_text):
    try:
        # strip off most common start and end sequences. if we get a valid json we don't really care that much
        # gen_text = re.sub('```json', '', gen_text)
        # gen_text = re.sub('```', '', gen_text)
        gen_text_json = json.loads(gen_text)
        assert type(gen_text_json) is dict
        for tissue_name in ['lung', 'heart', 'bone', 'mediastinal', 'others']:
            # add emtpy entries where tissue is not included
            if not tissue_name in gen_text_json.keys():
                gen_text_json[tissue_name] = ""
    except json.JSONDecodeError as e:
        print(f"Invalid JSON string: {e}")
        print(gen_text)
        assert False
        
    return gen_text_json

def generate_multi_try(image, model, n_tries=100):
    succeeded = False
    for k in range(n_tries):
        print("attempt generation, try {}".format(k+1))
        gen_text = generate(image, model)
        try:
            gen_text_json = convert_2_json(gen_text)
            succeeded = True
            break
        except:
            print("Failed, trying another round of generation")
    if succeeded:
        return gen_text_json
    else:
        # a bit crude and harsh, but if the data isn't even coming out in json format it ain't worth it
        print('text not in proper json format')
        print(gen_text)
        json_text = """
         {
             \"lung\": \"\",
             \"heart\": \"\",
             \"bone\": \"\",
             \"mediastinal\": \"\",
             \"others\": \"\"
         }
         """
        gen_text_json = json.loads(json_text)
        return gen_text_json
print("### Running Generation ###")
print(datetime.now())
for i in range(len(ds_val)):
    print("Generating for validation, {} of {}".format(i+1, len(ds_val)))
    img, _ = ds_val[i]
    id_i = ds_val.data_index[i]['id']
    gen_text_json = generate_multi_try(img, model)
    json_add = {
        "id": id_i,
        "report": gen_text_json
    }
    output_json["val"].append(json_add)
    print("Finished")
    print(datetime.now())
    
for i in range(len(ds_test)):
    print("Generating for test set, {} of {}".format(i+1, len(ds_test)))
    img, _ = ds_test[i]
    id_i = ds_test.data_index[i]['id']
    gen_text_json = generate_multi_try(img, model)
    json_add = {
        "id": id_i,
        "report": gen_text_json
    }
    output_json["test"].append(json_add)
    print("Finished")
    print(datetime.now())
    
print('### Saving Data ###')
print(datetime.now())
outfile_fname = os.path.join(output_dir, 'val_test_gen_reports.json')
with open(outfile_fname, "w") as outfile:
    json.dump(output_json, outfile, indent=4)
    
