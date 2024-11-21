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
from qwenl2_helpers import *

## General Settings
top_image_dir = '../data/images'
json_fpath = '../results/task1_convert_validation_annotations/annotation_quiz_all_modified.json'
peft_model_dir = '../results/task2_qwen2_vl2b_train/peft_model_epoch_2'
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



# setup output json
output_json = {
    "val": [],
    "test": []
}

print("### Running Generation ###")
print(datetime.now())
for i in range(len(ds_val)):
    print("Generating for validation, {} of {}".format(i+1, len(ds_val)))
    img, _ = ds_val[i]
    id_i = ds_val.data_index[i]['id']
    gen_text_json = generate_multi_try(img, model, processor)
    print("returned json")
    print(json.dumps(gen_text_json))
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
    gen_text_json = generate_multi_try(img, model, processor)
    print("returned json")
    print(json.dumps(gen_text_json))
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
    
