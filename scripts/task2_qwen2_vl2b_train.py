import json
from datetime import datetime
from functools import partial
import os
import sys
import gc
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import pipeline
from peft import LoraConfig
from torch.utils.data import DataLoader
from peft import get_peft_model
import torch

sys.path.append('../src')

from data_utils import *
from qwenl2_helpers import *

## General Settings
top_image_dir = '../data/images'
json_fpath = '../results/task1_convert_validation_annotations/annotation_quiz_all_modified.json'
output_dir = '../results/task2_qwen2_vl2b_train'
n_epochs=1
batch_size=4
loss_fun = torch.nn.NLLLoss(ignore_index=-100)

## Setup output directory
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

## Model Setup
print('Model Setup')
print(datetime.now())
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

lora_config = LoraConfig(r=8, target_modules = ['q_proj', 'v_proj'])
peft_model = get_peft_model(model, lora_config)
print('Trainable Parameters for PEFT model')
peft_model.print_trainable_parameters()

## Setup Training and Validation Datasets
ds_train = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='train', inference_mode=False, img_size=224)
ds_val = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='val', inference_mode=False, img_size=224)
ds_train.subsample(n_subsample=12)
ds_val.subsample(n_subsample=12)
print('number of training examples: {}'.format(len(ds_train))
print('number of validation examples: {}'format(len(ds_val))

train_loader = DataLoader(
                ds_train,
                batch_size=batch_size,
                collate_fn=partial(collate_fn, processor=processor, tokenizer=tokenizer, device=peft_model.device)
            )
val_loader = DataLoader(
                ds_val,
                batch_size=batch_size,
                collate_fn=partial(collate_fn, processor=processor, tokenizer=tokenizer, device=peft_model.device)
            )
## Setup Optimizer
# use only trainable parameters
trainable_parameters = [p for p in peft_model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_parameters, lr=1e-5)
## Training Loop
print('Begin Training')
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    print('##### Epoch {} of {} #####'.format(epoch + 1, n_epochs))
    print(datetime.now())
    ## Train
    print('## Train ##')
    print(datetime.now())
    steps = 0
    total_train_loss = 0
    total_train_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch
        outputs = peft_model.forward(**inputs)
        loss = loss_fun(torch.transponse(outputs, 1, 2), labels)
        steps += 1
        mean_train_batch_loss = loss.item()/inputs.shape[0]
        total_train_loss += loss.item()
        total_train_examples += inputs.shape[0]
        train_losses.append(mean_train_loss)
        print("Avg batch loss at step {}: {}".format(steps, mean_train_batch_loss))
        loss.backward()
        optimizer.step()
        gc.collect()
    mean_train_loss = total_train_loss/total_train_examples
    print('mean training loss: {}'.format(mean_train_loss))
    train_losses.append(mean_train_loss)
    ## Validation
    print('## Validation## ')
    steps = 0
    total_val_examples = 0
    total_val_loss = 0
    print(datetime.now())
    for batch in val_loader:
        optimizer.zero_grad()
        steps += 1
        print('Step {} of validation'.format(steps))
        inputs, labels = batch
        outputs = peft_model.forward(**inputs)
        loss = loss_fun(torch.transponse(outputs, 1, 2), labels)
        total_val_loss += loss.item()
        total_val_examples = inputs.shape[0]
        gc.collect()
    mean_val_loss = total_val_loss/total_val_examples
    val_losses.append(mean_val_loss)
    # we print both training and validation loss since we might have to scroll a while for the training loss.
    print('## Finished Training + Validation for Epoch {} of {} ##'.format(epoch + 1, n_epochs))
    print(datetime.now())
    print('mean training loss: {}'.format(mean_train_loss))
    print('mean validation loss: {}'.format(mean_train_loss))


## Save Final Model
peft_model.save_pretrained(output_dir)
## Save Statistics
train_loss_output_file = os.path.join(output_dir, 'train_losses.csv')
np.savetxt(x=np.array(train_losses), fname=train_loss_output_file)
val_loss_output_file = os.path.join(output_dir, 'val_losses.csv')
np.savetxt(x=np.array(val_losses), fname=val_loss_output_file)
