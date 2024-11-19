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


sys.path.append('../src')

from data_utils import *
from qwenl2_helpers import *

## General Settings
top_image_dir = '../data/images'
json_fpath = '../results/task1_convert_validation_annotations/annotation_quiz_all_modified.json'
output_dir = '../results/task2_qwen2_vl2b_train'
n_epochs=1
batch_size=4
loss_fun = torch.nn.CrossEntropyLoss(ignore_index=-100)

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

lora_config = LoraConfig(r=6, target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
peft_model = get_peft_model(model, lora_config)
print('Trainable Parameters for PEFT model')
peft_model.print_trainable_parameters()

## Setup Training and Validation Datasets
ds_train = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='train', inference_mode=False, img_size=224)
ds_val = XRayImageDataset(top_image_dir=top_image_dir, json_fpath=json_fpath, split='val', inference_mode=False, img_size=224)
ds_train.subsample(n_subsample=1000)
# ds_val.subsample(n_subsample=64)
print('number of training examples: {}'.format(len(ds_train)))
print('number of validation examples: {}'.format(len(ds_val)))

train_loader = DataLoader(
                ds_train,
                batch_size=batch_size,
                collate_fn=partial(collate_fn, processor=processor, tokenizer=tokenizer, device=peft_model.device),
                shuffle=True
            )
val_loader = DataLoader(
                ds_val,
                batch_size=batch_size,
                collate_fn=partial(collate_fn, processor=processor, tokenizer=tokenizer, device=peft_model.device),
                shuffle=True
            )
## Setup Optimizer
# use only trainable parameters
trainable_parameters = [p for p in peft_model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_parameters, lr=1e-3)
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
        loss = loss_fun(torch.transpose(outputs.logits, 1, 2), labels)
        steps += 1
        mean_train_batch_loss = loss.item()/labels.shape[0]
        total_train_loss += loss.item()
        total_train_examples += labels.shape[0]
        print("Avg batch loss at step {}: {}".format(steps, mean_train_batch_loss))
        print(datetime.now())
        loss.backward()
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
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
        loss = loss_fun(torch.transpose(outputs.logits, 1, 2), labels)
        total_val_loss += loss.item()
        total_val_examples += labels.shape[0]
        print('step complete')
        print(datetime.now())
        gc.collect()
        torch.cuda.empty_cache()
    mean_val_loss = total_val_loss/total_val_examples
    val_losses.append(mean_val_loss)
    # we print both training and validation loss since we might have to scroll a while for the training loss.
    print('## Finished Training + Validation for Epoch {} of {} ##'.format(epoch + 1, n_epochs))
    print(datetime.now())
    print('mean training loss: {}'.format(mean_train_loss))
    print('mean validation loss: {}'.format(mean_val_loss))
    del mean_val_loss
    del mean_train_loss
    ## Save Model for Epoch, as it turns out that the model params are in the MB for what ends up getting saved.
    pretrain_dir = os.path.join(output_dir, 'peft_model_epoch_{}'.format(epoch + 1))
    if not os.path.exists(pretrain_dir):
        os.mkdir(pretrain_dir)
    peft_model.save_pretrained(pretrain_dir)

## Save Statistics
train_loss_output_file = os.path.join(output_dir, 'train_losses.csv')
np.savetxt(X=np.array(train_losses), fname=train_loss_output_file)
val_loss_output_file = os.path.join(output_dir, 'val_losses.csv')
np.savetxt(X=np.array(val_losses), fname=val_loss_output_file)
