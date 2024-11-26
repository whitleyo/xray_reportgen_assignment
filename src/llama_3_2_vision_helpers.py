from functools import partial
import os
import json
from qwen_vl_utils import process_vision_info
import torch
from input_prep import *

def get_start_end_idx(l, tokenizer):
    """
    Get Start and End Indices for Assistant (Response) Portion of Chat Text (Tokenized)
    Args:
        l: list token sequences
        tokenizer: tokenizer
    returns: start and end indices for each token sequence, in list format [(start, end), (start, end), ...]
    """
    start_indexes = []
    end_indexes = []
    start_tokens = tokenizer.encode("<|start_header_id|>assistant")
    # llama adds a begin text id at front
    assert(tokenizer.decode(start_tokens[0]) == '<|begin_of_text|>')
    start_tokens = start_tokens[1:len(start_tokens)]
    assert(tokenizer.decode(start_tokens) == '<|start_header_id|>assistant')
    assert len(start_tokens) == 2
    end_token = tokenizer.encode("<|end_header_id|>")
    assert(tokenizer.decode(end_token[0]) == '<|begin_of_text|>')
    end_token = end_token[1:len(end_token)]
    assert(tokenizer.decode(end_token) == '<|end_header_id|>')
    assert len(end_token) == 1
     # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == start_tokens[0] and l[i + 1] == start_tokens[1]:
            start_indexes.append(i)
            # Now look for end index
            for j in range(i + 2, len(l)):
                if l[j] == end_token[0]:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(batch, processor, tokenizer, device):
    """
    Collate Function for Batch Processing
    Args:
        batch: iterable of items from XRayDataset
        processor: processor function
        tokenizer: tokenizer function
        device: model device
    returns: inputs and labels ids
    """
    # largely copied from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py, with slight modifications
    messages = []
    images = []
    for img, output_text in batch:
        messages.append(prep_message(img, output_text))
        images.append(img)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    inputs = processor(
        images,
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(device)
    inputs = inputs.to(device)
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)
    # convert output text to labels
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in get_start_end_idx(ids_list, tokenizer):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    labels_ids = labels_ids.to(device)
    return inputs, labels_ids
