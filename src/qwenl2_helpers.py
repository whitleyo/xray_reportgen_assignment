from functools import partial
import os

from qwen_vl_utils import process_vision_info
import torch

def prep_message(img, output_text):
    output_json_format = """
     {
         \"lung\": \"summary of lung related findings, empty if no findings\",
         \"heart\": \"summary of heart related findings, empty if no findings\",
         \"bone\": \"summary of bone related findings, empty if no findings\",
         \"mediastinal\": \"summary of mediastinal related findings, empty if no findings\",
         \"others\": \"summary of any findings not related to lung, heart, bone, or mediastinal\"
     }
     """
    message =   [
                    {
                        "role": "user",
                        "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": "Summarize the input image(s) in the following json format {}".format(output_json_format)},
                            ],
                    },
                    {
                        "role": "assistant", 
                        "content": output_text
                    },
                ]          
    
    return message

def get_start_end_idx(l, tokenizer):
    start_indexes = []
    end_indexes = []
    start_token_pair = tokenizer.encode("<|im_start|>assistant")
    assert len(start_token_pair) == 2
    end_token = tokenizer.encode("<|im_end|>")
    assert len(end_token) == 1
     # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == start_token_pair[0] and l[i + 1] == start_token_pair[1]:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == end_token[0]:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))
    

def collate_fn(batch, processor, tokenizer, device):
    # largely copied from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py, with slight modifications
    messages = []
    for img, output_text in batch:
        messages.append(prep_message(img, output_text))
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)
    # convert output text to labels
    labels_list = []
    
    for ids_list in input_ids_lists:
        # my understanding is that the -100 effectively nixes anything not in the assistant set of positions
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in get_start_end_idx(ids_list, tokenizer):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    labels_ids = labels_ids.to(device)
    return inputs, labels_ids