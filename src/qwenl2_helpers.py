from functools import partial
import os
import json
from qwen_vl_utils import process_vision_info
import torch

output_json_format = """
{
 \"lung\": \"summary of lung related findings, empty if no findings\",
 \"heart\": \"summary of heart related findings, empty if no findings\",
 \"bone\": \"summary of bone related findings, empty if no findings\",
 \"mediastinal\": \"summary of mediastinal related findings, empty if no findings\",
 \"others\": \"summary of any findings not related to lung, heart, bone, or mediastinal\"
}
"""

def prep_message(img, output_text=None, output_format=output_json_format):
    """
    Prepare Message for QWEN-VL-2B
    Args:
        img: image, as accessed from XRayImageDataset object (see data_utils.py), in PIL format
        output_text: string. output text that will be passed to model and tokenized into labels as part of the full message
        output_format: string, output format. defaults to output_json_format
    returns: list of dicts to be used as message input for model
    """
    message =   [
                    {
                        "role": "user",
                        "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": "Summarize the input image(s) in the following json format {}".format(output_json_format)},
                            ],
                    }, 
                ]
    if output_text is not None:
        try:
            assert type(output_text) is str
        except:
            raise TypeError('Expected str if output_text is not None, got {}'.format(type(output_text)))
        message.append({
                        "role": "assistant", 
                        "content": output_text
                        })
    
    return message

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
    """
    Collate Function for Batch Processing
    Args:
        batch: iterable of items from XRayDataset
        processor: processor function for QWEN-VL-2B
        tokenizer: tokenizer function for QWEN-VL-2B
        device: model device
    returns: inputs and labels ids
    """
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
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in get_start_end_idx(ids_list, tokenizer):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    labels_ids = labels_ids.to(device)
    return inputs, labels_ids


def generate(image, model, processor):
    """
    Generate text from image
    Args:
        image: PIL image
        model: QWEN-VL-2B model, or model with adapters
    returns: generated output text
    """
    messages = prep_message(image)
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


def convert_2_json(gen_text):
    """
    Attempt to convert text to json
    Args:
        gen_text: generated text from generate function
    returns: json output or throws error if text cannot be converted to json
    """
    try:
        # strip off most common start and end sequences. if we get a valid json we don't really care that much
        # gen_text = re.sub('```json', '', gen_text)
        # gen_text = re.sub('```', '', gen_text)
        gen_text_json = json.loads(gen_text)
        try:
            assert type(gen_text_json) is dict
        except:
            raise TypeError('generated json is of type {}'.format(type(gen_text_json)))
        for tissue_name in ['lung', 'heart', 'bone', 'mediastinal', 'others']:
            # add emtpy entries where tissue is not included
            if not tissue_name in gen_text_json.keys():
                gen_text_json[tissue_name] = ""
    except json.JSONDecodeError as e:
        print(f"Invalid JSON string: {e}")
        print(gen_text)
        assert False
        
    return gen_text_json

def generate_multi_try(image, model, processor, n_tries=100):
    """
    Try multiple times to run generation to get json output
    Args:
        image: PIL image
        model: QWEN-VL-2B model or adaptation thereof
        n_tries: maximum number of tries
    returns:
        json formatted text from model response
    """
    succeeded = False
    for k in range(n_tries):
        print("attempt generation, try {}".format(k+1))
        gen_text = generate(image, model, processor)
        print("Generated text")
        print(gen_text)
        try:
            gen_text_json = convert_2_json(gen_text)
            succeeded = True
        except:
            print("Failed, trying another round of generation")
        if succeeded:
            break
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