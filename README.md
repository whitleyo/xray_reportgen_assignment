

# XRAY Report Generation

This repository is an implementation of tasks related to NLP and multi-modal NLP models,
specifically for data pertaining to the [IU-X-Ray Dataset](https://paperswithcode.com/dataset/iu-x-ray).

## Environments and Requirements

Google Colab

Tesla T4 GPU (14GB VRAM).

Note, you'll need a google colab pro plus subscription to reliably run this. I've tried before
using my own laptop (8GB VRAM) which doesn't cut it for training anything but QWEN-VL-2B or a similarly small
model, and free google colab will randomly cut you out mid training/fine-tuning, which kind of nixes
its usefulness.

In a future iteration I might just consider
using google cloud or azure but for the sake of getting this done we'll use google colab pro for now.

## Dataset

[IU-X-Ray Dataset](https://paperswithcode.com/dataset/iu-x-ray)

Data is organized as follows
```
data/
├── annotation.json # includes data split, patient id and report findings.
Ignore the findings in "others"
└── images # each patient contains 1-4 images
├── CXR1000_IM-0003
├── 0.png
├── 1.png
└── 2.png
├── ...
└── CXR9_IM-2407
├── 0.png
└── 1.png
```

Note that the starting validation data was modified to contain original notes and not summaries converted into dictionaries. Task 1
is to re-convert the validation data.

##<a name="preprocessing"></a>Preprocessing

Image data is resized to 224 x 224 images, padding to square.
For MultiImage data, images are combined and resized to 224 x 224, with square padding applied.

For text, data is extracted from the relevant json files, converted to a single string, and tokenized. More details for each task.

## Task 1: Prompt Engineering:

Used [llama]([unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B)) to generate validation labels,
according to the following json format:

```
{
    'lung': 'summary of lung category related findings or leave as empty string if no lung related findings.',
    'heart': 'summary of heart category or leave as empty string if no heart related findings.',
    'bone': 'summary of bone category related findings or leave as empty string if no bone related findings.',
    'mediastinal': 'summary of mediastinal category related findings or leave as empty string if no mediastinal related findings.',
    'others': 'summary of any other findings that are NOT lung related, NOT heart related, NOT bone related, NOT mediastinal related. Leave as empty string if no findings.'
}
```

### Image processing:

See [Preprocessing](#preprocessing) section above

### Text Preprocessing:

Data is fed into the user prompt, which is then passed to llama's tokenizer.

### Running this Task

Run the following notebook: [notebooks/Task1_Llama-3.1_8b_get_validation_labels.ipynb](https://github.com/whitleyo/xray_reportgen_assignment/blob/master/notebooks/Task1_Llama-3.1_8b_get_validation_labels.ipynb)

The prompt for json report generation is included in that notebook.

### Results

Overall, the results appear to be sane enough perusing through a few examples. Occasionally there might be something missed, or something
along the lines of a mediastinal result being placed with a cardiac result, which is perhaps not surprising given that mediastinal results
often occur in context with cardiac findings.

Examples:

1.

original report: The trachea is midline. The heart XXXX is large, unchanged from prior exam. Slightly widened mediastinum, secondary to cardiomegaly and a tortuous aorta, is accentuated by AP portable technique. There are low lung volumes causing bibasilar atelectasis and bronchovascular crowding. The lungs do not demonstrate focal infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.
json extraction result: 
{
  'lung': 'Lungs do not demonstrate focal infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.', 
  'heart': 'The heart XXXX is large, unchanged from prior exam. Slightly widened mediastinum, secondary to cardiomegaly and a tortuous aorta, is accentuated by AP portable technique.', 
  'bone': '', 
  'mediastinal': '', 
  'others': ''
}

2.
original report: Cardiac and mediastinal contours are within normal limits. The lungs are clear. Mild prominence left hilar contour. Bony structures are intact.
json extraction result:
{
  'lung': 'The lungs are clear. Mild prominence left hilar contour.',
  'heart': 'Cardiac and mediastinal contours are within normal limits.',
  'bone': 'Bony structures are intact',
  'mediastinal': 'Mediastinal contours are within normal limits.',
  'others': ''
}

3.
original report: The heart is normal size. The mediastinum is unremarkable. A tortuous, calcified thoracic aorta is present. The lungs are hyperexpanded, consistent with emphysema. There is no pleural effusion, pneumothorax, or focal airspace disease. The XXXX are unremarkable.
json extraction result:
{
  'lung': 'The lungs are hyperexpanded, consistent with emphysema. There is no pleural effusion, pneumothorax, or focal airspace disease.',
  'heart': 'The heart is normal size',
  'bone': '',
  'mediastinal': 'A tortuous, calcified thoracic aorta is present.',
  'others': 'The XXXX are unremarkable.'
  }

It should also be noted that llama did not return correctly formated json output 100% of the time, and so I had to do a loop where output is generated,
cleaned for easy to fix items such as \```json and \``` on the ends, and often adding a curly brace (}) at the end (otherwise llama would, for some samples,
not give correctly formatted json in 100 tries even with otherwise pretty correct looking json output). It should be also noted that llama was fairly probabilistic
in whether or not it returned correctly formatted output or returned code to generate said output.

__Conclusion__

Llama 3.2-8B Instruct is OK at handling formatting but without doing any further fine tuning I'm afraid this is the best I'll be able to do here.
I tried a variety of simpler and more complex prompts than the one provided, and at some point adding examples seemed to saturate in terms of performance.



## Task 2: Fine Tuning Multimodal model

Here, I tried out [Llama 3.2-Vision-11B](https://huggingface.co/unsloth/Llama-3.2-11B-Vision) for fine tuning. 

### Text Preprocessing

The following function was used for text conversion:

```
def convert_to_conversation(img, response, output_format=output_json_format):
    instruction="Summarize the input image(s) in the following json format {}".format(output_json_format)
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : img} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : response} ]
        },
    ]
    return { "messages" : conversation }
```

It was imported from https://github.com/whitleyo/xray_reportgen_assignment/blob/master/src/llama_3_2_vision_unsloth_helpers.py

### Training

To run fine tuning for llama 3.2-vision-11B, run the following notebook: [notebooks/Task2_part1_Llama-3.2_vision_11b_fine_tune.ipynb](https://github.com/whitleyo/xray_reportgen_assignment/blob/master/notebooks/Task2_part1_Llama-3.2_vision_11b_fine_tune.ipynb)

### Trained Models

TODO

### Inference

TODO

### Evaluation

TODO


### Results

TODO


## Contributing

TODO

## Acknowledgement

We thank the contributors of the IU-XRAY dataset
