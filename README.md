

# XRAY Report Generation

This repository is an implementation of tasks related to NLP and multi-modal NLP models,
specifically for data pertaining to the [IU-X-Ray Dataset](https://paperswithcode.com/dataset/iu-x-ray).

## Environments and Requirements

- Ubuntu 22.04 on WSL2 on Windows 10 (10.0.1905)
- GPU: NVIDIA GeForce RTX 2070
- RAM: 16 GB Windows, 8GB available to WSL2
- CUDA version: 12.4.1
- python version: 3.8.2 (conda-forge)

To install requirements:

```setup
# for training models and running inference
conda env create -f xray_reportgen.yml
# for running green_score from https://github.com/Stanford-AIMI/GREEN
conda env create -f green_score.yml
```

To activate:

```
# for training models and running inference
conda activate xray_reportgen
# for running green_score
conda activate green_score
```

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

Used [llama](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (meta-llama/Llama-3.2-1B-Instruct from huggingface) to generate validation labels,
according to the following json format:

```
{
"lung": "Lungs are mildly hypoinflated but grossly clear of focal
airspace disease, pneumothorax, or pleural effusion. Pulmonary vasculature
are within normal limits in size.",
"heart": "Cardiac silhouette within normal limits in size.",
"mediastinal": "Mediastinal contours within normal limits in size.",
"bone": "Mild degenerative endplate changes in the thoracic spine. No
acute bony findings.",
"others": ""
}
```

### Image processing:

See [Preprocessing](#preprocessing) section above

### Text Preprocessing:

Data is fed into the user prompt, which is then passed to llama's tokenizer.

### Running this Task

```
conda activate xray_reportgen
python task1_convert_validation_annotations.py
```

### Results

The system and user prompts can be found in the following [script](https://github.com/whitleyo/xray_reportgen_assignment/blob/master/scripts/task1_convert_validation_annotations.py)

Overall, the results are range from somewhat incorrect in placement of tissue results but with proper formatting to 

Examples:

1.
```
original report:
Heart size is enlarged. The aorta is unfolded. Otherwise the mediastinal contour is normal.
There are streaky bibasilar opacities. There are no nodules or masses. No visible pneumothorax. No visible pleural fluid.
The XXXX are grossly normal. There is no visible free intraperitoneal air under the diaphragm.

json extraction result:

{
  'lung': '',
  'heart': 'The aorta is unfolded',
  'bone': 'The XXXX are grossly normal',
  'mediastinal': 'Mediastinal contour is normal',
  'others': 'No visible pneumothorax, no visible pleural fluid, streaky bibasilar opacities, no nodules or masses, free intraperitoneal air under the diaphragm.'
}
```

2.
```
original report:
The heart is mildly enlarged. The mediastinal contours are stable. The lungs are clear.

json extraction result:

{
  'lung': 'The lungs are clear.',
  'heart': 'The heart is mildly enlarged',
  'mediastinal': 'The mediastinal contours are stable',
  'bone': '',
  'others': ''
}
```

3.
```
original report: 2 images. Heart size and pulmonary vascular engorgement appear within limits of normal.
Mediastinal contour is unremarkable. No focal consolidation, pleural effusion, or pneumothorax identified. No convincing acute bony findings.

json extraction result:

{
  'lung': 'No focal consolidation, pleural effusion, or pneumothorax identified.',
  'heart': 'Heart size within normal limits',
  'bone': 'No acute bony findings.',
  'mediastinal': 'Mediastinal contour is unremarkable.',
  'others': 'No focal air space opacity to suggest pneumonia.'
}
```

It should also be noted that llama did not return correctly formated json output 100% of the time, and so I had to do a loop where output is generated,
cleaned for easy to fix items such as \```json and \``` on the ends, and often adding a curly brace (}) at the end (otherwise llama would, for some samples,
not give correctly formatted json in 100 tries even with otherwise pretty correct looking json output).

__Conclusion__

Llama 3.2-1B Instruct is OK at handling formatting but without doing any further fine tuning I'm afraid this is the best I'll be able to do here.
I tried a variety of simpler and more complex prompts than the one provided, and at some point adding examples seemed to saturate in terms of performance.



## Task 2: Fine Tuning Multimodal model

Here, I tried out [QWEN-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) for fine tuning. 

### Text Preprocessing

The following functions are used to process text data for QWEN-VL2V model
[prepare_message](https://github.com/whitleyo/xray_reportgen_assignment/blob/master/src/qwenl2_helpers.py#L7)
process_vision_info from qwen_vl_utils

### Training

To train the model(s) in this repo, run this command:

```train
conda activate xray_reportgen
cd scripts
python task2_qwen2_vl2b_train.py
```

Currently, I've only gotten QWEN2-VL2B to train or run inference out of QWEN2-VL-2B, llama 3.2 Vision-11B, and MOLMO.

We add lora adapters (of dimension 6) to the following modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] to give flexibility at the projection stage
before logits get calculated.

The training loss is torch.nn.CrossEntropyLoss calculated for tokens matching the indices of the reference response i.e. torch.nn.CrossEntropyLoss(outputs[start_response_idx:end_response_idx], labels[start_response_idx:end_response_idx])
This is done by making labels before start_response_idx (the start of the response) and after end_response_idx (after end of response) to -100, which signals for the loss function to ignore those indices.

As it currently stands, my first run at QWEN2-VL-2B resulted in a model that just spits out a blank response. I suspect this has to do with the learning rate I picked, and the fact that most of the 
responses are blank (and the high speed of convergence) suggests that the model just learned to call everything blank and that gets to a (very unhelpful) local minimum.

### Trained Models

Currently no models are pushed to the internet, but if you run the training, you'll get the trained model(s) in ./results/task2_qwen2_vl2b_train.

### Inference

```inference
conda activate xray_reportgen
cd scripts
python task2_qwen2_vl2b_label_val_test.py 
```

### Evaluation

We'd like to use the [Green Scorer](https://github.com/Stanford-AIMI/GREEN), but first we have to get the model successfully trained



### Results

TODO


## Contributing

TODO

## Acknowledgement

We thank the contributors of the IU-XRAY dataset
