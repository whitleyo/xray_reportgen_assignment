import os
import json
import torch
from PIL import Image
import re
# from transformers import LlamaTokenizer, LlamaForConditionalGeneration
# from torchvision import transforms
# from torchvision.transforms import ToTensor
from torchvision import transforms
# from torchvision.transforms import (
#     CenterCrop,
#     Compose,
#     Normalize,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     Resize,
#     ToTensor,
# )
from torch.utils.data import Dataset

class XRayImageDataset(Dataset):
    def __init__(self, top_image_dir, json_fpath, split, inference_mode=False, img_size=224):
        """
        Args:
            top_image_dir: top image directory (str)
            json_fpath: json file containing annotations paired with image directories (str)
            split: what split of data to pull (train, test, val)
        Notes:
            This object is intended to handle data passing for the X-ray image + annotation dataset for the assignment
        """
        super().__init__()
        self.top_image_dir = top_image_dir
        with open(json_fpath, "r") as file:
            json_data = json.load(file)
        self.data_index = json_data[split]
        self.inference_mode = inference_mode
        self.img_size = img_size

    def __getitem__(self, idx):
        """
        Get item given index
        Args:
            idx: integer index
        Returns:
            image, number of original images, json report in string format if 
        """
        json_entry = self.data_index[idx]
        id_pull = json_entry['id']
        image_return = self.load_images(image_dir=os.path.join(self.top_image_dir, id_pull))
        if self.inference_mode is False:
            # this assumes that there's a report in the json entry
            text = json.dumps(json_entry['report'])
            return image_return, text
        else:
            # for generic queries
            return image_return

    def __len__(self):
        return len(self.data_index)

    def load_images(self, image_dir):
        """
        Load images, concatenate into 1 large image
        Args:
            image_dir: full directory path to images files
            split_mode: train, val, or test. if train, do flips
        """
        all_images = []
        all_files_listed = os.listdir(image_dir)
        all_image_fnames = []
        for x in all_files_listed:
            if re.search('png$', x):
                all_image_fnames.append(x)
                
        all_image_paths = [os.path.join(image_dir, x) for x in all_image_fnames]
        for image_path in all_image_paths: 
            image = Image.open(image_path, 'r')
            image = self.transform_image(image)
            # note that any image in all images
            all_images.append(image)
        concat_images = torch.cat(all_images, axis=1)
        img_return = transforms.ToPILImage(mode='L')(concat_images)
        # a little inefficient but saves another write for resize and pad to square
        img_return = transforms.ToPILImage(mode='L')(self.transform_image(img_return))
        return img_return

    def transform_image(self, image): 
        image = self.pad_to_square(image)
        img_size = self.img_size
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), 
                                        transforms.ToTensor()]) 
        return transform(image)

    def pad_to_square(self, image): 
        width, height = image.size 
        max_dim = max(width, height) 
        padded_image = Image.new('L', (max_dim, max_dim)) 
        padded_image.paste(image, (0, 0)) 
        return padded_image







        