import os
from os.path import join, splitext
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir (str): Path to the parent directory containing subdirectories (e.g., 'label_folder').
            transform (callable, optional): Optional transform to be applied on an image.
            mode (str): Either "train" or "valid" to select the correct folder.
        """
        
        # self.label_folder = os.path.join(root_dir, 'chexbert_labels')
        self.root = root_dir
        self.img_path = os.path.join(self.root, 'PNG')
        self.img_folders = [folder for folder in os.listdir(self.img_path) if splitext(folder)[1] == '']

        self.label_folder = os.path.join(self.root, 'chexbert_labels')
        self.label_path = os.path.join(self.label_folder, 'findings_fixed.json')
        self.labels = []
        self.img_paths = []
        self.transform = transform
        self.mode = mode
        
        # load a dictionary of image paths and labels
        with open(self.label_path, 'r') as f:
            label_data = []
            for line in f:
                label_data.append(json.loads(line))

        for label_dict in label_data:
            label_list_per_sample = []
            for key, value in label_dict.items():
                if key == 'path_to_image': # save image paths
                    if splitext(value)[0].split('/')[0] == self.mode:
                        value = splitext(value)[0] + '.png'
                        for folder in self.img_folders:
                            img_subfolder_path = os.path.join(os.path.join(self.img_path, folder), 'PNG')
                            img_path = os.path.join(img_subfolder_path, value)
                            if os.path.exists(img_path):
                                self.img_paths.append(img_path)
                    else:
                        break # so labels for test data will not be saved
                else: # save label vectors
                    if value is None: 
                        label_list_per_sample.append(0) # if this disease is not mentioned, it is perhaps not present
                    elif value == -1:
                        label_list_per_sample.append(0) # if radiologist is uncertain, chances of having this disease or being healthy are half half
                    else:
                        label_list_per_sample.append(value) # either having this disease or not
            if len(label_list_per_sample) > 0: # empty list implies a testing smaple
                self.labels.append(torch.tensor(label_list_per_sample))
            
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")  # convert to RGB
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def main():
    data_folder = '../../../../../../../storage/ice1/shared/bmed6780/mip_group_2/CheXpert Plus'

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    dataset = CheXpertDataset(root_dir=data_folder, transform=transform, mode="train")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    for imgs, labels in tqdm(dataloader):
        pass

if __name__ == "__main__":
    main()