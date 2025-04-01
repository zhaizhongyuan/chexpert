import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from os.path import splitext
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torchvision.models import DenseNet121_Weights

# Path configuration
data_folder = '../../../../../../../storage/ice1/shared/bmed6780/mip_group_2/CheXpert Plus'
img_path_root = os.path.join(data_folder, 'PNG')
img_folders = [folder for folder in os.listdir(img_path_root) if splitext(folder)[1] == '']

label_folder = os.path.join(data_folder, 'chexbert_labels')
label_path = os.path.join(label_folder, 'findings_fixed.json')
mode = 'valid'

# Load a dictionary of image paths and labels
with open(label_path, 'r') as f:
    label_data = [json.loads(line) for line in f]
print(len(label_data))

transform = DenseNet121_Weights.DEFAULT.transforms()

# Function to process a single label dictionary
def process_label_dict(label_dict):
    for key, value in label_dict.items():
        if key == 'path_to_image':
            if splitext(value)[0].split('/')[0] != mode:
                return None  # skip if not in desired mode

            value = splitext(value)[0] + '.png'
            for folder in img_folders:
                img_subfolder_path = os.path.join(img_path_root, folder, 'PNG')
                full_img_path = os.path.join(img_subfolder_path, value)
                if os.path.exists(full_img_path):                
                    try:
                        img = Image.open(full_img_path).convert("RGB")
                        img = transform(img)
                    except (OSError, UnidentifiedImageError) as e:
                        print(f"Could not open {full_img_path}: {e}.")
                        continue
                    
                    return full_img_path
    return None

# Multithreaded processing with real-time progress bar
img_paths = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_label_dict, label_dict) for label_dict in label_data]

    for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        result = future.result()
        if result is not None:
            img_paths.append(result)

        # periodically run garbage collection
        if idx % 100 == 0:
            gc.collect()

print(len(img_paths))
print(len(np.unique(img_paths)))
