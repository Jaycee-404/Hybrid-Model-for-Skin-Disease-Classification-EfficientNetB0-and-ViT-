import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchmetrics.classification import MulticlassAccuracy
from PIL import Image

# Prompt user to specify dataset location
DATA_DIR = input("Enter the path to the dataset CSV file: ")
data = pd.read_csv(DATA_DIR)

# Mapping the diagnosis to numerical values
label_mapping = {
    'mel': 0,
    'nv': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6,
}
data['dx_num'] = data['dx'].map(label_mapping)

# Load images from user-specified directory
IMG_DIR_BASE = input("Enter the path to the image directory: ")

def load_images(img_ids, img_dir_base=IMG_DIR_BASE):
    images = []
    for img_id in img_ids:
        img_path_part1 = os.path.join(img_dir_base, 'HAM10000_images_part_1', img_id + '.jpg')
        img_path_part2 = os.path.join(img_dir_base, 'HAM10000_images_part_2', img_id + '.jpg')
        
        if os.path.exists(img_path_part1):
            img_path = img_path_part1
        elif os.path.exists(img_path_part2):
            img_path = img_path_part2
        else:
            raise FileNotFoundError(f"Image not found: {img_id}.jpg")
        
        img = Image.open(img_path)
        images.append(img)
    return images

# Define Dataset class for PyTorch
class SkinCancerDataset(Dataset):
    def __init__(self, img_ids, labels, img_dir_base=IMG_DIR_BASE, transform=None):
        self.img_ids = img_ids
        self.labels = labels
        self.img_dir_base = img_dir_base
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        label = self.labels[idx]
        img = load_images([img_id])[0]
        if self.transform:
            img = self.transform(img)
        return img, label