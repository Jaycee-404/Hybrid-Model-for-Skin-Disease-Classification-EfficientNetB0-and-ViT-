#Before running the main code, verify if all necessary files exist:

import os
import pandas as pd

# Set dataset paths (Modify this as per your setup)
DATA_DIR = r'path_to_your_dataset\HAM10000_metadata.csv'
IMG_DIR_1 = r'path_to_your_dataset\HAM10000_images_part_1'
IMG_DIR_2 = r'path_to_your_dataset\HAM10000_images_part_2'

# Check CSV file
if not os.path.exists(DATA_DIR):
    print(f"Metadata file not found: {DATA_DIR}")
else:
    print("Metadata file found!")

# Check image folders
missing_folders = []
for folder in [IMG_DIR_1, IMG_DIR_2]:
    if not os.path.exists(folder):
        missing_folders.append(folder)

if missing_folders:
    print(f"Missing image folders: {missing_folders}")
else:
    print("All image folders found!")

# Check dataset consistency
data = pd.read_csv(DATA_DIR)
if 'image_id' in data.columns and 'dx' in data.columns:
    print("Metadata structure is correct!")
else:
    print("Metadata structure is incorrect. Check CSV format.")