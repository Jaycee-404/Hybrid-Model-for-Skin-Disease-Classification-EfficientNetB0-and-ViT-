import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Load the Skin Cancer MNIST: HAM10000 dataset
DATA_DIR = r'path_to_your_dataset\HAM10000_metadata.csv'
img_dir_base = r'path_to_your_dataset'

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

# Define custom dataset
class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir_base, transform=None):
        self.df = df
        self.img_dir_base = img_dir_base
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['dx_num']
        
        img_path_part1 = os.path.join(self.img_dir_base, 'HAM10000_images_part_1', img_id + '.jpg')
        img_path_part2 = os.path.join(self.img_dir_base, 'HAM10000_images_part_2', img_id + '.jpg')
        
        img_path = img_path_part1 if os.path.exists(img_path_part1) else img_path_part2
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
])

# Create dataset and dataloaders
img_dir_base = r'C:\Users\pkpav\OneDrive\Desktop\DL Project\ham10000'
dataset = HAM10000Dataset(data, img_dir_base, transform=transform)
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['dx_num'], random_state=42)
train_dataset = HAM10000Dataset(train_data, img_dir_base, transform=transform)
val_dataset = HAM10000Dataset(val_data, img_dir_base, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define Vision Transformer components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.transpose(0, 1)
        x, _ = self.mha(x, x, x)
        return x.transpose(0, 1)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_chans=3, num_classes=7, embed_dim=64, num_heads=4, num_layers=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim),
                MultiHeadSelfAttention(embed_dim, num_heads),
                nn.LayerNorm(embed_dim),
                MLP(embed_dim, embed_dim * 2)
            ])
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for ln1, attn, ln2, mlp in self.layers:
            x = x + attn(ln1(x))
            x = x + mlp(ln2(x))

        return self.head(x[:, 0])

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Train Acc: {correct / len(train_loader.dataset):.4f}, "
              f"Val Acc: {val_correct / len(val_loader.dataset):.4f}")

train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate the model with a confusion matrix
def plot_confusion_matrix(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_mapping.keys(), 
                yticklabels=label_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(model, val_loader)