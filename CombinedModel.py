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

# Load the Skin Cancer MNIST: HAM10000 dataset
DATA_DIR = 'path_to_your_dataset\HAM10000_metadata.csv'
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

# Load images from local directory
def load_images(img_ids, img_dir_base='path_to_your_dataset'):
    images = []
    for img_id in img_ids:
        # Check both parts of the image directory
        img_path_part1 = os.path.join(img_dir_base, 'HAM10000_images_part_1', img_id + '.jpg')
        img_path_part2 = os.path.join(img_dir_base, 'HAM10000_images_part_2', img_id + '.jpg')

        if os.path.exists(img_path_part1):
            img_path = img_path_part1
        elif os.path.exists(img_path_part2):
            img_path = img_path_part2
        else:
            raise FileNotFoundError(f"Image not found: {img_id}.jpg")

        img = Image.open(img_path)  # Open image with PIL
        images.append(img)
    return images  # Return as list of PIL images

# Define Dataset class for PyTorch
class SkinCancerDataset(Dataset):
    def __init__(self, img_ids, labels, img_dir_base='path_to_your_dataset', transform=None):
        self.img_ids = img_ids
        self.labels = labels
        self.img_dir_base = img_dir_base
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        label = self.labels[idx]
        img = load_images([img_id])[0]  # Load single image
        if self.transform:
            img = self.transform(img)
        return img, label

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create dataset and dataloaders
X_train, X_val, y_train, y_val = train_test_split(data['image_id'].values, data['dx_num'].values, test_size=0.2, stratify=data['dx_num'].values, random_state=42)

train_dataset = SkinCancerDataset(X_train, y_train, transform=transform)
val_dataset = SkinCancerDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=7):
        super(VisionTransformer, self).__init__()
        # Vision Transformer (ViT) part - Basic architecture
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)  # Use latest weights
        
        # Extract the in_features from the classifier
        vit_in_features = self.vit.heads.head.in_features
        
        # Modify the final layer for the number of classes
        self.vit.heads.head = nn.Linear(vit_in_features, num_classes)

        # EfficientNetV2 part
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)  # Modify for number of classes

        # Combine the two models' output
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.vit.heads.head.out_features + self.efficientnet.classifier[1].out_features, num_classes)
        )

    def forward(self, x):
        vit_features = self.vit(x)
        efficientnet_features = self.efficientnet(x)
        combined = torch.cat((vit_features, efficientnet_features), dim=1)
        out = self.fc(combined)
        return out

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(num_classes=len(label_mapping)).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=15):
    best_val_accuracy = 0.0
    early_stopping_counter = 0
    patience = 3  # Early stopping patience
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_preds / total_preds

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


# Run training
train_model(model, train_loader, val_loader)

# Plotting confusion matrix
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

confusion_mtx = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display sample predictions
def display_predictions(model, dataloader, num_samples=5):
    model.eval()
    samples_displayed = 0
    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(imgs)):
                if samples_displayed >= num_samples:
                    break

                img = imgs[i].cpu().permute(1, 2, 0).numpy()  # Convert tensor to image
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
                img = np.clip(img, 0, 1)

                true_label = list(label_mapping.keys())[labels[i].item()]
                predicted_label = list(label_mapping.keys())[preds[i].item()]

                plt.subplot(1, num_samples, samples_displayed + 1)
                plt.imshow(img)
                plt.title(f'True: {true_label}\nPred: {predicted_label}')
                plt.axis('off')

                samples_displayed += 1

            if samples_displayed >= num_samples:
                break

    plt.show()

# Display 5 sample predictions from the validation set
display_predictions(model, val_loader, num_samples=5)
