# Hybrid Model for Automated Skin Disease Classification  
### Using EfficientNetB0 and Vision Transformer (ViT2)  

This project implements a **Hybrid Deep Learning Model** combining **EfficientNetB0** and **Vision Transformer (ViT2)** for automated skin disease classification. The model is trained on the **HAM10000 dataset** to identify **seven types of skin lesions** with high accuracy.  

Additionally, **separate models** using only **EfficientNetB0** and **ViT2** are also provided for individual performance comparison.  

---

## 📌 Project Overview  

Skin disease classification is a crucial medical challenge, and deep learning models have proven effective in diagnosing various skin conditions. This project explores three different approaches:  

- **Hybrid Model (EfficientNetB0 + ViT2)** → Combines CNN and Transformer architectures for enhanced accuracy.  
- **EfficientNetB0 Model** → Uses CNN-based feature extraction for classification.  
- **Vision Transformer (ViT2) Model** → Utilizes self-attention mechanisms for image classification.  

Each model is trained separately and evaluated for performance comparison.  

---

## 📂 Dataset Information  

The **HAM10000** dataset consists of **10,015 labeled dermatoscopic images** categorized into **seven types of skin lesions**:  

| Label | Description |  
|-------|------------|  
| **mel** | Melanoma |  
| **nv** | Melanocytic nevi |  
| **bkl** | Benign keratosis-like lesions |  
| **bcc** | Basal cell carcinoma |  
| **akiec** | Actinic keratoses and intraepithelial carcinoma |  
| **vasc** | Vascular lesions |  
| **df** | Dermatofibroma |  

---

## 🏗 Model Architectures  

### 🔹 1️⃣ Hybrid Model: EfficientNetB0 + ViT2  
- **EfficientNetB0** extracts high-level feature maps from input images.  
- **ViT2** processes the feature maps using self-attention mechanisms.  
- A **fully connected layer** makes the final classification decision.  

### 🔹 2️⃣ EfficientNetB0 Model  
- A CNN-based model optimized for **lightweight** feature extraction.  
- Uses **depthwise separable convolutions** for efficiency.  
- Pretrained on **ImageNet** for better generalization.  

### 🔹 3️⃣ Vision Transformer (ViT2) Model  
- Converts input images into **patch embeddings**.  
- Applies **multi-head self-attention** for learning spatial relationships.  
- Uses **transformer layers** instead of convolutional operations.  

---

## 📊 Model Evaluation  

### ✅ **Comparison of Models**  
| Model | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |  
|--------|------------|----------------|-----------------|----------------------|  
| **EfficientNetB0** | 0.1966 | 93.52% | 1.3750 | 68.60% |  
| **ViT2** | 0.6988 | 73.63% | 0.7094 | 73.84% |  
| **Hybrid Model** | 0.1491 | 95.06% | 0.5655 | 86.82% |  

---

This project demonstrates the effectiveness of a hybrid deep learning approach in **skin disease classification** by leveraging both convolutional and transformer-based architectures.  
