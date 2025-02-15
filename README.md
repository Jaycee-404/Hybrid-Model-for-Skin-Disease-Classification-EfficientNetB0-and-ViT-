# Hybrid Model for Automated Skin Disease Classification  

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

### Using EfficientNetB0 and Vision Transformer (ViT2)  



This project implements a **Hybrid Deep Learning Model** combining **EfficientNetB0** and **Vision Transformer (ViT2)** for automated skin disease classification. The model is trained on the **HAM10000 dataset** to identify **seven types of skin lesions** with high accuracy.  

Additionally, **separate models** using only **EfficientNetB0** and **ViT2** are also provided for individual performance comparison.  

---

## Project Overview  

Skin disease classification is a crucial medical challenge, and deep learning models have proven effective in diagnosing various skin conditions. This project explores three different approaches:  

- **Hybrid Model (EfficientNetB0 + ViT2)** → Combines CNN and Transformer architectures for enhanced accuracy.  
- **EfficientNetB0 Model** → Uses CNN-based feature extraction for classification.  
- **Vision Transformer (ViT2) Model** → Utilizes self-attention mechanisms for image classification.  

Each model is trained separately and evaluated for performance comparison.  

---

## Dataset Information  

Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


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

## Model Architectures  

### 1️⃣ Hybrid Model: EfficientNetB0 + ViT2  
- **EfficientNetB0** extracts high-level feature maps from input images.  
- **ViT2** processes the feature maps using self-attention mechanisms.  
- A **fully connected layer** makes the final classification decision.  

### 2️⃣ EfficientNetB0 Model  
- A CNN-based model optimized for **lightweight** feature extraction.  
- Uses **depthwise separable convolutions** for efficiency.  
- Pretrained on **ImageNet** for better generalization.  

### 3️⃣ Vision Transformer (ViT2) Model  
- Converts input images into **patch embeddings**.  
- Applies **multi-head self-attention** for learning spatial relationships.  
- Uses **transformer layers** instead of convolutional operations.  

---

### **Experimental Results**  

The model underwent several experimental phases to enhance performance and overcome key challenges like **overfitting, lack of generalization, and class imbalance**.  

---

### **Baseline Models & Issues**  

#### **ViT Model**  
- **Training Accuracy:** **73.63%**, but **Validation Accuracy** stagnated at **73.84%**.  
- **Training Loss:** **0.6988**, and **Validation Loss:** **0.7094**.  
- **Challenges:**  
  - Overfitting was reduced compared to other models, but validation accuracy remained low.  
  - Required more robust feature extraction to improve generalization.  

#### **EfficientNetB0 Model**  
- **Training Accuracy:** **93.52%**, but **Validation Accuracy** dropped to **68.60%**.  
- **Training Loss:** **0.1966**, and **Validation Loss:** **1.3750**.  
- **Challenges:**  
  - Strong overfitting—high training accuracy but **poor generalization to validation data**.  
  - Validation loss increased significantly, indicating a need for **better regularization techniques**.  

---

### **Improvements with Data Augmentation & Early Stopping**  

#### **Techniques Applied**  
- **Transformations:** Random rotation, horizontal flipping, zoom, and brightness adjustments.  
- **Early Stopping:** Stopped training when validation loss no longer improved.  

#### **Impact**  
- **Improved Generalization:** Reduced overfitting significantly.  
- **Stable Training:**  
  - **Training Loss:** Decreased from **0.6988 → 0.1491**.  
  - **Training Accuracy:** Increased from **73.63% → 95.06%**.  
- **Validation Accuracy:** Increased from **73.84% → 86.82%**, showing strong improvement.  

---

### **Performance Evaluation & Confusion Matrix Analysis**  

- **Overall Accuracy:** Achieved **86.82%**, a major improvement from the baseline models.  
- **Precision & Recall:** Improved for **melanoma and benign keratosis**, leading to better classification of critical skin lesions.  
- **F1-Score:** Increased across most classes, particularly for underrepresented lesion types.  
- **Confusion Matrix Insights:**  
  - The model **still struggles with differentiating melanocytic nevus (nv) and melanoma (mel)** due to their visual similarities.  
  - More advanced feature extraction or additional training data could further improve differentiation.

### **Predicted Images**
![Predicted Images](https://github.com/user-attachments/assets/0baa50a2-efbc-45fe-98ca-7b05bd284753)

### **Confusion Matrix**
![Confusion Matrix](https://github.com/user-attachments/assets/2e591e7a-235c-42b6-93b6-f8d086650bb2)

---

### **Conclusion**  

This project successfully implemented a **Hybrid Deep Learning Model** using **ViT and EfficientNetB0** for **skin lesion classification**. By combining **ViT's global feature extraction** with **EfficientNetB0's local feature extraction**, the model demonstrated **high accuracy** and **robust generalization**.  

### **Key Takeaways**  
**Data Augmentation & Early Stopping** significantly improved performance.  
**Overfitting was reduced**, leading to **better validation accuracy** (**86.82%**).  
**Precision, Recall, and F1-Scores improved**, especially for underrepresented lesion types.  

While the model showed **strong classification ability**, challenges remain in **distinguishing visually similar lesions**. Future improvements could include **additional metadata, advanced augmentation techniques, and larger datasets** to further enhance classification accuracy.  

This work highlights how **hybrid architectures and optimized training strategies** can **advance automated skin disease diagnosis**, contributing to the field of **medical image analysis**. 
