# 🌸 Flower Species Classification using Transfer Learning

> Fine-tuning pretrained CNNs (ResNet50, EfficientNetB0, VGG16) in TensorFlow & PyTorch to classify 5 flower species with 90.19% accuracy.

---

## 📌 Project Overview

This project builds a deep learning image classifier for flower species recognition using **Transfer Learning** on the [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle. Rather than training from scratch, we fine-tune pretrained Convolutional Neural Networks (CNNs) that were originally trained on ImageNet — adapting their learned features to distinguish between 5 flower classes.

The project was completed as part of **CS672 – Introduction to Deep Learning** at Pace University (Spring 2026).

---

## 🌼 Classes

| Label | Class |
|-------|-------|
| 0 | Daisy |
| 1 | Dandelion |
| 2 | Rose |
| 3 | Sunflower |
| 4 | Tulip |

---

## 🏗️ Models

6 models trained across 3 architectures and 2 frameworks:

| # | Model | Framework | Strategy |
|---|-------|-----------|----------|
| 1 | ResNet50 | TensorFlow | Last 30 layers unfrozen |
| 2 | EfficientNetB0 | TensorFlow | Last 40 layers unfrozen |
| 3 | VGG16 | TensorFlow | Last conv block unfrozen |
| 4 | ResNet50 | PyTorch | layer4 + fc unfrozen |
| 5 | EfficientNetB0 | PyTorch | features.7/8 + classifier unfrozen |
| 6 | VGG16 | PyTorch | Last conv block + classifier unfrozen |

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **PT ResNet50** | **90.19%** | **0.9030** | **0.9019** | **0.9021** |
| TF ResNet50 | 89.07% | 0.8922 | 0.8907 | 0.8908 |
| TF EfficientNetB0 | 88.33% | 0.8843 | 0.8833 | 0.8833 |
| PT VGG16 | 87.04% | 0.8713 | 0.8704 | 0.8704 |
| PT EfficientNetB0 | 86.85% | 0.8689 | 0.8685 | 0.8685 |
| TF VGG16 | 86.20% | 0.8647 | 0.8620 | 0.8622 |

> 🏆 **Best Model:** PyTorch ResNet50 — 90.19% test accuracy

---

## 🗂️ Project Structure

```
flower-classification/
│
├── CS672_Project3_v4_6models.ipynb     # Main training notebook (6 models)
├── Flower_Predictor_PT_ResNet50.ipynb  # Prediction notebook (upload & classify)
│
├── saved_models/
│   ├── tf_resnet50_flowers.keras
│   ├── tf_efficientnetb0_flowers.keras
│   ├── tf_vgg16_flowers.keras
│   ├── pt_resnet50_flowers.pth
│   ├── pt_efficientnetb0_flowers.pth
│   └── pt_vgg16_flowers.pth
│
└── README.md
```

---

## ⚙️ Key Techniques

- **Transfer Learning** — pretrained ImageNet weights adapted to flower classification
- **Partial Fine-tuning** — only top layers unfrozen to preserve universal features while adapting task-specific ones
- **Framework-specific preprocessing** — each architecture gets its own preprocessing (ResNet/VGG: caffe-style BGR mean subtraction, EfficientNet: [-1, 1] scaling, PyTorch: [0, 1] normalization)
- **Data Augmentation** — rotation, shift, shear, zoom, horizontal flip to prevent overfitting on ~4,300 images
- **BatchNormalization + Dropout** in custom classifier heads for training stability and regularization
- **ReduceLROnPlateau** — adaptive learning rate scheduling
- **EarlyStopping** — restores best weights automatically

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/flower-classification.git
cd flower-classification
```

### 2. Install dependencies
```bash
pip install tensorflow torch torchvision kagglehub opencv-python scikit-learn matplotlib seaborn tqdm
```

### 3. Run the training notebook
Open `CS672_Project3_v4_6models.ipynb` in Google Colab or Jupyter and run all cells top to bottom.

> ⚠️ **RAM Note:** Training is memory-intensive. Free Google Colab (12 GB RAM) is sufficient — the notebook uses on-demand preprocessing and garbage collection to stay within limits.

### 4. Predict on your own images
Open `Flower_Predictor_PT_ResNet50.ipynb`, upload the saved `pt_resnet50_flowers.pth`, then upload any flower image to classify it.

---

## 📈 Visualizations Included

- Sample images per class
- Class distribution charts
- Training & validation accuracy/loss curves for all 6 models
- Confusion matrices (raw counts + normalized %)
- Per-class confidence distribution histograms
- Prediction confidence breakdown charts
- Radar chart — multi-metric model comparison
- Best vs. worst model confusion matrix difference
- TensorFlow vs. PyTorch framework comparison
- Per-class accuracy heatmap across all 6 models
- Cross-model misclassification analysis
- Most confused flower pairs (Tulip → Rose identified as universal challenge)
- Model efficiency plot (accuracy vs. trainable parameters)
- Final ranking table

---

## 🔍 Key Findings

- **ResNet50 wins** in both frameworks — skip connections make it the best architecture for transfer learning on this dataset
- **EfficientNetB0 is most efficient** — ~8x fewer parameters than ResNet50 with competitive accuracy, best for deployment on resource-constrained devices
- **Framework matters but not consistently** — PyTorch wins on ResNet50 (+1.11%) and VGG16 (+0.83%), TensorFlow wins on EfficientNetB0 (+1.48%)
- **Dandelion is the easiest class** (8.6% avg error) — distinctive yellow color and circular shape
- **Sunflower is the hardest class** (14.4% avg error) — visual overlap with dandelion color and daisy petal shape
- **Tulip → Rose is the most common mistake** (117 total errors across all models) — shared petal colors and shapes cause universal confusion across all architectures

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-yellow?logo=googlecolab)

---

## 📦 Dataset

**Flowers Recognition** by Alexander Mamaev
- 4,317 images across 5 classes
- ~800 images per class
- Source: [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

---

## 📚 References

- He et al. (2016) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — ResNet
- Tan & Le (2019) — [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) — EfficientNet
- Simonyan & Zisserman (2014) — [Very Deep CNNs for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) — VGG
- Deng et al. (2009) — ImageNet: A Large-Scale Hierarchical Image Database

---

## 👤 Author

**[Your Name]**
Pace University — MS Computer Science
CS672 Introduction to Deep Learning — Spring 2026
