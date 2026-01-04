# 🌿 Plant Disease Classification with Explainable Deep Learning

This project implements an end-to-end image classification pipeline for **plant disease detection** using convolutional neural networks, with a strong emphasis on **model benchmarking** and **explainability** using Grad-CAM.

The goal is not only to achieve high classification accuracy, but also to **understand *why* a model makes its predictions**, which is critical for real-world deployment in agriculture and decision-support systems.

---

## 📌 Problem Statement

Plant diseases can significantly impact crop yield and food security. Manual disease identification is time-consuming and requires expert knowledge.

This project addresses the problem by:
- Automatically classifying plant leaf images into disease categories.
- Comparing a **Baseline CNN** with a **Transfer Learning model (ResNet50)**.
- Applying **Grad-CAM** to visually interpret model decisions.

---

## 📊 Dataset

- **Dataset:** PlantVillage
- **Type:** Multi-class image classification
- **Input:** RGB images of plant leaves
- **Classes:** Multiple crop–disease combinations (pepper, potato, tomato, healthy vs diseased)
- **Splitting:** Data was split into **train / validation / test** sets using **stratified sampling** to preserve class distributions.

---

## 🧠 Models Implemented

### 1️⃣ Baseline CNN (From Scratch)
A custom convolutional neural network trained without pretrained weights to establish a realistic performance baseline.
* **Test Accuracy:** 0.8608

### 2️⃣ ResNet50 with Transfer Learning (ImageNet)
A ResNet50 backbone pretrained on ImageNet with a custom classification head. 
* **Design:** Frozen convolutional backbone, Global Average Pooling, and a fully connected classifier.
* **Test Accuracy:** 0.9616

**Verdict:** Transfer learning provided a **substantial performance gain** under identical evaluation conditions.

---

## 🔍 Explainability with Grad-CAM

High accuracy alone is not sufficient for trustable ML systems. To address this, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to visualize where models focus their attention.
Grad-CAM visualizations were generated using standalone scripts to avoid notebook GPU instability.


### 🧪 Error Analysis Workflow
Grad-CAM was applied to both correct and incorrect predictions to understand model behavior:

**Correct Predictions:**
- Activation concentrated on lesion or infected regions.
- Minimal background attention.

**Incorrect Predictions:**
- Attention spread over background or leaf edges.
- Confusion between visually similar diseases (e.g., early vs late blight).

---

## 🔁 Reproducibility & Execution Flow

This project follows a clear separation between **experimentation**, **execution**, and **analysis** to ensure stability and reproducibility.

- **Notebooks** were used for exploration, prototyping, and result interpretation.
- **Standalone scripts** were used for Grad-CAM generation to avoid notebook GPU instability.
- **Final artifacts** (trained models, metrics, and visualizations) are stored under `results/` and treated as frozen outputs.

### Execution Order

1. Dataset exploration and visualization  
   → `notebooks/01_data_exploration.ipynb`

2. Baseline CNN training and evaluation  
   → `notebooks/02_baseline_cnn.ipynb`

3. Error analysis and explainability documentation  
   → `notebooks/03_error_analysis_and_gradcam.ipynb`

4. Transfer learning with ResNet50  
   → `notebooks/04_transfer_learning_resnet50.ipynb`

5. Stable Grad-CAM generation (outside notebooks)  
   ```bash
   python scripts/run_gradcam.py
   python scripts/run_gradcam_resnet.py
   
---

## ⚙️ Tech Stack

* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib
* **Machine Learning Utilities:** Scikit-learn

---

## 📂 Project Structure

```text
cnn-image-analysis/
├── data/
│   └── raw/PlantVillage/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_error_analysis_and_gradcam.ipynb
│   └── 04_transfer_learning_resnet50.ipynb
├── scripts/
│   ├── run_gradcam.py
│   └── run_gradcam_resnet.py
├── src/
│   ├── data_loader.py
│   ├── models.py
│   └── train.py
├── results/
│   ├── metrics/
│   ├── plots/
│   └── gradcam_resnet/
└── README.md
