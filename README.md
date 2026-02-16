🍃 Mango Leaf Disease Detection and Classification

Deep learning–based hybrid framework for automated detection and classification of mango leaf diseases using transfer learning and ensemble machine learning.

📌 Project Overview

Mango cultivation is highly affected by leaf diseases such as Anthracnose, Powdery Mildew, and Bacterial Canker, which reduce crop yield and quality. Manual disease identification is time-consuming and error-prone due to similar visual symptoms across diseases.

This project proposes an automated deep learning–based system that detects and classifies mango leaf diseases from images using transfer learning and hybrid ensemble classification.

The proposed framework combines multiple CNN feature extractors with machine learning meta-classifiers to achieve high accuracy and robust disease recognition.

🎯 Objectives

Detect mango leaf diseases from images

Classify leaves into 7 disease categories + healthy

Improve accuracy using hybrid deep learning + ML ensemble

Provide automated decision support for agriculture

📂 Dataset

Dataset: MangoLeafBD (Kaggle)

4,000 RGB leaf images

8 classes (7 diseases + healthy)

500 images per class

Captured under natural orchard conditions

Classes:

Anthracnose

Bacterial Canker

Cutting Weevil

Die Back

Gall Midge

Powdery Mildew

Sooty Mould

Healthy

⚙️ Methodology
1️⃣ Preprocessing

CLAHE contrast enhancement

HSV color segmentation

Leaf region extraction

Image normalization

2️⃣ Deep Feature Extraction

Transfer learning models:

EfficientNetB0

ResNet50

DenseNet121

Extracted deep features are concatenated into a fused representation.

3️⃣ Hybrid Classification

Meta-classifier ensemble:

XGBoost

Random Forest

Weighted prediction improves generalization and reduces overfitting.

🧠 Model Architecture

Input Image
→ CLAHE + HSV Segmentation
→ CNN Feature Extraction (3 models)
→ Feature Fusion
→ XGBoost + RandomForest
→ Disease Prediction

📊 Results

Accuracy: 97.5%

High Precision, Recall, F1-score

Robust classification across 8 classes

Improved performance over single CNN models

DenseNet121 showed best individual performance.
Hybrid classifier further improved final accuracy.

🖥️ Implementation

Language: Python 3.10

Libraries:

TensorFlow / Keras

OpenCV

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib

Hardware:

Intel i5

16GB RAM

NVIDIA GPU

📁 Repository Structure
mango-leaf-disease-detection/
│
├── dataset_sample/
├── baseline_models/
├── proposed_model/
├── results/
├── report/
│   └── Mango_Leaf_Disease_Report.pdf
├── README.md

🚀 Applications

Precision agriculture

Automated crop disease monitoring

Smart farming systems

Early disease diagnosis tools

📚 Publication / Report

This work was developed as part of:

Technical Answers for Real World Problems (TARP)
Integrated M.Tech CSE (Data Science)
Vellore Institute of Technology, Vellore
October 2025

👩‍💻 Authors
Boomika S
Integrated M.Tech CSE (Data Science)
VIT Vellore

📜 License

This project is released under the MIT License.
