🍃**Mango Leaf Disease Detection and Classification**

Deep learning–based hybrid framework for automated detection and classification of mango leaf diseases using transfer learning and ensemble machine learning.

📌**Project Overview**

Mango cultivation is significantly affected by leaf diseases such as Anthracnose, Powdery Mildew, and Bacterial Canker, which reduce crop yield and quality. Manual identification of these diseases is difficult and time-consuming because many diseases exhibit visually similar symptoms.

This project presents an automated image-based disease detection system that classifies mango leaf diseases using deep learning and hybrid machine learning techniques. The proposed framework combines multiple CNN feature extractors with ensemble meta-classifiers to achieve robust and accurate disease recognition.

🎯 **Objectives**

Detect mango leaf diseases from leaf images

Classify leaves into seven disease categories and healthy class

Improve classification accuracy using hybrid deep learning + ensemble ML

Provide an automated decision-support approach for agricultural monitoring

📂**Dataset**

Dataset: MangoLeafBD (Kaggle)

4,000 RGB images

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

⚙️**Methodology**
1️⃣ **Image Preprocessing**

CLAHE contrast enhancement

HSV-based leaf segmentation

Background removal and leaf region extraction

Image resizing and normalization

2️⃣ **Deep Feature Extraction**

Transfer learning models used:

EfficientNetB0

ResNet50

DenseNet121

Deep features from all models were extracted and concatenated to form a fused feature representation.

3️⃣**Hybrid Classification
**
Ensemble meta-classifier:

XGBoost

Random Forest

Weighted prediction from both classifiers improved generalization and reduced overfitting.

🧠**Model Architecture**
Input Image
→ CLAHE + HSV Segmentation
→ CNN Feature Extraction (EfficientNetB0, ResNet50, DenseNet121)
→ Feature Fusion
→ XGBoost + Random Forest
→ Disease Prediction

📊 **Results**

Accuracy: 97.5%

High Precision, Recall, and F1-score across all classes

Hybrid classifier outperformed individual CNN models

DenseNet121 showed strongest standalone performance

Ensemble classification improved final prediction robustness

🖥️ **Implementation**

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

16 GB RAM

NVIDIA GPU

📁**Repository Structure**
mango-leaf-disease-detection/
│
├── baseline_models/
├── proposed_model/
├── results/
├── report/
│   └── Mango_Leaf_Disease_Report.pdf
├── README.md

🚀**Applications**

Precision agriculture

Automated crop disease monitoring

Smart farming systems

Early disease diagnosis tools

📚**Academic Context**

This work was carried out as part of the course:

Technical Answers for Real World Problems (TARP)
Integrated M.Tech CSE (Data Science)
Vellore Institute of Technology, Vellore
October 2025

👩‍💻**Author**

Boomika S
Integrated M.Tech CSE (Data Science)
VIT Vellore

📜**License**

This project is released under the MIT License.
