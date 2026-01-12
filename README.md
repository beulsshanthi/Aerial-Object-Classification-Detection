# Aerial-Object-Classification-Detection
Aerial Object Classification and Detection using Deep Learning
📌 Project Overview

This project focuses on binary classification of aerial objects to distinguish between Birds and Drones using deep learning techniques. Multiple models were implemented and evaluated, including a custom CNN and transfer learning approaches using ResNet50, MobileNetV2, and EfficientNetB0. The goal is to compare model performance and identify the most accurate architecture for aerial image classification.

🎯 Objectives

Classify aerial images into Bird and Drone

Compare custom CNN vs pretrained models

Leverage GPU acceleration for faster training

Evaluate models using standard classification metrics

Identify the best-performing model

🧰 Technologies & Libraries

Python

PyTorch

Torchvision

NumPy

Matplotlib

Scikit-learn

🖥️ Hardware Acceleration

The system automatically detects GPU availability and uses CUDA when available to improve training speed.

📂 Dataset Description

The dataset consists of aerial images organized into training, validation, and testing sets using the ImageFolder format.

Classes:

Bird

Drone

Dataset Split:

Training: 2662 images

Validation: 442 images

Testing: 215 images

🔄 Data Preprocessing & Augmentation

To improve generalization, extensive data augmentation was applied during training:

Image resizing to 224×224

Random rotation

Horizontal flipping

Random resized cropping

Color jittering

Tensor conversion

Validation and test images were resized and normalized without augmentation.

🧠 Model Architectures
1️⃣ Custom CNN

A lightweight convolutional neural network built from scratch to establish a baseline.

Key Features:

Two convolutional blocks with ReLU and MaxPooling

Fully connected layers with Dropout

Binary classification using BCEWithLogitsLoss

2️⃣ Transfer Learning Models

Pretrained models were fine-tuned using ImageNet weights by freezing early layers and training deeper layers only.

🔹 ResNet50

Deep residual architecture

Strong feature extraction capability

🔹 MobileNetV2

Lightweight and efficient

Suitable for mobile and edge devices

🔹 EfficientNetB0

Compound scaling strategy

Best balance of accuracy and efficiency

⚙️ Training Strategy

Loss Function: Binary Cross-Entropy with Logits

Optimizer: Adam

Learning Rate: 0.001

Epochs: 5

Batch Size: 32

Evaluation performed after each epoch

📊 Model Performance
Model	Test Accuracy
Custom CNN	80.46%
ResNet50	93.49%
MobileNetV2	91.63%
EfficientNetB0	⭐ 98.14%
📈 Results & Analysis

Transfer learning models significantly outperformed the custom CNN

EfficientNetB0 achieved the highest accuracy with balanced precision and recall

Training and validation loss curves indicate stable convergence

No major overfitting observed

🏆 Best Model

EfficientNetB0 was selected as the best-performing model based on:

Highest test accuracy

Consistent validation performance

Balanced classification metrics

📉 Evaluation Metrics

Each model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Classification Report

🔮 Future Enhancements

Extend to multi-class aerial object classification

Implement object detection (YOLO, Faster R-CNN)

Real-time drone video inference

Deploy using ONNX or TensorRT

Integrate with surveillance systems

👩‍💻 Author

Shanthi Beula
