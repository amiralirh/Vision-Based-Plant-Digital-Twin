# 🌱 Tomato Plant Disease Detection: Edge-Ready Digital Twin Framework

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An end-to-end, highly optimized Deep Learning pipeline for automated phytopathological diagnosis. Designed specifically for memory-constrained edge devices (Raspberry Pi / Jetson Nano), this model serves as the computer vision engine for **Agricultural Digital Twins** and automated greenhouse monitoring systems.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features & Engineering](#-key-features--engineering)
- [Dataset & Preprocessing Pipeline](#-dataset--preprocessing-pipeline)
- [Model Architecture & Fine-Tuning](#-model-architecture--fine-tuning)
- [Evaluation & Results](#-evaluation--results)
- [Getting Started](#-getting-started)
- [Next Steps: Digital Twin Integration](#-next-steps-digital-twin-integration)
- [Author](#-author)

---

## 🎯 Overview
Traditional crop disease identification relies on manual visual inspection, which is slow and error-prone. This project automates the diagnosis of 9 distinct tomato plant diseases (plus a healthy baseline) using convolutional neural networks. 

Instead of relying on heavy server-side models, this framework achieves **95.94% accuracy using an ultra-lightweight EfficientNetB0 architecture (~4.0M parameters)**, making it vastly superior for RAM-limited embedded edge devices.

---

## 🚀 Key Features & Engineering
- **Asynchronous GPU-Accelerated Segmentation:** Utilizes the ONNX runtime (`CUDAExecutionProvider`) to run a U-2-Net salient object detection model, separating disk I/O from GPU inference to eliminate data bottlenecks.
- **Biologically-Accurate Texture Enhancement:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) *strictly* to the Luminance (L) channel of the CIELAB color space, amplifying necrotic lesions without distorting true biological colors.
- **Dynamic Tensor Batching & Class Weighting:** Processes data out-of-core (`tf.data.Dataset` with `AUTOTUNE`) and mathematically penalizes the loss function to counteract natural dataset imbalances.
- **Precision Transfer Learning:** Implements a two-phase training protocol—frozen base extraction followed by a micro-learning rate ($1 \times 10^{-5}$) fine-tuning of the top 30 layers to prevent catastrophic forgetting.

---

## 🧪 Dataset & Preprocessing Pipeline
The model is trained on a strictly filtered subset of the **PlantVillage** dataset, containing 10 categories (1 Healthy, 9 Diseases). 

To prevent the CNN from learning "background shortcuts," the data undergoes a rigorous preprocessing pipeline before training:
1. **U-2-Net Background Subtraction:** The leaf is perfectly isolated. Background pixels are mathematically forced to `[0, 0, 0]` (absolute black), ensuring zero gradient updates occur for non-plant regions during backpropagation.
2. **CLAHE Feature Amplification:** The isolated leaf is converted to LAB color space. Enhancing only the luminance drastically improves the network's ability to distinguish visually overlapping conditions (e.g., Early Blight vs. Target Spot).

---

## 🧠 Model Architecture & Fine-Tuning
Three base architectures were evaluated for edge viability: MobileNetV2, ResNet50, and EfficientNetB0. 

After initial transient response evaluation, **MobileNetV2 was eliminated**, and the remaining models were advanced to deep fine-tuning.

### Custom Classification Head
```python
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
outputs = Dense(10, activation='softmax')(x)
