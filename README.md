# 🌱 AgriVision-Edge: Tomato Plant Disease Diagnostics

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</div>

<br>

> An edge-ready deep learning framework using U-2-Net and EfficientNetB0 for high-accuracy (95.94%) tomato plant disease detection. Designed as the computer vision engine for **Agricultural Digital Twins**.

---


## 🎯 Overview
Traditional crop disease identification relies on manual visual inspection, which is slow, subjective, and difficult to scale. This project automates the diagnosis of 9 distinct tomato plant diseases (plus a healthy baseline) using convolutional neural networks. 

Instead of relying on heavy server-side models, this framework achieves **95.94% overall accuracy utilizing an ultra-lightweight EfficientNetB0 architecture (~4.0M parameters)**. This massive reduction in computational weight makes it ideal for deployment on RAM-limited embedded edge devices (e.g., Raspberry Pi / Jetson Nano) inside active greenhouses.

---

## 🚀 Key Features & Engineering
- **Asynchronous GPU Segmentation:** Utilizes ONNX (`CUDAExecutionProvider`) to run a U-2-Net salient object detection model. Multi-threading separates disk I/O from inference to eliminate bottlenecks.
- **Biologically-Accurate Feature Enhancement:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) *strictly* to the Luminance (L) channel (CIELAB color space), amplifying necrotic lesions without distorting true biological colors.
- **Dynamic Data Engineering:** Processes data out-of-core (`tf.data.Dataset` with `AUTOTUNE`) and mathematically penalizes the loss function to counteract natural dataset class imbalances.
- **Precision Transfer Learning:** Implements a two-phase training protocol—frozen base extraction followed by a micro-learning rate ($1 \times 10^{-5}$) fine-tuning of the top 30 layers to prevent catastrophic forgetting.

---

## 📸 Visual Pipeline & Results

### 1. Preprocessing: Segmentation & CLAHE
By forcing the background to absolute black (`[0,0,0]`), the network computes zero gradient updates for non-plant regions, physically forcing it to learn only from the leaf morphology.
<div align="center">
  <!-- REPLACE THE SRC LINK BELOW WITH YOUR ACTUAL SAVED IMAGE URL/PATH -->
  <img src="docs/preprocessing_sample.png" alt="Raw vs Segmented vs CLAHE" width="800">
  <p><i>Left: Raw Image | Middle: U-2-Net Masked | Right: CLAHE Enhanced (L-Channel)</i></p>
</div>

### 2. Final Diagnostic Performance
The model was evaluated against a strictly isolated test dataset. EfficientNetB0 vastly outperformed ResNet50 in both accuracy and parameter efficiency.
<div align="center">
  <!-- REPLACE THE SRC LINK BELOW WITH YOUR ACTUAL SAVED IMAGE URL/PATH -->
  <img src="docs/confusion_matrix.png" alt="EfficientNetB0 Confusion Matrix" width="600">
  <p><i>Confusion Matrix of the Fine-Tuned EfficientNetB0 Architecture</i></p>
</div>

| Architecture | Test Accuracy | Test Loss | Parameters | Status |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet-50** | 94.44% | 0.1922 | ~23.5M | Eliminated |
| **EfficientNet-B0** | **95.94%** | **0.1189** | **~4.0M** | **Deployed** |

