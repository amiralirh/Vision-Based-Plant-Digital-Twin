# 🌿 Intelligent Agricultural Digital Twin System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


## 📖 Project Overview

The **Intelligent Agricultural Digital Twin System** is a vision-based cyber-physical framework designed for precision agriculture. This repository houses the core implementation of a plant Digital Twin that bridges the gap between traditional farming and autonomous smart greenhouses. By combining deep learning image segmentation, robust disease classification, and kinematic state estimation via Extended Kalman Filtering (EKF), the system dynamically adapts to plant growth phases and biological stresses while rejecting environmental noise and sensor faults. 

## ✨ Key Features

*   **Macroscopic Canopy Tracking:** High-precision 2D segmentation of the plant's structural growth using a specialized U-Net architecture.
*   **Microscopic Pathology Detection:** Fast and accurate plant disease classification utilizing an optimized EfficientNetB0 network.
*   **Explainable AI (XAI) Telemetry:** A transparent auditing dashboard generating automated telemetry logs, Grad-CAM heatmaps, and probabilistic uncertainty maps.
*   **Fault-Tolerant State Estimation:** Sustained kinematic tracking and stability during visual hardware blackouts, achieved through deterministic mathematical filtration.
*   **Decision-Making & Control:** Features **active environmental regulation** and **majority voting logic** to ensure safe, stable, and autonomous management of greenhouse actuators.

## 🏗️ System Architecture

The pipeline is modularly designed into distinct subsystems that feed into a central processing core:

1.  **Subsystem 1: Disease Diagnosis (Classification)**
    *   Processes macro image sequences to detect microscopic biological stresses. Powered by a fine-tuned convolutional neural network (EfficientNetB0), this layer isolates infectious anomalies with high diagnostic confidence.
2.  **Subsystem 2: Growth & Canopy Tracking (Segmentation)**
    *   Analyzes overhead visual data using a U-Net model. It calculates the plant's precise area and shape, establishing the foundational physical metrics for the Digital Twin.
3.  **Data Fusion & State Estimation**
    *   The raw outputs from the perception networks are inherently noisy. Integration using **Extended Kalman Filtering (EKF)** within the Digital Twin framework fuses these signals, extracts the hidden growth velocity, and maintains a continuous state prediction even during temporary sensor dropouts.

## 🛠️ Tech Stack

*   **Language:** Python 3.8+
*   **Deep Learning:** PyTorch (Segmentation & Classification), TensorFlow (Grad-CAM & XAI mapping)
*   **Computer Vision:** OpenCV, Scikit-Image
*   **Data Processing & Telemetry:** NumPy, Pandas
*   **State Estimation:** SciPy, Custom EKF implementations


# 4. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
