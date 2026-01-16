# ⚡ Welding Fault Detection using Spiking Neural Networks (SNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch_&_SpikingJelly-orange?style=for-the-badge&logo=pytorch)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry_Pi-green?style=for-the-badge&logo=raspberrypi)
![Status](https://img.shields.io/badge/Status-Prototype_Complete-success?style=for-the-badge)


---

## 📌 Project Overview
This project implements an **Edge AI system** capable of detecting welding faults in real-time by analyzing acoustic signatures (welding sounds). Unlike traditional Deep Learning models (ANN/CNN), this project utilizes **Spiking Neural Networks (SNNs)**—specifically Leaky Integrate-and-Fire (LIF) neurons—to process temporal audio data efficiently .

The system is optimized for deployment on **Embedded Linux (Raspberry Pi)** environments, offering a low-power, high-efficiency solution for **Industry 4.0** Quality Assurance.

---

## 🚀 Key Features
🧠 Neuromorphic Computing:** Uses event-driven SNNs (LIF Neurons) to mimic biological neural processing for high efficiency.
🎧 Acoustic Analysis:** Detects defects based on welding sound variations, eliminating the need for visual inspection or destructive testing.
⚡ Real-Time Inference:** Optimized for low-latency performance on constrained hardware (Raspberry Pi).
📊 High Accuracy:** Achieved **91.48% overall accuracy** on the test dataset.

---

## 🛠️ System Architecture
The data pipeline follows a structured approach:

1.  **Input:** Raw audio captured via USB Microphone (16kHz).
2.  **Preprocessing:** Noise reduction, padding/trimming to 5 seconds, and **MFCC Feature Extraction** (13 coefficients).
3.  **Encoding:** Converting continuous MFCC features into **Spike Trains** using Rate Coding.
4.  **SNN Model:** A multi-layer network utilizing **LIF Nodes** (Leaky Integrate-and-Fire) to process spikes over time.
5.  **Output:** Classification into one of 5 weld categories.

---

## 🔍 Classification Performance

The system achieved an **overall accuracy of 96.36%** using the Intel Robotic Welding Multimodal Dataset.

| Defect Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Porosity** | 0.991 | 0.991 | 0.991 |
| **Good Weld** | 0.966 | 0.973 | 0.970 |
| **Spatter** | 0.967 | 0.990 | 0.979 |
| **Excessive Penetration** | 0.949 | 0.931 | 0.940 |
| **Burn-through** | 0.934 | 0.922 | 0.928 |

- **Macro Average F1:** 0.961 [cite: 261]
- **Weighted Average F1:** 0.963 [cite: 261]

---

## 📂 Repository Structure
```text
├── data/                  # Sample audio files for testing (add small samples here)
├── src/
│   ├── preprocessing.py   # Audio loading, padding, and MFCC extraction logic
│   ├── snn_model.py       # PyTorch/SpikingJelly SNN Architecture definition
│   ├── train_model.py     # Training loop, optimization, and model saving
├── requirements.txt       # List of Python dependencies
└── README.md              # Project Documentation
