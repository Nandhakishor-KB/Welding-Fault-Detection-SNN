# ⚡ Welding Fault Detection using Spiking Neural Networks (SNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch_&_SpikingJelly-orange?style=for-the-badge&logo=pytorch)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry_Pi-green?style=for-the-badge&logo=raspberrypi)
![Status](https://img.shields.io/badge/Status-Prototype_Complete-success?style=for-the-badge)


---

## 📌 Project Overview
This project implements an **Edge AI system** capable of detecting welding faults in real-time by analyzing acoustic signatures (welding sounds). [cite_start]Unlike traditional Deep Learning models (ANN/CNN), this project utilizes **Spiking Neural Networks (SNNs)**—specifically Leaky Integrate-and-Fire (LIF) neurons—to process temporal audio data efficiently [cite: 1432-1436].

[cite_start]The system is optimized for deployment on **Embedded Linux (Raspberry Pi)** environments, offering a low-power, high-efficiency solution for **Industry 4.0** Quality Assurance[cite: 1208, 1224].

---

## 🚀 Key Features
**🧠 Neuromorphic Computing:** Uses event-driven SNNs (LIF Neurons) to mimic biological neural processing for high efficiency[cite: 1203].
* [cite_start]**🎧 Acoustic Analysis:** Detects defects based on welding sound variations, eliminating the need for visual inspection or destructive testing[cite: 1205].
* [cite_start]**⚡ Real-Time Inference:** Optimized for low-latency performance on constrained hardware (Raspberry Pi)[cite: 1208].
* [cite_start]**📊 High Accuracy:** Achieved **91.48% overall accuracy** on the test dataset[cite: 1632].

---

## 🛠️ System Architecture
[cite_start]The data pipeline follows a structured approach [cite: 1399-1427]:

1.  **Input:** Raw audio captured via USB Microphone (16kHz).
2.  [cite_start]**Preprocessing:** Noise reduction, padding/trimming to 5 seconds, and **MFCC Feature Extraction** (13 coefficients) [cite: 1496-1502].
3.  [cite_start]**Encoding:** Converting continuous MFCC features into **Spike Trains** using Rate Coding[cite: 1531].
4.  [cite_start]**SNN Model:** A multi-layer network utilizing **LIF Nodes** (Leaky Integrate-and-Fire) to process spikes over time[cite: 1616].
5.  **Output:** Classification into one of 5 weld categories.

---

## 🔍 Classification Performance
[cite_start]The model was trained and validated on a dataset of welding sounds, identifying the following conditions :

| Defect Class | F1-Score | Description |
| :--- | :--- | :--- |
| **Porosity** | **0.98** | Gas trapped in the weld metal. |
| **Good Weld** | 0.89 | A clean, defect-free weld. |
| **Burn-through** | 0.86 | Excessive heat causing holes in the base metal. |
| **Spatter** | 0.86 | Droplets of molten material splashing. |
| **Excessive Penetration**| 0.84 | Weld metal protruding through the root. |

* **Overall Accuracy:** 91.48%
* **Macro F1-Score:** 0.8850
* **Weighted F1-Score:** 0.89

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
