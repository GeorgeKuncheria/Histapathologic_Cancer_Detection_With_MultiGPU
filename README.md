# Histopathologic Oral Cancer Detection: High-Performance Parallel Computing Report

**Team 7:** George Chempumthara & Revanth Padala  
**Course:** CSYE 7105 High Performance Parallel Machine Learning & AI  
**Institution:** Northeastern University

---

## 📋 Project Overview
This project addresses the computational bottlenecks in medical AI by implementing a high-performance deep learning pipeline for detecting **Oral Squamous Cell Carcinoma (OSCC)**. By leveraging parallel computing techniques on both CPUs (for data preparation) and GPUs (for distributed training), we accelerated the end-to-end pipeline while maintaining diagnostic precision.

### Key Goals
* **Parallel Data Processing:** Accelerate file operations (scanning, label assignment) using Python’s multiprocessing libraries.
* **Parallel CNN Training:** Implement and compare **PyTorch Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)**.
* **Performance Analysis:** Measure speedup, efficiency, and accuracy across 1, 2, and 4 GPU configurations.

---

## 💾 Dataset
* **Source:** Aggregated from Kaggle Histopathologic Cancer Detection datasets.
* **Total Images:** 287,832 histopathologic patches.
* **Class Balance:** Evenly split between **Normal (0)** and **OSCC (1)** (120,283 images each).
* **Preprocessing:** Images resized to 96x96 pixels.

---

## 🛠 Methodology

### 1. Parallel Data Processing (CPU)
We benchmarked three Python frameworks to bypass the Global Interpreter Lock (GIL) and accelerate the scanning of 280k+ files.
* **Standard Multiprocessing (Pool):** **Winner.** Achieved 1.12x speedup with lowest overhead (0.25s execution).
* **Dask & Joblib:** Proved less efficient for this specific lightweight task due to scheduler overhead.

### 2. Distributed Training (GPU)
* **Hardware:** Tesla P100-PCIE-12GB GPUs.
* **Model:** Custom "SimpleCNN" (3 Convolutional Blocks, 256-neuron Dense Layer).
* **Strategies:** Evaluated DDP vs. FSDP scaling efficiency.

---

## 📊 Results & Analysis

### Speedup and Efficiency
* **DDP Performance:** Achieved a **3.7x speedup** on 4 GPUs, reducing training time from ~51.5 mins (1 GPU) to ~13.8 mins.
* **Comparison:** DDP outperformed FSDP in raw throughput for this model size (806s vs 956s on 4 GPUs) because the model fit comfortably in VRAM, making FSDP's sharding overhead unnecessary.

### Accuracy vs. Scale (The Generalization Gap)
* **Best Accuracy:** **94.10%** (FSDP, 1 GPU, Batch 64) and **93.62%** (DDP, 1 GPU, Batch 64).
* **Batch Size Impact:** Smaller batch sizes (64) consistently yielded higher accuracy than larger batches (512).
* **Trade-off:** Scaling to 4 GPUs increased the effective global batch size, causing a slight accuracy drop (~1-2%) known as the "Generalization Gap".

| Configuration | GPU Count | Batch Size | Accuracy | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **Most Accurate** | 1 | 64 | **93.62%** | ~51 min |
| **Fastest** | 4 | 512 | 88.18% | **~13 min** |

---

## 🚀 Recommendations
1.  **For Production (Diagnosis):** Use **1 GPU | Batch 64**. In medical diagnosis, the ~3-4% accuracy advantage is critical for patient safety.
2.  **For Prototyping:** Use **4 GPUs (DDP)**. The speed advantage allows for rapid hyperparameter iteration.
3.  **Framework Choice:** **DDP** is the preferred strategy for this specific CNN architecture over FSDP.

---

## 🔮 Future Scope
* **Automatic Mixed Precision (AMP):** To reduce memory usage by 50% and increase throughput.
* **Advanced Architectures:** Apply FSDP to larger models like Vision Transformers (ViT) or EfficientNet.
* **NVIDIA DALI:** To offload image decoding and augmentation to the GPU.
