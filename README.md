# Histopathologic Oral Cancer Detection: High-Performance Parallel Computing Report

[cite_start]**Team 7:** George Chempumthara & Revanth Padala [cite: 3]  
[cite_start]**Course:** CSYE 7105 High Performance Parallel Machine Learning & AI [cite: 4]  
[cite_start]**Institution:** Northeastern University [cite: 1]

---

## 📋 Project Overview
[cite_start]This project addresses the computational bottlenecks in medical AI by implementing a high-performance deep learning pipeline for detecting **Oral Squamous Cell Carcinoma (OSCC)**[cite: 38, 47]. [cite_start]By leveraging parallel computing techniques on both CPUs (for data preparation) and GPUs (for distributed training), we accelerated the end-to-end pipeline while maintaining diagnostic precision[cite: 46, 49].

### Key Goals
* [cite_start]**Parallel Data Processing:** Accelerate file operations (scanning, label assignment) using Python’s multiprocessing libraries[cite: 50].
* [cite_start]**Parallel CNN Training:** Implement and compare **PyTorch Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)**[cite: 52, 53, 68].
* [cite_start]**Performance Analysis:** Measure speedup, efficiency, and accuracy across 1, 2, and 4 GPU configurations[cite: 131].

---

## 💾 Dataset
* [cite_start]**Source:** Aggregated from Kaggle Histopathologic Cancer Detection datasets[cite: 104].
* [cite_start]**Total Images:** 287,832 histopathologic patches[cite: 108].
* [cite_start]**Class Balance:** Evenly split between **Normal (0)** and **OSCC (1)** (120,283 images each)[cite: 109, 110, 111].
* [cite_start]**Preprocessing:** Images resized to 96x96 pixels[cite: 71].

---

## 🛠 Methodology

### 1. Parallel Data Processing (CPU)
[cite_start]We benchmarked three Python frameworks to bypass the Global Interpreter Lock (GIL) and accelerate the scanning of 280k+ files[cite: 67].
* [cite_start]**Standard Multiprocessing (Pool):** **Winner.** Achieved 1.12x speedup with lowest overhead (0.25s execution)[cite: 313, 335].
* [cite_start]**Dask & Joblib:** Proved less efficient for this specific lightweight task due to scheduler overhead[cite: 336].

### 2. Distributed Training (GPU)
* [cite_start]**Hardware:** Tesla P100-PCIE-12GB GPUs[cite: 121].
* [cite_start]**Model:** Custom "SimpleCNN" (3 Convolutional Blocks, 256-neuron Dense Layer)[cite: 72].
* [cite_start]**Strategies:** Evaluated DDP vs. FSDP scaling efficiency[cite: 68].

---

## 📊 Results & Analysis

### Speedup and Efficiency
* [cite_start]**DDP Performance:** Achieved a **3.7x speedup** on 4 GPUs, reducing training time from ~51.5 mins (1 GPU) to ~13.8 mins[cite: 141, 204].
* [cite_start]**Comparison:** DDP outperformed FSDP in raw throughput for this model size (806s vs 956s on 4 GPUs) because the model fit comfortably in VRAM, making FSDP's sharding overhead unnecessary[cite: 353, 347].

### Accuracy vs. Scale (The Generalization Gap)
* [cite_start]**Best Accuracy:** **94.10%** (FSDP, 1 GPU, Batch 64) and **93.62%** (DDP, 1 GPU, Batch 64)[cite: 186, 271].
* [cite_start]**Batch Size Impact:** Smaller batch sizes (64) consistently yielded higher accuracy than larger batches (512)[cite: 169].
* [cite_start]**Trade-off:** Scaling to 4 GPUs increased the effective global batch size, causing a slight accuracy drop (~1-2%) known as the "Generalization Gap"[cite: 176].

| Configuration | GPU Count | Batch Size | Accuracy | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **Most Accurate** | 1 | 64 | **93.62%** | [cite_start]~51 min [cite: 186] |
| **Fastest** | 4 | 512 | 88.18% | [cite_start]**~13 min** [cite: 216] |

---

## 🚀 Recommendations
1.  **For Production (Diagnosis):** Use **1 GPU | Batch 64**. [cite_start]In medical diagnosis, the ~3-4% accuracy advantage is critical for patient safety[cite: 161, 162].
2.  **For Prototyping:** Use **4 GPUs (DDP)**. [cite_start]The speed advantage allows for rapid hyperparameter iteration[cite: 159].
3.  [cite_start]**Framework Choice:** **DDP** is the preferred strategy for this specific CNN architecture over FSDP[cite: 375].

---

## 🔮 Future Scope
* [cite_start]**Automatic Mixed Precision (AMP):** To reduce memory usage by 50% and increase throughput[cite: 366, 367].
* [cite_start]**Advanced Architectures:** Apply FSDP to larger models like Vision Transformers (ViT) or EfficientNet[cite: 368].
* [cite_start]**NVIDIA DALI:** To offload image decoding and augmentation to the GPU[cite: 370].
