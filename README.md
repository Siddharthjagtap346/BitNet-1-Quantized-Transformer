# BitNet-1 Quantized Transformer for Protein Function Prediction

A research-oriented implementation of a **BitNet-inspired 1-bit Transformer** designed for **efficient protein sequence modeling and function prediction**.
The system applies **quantization-aware techniques and binarized linear layers** to significantly reduce compute and memory requirements while maintaining meaningful prediction capability.

This project explores how **ultra-low-precision neural networks** can be used for **bioinformatics tasks** and **edge-AI deployments**.

---

# Project Overview

Modern protein modeling systems require extremely large models and high compute.
This project investigates whether **binarized transformer architectures** can perform meaningful biological sequence analysis while remaining lightweight enough for **edge hardware deployment**.

The system:

• Processes **DNA or protein sequences**
• Learns contextual relationships using a **BitNet-style Transformer**
• Predicts **protein function, domains, localization, and GO annotations**
• Runs inference through a **lightweight web dashboard**

The goal is to demonstrate that **quantized transformer architectures can support biological sequence intelligence on low-power devices**.

---

# Key Features

• BitNet-style **1-bit quantized linear layers**
• **Quantization-aware training (QAT)** pipeline
• Multi-task prediction:

* Protein Function
* Protein Domains
* Subcellular Localization
* Gene Ontology (GO) terms
  • **Transformer-based sequence modeling**
  • Efficient inference with **KV caching**
  • Interactive **FastAPI web interface**
  • Designed with **edge AI deployment in mind**

---

# System Architecture

The model follows a **Transformer decoder architecture with BitLinear layers** replacing traditional dense layers.

### Processing Pipeline

Input Sequence
↓
Tokenization
↓
Embedding + Positional Encoding
↓
BitTransformer Blocks

* SubLayer LayerNorm
* BitLinear Q/K/V projections
* Scaled Dot Product Attention
* Residual connections
* BitLinear Feed Forward Network
  ↓
  Multi-Task Prediction Heads
* Function classification
* Domain detection
* Localization prediction
* GO term prediction

---

# Architecture Diagram

<WhatsApp Image 2026-03-14 at 10.02.33 PM.jpeg>



---

# Dataset

The model is trained using data derived from the **UniProt protein database**.

UniProt is one of the largest biological databases containing protein sequences and annotations.

### Dataset Features

| Feature              | Description                       |
| -------------------- | --------------------------------- |
| Sequence             | Amino-acid sequence               |
| Domains              | Conserved functional regions      |
| Motifs               | Functional patterns               |
| Subcellular location | Cellular compartment              |
| Interactions         | Binding partners                  |
| GO Terms             | Standardized function annotations |
| PTMs                 | Post-translational modifications  |

Example:

A protein containing a **kinase domain and ATP binding motif** is likely classified as a **Kinase enzyme**.

---

# Mathematical Model

### BitLinear Layer

The BitLinear layer replaces standard dense layers with **binarized weights**.

Forward pass:

Wb = sign(W − α)

y = (Wb × x) × scale

Where:

• Wb = binarized weight matrix
• α = group mean normalization
• scale = activation scaling factor

---

### Quantization Aware Training

Weights transition progressively during training:

Float32 → Int8 → Binary

Quantization step:

W_int8 = round(W / max(|W|) × 127)

Binary projection:

W_bin = sign(W_int8)

Gradients are approximated using the **Straight-Through Estimator (STE)**.

---

# Model Implementation

The implementation contains several core components.

### BitLinear Layer

Implements:

• Weight binarization
• Group quantization
• Activation quantization
• Straight-Through gradient estimator

---

### Transformer Decoder

Key components:

• Multi-Head Self Attention
• BitLinear projections
• Feed-Forward networks
• Layer normalization
• Residual connections

---

### Multi-Task Heads

The model simultaneously predicts:

• Protein function
• Protein domain presence
• Subcellular localization
• Gene Ontology labels

This allows the model to learn **shared biological representations**.

---

# Training Pipeline

Training includes:

1. Dataset loading from UniProt TSV
2. Tokenization of protein sequences
3. Batch padding and attention masking
4. Transformer forward pass
5. Multi-task loss computation
6. Gradient clipping
7. Learning rate scheduling
8. Periodic checkpoint saving

Loss function combines:

Token prediction loss

* Function classification loss
* Domain classification loss
* Localization loss
* GO prediction loss

---

# Training Logs

Example training output:

```
Epoch 1 | Step 50 | Total: 3.21 | Token: 2.84 | Func: 0.37
Epoch 2 | Step 100 | Total: 2.98 | Token: 2.61 | Func: 0.36
```
<Screenshot 2026-02-14 212437.png>

<Screenshot 2026-02-14 212418.png>


---

# Web Interface

The project includes a **FastAPI based inference dashboard**.

Features:

• Interactive protein sequence input
• Model inference display
• Function prediction visualization
• Domain and GO term predictions
• Token probability visualization
• Latency and model statistics


---

# Output

<WhatsApp Image 2026-02-17 at 10.13.12 PM.jpeg>

<WhatsApp Image 2026-02-17 at 10.13.12 PM (1).jpeg>


---

# Installation

Clone repository

```
git clone https://github.com/Siddharthjagtap346/bitnet-1-quantized-transformer
cd bitnet-1-quantized-transformer
```

Install dependencies

```
pip install torch fastapi uvicorn
```

---

# Training

Run training script

```
python train_full.py
```

Model checkpoints will be saved in

```
/checkpoints
```

---

# Run Inference Dashboard

```
uvicorn webapp:app --reload
```

Open browser:

```
http://127.0.0.1:8000
```

---

# Future Improvements

• Larger protein datasets
• Improved tokenization strategies
• Model distillation for edge deployment
• Hardware acceleration for Raspberry Pi
• Integration with biological evaluation benchmarks

---

# Research Motivation

This project explores the intersection of:

• **Efficient Transformers**
• **Quantized neural networks**
• **Edge AI deployment**
• **Computational biology**

The work demonstrates how **low-precision architectures can enable biological sequence modeling on resource-constrained devices**.

---

# Author

**Siddharth Jagtap**

Computer Engineering Student
Interested in:

• Artificial Intelligence
• Efficient Neural Networks
• Blockchain Systems
• Edge AI

---

# License

MIT License

---

# Acknowledgements

• PyTorch
• UniProt protein database
• Research work on **BitNet and low-precision transformers**
