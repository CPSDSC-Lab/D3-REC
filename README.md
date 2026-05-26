# D3-REC

This repository contains the inference code and evaluation scripts for our paper:

> **Exploring Contextual Attribute Density in Referring Expression Counting**  
> [arXiv:2503.12460](https://arxiv.org/abs/2503.12460)

---

## Overview

D3-REC introduces three key modules for referring expression counting:

- **ADM** (Attribute Decomposition Module): Token-level attribute classification + Slot Attention aggregation.
- **DDG** (Dual-path Density Guidance): Context-aware density guidance with multi-scale learning.
- **SDPR** (Stable Diffusion Prior Refinement): Diffusion prior density refinement with gated fusion.

> **Note:** This release provides **inference and evaluation code** for reproducibility. Training scripts are excluded per review policy.

---

## Installation

Our code has been tested on Python 3.10 and PyTorch 2.4.0.

1. Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

We evaluate on **REC8K** and **FSC-147** datasets. Please follow the official repositories to download and unzip the datasets:

- [REC-8k](https://github.com/sydai/referring-expression-counting)
- [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
- [FSC147-D](https://github.com/niki-amini-naieni/countx)

**Density maps:** For FSC-147, we use the official density maps directly. For REC8K, we generate density maps using a fixed kernel size. You can download the generated density maps from [this link](https://pan.baidu.com/s/10PjtyFNUpBuDdBun1SEINw?pwd=8wa3).

---

## Inference

Run the following commands to perform inference on REC8K and FSC-147:

```bash
python test_rec.py
python test_fsc.py
```

---

## Core Modules

Key model implementations are located under `GroundingDINO/groundingdino/models/GroundingDINO/`:

| Module | File | Description |
|--------|------|-------------|
| ADM | `attribute_decomposition_module.py` | Attribute decomposition with Slot Attention |
| DDG | `context_density_module.py` | Context-aware density guidance modules |
| SDPR | `diffusion_density_refinement.py` | Diffusion prior refinement with gated fusion |
| Dual-Head Regressor | `counter.py` | Joint category & attribute density regression |
| Density-Aware Attention | `coutning_attention.py` | Spatial & channel attention from density maps |
| Query Enhancement | `query_enhanced_module.py` | Dual-path linguistic query enhancement |
| Main Model | `groundingdino_counting_guide.py` | Full model integrating ADM into forward pass |
| Transformer Core | `transformer_counting_guide.py` | Transformer with DDG/ADM/SDPR integration |

Evaluation logic and loss functions are in `utils/`:

- `utils/tester_rec.py` — REC8K evaluation (dynamic threshold, Support Memory, HSV pre-filtering, Hungarian matching)
- `utils/tester_fsc.py` — FSC-147 evaluation (density-guided fusion & top-k selection)
- `utils/criterion.py` — Loss functions (`loss_adm`, `loss_attr_density`, `loss_sd_kl`, `loss_count_consistency`)
- `utils/support_memory.py` — Adaptive threshold prior retrieval at inference time

---
