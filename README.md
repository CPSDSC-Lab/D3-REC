# D3-REC

This repository contains the inference code and evaluation scripts for our paper:

> **D3-REC: Decoupled Density Guidance with Attribute Decomposition and Diffusion Prior for Referring Expression Counting**

---

## Overview

**Referring Expression Counting (REC)** estimates the number of objects described by a natural-language query in an image. Existing contextual attribute density (CAD) methods usually rely on a single density map derived from the CLS token to guide both object localization and attribute filtering, introducing semantic mismatch between *where* objects are and *which* attributes they satisfy.

**D3-REC** addresses this through three coordinated improvements:

- **DDG** (Decoupled Density Guidance): Separates density estimation into two complementary paths:
  - **Class-density path**: focuses on object-center localization (WHERE).
  - **Attribute-density path**: highlights attribute-relevant regions (WHICH).
  The two density cues are injected into the decoder differently: class density drives spatial attention, while attribute density drives channel-wise discrimination.

- **ADM** (Attribute Decomposition Module): Models the linguistic structure of referring expressions by softly assigning text tokens to **category**, **attribute**, and **context** roles via a lightweight token classifier. Category representation is obtained by weighted average pooling, and attribute-aware token weighting emphasizes tokens most relevant to attribute cues.

- **SDPR** (Stable Diffusion Prior Refinement): Leverages cross-attention maps from a pretrained Stable Diffusion UNet as auxiliary density priors during training. A gated refinement mechanism combines the diffusion prior with density features, and KL divergence encourages alignment between the refined density and the diffusion attention map. **No additional inference-time overhead** is introduced.

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

We evaluate on **REC-8K** and **FSC-147** datasets. Please follow the official repositories to download and unzip the datasets:

- [REC-8k](https://github.com/sydai/referring-expression-counting)
- [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

**Density maps:** For FSC-147, we use the official density maps directly. For REC8K, we generate density maps using a fixed kernel size. You can download the generated density maps from [this link](https://pan.baidu.com/s/10PjtyFNUpBuDdBun1SEINw?pwd=8wa3).

### ATTR Subset

To evaluate attribute discrimination directly, we construct an **ATTR subset** from the REC-8K test split. Expressions are grouped by image, object class, and attribute type, retaining only groups with at least two attribute values. This yields contrastive pairs such as *red car* vs. *blue car* within the same image. The subset covers six attribute types:

| Type       | Groups | Entries | Pairs |
|------------|--------|---------|-------|
| Action     | 299    | 601     | 305   |
| Age        | 191    | 438     | 303   |
| Color      | 122    | 553     | 1,125 |
| Gender     | 201    | 402     | 201   |
| Location   | 81     | 188     | 133   |
| Orientation| 276    | 552     | 276   |
| **Total**  | **1,170** | **2,734** | **2,343** |

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
| ADM | `attribute_decomposition_module.py` | Token-role decomposition (category / attribute / context) with Slot Attention aggregation |
| DDG | `context_density_module.py` | Decoupled class-density and attribute-density guidance modules |
| SDPR | `diffusion_density_refinement.py` | Stable Diffusion cross-attention prior refinement with gated fusion |
| Dual-Head Regressor | `counter.py` | Shared backbone with task-specific class-density and attribute-density heads |
| Density-Aware Attention | `coutning_attention.py` | Spatial attention from class density and channel attention from attribute density |
| Query Enhancement | `query_enhanced_module.py` | Dual-path linguistic query enhancement (text + class density + attribute density) |
| Main Model | `groundingdino_counting_guide.py` | Full D3-REC model integrating ADM into forward pass |
| Transformer Core | `transformer_counting_guide.py` | Transformer decoder with DDG/ADM/SDPR integration |

Evaluation logic and loss functions are in `utils/`:

- `utils/tester_rec.py` — REC8K evaluation (dynamic threshold, Support Memory, HSV pre-filtering, Hungarian matching)
- `utils/tester_fsc.py` — FSC-147 evaluation (density-guided fusion & top-k selection)
- `utils/criterion.py` — Loss functions (`loss_adm`, `loss_attr_density`, `loss_sd_kl`, `loss_count_consistency`)
- `utils/support_memory.py` — Adaptive threshold prior retrieval at inference time

---

## Evaluation Metrics

### Standard REC Metrics
- **MAE** / **RMSE**: Counting accuracy
- **Precision** / **Recall** / **F1**: Localization quality

### Attribute Discrimination Metrics (ATTR Subset)
- **AD-MAE** (Attribute Discrimination MAE): Accuracy of predicted count gaps between competing attribute queries
- **APDR** (Attribute Pair Discrimination Rate): Percentage of pairs where the predicted count relation matches ground truth
- **ACR** (Attribute Confusion Rate): Complementary to APDR (`ACR = 100 - APDR`)


