# CAD-GD 三项创新改进：解耦密度引导 / 属性分解模块 / 扩散先验密度精炼

---

## 1. 概述

本文档描述基于 **CAD-GD（CVPR 2025）** 论文的三项系统性创新改进，旨在提升指代表达计数（Referring Expression Counting, REC）任务中密度估计的精度与鲁棒性。

### 核心问题

| 问题 | 原始模型的不足 |
|------|----------------|
| **属性-空间错配** | 原始 CAD 仅使用 `[CLS]` token 生成单一密度图，无法区分"对象在哪里（WHERE）"和"属性特征在哪里（WHICH）"，导致数量-位置联合预测时存在语义混淆 |
| **文本理解不足** | BERT 编码后直接使用全体 token，缺乏对指代表达结构的显式建模（类别词 vs. 属性词 vs. 上下文词） |
| **密度估计缺乏先验** | `DensityRegressor` 完全依靠视觉-文本对齐信号，缺少来自大规模生成模型的强语义先验指导 |

### 三项创新目标

1. **DDG（Decoupled Density Guidance，解耦密度引导）**：将单一密度路径拆分为类别密度路径 + 属性密度路径，分别定位"对象中心"与"属性特征区域"。
2. **ADM（Attribute Decomposition Module，属性感知分层文本编码）**：在 BERT 编码后插入轻量级属性分解层，将文本 token 软分类为 category / attribute / context 三种角色。
3. **SDPR（SD Prior Density Refinement，扩散模型先验密度精炼）**：利用 Stable Diffusion UNet 的跨模态注意力图作为辅助密度先验信号，训练时以 KL 散度损失进行软引导，推理时无需 SD 模型。

---

## 2. 创新点详解

### 2.1 解耦密度引导（Decoupled Density Guidance, DDG）

#### 问题背景

原始 CAD 模型将 BERT 编码的 `[CLS]` token（位置 0）与视觉特征计算相似度，生成单一密度图作为计数引导。然而：
- `[CLS]` token 代表整体语义，其相似度图侧重对象中心区域
- 属性词（如颜色、形状、纹理）对应的特征分布在图像中与对象中心位置未必重合
- 混用单一密度图既引导定位又引导属性过滤，造成语义级别的错配

#### 解决方案：双路径密度估计

**类别密度路径（Class Density）**：使用 `[CLS]` token 计算视觉-语义相似度，定位对象中心区域，生成 `cls_density`。

**属性密度路径（Attribute Density）**：使用 ADM 输出的属性权重（`attr_weights`）对文本 token 加权聚合后，计算视觉-属性相似度，生成 `attr_density`，定位属性特征激活区域。

#### 涉及模块

| 模块类名 | 文件 | 功能 |
|----------|------|------|
| `AttrSCGM` | `context_density_module.py` | 属性相似度计算引导模块，使用 ADM 输出的 `attr_weights` 加权聚合属性 token，与视觉特征计算相似度图 |
| `DualHeadDensityRegressor` | `counter.py` | 双头密度回归器，共享特征提取骨干（继承 `DensityRegressor`），分别为类别路径和属性路径输出密度图，内置可选的 SDPR 模块 |
| `DualDensityAwareEnhance` | `coutning_attention.py` | 双密度感知增强模块，类别密度驱动空间注意力（WHERE），属性密度驱动通道注意力（WHICH） |
| `DualQueryLinguishDensityModule` | `query_enhanced_module.py` | 双路径查询增强，顺序执行文本增强 → 类别密度增强 → 属性密度增强 |

#### DDG 数据流

```
text_dict['encoded_text']
         │
         ├─── [CLS] token ──────────────────── SCGM ──────► cls_sim_maps
         │                                                       │
         └─── ADM attr_weights ──── AttrSCGM ──────────────► attr_sim_maps
                                                                 │
cas_feature + cnn                                                │
         │                                                       │
         └──────────────── DualHeadDensityRegressor ────────────┘
                              │              │
                        cls_density    attr_density
                              │              │
                        DualDensityAwareEnhance
                         空间注意力(WHERE) + 通道注意力(WHICH)
                              │
                         增强后的 cnn 特征
                              │
                     DualQueryLinguishDensityModule
                    文本增强 → cls密度增强 → attr密度增强
                              │
                           解码器
```

---

### 2.2 属性感知分层文本编码（Attribute Decomposition Module, ADM）

#### 问题背景

指代表达通常具有明确的结构，例如：
- **"5 red cars parked on the road"**：`cars`（类别词）、`red`（属性词）、`parked on the road`（上下文词）

原始模型将所有 BERT 编码 token 均等对待，无法区分不同类型词汇对视觉特征的贡献差异，导致属性感知能力弱。

#### 解决方案

在 BERT 编码后（特征投影之后）插入轻量级 `AttributeDecompositionModule`：

1. **Token 分类器**：一个 `Linear → ReLU → Dropout → Linear(3)` 的 MLP，对每个 token 输出三类软概率：
   - `0`：category（类别词）
   - `1`：attribute（属性词）
   - `2`：context（上下文词）

2. **Category Aggregator**：使用类别权重 `cat_weights` 加权平均聚合，输出类别表示 `category_repr`（形状 `B×1×d_model`）。

3. **Attribute Aggregator**：使用 Slot Attention 机制，以 `num_attr_slots` 个可学习 slot 为 Query，BERT 特征为 Key/Value，聚合出 K 个属性 slot 表示 `attr_repr`（形状 `B×K×d_model`）。

4. **attr_weights 输出**：ADM 将属性软概率 `attr_weights`（形状 `B×L`）写入 `text_dict['adm_attr_weights']`，供 DDG 的 `AttrSCGM` 模块使用。

#### 模块接口

**文件**：`attribute_decomposition_module.py`

```python
class AttributeDecompositionModule(nn.Module):
    def __init__(self, d_model=256, num_attr_slots=4, nhead=4, dropout=0.1)
    
    def forward(self, text_features, text_mask):
        # 返回: category_repr, attr_repr, cat_weights, attr_weights, token_logits
```

#### 与 DDG 的联动

```
BERT 编码
    │
    ▼
AttributeDecompositionModule
    │
    ├── category_repr ──► DualQueryLinguishDensityModule（可选）
    ├── attr_weights ───► text_dict['adm_attr_weights']
    │                              │
    │                              ▼
    │                          AttrSCGM (DDG)
    │                          计算属性相似度图
    └── token_logits ──► loss_adm（辅助交叉熵损失）
```

---

### 2.3 扩散模型先验密度精炼（SD Prior Density Refinement, SDPR）

#### 问题背景

`DensityRegressor` 的训练信号完全来源于 GT 密度图和文本-视觉对齐，缺乏对指代对象位置分布的强先验知识。Stable Diffusion UNet 在生成任务中形成了丰富的跨模态注意力模式，其交叉注意力图（cross-attention map）能够反映文本描述在图像空间中的响应区域，与密度图的语义接近。

#### 解决方案

**离线预计算（前置步骤）**：

使用 `precompute_sd_attention.py` 脚本，对 REC-8K 数据集中所有图像-表达式对，利用 SD v1.5 的 UNet 提取跨模态注意力图，聚合多层多头注意力后归一化保存为 `.pt` 文件（形状 `1×H×W`，分辨率默认 64×64）。

**训练时使用（门控融合 + KL 辅助损失）**：

`DiffusionDensityRefinement` 模块嵌入在 `DualHeadDensityRegressor` 中，接收共享特征 `shared_feat` 和 SD 注意力图 `sd_attn_map`：
1. SD 注意力图经过两层卷积对齐到特征空间
2. 与密度特征拼接后，分别生成精炼信号 `refined` 和门控权重 `gate_weight`（范围 `[0,1]`）
3. 最终类别密度：`cls_density = cls_density_base + gate_weight × refined`

同时，`SetCriterion.loss_sd_kl()` 计算预测密度与 SD 注意力图之间的 KL 散度辅助损失，温度参数为 0.1。

**推理时行为**：

`sd_attn_map=None` 时，`DiffusionDensityRefinement` 自动走 `fallback_head` 路径（直接返回小量精炼信号，门控为 None），不改变主路径密度图，**无需加载 SD 模型**。

#### 模块接口

**文件**：`diffusion_density_refinement.py`

```python
class DiffusionDensityRefinement(nn.Module):
    def __init__(self, feat_channels=256, hidden_channels=64)
    
    def forward(self, density_feat, sd_attn_map=None):
        # 返回: refined_density (B,1,H,W), gate_weight (B,1,H,W) or None
    
    def forward_with_residual(self, density_feat, original_density, sd_attn_map=None):
        # 返回: final_density (B,1,H,W), gate_weight
```

---

## 3. 整体架构

### 改进后的完整数据流

```
输入图像                    指代表达文本（如 "5 red cars on the road"）
    │                                       │
    ▼                                       ▼
SwinT 骨干特征提取              BERT 文本编码
    │                                       │
    │                           [可选 ADM] AttributeDecompositionModule
    │                            │   │   │
    │                       cat_repr attr_repr attr_weights → text_dict
    │                                       │
    └─────────── TransformerEncoder (6层 BiAttentionBlock 视觉-语言融合) ────┘
                              │
                         memory (视觉特征) + memory_text (文本特征)
                              │
          ┌───────────────────┴────────────────────┐
          │                                         │
    [CLS] token → SCGM                 ADM attr_weights → AttrSCGM
     cls_sim_maps (4尺度)               attr_sim_maps (4尺度)
          │                                         │
          └─────────────────────────────────────────┘
                              │
                    DualHeadDensityRegressor
                    ├── 共享骨干特征提取 (FPN式解码)
                    ├── 类别路径: cls_sim 调制 → cls_density
                    ├── 属性路径: attr_sim 调制 → attr_density
                    └── [可选 SDPR] DiffusionDensityRefinement
                              │
                    DualDensityAwareEnhance
                    ├── cls_density → 空间注意力 → WHERE引导
                    └── attr_density → 通道注意力 → WHICH引导
                              │
                         增强后的视觉特征
                              │
                    DualQueryLinguishDensityModule
                    ├── 文本增强 (QueryLinguishModule)
                    ├── 类别密度增强 (CrossAttentionLayer)
                    └── 属性密度增强 (CrossAttentionLayer)
                              │
                        Transformer 解码器 (6层)
                              │
                    pred_logits + pred_points
                              │
                      HungarianMatcher 匹配
                              │
                      最终计数与定位结果
```

### 三项创新协同工作

- **ADM → DDG**：ADM 输出的 `attr_weights` 直接提供给 `AttrSCGM`，是 DDG 属性密度路径的语义输入；`token_logits` 用于辅助监督。
- **DDG → SDPR**：`DualHeadDensityRegressor` 内置 `DiffusionDensityRefinement`，SDPR 利用 DDG 的共享骨干特征 `shared_feat` 进行精炼。
- **SDPR → DDG**：精炼后的 `cls_density` 更准确，反馈给 `DualDensityAwareEnhance` 的空间注意力模块，提升视觉特征增强质量。

---

## 4. 文件结构

### 新增文件（3个）

| 文件 | 说明 |
|------|------|
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/attribute_decomposition_module.py` | ADM 模块实现，包含 `AttributeDecompositionModule` 类（Token 分类器 + Category Aggregator + Slot Attention 属性聚合） |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/diffusion_density_refinement.py` | SDPR 模块实现，包含 `DiffusionDensityRefinement` 类（SD 特征对齐层 + 门控融合 + fallback 路径） |
| `CAD-GD/precompute_sd_attention.py` | SD 注意力图离线预计算脚本，包含 `SDAttentionExtractor`、`AttentionMapProcessor`、`CrossAttentionHook` 等工具类 |

### 修改文件（8个）

| 文件 | 修改内容 |
|------|----------|
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/context_density_module.py` | 新增 `AttrSCGM` 类（属性相似度计算引导模块），使用 ADM 输出的属性权重计算视觉-属性相似度图 |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/coutning_attention.py` | 新增 `DualDensityAwareEnhance` 类，组合现有 `SpatialModule`（类别密度驱动）和 `ChannelModule`（属性密度驱动）实现双路增强 |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/counter.py` | 新增 `DualHeadDensityRegressor` 类（继承 `DensityRegressor`），增加属性密度头 `attr_head` 和可选的 `DiffusionDensityRefinement` 子模块 |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/query_enhanced_module.py` | 新增 `DualQueryLinguishDensityModule` 类，实现三阶段查询增强：文本 → 类别密度 → 属性密度 |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/transformer_counting_guide.py` | 在 `Transformer` 类中新增 DDG/ADM/SDPR 开关参数，条件初始化对应子模块，`forward` 方法中增加属性相似度计算、双头回归、双路增强、双路查询增强等逻辑分支；`build_transformer` 中传递创新开关 |
| `CAD-GD/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino_counting_guide.py` | 在主模型 `forward` 中将 DDG/ADM/SDPR 开关从 args 透传至 Transformer；将 `attr_density` 和 ADM 相关输出写入 `outputs` 字典供损失函数使用 |
| `CAD-GD/utils/criterion.py` | 在 `SetCriterion` 中新增 `loss_attr_density()`（DDG 属性密度 MSE 损失）、`loss_adm()`（ADM Token 分类交叉熵损失）、`loss_sd_kl()`（SDPR KL 散度损失）方法及其在 `forward` 中的条件调用 |
| `CAD-GD/train_rec8k.py` | 新增命令行参数 `--use_ddg`、`--use_adm`、`--use_sdpr`、`--lambda_attr`、`--lambda_adm`、`--lambda_sd_kl`、`--sd_attn_dir`、`--num_attr_slots`；将这些参数传递给 `SetCriterion` 和数据加载器 |

---

## 5. 损失函数设计

### 完整损失公式

$$\mathcal{L}_{total} = w_{label} \cdot \mathcal{L}_{label} + w_{point} \cdot \mathcal{L}_{point} + w_{contrast} \cdot \mathcal{L}_{contrast} + \lambda_{attr} \cdot \mathcal{L}_{attr} + \lambda_{adm} \cdot \mathcal{L}_{adm} + \lambda_{kl} \cdot \mathcal{L}_{kl}$$

### 各损失说明

| 损失项 | 权重 | 公式/说明 | 模块来源 |
|--------|------|-----------|----------|
| $\mathcal{L}_{label}$ | `weight_dict['loss_label'] = 5` | Sigmoid Focal Loss（分类损失），α=0.25，γ=2 | 原始 CAD-GD |
| $\mathcal{L}_{point}$ | `weight_dict['loss_point'] = 1` | L1 Loss（点定位损失），通过 HungarianMatcher 匹配 | 原始 CAD-GD |
| $\mathcal{L}_{contrast}$ | `weight_dict['loss_contrast'] = 0.06` | 图像-文本对比损失，正负样本 BCE | 原始 CAD-GD |
| $\mathcal{L}_{attr}$ | `lambda_attr = 1.0`（默认） | 属性密度图 MSE 损失：`MSE(attr_density_pred, density_gt)`，需 GT 密度图；仅 DDG 启用时计算 | DDG 新增 |
| $\mathcal{L}_{adm}$ | `lambda_adm = 0.1`（默认） | Token 分类交叉熵损失：`CrossEntropy(token_logits, labels)`，标签由 `text_subject_mask` 和 `text_attribute_mask` 构建；仅 ADM 启用时计算 | ADM 新增 |
| $\mathcal{L}_{kl}$ | `lambda_sd_kl = 0.5`（默认） | KL 散度损失：`KL(log_softmax(cls_density), softmax(sd_attn_map / 0.1))`，温度参数 τ=0.1；仅 SDPR 启用且 `sd_attn_map` 不为 None 时计算 | SDPR 新增 |

### 推荐权重配置

- **完整模型（DDG + ADM + SDPR）**：`--lambda_attr 1.0 --lambda_adm 0.1 --lambda_sd_kl 0.5`（使用默认值）
- **仅 DDG**：`--lambda_attr 1.0`
- **仅 SDPR**：`--lambda_sd_kl 0.5`（注意：必须同时启用 `--use_ddg` 因为 SDPR 内嵌于 `DualHeadDensityRegressor`）
- **调参建议**：如果 `loss_adm` 过大影响收敛，可降低 `--lambda_adm` 至 `0.05`；如果 `loss_sd_kl` 过大，可降至 `0.1`

---

## 6. 使用指南

### 6.1 环境要求

```
Python >= 3.10
PyTorch >= 2.6
CUDA >= 11.8
diffusers >= 0.20.0    # 用于 SDPR 预计算
transformers >= 4.30.0
tqdm
pillow
```

安装依赖：

```bash
cd /home/charles/Projects/REC/CAD-GD/GroundingDINO
pip install -e .
cd /home/charles/Projects/REC/CAD-GD
pip install -r requirements.txt
# SDPR 额外依赖
pip install diffusers transformers accelerate
```

---

### 6.2 预计算 SD 注意力图（SDPR 前置步骤）

**注意**：仅在启用 `--use_sdpr` 时需要此步骤。

#### 基本命令

```bash
cd /home/charles/Projects/REC/CAD-GD

python precompute_sd_attention.py \
    --sd_model runwayml/stable-diffusion-v1-5 \
    --data_root /home/charles/Projects/REC/shared/data/rec-8k/ \
    --anno_file /home/charles/Projects/REC/referring-expression-counting/anno/annotations.json \
    --output_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --timestep 50 \
    --resolution 64 \
    --device cuda
```

#### 使用 FP32（显存充足时，精度更高）

```bash
python precompute_sd_attention.py \
    --sd_model runwayml/stable-diffusion-v1-5 \
    --data_root /home/charles/Projects/REC/shared/data/rec-8k/ \
    --anno_file /home/charles/Projects/REC/referring-expression-counting/anno/annotations.json \
    --output_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --fp32 \
    --device cuda
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sd_model` | str | `runwayml/stable-diffusion-v1-5` | SD 模型路径或 HuggingFace ID；可指向本地目录 |
| `--data_root` | str | `/home/charles/Projects/REC/shared/data/rec-8k/` | REC-8K 图像目录 |
| `--anno_file` | str | `…/anno/annotations.json` | 标注 JSON 文件路径 |
| `--output_dir` | str | `…/rec-8k-sd-attn/` | 输出目录，`.pt` 文件按 `{image_base}_{expr_index}.pt` 命名 |
| `--timestep` | int | `50` | 噪声时间步（0-1000），较小值保留更多语义信息 |
| `--resolution` | int | `64` | 注意力图输出分辨率（像素） |
| `--batch_size` | int | `1` | 当前仅支持 1 |
| `--device` | str | `cuda` | 运行设备 |
| `--fp32` | flag | 否 | 使用 FP32 精度（默认 FP16） |

#### 预计时间和存储估算

| 数据集规模 | 设备 | 预计时间 | 存储占用（64×64 FP32）|
|-----------|------|----------|----------------------|
| REC-8K 约 ~20K 条表达式 | A100 FP16 | 约 2-4 小时 | ~20K × (64×64×4 bytes) ≈ 320 MB |

脚本支持**断点续处理**：若 `.pt` 文件已存在则自动跳过，可安全中断后重启。

---

### 6.3 训练命令

所有训练命令均在 `CAD-GD/` 目录下运行。

#### Baseline（原始 CAD-GD，无创新）

```bash
cd /home/charles/Projects/REC/CAD-GD

python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/baseline
```

#### 仅 DDG

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/ddg_only \
    --use_ddg \
    --lambda_attr 1.0
```

#### 仅 ADM

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/adm_only \
    --use_adm \
    --num_attr_slots 4 \
    --lambda_adm 0.1
```

#### 仅 SDPR（需先预计算 SD 注意力图，且必须同时启用 DDG）

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/sdpr_only \
    --use_ddg \
    --use_sdpr \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 \
    --lambda_sd_kl 0.5
```

#### DDG + ADM

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/ddg_adm \
    --use_ddg \
    --use_adm \
    --num_attr_slots 4 \
    --lambda_attr 1.0 \
    --lambda_adm 0.1
```

#### DDG + SDPR

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/ddg_sdpr \
    --use_ddg \
    --use_sdpr \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 \
    --lambda_sd_kl 0.5
```

#### 完整模型（DDG + ADM + SDPR）

```bash
python train_rec8k.py \
    --epochs 4 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/full_model \
    --use_ddg \
    --use_adm \
    --use_sdpr \
    --num_attr_slots 4 \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 \
    --lambda_adm 0.1 \
    --lambda_sd_kl 0.5
```

#### 带密度预热的完整模型

```bash
python train_rec8k.py \
    --epochs 6 \
    --batch 1 \
    --lr 1e-5 \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --stats_dir ./exp/cvpr/full_model_warmup \
    --use_ddg \
    --use_adm \
    --use_sdpr \
    --density_warmup \
    --warmup_epoch 1 \
    --warmup_ignore_localization \
    --num_attr_slots 4 \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 \
    --lambda_adm 0.1 \
    --lambda_sd_kl 0.5
```

---

### 6.4 命令行参数说明

#### 三项创新开关

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--use_ddg` | flag | `False` | 启用解耦密度引导（DDG），替换原始单路径密度估计为双路径 |
| `--use_adm` | flag | `False` | 启用属性分解模块（ADM），在 BERT 之后添加 token 软分类层 |
| `--use_sdpr` | flag | `False` | 启用 SD 先验密度精炼（SDPR），需同时启用 `--use_ddg` |

#### 损失权重参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--lambda_attr` | float | `1.0` | 属性密度 MSE 损失权重（DDG），控制 `attr_density` 的监督强度 |
| `--lambda_adm` | float | `0.1` | ADM token 分类辅助损失权重，建议保持较小值避免干扰主任务 |
| `--lambda_sd_kl` | float | `0.5` | SD KL 散度损失权重（SDPR），控制 SD 先验的引导强度 |

#### SD 注意力图参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--sd_attn_dir` | str | `None` | 预计算 SD 注意力图的目录路径；为 `None` 时即使启用 SDPR 也退化为 fallback 路径（无 KL 损失） |

#### ADM 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--num_attr_slots` | int | `4` | ADM 中属性 Slot Attention 的 slot 数量 K，建议范围 2~8 |

#### 原有训练参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--epochs` | int | `4` | 训练总 epoch 数 |
| `--batch` | int | `1` | Batch size |
| `--lr` | float | `1e-5` | 初始学习率 |
| `--weight_decay` | float | `0.0001` | AdamW 权重衰减 |
| `--density_warmup` | flag | `False` | 启用密度预热阶段 |
| `--warmup_epoch` | int | `1` | 密度预热结束的 epoch |
| `--warmup_ignore_localization` | flag | `False` | 预热阶段是否忽略定位损失 |
| `--prompt_detach` | flag | `False` | 是否 detach prompt 特征梯度 |
| `--localization_weight` | float | `1.0` | 定位损失总权重 |
| `--regression_weight` | float | `1.0` | 回归损失权重 |
| `--stats_dir` | str | `./exp/cvpr/rec_8k_debug` | 实验结果保存目录 |
| `--resume` | str | `""` | 断点续训的 checkpoint 路径 |

---

## 7. 消融实验设计

以下 7 组实验用于验证各创新点的独立贡献及组合效果：

| 实验编号 | 配置 | DDG | ADM | SDPR | 对应命令行关键参数 |
|----------|------|-----|-----|------|-------------------|
| Exp-1 | Baseline | ✗ | ✗ | ✗ | 无额外参数 |
| Exp-2 | + DDG | ✓ | ✗ | ✗ | `--use_ddg` |
| Exp-3 | + ADM | ✗ | ✓ | ✗ | `--use_adm` |
| Exp-4 | + SDPR（需 DDG） | ✓ | ✗ | ✓ | `--use_ddg --use_sdpr --sd_attn_dir ...` |
| Exp-5 | + DDG + ADM | ✓ | ✓ | ✗ | `--use_ddg --use_adm` |
| Exp-6 | + DDG + SDPR | ✓ | ✗ | ✓ | `--use_ddg --use_sdpr --sd_attn_dir ...` |
| Exp-7 | 完整模型 | ✓ | ✓ | ✓ | `--use_ddg --use_adm --use_sdpr --sd_attn_dir ...` |

### 各实验的完整命令行

```bash
# Exp-1: Baseline
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp1_baseline \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth

# Exp-2: + DDG
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp2_ddg \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_ddg --lambda_attr 1.0

# Exp-3: + ADM
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp3_adm \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_adm --num_attr_slots 4 --lambda_adm 0.1

# Exp-4: + DDG + SDPR
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp4_ddg_sdpr \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_ddg --use_sdpr \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 --lambda_sd_kl 0.5

# Exp-5: + DDG + ADM
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp5_ddg_adm \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_ddg --use_adm --num_attr_slots 4 \
    --lambda_attr 1.0 --lambda_adm 0.1

# Exp-6: + DDG + SDPR（与 Exp-4 相同，用作对照）
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp6_ddg_sdpr \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_ddg --use_sdpr \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 --lambda_sd_kl 0.5

# Exp-7: 完整模型（DDG + ADM + SDPR）
python train_rec8k.py --epochs 4 --stats_dir ./exp/ablation/exp7_full \
    --config ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model /home/charles/Projects/REC/shared/weights/groundingdino_swint_ogc.pth \
    --use_ddg --use_adm --use_sdpr \
    --num_attr_slots 4 \
    --sd_attn_dir /home/charles/Projects/REC/shared/data/rec-8k-sd-attn/ \
    --lambda_attr 1.0 --lambda_adm 0.1 --lambda_sd_kl 0.5
```

---

## 8. 技术细节

### 8.1 显存估算

| 配置 | 额外显存（相对 Baseline） | 备注 |
|------|--------------------------|------|
| Baseline | - | ~8 GB（A100，batch=1） |
| + DDG | +约 500 MB | 双头密度回归器，`DualDensityAwareEnhance`，`DualQueryLinguishDensityModule` |
| + ADM | +约 100 MB | 轻量级，仅 token 分类器 + Slot Attention |
| + SDPR（训练时） | +约 300 MB | `DiffusionDensityRefinement` 及 SD 注意力图加载，SD 模型不常驻内存（已预计算） |
| 完整模型 | +约 800-900 MB | 总计约 9 GB |

### 8.2 训练策略建议

**密度预热阶段（推荐用于完整模型）**：

启用 `--density_warmup --warmup_epoch 1 --warmup_ignore_localization`，第 0 epoch 仅优化密度回归相关损失，跳过定位损失，使 `DualHeadDensityRegressor` 在进入联合优化前先收敛到合理的密度分布。

**学习率建议**：

- 默认 `lr=1e-5`（AdamW）适合大多数场景
- 若启用 SDPR 且密度图质量不佳，可尝试降低 `--lambda_sd_kl 0.1` 避免 KL 损失主导训练早期

**epoch 数**：

- Baseline / 单项创新：4 epochs 通常足够
- 完整模型（三项创新）：建议 5-6 epochs，使多路径梯度充分收敛

**`--prompt_detach`**：

启用后，密度特征梯度不会反传到 BERT 编码器，可在显存受限或调试阶段使用。默认关闭以允许端到端优化。

### 8.3 推理时的行为

| 创新 | 推理时是否需要额外输入 | 推理时行为 |
|------|------------------------|------------|
| DDG | 否 | `sd_attn_map=None`，使用双路径密度估计，`DualHeadDensityRegressor` 正常运行，`attr_density` 输出但不用于损失计算 |
| ADM | 否 | 正常运行，`attr_weights` 由 ADM 自动计算，`token_logits` 不计算辅助损失 |
| SDPR | 否 | `sd_attn_map=None` 触发 `fallback_head` 路径，精炼信号几乎为零，不改变主路径密度图；无需加载 Stable Diffusion 模型 |

推理时所有三项创新均可正常运行，无需额外数据。

### 8.4 checkpoint 兼容性

启用创新开关后，模型权重包含新增参数（如 `attr_head`、`sdpr.*`、ADM 相关参数）。**已训练的 Baseline checkpoint 不能直接用于有创新开关的模型**，需从头训练或使用 `strict=False` 加载（会有未初始化参数警告）。

断点续训时，checkpoint 会自动恢复 epoch、optimizer、scheduler 及 RNG 状态：

```bash
python train_rec8k.py \
    --resume ./exp/ablation/exp7_full/last_checkpoint.pth \
    --use_ddg --use_adm --use_sdpr \
    # ... 其他参数与原始训练一致 ...
```

---

*文档基于代码库实际实现撰写，所有模块名、参数名均与源代码保持一致。*
