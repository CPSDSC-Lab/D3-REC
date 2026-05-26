import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeDecompositionModule(nn.Module):
    """
    属性分解模块 (ADM)
    将BERT编码的文本特征分层为类别(category)、属性(attribute)、上下文(context)三类表示。
    
    输入:
        text_features: (B, L, d_model) - BERT编码并投影后的文本特征
        text_mask: (B, L) - 文本有效token掩码 (True=有效)
    
    输出:
        category_repr: (B, 1, d_model) - 类别聚合表示
        attr_repr: (B, K, d_model) - K个属性slot表示
        cat_weights: (B, L) - 每个token的类别权重
        attr_weights: (B, L) - 每个token的属性权重
        token_logits: (B, L, 3) - token分类logits (用于辅助损失)
    """
    def __init__(self, d_model=256, num_attr_slots=4, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_attr_slots = num_attr_slots
        
        # Token-level 分类器：预测每个token属于 {category=0, attribute=1, context=2}
        self.token_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
        # Category Aggregator：使用加权平均
        # (直接用 cat_weights 加权聚合)
        
        # Attribute Aggregator：使用 Slot Attention
        self.attr_slots = nn.Parameter(torch.randn(num_attr_slots, d_model) * 0.02)
        self.attribute_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.attr_norm = nn.LayerNorm(d_model)
        
    def forward(self, text_features, text_mask):
        """
        Args:
            text_features: (B, L, d_model)
            text_mask: (B, L) bool, True=valid token
        Returns:
            category_repr: (B, 1, d_model)
            attr_repr: (B, num_attr_slots, d_model)
            cat_weights: (B, L)
            attr_weights: (B, L) 
            token_logits: (B, L, 3)
        """
        B, L, D = text_features.shape
        
        # 1. Token分类（软分类，可微分）
        token_logits = self.token_classifier(text_features)  # (B, L, 3)
        
        # 应用text_mask：无效token的logits设为极小值
        if text_mask is not None:
            token_logits = token_logits.masked_fill(~text_mask.unsqueeze(-1), -1e9)
        
        token_probs = F.softmax(token_logits, dim=-1)  # (B, L, 3)
        
        cat_weights = token_probs[:, :, 0]   # (B, L) 类别权重
        attr_weights = token_probs[:, :, 1]  # (B, L) 属性权重
        
        # 2. 类别聚合（加权平均）
        cat_weights_norm = cat_weights / (cat_weights.sum(dim=-1, keepdim=True) + 1e-8)
        category_repr = torch.bmm(cat_weights_norm.unsqueeze(1), text_features)  # (B, 1, D)
        
        # 3. 属性聚合（Slot Attention）
        attr_queries = self.attr_slots.unsqueeze(1).expand(-1, B, -1)  # (K, B, D)
        text_kv = text_features.transpose(0, 1)  # (L, B, D)
        
        key_padding_mask = ~text_mask if text_mask is not None else None
        
        attr_repr, _ = self.attribute_attn(
            query=attr_queries,      # (K, B, D)
            key=text_kv,             # (L, B, D)
            value=text_kv,           # (L, B, D)
            key_padding_mask=key_padding_mask  # (B, L)
        )  # (K, B, D)
        
        attr_repr = self.attr_norm(attr_repr)
        attr_repr = attr_repr.transpose(0, 1)  # (B, K, D)
        
        return category_repr, attr_repr, cat_weights, attr_weights, token_logits
