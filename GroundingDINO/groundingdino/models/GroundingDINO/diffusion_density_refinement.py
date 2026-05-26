"""
Diffusion Prior Density Refinement (SDPR) module.

Refines density estimation using precomputed Stable Diffusion cross-attention maps
as auxiliary supervision signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionDensityRefinement(nn.Module):
    """
    扩散模型先验密度精炼模块 (SDPR)
    
    利用预计算的 Stable Diffusion 交叉注意力图作为辅助信号，
    精炼 DensityRegressor 的密度估计。
    
    仅在训练时使用 SD 注意力图，推理时可跳过（直接返回原始密度）。
    
    输入:
        density_feat: (B, C, H, W) - DensityRegressor的中间共享特征
        sd_attn_map: (B, 1, H_sd, W_sd) - 预计算的SD交叉注意力图，可能为None（推理时）
    
    输出:
        refined_density: (B, 1, H, W) - 精炼后的密度图
        gate_weight: (B, 1, H, W) - 门控权重（仅当有SD注意力图时）
    """
    
    def __init__(self, feat_channels=256, hidden_channels=64):
        """
        Args:
            feat_channels: 输入密度特征的通道数（与 DualHeadDensityRegressor 的 counter_dim 一致）
            hidden_channels: 隐藏层通道数
        """
        super().__init__()
        
        self.feat_channels = feat_channels
        self.hidden_channels = hidden_channels
        
        # SD注意力图对齐层：将单通道注意力图映射到特征空间
        self.sd_align = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 融合层：合并密度特征和SD特征
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_channels + hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )
        
        # 门控机制：控制SD先验的影响力度
        self.gate = nn.Sequential(
            nn.Conv2d(feat_channels + hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 无 SD 注意力图时的 fallback 路径
        # 直接从密度特征生成精炼信号（用于推理时保持一致的计算图）
        self.fallback_head = nn.Sequential(
            nn.Conv2d(feat_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        
        self._weight_init_()
    
    def _weight_init_(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, density_feat, sd_attn_map=None):
        """
        Args:
            density_feat: (B, C, H, W) - DensityRegressor的中间特征
            sd_attn_map: (B, 1, H_sd, W_sd) or None - SD注意力图
        
        Returns:
            refined_density: (B, 1, H, W) - 精炼后的密度输出
            gate_weight: (B, 1, H, W) or None - 门控权重（None when sd_attn_map is None）
        """
        if sd_attn_map is None:
            # 推理时无SD注意力图，使用 fallback 路径
            # 输出接近零的精炼信号，使其对最终密度影响最小
            refined = self.fallback_head(density_feat)
            # 返回零门控，表示不使用 SD 精炼
            return refined, None
        
        # 1. 将SD注意力图resize到与density_feat相同的空间尺寸
        if sd_attn_map.shape[2:] != density_feat.shape[2:]:
            sd_attn_map = F.interpolate(
                sd_attn_map, 
                size=density_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 2. SD特征对齐
        sd_feat = self.sd_align(sd_attn_map)  # (B, hidden_channels, H, W)
        
        # 3. 拼接
        combined = torch.cat([density_feat, sd_feat], dim=1)  # (B, C+hidden, H, W)
        
        # 4. 门控融合
        gate_weight = self.gate(combined)  # (B, 1, H, W) 范围[0,1]
        refined = self.fusion(combined)    # (B, 1, H, W)
        
        return refined, gate_weight
    
    def forward_with_residual(self, density_feat, original_density, sd_attn_map=None):
        """
        带残差连接的前向传播，直接返回精炼后的密度。
        
        Args:
            density_feat: (B, C, H, W) - DensityRegressor的中间特征
            original_density: (B, 1, H, W) - 原始密度图
            sd_attn_map: (B, 1, H_sd, W_sd) or None - SD注意力图
        
        Returns:
            refined_density: (B, 1, H, W) - 精炼后的密度
            gate_weight: (B, 1, H, W) or None - 门控权重
        """
        refined, gate_weight = self.forward(density_feat, sd_attn_map)
        
        if sd_attn_map is not None and gate_weight is not None:
            # 有 SD 注意力图时，使用门控融合
            # 需要将 refined 和 gate_weight resize 到 original_density 的尺寸
            if refined.shape[2:] != original_density.shape[2:]:
                refined = F.interpolate(
                    refined, 
                    size=original_density.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                gate_weight = F.interpolate(
                    gate_weight, 
                    size=original_density.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            final_density = original_density + gate_weight * refined
        else:
            # 无 SD 注意力图时，直接返回原始密度
            final_density = original_density
        
        return final_density, gate_weight
