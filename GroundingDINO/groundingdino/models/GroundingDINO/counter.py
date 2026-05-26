
"""
Counter modules.
"""
from torch import nn
import torch
import torch.nn.functional as F

from .diffusion_density_refinement import DiffusionDensityRefinement


# 4 8 16 32 x
class DensityRegressor(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        # 1/32 -> 1/16
        self.conv0 = nn.Sequential(nn.Conv2d(counter_dim * 2, counter_dim, 7, padding=3),
                                   nn.ReLU())
        # 1/16 -> 1/8
        self.conv1 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 5, padding=2),
                                   nn.ReLU())
        # 1/8 -> 1/4
        self.conv2 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/4 -> 1/2
        self.conv3 = nn.Sequential(nn.Conv2d(counter_dim * 3, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/2 -> 1
        # self.up2x = nn.Sequential(
        #                 nn.Conv2d(counter_dim, counter_dim//2, 1),
        #                 nn.ReLU(),)
        self.up2x = nn.Sequential(
                                nn.Conv2d(counter_dim, counter_dim//2, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(counter_dim//2, counter_dim//4, 1),
                                nn.ReLU(),)
        self.conv4 = nn.Sequential(
                                nn.Conv2d(counter_dim//4, 1, 1),
                                nn.ReLU())
        # self.conv4 = nn.Conv2d(counter_dim//2, 1, 1)
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.down2x = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4x = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.down8x = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        
        self._weight_init_()
        
    def forward(self, features, cnns, img_shape = [1000,1000], hidden_output=False):
        ## multi-layer context-aware density feature encode
        # f = features[3] * cnns[3]
        x = torch.cat([features[3], cnns[3]], dim=1)
        x1 = self.conv0(x)

        x = F.interpolate(x1, size = features[2].shape[-2:], mode='bilinear')
        # x = self.pixel_shuffle(x1) ##
        # f = features[2] * cnns[2]
        x = torch.cat([x, features[2], cnns[2]], dim=1)
        # smap = F.interpolate(smaps[0], size = features[2].shape[-2:], mode='bilinear')

        x2 = self.conv1(x)

        x = F.interpolate(x2, size = features[1].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[1], cnns[1]], dim=1)
        
        #smap = F.interpolate(smaps[0], size = features[1].shape[-2:], mode='bilinear')
        # x = x
        x3 = self.conv2(x)

        x = F.interpolate(x3, size = features[0].shape[-2:], mode='bilinear')

        x = torch.cat([x, features[0], cnns[0]], dim=1)
        # x = x
        x4 = self.conv3(x)
        xx4 = x4
        ## down-scale density feature
        xx3 = self.down2x(x4)
        xx2 = self.down4x(x4)
        xx1 = self.down8x(x4)
        
        ## multi-layer context-aware density feature decode
        
        x = self.up2x(x4)

        x = F.interpolate(x, size = img_shape, mode='bilinear')
        
        x = self.conv4(x)
        
        # x = F.sigmoid(x)
        # x = x * 60
        hidden_mode = 'fpn'
        if hidden_output:
            if hidden_mode == 'down':
                return x, [xx4,xx3,xx2,xx1]
            if hidden_mode == 'fpn':
                return x, [x4,x3,x2,x1]
        else:
            return x

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.01, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# # 4 8 16 32 x
# class DensityRegressor(nn.Module):
#     def __init__(self, counter_dim):
#         super().__init__()
#         # 1/32 -> 1/16
#         self.conv0 = nn.Sequential(nn.Conv2d(counter_dim + 1, counter_dim, 7, padding=3),
#                                    nn.ReLU())
#         # 1/16 -> 1/8
#         self.conv1 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 5, padding=2),
#                                    nn.ReLU())
#         # 1/8 -> 1/4
#         self.conv2 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 3, padding=1),
#                                    nn.ReLU())
#         # 1/4 -> 1/2
#         self.conv3 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 3, padding=1),
#                                    nn.ReLU())
#         # 1/2 -> 1
#         self.up2x = nn.Sequential(
#                                 nn.Conv2d(counter_dim, counter_dim//2, 1),
#                                 nn.ReLU())
#         # self.conv4 = nn.Sequential(
#         #                         nn.Conv2d(counter_dim//2, 1, 1),
#         #                         nn.ReLU())
#         self.conv4 = nn.Conv2d(counter_dim//2, 1, 1)
#         # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self._weight_init_()
        
#     def forward(self, features, smaps, img_shape = [1000,1000], hidden_output=False):
#         x = torch.cat([features[3],smaps[3]], dim=1)
#         x1 = self.conv0(x)

#         x = F.interpolate(x1, size = features[2].shape[-2:], mode='bilinear')
#         # x = self.pixel_shuffle(x1) ##
#         x = torch.cat([x, features[2], smaps[2]], dim=1)
#         x2 = self.conv1(x)

#         x = F.interpolate(x2, size = features[1].shape[-2:], mode='bilinear')
#         x = torch.cat([x, features[1], smaps[1]], dim=1)
#         x3 = self.conv2(x)

#         x = F.interpolate(x3, size = features[0].shape[-2:], mode='bilinear')
#         x = torch.cat([x, features[0], smaps[0]], dim=1)
#         x4 = self.conv3(x)
#         x = self.up2x(x4)

#         x = F.interpolate(x, size = img_shape, mode='bilinear')
#         x = self.conv4(x)

#         if hidden_output:
#             return x, [x4,x3,x2,x1]
#         else:
#             return x

#     def _weight_init_(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 # nn.init.kaiming_uniform_(
#                 #         m.weight, 
#                 #         mode='fan_in', 
#                 #         nonlinearity='relu'
#                 #         )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


class DualHeadDensityRegressor(DensityRegressor):
    """
    双头密度回归器：共享特征提取骨干，分别输出类别密度和属性密度
    
    支持可选的 SDPR（扩散先验密度精炼）模块，利用预计算的 Stable Diffusion
    交叉注意力图辅助密度估计。
    """
    def __init__(self, counter_dim, use_sdpr=False, sdpr_hidden=64):
        super().__init__(counter_dim)
        # 属性密度头（新增）- 输入通道数与 up2x 输出一致（counter_dim // 4）
        self.attr_head = nn.Sequential(
            nn.Conv2d(counter_dim // 4, counter_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(counter_dim // 4, 1, 1),
            nn.ReLU()
        )
        self._weight_init_attr_head_()
        
        # SDPR 模块（可选）
        self.use_sdpr = use_sdpr
        if use_sdpr:
            # feat_channels 与 shared_feat 的通道数一致（即 counter_dim）
            self.sdpr = DiffusionDensityRefinement(
                feat_channels=counter_dim,
                hidden_channels=sdpr_hidden
            )
    
    def _weight_init_attr_head_(self):
        for m in self.attr_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.01, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _extract_shared_features(self, features, cnns, img_shape):
        """
        提取共享特征（复用父类的特征提取逻辑）
        Args:
            features: List of CAS features (4 scales)
            cnns: List of visual features (4 scales)
            img_shape: target image shape
        Returns:
            shared_feat: (B, counter_dim, H, W) - 共享特征
            hidden_features: List of multi-scale hidden features
        """
        # 复用父类的多层特征融合逻辑
        x = torch.cat([features[3], cnns[3]], dim=1)
        x1 = self.conv0(x)

        x = F.interpolate(x1, size=features[2].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[2], cnns[2]], dim=1)
        x2 = self.conv1(x)

        x = F.interpolate(x2, size=features[1].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[1], cnns[1]], dim=1)
        x3 = self.conv2(x)

        x = F.interpolate(x3, size=features[0].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[0], cnns[0]], dim=1)
        x4 = self.conv3(x)
        
        # 下采样生成多层隐藏特征
        xx4 = x4
        xx3 = self.down2x(x4)
        xx2 = self.down4x(x4)
        xx1 = self.down8x(x4)
        
        return x4, [x4, x3, x2, x1], [xx4, xx3, xx2, xx1]
    
    def forward(self, features, cnns, cls_sim_maps, attr_sim_maps, img_shape=[1000, 1000], 
                sd_attn_map=None, hidden_output=False):
        """
        Args:
            features: List of CAS features (4 scales)
            cnns: List of visual features (4 scales)
            cls_sim_maps: List of class similarity maps (from SCGM, 4 scales)
            attr_sim_maps: List of attribute similarity maps (from AttrSCGM, 4 scales)
            img_shape: target image shape
            sd_attn_map: (B, 1, H_sd, W_sd) or None - 预计算的SD交叉注意力图（可选）
            hidden_output: whether to output intermediate features
        Returns:
            cls_density: (B, 1, H, W) - 类别密度图
            attr_density: (B, 1, H, W) - 属性密度图
            hidden_feats: optional intermediate features for enhancement
            sd_outputs: optional dict with sd_refined and gate_weight (when use_sdpr=True)
        """
        # 使用共享骨干提取特征
        shared_feat, fpn_features, down_features = self._extract_shared_features(features, cnns, img_shape)
        
        # 类别密度：用类别相似度图调制
        cls_sim_resized = F.interpolate(
            cls_sim_maps[-1].unsqueeze(1) if cls_sim_maps[-1].dim() == 3 else cls_sim_maps[-1],
            size=shared_feat.shape[2:], mode='bilinear', align_corners=False
        )
        cls_modulated = shared_feat * cls_sim_resized + shared_feat
        cls_up = self.up2x(cls_modulated)
        cls_up = F.interpolate(cls_up, size=img_shape, mode='bilinear')
        cls_density = self.conv4(cls_up)
        
        # 属性密度：用属性相似度图调制（detach 共享特征，阻断梯度回流到共享骨干）
        attr_sim_resized = F.interpolate(
            attr_sim_maps[-1].unsqueeze(1) if attr_sim_maps[-1].dim() == 3 else attr_sim_maps[-1],
            size=shared_feat.shape[2:], mode='bilinear', align_corners=False
        )
        attr_modulated = shared_feat.detach() * attr_sim_resized + shared_feat.detach()
        attr_up = self.up2x(attr_modulated)
        attr_up = F.interpolate(attr_up, size=img_shape, mode='bilinear')
        attr_density = self.attr_head(attr_up)
        
        # SDPR 精炼（可选）— 限制门控量级，防止系统性偏差
        sd_refined = None
        gate_weight = None
        if self.use_sdpr:
            sd_refined, gate_weight = self.sdpr(shared_feat, sd_attn_map)
            # 将 SD 精炼结果融合到类别密度
            if sd_attn_map is not None and gate_weight is not None:
                # 将精炼结果 resize 到密度图尺寸
                if sd_refined.shape[2:] != cls_density.shape[2:]:
                    sd_refined = F.interpolate(
                        sd_refined, size=cls_density.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                    gate_weight = F.interpolate(
                        gate_weight, size=cls_density.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                # 限制 gate 最大值为 0.3，防止 SD 先验过度覆盖密度校准
                gate_weight = gate_weight * 0.3
                # 只修正分布形状，不改变总量：对 sd_refined 做 zero-mean 归一化
                sd_refined = sd_refined - sd_refined.mean(dim=[2, 3], keepdim=True)
                cls_density = cls_density + gate_weight * sd_refined
        
        if hidden_output:
            return cls_density, attr_density, fpn_features, {
                'sd_refined': sd_refined,
                'gate_weight': gate_weight,
                'shared_feat': shared_feat
            }
        return cls_density, attr_density


if __name__ == '__main__':
    img_shape = [400, 600]
    encode_vision_feature = torch.rand(2,19947,256)
    spatial_shapes = torch.tensor([[100, 150],
        [ 50,  75],
        [ 25,  38],
        [ 13,  19]])
    N_, S_, C_ = encode_vision_feature.shape
    _cur = 0
    cnn = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        memory_flatten_ = encode_vision_feature[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, -1)
        cnn.append(memory_flatten_.permute(0,3,1,2))
        _cur += H_ * W_
    model = DensityRegressor(counter_dim=256)
    output = model(cnn, img_shape)
    print(1)
