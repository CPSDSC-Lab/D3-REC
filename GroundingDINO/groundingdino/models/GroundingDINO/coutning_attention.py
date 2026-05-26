import copy
from typing import List
import json
import torch
import torch.nn.functional as F
from torch import nn
from utils.util import attention_map_save, channel_map_save
## TODO: a module that make use of the regression density information to enhance the original feature.
### part1: spatial attention for feature enhance
### part2: channel attention for feature enhance
### part3: combination of these two modules

class DensityAwareEnhance(nn.Module):
    def __init__(self, channel_attention=False):
        super().__init__()
        self.spatial_module = SpatialModule()
        self.channel_attention = channel_attention
        if self.channel_attention:
            self.channel_module = ChannelModule()

    def forward(self, vision_features, density_features):
        vision_features = self.spatial_module(vision_features, density_features)
        if self.channel_attention:
            vision_features = self.channel_module(vision_features)
        return vision_features

class SpatialModule(nn.Module):
    def __init__(self):
        super().__init__()
        # get max pool and avg pool of every layer
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 
        self.spatial_attention = nn.ModuleList([nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) for i in range(4)])

        self.sigmoid = nn.Sigmoid()

    def forward(self, vision_features, density_features):
        for i, vision_feature in enumerate(vision_features):
            b, c, h, w = vision_feature.shape
            density_max = self.max_pool(density_features[i].permute(0,2,3,1).view(b, -1, c)).view(b, h, w, -1)
            density_mean = self.avg_pool(density_features[i].permute(0,2,3,1).view(b, -1, c)).view(b, h, w, -1)
            spatial_feature = torch.cat([density_max, density_mean], dim=-1)
            spatial_feature = spatial_feature.permute(0,3,1,2)
            attention_spatial = self.sigmoid(self.spatial_attention[i](spatial_feature))
            vision_features[i] = vision_features[i] * attention_spatial
            # for j in range(attention_spatial.shape[0]):
            #     attention_map_save(attention_spatial[j,0,...], f'exp/visualization_experiment/feature/SDensityGD/visual_features/{i}_{j}_attention.jpg')
        return vision_features
    
class ChannelModule(nn.Module):
    def __init__(self):
        super().__init__()
        ## Channel attention map
        self.max_pool2d = nn.AdaptiveMaxPool2d((1,1))
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1,1))

        self.mlp = MLP(256, 256, 256, num_layers=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vision_features):
        for i, vision_feature in enumerate(vision_features):
            max_x = self.mlp(self.max_pool2d(vision_feature).squeeze())
            avg_x = self.mlp(self.avg_pool2d(vision_feature).squeeze())
            attention_channel = self.sigmoid(max_x + avg_x)
            # for j in range(attention_channel.shape[0]):
            #     channel_map_save(attention_channel[j,...].unsqueeze(0).cpu(), f'exp/visualization_experiment/feature/SDensityGD/visual_features/{i}_{j}_channel_attention.jpg')
            vision_features[i] = attention_channel.unsqueeze(-1).unsqueeze(-1) * vision_features[i]
        return vision_features
    
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DualDensityAwareEnhance(nn.Module):
    """
    双密度感知增强模块
    类别密度 → 空间注意力（WHERE: 对象在哪里）
    属性密度 → 通道注意力（WHICH: 哪些通道响应属性）
    """
    def __init__(self, feature_channels_list=None, hidden_channels=64):
        super().__init__()
        # 空间注意力模块（由类别密度驱动）- 复用现有SpatialModule
        self.cls_spatial_module = SpatialModule()
        # 通道注意力模块（由属性密度驱动）- 复用现有ChannelModule
        self.attr_channel_module = ChannelModule()
    
    def forward(self, vision_features, cls_density_features, attr_density_features):
        """
        Args:
            vision_features: List of 4 multi-scale visual features
            cls_density_features: List of 4 class density hidden features
            attr_density_features: List of 4 attribute density hidden features
        Returns:
            enhanced_features: List of 4 enhanced visual features
        """
        # Step 1: 类别密度驱动的空间注意力
        spatially_enhanced = self.cls_spatial_module(vision_features, cls_density_features)
        # Step 2: 属性密度驱动的通道注意力
        fully_enhanced = self.attr_channel_module(spatially_enhanced)
        return fully_enhanced


if __name__ == '__main__':
    model = SpatialModule()
    x1 = torch.rand(8,256,256,256)
    x2 = torch.rand(8,256,128,128)
    x3 = torch.rand(8,256,64,64)
    x4 = torch.rand(8,256,32,32)
    x = [x1,x2,x3,x4]
    y1 = torch.rand(8,256,256,256)
    y2 = torch.rand(8,256,128,128)
    y3 = torch.rand(8,256,64,64)
    y4 = torch.rand(8,256,32,32)
    y = [y1,y2,y3,y4]
    score = model(x, y)



    # density_feature, score = model(x1)
    # label = torch.tensor([0,1,2,3,0,1,2,3])
    # cri = nn.CrossEntropyLoss()
    # loss = cri(score, label)
    # pred_index = torch.argmax(score,dim=1)
    # accuracy = torch.sum(pred_index == label)

    # print(score)
    
