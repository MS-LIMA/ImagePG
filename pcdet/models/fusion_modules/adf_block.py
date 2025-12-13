import torch
from torch import nn
from torch.nn import functional as F


class AttentiveFusion(nn.Module):
    # reproduced perspective-channel fusion according to 3D iou-net
    # 3D iou-net: Iou guided 3D object detector for point clouds (arXiv:2004.04962)
    def __init__(self, grid_points=6*6*6, pre_channels=144):
        super().__init__()
        self.pre_channels = pre_channels
        self.pre_points = grid_points

        self.point_wise_attention = nn.Sequential(
            nn.Linear(in_features=self.pre_points, out_features=self.pre_points), 
            nn.Linear(in_features=self.pre_points, out_features=self.pre_points),
            nn.ReLU(), 
        )
        self.channel_wise_attention = nn.Sequential(
            nn.Linear(in_features=self.pre_channels, out_features=self.pre_channels), 
            nn.Linear(in_features=self.pre_channels, out_features=self.pre_channels),
            nn.ReLU(), 
        )
   
    def forward(self, features):
        # features [B, P, C]
        point_features = F.max_pool2d(features, kernel_size=[1, features.size(2)]).squeeze(-1) # [B, P]
        channel_features = F.max_pool2d(features, kernel_size=[features.size(1), 1]).squeeze(1) # [B, C]

        point_attention = self.point_wise_attention(point_features).unsqueeze(-1)
        channel_attention = self.channel_wise_attention(channel_features).unsqueeze(1)
        
        attention = point_attention * channel_attention
        attention = F.sigmoid(attention) # [B, P, C]
        
        features = attention * features

        return features
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ADF(nn.Module):
    def __init__(self, planes):
        super(ADF, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x