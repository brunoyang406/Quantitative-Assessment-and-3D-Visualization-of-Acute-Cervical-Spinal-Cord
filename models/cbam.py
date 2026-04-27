"""
CBAM: Convolutional Block Attention Module (3D version)
Based on the original CBAM implementation, adapted for 3D convolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    """Basic 3D Convolution with optional BatchNorm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3d, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    """Flatten module for 3D tensors"""
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_3d(tensor):
    """Log-sum-exp pooling for 3D tensors"""
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate3D(nn.Module):
    """3D Channel Attention Gate"""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # Global average pooling: (B, C, D, H, W) -> (B, C, 1, 1, 1)
                avg_pool = F.adaptive_avg_pool3d(x, 1)
                # (B, C, 1, 1, 1) -> (B, C) via Flatten in MLP
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                # Global max pooling: (B, C, D, H, W) -> (B, C, 1, 1, 1)
                max_pool = F.adaptive_max_pool3d(x, 1)
                # Squeeze spatial dimensions: (B, C, 1, 1, 1) -> (B, C)
                max_pool = max_pool.view(max_pool.size(0), -1)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                # Lp pooling (not commonly used, but kept for compatibility)
                # For 3D: use adaptive pooling as approximation
                lp_pool = F.adaptive_avg_pool3d(x, 1)  # Approximation
                # Squeeze spatial dimensions: (B, C, 1, 1, 1) -> (B, C)
                lp_pool = lp_pool.view(lp_pool.size(0), -1)
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # Log-sum-exp pooling
                lse_pool = logsumexp_3d(x)
                channel_att_raw = self.mlp(lse_pool)
            else:
                continue

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Reshape back to (B, C, 1, 1, 1) for broadcasting
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool3D(nn.Module):
    """Channel pooling for 3D spatial attention"""
    def forward(self, x):
        # x: (B, C, D, H, W)
        # Return: (B, 2, D, H, W) - max and mean across channels
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), 
            dim=1
        )


class SpatialGate3D(nn.Module):
    """3D Spatial Attention Gate"""
    def __init__(self, kernel_size=7):
        super(SpatialGate3D, self).__init__()
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3d(
            2, 1, kernel_size, 
            stride=1, 
            padding=(kernel_size-1) // 2, 
            relu=False
        )
    
    def forward(self, x):
        x_compress = self.compress(x)  # (B, 2, D, H, W)
        x_out = self.spatial(x_compress)  # (B, 1, D, H, W)
        scale = torch.sigmoid(x_out)  # Broadcasting
        return x * scale


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module
    
    Based on the original CBAM paper, adapted for 3D convolutions.
    Supports multiple pooling types and includes BatchNorm in spatial attention.
    
    Args:
        gate_channels: Number of input channels
        reduction_ratio: Channel reduction ratio (default: 16)
        pool_types: List of pooling types ['avg', 'max', 'lp', 'lse'] (default: ['avg', 'max'])
        no_spatial: If True, only use channel attention (default: False)
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()
    
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# Alias for backward compatibility
ChannelAttention3D = ChannelGate3D
SpatialAttention3D = SpatialGate3D

