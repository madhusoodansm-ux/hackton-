"""
U-Net with Attention Mechanisms - Hybrid Approach
Balances CNN efficiency with attention mechanisms for better feature selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class AttentionBlock(nn.Module):
    """Attention Block combining Channel and Spatial Attention"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    """Convolution Block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetAttention(nn.Module):
    """U-Net with Attention Mechanisms"""
    
    def __init__(self, in_channels=4, out_channels=4, base_filters=64, depth=5, 
                 attention_blocks=None, dropout=0.2):
        super(UNetAttention, self).__init__()
        
        if attention_blocks is None:
            attention_blocks = [2, 3, 4]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.attention_blocks = attention_blocks
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attention_encoder = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoder.append(ConvBlock(in_ch, out_ch))
            
            # Add attention if specified
            if i in attention_blocks:
                self.attention_encoder.append(AttentionBlock(out_ch))
            else:
                self.attention_encoder.append(nn.Identity())
            
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        
        # Bottleneck with attention
        self.bottleneck = ConvBlock(base_filters * (2 ** (depth - 1)), base_filters * (2 ** depth))
        self.bottleneck_attention = AttentionBlock(base_filters * (2 ** depth))
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.attention_decoder = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upsamples.append(nn.ConvTranspose2d(out_ch * 2, out_ch, 2, 2))
            self.decoder.append(ConvBlock(out_ch * 2, out_ch))
            
            # Add attention in decoder
            if i in attention_blocks:
                self.attention_decoder.append(AttentionBlock(out_ch))
            else:
                self.attention_decoder.append(nn.Identity())
        
        # Output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x):
        # Encoder with skip connections and attention
        encoder_features = []
        for i in range(self.depth):
            x = self.encoder[i](x)
            x = self.attention_encoder[i](x)
            encoder_features.append(x)
            x = self.pools[i](x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.bottleneck_attention(x)
        
        # Decoder with skip connections and attention
        for i in range(self.depth - 1, -1, -1):
            x = self.upsamples[self.depth - 1 - i](x)
            skip = encoder_features[i]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[self.depth - 1 - i](x)
            x = self.attention_decoder[self.depth - 1 - i](x)
        
        # Output
        x = self.final_conv(x)
        return x