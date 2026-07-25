"""
U-Net Architecture for Medical Image Segmentation
Fast, efficient baseline model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm and ReLU"""
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


class UNet(nn.Module):
    """U-Net Model for Medical Image Segmentation"""
    
    def __init__(self, in_channels=4, out_channels=4, base_filters=64, depth=5, dropout=0.2):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoder.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * (2 ** (depth - 1)), base_filters * (2 ** depth))
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upsamples.append(nn.ConvTranspose2d(out_ch * 2, out_ch, 2, 2))
            self.decoder.append(ConvBlock(out_ch * 2, out_ch))
        
        # Output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
        
    def forward(self, x):
        # Encoder with skip connections
        encoder_features = []
        for i in range(self.depth):
            x = self.encoder[i](x)
            encoder_features.append(x)
            x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i in range(self.depth - 1, -1, -1):
            x = self.upsamples[self.depth - 1 - i](x)
            skip = encoder_features[i]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[self.depth - 1 - i](x)
        
        # Output
        x = self.final_conv(x)
        return x