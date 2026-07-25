"""
Vision Transformer for Medical Image Segmentation
State-of-the-art approach with better long-range dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head Self-Attention"""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSegmentation(nn.Module):
    """Vision Transformer for Segmentation"""
    
    def __init__(self, in_channels=4, out_channels=4, image_size=256, patch_size=16, 
                 dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.to_segmentation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_channels * patch_size * patch_size),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     h=image_size // patch_size, w=image_size // patch_size, 
                     p1=patch_size, p2=patch_size, c=out_channels),
        )
    
    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        for transformer_block in self.transformer:
            x = transformer_block(x)
        
        x = self.to_segmentation(x)
        return x