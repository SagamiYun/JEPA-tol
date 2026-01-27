"""
JEPA Encoder 模块

Encoder 是 JEPA 架构的核心组件，负责将输入数据映射到嵌入空间。
根据 LeCun 的论文，Encoder 学习的是输入的"本质"表示，忽略不可预测的细节。
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    """
    JEPA Encoder 基类
    
    Encoder 的职责是将输入 x 映射到嵌入空间，产生表示 s_x。
    这个表示应该捕获输入的"可预测"和"显著"的方面。
    
    核心设计原则（来自 LeCun 论文）：
    1. 表示应该是紧凑的，忽略不可预测的细节
    2. 表示应该对下游任务有用
    3. 表示应该可以被 Predictor 预测
    """
    
    def __init__(self, embed_dim: int = 768):
        """
        初始化 Encoder
        
        Args:
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.embed_dim = embed_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入映射到嵌入空间
        
        Args:
            x: 输入张量
            
        Returns:
            嵌入表示 s_x
        """
        pass
    
    @abstractmethod
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取 patch 级别的嵌入（用于 mask 预测）
        
        Args:
            x: 输入张量
            
        Returns:
            patch 嵌入，形状为 (B, N, D)
        """
        pass


class ViTEncoder(Encoder):
    """
    基于 Vision Transformer 的 Encoder 实现
    
    这是 I-JEPA 中使用的标准 Encoder 架构。
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        初始化 ViT Encoder
        
        Args:
            img_size: 输入图像大小
            patch_size: Patch 大小
            in_channels: 输入通道数
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层比例
            dropout: Dropout 概率
        """
        super().__init__(embed_dim=embed_dim)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回全局表示
        
        Args:
            x: 输入图像，形状为 (B, C, H, W)
            
        Returns:
            全局嵌入，形状为 (B, D)
        """
        patch_embeds = self.get_patch_embeddings(x)
        # 使用平均池化获取全局表示
        return patch_embeds.mean(dim=1)
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取 patch 级别的嵌入
        
        Args:
            x: 输入图像，形状为 (B, C, H, W)
            
        Returns:
            patch 嵌入，形状为 (B, N, D)
        """
        # Patch embedding: (B, C, H, W) -> (B, D, H', W') -> (B, N, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer 编码
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
