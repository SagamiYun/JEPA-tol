"""
JEPA 时序编码器模块

用于视频理解的时空编码器，支持：
- 时空自注意力 (Space-Time Attention)
- 分离式时空注意力 (Divided Space-Time Attention)
- 时序位置编码
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_tol.core.encoder import Encoder


class TemporalPositionalEncoding(nn.Module):
    """
    时序位置编码
    
    为视频帧添加时间维度的位置信息。
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_frames: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        
        # 可学习的时序位置编码
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, max_frames, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
    
    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        添加时序位置编码
        
        Args:
            x: 输入张量 (B, T*N, D) 或 (B, T, N, D)
            num_frames: 帧数
            
        Returns:
            添加时序位置编码后的张量
        """
        temporal_pos = self.temporal_embed[:, :num_frames]
        
        if x.dim() == 4:
            # (B, T, N, D)
            B, T, N, D = x.shape
            temporal_pos = temporal_pos.unsqueeze(2)  # (1, T, 1, D)
            x = x + temporal_pos
        else:
            # (B, T*N, D) - 需要先 reshape
            B, TN, D = x.shape
            N = TN // num_frames
            x = x.view(B, num_frames, N, D)
            temporal_pos = temporal_pos.unsqueeze(2)
            x = x + temporal_pos
            x = x.view(B, TN, D)
        
        return self.dropout(x)


class SpaceTimeAttention(nn.Module):
    """
    时空自注意力模块
    
    联合处理空间和时间维度的注意力。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        attention_type: str = "joint",  # "joint" 或 "divided"
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 概率
            attention_type: 注意力类型
                - "joint": 联合时空注意力
                - "divided": 分离式时空注意力 (先空间后时间)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        if attention_type == "joint":
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:  # divided
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if attention_type == "divided" else None
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        num_patches: int,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, T*N, D)
            num_frames: 帧数 T
            num_patches: 每帧的 patch 数 N
            
        Returns:
            注意力输出 (B, T*N, D)
        """
        B, TN, D = x.shape
        
        if self.attention_type == "joint":
            # 联合时空注意力
            residual = x
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = residual + x
        else:
            # 分离式时空注意力
            # 1. 空间注意力 (每帧内部)
            x = x.view(B * num_frames, num_patches, D)
            residual = x
            x = self.norm1(x)
            x, _ = self.spatial_attn(x, x, x)
            x = residual + x
            
            # 2. 时间注意力 (跨帧)
            x = x.view(B, num_frames, num_patches, D)
            x = x.permute(0, 2, 1, 3).reshape(B * num_patches, num_frames, D)
            residual = x
            x = self.norm2(x)
            x, _ = self.temporal_attn(x, x, x)
            x = residual + x
            
            # 恢复形状
            x = x.view(B, num_patches, num_frames, D)
            x = x.permute(0, 2, 1, 3).reshape(B, TN, D)
        
        return x


class TemporalEncoderBlock(nn.Module):
    """
    时序编码器块
    
    包含时空注意力和 FFN。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_type: str = "divided",
    ):
        super().__init__()
        
        self.attn = SpaceTimeAttention(
            embed_dim, num_heads, dropout, attention_type
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        num_patches: int,
    ) -> torch.Tensor:
        x = self.attn(x, num_frames, num_patches)
        residual = x
        x = self.norm(x)
        x = residual + self.mlp(x)
        return x


class TemporalEncoder(Encoder):
    """
    视频时序编码器
    
    扩展 I-JEPA 的 ViT Encoder，添加时序建模能力。
    用于 V-JEPA 的视频理解任务。
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
        max_frames: int = 64,
        dropout: float = 0.0,
        attention_type: str = "divided",
    ):
        """
        初始化时序编码器
        
        Args:
            img_size: 输入图像大小
            patch_size: Patch 大小
            in_channels: 输入通道数
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层比例
            max_frames: 最大帧数
            dropout: Dropout 概率
            attention_type: 注意力类型 ("joint" 或 "divided")
        """
        super().__init__(embed_dim=embed_dim)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.max_frames = max_frames
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 空间位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # 时序位置编码
        self.temporal_pos = TemporalPositionalEncoding(
            embed_dim, max_frames, dropout
        )
        
        # Transformer 块
        self.blocks = nn.ModuleList([
            TemporalEncoderBlock(
                embed_dim, num_heads, mlp_ratio, dropout, attention_type
            )
            for _ in range(depth)
        ])
        
        # 输出 Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回全局表示
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            
        Returns:
            全局嵌入 (B, D)
        """
        patch_embeds = self.get_patch_embeddings(x)
        # 平均池化获取全局表示
        return patch_embeds.mean(dim=1)
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取 patch 级别的嵌入
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            
        Returns:
            patch 嵌入 (B, T*N, D)
        """
        B, T, C, H, W = x.shape
        
        # Patch embedding: 每帧独立处理
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        
        N = x.shape[1]  # patch 数量
        
        # 添加空间位置编码
        x = x + self.pos_embed[:, :N]  # 确保 pos_embed 尺寸匹配
        
        # Reshape 为视频格式
        x = x.view(B, T, N, self.embed_dim)
        
        # 添加时序位置编码 (输入是 4D，输出也是 4D)
        x = self.temporal_pos(x, T)
        
        # 确保是 4D 后再 Reshape 为序列格式
        if x.dim() == 4:
            x = x.view(B, T * N, self.embed_dim)
        
        # Transformer 编码
        for block in self.blocks:
            x = block(x, T, N)
        
        x = self.norm(x)
        
        return x
    
    def get_frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取帧级别的嵌入
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            
        Returns:
            帧嵌入 (B, T, D)
        """
        B, T, C, H, W = x.shape
        
        patch_embeds = self.get_patch_embeddings(x)  # (B, T*N, D)
        
        # Reshape 并平均池化每帧
        N = self.num_patches
        patch_embeds = patch_embeds.view(B, T, N, self.embed_dim)
        frame_embeds = patch_embeds.mean(dim=2)  # (B, T, D)
        
        return frame_embeds
