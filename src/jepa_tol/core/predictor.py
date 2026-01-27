"""
JEPA Predictor 模块

Predictor 是 JEPA 架构的核心组件，负责在嵌入空间中进行预测。
根据 LeCun 的论文，Predictor 从 context 表示预测 target 表示。

关键区别于生成式模型：
- 生成式模型预测原始像素/token
- JEPA Predictor 预测抽象表示
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class Predictor(ABC, nn.Module):
    """
    JEPA Predictor 基类
    
    Predictor 的职责是从 context 嵌入 s_x 预测 target 嵌入 s_y。
    
    核心设计原则（来自 LeCun 论文）：
    1. 预测在嵌入空间进行，而非原始数据空间
    2. 可以使用额外信息（如 mask 位置）辅助预测
    3. 预测应该是"能量最小化"的
    """
    
    def __init__(self, embed_dim: int = 768, predictor_dim: int = 384):
        """
        初始化 Predictor
        
        Args:
            embed_dim: 输入嵌入维度
            predictor_dim: Predictor 内部维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
    
    @abstractmethod
    def forward(
        self, 
        context_embeddings: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 context 嵌入预测 target 嵌入
        
        Args:
            context_embeddings: Context 嵌入，形状为 (B, N_ctx, D)
            target_positions: Target 位置编码，形状为 (B, N_tgt, D)
            
        Returns:
            预测的 target 嵌入，形状为 (B, N_tgt, D)
        """
        pass


class TransformerPredictor(Predictor):
    """
    基于 Transformer 的 Predictor 实现
    
    这是 I-JEPA 中使用的标准 Predictor 架构。
    使用 cross-attention 从 context 预测 target。
    """
    
    def __init__(
        self,
        num_patches: int = 196,
        embed_dim: int = 768,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        """
        初始化 Transformer Predictor
        
        Args:
            num_patches: Patch 数量（用于位置编码）
            embed_dim: 输入/输出嵌入维度
            predictor_dim: Predictor 内部维度
            depth: Transformer 层数
            num_heads: 注意力头数
        """
        super().__init__(embed_dim=embed_dim, predictor_dim=predictor_dim)
        
        self.num_patches = num_patches
        
        # 输入投影
        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        
        # Mask token（用于表示需要预测的位置）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # 输出投影
        self.output_proj = nn.Linear(predictor_dim, embed_dim)
        
        # Layer Norm
        self.norm = nn.LayerNorm(predictor_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 context 嵌入预测 target 嵌入
        
        Args:
            context_embeddings: Context 嵌入，形状为 (B, N_ctx, D)
            target_positions: Target 位置编码，形状为 (B, N_tgt, D)（可选）
            target_indices: Target patch 的索引，形状为 (B, N_tgt)（可选）
            
        Returns:
            预测的 target 嵌入，形状为 (B, N_tgt, D)
        """
        B = context_embeddings.shape[0]
        
        # 投影 context 到 predictor 维度
        context = self.input_proj(context_embeddings)
        
        # 准备 target queries
        if target_indices is not None:
            N_tgt = target_indices.shape[1]
            # 使用 mask token + 位置编码
            queries = self.mask_token.expand(B, N_tgt, -1)
            # 添加对应位置的位置编码
            pos = self.pos_embed.expand(B, -1, -1)
            target_pos = torch.gather(
                pos, 1, 
                target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
            )
            queries = queries + target_pos
        elif target_positions is not None:
            N_tgt = target_positions.shape[1]
            queries = self.mask_token.expand(B, N_tgt, -1) + target_positions
        else:
            # 默认预测所有位置
            queries = self.mask_token.expand(B, self.num_patches, -1) + self.pos_embed
        
        # Transformer 解码
        output = self.transformer(queries, context)
        output = self.norm(output)
        
        # 投影回原始维度
        output = self.output_proj(output)
        
        return output
