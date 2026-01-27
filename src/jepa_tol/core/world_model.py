"""
JEPA World Model 模块

World Model 是 LeCun 论文中提出的认知架构的核心组件。
它整合了 Encoder 和 Predictor，实现"在嵌入空间中预测世界状态"的能力。

World Model 的关键特性：
1. 可配置性：可以根据不同任务调整预测行为
2. 层次化：支持多尺度的预测（H-JEPA）
3. 能量基础：使用能量函数评估预测质量
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_tol.core.encoder import Encoder, ViTEncoder
from jepa_tol.core.predictor import Predictor, TransformerPredictor


@dataclass
class WorldModelConfig:
    """World Model 配置"""
    
    # Encoder 配置
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    
    # Predictor 配置
    predictor_dim: int = 384
    predictor_depth: int = 6
    predictor_heads: int = 6
    
    # 训练配置
    momentum: float = 0.996  # EMA 动量
    

class WorldModel(nn.Module):
    """
    JEPA World Model
    
    整合 Encoder 和 Predictor，实现完整的 JEPA 架构。
    
    架构组成（来自 LeCun 论文）：
    - Context Encoder: 编码上下文区域
    - Target Encoder: 编码目标区域（使用 EMA 更新）
    - Predictor: 从上下文预测目标表示
    
    训练目标：
    最小化预测表示与目标表示之间的距离（在嵌入空间中）
    """
    
    def __init__(
        self,
        config: Optional[WorldModelConfig] = None,
        context_encoder: Optional[Encoder] = None,
        target_encoder: Optional[Encoder] = None,
        predictor: Optional[Predictor] = None,
    ):
        """
        初始化 World Model
        
        Args:
            config: 配置对象（如果不提供 encoder/predictor）
            context_encoder: Context Encoder（可选）
            target_encoder: Target Encoder（可选，通常是 context_encoder 的 EMA）
            predictor: Predictor（可选）
        """
        super().__init__()
        
        config = config or WorldModelConfig()
        num_patches = (config.img_size // config.patch_size) ** 2
        
        # Context Encoder（在线更新）
        self.context_encoder = context_encoder or ViTEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
        )
        
        # Target Encoder（EMA 更新，不参与梯度计算）
        self.target_encoder = target_encoder or ViTEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
        )
        
        # 冻结 target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # 初始化 target encoder 为 context encoder 的副本
        self._copy_encoder_weights()
        
        # Predictor
        self.predictor = predictor or TransformerPredictor(
            num_patches=num_patches,
            embed_dim=config.embed_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
        )
        
        self.momentum = config.momentum
    
    def _copy_encoder_weights(self):
        """复制 context encoder 权重到 target encoder"""
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            target_param.data.copy_(context_param.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """使用 EMA 更新 target encoder"""
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            target_param.data.mul_(self.momentum).add_(
                context_param.data, alpha=1 - self.momentum
            )
    
    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 (B, C, H, W)
            context_mask: Context patch 的 mask，形状为 (B, N)，True 表示保留
            target_mask: Target patch 的 mask，形状为 (B, N)，True 表示需要预测
            
        Returns:
            pred_embeddings: 预测的 target 嵌入
            target_embeddings: 真实的 target 嵌入（用于计算 loss）
        """
        # 获取所有 patch 嵌入
        context_patches = self.context_encoder.get_patch_embeddings(x)
        
        # 选择 context patches
        B, N, D = context_patches.shape
        context_indices = context_mask.nonzero(as_tuple=True)[1].view(B, -1)
        context_embeddings = torch.gather(
            context_patches, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # 获取 target indices
        target_indices = target_mask.nonzero(as_tuple=True)[1].view(B, -1)
        
        # 预测 target 嵌入
        pred_embeddings = self.predictor(
            context_embeddings,
            target_indices=target_indices,
        )
        
        # 获取真实的 target 嵌入（使用 target encoder）
        with torch.no_grad():
            target_patches = self.target_encoder.get_patch_embeddings(x)
            target_embeddings = torch.gather(
                target_patches, 1,
                target_indices.unsqueeze(-1).expand(-1, -1, D)
            )
        
        return pred_embeddings, target_embeddings
    
    def compute_loss(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 JEPA loss
        
        使用 smooth L1 loss（或 L2 loss）在嵌入空间中计算距离。
        
        Args:
            pred_embeddings: 预测的嵌入
            target_embeddings: 目标嵌入
            
        Returns:
            loss 值
        """
        # 归一化嵌入
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)
        
        # 计算 L2 距离
        loss = F.mse_loss(pred_norm, target_norm)
        
        return loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入，获取全局表示
        
        用于推理阶段，直接获取输入的嵌入表示。
        
        Args:
            x: 输入图像
            
        Returns:
            全局嵌入表示
        """
        return self.context_encoder(x)
    
    def predict(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测指定位置的表示
        
        用于推理阶段，预测被遮挡区域的表示。
        
        Args:
            x: 输入图像
            context_mask: Context mask
            target_indices: 需要预测的 patch 索引
            
        Returns:
            预测的表示
        """
        context_patches = self.context_encoder.get_patch_embeddings(x)
        
        B, N, D = context_patches.shape
        context_indices = context_mask.nonzero(as_tuple=True)[1].view(B, -1)
        context_embeddings = torch.gather(
            context_patches, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        
        return self.predictor(context_embeddings, target_indices=target_indices)
