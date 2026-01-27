"""
视觉领域的 JEPA 实现

包含 I-JEPA (Image JEPA) 风格的模型实现。
基于 Meta AI 的 I-JEPA 论文和开源实现。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_tol.core.encoder import ViTEncoder
from jepa_tol.core.predictor import TransformerPredictor
from jepa_tol.models.base import BaseModel, ModelRegistry


class VisionEncoder(nn.Module):
    """
    视觉 Encoder 包装器
    
    提供额外的预处理和后处理功能。
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """获取全局表示"""
        return self.encoder(x)
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """获取 patch 级别的嵌入"""
        return self.encoder.get_patch_embeddings(x)


@ModelRegistry.register("ijepa")
class IJEPAModel(BaseModel):
    """
    I-JEPA (Image JEPA) 模型
    
    实现基于图像的 JEPA 架构，用于自监督视觉表示学习。
    
    核心思想：
    - 随机 mask 图像的一些区域
    - 从未 mask 的区域预测 mask 区域的表示
    - 使用 Target Encoder (EMA) 提供监督信号
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        momentum: float = 0.996,
    ):
        """
        初始化 I-JEPA 模型
        
        Args:
            img_size: 输入图像大小
            patch_size: Patch 大小
            embed_dim: 嵌入维度
            encoder_depth: Encoder 层数
            encoder_heads: Encoder 注意力头数
            predictor_dim: Predictor 内部维度
            predictor_depth: Predictor 层数
            predictor_heads: Predictor 注意力头数
            momentum: EMA 动量
        """
        super().__init__(embed_dim=embed_dim)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.momentum = momentum
        
        # Context Encoder（在线更新）
        self.context_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        
        # Target Encoder（EMA 更新）
        self.target_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        
        # 冻结 target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # 初始化 target encoder
        self._copy_weights()
        
        # Predictor
        self.predictor = TransformerPredictor(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
    
    def _copy_weights(self):
        """复制权重到 target encoder"""
        for t_param, c_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            t_param.data.copy_(c_param.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA 更新 target encoder"""
        for t_param, c_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            t_param.data.mul_(self.momentum).add_(c_param.data, alpha=1 - self.momentum)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入图像"""
        return self.context_encoder(x)
    
    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            context_mask: Context mask (B, N)，True 表示保留
            target_mask: Target mask (B, N)，True 表示需要预测
            
        Returns:
            包含预测和目标嵌入的字典
        """
        B = x.shape[0]
        
        # 如果没有提供 mask，生成随机 mask
        if context_mask is None or target_mask is None:
            context_mask, target_mask = self._generate_masks(B, x.device)
        
        # 编码context
        context_patches = self.context_encoder.get_patch_embeddings(x)
        
        # 选择 context patches
        D = context_patches.shape[-1]
        context_indices = context_mask.nonzero(as_tuple=False)
        # 重塑为 (B, N_ctx)
        context_indices_per_batch = context_mask.sum(dim=1)
        max_ctx = context_indices_per_batch.max().item()
        
        # 简化处理：使用 gather
        context_idx = context_mask.float().topk(max_ctx, dim=1).indices
        context_embeddings = torch.gather(
            context_patches, 1,
            context_idx.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Target indices
        target_idx = target_mask.float().topk(target_mask.sum(dim=1).max().item(), dim=1).indices
        
        # 预测
        pred_embeddings = self.predictor(context_embeddings, target_indices=target_idx)
        
        # 获取 target 嵌入
        with torch.no_grad():
            target_patches = self.target_encoder.get_patch_embeddings(x)
            target_embeddings = torch.gather(
                target_patches, 1,
                target_idx.unsqueeze(-1).expand(-1, -1, D)
            )
        
        return {
            "pred_embeddings": pred_embeddings,
            "target_embeddings": target_embeddings,
            "context_embeddings": context_embeddings,
        }
    
    def _generate_masks(
        self, 
        batch_size: int, 
        device: torch.device,
        context_ratio: float = 0.5,
        target_ratio: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成随机 mask
        
        使用 block masking 策略（类似 I-JEPA）
        """
        N = self.num_patches
        
        # 简化版：随机选择 patches
        context_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        
        n_context = int(N * context_ratio)
        n_target = int(N * target_ratio)
        
        for b in range(batch_size):
            # 随机选择 context
            perm = torch.randperm(N, device=device)
            context_mask[b, perm[:n_context]] = True
            
            # 从剩余位置选择 target
            remaining = perm[n_context:]
            target_mask[b, remaining[:n_target]] = True
        
        return context_mask, target_mask
    
    def compute_loss(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 JEPA loss"""
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)
        return F.mse_loss(pred_norm, target_norm)
    
    @classmethod
    def from_pretrained(
        cls, 
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> "IJEPAModel":
        """从预训练权重加载"""
        model = cls(**kwargs)
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
        
        return model
