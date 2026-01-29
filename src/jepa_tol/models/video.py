"""
V-JEPA 视频模型

Video Joint Embedding Predictive Architecture
基于 Meta AI 的 V-JEPA 论文实现。

核心思想：
- 在视频的时空嵌入空间中进行预测
- 使用 tube masking 策略 (时空连续的 mask 块)
- 从可见的时空区域预测被 mask 的区域
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_tol.core.temporal_encoder import TemporalEncoder
from jepa_tol.core.predictor import TransformerPredictor
from jepa_tol.models.base import BaseModel, ModelRegistry


@dataclass
class VJEPAConfig:
    """V-JEPA 配置"""
    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    embed_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    predictor_dim: int = 384
    predictor_depth: int = 6
    predictor_heads: int = 6
    attention_type: str = "divided"
    momentum: float = 0.996


class TubeMaskGenerator:
    """
    Tube Masking 生成器
    
    生成时空连续的 mask 块，用于 V-JEPA 训练。
    """
    
    def __init__(
        self,
        num_frames: int,
        num_patches_per_frame: int,
        mask_ratio: float = 0.9,
        tube_length: int = 4,
    ):
        """
        Args:
            num_frames: 帧数
            num_patches_per_frame: 每帧的 patch 数
            mask_ratio: mask 比例
            tube_length: tube 在时间维度的长度
        """
        self.num_frames = num_frames
        self.num_patches = num_patches_per_frame
        self.mask_ratio = mask_ratio
        self.tube_length = tube_length
        
        # 计算空间网格
        self.grid_size = int(num_patches_per_frame ** 0.5)
    
    def __call__(
        self, 
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 mask
        
        Returns:
            context_mask: (B, T*N) True 表示可见
            target_mask: (B, T*N) True 表示需要预测
        """
        T, N = self.num_frames, self.num_patches
        total_tokens = T * N
        
        # 计算需要 mask 的 token 数
        num_mask = int(total_tokens * self.mask_ratio)
        num_visible = total_tokens - num_mask
        
        context_mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # 随机选择 tube 的起始位置
            num_tubes = num_mask // (self.tube_length * 4)  # 每个 tube 大约 4x4 空间块
            
            mask_indices = set()
            
            for _ in range(num_tubes):
                # 随机起始帧
                t_start = torch.randint(0, max(1, T - self.tube_length + 1), (1,)).item()
                t_end = min(t_start + self.tube_length, T)
                
                # 随机空间位置 (2x2 或 4x4 块)
                block_size = min(2, self.grid_size)
                h_start = torch.randint(0, self.grid_size - block_size + 1, (1,)).item()
                w_start = torch.randint(0, self.grid_size - block_size + 1, (1,)).item()
                
                # 添加 tube 中的所有 token
                for t in range(t_start, t_end):
                    for h in range(h_start, h_start + block_size):
                        for w in range(w_start, w_start + block_size):
                            idx = t * N + h * self.grid_size + w
                            if idx < total_tokens:
                                mask_indices.add(idx)
            
            # 如果 mask 不够，随机补充
            all_indices = set(range(total_tokens))
            remaining = all_indices - mask_indices
            
            if len(mask_indices) < num_mask:
                additional = torch.randperm(len(remaining))[:num_mask - len(mask_indices)]
                remaining_list = list(remaining)
                for i in additional:
                    mask_indices.add(remaining_list[i])
            
            # 截断到目标数量
            mask_list = list(mask_indices)[:num_mask]
            visible_list = list(all_indices - set(mask_list))[:num_visible]
            
            context_mask[b, visible_list] = True
            target_mask[b, mask_list[:num_mask // 2]] = True  # 只预测部分 mask 区域
        
        return context_mask, target_mask


@ModelRegistry.register("vjepa")
class VJEPAModel(BaseModel):
    """
    V-JEPA 视频模型
    
    实现基于视频的 JEPA 架构：
    - 使用时空 Transformer 编码视频
    - 采用 tube masking 策略
    - 在嵌入空间预测被 mask 的时空区域
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        attention_type: str = "divided",
        momentum: float = 0.996,
    ):
        super().__init__(embed_dim=embed_dim)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        self.momentum = momentum
        
        # Context Encoder
        self.context_encoder = TemporalEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            max_frames=num_frames,
            attention_type=attention_type,
        )
        
        # Target Encoder (EMA)
        self.target_encoder = TemporalEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            max_frames=num_frames,
            attention_type=attention_type,
        )
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self._copy_weights()
        
        # Predictor
        self.predictor = TransformerPredictor(
            num_patches=num_frames * self.num_patches,
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
        
        # Mask 生成器
        self.mask_generator = TubeMaskGenerator(
            num_frames=num_frames,
            num_patches_per_frame=self.num_patches,
        )
    
    def _copy_weights(self):
        for t_param, c_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            t_param.data.copy_(c_param.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        for t_param, c_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            t_param.data.mul_(self.momentum).add_(c_param.data, alpha=1 - self.momentum)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码视频，获取全局表示
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            
        Returns:
            全局嵌入 (B, D)
        """
        return self.context_encoder(x)
    
    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码视频，获取帧级别表示
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            
        Returns:
            帧嵌入 (B, T, D)
        """
        return self.context_encoder.get_frame_embeddings(x)
    
    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入视频 (B, T, C, H, W)
            context_mask: 可见区域 mask (B, T*N)
            target_mask: 预测区域 mask (B, T*N)
            
        Returns:
            包含预测和目标嵌入的字典
        """
        B = x.shape[0]
        device = x.device
        
        # 生成 mask
        if context_mask is None or target_mask is None:
            context_mask, target_mask = self.mask_generator(B, device)
        
        # Context 编码
        context_patches = self.context_encoder.get_patch_embeddings(x)  # (B, T*N, D)
        
        # 选择 context patches
        D = context_patches.shape[-1]
        max_ctx = context_mask.sum(dim=1).max().item()
        context_idx = context_mask.float().topk(max_ctx, dim=1).indices
        context_embeddings = torch.gather(
            context_patches, 1,
            context_idx.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Target indices
        max_tgt = target_mask.sum(dim=1).max().item()
        target_idx = target_mask.float().topk(max_tgt, dim=1).indices
        
        # 预测
        pred_embeddings = self.predictor(context_embeddings, target_indices=target_idx)
        
        # Target 编码
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
            "context_mask": context_mask,
            "target_mask": target_mask,
        }
    
    def compute_loss(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 V-JEPA loss"""
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)
        return F.mse_loss(pred_norm, target_norm)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> "VJEPAModel":
        model = cls(**kwargs)
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    @classmethod
    def from_config(cls, config: VJEPAConfig) -> "VJEPAModel":
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            predictor_dim=config.predictor_dim,
            predictor_depth=config.predictor_depth,
            predictor_heads=config.predictor_heads,
            attention_type=config.attention_type,
            momentum=config.momentum,
        )
