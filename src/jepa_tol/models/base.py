"""
JEPA 模型基类和注册机制

提供统一的模型接口和注册表，便于管理不同的 JEPA 变体。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn


class ModelRegistry:
    """
    模型注册表
    
    用于管理和获取不同的 JEPA 模型实现。
    """
    
    _registry: Dict[str, Type["BaseModel"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册模型类"""
        def decorator(model_class: Type["BaseModel"]):
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type["BaseModel"]:
        """获取已注册的模型类"""
        if name not in cls._registry:
            raise ValueError(
                f"模型 '{name}' 未注册。可用模型: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_models(cls) -> list[str]:
        """列出所有已注册的模型"""
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls, 
        name: str, 
        pretrained: bool = False,
        **kwargs
    ) -> "BaseModel":
        """创建模型实例"""
        model_class = cls.get(name)
        return model_class.from_pretrained(**kwargs) if pretrained else model_class(**kwargs)


class BaseModel(ABC, nn.Module):
    """
    JEPA 模型基类
    
    所有 JEPA 模型变体都应该继承这个类。
    """
    
    def __init__(self, embed_dim: int = 768):
        """
        初始化模型
        
        Args:
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.embed_dim = embed_dim
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入，获取嵌入表示
        
        Args:
            x: 输入数据
            
        Returns:
            嵌入表示
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据
            **kwargs: 额外参数
            
        Returns:
            包含各种输出的字典
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint_path: Optional[str] = None, **kwargs) -> "BaseModel":
        """
        从预训练权重加载模型
        
        Args:
            checkpoint_path: 权重文件路径
            **kwargs: 额外参数
            
        Returns:
            加载权重后的模型实例
        """
        pass
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
