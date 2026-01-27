"""
JEPA 表示提取工具

提供从输入数据提取 JEPA 嵌入表示的功能。
可用于下游任务如分类、检索等。
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jepa_tol.models.base import BaseModel, ModelRegistry


class RepresentationExtractor:
    """
    JEPA 表示提取器
    
    从输入数据中提取 JEPA 嵌入表示。
    
    使用示例：
        >>> extractor = RepresentationExtractor.from_pretrained("ijepa")
        >>> embeddings = extractor.extract(images)
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        初始化提取器
        
        Args:
            model: JEPA 模型
            device: 运行设备
            normalize: 是否 L2 归一化嵌入
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.normalize = normalize
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "ijepa",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        **model_kwargs,
    ) -> "RepresentationExtractor":
        """
        从预训练模型创建提取器
        
        Args:
            model_name: 模型名称
            checkpoint_path: 权重路径
            device: 运行设备
            normalize: 是否归一化
            **model_kwargs: 模型参数
            
        Returns:
            RepresentationExtractor 实例
        """
        model = ModelRegistry.create(
            model_name,
            pretrained=checkpoint_path is not None,
            checkpoint_path=checkpoint_path,
            **model_kwargs,
        )
        return cls(model=model, device=device, normalize=normalize)
    
    @torch.no_grad()
    def extract(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        提取嵌入表示
        
        Args:
            inputs: 输入数据（单个 tensor 或 tensor 列表）
            batch_size: 批次大小
            
        Returns:
            嵌入表示，形状为 (N, D)
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        
        all_embeddings = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(self.device)
            embeddings = self.model.encode(batch)
            
            if self.normalize:
                embeddings = nn.functional.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        从 DataLoader 提取嵌入
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数
            
        Returns:
            嵌入表示
        """
        all_embeddings = []
        total = 0
        
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # 假设第一个元素是输入
            
            batch = batch.to(self.device)
            embeddings = self.model.encode(batch)
            
            if self.normalize:
                embeddings = nn.functional.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings.cpu())
            total += len(batch)
            
            if max_samples and total >= max_samples:
                break
        
        result = torch.cat(all_embeddings, dim=0)
        if max_samples:
            result = result[:max_samples]
        
        return result
    
    def save_embeddings(
        self,
        embeddings: torch.Tensor,
        path: Union[str, Path],
    ):
        """保存嵌入到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, path)
    
    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> torch.Tensor:
        """从文件加载嵌入"""
        return torch.load(path, map_location="cpu")
