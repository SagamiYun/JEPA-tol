"""
JEPA 相似度搜索工具

基于 JEPA 嵌入进行语义相似度搜索。
支持构建索引和高效检索。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class SearchResult:
    """搜索结果"""
    indices: torch.Tensor  # 匹配项的索引
    scores: torch.Tensor   # 相似度分数
    embeddings: Optional[torch.Tensor] = None  # 匹配项的嵌入（可选）


class SimilaritySearch:
    """
    基于 JEPA 嵌入的相似度搜索
    
    支持：
    - 构建嵌入索引
    - k-NN 搜索
    - 批量搜索
    
    使用示例：
        >>> search = SimilaritySearch()
        >>> search.build_index(embeddings)
        >>> results = search.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        metric: str = "cosine",
        device: Optional[str] = None,
    ):
        """
        初始化搜索器
        
        Args:
            metric: 相似度度量 ("cosine" 或 "l2")
            device: 运行设备
        """
        self.metric = metric
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.index: Optional[torch.Tensor] = None
        self.metadata: Optional[List] = None  # 可选的元数据
    
    def build_index(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[List] = None,
        normalize: bool = True,
    ):
        """
        构建搜索索引
        
        Args:
            embeddings: 嵌入矩阵，形状为 (N, D)
            metadata: 每个嵌入对应的元数据
            normalize: 是否 L2 归一化
        """
        self.index = embeddings.to(self.device)
        
        if normalize and self.metric == "cosine":
            self.index = F.normalize(self.index, dim=-1)
        
        self.metadata = metadata
    
    def add_to_index(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[List] = None,
        normalize: bool = True,
    ):
        """
        向索引添加新嵌入
        
        Args:
            embeddings: 新嵌入
            metadata: 新元数据
            normalize: 是否归一化
        """
        new_embeddings = embeddings.to(self.device)
        
        if normalize and self.metric == "cosine":
            new_embeddings = F.normalize(new_embeddings, dim=-1)
        
        if self.index is None:
            self.index = new_embeddings
            self.metadata = metadata
        else:
            self.index = torch.cat([self.index, new_embeddings], dim=0)
            if metadata and self.metadata:
                self.metadata.extend(metadata)
    
    @torch.no_grad()
    def search(
        self,
        query: torch.Tensor,
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> SearchResult:
        """
        搜索最相似的项
        
        Args:
            query: 查询嵌入，形状为 (D,) 或 (B, D)
            top_k: 返回的最相似项数量
            return_embeddings: 是否返回匹配项的嵌入
            
        Returns:
            SearchResult 包含索引和分数
        """
        if self.index is None:
            raise ValueError("索引为空，请先调用 build_index()")
        
        # 确保 query 是 2D
        single_query = query.dim() == 1
        if single_query:
            query = query.unsqueeze(0)
        
        query = query.to(self.device)
        
        # 归一化 query
        if self.metric == "cosine":
            query = F.normalize(query, dim=-1)
        
        # 计算相似度
        if self.metric == "cosine":
            scores = torch.mm(query, self.index.T)  # (B, N)
        else:  # L2
            # 计算负 L2 距离（越大越相似）
            scores = -torch.cdist(query, self.index, p=2)
        
        # 获取 top-k
        top_k = min(top_k, self.index.shape[0])
        top_scores, top_indices = scores.topk(top_k, dim=-1)
        
        # 如果是单个查询，去掉 batch 维度
        if single_query:
            top_indices = top_indices.squeeze(0)
            top_scores = top_scores.squeeze(0)
        
        result_embeddings = None
        if return_embeddings:
            if single_query:
                result_embeddings = self.index[top_indices]
            else:
                # 批量获取
                result_embeddings = torch.stack([
                    self.index[idx] for idx in top_indices
                ])
        
        return SearchResult(
            indices=top_indices.cpu(),
            scores=top_scores.cpu(),
            embeddings=result_embeddings.cpu() if result_embeddings is not None else None,
        )
    
    def get_metadata(self, indices: torch.Tensor) -> List:
        """
        获取指定索引的元数据
        
        Args:
            indices: 索引张量
            
        Returns:
            元数据列表
        """
        if self.metadata is None:
            return []
        
        indices = indices.tolist()
        if isinstance(indices, int):
            return [self.metadata[indices]]
        return [self.metadata[i] for i in indices]
    
    def save(self, path: Union[str, Path]):
        """保存索引到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "index": self.index.cpu() if self.index is not None else None,
            "metadata": self.metadata,
            "metric": self.metric,
        }
        torch.save(data, path)
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "SimilaritySearch":
        """从文件加载索引"""
        data = torch.load(path, map_location="cpu")
        
        search = cls(metric=data["metric"], device=device)
        if data["index"] is not None:
            search.index = data["index"].to(search.device)
        search.metadata = data["metadata"]
        
        return search
    
    def __len__(self) -> int:
        """返回索引大小"""
        return 0 if self.index is None else len(self.index)
