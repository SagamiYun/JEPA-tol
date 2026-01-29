"""
JEPA 语义检索工具

基于 JEPA 嵌入的高效语义检索系统。
支持 FAISS 索引加速和增量更新。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from jepa_tol.models.base import BaseModel
from jepa_tol.tools.similarity_search import SimilaritySearch, SearchResult


@dataclass
class RetrievalResult:
    """检索结果"""
    indices: List[int]
    scores: List[float]
    metadata: List[Any] = field(default_factory=list)


class JEPARetriever:
    """
    基于 JEPA 的语义检索器
    
    使用 JEPA 模型提取嵌入，结合 SimilaritySearch 实现高效检索。
    
    使用示例：
        >>> retriever = JEPARetriever(model)
        >>> retriever.build_index(images, metadata=filenames)
        >>> results = retriever.search(query_image, top_k=5)
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
        use_faiss: bool = True,
        normalize: bool = True,
    ):
        """
        初始化检索器
        
        Args:
            model: JEPA 模型
            device: 运行设备
            use_faiss: 是否使用 FAISS (需要安装)
            normalize: 是否 L2 归一化嵌入
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.normalize = normalize
        self.use_faiss = use_faiss
        
        # 基础搜索引擎
        self.search_engine = SimilaritySearch(
            metric="cosine" if normalize else "l2",
            device=self.device,
        )
        
        # FAISS 索引 (可选)
        self.faiss_index = None
        self._try_init_faiss()
        
        # 元数据存储
        self.metadata_store: List[Any] = []
    
    def _try_init_faiss(self):
        """尝试初始化 FAISS"""
        if not self.use_faiss:
            return
        
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            self._faiss = None
            print("FAISS 未安装，使用基础搜索引擎")
    
    @torch.no_grad()
    def encode(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        编码输入为嵌入向量
        
        Args:
            inputs: 输入数据
            batch_size: 批次大小
            
        Returns:
            嵌入向量 (N, D)
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        
        all_embeddings = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(self.device)
            embeddings = self.model.encode(batch)
            
            if self.normalize:
                embeddings = F.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def build_index(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        metadata: Optional[List[Any]] = None,
        batch_size: int = 32,
    ):
        """
        构建检索索引
        
        Args:
            data: 图像数据
            metadata: 每个图像的元数据 (文件名、标签等)
            batch_size: 编码批次大小
        """
        # 编码
        embeddings = self.encode(data, batch_size=batch_size)
        
        # 保存元数据
        self.metadata_store = metadata if metadata else list(range(len(embeddings)))
        
        # 构建 FAISS 索引
        if self._faiss is not None:
            embed_np = embeddings.numpy().astype(np.float32)
            embed_dim = embed_np.shape[1]
            
            # 使用 IVF 索引加速大规模检索
            if len(embeddings) > 10000:
                nlist = min(100, len(embeddings) // 100)
                quantizer = self._faiss.IndexFlatIP(embed_dim)
                self.faiss_index = self._faiss.IndexIVFFlat(
                    quantizer, embed_dim, nlist, self._faiss.METRIC_INNER_PRODUCT
                )
                self.faiss_index.train(embed_np)
            else:
                self.faiss_index = self._faiss.IndexFlatIP(embed_dim)
            
            self.faiss_index.add(embed_np)
        else:
            # 使用基础搜索引擎
            self.search_engine.build_index(embeddings, normalize=False)
    
    def add_to_index(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        metadata: Optional[List[Any]] = None,
        batch_size: int = 32,
    ):
        """
        增量添加到索引
        
        Args:
            data: 新图像数据
            metadata: 新元数据
        """
        embeddings = self.encode(data, batch_size=batch_size)
        
        # 更新元数据
        if metadata:
            self.metadata_store.extend(metadata)
        else:
            start_idx = len(self.metadata_store)
            self.metadata_store.extend(range(start_idx, start_idx + len(embeddings)))
        
        # 更新索引
        if self.faiss_index is not None:
            embed_np = embeddings.numpy().astype(np.float32)
            self.faiss_index.add(embed_np)
        else:
            self.search_engine.add_to_index(embeddings, normalize=False)
    
    @torch.no_grad()
    def search(
        self,
        query: Union[torch.Tensor, List[torch.Tensor]],
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        检索最相似的项
        
        Args:
            query: 查询图像
            top_k: 返回数量
            
        Returns:
            检索结果
        """
        # 编码查询
        if isinstance(query, list):
            query = torch.stack(query)
        if query.dim() == 3:
            query = query.unsqueeze(0)
        
        query_embedding = self.encode(query)[0]
        
        # 搜索
        if self.faiss_index is not None:
            query_np = query_embedding.numpy().astype(np.float32).reshape(1, -1)
            scores, indices = self.faiss_index.search(query_np, top_k)
            indices = indices[0].tolist()
            scores = scores[0].tolist()
        else:
            result = self.search_engine.search(query_embedding, top_k=top_k)
            indices = result.indices.tolist()
            scores = result.scores.tolist()
        
        # 获取元数据
        metadata = [self.metadata_store[i] for i in indices if i < len(self.metadata_store)]
        
        return RetrievalResult(
            indices=indices,
            scores=scores,
            metadata=metadata,
        )
    
    def batch_search(
        self,
        queries: Union[torch.Tensor, List[torch.Tensor]],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[RetrievalResult]:
        """
        批量检索
        
        Args:
            queries: 查询图像列表
            top_k: 每个查询返回数量
            batch_size: 编码批次大小
            
        Returns:
            检索结果列表
        """
        # 编码所有查询
        query_embeddings = self.encode(queries, batch_size=batch_size)
        
        results = []
        for query_emb in query_embeddings:
            if self.faiss_index is not None:
                query_np = query_emb.numpy().astype(np.float32).reshape(1, -1)
                scores, indices = self.faiss_index.search(query_np, top_k)
                indices = indices[0].tolist()
                scores = scores[0].tolist()
            else:
                result = self.search_engine.search(query_emb, top_k=top_k)
                indices = result.indices.tolist()
                scores = result.scores.tolist()
            
            metadata = [self.metadata_store[i] for i in indices if i < len(self.metadata_store)]
            results.append(RetrievalResult(indices=indices, scores=scores, metadata=metadata))
        
        return results
    
    def save(self, path: Union[str, Path]):
        """保存检索器状态"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        torch.save({
            "metadata": self.metadata_store,
            "normalize": self.normalize,
        }, path / "retriever_state.pt")
        
        # 保存 FAISS 索引
        if self.faiss_index is not None:
            self._faiss.write_index(self.faiss_index, str(path / "faiss_index.bin"))
        else:
            self.search_engine.save(path / "search_index.pt")
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: BaseModel,
        device: Optional[str] = None,
    ) -> "JEPARetriever":
        """加载检索器"""
        path = Path(path)
        
        state = torch.load(path / "retriever_state.pt", map_location="cpu")
        
        retriever = cls(model=model, device=device, normalize=state["normalize"])
        retriever.metadata_store = state["metadata"]
        
        # 加载索引
        if (path / "faiss_index.bin").exists() and retriever._faiss is not None:
            retriever.faiss_index = retriever._faiss.read_index(str(path / "faiss_index.bin"))
        elif (path / "search_index.pt").exists():
            retriever.search_engine = SimilaritySearch.load(path / "search_index.pt", device=device)
        
        return retriever
    
    def __len__(self) -> int:
        """索引大小"""
        return len(self.metadata_store)
