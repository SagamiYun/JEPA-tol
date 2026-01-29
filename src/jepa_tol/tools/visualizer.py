"""
JEPA 可视化工具

提供 JEPA 表示空间的可视化功能：
- t-SNE / UMAP 降维可视化
- 注意力热力图
- 聚类可视化
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from jepa_tol.models.base import BaseModel


class JEPAVisualizer:
    """
    JEPA 表示可视化器
    
    使用示例：
        >>> viz = JEPAVisualizer(model)
        >>> embeddings = viz.extract_embeddings(images)
        >>> viz.plot_tsne(embeddings, labels, save_path="tsne.png")
    """
    
    def __init__(
        self,
        model: Optional[BaseModel] = None,
        device: Optional[str] = None,
    ):
        """
        初始化可视化器
        
        Args:
            model: JEPA 模型 (可选，用于提取嵌入)
            device: 运行设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        提取嵌入向量
        
        Args:
            data: 输入数据
            batch_size: 批次大小
            normalize: 是否 L2 归一化
            
        Returns:
            嵌入数组 (N, D)
        """
        if self.model is None:
            raise ValueError("模型未设置，请在初始化时传入模型")
        
        if isinstance(data, list):
            data = torch.stack(data)
        
        all_embeddings = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(self.device)
            embeddings = self.model.encode(batch)
            
            if normalize:
                embeddings = F.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "tsne",
        n_components: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """
        降维
        
        Args:
            embeddings: 嵌入数组 (N, D)
            method: 降维方法 ("tsne", "umap", "pca")
            n_components: 目标维度
            **kwargs: 降维方法的额外参数
            
        Returns:
            降维后的数组 (N, n_components)
        """
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=n_components,
                    perplexity=kwargs.get("perplexity", 30),
                    random_state=kwargs.get("random_state", 42),
                )
                return reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
        
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=kwargs.get("n_neighbors", 15),
                    min_dist=kwargs.get("min_dist", 0.1),
                    random_state=kwargs.get("random_state", 42),
                )
                return reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("需要安装 umap-learn: pip install umap-learn")
        
        elif method == "pca":
            try:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
                return reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
        
        else:
            raise ValueError(f"不支持的降维方法: {method}")
    
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        title: str = "JEPA Embedding Space",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
        **kwargs,
    ):
        """
        绘制嵌入空间可视化
        
        Args:
            embeddings: 嵌入数组
            labels: 标签数组 (用于着色)
            method: 降维方法
            title: 图表标题
            save_path: 保存路径
            figsize: 图表大小
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装 matplotlib: pip install matplotlib")
        
        # 降维
        if embeddings.shape[1] > 2:
            reduced = self.reduce_dimensions(embeddings, method=method, **kwargs)
        else:
            reduced = embeddings
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=labels, cmap="tab10", alpha=0.7, s=10
            )
            plt.colorbar(scatter, ax=ax, label="Class")
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=10)
        
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")
        
        plt.close()
        return fig
    
    def plot_similarity_matrix(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Similarity Matrix",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        绘制相似度矩阵热力图
        
        Args:
            embeddings: 嵌入数组
            labels: 样本标签
            title: 图表标题  
            save_path: 保存路径
            figsize: 图表大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("需要安装 matplotlib 和 seaborn")
        
        # 计算相似度矩阵
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity = embeddings_norm @ embeddings_norm.T
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            similarity,
            xticklabels=labels if labels else False,
            yticklabels=labels if labels else False,
            cmap="viridis",
            ax=ax,
            square=True,
        )
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        plt.close()
        return fig
    
    def compute_cluster_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算聚类质量指标
        
        Args:
            embeddings: 嵌入数组
            labels: 真实标签
            
        Returns:
            指标字典
        """
        try:
            from sklearn.metrics import (
                silhouette_score,
                calinski_harabasz_score,
                davies_bouldin_score,
            )
        except ImportError:
            raise ImportError("需要安装 scikit-learn")
        
        # 归一化
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return {
            "silhouette_score": silhouette_score(embeddings_norm, labels),
            "calinski_harabasz_score": calinski_harabasz_score(embeddings_norm, labels),
            "davies_bouldin_score": davies_bouldin_score(embeddings_norm, labels),
        }
    
    @staticmethod
    def visualize_attention(
        attention_weights: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 5),
    ):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重 (H, N, N) 或 (N, N)
            image: 原始图像 (可选)
            save_path: 保存路径
            figsize: 图表大小
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装 matplotlib")
        
        # 处理注意力权重
        if attention_weights.dim() == 3:
            # 多头注意力，取平均
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights
        
        attn = attn.cpu().numpy()
        
        # 绘图
        if image is not None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # 显示原图
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # 显示注意力
            im = axes[1].imshow(attn, cmap="viridis")
            axes[1].set_title("Attention Weights")
            plt.colorbar(im, ax=axes[1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(attn, cmap="viridis")
            ax.set_title("Attention Weights")
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        plt.close()
        return fig
