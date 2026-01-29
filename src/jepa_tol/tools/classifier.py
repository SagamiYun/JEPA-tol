"""
JEPA 分类器工具

基于 JEPA 表示的线性分类器，支持：
- 冻结 backbone + 线性探测 (Linear Probing)
- 全量微调 (Fine-tuning)
- 多标签分类
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jepa_tol.models.base import BaseModel


@dataclass
class ClassifierConfig:
    """分类器配置"""
    num_classes: int = 1000
    hidden_dim: Optional[int] = None  # None 表示直接线性分类
    dropout: float = 0.0
    freeze_backbone: bool = True  # 线性探测模式


class LinearClassifier(nn.Module):
    """
    线性分类头
    
    可选择使用单层线性或带隐藏层的 MLP。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class JEPAClassifier(nn.Module):
    """
    基于 JEPA 的分类器
    
    使用 JEPA 模型作为 backbone 提取表示，
    然后通过线性分类头进行分类。
    
    使用示例：
        >>> from jepa_tol.models.vision import IJEPAModel
        >>> backbone = IJEPAModel.from_pretrained(checkpoint_path="...")
        >>> classifier = JEPAClassifier(backbone, num_classes=100)
        >>> logits = classifier(images)
    """
    
    def __init__(
        self,
        backbone: BaseModel,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
    ):
        """
        初始化分类器
        
        Args:
            backbone: JEPA 模型 (需要有 encode 方法)
            num_classes: 类别数量
            hidden_dim: 隐藏层维度 (None 表示纯线性)
            dropout: Dropout 概率
            freeze_backbone: 是否冻结 backbone
        """
        super().__init__()
        
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            self.backbone.freeze()
        
        self.classifier = LinearClassifier(
            embed_dim=backbone.embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            分类 logits (B, num_classes)
        """
        # 提取表示
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.backbone.encode(x)
        else:
            embeddings = self.backbone.encode(x)
        
        # 分类
        logits = self.classifier(embeddings)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        logits = self.forward(x)
        return logits.argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def unfreeze_backbone(self):
        """解冻 backbone 进行微调"""
        self.freeze_backbone = False
        self.backbone.unfreeze()
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        if self.freeze_backbone:
            return list(self.classifier.parameters())
        return list(self.parameters())


class MultiLabelClassifier(JEPAClassifier):
    """
    多标签分类器
    
    使用 sigmoid 激活，支持每个样本有多个标签。
    """
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """预测多标签"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def train_classifier(
    classifier: JEPAClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    训练分类器
    
    Args:
        classifier: 分类器实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 运行设备
        
    Returns:
        训练历史 (losses, accuracies)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    
    # 只优化可训练参数
    optimizer = torch.optim.AdamW(classifier.get_trainable_params(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # 训练
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = classifier(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # 验证
        if val_loader is not None:
            classifier.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = classifier(images)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    
    return history
