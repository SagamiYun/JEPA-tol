"""
V-JEPA 训练示例脚本

演示如何使用 JEPA-tol 进行视频自监督表示学习。
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from jepa_tol.models.video import VJEPAModel, VJEPAConfig


class RandomVideoDataset(Dataset):
    """
    随机视频数据集 (用于演示)
    
    实际使用时应替换为真实视频数据集。
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 16,
        img_size: int = 224,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # 返回随机视频张量 (T, C, H, W)
        return torch.randn(self.num_frames, 3, self.img_size, self.img_size)


def train_one_epoch(
    model: VJEPAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, videos in enumerate(dataloader):
        videos = videos.to(device)
        
        # 前向传播
        outputs = model(videos)
        loss = model.compute_loss(
            outputs["pred_embeddings"],
            outputs["target_embeddings"]
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA 更新 target encoder
        model.update_target_encoder()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 训练脚本")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num-frames", type=int, default=16, help="视频帧数")
    parser.add_argument("--img-size", type=int, default=224, help="图像大小")
    parser.add_argument("--embed-dim", type=int, default=384, help="嵌入维度")
    parser.add_argument("--encoder-depth", type=int, default=6, help="编码器深度")
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集 (演示用随机数据)
    print("注意: 当前使用随机数据进行演示。")
    print("实际训练请替换为真实视频数据集 (如 Kinetics, SSv2 等)。")
    
    dataset = RandomVideoDataset(
        num_samples=500,
        num_frames=args.num_frames,
        img_size=args.img_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # 模型配置
    config = VJEPAConfig(
        img_size=args.img_size,
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.embed_dim // 64,
        predictor_dim=args.embed_dim // 2,
        predictor_depth=4,
        predictor_heads=args.embed_dim // 128,
    )
    
    model = VJEPAModel.from_config(config).to(device)
    
    print(f"模型参数量: {model.get_num_parameters() / 1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # 训练
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch} 平均 Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, output_dir / "best_vjepa_model.pt")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, output_dir / f"vjepa_checkpoint_epoch{epoch}.pt")
    
    print("训练完成！")


if __name__ == "__main__":
    main()
