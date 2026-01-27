"""
I-JEPA 训练示例脚本

演示如何使用 JEPA-tol 进行自监督视觉表示学习。
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from jepa_tol.models.vision import IJEPAModel


def get_transforms(img_size: int = 224):
    """获取数据增强 transforms"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def train_one_epoch(
    model: IJEPAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        # 前向传播
        outputs = model(images)
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
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="I-JEPA 训练脚本")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--img-size", type=int, default=224, help="图像大小")
    parser.add_argument("--embed-dim", type=int, default=768, help="嵌入维度")
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集（使用 CIFAR-10 作为示例）
    transform = get_transforms(args.img_size)
    
    try:
        dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    except Exception as e:
        print(f"无法加载 CIFAR-10: {e}")
        print("请确保网络连接正常，或手动下载数据集。")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # 模型
    model = IJEPAModel(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
    ).to(device)
    
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
            }, output_dir / "best_model.pt")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")
    
    print("训练完成！")


if __name__ == "__main__":
    main()
