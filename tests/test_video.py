"""V-JEPA 视频模型测试"""

import torch
import pytest


class TestTemporalEncoder:
    """TemporalEncoder 测试"""
    
    def test_temporal_encoder_forward(self):
        """测试时序编码器前向传播"""
        from jepa_tol.core.temporal_encoder import TemporalEncoder
        
        encoder = TemporalEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=2,
            num_heads=3,
            max_frames=8,
            attention_type="divided",
        )
        
        # 输入视频 (B, T, C, H, W)
        x = torch.randn(2, 4, 3, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 192), f"期望 (2, 192)，实际 {output.shape}"
    
    def test_get_patch_embeddings(self):
        """测试 patch 嵌入"""
        from jepa_tol.core.temporal_encoder import TemporalEncoder
        
        encoder = TemporalEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=2,
            num_heads=3,
            max_frames=8,
        )
        
        x = torch.randn(2, 4, 3, 224, 224)
        patch_embeds = encoder.get_patch_embeddings(x)
        
        # T=4, N=196 (14x14)
        num_patches = (224 // 16) ** 2
        assert patch_embeds.shape == (2, 4 * num_patches, 192)
    
    def test_get_frame_embeddings(self):
        """测试帧嵌入"""
        from jepa_tol.core.temporal_encoder import TemporalEncoder
        
        encoder = TemporalEncoder(
            img_size=224, embed_dim=192, depth=2, num_heads=3
        )
        
        x = torch.randn(2, 8, 3, 224, 224)
        frame_embeds = encoder.get_frame_embeddings(x)
        
        assert frame_embeds.shape == (2, 8, 192)


class TestVJEPAModel:
    """VJEPAModel 测试"""
    
    def test_vjepa_forward(self):
        """测试 V-JEPA 前向传播"""
        from jepa_tol.models.video import VJEPAModel
        
        model = VJEPAModel(
            img_size=224,
            patch_size=16,
            num_frames=4,
            embed_dim=192,
            encoder_depth=2,
            encoder_heads=3,
            predictor_depth=2,
            predictor_heads=3,
        )
        
        x = torch.randn(2, 4, 3, 224, 224)
        outputs = model(x)
        
        assert "pred_embeddings" in outputs
        assert "target_embeddings" in outputs
        assert outputs["pred_embeddings"].shape == outputs["target_embeddings"].shape
    
    def test_vjepa_encode(self):
        """测试 V-JEPA 编码"""
        from jepa_tol.models.video import VJEPAModel
        
        model = VJEPAModel(
            img_size=224,
            num_frames=4,
            embed_dim=192,
            encoder_depth=2,
            predictor_depth=2,
        )
        
        x = torch.randn(2, 4, 3, 224, 224)
        embeddings = model.encode(x)
        
        assert embeddings.shape == (2, 192)
    
    def test_vjepa_encode_frames(self):
        """测试帧级别编码"""
        from jepa_tol.models.video import VJEPAModel
        
        model = VJEPAModel(
            img_size=224,
            num_frames=8,
            embed_dim=192,
            encoder_depth=2,
            predictor_depth=2,
        )
        
        x = torch.randn(2, 8, 3, 224, 224)
        frame_embeds = model.encode_frames(x)
        
        assert frame_embeds.shape == (2, 8, 192)
    
    def test_vjepa_loss(self):
        """测试 loss 计算"""
        from jepa_tol.models.video import VJEPAModel
        
        model = VJEPAModel(
            embed_dim=192, num_frames=4, encoder_depth=2, predictor_depth=2
        )
        
        pred = torch.randn(2, 50, 192)
        target = torch.randn(2, 50, 192)
        
        loss = model.compute_loss(pred, target)
        
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestTubeMaskGenerator:
    """TubeMaskGenerator 测试"""
    
    def test_mask_generation(self):
        """测试 mask 生成"""
        from jepa_tol.models.video import TubeMaskGenerator
        
        generator = TubeMaskGenerator(
            num_frames=4,
            num_patches_per_frame=196,
            mask_ratio=0.9,
            tube_length=2,
        )
        
        device = torch.device("cpu")
        context_mask, target_mask = generator(batch_size=2, device=device)
        
        total_tokens = 4 * 196
        
        assert context_mask.shape == (2, total_tokens)
        assert target_mask.shape == (2, total_tokens)
        
        # 验证 mask 比例大致正确
        visible_ratio = context_mask[0].float().mean().item()
        assert 0.05 < visible_ratio < 0.3  # 大约 10% 可见


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
