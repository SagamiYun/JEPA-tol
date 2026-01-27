"""jepa_tol 模块测试"""

import torch
import pytest


class TestEncoder:
    """Encoder 测试"""
    
    def test_vit_encoder_forward(self):
        """测试 ViT Encoder 前向传播"""
        from jepa_tol.core.encoder import ViTEncoder
        
        encoder = ViTEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6,
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 384), f"期望 (2, 384)，实际 {output.shape}"
    
    def test_vit_encoder_patch_embeddings(self):
        """测试 patch 嵌入输出"""
        from jepa_tol.core.encoder import ViTEncoder
        
        encoder = ViTEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6,
        )
        
        x = torch.randn(2, 3, 224, 224)
        patch_embeds = encoder.get_patch_embeddings(x)
        
        num_patches = (224 // 16) ** 2
        assert patch_embeds.shape == (2, num_patches, 384)


class TestPredictor:
    """Predictor 测试"""
    
    def test_transformer_predictor(self):
        """测试 Transformer Predictor"""
        from jepa_tol.core.predictor import TransformerPredictor
        
        predictor = TransformerPredictor(
            num_patches=196,
            embed_dim=384,
            predictor_dim=192,
            depth=4,
            num_heads=4,
        )
        
        context = torch.randn(2, 100, 384)  # 100 个 context patches
        target_indices = torch.randint(0, 196, (2, 50))  # 50 个 target indices
        
        output = predictor(context, target_indices=target_indices)
        
        assert output.shape == (2, 50, 384)


class TestIJEPAModel:
    """I-JEPA 模型测试"""
    
    def test_ijepa_forward(self):
        """测试 I-JEPA 前向传播"""
        from jepa_tol.models.vision import IJEPAModel
        
        model = IJEPAModel(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            encoder_depth=4,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=2,
        )
        
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        
        assert "pred_embeddings" in outputs
        assert "target_embeddings" in outputs
        assert outputs["pred_embeddings"].shape == outputs["target_embeddings"].shape
    
    def test_ijepa_encode(self):
        """测试 I-JEPA encoding"""
        from jepa_tol.models.vision import IJEPAModel
        
        model = IJEPAModel(
            img_size=224,
            embed_dim=384,
            encoder_depth=4,
            encoder_heads=4,
        )
        
        x = torch.randn(2, 3, 224, 224)
        embeddings = model.encode(x)
        
        assert embeddings.shape == (2, 384)
    
    def test_ijepa_loss(self):
        """测试 loss 计算"""
        from jepa_tol.models.vision import IJEPAModel
        
        model = IJEPAModel(embed_dim=384, encoder_depth=2, predictor_depth=2)
        
        pred = torch.randn(2, 50, 384)
        target = torch.randn(2, 50, 384)
        
        loss = model.compute_loss(pred, target)
        
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0


class TestTools:
    """工具测试"""
    
    def test_similarity_search(self):
        """测试相似度搜索"""
        from jepa_tol.tools.similarity_search import SimilaritySearch
        
        search = SimilaritySearch(metric="cosine")
        
        # 构建索引
        embeddings = torch.randn(100, 384)
        search.build_index(embeddings)
        
        # 搜索
        query = torch.randn(384)
        results = search.search(query, top_k=5)
        
        assert len(results.indices) == 5
        assert len(results.scores) == 5
        assert results.scores[0] >= results.scores[-1]  # 降序


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
