"""Phase 1 下游任务工具测试"""

import torch
import pytest


class TestJEPAClassifier:
    """JEPAClassifier 测试"""
    
    def test_classifier_forward(self):
        """测试分类器前向传播"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.classifier import JEPAClassifier
        
        # 创建小型模型
        backbone = IJEPAModel(
            img_size=224,
            embed_dim=192,
            encoder_depth=2,
            predictor_depth=2,
        )
        
        classifier = JEPAClassifier(
            backbone=backbone,
            num_classes=10,
            freeze_backbone=True,
        )
        
        x = torch.randn(2, 3, 224, 224)
        logits = classifier(x)
        
        assert logits.shape == (2, 10)
    
    def test_classifier_predict(self):
        """测试分类预测"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.classifier import JEPAClassifier
        
        backbone = IJEPAModel(embed_dim=192, encoder_depth=2, predictor_depth=2)
        classifier = JEPAClassifier(backbone, num_classes=5)
        
        x = torch.randn(4, 3, 224, 224)
        predictions = classifier.predict(x)
        
        assert predictions.shape == (4,)
        assert predictions.min() >= 0
        assert predictions.max() < 5
    
    def test_multilabel_classifier(self):
        """测试多标签分类器"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.classifier import MultiLabelClassifier
        
        backbone = IJEPAModel(embed_dim=192, encoder_depth=2, predictor_depth=2)
        classifier = MultiLabelClassifier(backbone, num_classes=10)
        
        x = torch.randn(2, 3, 224, 224)
        predictions = classifier.predict(x, threshold=0.5)
        
        assert predictions.shape == (2, 10)
        assert set(predictions.unique().tolist()).issubset({0, 1})


class TestJEPARetriever:
    """JEPARetriever 测试"""
    
    def test_retriever_build_index(self):
        """测试索引构建"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.retriever import JEPARetriever
        
        model = IJEPAModel(embed_dim=192, encoder_depth=2, predictor_depth=2)
        retriever = JEPARetriever(model, use_faiss=False)
        
        # 构建索引
        images = torch.randn(10, 3, 224, 224)
        metadata = [f"image_{i}" for i in range(10)]
        retriever.build_index(images, metadata=metadata)
        
        assert len(retriever) == 10
    
    def test_retriever_search(self):
        """测试检索"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.retriever import JEPARetriever
        
        model = IJEPAModel(embed_dim=192, encoder_depth=2, predictor_depth=2)
        retriever = JEPARetriever(model, use_faiss=False)
        
        # 构建索引
        images = torch.randn(20, 3, 224, 224)
        retriever.build_index(images)
        
        # 搜索
        query = torch.randn(3, 224, 224)
        results = retriever.search(query, top_k=5)
        
        assert len(results.indices) == 5
        assert len(results.scores) == 5


class TestJEPAVisualizer:
    """JEPAVisualizer 测试"""
    
    def test_extract_embeddings(self):
        """测试嵌入提取"""
        from jepa_tol.models.vision import IJEPAModel
        from jepa_tol.tools.visualizer import JEPAVisualizer
        
        model = IJEPAModel(embed_dim=192, encoder_depth=2, predictor_depth=2)
        viz = JEPAVisualizer(model)
        
        images = torch.randn(5, 3, 224, 224)
        embeddings = viz.extract_embeddings(images)
        
        assert embeddings.shape == (5, 192)
    
    def test_reduce_dimensions_pca(self):
        """测试 PCA 降维"""
        from jepa_tol.tools.visualizer import JEPAVisualizer
        
        viz = JEPAVisualizer()
        
        # 随机嵌入
        import numpy as np
        embeddings = np.random.randn(50, 192)
        
        try:
            reduced = viz.reduce_dimensions(embeddings, method="pca", n_components=2)
            assert reduced.shape == (50, 2)
        except ImportError:
            pytest.skip("scikit-learn 未安装")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
