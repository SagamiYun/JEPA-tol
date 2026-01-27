# JEPA-tol

基于 Yann LeCun 论文 **"A Path Towards Autonomous Machine Intelligence" (2022)** 的 JEPA (Joint Embedding Predictive Architecture) 研究与工具集。

## 📚 项目背景

JEPA 是一种**非生成式**的自监督学习架构：
- **预测表示而非原始数据**：从输入的嵌入预测输出的嵌入
- **忽略不可预测的细节**：聚焦于数据中可预测和显著的方面
- **层次化结构 (H-JEPA)**：低层处理短期预测，高层处理长期抽象

## 🏗️ 项目结构

```
JEPA-tol/
├── src/jepa_tol/
│   ├── core/          # JEPA 核心架构 (Encoder, Predictor, World Model)
│   ├── models/        # 预训练模型适配器
│   └── tools/         # 可复用工具集
├── experiments/       # 实验脚本
└── tests/             # 单元测试
```

## 🚀 快速开始

```bash
# 安装
pip install -e .

# 或使用 uv
uv pip install -e .
```

## 🔧 核心模块

### Core
- `Encoder` - 将输入映射到嵌入空间
- `Predictor` - 在嵌入空间中进行预测
- `WorldModel` - 整合 Encoder 和 Predictor 的世界模型

### Tools
- `RepresentationExtractor` - 提取 JEPA 表示
- `SimilaritySearch` - 基于语义的相似度搜索

## 📖 参考资料

- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) - Yann LeCun, 2022
- [I-JEPA (Image JEPA)](https://github.com/facebookresearch/ijepa) - Meta AI
- [V-JEPA (Video JEPA)](https://github.com/facebookresearch/vjepa) - Meta AI

## 📄 License

MIT License
