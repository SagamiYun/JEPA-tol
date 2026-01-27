# JEPA 研究笔记

## 论文核心思想

### 来源
- **论文**: "A Path Towards Autonomous Machine Intelligence" (Yann LeCun, 2022)
- **链接**: https://openreview.net/forum?id=BZ5a1r-kVsf

### JEPA vs 生成式模型

| 特性 | 生成式模型 (GPT/Diffusion) | JEPA |
|------|---------------------------|------|
| 预测目标 | 原始像素/token | 抽象表示 |
| 处理细节 | 必须预测所有细节 | 忽略不可预测的细节 |
| 效率 | 低（需要大量计算） | 高（在嵌入空间操作） |
| 适用场景 | 生成任务 | 理解/推理任务 |

### 认知架构组件

1. **Encoder (编码器)** - 将输入映射到嵌入空间
2. **Predictor (预测器)** - 在嵌入空间中预测
3. **World Model (世界模型)** - 整合上述组件，实现对世界的预测
4. **Cost Module** - 评估预测质量
5. **Actor** - 基于预测采取行动
6. **Configurator** - 配置系统行为

### 层次化 JEPA (H-JEPA)

- **底层**: 处理短期、细粒度预测
- **高层**: 处理长期、抽象预测
- **优势**: 支持多尺度规划和推理

## 研究方向

### 短期目标
- [ ] 复现 I-JEPA 基础结果
- [ ] 理解 EMA target encoder 的作用
- [ ] 分析 mask 策略对学习的影响

### 中期目标
- [ ] 实现 V-JEPA (Video JEPA)
- [ ] 探索多模态 JEPA
- [ ] 构建基于 JEPA 的检索系统

### 长期目标
- [ ] 实现 H-JEPA 层次化架构
- [ ] 集成规划和推理能力
- [ ] 探索 World Model 在机器人领域的应用

## 参考资源

- [I-JEPA 官方代码](https://github.com/facebookresearch/ijepa)
- [V-JEPA 官方代码](https://github.com/facebookresearch/vjepa)
- [LeCun 的演讲录像](https://www.youtube.com/watch?v=DokLw1tILlw)
