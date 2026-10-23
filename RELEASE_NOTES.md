# Release 0.1.0 — Paper Code Release

本版本开放以下核心组件：

- 检索 log-softmax、余弦语义损失和真实 LPIPS 接口。
- 单编码器 L∞ 约束 PGD，默认采用论文附录 A.2 的超参数。
- 给定计划嵌入后的 RDS，以及三次相关性评分的多数决策。
- 前向加噪和 eta=0 的 DDIM 单步数值工具。

随包提供合成测试、示例、论文对应关系、实现约定、源码来源对应表及安装验证记录。代码许可为 MIT。

这是从现有工程整理的核心组件版本。完整模型与数据、Agent、完整防御流水线和实验评估器不包含在 `0.1.0` 中。算法输入契约见 [ALGORITHMS.md](ALGORITHMS.md)，验证内容见 [VALIDATION.md](VALIDATION.md)。
