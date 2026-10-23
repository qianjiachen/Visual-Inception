# 发布介绍

## 标题

Visual Inception 核心代码发布（ACL 2026）

## 简介

我们发布论文 *Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning* 的核心代码。

本次开放的组件包括检索与语义损失、真实 LPIPS 的接入接口、L∞ 约束下的单编码器 PGD、基于计划嵌入的 RDS，以及三次相关性评分的多数决策规则。同时提供前向加噪与确定性 DDIM 单步工具，便于检查扩散数值关系。

默认 PGD 配置为 100 步、步长 2/255、扰动预算 8/255，检索温度为 0.07，语义和感知损失权重分别为 1.0 和 0.1。公开说明同时列出检索分母、初始化、查询批次与最佳迭代选择等实现约定。

版本 0.1.0 限于上述核心组件。完整 Agent、完整 CognitiveGuard 流水线、跨编码器集成、实验评估器、数据与模型权重不随本版本提供。随包附有算法说明、论文对应关系、70 项合成测试及独立安装验证记录；论文结果请引用正式发表版本。

代码采用 MIT 许可。论文：[ACL Anthology](https://aclanthology.org/2026.acl-long.954/)。

## 简短公告

Visual Inception（ACL 2026）核心代码现已开放，包含单编码器 PGD、损失函数、RDS 与相关性投票规则及扩散数值工具，并附实现约定、测试与运行说明。本版本的具体开放范围见仓库 README。
