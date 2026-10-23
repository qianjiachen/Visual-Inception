# Visual Inception — 核心代码

本仓库提供 Visual Inception 的核心代码，包括单编码器损失与有界 PGD、RDS 与三次相关性评分的决策规则，以及前向加噪和确定性 DDIM 单步更新。

Jiachen Qian, *Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning*, ACL 2026：[论文页面](https://aclanthology.org/2026.acl-long.954/)、[PDF](https://aclanthology.org/2026.acl-long.954.pdf)。

版本 `0.1.0` 为经过接口整理与数值检查的核心组件发布；具体实现约定见下文，版本内容见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。

## 公开范围

| 内容 | 提供内容 | 对应依据与范围 |
| --- | --- | --- |
| 损失 | 检索 log-softmax、目标语义余弦损失、真实 LPIPS 的调用接口 | §3.2；检索分母采用下文明确的实现约定 |
| 优化 | 单编码器、零初始化、有界符号梯度下降；默认 100 步 | 附录 A.2 的步数、步长、扰动预算和损失权重 |
| 推理校验 | 已给定计划嵌入的 RDS，以及三次相关性评分的多数决策 | §4.2；不包含 Agent、SBERT 或 LLM judge 本身 |
| 扩散数值工具 | 前向加噪；eta=0 的 DDIM 单步，支持三种预测类型 | 前向式见 §4.1；DDIM 为公开说明的实现选择 |

本次发布范围限于上述组件。跨编码器集成、负样本配方、查询来源构造、真实编码器预处理、记忆库、完整 Agent、完整 System 1/2、指标评估器、第三方基线、数据与模型权重不在本版本中。论文实验结果请引用论文；本包的合成测试用于验证已开放代码的数值行为。

## 安装和运行

要求 Python ≥ 3.10、PyTorch ≥ 2.2；本次实际验证环境见 [VALIDATION.md](VALIDATION.md)，其他版本未逐一验证。

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
python -m examples.toy_demo
```

安装依赖可能访问包索引；算法与演示本身不发起网络请求，不自动下载模型或调用 API。已有依赖时，可在本目录直接执行测试和演示，无需安装本包。

演示使用合成张量和 toy encoder，并显式设置 `lambda_perceptual=0`，运行删去感知项的简化目标。默认配置保留论文的 `lambda_perceptual=0.1`，需要调用者提供真实 LPIPS 模型；缺少模型时会报错。

## 实现约定

1. 检索分母为“攻击图 + 已有代理记忆”，攻击图只出现一次。论文未明确记忆集合是否已经包含攻击图；这里公开采用包含正类的 softmax 分类定义。
2. 查询嵌入由调用者提供，每步从固定池的索引中无放回抽取至多 16 项。论文的 100 条查询、50%/30%/20% 来源构造不在本包内实现；任意输入矩阵不自动满足该协议。
3. PGD 从原图开始；用同一个完整查询池比较初始点和每次更新，返回其中目标值最低的图。这些是公开的实现选择，论文未完全规定初始化和检查点选择。
4. 优化器接收单个编码器，所有图像、目标、查询和记忆嵌入须来自同一空间。形状相同不能证明模型和预处理一致，调用者必须保证这一点。
5. RDS 需来自同一配置下、有无指定记忆的成对 Agent 计划的 SBERT 嵌入；相关性输入需来自三次真实 judge 调用。本包只计算分数和规则，不生成这些输入。

完整公式、输入契约和伪代码见 [ALGORITHMS.md](ALGORITHMS.md)；论文对应关系与版本边界见 [SCOPE_AND_ALIGNMENT.md](SCOPE_AND_ALIGNMENT.md)。

## 验证与许可

测试覆盖公式、梯度、投影约束、检查点选择、阈值边界和 DDIM 解析关系。[VALIDATION.md](VALIDATION.md) 记录独立测试、与原工程选定路径的数值对照，以及脱离原工程的安装检查；真实模型和论文实验不在本次数值验证范围内。

本代码采用 [MIT 许可](LICENSE)。第三方模型、数据与论文遵循各自许可。代码整理范围见 [SOURCE_ORIGINS.md](SOURCE_ORIGINS.md)。

## 引用

使用本代码时，请引用论文；BibTeX 见 [references.bib](references.bib)，软件引用元数据见 [CITATION.cff](CITATION.cff)。
