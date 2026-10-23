# 来源与整理边界

本包从 Visual Inception 现有工程中整理发布，保留原有 MIT 许可与版权声明。以下记录接口整理及其与原工程的对应关系。

本版本整理了 API、补充了输入契约与错误检查，并将依赖限制在已开放的数值组件。原工程保留原状。所列可比较路径已通过合成输入交叉核验；这记录了本版本与现有工程的关系，不将整理后的 API 视为历史实验源码的逐字快照。

| 发布文件 | 原工程依据 | 整理内容 |
| --- | --- | --- |
| `visual_inception_core/losses.py` | `src/attack/loss.py` | 只保留检索、语义、感知和总目标；去掉自动模型加载及未明确的负样本项；增加输入检查和梯度接口检查 |
| `visual_inception_core/pgd.py` | `src/attack/visual_inception.py`、`src/attack/samplers.py` | 只保留单编码器有界下降与固定池最佳点；接收预计算嵌入；不包含集成、EOT、自适应扩展、上传和计时 |
| `visual_inception_core/verification.py` | `src/defense/system2.py` | 从模型调用流水线中分离 RDS、三次中位数投票和 OR 合并；数值余弦舍入裁界 |
| `visual_inception_core/diffusion.py` | `src/defense/system1.py` 和论文 §4.1 | 独立前向关系与 DDIM 单步；不提供模型、参考集或停止规则；sample 分支要求 alpha<1 后使用真实分母 |
| `tests/`、`examples/` | 本次整理新增 | 合成验证与接口演示，不是论文实验 |

原工程关键源文件、所核对稿件与正式 PDF 的 SHA-256 见 `validation/original_comparison.json`；其中的路径是来源标识，不表示这些资产包含在本包内。可复用 `tools/compare_with_original.py --source-root PATH --output PATH` 对另一个原工程副本运行相同的合成对照。加入可选 `--manuscript` 和 `--paper-pdf` 参数可记录对应材料的哈希。

`MANIFEST.sha256` 记录本包其他所有文件的内容哈希（不包含自身），用于确认文件完整性。测试与安装记录见 `VALIDATION.md`。
