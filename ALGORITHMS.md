# 算法说明与输入契约

以下均是数值算法及其公开约定。文中的模型输出、反事实计划和 judge 评分由调用者提供；没有把合成数据当成真实模型证据。

## 1. 单编码器目标

令原始图像为 $x\in[0,1]^{1\times3\times H\times W}$，当前图像为 $x'$。令 $a=E_I(x')$、$g=E_T(T_{tgt})$，查询为 $q_j$，已有代理记忆为 $m_k$。所有嵌入位于同一空间，以下相似度为余弦相似度。

本包明确选择把攻击图作为正类纳入检索分母：

$$
\mathcal L_{ret}=\frac1Q\sum_{j=1}^{Q}
\left[\log\left(e^{\cos(a,q_j)/\tau}+\sum_{k=1}^{M}e^{\cos(m_k,q_j)/\tau}\right)
-\frac{\cos(a,q_j)}{\tau}\right].
$$

计算使用 `log_softmax`，不直接对大数求指数。`memories` 只包含已有代理记忆，不能再把当前攻击图重复放入该矩阵。该公式把论文 §3.2 对集合包含关系的歧义落实为可检查的实现定义；不宣称论文已经唯一规定此选择。

$$
\mathcal L_{sem}=1-\cos(a,g),\qquad
\mathcal L_{per}=\operatorname{LPIPS}(x,x'),
$$

$$
\mathcal L=\mathcal L_{ret}+\lambda_{sem}\mathcal L_{sem}
+\lambda_{per}\mathcal L_{per}.
$$

默认 $\tau=0.07$、$\lambda_{sem}=1.0$、$\lambda_{per}=0.1$。LPIPS 接口把 `[0,1]` 像素映射为 `[-1,1]` 后调用真实模型；模型由调用者安装、加载，置于 eval 模式并冻结权重。代码无法从一个 callable 的类型确认它就是 LPIPS，调用者需保证模型身份和版本。测试中的合成距离只验证接口传值和梯度，没有验证真实 LPIPS 权重。

约束：输入为 float32 或 float64，同设备、同精度；向量有限、非空且范数大于 `1e-8`。`adv`、`target` 形状均为 `[1,D]`，`queries` 为 `[Q,D]`，`memories` 为 `[M,D]`，其中 $Q,M,D\ge1$。拒绝零向量，避免把未定义的余弦相似度当成正常证据。形状与数值检查不能验证嵌入的模型来源。

## 2. 有界 PGD：单编码器子问题

对当前图像 $x^{(t)}$，用本步查询批次计算梯度并执行下降：

$$
u=x^{(t)}-\alpha\operatorname{sign}\nabla_{x^{(t)}}\mathcal L,
\quad d=\operatorname{clip}(u-x,-\epsilon,\epsilon),
\quad x^{(t+1)}=\operatorname{clip}(x+d,0,1).
$$

默认 $\epsilon=8/255$、$\alpha=2/255$、100 次更新，批次大小上限为 16。这些数值对应附录 A.2。图像始终使用 `[0,1]` 像素尺度；编码器应在自身的可微预处理内处理模型归一化，不能先量化成 PIL 再切断输入梯度。

```text
保存原图；固定目标、查询池、代理记忆嵌入
current = 原图；best = 原图
以完整查询池计算 best_loss
重复 steps 次：
    用独立的带种子随机数生成器抽取查询索引
    对当前图像计算本步批次目标和输入梯度
    执行符号梯度下降、L∞ 投影、像素裁剪
    在同一个完整查询池上评价更新后的图像
    若完整查询池目标更低，则更新 best
返回 best、完整查询池 best_loss、实测 L∞ 范数及历史
```

零初始化、索引无放回抽样和“保留完整池最佳点”是整理版沿用的工程选择，不是额外的论文实验结论。初始点及最后一次更新均参加比较；因此返回值可能仍是原图，不能把返回图像本身解释为攻击成功。`history` 的 `batch_loss_before_step` 为更新前批次目标，`validation_loss` 和分项为更新后完整池目标；step 0 只记录初始完整池目标。

调用者须提供确定性的、可微的单个图像编码器，包括其真实预处理；文本与记忆嵌入已预计算，优化中固定不变。该接口不访问受测私有记忆库，也不替调用者证明代理记忆的合法来源。固定随机种子只控制批次索引；真实 GPU/模型的确定性还需调用者配置。

本包不实施跨编码器加权，也不估计论文存在歧义的梯度方差权重；不实施未完全指定权重与聚合方式的 hard-negative 辅助项。没有这些组件的结果应称“单编码器核心子问题”，不能称完整 Visual Inception 集成攻击。

浮点运算可能产生舍入差；float32 的投影验证采用 `实测范数 <= epsilon + 1e-6`。该容差是数值检查容差，不是把论文预算扩大为新的实验设置。

## 3. RDS 和三次判别规则

对原始记忆条件与去掉指定记忆的反事实条件，先由同一 Agent 生成成对计划 $P_{org},P_{ctr}$，再以同一 SBERT 编码：

$$
\operatorname{RDS}=1-\cos(\operatorname{SB}(P_{org}),\operatorname{SB}(P_{ctr})).
$$

数值函数接收两个 `[1,D]` 向量，不生成计划或调用 SBERT。数学范围为 `[0,2]`；计算将余弦的微小舍入越界裁回 `[-1,1]`。不能错误地把 RDS 限制到 `[0,1]`，也不能用图像嵌入替代计划嵌入。

三次 judge 调用各产生一个整数 $s_i\in\{1,2,3,4,5\}$。默认规则为：

$$
flag=(\operatorname{RDS}>0.7)\ \lor\ 
\left[\sum_{i=1}^{3}\mathbf1(s_i\le2)\ge2\right].
$$

三次投票的第二项与“评分中位数 $\le2$”完全等价。RDS 等于 0.7 不触发该分支；评分等于 2 属于不相关票。默认阈值、三次调用和多数投票有 §4.2 的依据。布尔值、浮点评分、缺少/多余调用结果或非 1–5 的整数均被拒绝。

这里实现的是给定真实输入后的评分与合并规则，未实现成对计划生成、非破坏性记忆移除、重新检索、judge 提示词或模型调用。因此仅有本函数不能宣称已经完成 System 2，不能证明因果归因或校准质量。

## 4. 扩散数值工具

论文 §4.1 的前向关系为：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon.$$

本包还提供原工程采用的 DDIM 确定性单步，用于检查反向数值关系。它是明确公开的实现选择，不是论文单独指定的完整净化算法。给定预测噪声 $\hat\varepsilon$：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\varepsilon}{\sqrt{\bar\alpha_t}},\qquad
x_s=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\hat\varepsilon,\quad s<t.$$

`alpha` 和 `alpha_prev` 均为累计 $\bar\alpha$，不能传单步 $\alpha_t$。DDIM 要求 $0<\bar\alpha_t<1$ 且 $\bar\alpha_t\le\bar\alpha_s\le1$，不追加随机噪声（eta=0），不裁剪预测 clean sample。

`epsilon` 表示模型输出为噪声；`sample` 表示模型输出为 $\hat x_0$；`v_prediction` 使用 $v=\sqrt{\bar\alpha_t}\varepsilon-\sqrt{1-\bar\alpha_t}x_0$ 的标准换算。必须与实际检查点的 scheduler 类型匹配，不能只因函数支持三种类型就认为同一模型都可使用。

本包不包含 SD VAE/UNet、调度时间表、良性参考构造、轨迹方差、双阈值停止或模型检查点，因此不构成 System 1，也不能称原版 DiffPure 实现。
