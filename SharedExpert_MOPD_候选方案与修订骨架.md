# SharedExpert-MOPD：四个候选方案与修订论文骨架

> 研究设计版本：2026-10-19。保留原始《SharedExpert_MOPD_论文骨架.md》不变。
> 本文区分已核验的文献事实、本次提出的设计、带假设的数学结论和待完成的实验。所有性能表均为实验计划，不代表已经获得结果。
> 配套文件：[文献核验](SharedExpert_MOPD_文献核验.md)、[完整数学审查与证明](SharedExpert_MOPD_数学审查.md)、[方法与实验审查](SharedExpert_MOPD_方法与实验审查.md)。

## 0. 先选择研究路线

你的想法可以保留，但不宜把“shared expert + specialist 残差”本身作为主创新。更有价值的问题是：**在当前学生前缀上，如何判断 specialist 有值得引入的增量，以及引入多少才合适？**

下面四条路线可以独立选择；不建议把它们全部叠加为主方法。

| 方案 | 核心决策 | shared 权重 | specialist 权重依据 | 额外成本 | 论文潜力与主要问题 |
|---|---|---|---|---|---|
| **A：固定／缓慢调度的共享锚** | 先确定哪个任务交给哪个专家 | 同轮固定，可跨轮 warm-up | 领域标签或校准子群表现 | 低；通常 generalist + 一个 specialist 评分 | 最容易实施；与 ChemOPD 很近，适合基线或可行性验证 |
| **B：可靠性筛选 + token 对齐门控** | 专家有资格，再看当前 token 是否有可学差异 | 随 token 变化，有下限 | 子群可靠性 × Fisher/中心化对齐 | 中；不需要大量教师续写 | 工程上折中；与 ExpertAlign、TrustMOPD 重叠较多，需要扎实机制实验 |
| **C：前缀条件的可验证增益路由** | 从同一学生前缀出发，谁比 generalist 更可能完成任务 | 随 prefix/step 变化，有下限 | 预测的相对续写收益及其不确定性 | 较高；需要 probe、校准、轻量 router | **较适合主论文探索**；必须验证“续写优势”是否能变成学生可利用的 token 指导 |
| **D：共同续写下的一步价值路由** | 只改变下一动作，后面用相同策略，隔离真正的局部收益 | 随 prefix 变化，有下限 | 当前动作的可验证 advantage | 高；需要 continuation 采样或 critic | **适合理论优先**；可给策略改进命题，但估值、覆盖和学生拟合条件更苛刻 |

选择建议：先用 A 检查专家互补性；资源有限选 B；想把创新集中在一个明确问题上选 C；愿意付出价值估计成本、希望理论更直接对应任务奖励则选 D。可以用 B 做原型、C 做正式方案，但应分别报告，不能把 B 的便宜成本写到 C 上。

## 1. 文献核验后，原骨架必须修正的地方

### 1.1 最接近的已有工作

| 工作 | 已有内容 | 本文不能再声称什么 |
|---|---|---|
| [MOPD](https://arxiv.org/abs/2606.30406) | 学生 rollout；按 prompt 领域选择教师；token reverse-KL | 原始 MOPD 不是“三教师等权平均”；硬路由与均匀软聚合须分开 |
| [ChemOPD](https://arxiv.org/abs/2609.33838) | generalist anchor、相对 generalist 的 specialist correction、warm-up | 不能说首次引入 generalist-relative residual |
| [MOPD-Router](https://arxiv.org/abs/2609.30837) | teacher−base 与 teacher−student 对齐的 token gate | 原 gate 更换 reference，不能当作从零提出的新路由原理 |
| [TrustMOPD](https://arxiv.org/abs/2609.23697) | 相对 reference 的中心化位移、校准尺度、token 权重 | “动态权重”“相对参考”“可靠性”都不足以单独构成创新 |
| [VG-OPD](https://arxiv.org/abs/2609.15404) | verifier 判定专家是否解决学生失败 criterion，再定位 token | C/D 必须与其正面比较；差异在同一 prefix、generalist 参照和不确定性处理 |
| [LEGO-OPD](https://arxiv.org/abs/2610.00333) | 语言先验 + 视觉残差；prefix 自适应系数与 KL 预算 | 它不是固定因子方法；KL-budget 本身不是本提案独有 |
| [Δ-MOPD](https://arxiv.org/abs/2610.10460) | teacher 相对自身 base 的 shift，重新锚定学生初始化 | generalist reference 和 teacher precursor 应区分；同源下部分对照可能退化为相同目标 |
| [H-OPD](https://arxiv.org/abs/2607.02592) | 多模态异构教师、token 置信仲裁 | 仅改成 token 权重不够，需要证明增益判据更有效 |

以上均为截止日期前的原始论文。详细核验范围、版本与机制见配套文献报告。这里只提出有限检索支持的候选缺口，不宣称全球首次。

### 1.2 四个关键数学纠正

1. **残差组合与几何聚合精确等价。**

   \[
   z_g+\sum_i w_i(z_i-z_g)=(1-\sum_iw_i)z_g+\sum_iw_i z_i.
   \]

   同权重下不是两个算法；“residual vs geometric pooling”应是数值一致性检查，不是性能消融。

2. **公共分量也会在两个 specialist 相减时消失。**因此不能以“公共语言分量很大”为由，推出残差路由一定比硬路由鲁棒。
3. **对齐不等于正确。**学生等于 generalist 时，原式的两个向量相等，所有非零 specialist 残差都通过；错误专家同样通过。
4. **token KL 不等于最终准确率。**需要状态分布、序列长度、学生拟合误差和奖励范围，才可以推出相应的奖励偏离界。

另一个容易混淆的点：generalist 是完整多模态模型，不是“纯语言先验”；specialist−generalist 同时包含能力、风格、温度和错误差异，不是自动提纯后的能力向量。

## 2. 所有候选共用的设定

### 2.1 三个教师如何获得和分工

| 教师 | 主要职责 | 建议训练和诊断 |
|---|---|---|
| \(T_g\)：shared generalist | 通用视觉问答、常规推理、指令、格式、广域保留 | 优先采用已验证的通用 VLM；广域 SFT 或 RL 都可，不必为了形式强制再训一个 RL 教师 |
| \(T_p\)：perception specialist | 计数、空间、属性、细粒度定位、视觉证据读取 | 与 g 同源初始化；可验证感知数据；检验在这些子群上是否真正超过 g |
| \(T_r\)：reasoning specialist | 给定视觉证据后的数学、逻辑、STEM 多步推理 | 优先仍用能看同一图像的 VLM；检验是否只靠语言捷径、是否损伤感知 |

初版优先同一模型家族、词表、输入格式、温度与共同起点。纯 LLM 推理教师需要图像转写或其他接口，会引入额外信息瓶颈和异构对齐问题，宜作为单独扩展。

学生初始化选同一通用 checkpoint 或与其同源的 SFT checkpoint，并在所有基线中保持一致。教师强度与“是否 RL”不是一回事，应由留出集确认。若已有 generalist 就合格，不必额外制造第四个训练阶段。

**先做教师能力矩阵。**只有当 p、r 分别在一些子群可靠胜过 g，且胜出样本不完全重合，路由才有可用空间。观察不到互补性时，应先改专家训练或数据，而不是继续调 gate。

### 2.2 “每轮固定”究竟固定什么

用 \(k\) 表示一批学生 rollout 及其随后的优化阶段，用 \(h_t=(x,y_{<t})\) 表示生成前缀。

| 量 | 推荐处理 |
|---|---|
| 教师模型参数 | 整个蒸馏阶段冻结；它们不随每轮学生更新 |
| router 参数／校准统计 | 至少在同一轮冻结；必要时在轮次之间用新前缀刷新 |
| 总 specialist 上限 \(B_k\) | A 可 warm-up；B/C/D 首版固定，后续只比较慢速调度 |
| 具体 \(w_{g,t},w_{p,t},w_{r,t}\) | A 对同一 prompt 固定；B/C/D 随 prefix 或推理 step 改变 |
| 已采样这批 rollout 对应的目标和权重 | 学生内层优化期间缓存并 stop-gradient，避免目标追着学生变化 |

**rollout 始终由学生生成。**这里讨论的是学生生成后，哪些教师在其前缀上提供蒸馏监督，不是把多个教师混合起来代替学生生成训练轨迹。

### 2.3 A/B/C 的统一目标

在共享词表、相同前缀和统一温度下，令

\[
\Delta_{i,t}(v)=\log p_i(v\mid h_t)-\log p_g(v\mid h_t),\quad i\in\{p,r\}.
\]

\[
q_t(v)\propto p_g(v\mid h_t)
\exp\!\left(\sum_i w_{i,t}\Delta_{i,t}(v)\right),
\quad w_{g,t}=1-w_{p,t}-w_{r,t}.
\]

要求

\[
w_i\ge0,\qquad w_p+w_r\le B_k<1,
\qquad w_g\ge1-B_k>0.
\]

“always-on”的实质是这个**有效权重下限**。式子前面写了一个 \(z_g\)，并不意味着 generalist 的有效权重永远为 1。

本轮固定状态分布 \(d_k\) 和目标后，学生优化

\[
\mathcal L_k(\theta)
=\mathbb E_{h\sim d_k}\operatorname{KL}(\pi_\theta(\cdot|h)\Vert\operatorname{sg}[q_k(\cdot|h)]).
\]

这是每轮的 fixed-prefix surrogate。不能把它的梯度直接写成包含全部未来状态分布导数的完整 trajectory objective 梯度。

初版只设置一个总预算 \(B\)，例如在开发集比较 \(\{0.25,0.5,0.75\}\)。这些是搜索起点，不是已验证最优值。固定状态下，额外的 \(\mu\operatorname{KL}(\pi_\theta\Vert p_g)\) 等价于把全部 residual 权重缩为 \(w_i/(1+\mu)\)，并整体缩放 loss；因此先不同时搜索 \(B\) 与 \(\mu\)。

## 3. 候选 A：固定／缓慢调度的共享锚

**适合：想迅速知道 shared anchor 有没有用，或者算力有限。**

由任务域或校准子群给出 \(a_p(x)+a_r(x)=1\)，不确定或无需 specialist 的样本设两者为 0。例如感知题选择 p，纯推理题选择 r，混合题用校准得分确定比例。

\[
w_{i,k,t}=B_k a_i(x),\qquad
w_{g,k,t}=1-B_k\sum_i a_i(x).
\]

两个可独立比较的版本：

- **A1 固定：**全程 \(B_k=B\)。这是最干净的因果对照。
- **A2 warm-up：**\(B_k=B\min(1,k/K_w)\)，前期偏向 generalist，之后增加 specialist。\(K_w\) 用开发集选定；不预设学生越强就一定越应该听 specialist。

若要动态跨轮调预算，应依据开发集上的能力保留与特化收益，而非训练 loss 大小。例如通用保留指标超过预设容忍退化时降低 \(B_k\)。这是控制启发式，不能保证下游分数单调。

**优点：**权重可解释、无需训练 router；top-1 版本只需 g 加一个 specialist 评分，混合题若同时启用 p、r 则需三教师评分。

**缺点：**同一题内部不能从读图切换到推理；领域路由仍可能错。与 ChemOPD 很接近，单独作为主要创新较弱。[ChemOPD 原文](https://arxiv.org/html/2609.33838v1)

**可以证明：**锚点收缩的条件风险改善、固定权重几何池恒等式。不能声称发现了新的残差聚合机制。

## 4. 候选 B：可靠性筛选 + token 对齐门控

**适合：希望逐 token 利用两个 specialist，但不想支付大规模续写探针成本。**

### 4.1 先判断“有没有资格”

在训练之外的校准子集，按可观察来源／任务能力划分少量子群 \(d\)。同题同解码预算评估教师，得到

\[
\widehat u_i(d)=\frac1{n_d}\sum_{x\in C_d}[R_i(x)-R_g(x)].
\]

按题目估计下置信界，定义

\[
c_i(x)=[\operatorname{LCB}(u_i(d(x)))-\delta]_+.
\]

\(c_i\) 只是子群平均可靠性，不是逐题正确性 oracle。域标签可见性必须与基线一致；未知子群不应伪装成已校准。

### 4.2 再判断“当前有没有可学的差异”

令 \(m_t=\pi_{\theta_k}(\cdot|h_t)\)，定义中心化算子与内积

\[
C_m[f]=f-\mathbb E_m f,\qquad
\langle f,g\rangle_m=\mathbb E_m[fg].
\]

\[
e_i=C_m[\log p_i-\log p_g],\quad
d_i=C_m[\log p_i-\log\pi_{\theta_k}],
\]

\[
\gamma_i(h_t)=\left[\frac{\langle e_i,d_i\rangle_m}
{\|e_i\|_m\|d_i\|_m+\epsilon}\right]_+,
\quad s_i(h_t)=c_i(x)\gamma_i(h_t).
\]

零范数时取 \(\gamma_i=0\)。中心化／Fisher 度量使 gate 不受 logits 任意常数平移影响，但没有让它自动获得 correctness guarantee。

### 4.3 给出实际权重

令 \(S_t=s_p+s_r\)，取 \(\tau>0\)：

\[
\boxed{w_{i,t}=B_k\frac{s_i(h_t)}{\tau+S_t},\qquad
w_{g,t}=1-B_k\frac{S_t}{\tau+S_t}.}
\]

两位专家都不合格时自动退回 g；只有 p 合格则只加感知残差；两者都合格时可以同时贡献。\(\tau\) 控制“证据弱时少用专家”，须按收益分数尺度选择，不能任意比较不同量纲的 score。

**优点：**便宜、可解释；纠正原 gate 最明显的问题。

**缺点：**子群可靠性不能保证当前 token 正确；alignment 仍可能误判。主创新不能只是将 ExpertAlign 的 base 换成 generalist。[MOPD-Router](https://arxiv.org/abs/2609.30837)、[TrustMOPD](https://arxiv.org/abs/2609.23697)

**可证明范围：**权重扰动界、锚点偏离界，以及明确 FP/FN 条件下的稀疏门控风险改善。实际正确率提升仍需实验。

## 5. 候选 C：前缀条件的可验证增益路由

**适合：把论文集中在“specialist 何时真正比 generalist 有用”这一问题上。**

### 5.1 定义不依赖 logit 大小的收益

在相同学生前缀 \(h\) 上，教师分别继续生成，使用同一个有界验证器：

\[
V_i(h)=\mathbb E_{c\sim T_i(\cdot|h)}R(x,y_{<t}\oplus c),
\qquad u_i(h)=V_i(h)-V_g(h).
\]

\(R\in[0,1]\) 可以是可验证答案、定位／计数正确性、或明确加权的可验证子目标。统一 continuation 长度和采样预算，截断与无法解析的答案需预先定义处理规则。

它与低熵、位移大、与学生差距大不同：这些只能是预测特征；\(u_i\) 的监督来自实际任务结果。

### 5.2 如何把昂贵 probes 变成可用 router

1. 先在少量训练题上做 pilot，每条学生轨迹取少数不同阶段的前缀；不要对全部 token 做全教师续写。
2. 用 g、p、r 的重复续写估计 \(u_i(h)\)，训练小预测器输出 \(\widehat u_i(h)\)。输入可包含学生隐藏状态、问题特征和教师分布统计；不使用测试答案。
3. 按**题目**划分 router 训练、校准、开发和最终测试，不能把同一题的不同前缀拆进两边。
4. 留出校准估计误差尺度 \(b_i(h)\)，然后取

   \[
   s_i(h)=[\widehat u_i(h)-b_i(h)-\delta]_+.
   \]

5. 将 \(s_i\) 放入方案 B 的权重公式。router 在新轮次的前缀上运行；本轮学生优化时缓存权重。

**统计边界：**少量 Monte Carlo 的样本标准差，不自动构成有效置信界；全成功样本也不代表零不确定性。固定前缀的采样误差界不等于新前缀的预测误差界。校准覆盖通常只能在给定分布／子群和交换性条件下解释；on-policy 分布变化时必须复测。样本太少时严格分布无关下界可能全部为负，应报告覆盖不足，而不是把统计保证写强。

实现可从 step 级开始：只在句子／推理步骤边界计算，再在该段使用相同权重；如果采用这一版，正文写 step-adaptive，不要称为每 token 独立验证。token predictor 与 step predictor 应做对照。

### 5.3 两个 specialist 何时参与

| 当前前缀诊断 | 权重行为 |
|---|---|
| g 已足够，或者两位专家增益都不确定 | \(s_p=s_r=0\)，纯 generalist 目标 |
| 主要需要精细读图，p 的下界为正 | \(w_p>0,w_r=0\) |
| 证据已明确、需要逻辑／计算，r 的下界为正 | \(w_r>0,w_p=0\) |
| 两种条件增益都可靠 | 同时使用，但总权重不超过 \(B\) |
| 两专家互相冲突 | 依校准增益选择或混合；测试 top-1 与 top-2，不能仅靠平均宣称“互补” |

上述角色由实测收益判定，不是按“因此”“图中”这类词表规则强行分配。感知可能在后期复核时再次重要，推理也可能在读图前决定观察什么。

### 5.4 必须验证的桥接假设

**整段续写更强，不等于当前 token 对学生更有用。**专家可能选择一个只有自己能继续做对的分支，学生学到开头后却无法完成。又或者教师们下一 token 相同，区别只在后续策略，当前混合并无收益。

一个两步反例：g 和学生在第一步选择分支 A 的概率为 0.6，而且只有 A 能由它们继续做对；specialist 选择 A 的概率只有 0.1，但自己能完成 A、B 两个分支。因此 specialist 的整段成功率为 1，高于 g 的 0.6。然而二者等权几何池选择 A 的概率只有约 0.290；交给原学生继续，成功率反而降低。这是构造反例，数值已检查，不是实际模型实验。

因此抽取诊断前缀：分别从 g、specialist、混合目标采样当前动作，然后全部接同一个冻结学生续写，检验 C 的 score 是否预测了真正可利用的局部改善。这是 C 必须做的机制验证；若关联很弱，应考虑 D，而不是硬写性能定理。

**相对已有工作的候选区别：**同一学生 prefix、specialist 相对 generalist 的价值、校准不确定性和回退。VG-OPD 使用原 prompt 的专家 probe 与学生失败 criterion；应分别复现其原则和其完整训练配方，不能把本文改动后的版本称为原论文复现。[VG-OPD](https://arxiv.org/html/2609.15404v1)

## 6. 候选 D：共同续写下的一步价值路由

**适合：希望数学结论直接涉及任务奖励，而不仅是 logit 风险。**

### 6.1 改掉 C 的价值定义

选定冻结参考策略 \(b\)。最简单取 generalist \(b=g\)。定义

\[
Q_b(h,v)=\mathbb E[R\mid\text{当前选 }v,\text{其后统一由 }b\text{继续}],
\]

\[
V_b(h)=\mathbb E_{v\sim b}Q_b(h,v),\qquad
A_i^b(h)=\mathbb E_{v\sim p_i}Q_b(h,v)-V_b(h).
\]

这里各候选的**后续策略相同**，因此能隔离当前分布的贡献；它不再把 specialist 自己的长程续写能力算到当前 token 上。

不需要枚举词表：可以从各候选分布采样动作、统一续写，估计 action-value，或用这些数据拟合 critic。但 critic 误差和采样开销必须单列。

若有 \(N\) 个前缀、\(K\) 个 specialist、每个候选采样 \(M\) 次、平均剩余长度 \(L\)，直接估计约需 \(N(K+1)ML\) 个 continuation tokens，另加 prefill；g 的对照样本可以在不同 specialist 比较中复用。对有界奖励，分辨量级为 \(\epsilon\) 的微小 advantage，保守样本量通常随 \(\log(K/\delta)/\epsilon^2\) 增长。单 token 的收益往往很小，这可能比教师评分本身更昂贵；critic 省采样但引入估值误差。

### 6.2 使用概率混合，不再使用 logit 几何池

当 \(b=g\)，令 \(s_i=[\widehat A_i^g-b_i-\delta]_+\)，按 B 的公式取权重，然后

\[
\boxed{q_D(v|h)=w_g(h)p_g(v|h)+w_p(h)p_p(v|h)+w_r(h)p_r(v|h).}
\]

学生仍可用 \(\operatorname{KL}(\pi_\theta\Vert q_D)\) 蒸馏。注意：**这是 reverse-KL 到概率混合目标，不是加权的多个 reverse-KL**；后者对应几何池。这个差异是 D 的实质设计，不能在实现中替换。

### 6.3 能证明什么

以下是经典 policy improvement／performance-difference identity 在多教师概率混合中的实例化，不把该基础定理作为新的理论发现；本方案需要研究的是估值是否可信，以及学生能否保留这个增益。

若所有可能访问的前缀上，启用的专家都满足 \(A_i^g(h)\ge0\)，则

\[
\mathbb E_{v\sim q_D}Q_g(h,v)-V_g(h)
=\sum_i w_i(h)A_i^g(h)\ge0.
\]

由有限时域 performance-difference identity，

\[
J(q_D)-J(g)=\mathbb E_{h\sim q_D}\sum_t\sum_iw_i(h_t)A_i^g(h_t)\ge0.
\]

这是**组合策略相对 generalist 的有条件改进**。若学生与组合策略的完整轨迹 KL 不超过 \(\varepsilon_{\rm dist}\)，奖励范围长度为 \(R_{\rm span}\)，则

\[
J(\pi_\theta)\ge J(g)+\underbrace{[J(q_D)-J(g)]}_{\text{组合策略增益}}
-R_{\rm span}\sqrt{\varepsilon_{\rm dist}/2}.
\]

严格学生改进还要求组合增益大于拟合误差项。只在少量校准前缀上成立的下界，不能自动推广到所有 q 可访问状态。

**若一定要直接比较朴素 MOPD：**令 \(b\) 为预先冻结的具体 MOPD 学生，按 \(Q_b\) 估计所有教师相对 b 的一步 advantage。需找到满足 shared 下限且 \(\sum_jw_jA_j^b(h)\ge0\) 的权重。若每个相关状态都可行，可给相对 b 的组合策略改进界；但可行性不是免费的，g 可能比 b 差。此时“回退 g”不保证安全，“回退 b”则改变 always-on g 的设计。应明确接受哪种假设／回退规则，不能同时无条件承诺二者。

**优点：**正确性假设、权重和奖励结论之间联系最直接。

**缺点：**估值成本高，保守下界可能让 gate 几乎不激活；一步收益可能很小、难以估计。其理论基础来自已有策略改进思想，而非本提案新发现：[Kakade–Langford 的 conservative policy iteration](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)、[High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.pdf) 与 [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) 分别提供改进、置信评估及不确定时依赖基线的相关思路。它们的保证依赖各自假设，不能直接移植为大模型 critic 的保证；D 的研究贡献需要落在多教师可利用价值估计及其蒸馏验证上。

## 7. 可选稳健组件：怎样定义 shared 权重预算

### 7.1 建议先固定总上限，后验证是否需要动态预算

不建议一开始就同时引入：可学习 shared gate、每域归一化、EMA anchor、第四专家、梯度投影和动态预算。先回答权重判据是否有效，否则结果难归因。

对 A/B/C，可以在既定方向上增加 KL 限制：设 \(a_i=s_i/\sum_js_j\)，\(r_t=\sum_i a_i\Delta_i\)，

\[
q_\beta\propto p_g e^{\beta r_t},\qquad
\beta_0=B_k\frac{\sum_i s_i}{\tau+\sum_i s_i}.
\]

选取

\[
\beta_t=\max\{\beta\in[0,\beta_0]:\operatorname{KL}(q_\beta\Vert p_g)\le\varepsilon_{\rm anchor}\},
\quad w_i=\beta_ta_i.
\]

固定 \(r_t\) 时，KL 对 \(\beta\ge0\) 单调不减，可用一维二分；残差为常数时目标等于 g。它控制的是目标偏离，不是保证 benchmark 不掉分。LEGO-OPD 已有相关自适应 KL 校准，这里作为借鉴组件和消融，不作为单独主创新。[LEGO-OPD §3 与附录 A](https://arxiv.org/html/2610.00333v1)

### 7.2 为什么不能给所有专家固定 0.25 就结束

固定权重适合作为 A1，但不能识别：p/r 在当前题其实弱于 g、学生已经超过某教师、当前 prefix 已不可恢复、或某个教师仅靠更尖锐的分布支配梯度。动态 gate 的价值必须通过这些场景的错误率下降来证明。

### 7.3 为什么不直接最大熵／最小熵选专家

低熵可能来自更准，也可能来自过度自信；教师温度和输出风格不同会改变熵。可以把熵路由作为强而便宜的 baseline，不能把熵直接定义成真实可靠性。

## 8. 要不要再加专家

**默认：先不加，先做 g+p+r。**本地轻量配置能确认通用模型和评测／Vision-OPD 探针，但不能据此确认已有合格的三个专门教师；需先补教师能力矩阵。

| 新增角色 | 何时值得加入 | 为什么不能现在默认加入 |
|---|---|---|
| OCR／文档专家 | 三教师在小字、表格、文档结构上共同失败，新增教师在这些样本明显更强 | 感知专家可能已经覆盖；重复专家增加成本和校准困难 |
| 视频／时序专家 | 主任务确有视频、跨帧因果／时序关系，且有足够训练与评测资源 | 会改变输入分布、计算预算和论文范围 |
| 工具／程序专家 | 错误主要来自可执行计算，任务允许工具动作与执行监督 | 需要统一动作空间，不能直接混合不兼容 token 策略 |
| verifier／critic | C/D 的价值标签太贵，需要预测或检查收益 | 优先作为评估／路由组件，不一定成为第四个生成教师 |

新增专家的准入指标可定义为

\[
\Delta_{\rm oracle}(j)
=\mathbb E_x\left[\max\{V_g,V_p,V_r,V_j\}-\max\{V_g,V_p,V_r\}\right].
\]

它衡量潜在覆盖，不代表能全部蒸馏给学生。最好再报告“现有教师全错但新教师正确”的比例、错误相关性，以及新增训练／评分／probe GPU-hours。先以留出探针估计增量和不确定性，只有存在可识别的收益才扩展。

新增专家后固定总预算 \(\sum_iw_i\le B\)，而不是每加一个专家再加一份固定权重；后者会让 shared 权重下降甚至变负，混淆“更多知识”与“更弱锚定”。

## 9. 数学上怎样比较朴素 MOPD

### 9.1 必须先指定比较对象

- **MOPD-hard：**按 prompt 领域选一个教师，原始 MOPD 的核心路由设置。
- **Uniform-geometric：**三个教师 logits 等权，等价于加权 reverse-KL 的目标。
- **Uniform-arithmetic：**三个教师概率等权；与上者不同。
- **Anchor-fixed：**与本文相同 shared 预算，但不做质量路由。

不存在对所有教师、任务和预算均成立的“本文优于 MOPD”定理：若 specialist 是完美目标、generalist 有系统偏差，强行保留 g 会损害特化性能。理论应说明**何时有优势、优势与什么量有关、何时失效**。

### 9.2 定理一：几何聚合恒等式，用于厘清方法

固定状态与权重时，\(q_w\propto\prod_jp_j^{w_j}\)，有

\[
\sum_jw_j\operatorname{KL}(p\Vert p_j)
=\operatorname{KL}(p\Vert q_w)-\log Z_w.
\]

这是精确恒等式，不是一阶近似。硬路由是 one-hot 权重特例。贡献在“权重怎么来”，不在代数重写。

### 9.3 定理二：锚定收缩什么时候严格优于单教师

在去掉 logit 常数方向的空间，设理想目标为 \(z^\star\)，

\[
z_g=z^\star-s+\epsilon_g,\qquad z_i=z^\star+\epsilon_i,
\]

噪声零均值，\(V_g=\mathbb E\|\epsilon_g\|^2\)、\(V_i=\mathbb E\|\epsilon_i\|^2\)、\(C=\mathbb E\langle\epsilon_g,\epsilon_i\rangle\)。对固定权重 \(\beta\)，

\[
\widehat z_\beta=(1-\beta)z_g+\beta z_i,
\]

\[
\mathcal R_\beta=(1-\beta)^2(\|s\|^2+V_g)+\beta^2V_i+2\beta(1-\beta)C.
\]

硬路由风险 \(\mathcal R_{\rm hard}=V_i\)。令

\[
D=\|s\|^2+V_g+V_i-2C,
\]

则当 \(\beta<1\) 时，严格改善的条件为

\[
\boxed{(1-\beta)D<2(V_i-C).}
\]

独立噪声 \(C=0\) 时，最优无预算约束权重为

\[
\boxed{\beta^*=\frac{\|s\|^2+V_g}{\|s\|^2+V_g+V_i}.}
\]

这给出明确直觉：generalist 偏差大，应更多依赖 specialist；specialist 噪声大，应更多依赖 g；教师错误高度相关时，平均的收益会变弱。加上 shared 下限后要把最优权重投影到可行区间，不能再保证达到无约束最优风险。

**边界：**这是目标估计 MSE，不是准确率定理；神经网络固定 checkpoint 的“噪声”需在明确的随机训练／估计模型中定义。固定权重结论不能无条件套到依赖同一噪声的自适应 gate。可用固定参考下的 Fisher 加权范数解释局部 KL，但仍不能跳到 benchmark 分数。

### 9.4 定理三：路由错误的影响由预算控制

令 \(\operatorname{osc}(f)=\max_vf(v)-\min_vf(v)\)，比较两组 specialist 权重 \(w,\widetilde w\)，定义

\[
D_w=\sum_i|w_i-\widetilde w_i|\operatorname{osc}(\Delta_i).
\]

则

\[
\boxed{\operatorname{KL}(q_w\Vert q_{\widetilde w})\le D_w^2/8,\qquad
\operatorname{TV}(q_w,q_{\widetilde w})\le\min(1,D_w/4).}
\]

证明由指数倾斜、Hoeffding 引理与 Pinsker 不等式得到。它表明“限制权重错误和残差幅度，可以限制目标扰动”，并不表明 gate 的误判率更低。后者要测。

### 9.5 定理四：通用奖励偏离要加序列与拟合误差

假设所有相关状态上 \(\operatorname{osc}(\sum_iw_i\Delta_i)\le B_z\)，序列最多 \(H\) 步，记完整轨迹分布为 \(P_q,P_g,P_s\)，则

\[
\operatorname{KL}(P_q\Vert P_g)\le HB_z^2/8.
\]

若另有 \(\operatorname{KL}(P_s\Vert P_q)\le\varepsilon_{\rm dist}\)，则对范围长度 \(R_{\rm span}\) 的序列奖励，

\[
\boxed{|J(s)-J(g)|\le R_{\rm span}
\left(\sqrt{\varepsilon_{\rm dist}/2}+B_z\sqrt H/4\right).}
\]

可与奖励最大范围取较小值。仅在训练 minibatch 上观察到平均 token KL 小，不足以证明这个全轨迹界的条件；尤其是通用评测输入需要相应覆盖。应保留通用训练／回放数据，并在留出通用前缀上诊断。

### 9.6 对 C 的路由界与对 D 的奖励界要分开

C 可以在 \(|\widehat u_i-u_i|\le e\) 等条件下证明，对同一权重可行集的线性代理效用，最优化估计效用的 router 相对 oracle 的 regret 至多 \(2Be\)。这只适用于实际求解该优化问题的 router；上面的平滑正分数权重不是自动的精确 argmax。也不能把 teacher continuation 的线性代理效用等同于几何目标 reward。

D 则可以在有效一步 advantage 和状态覆盖条件下，证明组合策略相对参考的改进，见 §6。若论文特别重视“数学保证优于基线”，优先认真评估 D 的可实施性，而不是把 C 的代理界包装成奖励定理。

完整证明、稀疏 gate 的误检／漏检风险条件、反例和梯度空间区别见[数学审查](SharedExpert_MOPD_数学审查.md)。

## 10. 修订后的完整论文骨架

### 10.1 标题按选择的路线确定

- A：*Generalist-Anchored Distillation for Multimodal Capability Integration*。
- B：*Reliable Residual Routing for Shared-Expert Multimodal Distillation*。
- C：*When Should Specialists Teach? Generalist-Relative Value Routing for Multimodal On-Policy Distillation*。
- D：*Value-Guided Shared-Expert Distillation with Conditional Policy Improvement*。

“Common-Mode Rejection”不建议作为标题，因为原先相关推论并未成立。“Safe”也不宜在仅有目标 KL 界时无条件使用。

### 10.2 Abstract：五句模板

1. 多模态能力整合需要吸收感知、推理等专家，同时维持通用表现。
2. 任务域、置信度和教师位移不能直接确定专家在当前学生前缀上的增量价值。
3. 提出【B/C/D 选定方法】：用【所选可靠性或价值信号】分配 specialist 监督，并保留 generalist 参考。
4. 在明确假设下给出【选定的两到三个理论结论】，分析适用条件和失败边界。
5. 在统一教师池与预算下评测感知、推理、通用保留和混合任务；结果填写【待实验】，不提前写超过所有基线。

### 10.3 Introduction：五段

1. 多模态能力互补的需求，以及 student-prefix supervision 的价值。
2. 经验问题：同一任务内部专家收益会变化；强 specialist 仍可能提供错误或学生无法利用的指导。
3. shared generalist 的作用是明确参照与回退；必须判断相对增量，而非把差异当作能力。
4. 概述选定路线与最关键设计，不把可选模块一起堆进主算法。
5. 贡献限定为问题诊断、可复现决策规则、带条件分析、受控实验；不写“首次 shared expert”。

### 10.4 Related Work：四组

1. OPD 与多教师整合：GKD、MOPD、Open-MOPD、DN-MOPD。
2. 教师选择与可靠性：MOPD-Router、TrustMOPD、VG-OPD、H-OPD、Compatibility Matters。
3. generalist／残差／能力保留：ChemOPD、Δ-MOPD、CaMOPD、UCMOPD、SF-MOPD。
4. 多模态证据：LEGO-OPD、VA-OPD、VGS；shared expert 的架构启发简述 DeepSeekMoE。

若选 D，增加 §6 所列 conservative policy iteration、High Confidence Policy Improvement 和 SPIBB，并区分直接采样与离策略估值的假设。架构 MoE 的共享专家不能直接证明教师空间分解正确。[DeepSeekMoE](https://arxiv.org/abs/2401.06066)

### 10.5 Problem and Diagnosis

- 清楚定义同词表、教师输入、rollout 分布、参考策略、reward/verifier 和预算。
- 用校准子集画“教师 × 能力子群”矩阵；报告互补覆盖，而不只比较平均分。
- 用反例或真实案例展示：低熵、位移大、ExpertAlign 通过，但专家不更正确。
- C/D 增加相同 prefix 的对照，说明 prompt 能力标签为什么不够。

### 10.6 Method

1. 教师构建与学生初始化。
2. 所选方案的效用／可靠性定义。
3. shared/specialist 三个实际权重及全拒绝回退规则。
4. 几何池或概率池及相应训练损失。
5. 冻结、刷新、校准 split 与复杂度。
6. 可选 budget/step 粒度放消融，不掩盖主线。

### 10.7 Theory

正文选两到三个最贴近主方法的结论：

- A/B：收缩风险条件 + 路由扰动界；
- C：校准假设下的决策分析 + 目标偏离界，并承认续写代理的限制；
- D：一步价值条件下的组合策略改进 + 学生拟合误差修正。

几何池恒等式作为 lemma；反例和完整证明放附录。不需要为了凑数保留原来的五个 Proposition。

### 10.8 Experiments

主结果回答三个问题：有没有学到专长、有没有损害通用能力、是否值得新增算力。机制实验回答 gate 是否在正确场景启用正确教师。详细矩阵见下一节。

### 10.9 Limitations / Conclusion

讨论同源约束、verifier 偏差、teacher–student 可利用性差距、校准漂移、长序列界变松、shared 下限可能压制强 specialist，以及推断只保留学生。结论只陈述实际结果与成立的理论条件。

## 11. 最小实验闭环与公平比较

### 11.1 数据与评测边界

先做图像任务；视频作为扩展。把数据按题目／图像来源拆成 teacher training、router/probe training、calibration、development、final evaluation。重复图像和改写题应按组去重。

原稿一边列 MathVista/MathVision 为教师训练数据，一边又用于评测，边界不清。主实验建议将选定评测集整体留出；若使用官方训练 split，必须精确记录 split 并去重，不能把 benchmark 名称笼统写进训练表。

评测先沿用现有 pipeline 的 MathVista、MathVision、MathVerse、MMMU、MMStar、MMBench，再补能直接检验感知专长的计数／空间／grounding 指标。广域基准中有推理成分，不能把其全部当成纯感知分数。各 benchmark 按既定协议评测，原始指标与域汇总一起报告。

### 11.2 分阶段执行，避免一开始十几个大实验

| 阶段 | 实验 | 继续条件 |
|---|---|---|
| P0：教师诊断 | g/p/r 能力矩阵、共同失败率、相对 g 的胜率、分布距离 | 两 specialist 的互补优势真实存在 |
| P1：最低成本整合 | g-only、MOPD-hard、A1、均匀几何／概率池 | 存在保留–特化 trade-off，且整合不完全失败 |
| P2：选择路线 | B 或 C 或 D；增加其最近邻方法 | 所选质量／价值信号有预测力 |
| P3：主实验 | 多个训练随机种子、两种模型规模或独立设置 | 优势不只来自单种子或单个任务 |
| P4：扩展 | 第四专家／视频／跨家族 | 只有前三阶段成功且有剩余预算时做 |

oracle 从同一批 probe 里挑最好再在同批上评估会有选择偏差；应独立采样评估选出的专家，或明确把该数值标成带乐观偏差的诊断参考，不当主结果。

### 11.3 Baselines

所有路线共用：初始学生、g-only OPD、每个单教师 OPD、MOPD-hard、均匀几何池、均匀概率池、固定 shared anchor。

- B：加 ExpertAlign、熵路由、TrustMOPD、无可靠性筛选／无对齐消融。
- C：加 B、VG-OPD-style prompt verifier gate、同样 probe 预算但只用 prompt 信息的 router、去掉校准／改比较 student 的版本。
- D：加相同权重的几何池、C 的完整续写 utility、critic/Monte Carlo 估值对照，以及冻结参考策略本身。
- preservation：选 SF-MOPD 或可复现的通用保留方法；LEGO-style budget 作为机制对照。不同输入或教师结构的原方法不能假装完全同设定。

Open-MOPD 的 token-share balancing 与 DN-MOPD 的 advantage-scale normalization 是优化预算／尺度组件，不应不加区分地乘入 residual 权重；那会改变 shared 总预算。采用时保持各主基线一致，并单独消融。[Open-MOPD](https://arxiv.org/abs/2608.19098)、[DN-MOPD](https://arxiv.org/abs/2609.35347)

### 11.4 预算必须有两条公平线

1. **同学生 rollout tokens：**检验监督质量；所有方法学生生成预算相同。
2. **同总 GPU-hours／总训练成本：**纳入教师训练、全教师评分、probe 续写、router/critic 和校准，允许便宜基线使用剩余预算。

原始 MOPD 每个 prompt 常只评分一个教师；全教师 token gate 通常需要三位教师评分。gate 为零不代表这次前向免费。若想省计算，需先用廉价 router 选候选再调用教师，并测漏选损失。

不能把其他论文的 1.18× 成本数字移植到本方案。最终部署只保留学生时，无多教师路由开销；训练成本仍需完整计入。

### 11.5 主表与指标

| 方法 | 感知 | 推理 | 通用 | 混合题 | 能力税 | 训练 GPU-hours | probe/teacher 成本 |
|---|---|---|---|---|---|---|---|
| g-only | 待测 | 待测 | 待测 | 待测 | 0 | 待测 | 待测 |
| MOPD-hard | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Anchor-fixed | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 最近邻强基线 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 所选方案 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

能力税相对 **g-only 蒸馏学生**定义，另外报告直接与 g 教师的差距。归一化恢复率只有在教师分数明显高于初始学生、分母可靠时使用；否则报告原始提升，避免小分母制造巨大增益。对样本不平衡的任务报告每域与最差域指标，而不是只给总体均值。

### 11.6 必要机制实验

- 固定、prompt、step、token 四种权重粒度；同样 teacher/probe 预算。
- shared 下限与 total residual budget sweep；画特化收益–通用保留 Pareto 曲线。
- gate 预测正收益的 precision、覆盖率、校准误差；报告高自信错误专家。
- 注入路由错误，分开看“误判率”与“同等误判率下的伤害”。
- C：完整续写增益与共同学生续写下的局部增益相关性。
- D：LCB 覆盖、critic 误差与实际策略改进；报告无可行 shared 权重的比例。
- 真实视觉干预：遮挡有关区域、无关区域及图像替换，观察感知路由是否响应真实证据；避免把对 OOD 扰动敏感直接等同视觉正确。
- 测实际参数梯度冲突；不要用 logit 残差正交代替。VGS 的语言／视觉损失梯度和本文两教师残差不是同一对象。[VGS](https://arxiv.org/abs/2606.00564)

### 11.7 图表计划

1. 统一架构图：student rollout → 三教师评分 → 所选 gate → anchored target → student update。
2. 教师能力／错误重合矩阵，先展示为何值得融合。
3. 同一轨迹的权重热图，并用 probe 结果核对，不靠人工挑好看的例子。
4. 特化–保留–成本对照；注明预算口径。
5. 与所选理论对应的图：风险条件、扰动、校准或一步 advantage。

## 12. 实现中必须保持一致的细节

1. **词表与温度：**同一 token 定义、特殊 token 和温度；异构词表先解决映射再谈 KL。
2. **因果权重：**若声称 q 是序列策略，权重只依赖当前前缀；使用未来正确答案或整条轨迹信息的后验 mask，应明确只是训练选择机制。
3. **stop-gradient：**教师、router、置信区间与权重在本轮学生更新中固定；不允许学生通过操纵 gate 逃避监督。
4. **top-k：**共同 token 支持、候选外质量和归一化必须明确；截断后重新归一化的 KL 与全词表 KL 不相同，不能把子集界写成完整分布保证。原 MOPD 的 top-k generalized KL 带 \(-p+q\) 修正，不能悄悄换成另一损失。
5. **采样估计：**逐 token teacher−student log-prob advantage 只在相应固定状态／on-policy 条件下解释；多次内层更新、采样温度变化或异步 rollout 需使用一致的概率与校正。
6. **锚点保护的分布：**仅在专门任务上约束到 g，不自动保护未知通用输入；加入广域回放或覆盖诊断，所有基线同样处理。
7. **学生超过教师：**额外测 student vs g/p/r 的胜率。always-on g 是保留参照，也可能成为能力上限；不应掩盖这种失败模式。

## 13. 推荐的选择方式

- **优先省算力与尽快验证：A → B。**主要结论围绕“可靠性筛选是否比单纯 alignment 有效”，不追求不成立的全局性能定理。
- **优先论文问题清晰与方法空间：A 作为基线，C 作为主方案。**把创新压在“当前前缀相对 generalist 的可验证增量”，并用共同续写诊断补上关键证据。
- **优先奖励层面的理论：D。**接受概率混合、critic/probe 成本，以及严格 shared 下限和对任意 baseline 保证之间的可行性约束。

无论选哪条，最先应拿到的都不是更多专家，而是三个教师的互补性证据，以及“路由分数是否真的预测帮助”的小规模结果。
