# MOPD × Shared Expert：三个改进构想与独立审稿修订

> 日期：2026-10-25  
> 证据快照：2026-10-25 02:12（Asia/Shanghai）；以当时已完整汇总的结果为准  
> 状态：三位独立模拟审稿后的修订汇总（非真实外审）  
> 范围：基于当前工作区的论文设计、教师训练与统一评测结果；不把未完成的学生蒸馏或扩展评测写成已有结论。

## 1. 当前证据与构思约束

### 1.1 已完成的教师结果

统一生成协议为 `thinking=True`、`temperature=0`、响应上限 8192 tokens，长度截断记零。主套件有 3,423 行；BLINK/MMVP 增量套件有 2,201 行。下表数值为百分比。

| 基准 | Base-4B | Base-9B | Generalist | Perception | Reasoning | Perception-v2 |
|---|---:|---:|---:|---:|---:|---:|
| MMBench | 85.20 | 88.00 | 88.00 | 87.20 | **88.40** | 86.40 |
| MME perception 子集 | 92.92 | 94.58 | **95.42** | 95.00 | 95.00 | 95.00 |
| MMMU | 57.02 | 62.36 | 67.26 | 63.47 | **68.37** | 60.13 |
| MMStar | 67.60 | 68.20 | 72.40 | 71.60 | **73.80** | 68.80 |
| MathVision | 35.86 | 40.13 | 45.07 | 44.41 | **48.68** | 35.53 |
| MathVista | 71.60 | 74.60 | 79.60 | 78.00 | **81.20** | 75.40 |
| RefCOCO heldout | 91.80 | 91.60 | 91.20 | 92.00 | 92.00 | **93.00** |
| BLINK val | 59.18 | 未完成 | 未完成 | 66.86 | **67.54** | 未完成 |
| MMVP | 79.00 | 未完成 | 未完成 | **81.33** | 80.33 | 未完成 |

G/P/R/P2 分别为 generalist、perception-step100、reasoning-step100、perception-v2-step100，均为 step100 教师；P2 使用不同的感知数据配方。

来源：`experiments/sharedexpert_c/evaluation/results/summary.md`、`experiments/sharedexpert_c/evaluation/results/summary_main.md` 与 `experiments/sharedexpert_c/evaluation/results_blink_mmvp/summary.md`。

直接约束：

1. **Generalist 不是弱先验。**它在 MME 上最好，并在 MMMU、MathVision、MathVista 上明显超过 perception-v1；shared anchor 有实证动机，但也可能限制 specialist 增益。
2. **教师名字不等于逐样本能力。**Reasoning 在 BLINK 上略高于 perception；perception-v1 没有形成干净的感知专属优势。不能靠 `P 管读图、R 管推理` 的先验完成路由创新。
3. **Perception-v2 只在 RefCOCO 上取得数值最高分，其他维度相对偏弱。**其 RefCOCO 达 93.0，仅比 P/R 高 1.0 个百分点且置信区间重叠，不能声称稳定优于；其 reasoning 宏平均只有 57.02，虽然比 base 的 54.82 高 2.20 个百分点，但低于 perception-v1 的 61.96、generalist 的 63.98 和 reasoning 的 66.09。它更适合作为“条件增量是否可识别”的压力测试，而非默认替换全部 perception 教师。
4. **训练 reward 不等于下游专长。**教师训练 reward 均上升；perception-v2 的 step100 平均 reward 约 0.931，同时其数学与通用成绩明显低于 G/P/R。路由和教师准入必须使用独立任务结果，而不是训练 reward。
5. **数值最高不等于稳定占优。**本次用 20,000 次 paired cluster bootstrap 复核 R−G：七个主基准的 95% 区间均跨 0。例如 MathVision 差值为 +3.62、区间约 [-0.66, 7.89]，MMMU 为 +1.11、[-2.23, 4.23]。这不能证明 R 与 G 相等，也不排除宏平均差异，但不支持“R 已在各基准稳定压倒 G”的写法。

### 1.2 互补性机会：存在，但不是学生可达上界

已有 P/R 分析显示，二者主套件 raw-row oracle 为 84.55%，相对 R 的 80.98% 高 3.56 个百分点；按原报告主评测单位，P 独有 109 题、R 独有 174 题。BLINK 上 P/R oracle union 为 76.28%，较最佳单教师 R 高 8.73 个百分点，P-only/R-only 分别为 166/179 行。原报告的 P/R 按基准等权分析中，oracle−R 在 perception、reasoning、general 分别为 +2.03、+5.67、+3.10 个百分点，对应点态 95% bootstrap 区间约为 [1.05, 3.15]、[4.38, 7.07]、[1.90, 4.40]；这些区间不是多重性校正后的区间，也不消除 oracle 的选择偏差。

本次另用当前 `results/per_sample.csv` 做了只读重算：G/P/R/P2 分别指 generalist、perception-step100、reasoning-step100、perception-v2-step100。MMBench 先把每个教师的 circular group 定义为“该教师在组内全部变体正确”，再做教师 union；即同一教师必须通过整组，与既有 P/R 分析口径一致。其余基准按题计算。

| 基准 | 最优单教师 | G/P/R union | 四教师 union | union−最优单教师 |
|---|---:|---:|---:|---:|
| MMBench | 88.40 (R) | 91.60 | 91.60 | +3.20 |
| MME perception 子集 | 95.42 (G) | 97.08 | 97.92 | +2.50 |
| MMMU | 68.37 (R) | 75.50 | 76.39 | +8.02 |
| MMStar | 73.80 (R) | 80.60 | 81.80 | +8.00 |
| MathVision | 48.68 (R) | 58.55 | 59.21 | +10.53 |
| MathVista | 81.20 (R) | 86.60 | 88.20 | +7.00 |
| RefCOCO | 93.00 (P2) | 95.00 | 95.40 | +2.40 |

按各基准等权，四教师 union 相对“逐基准事后最优单教师”的差距为：reasoning +8.51、perception +2.45、general +5.60 个百分点。按相同评测单位，G 有 32 题是其余三教师全错而自己答对，P2 也有 24 题这样的“独有正确”（其中 MathVista 8、MMStar 6、MMMU 4）。这说明强 anchor 与平均较弱的教师都可能有局部价值，但数量较小，且仍是未做独立 router 评估的事后统计。

这些数字只说明**最终答案正确集合有残余互补性**。它们使用 ground truth 事后选教师，具有选择偏差；混合了知识、输出长度、终止和格式差异；既不是 prefix continuation advantage，也不是可学习 router 的覆盖率或学生蒸馏上界。增加教师数量只会保持或扩大 union，因此“四教师 union 高于三教师 union”本身也不证明 P2 值得增加训练/评分成本。测试题标签不得用于 router 训练、校准或模型选择。

### 1.3 截断是不可忽略的混杂因素

主套件长度截断数为：base 530、generalist 277、perception 302、reasoning 229、perception-v2 431（不是各自的抽取失败数）。已有 both-finished 诊断中：

- MathVision：全量 P/R 为 44.41/48.68；二者都正常完成的 151 题上均为 77.48。
- MathVista：全量 P/R 为 78.00/81.20；二者都正常完成的 443 题上为 87.13/86.68。

这提示部分教师差异与完成行为相关，而非纯粹“推理知识”差异。both-finished 是按模型行为条件化的子集，不能解释为去除截断后的因果能力。后续路由必须单独报告严格准确率、完成率、条件正确率和生成 token 成本。

### 1.4 尚无的关键结果

- `domain-opd` 学生与 `mix-rl` 对照均未开始；当前没有任何 SharedExpert/MOPD 学生增益证据。
- IFEval/OCRBench 扩展尚无完整结果；generalist 与 perception-v2 的 BLINK/MMVP 也未完成。部分扩展任务可能仍在运行，本文只采用已经完整汇总的结果。
- 合并 `summary.md` 的 perception 宏平均对 P/R 包含 BLINK/MMVP，而对 G/P2 目前只包含主套件中的感知基准，覆盖不一致；本文不拿该列做四教师横向结论，只比较同一基准上的完整结果。
- 本次新增的四教师 union 是描述性重算，未重新计算置信区间；既有 P/R 分析的 20,000 次 cluster bootstrap 也不能自动覆盖新增的 G/P2 选择。
- 目前没有 prefix-level continuation probe、router 校准、视觉反事实门控或 lookahead 学生更新实验。
- 当前 `distillation` 仍是 prompt-domain 的单教师评分与 sampled-token surrogate；真实多教师 `q_w` 解码、三图评分、组合梯度和候选 shadow update 都不是现有开关。已有 CPU 测试不能证明这些新路线已经实现或有效。

因此，下文全部是待证伪的研究方案，不预写“超过 MOPD”“保证能力保持”或“首次”。

### 1.5 对原 SharedExpert-MOPD 构思的审理

| 原始主张或设计 | 审查结论 | 后续处理 |
|---|---|---|
| “首次把 shared expert + specialist residual 用于教师空间” | 不能成立为核心新颖性；ChemOPD 已有 generalist-relative correction | 保留 shared anchor 作为功能角色，创新转移到增量价值或视觉证据的识别 |
| “公共分量很大，所以残差天然更能抗误路由” | 推论不成立；两个 specialist 相减也会消去公共分量 | 只研究显式权重预算带来的扰动控制，不使用 common-mode rejection 叙事 |
| “teacher−generalist 与 teacher−student 对齐说明专家值得学习” | 对齐不等于正确；学生等于 generalist 时任何非零残差都可通过 | 使用中心化、学生分布加权的 alignment 作为便宜 baseline，而非 correctness gate |
| “P 负责感知、R 负责推理” | 当前逐基准结果不支持干净的互斥分工 | 教师名称只记录训练来源，路由角色由独立校准中的条件增量决定 |
| “always-on generalist 可以保护通用能力” | 只能约束部分目标分布，不能保证未见通用输入或最终准确率 | 保留广域回放与通用留出评测，报告 anchor 也可能限制强 specialist 的失败情形 |
| “token gate 是必要且新颖的” | token 动态权重已有多项近邻，且当前没有粒度优势证据 | 先比较 prompt、segment/step、token；主创新不得只写粒度 |
| “教师 oracle union 证明蒸馏空间” | 只能证明最终答案正确集合不完全重合 | 降级为动机诊断；必须补同 prefix、同后续策略的可转移价值实验 |

## 2. 三个初稿（提交独立审稿前）

### Idea 1：Student-Completed Counterfactual Teachability（SCCT）

**问题。**已有完整教师续写价值会混入教师自身的后续恢复能力；一步 advantage 又可能太局部。真正需要评价的是：当前 anchored target 提供一小段指导后，**同一个学生**能否更好地完成。

**初稿。**在真实学生 prefix 的稀疏 segment 边界，分别让 generalist、specialist 和实际 anchored mixture 只生成一个短 segment，随后全部由同一冻结学生续写。定义

\[
V_s(\rho;h)=\mathbb E_{c\sim\rho(\cdot|h),\,y_{\rm rest}\sim\pi_k}R(h,c,y_{\rm rest}).
\]

直接以 `V_s(q_w)-V_s(p_g)` 评价实际组合目标，而非由单教师整段成功率推断组合收益。以不确定下界筛选权重，保留 generalist shared floor，并把稀疏 probe 标签蒸馏给轻量 router；每轮冻结已采样权重，跨轮复测校准。

**与已有 A–D 的关系。**它把既有文档中的 target-verifier 机制诊断升级为路由监督；与 C 的差别是 common student continuation，与 D 的差别是 segment 级、直接评价实际 q，而非一 token + generalist continuation 的线性 advantage。

### Idea 2：Visual Difference-in-Differences Residual（VDiD）

**问题。**`specialist-generalist` 同时含有视觉能力、语言风格、校准与错误差异。需要识别“相对 generalist 的优势是否真的来自视觉证据”，而不是按教师名字猜角色。

**初稿。**固定学生 prefix，构造原图 `I`、问题相关区域扰动 `I_rel` 和无关区域扰动 `I_irr`。令

\[
r_i^I=\log p_i^I-\log p_g^I,\qquad
v_i=(r_i^I-r_i^{I_{rel}})-(r_i^I-r_i^{I_{irr}})
=r_i^{I_{irr}}-r_i^{I_{rel}}.
\]

视觉反事实只负责 gate，不进入训练目标；目标仍使用原图的 generalist-anchored residual。用少量可验证生成结果校准视觉增量，不限制 reasoning 教师不能在感知题上获胜。

**与已有工作的候选差别。**不是把单一 VLM 的有图/无图残差直接当作组合因子，而是比较 specialist 相对 generalist 的残差在相关/无关视觉干预下如何改变。该差别是否足够新颖，交由审稿检验。

### Idea 3：Lookahead Learning-Gain Routing（LLGR）

**问题。**教师更强、更接近学生或续写奖励更高，都不必然意味着一次有限学生更新能学到更多。能否直接按“学生更新后的改善”选择教师组合？

**初稿。**周期性构造若干含 generalist floor 的候选目标 `q_w`，求解

\[
w^*=\arg\min_{w\in\mathcal W_\eta}
\mathcal L_{\rm val}\left(\theta-\eta\nabla_\theta
\mathcal L_{\rm OPD}(\theta;q_w)\right),
\]

其中验证 prompt 与更新 batch 独立且具有可验证监督。用一阶梯度内积或少量离散候选降低成本，只采用粗粒度、周期性权重；与 teacher accuracy、teacher–student distance 等路由比较。

**候选贡献。**将路由目标从教师侧代理改为学生实际学习增益，并显式核算“更多候选反传”是否值得。

## 3. 三条路线共用的实验纪律

### 3.1 数据边界

1. 按题目及来源图像组划分 teacher training、router/probe training、calibration、development 和 final evaluation；同题不同 prefix 不得跨 split。
2. 最终测试标签只用于冻结评测，不能用于 router 训练、校准、超参选择或教师准入。本文的四教师 oracle 仅作事后分析。
3. on-policy 分布变化后，需要用新学生 rollout 重新校准；旧学生前缀上的区间不能自动当作新学生的覆盖保证。
4. 探针选择、权重选择与独立评估尽量使用不同样本，避免从多个候选中挑最优后再在同一批数据上报告收益的 winner's curse。

### 3.2 共同对照

- 初始学生、generalist-only OPD、各单教师 OPD。
- 原始 domain-hard MOPD、均匀几何池、均匀概率池、固定 shared anchor、ChemOPD-style warm-up。
- ExpertAlign、熵路由、TrustMOPD；CoSEV 增加 SPOT、FutureBridge、VG-OPD-style gate、完整教师续写与线性 advantage；PVM 增加 VC-OPD、VAD、CVPD 和 LEGO-style 图像残差/budget；LCI 增加静态权重、Ren/LGTM 式影响路由及 teacher–student compatibility 路由。
- 同数据、同学生初始化的 Mix-RL 作为独立训练范式对照；不与 OPD 的教师目标混为一谈。
- prompt、segment/step、token 粒度必须在相近信息与计算预算下比较，不能默认 token 最好。

### 3.3 两条公平预算线

1. **相同学生 rollout tokens / optimizer steps：**检验监督质量，学生初始化、prompt 配比、采样、长度上限和 loss 口径保持一致。
2. **相同总 GPU-hours：**纳入教师训练（另列表）、全教师评分、probe 续写、图像反事实前向、候选反传、router/critic 和校准；便宜基线可以使用剩余预算增加训练。

### 3.4 指标与统计

- 每个 benchmark 的原始严格准确率及维度宏平均，不将不同覆盖范围的宏平均直接横比。
- 相对 generalist-only 学生的特化增益、通用能力税、最差能力组回归，而不是只报一个混合平均分。
- 完成率、长度截断率、已完成样本正确率（诊断）、平均及总生成 tokens。
- Router 的正增益 precision、误激活率、覆盖率、校准误差；在独立 probe 上评价，而非用 ground-truth test oracle 代替。
- 特化收益–通用保留–总成本 Pareto 曲线；shared floor / residual budget 做相同 sweep。
- 正式训练至少 3 个 seed；按图像或循环选项组做 paired bootstrap，题目区间与跨 seed 方差分开报告。
- 预注册主要比较、候选集合和最小有用效应；多候选/多主比较采用独立确认或适当多重性控制。区间上界低于最小有用效应可判 futility，区间过宽则只能报不确定。

## 4. 独立模拟审稿反馈与裁决

本节汇总三个独立 subagent 的模拟审稿，不代表真实人类外审或会议决定。R1 负责新颖性，R2 负责数学与算法，R3 负责实验、统计和两卡可实施性；三者看到相同初稿并独立返回意见。

### 4.1 初稿评分与排序

R1 对**初稿**的评分为：

| Idea | Novelty | Significance | Clarity | R1 主要判断 |
|---|---:|---:|---:|---|
| SCCT | 3/5 | 4/5 | 3/5 | 直接评价实际组合仍有空间，但 student continuation 本身已有近邻 |
| VDiD | 2/5 | 3/5 | 2/5 | 视觉反事实近邻很强，原公式不识别原图视觉收益 |
| LLGR | 2/5 | 4/5 | 3/5 | 问题重要，但一步 lookahead 是经典优化思路 |

三方一致首选 **收窄后的 SCCT**；R1 与 R3 均将稀疏离散 LLGR 列第二、VDiD 列第三。R2 进一步建议 VDiD 只作辅助视觉敏感性特征、LLGR 作粗粒度验证选权基线；R3 给出 `major revision`，强调任何路线都尚无学生结果。

### 4.2 新颖性审稿（R1）带来的关键变化

1. **SCCT 不能再把 student-completed counterfactual 本身当新原理。**经主会话再次读取 arXiv 摘要核验：
   - [SPOT](https://arxiv.org/abs/2608.04419) 已包含稀疏 probing、教师候选和 verifier-scored student continuations，再用 outcome 校准目标。
   - [FutureBridge-OPD](https://arxiv.org/abs/2608.01953) 已包含 short teacher bridge 及其后的 student continuation 评估。
   因此，唯一可辩护的主线收窄为：**组合级、可能非可加的学生可执行价值**——实际 `q_w` 是否超过最佳组成教师，并且是否提供超出单教师价值/advantage 线性预测的信息。
2. **VDiD 面临比原文献表更近的工作。**[VC-OPD](https://arxiv.org/abs/2610.14939) 已用原图教师目标与 counterfactual gap 加权；[VAD](https://arxiv.org/abs/2607.28590) 已在学生 prefix 上比较 evidence present/removed 的 centered log-prob 并重构目标；[CVPD](https://arxiv.org/abs/2608.09931) 已使用区域反事实识别视觉盲点。仅换 generalist reference 不够，修订为 **placebo 控制的、verifier outcome 层面的视觉优势归因**。
3. **LLGR 不能靠双层公式承担创新。**它与 [Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html)、MAML、distillation influence（[LGTM](https://aclanthology.org/2023.acl-long.111/)）及学生反馈多教师选择相近。修订为研究**相对 no-op/G-only 的组合学习交互、通用保持约束及一步到多步的可预测性**。

以上新增文献由 R1 补检；SPOT、FutureBridge、VC-OPD、VAD、CVPD 的标题与摘要机制已由主会话再次读取原始 arXiv 页面核对。仍不据此声称全球首次。

### 4.3 理论与算法审稿（R2）带来的关键变化

- SCCT 的主标签必须是 `V_s(q_w)-V_s(student)`，相对 G 的差只作辅助；G fallback 与 shared floor 都不是相对学生的安全证书。未认证时应允许 **student abstain/no-op**。
- 必须直接解码真实 token-level `q_w` 的整个 segment。一般 `V_s(q_w)≠Σ_iw_iV_s(p_i)`；“按权重先选一个教师生成整段”也是另一种策略。
- 几何或算术目标都可使用，因为直接估计组合价值；但不能把单教师 segment advantage 的加权和冒充实际组合价值。
- VDiD 的 `r^{I_irr}-r^{I_rel}` 只衡量两种扰动下的 log-ratio contrast，原图项完全抵消；gold answer/target bbox 也不能成为实际 gate 的输入。
- LLGR 的真实离散 verifier 不可直接普通反向传播。首版应采用少量离散候选的黑盒验证，而非伪装成精确 hypergradient。
- 几何目标、固定状态和精确 full-KL 下，一阶梯度对权重是线性的，最优通常落在可行集顶点；不能声称一阶 LLGR 自然学到有意义的软混合。

### 4.4 实验审稿（R3）带来的关键变化

- 所有路线先过 inference-first 证伪，再决定是否进入正式学生训练；固定 G-only、domain OPD 和固定 anchored mixture 是不可省略的简单基线。
- SCCT 必须加入 token-matched sham segment 与 direct-answer 对照，排除“短段只是泄露答案”及只学到截断/长度差异。
- VDiD 的主指标应是原图独立 reward 的决策价值，不是视觉分数 AUC；相关/无关扰动要匹配面积与退化强度，并测试多种扰动。
- LLGR 只允许稀疏、离散、coarse 版本起步；selection、update、validation 使用不同 batch，并采用 rotating validation，最终 dev/test 不参与 lookahead。
- P2 不应默认加入所有实验臂；只能由训练/validation 的边际价值决定，不能因测试 oracle 中有少量独有正确就增加成本。
- 预注册最小有用效应与最大预算。futility 应写成“配对区间上界仍低于最小有用效应”，不能把“不显著”当成“已证明无效”。

### 4.5 对审稿数字的复核与修正

R3 提示 MMBench 的 general headroom 复现不一致。主会话进一步分解后确认：

- **96.80** 对应“一个 circular group 中至少一个变体被至少一个教师答对”的存在性口径；它比主指标宽松。
- **92.00** 对应“组内每个变体都至少有一个教师答对”，允许不同变体选择不同教师。
- **91.60** 才是本文与既有 P/R 分析采用的口径：存在至少一位教师，且**同一位教师通过该组全部变体**。

因此主表保留 MMBench 四教师 union=91.60、headroom=+3.20，general 宏平均 headroom=+5.60；不采用另外两种口径。R3 对截断数的提醒则成立：本文已把 G/R/P2 从误写的抽取失败数 275/225/426 更正为长度截断数 **277/229/431**。

### 4.6 修订后的路线名称

| 初稿 | 修订后 | 核心变化 |
|---|---|---|
| SCCT | **CoSEV-MOPD：Composition-level Student-Executable Value** | 从“学生续写”收窄为“实际组合是否超过最佳教师及线性预测”；学生参考、真实 `q_w`、abstain |
| VDiD | **PVM-OPD：Placebo-controlled Visual Mediation** | 从 logit 双差分 gate 改为 verifier outcome 的视觉优势归因与无关区域 placebo；logit 仅作辅助特征 |
| LLGR | **LCI-MOPD：Learning Composition Interactions** | 从通用双层路由改为相对 no-op/G-only 的组合学习交互，离散黑盒选择、通用约束、一步到多步验证 |


## 5. 审稿后的三个修订方案

### 5.1 路线一：CoSEV-MOPD，组合级学生可执行价值

**核心科学问题。**实际 anchored teacher composition 的一个短干预，是否比 G、当前学生、最佳单教师及线性 advantage 预测更能被当前学生接续利用？论文不再声称 student continuation 或 segment bridge 是新原理；SPOT 与 FutureBridge 是必须同预算比较的直接先例。

**最小方法。**固定第 `k` 轮学生 `π_k`，在其真实 rollout 中稀疏选择 segment 边界。对候选策略 `ρ` 生成长度为 `m` 的片段 `c`，随后由同一个 `π_k` 完成：

\[
V_k^m(\rho;h)=\mathbb E_{c\sim\rho_m(\cdot|h),\,y_{\rm rest}\sim\pi_k(\cdot|h,c)}
[R(h,c,y_{\rm rest})].
\]

首版只比较少量离散候选：G-only、P/R 单教师、固定 shared anchor，以及若干具有相同 generalist floor 的组合 `q_w`。P/R 单教师是 probe 与公平性对照，正式 SharedExpert 组合仍满足 G floor；若真实组合不优于最佳单教师，不能把“胜过弱 G”当作方法优势。直接估计 `V_k^m(q_w;h)`，不再由教师独立成功率推断组合价值。**实现必须按 token-level `q_w` 真正解码整个 segment；不能先按权重抽一个教师生成整段，也不能用各教师整段价值的加权和代替组合价值。**至少同时保留两个对照：

\[
D_w^G=V_k^m(q_w;h)-V_k^m(p_g;h),\qquad
D_w^S=V_k^m(q_w;h)-V_k^m(\pi_k;h).
\]

为把新颖性压在“组合”而不是续写本身，另定义两个独立诊断量：

\[
C_w=V_k^m(q_w;h)-\max_{j\in\{g,p,r\}}V_k^m(p_j;h),
\]
\[
N_w=V_k^m(q_w;h)-\sum_{j\in\{g,p,r\}}w_jV_k^m(p_j;h).
\]

`C_w` 检查实际组合是否超过最佳组成教师；`N_w` 度量不能被单教师价值线性预测解释的交互，可正可负。若 `N_w` 始终接近 0，线性 predictor 已足够，昂贵的组合 probe 没有新增信息。对 **一 token 算术混合**，线性关系在相同 continuation 下本应精确成立，只能作数值 sanity check；研究非可加性需要 `m>1` 的真实 segment 解码或几何组合，不能人为制造交互。

specialist 只有在独立校准支持正学生参考优势时才激活；G-only 也必须实测 `A_G^b`，不能默认安全。三态决策为：选择经认证且优于学生/G/最佳组成教师的组合；若组合未通过但 G-only 相对学生有认证优势，则退回 G-only；否则跳过该次蒸馏。shared floor 的准确含义是：**所有启用教师的组合目标都保留 G；跳过更新不伪装成 G 回退。**

首版固定一种目标参数化。若采用算术混合，

\[
q_w=w_g p_g+\sum_i w_i p_i,\qquad w_g\ge\eta,\quad w_i\ge0,\quad w_g+\sum_iw_i=1;
\]

若采用几何残差，则写成 `softmax(log p_g + Σ_i w_i(log p_i-log p_g))`。二者是不同的实验条件，不能共享算术混合的线性 advantage 结论。CoSEV 的核心是直接评价实际 `q`，不是再换一个 gate 公式。

**Probe 到 router。**用训练题目的 probe 标签训练轻量预测器，输入只能包含当时可观察的学生/教师分布特征、prefix 和位置。按题目划分训练、校准和独立评估；每轮刷新学生前缀，已采样目标与权重在优化阶段缓存并 stop-gradient。候选选择和最终报告使用独立续写样本，降低多候选选择偏差。

**可主张与不可主张。**令 `b=π_k`，主标签写成 `A_ρ^b(h)=V_s(ρ;h)-V_s(b;h)`；`D_w^G` 只作辅助排序特征，`A_G^b` 也必须实测，不能默认是 0。若宏策略 `ρ̄` 在所有被激活状态上的证书同时有效、未激活状态严格取 `b`，则 segment-level performance difference 给出

\[
J(\bar\rho)-J(b)=\sum_s\mathbb E_{h\sim d_{\bar\rho,s}}A_{\bar\rho(h)}^b(h)
\ge \Gamma\ge0,
\]

其中 `Γ` 是激活状态上非负证书下界的总和。单轮固定目标内，最终学生仍有拟合误差惩罚：若最终学生访问状态下对该固定目标的平均真实拟合误差为 `ε_fit`，`T` 为 segment horizon、`R_span` 为奖励范围长度，则相应下界为 `Γ-R_span·min(1,sqrt(Tε_fit/2))`；严格改善要求正 margin 大于该项。离线 router 在新状态误激活、学生更新后标签失效或有限 probe 未覆盖目标可达状态时，不能直接套用该结论。

- 可主张：提出 target-level、student-completed 的片段价值估计，并检验它是否比教师完整续写和 logit proxy 更能预测可蒸馏收益。
- 可分析：固定前缀上的 Monte Carlo 误差、固定有限候选与明确采样假设下的同时区间、给定校准误差假设下的代理决策 regret。
- 不可直接主张：`m>1` 时沿用一 token 算术混合的线性 advantage 分解；稀疏校准前缀覆盖所有新状态；局部片段价值保证最终学生无条件优于 G/MOPD；单一标量 reward 改善推出所有能力分项改善。连续搜索或反复自适应加候选时，也不能直接套用预先固定候选的 union bound。

**最低成本证伪实验。**先在训练/校准题目的随机与分层 prefix 上估计方差，再按可检测效应确定样本量。比较 G、P/R、静态锚与实际组合在 `m=1` 和一个短句/推理步下的价值；核心对照包括 SPOT 式单教师候选价值、FutureBridge 式 bridge、C 的完整教师续写、D 的线性 advantage，以及单教师价值的线性 predictor。另加 **token-matched sham segment** 与 **direct-answer** 对照，并审计短段是否直接泄露选项或最终答案。通过后再做正式蒸馏，不额外安排与科研问题无关的预训练 smoke。

R3 给出的起步规模可作为方差估计参考而非功效保证：192 个分层训练 prompt、每题 2 个 prefix，共 384 个状态；no-op/G/P/R/固定组合五臂，每臂 4 次配对续写，共 7,680 条学生 continuation，另有 1,536 条短教师 segment；sham/direct-answer 对照另计。其学生 continuation 数已约为既定 6,400 条 OPD 响应的 1.2 倍，因此必须先做方差与 futility 评估，不能把“教师只生成短段”误写成总成本很低。

**Go / no-go。**

- Go：在可承受采样数内存在可重复的正学生参考 margin；真实组合还优于最佳单教师/静态锚；直接组合价值提供超出 SPOT/FutureBridge、C/D 与线性 predictor 的增量；优势在 sham 和答案剥离后仍存在；独立评估和等 GPU-hours 下仍有收益。
- No-go：主要只是胜 G、却经常输给学生；真实组合不优于最佳组成教师；线性 predictor 已足够；优势主要来自 direct answer、长度或截断；可行预算下区间始终过宽或 probe 成本抵消全部学生收益。置信区间太宽只能判为不确定，不能写成已经证明无效。

### 5.2 路线二：PVM-OPD，placebo 控制的视觉优势归因

**核心科学问题。**specialist 相对 G 的一小段指导优势中，有多少会在相关视觉证据受扰后消失？这一效应能否通过无关区域 placebo，并转化为原图上的独立任务收益？这条路线收窄为**局部可验证任务上的视觉优势归因**，不再声称通用 P/R 因果路由器。

**Outcome-level 主估计量。**固定冻结学生 `b=π_k`、明确的前缀与解码预算。令教师 `j` 在图像 `X` 下生成候选 segment，随后由 `b` 在预先指定的相同图像条件 `X` 下续写：

\[
V_j(X)=\mathbb E_{c\sim p_j(\cdot|h,X),\,y\sim b(\cdot|h,c,X)}R(h,c,y),
\qquad U_i(X)=V_i(X)-V_g(X).
\]

定义

\[
\tau_i^{vis}=U_i(I)-U_i(I_{rel}),\qquad
\mathcal P_i=U_i(I_{irr})-U_i(I).
\]

`τ_i^vis` 是相关干预消除的相对优势，`𝒫_i` 是无关 placebo 引起的漂移。若 `𝒫_i≈0`，则 `U_i(I_irr)-U_i(I_rel)≈τ_i^vis`；若还要求它等于原图全部相对优势，则另需 `U_i(I_rel)≈0`，不能默认。候选 segment 与 continuation 若使用不同图像条件，估计的是另一个量，必须单列。

用于路由前至少需要：原图相对优势为正、相对学生也不更差、相关干预使优势下降、placebo 在预注册容差内。图像敏感但原图错误的教师不能通过。只用 prompt 级结果时，就只能声称 prompt 级归因。

**Logit 仅作辅助预测器。**初稿的 `r^{I_irr}-r^{I_rel}` 以及相对原图的 `r^I-r^{I_rel}`、`r^I-r^{I_irr}` 可用于低成本预测上述 outcome 标签，但都不是效用本身。实现使用 normalized log-prob、共同 token 支持和一致的 tail 处理；内积/范数前在同一参考分布下中心化。原图 residual 才进入蒸馏目标，扰动图目标不蒸馏。

**区域与扰动。**实际 gate 的区域生成器只能读取可观察的图像、问题和 prefix，不能读取 gold answer 或 target bbox。RefCOCO target bbox 等 oracle region 只用于上限/诊断，不进入可部署结果。relevant 与 irrelevant placebo 应在可行范围内匹配面积、位置和退化强度，或通过多组随机 placebo 使这些 nuisance 分布一致；同时检查 mask 自身泄露和 OOD 敏感性。固定原图 prefix 已包含视觉事实时，只能解释为条件敏感性，并补 native-prefix 对照。

**与最近邻的区别。**VC-OPD 已做原图目标 + counterfactual gap 加权，VAD 已做 evidence present/removed 的视觉方向与目标重构，CVPD 已做区域反事实监督。PVM 的候选增量仅是：**以 specialist-over-G 的 verifier outcome 为对象，用相关干预和 matched placebo 分解可归因优势，并检验这种归因是否有独立决策价值。**若仅表现为又一个 logit gate，或相对这些方法没有增量，就不应成立为主线。

**最低成本证伪实验。**V1 可先从 256–512 个训练/校准图像起步，覆盖两类来源与两类扰动，做离线敏感性和 placebo 检查；该规模只用于方差估计。V2 在新 prompt 上比较 PVM gate、ungated original residual、absolute-q、VC-OPD/VAD-style、固定/domain、随机及 sham gate，阈值 cross-fit，主指标为**原图独立 reward 的决策价值**而非 AUC。R3 建议 V2 先封顶约 3,200 条学生 continuation，再根据盲态方差调整。G + 一个 specialist + 三图条件已需约 6 次教师序列评分/prefix；四教师则约 12 次，不能按普通单教师 OPD 成本报告。

**Go / no-go。**

- Go：多种来源和扰动下，相关效应稳定而 placebo 接近预注册容差；PVM 相对 VC-OPD/VAD-style、质量-only、ungated/sham 有独立原图 reward 增益；增益在等 GPU-hours 下仍存在。
- No-go：placebo 明显改变效用；效应只反映 OOD 或 oracle mask；简单原图特征/已有视觉反事实已同样好；预注册最小有用效应的区间上界仍不通过，或三倍评分抵消收益。

**措辞边界。**可以报告“在指定干预与冻结 continuation 下的视觉优势消除量和 placebo 稳定性”，不能声称识别了教师内部的无条件因果机制。当前主套件 perception 四教师机会差只有 2.45 且 MME 接近饱和，BLINK 又缺 G/P2，所以该路线优先定位为局部机制研究或备选论文，而非当前整合论文的唯一主线。

### 5.3 路线三：LCI-MOPD，组合学习交互与稀疏离散 lookahead

**核心科学问题。**多教师目标在学生参数空间中何时产生非可加的学习收益或干扰？相对 no-op 和 G-only 的一步组合增益，何时能预测后续 `K` 步乃至完整训练？论文不再把经典 bilevel/example-reweighting 公式本身当创新。

**黑盒离散首版。**每 `K` 个训练步，在同一批学生 rollout、缓存动作和固定教师评分上构造少量离散候选。对每个候选实际执行与正式优化器一致的 shadow update：

\[
\theta'_w=\operatorname{Update}(\theta,B_{train};q_w),\qquad
\Delta_w^d=R_d(\theta'_w)-R_d(\theta),
\]

其中 `R_d` 是在独立 rotating validation 上测得的真实可验证 reward，`d` 至少区分 specialist 与 general retention。定义相对 G-only 的增量和组合学习交互：

\[
I_w^d=\Delta_w^d-\Delta_G^d,\qquad
N_w^d=\Delta_w^d-\sum_jw_j\Delta_j^d.
\]

`N_w^d` 可正可负，用于测量非可加性，不预设它一定为正。候选只有在独立验证上同时满足“相对 no-op 有可接受收益、相对 G-only 有最小组合增益、general 退化不超过预注册容差”时才可选择；没有候选通过则 **no-op**。G floor 不能替代这些外部约束。

首版候选只需 no-op、G-only、训练/校准集选出的最佳 hard specialist、一个固定 anchored interior。selection、内层 update、lookahead validation 使用不同 batch；反复使用的 validation 是开发流，不是最终测试。所有 shadow update 使用相同优化器状态与学习率，未选候选完整恢复；模型加载、恢复、validation 生成和 verifier 都计入成本。

**一阶版本只作可选筛查。**若另用可微 `L_val`，可在固定状态下做

\[
L_{val}(\theta-\eta\widetilde g_w)\approx L_{val}(\theta)-\eta\langle g_{val},\widetilde g_w\rangle,
\]

但结论只属于 surrogate。几何目标、精确 full-KL 下 `g_w=Σ_jw_jg_j`，所以一阶目标对 `w` 线性，通常只选择可行集顶点，不能宣称学到软混合。当前 sampled surrogate 还有 capped correction、采样方差和目标 detach；真实离散 reward 也不能直接普通反传。黑盒离散版不需要穿过 `w` 或 reward 求导，因而作为首版。

**粒度与理论边界。**同批样本共享一个梯度时，只能称 round/domain/batch-level 周期路由，不是 prompt/token adaptive。可以分析精确确定性、固定状态和光滑条件下的局部 validation-surrogate 下降；不能声称 capped sampled surrogate 是完整 on-policy 轨迹梯度、一步增益保证长期 reward，或同批选择结果具有独立测试泛化。

**与经典工作的差异。**Ren 式 example reweighting、MAML、LGTM/distillation influence 和学生反馈多教师选择都是直接先例。LCI 只有在“**组合相对 G-only 的学习交互、独立 general 约束、一步到多步可预测性**”提供新证据时才有增量；仅把样本权重换成教师权重不够。

**最低成本证伪实验。**在若干冻结学生状态上比较 no-op/G/best specialist/固定组合：真实一步独立 reward、相对 G 的 `I_w`、组合交互 `N_w`，以及可选一阶得分对真实更新排序的预测力。随后在稀疏 lookahead 内检验所选策略与 `K` 步真实结果是否一致；不通过就不进入完整训练。该评估是方法本身，不另设一次性 GPU smoke 或临时导出。

作为操作量参考，若 200 步中每 10 步做一次、每次 4 个候选、validation batch 32，额外 validation 响应约为 `20×4×32=2,560`，相当于既定 6,400 条 OPD 响应的 0.4 倍；候选反向、临时更新、教师评分和恢复另计，实际墙钟必须测量。

**Go / no-go。**

- Go：组合相对 no-op 与 G-only 的增益在新鲜 validation 上可重复；一步指标能预测 `K` 步结果；等总 GPU-hours 下优于固定权重、静态锚、Ren/LGTM 式影响路由和 compatibility 基线。
- No-go：一阶排序与真实更新不一致；只改善可微 surrogate；组合优势被多候选选择噪声主导；一步不能预测多步；等计算不优于最佳固定权重。若权重最终总收敛到同一教师，应保留简单固定基线，而不是包装成自适应成果。

## 6. 最终选择与论文定位

### 6.1 三条路线的优先级

| 路线 | 当前定位 | 成立的必要条件 | 主要风险 |
|---|---|---|---|
| **CoSEV-MOPD** | **唯一推荐主线** | 实际组合超过最佳组成教师，且直接组合价值超出 SPOT/FutureBridge、C/D 和线性 predictor | probe 成本、学生覆盖/拟合、组合优势可能不存在 |
| **LCI-MOPD** | 离散、粗粒度备选 | 相对 no-op/G-only 的学习交互可重复，并能预测多步结果 | 经典 meta-reweighting 移植感、候选反传与验证成本 |
| **PVM-OPD** | 局部视觉归因探索/备选论文 | outcome-level 视觉消除量通过 placebo，并对原图独立 reward 有增量 | VC-OPD/VAD 重叠、OOD 与 mask 泄漏、perception 机会差较小 |

不要把三者叠加成一个主方法。若 CoSEV 的最小 probe 失败，应回到静态锚或检验稀疏 LCI；若 PVM 只有 logit 敏感性而无 outcome 决策价值，则只保留为诊断。

### 6.2 推荐论文题目与核心贡献

**题目候选：**

*When Is a Teacher Mixture More Teachable Than Its Teachers? Composition-Level Student-Executable Value for Shared-Expert On-Policy Distillation*

**一句话定位：**在共享 generalist 的多教师 OPD 中，直接估计真实组合教师的一小段指导对当前学生的可执行价值，并研究它何时超过最佳组成教师及单教师价值的线性预测。

**可辩护的四项贡献（均需实验兑现）：**

1. **问题与诊断：**区分教师独立能力、单教师分支价值、实际组合价值和学生可执行性；报告 `D^S`、`C_w`、`N_w`，而不是只看教师平均分或 logit 距离。
2. **方法：**有限候选的真实 `q_w` segment probing、冻结学生 common continuation、独立校准、active G floor 与未认证时 student abstain；再用 router 摊销在线成本。
3. **条件性分析：**在证书覆盖激活状态且未激活严格回退学生时，给出 segment 宏策略的条件性改进及最终学生拟合误差项。经典 performance-difference 本身不是新定理。
4. **受控实证：**在同 continuation-token 与同总 GPU-hours 下，比较 SPOT、FutureBridge、C/D、最佳单教师、静态锚和朴素 MOPD，并检验 sham、direct answer、截断及多步可迁移性。

**必须接受的概念调整。**若采用学生安全标签，就不应再把“每个状态都强制保留 G 且回退 G”与“相对学生不退化”同时作为无条件承诺。修订后的 shared expert 指：**active teacher composition 中 G 始终参与；无可认证候选时允许学生 no-op。**若坚持每状态 G fallback，则必须另证可行非负优势，或保留未认证状态的负优势惩罚。

### 6.3 推荐的阶段决策

1. **P0：固定证据与协议。**冻结学生、教师池、候选集合、segment/continuation 预算、verifier、最小有用效应和数据 split。现有学生基线补齐前不写成论文主表，也不伪造完成状态；独立的科研 probe 可以先回答机制问题。
2. **P1：组合价值证伪。**只做 CoSEV 的最小真实 `q_w` probe，包含 student control、最佳单教师、线性 predictor、sham 与 direct answer。先判断组合增益是否存在及其采样成本。
3. **P2：路由泛化。**仅在 P1 通过后训练轻量 router，在全新题目的新学生 prefix 上与固定 G、domain、固定 mixture、SPOT/FutureBridge 比较。oracle 只作描述，不参与选择。
4. **P3：正式学生训练。**按项目既定训练检查流程直接启动获授权的正式实验；同时做 iso-rollout 与 iso-GPU-hour 对照，至少 3 seed，最后冻结导出和统一评测。不另加默认 GPU smoke。
5. **P4：备选与扩展。**CoSEV 不可行时再评估稀疏 LCI；PVM 只有在独立视觉归因价值成立后作为单独机制方向或后续扩展，不与主方法捆绑。

### 6.4 主线必须回答的四个结果问题

1. **组合存在性：**有多少 prefix 上实际 `q_w` 同时优于学生、G 和最佳单教师？独立评估的正 margin 与激活覆盖率是多少？
2. **非可加性：**`N_w` 是否足够大且方向可预测？若线性 predictor 已足够，为什么还要付出真实组合 probe 成本？
3. **可蒸馏性：**segment 价值是否预测实际 OPD 后学生收益，而不是只预测 student continuation 或短段答案泄露？最终学生保留了多少目标增益？
4. **成本与混杂：**优势是否在相同学生预算和相同总 GPU-hours 下都存在，并在控制长度、截断、sham、direct answer 后仍成立？

若第 1 项失败，停止主线；若第 1 项通过而第 2 项失败，改用便宜的线性/单教师方法；若第 3 项失败，结论只能是 continuation 诊断而非有效蒸馏；若第 4 项失败，只能报告监督质量或特定预算结果，不能声称效率优势。

### 6.5 当前可以写与不能写的结论

- **可以写：**现有教师最终答案集合存在描述性互补；generalist 是强基线；按领域命名的教师分工不稳定；截断是重要混杂；这些结果共同激励学生可执行组合价值研究。
- **不能写：**SharedExpert/MOPD 学生已经优于 baseline；oracle union 是学生上界；R 显著全面强于 G；P2 已稳定改善感知；任何 router 已具备视觉因果识别或安全保证。
- **新颖性措辞：**使用“我们研究组合级学生可执行价值”而非“首次学生续写验证”“首次 shared expert”“首次视觉反事实”或“首次验证集 lookahead”。

## 7. 证据与文献索引

### 7.1 本地实验与设计

- 完整汇总：`experiments/sharedexpert_c/evaluation/results/summary.md`
- 主套件：`experiments/sharedexpert_c/evaluation/results/summary_main.md`
- BLINK/MMVP：`experiments/sharedexpert_c/evaluation/results_blink_mmvp/summary.md`
- 尚未完整的 IFEval/OCR：`experiments/sharedexpert_c/evaluation/results_general/summary.md`
- 逐题输入：`experiments/sharedexpert_c/evaluation/results/per_sample.csv`
- P/R 逐题重合：`experiments/sharedexpert_c/analysis/teacher_overlap_20261022/analysis.json`
- 教师训练曲线：`experiments/sharedexpert_c/results_generalist_step100/metrics.csv`、`experiments/sharedexpert_c/monitoring_perception_v2/metrics.csv`
- 当前 OPD 实现边界：`experiments/sharedexpert_c/distillation/README.md`
- 最近邻文献与已有 A–D：`SharedExpert_MOPD_文献核验.md`、`SharedExpert_MOPD_候选方案与修订骨架.md`、`SharedExpert_MOPD_数学审查.md`

本次四教师 union 使用上述 `per_sample.csv` 重算；MMBench 采用 same-teacher circular-group 口径。R−G 复核使用 20,000 次 paired cluster bootstrap、seed 20261025；MME 与 RefCOCO 按 `group` 聚类，MMBench 使用 250 个 circular groups。新增统计是描述性研究诊断，不改变原始评分文件。

部分 README 的“generalist 尚未完成”等状态属于历史记录；本文以证据快照时的完整结果汇总与逐题文件为准。domain-opd/mix-rl 无学生结果这一点在快照时仍成立。

### 7.2 本次新增直接近邻（摘要已由主会话复核）

- [SPOT: Sparse Probing and Outcome Calibration for On-Policy Distillation](https://arxiv.org/abs/2608.04419)
- [Look Ahead Before You Distill: Future Trajectory Validation of Teacher Guidance for Agentic On-Policy Distillation](https://arxiv.org/abs/2608.01953)
- [VC-OPD: Visual Counterfactual On-Policy Distillation for Grounded Vision-Language Reasoning](https://arxiv.org/abs/2610.14939)
- [VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation](https://arxiv.org/abs/2607.28590)
- [Perception Before Supervision: Self-Contained Visual Distillation from Counterfactual Blind Spots](https://arxiv.org/abs/2608.09931)

### 7.3 LCI 需要正面对照的经典脉络

- [Learning to Reweight Examples for Robust Deep Learning](https://proceedings.mlr.press/v80/ren18a.html)
- [Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v70/finn17a.html)
- [Tailoring Instructions to Student’s Learning Levels / LGTM](https://aclanthology.org/2023.acl-long.111/)
- [Reinforced Multi-Teacher Selection for Knowledge Distillation](https://arxiv.org/abs/2012.06048)
- [Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html)

文献检索用于界定候选差异，不构成系统综述或“没有更早工作”的证明。所有性能结论仍须由后续独立实验获得。
