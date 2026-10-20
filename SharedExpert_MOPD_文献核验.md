# Shared-Expert MOPD 文献核验与新颖性审查

核验日期：2026-10-19。范围：原骨架中最接近的教师锚定、残差、动态路由和能力保持论文；以 arXiv 原始摘要和 HTML 全文为准。下文的“已核验”表示标题、版本与所述机制核对过，不表示其结果已经独立复现。本文是研究设计备忘录，不是系统综述。

## 1. 最重要的结论

原骨架不能把“generalist + specialist 残差”或“token 动态门控”单独作为首次贡献。ChemOPD 已经使用 generalist 作为残差参考；MOPD-Router 与 TrustMOPD 已经做 token 级分配；LEGO-OPD 已经根据每个 prefix 的 KL 预算自适应调整残差权重。把这些组件组合到 VLM 的感知／推理专家上有研究价值，但仅替换参考模型、加一层 gate，创新论证较弱。

更有希望的问题是：**如何判断一个 specialist 在学生当前前缀上的增量价值确实超过 generalist，并在判断不确定时保留 generalist 的监督？** 这需要显式校准增益估计、回退规则，以及说明估计错误如何影响学生的可检验风险界。它仍需与 VG-OPD 的 verifier gate 正面比较。

## 2. 核验表

| 论文与核验源 | 截止前可用版本 | 已确认的机制 | 对本提案的影响 |
|---|---|---|---|
| [ChemOPD: Multi-Teacher On-Policy Distillation for Multi-Task Chemical Reasoning](https://arxiv.org/html/2609.33838v1) | v1，2026-09-27 | SFT 梯度估计任务亲和度，MIP 组织部分重叠专家组；保留全任务 generalist；固定任务路由；generalist advantage 加 specialist-relative correction；线性 warm-up | 是 anchor-residual 的直接先例；不能宣称“本文才用 generalist 作残差参考”。其教师采用同源 SFT 配方，不应误写成 RL 教师。 |
| [MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2609.30837v2) | v2，2026-09-28 | 每个 token 计算全教师池指标；ExpertAlign 衡量 teacher−base 与 teacher−student 的正向余弦对齐，保留教师再均匀或余弦归一化；没有教师通过时跳过该 token | 原骨架 gate 基本是其更换 reference 的版本。需承认继承关系，并证明 generalist-relative 判据为什么更可靠。 |
| [Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2609.23697v3) | v3，2026-10-11 | teacher 相对 shared pre-RL reference 的中心化 log-ratio 范数；除以固定校准集均值；幂归一化为 token weights；无训练 router、无 outcome verifier | “可靠性”“相对 reference”“跨教师校准”“混合能力 prompt”都已有。其 calibration 是尺度归一化，并非统计意义的正确率或置信区间校准。 |
| [Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts](https://arxiv.org/html/2610.10460v1) | v1，2026-10-07 | Δ-MOPD 将每个 teacher 相对其自身 base 的 logit shift 加到冻结学生初始化上；固定 selection 比较 target construction；主要针对异源 base pull | 应准确区分 teacher-specific precursor 与 generalist。原论文说明同源情况下 shift 与 endpoint 项相同，不能把其异源收益直接外推到本提案的同源三教师。 |
| [LEGO-OPD: Factorized Teacher Composition for Multimodal On-Policy Distillation](https://arxiv.org/html/2610.00333v1) | v1，2026-09-29 | LLM prior 加 VLM 有图／无图 logit residual；每个 prefix 用 source KL 校准组合系数；给出精确 KL matching 与二阶方差近似 | **不是固定因子或固定权重方法**。VLM 残差、prefix adaptive weighting、KL budget 与通用推理保持均有较强重叠。 |
| [Who Teaches Which Token? Verifier-Gated Multi-Expert On-Policy Distillation for Scientific Reasoning](https://arxiv.org/html/2609.15404v1) | v1，2026-09，截止前 | VG-OPD 用专家从原问题生成的 greedy probe，验证其是否解决学生失败 criterion；通过后按 teacher disagreement 定位 token，并加权蒸馏；主方案与 GRPO 联用 | 若改用 verifier 增益 gate，这是必须加入的最近邻。其 probe 从原 prompt 开始，不是从相同学生 prefix 续写；比较基准是学生失败答案，不是 generalist。 |
| [Compatibility Matters: A Framework for Token-Level Teacher Selection in On-Policy Distillation](https://arxiv.org/html/2610.21234v1) | v1，2026-10-15 | 分离 teacher competence 与 student–teacher gap；比较 entropy 与 representation routing，后者用 Hellinger／最终隐藏特征距离；分析优化与多样性 | **截止日前已公开，应保留引用**。不能仅凭 2610.21234 的编号判断它来自未来。其理论也提醒：更容易拟合某教师不等于任务奖励更高。 |

日期可分别通过对应 [ChemOPD 摘要](https://arxiv.org/abs/2609.33838)、[MOPD-Router 摘要](https://arxiv.org/abs/2609.30837)、[TrustMOPD 摘要](https://arxiv.org/abs/2609.23697)、[Δ-MOPD 摘要](https://arxiv.org/abs/2610.10460)、[LEGO-OPD 摘要](https://arxiv.org/abs/2610.00333)、[VG-OPD 摘要](https://arxiv.org/abs/2609.15404)、[Compatibility Matters 摘要](https://arxiv.org/abs/2610.21234)中的 submission history 复查。

## 3. 三个容易误判的新颖性点

### 3.1 ChemOPD 已经以 generalist 为残差参考

其 §4.3 的组合信号为

\[
\widehat A_{\mathrm{Chem}}=\widehat A_g+\alpha(s)(\widehat A_i-\widehat A_g),
\]

其中 \(\alpha(s)\) 随 OPD update 线性升高，在同一轮的所有位置共享。论文报告配置从 0 升至 0.5，warm-up 覆盖前 65% updates。因此合理的差异是**动态估计什么时候 specialist 比 generalist 更有用**，而不是“使用 generalist anchor”。来源：[ChemOPD §4.3、Appendix G](https://arxiv.org/html/2609.33838v1)。

### 3.2 LEGO-OPD 已有自适应残差预算

其目标是

\[
q_{\alpha,t}=\operatorname{softmax}(z_{L,t}+\alpha_t[z^I_{V,t}-z^0_{V,t}]),
\]

并令 \(D_{\rm KL}(q_{\alpha,t}\|q_{L,t})\) 匹配缩放后的 grounding expert 有图／无图 KL。二阶实现按 residual 在 receiving prior 下的方差校准系数。因此“KL budget + token adaptive coefficient”应作为借鉴组件，不应承担主创新。其 residual 分离的是图像影响，本提案 residual 则表示两个完整 VLM 教师的差异，二者不能直接等同为纯感知／推理信号。来源：[LEGO-OPD §3.1–3.2](https://arxiv.org/html/2610.00333v1)。

### 3.3 TrustMOPD 没有验证真实任务增益

其分数来自中心化位移范数 \(\rho_{i,t}\)，使用 \(r_{i,t}=\rho_{i,t}/\mu_i\)、\(w_{i,t}\propto r_{i,t}^{\gamma}\)。它将位移当作可靠性的代理，并在理想 KL-regularized Bellman optimum 假设下解释其 utility contrast；这不等于观测到该教师在当前 prefix 的可验证回报优于 generalist。来源：[TrustMOPD §3、§5](https://arxiv.org/html/2609.23697v3)。

## 4. 候选新方法的可区分空间

考虑以下待研究方案：在相同学生前缀 \(h\) 上，让 generalist 与 specialists 分别续写，通过可验证任务回报估计

\[
\Delta_i(h)=V_i(h)-V_g(h),
\]

用这些标签训练轻量预测器，另留 calibration split 估计误差；只有可信的正增益才启用 specialist，否则回退 generalist。当前核验的最近邻中，尚未发现完整采用这一组合的论文。但这是**有限检索下的候选缺口**，不能写成已证明的全球首次。

| 设计维度 | 最近邻已做到 | 候选方案需要拿出什么 |
|---|---|---|
| 前缀条件增益 | VG-OPD 用 prompt-level probe 和 criterion gain；TrustMOPD 用 prefix displacement proxy | 同一 prefix 续写标签，并证明它优于 prompt/domain 标签、位移、熵和 ExpertAlign |
| 比较参照 | ChemOPD generalist anchor；VG-OPD 对学生失败项验证 | 直接检验 specialist 相对 generalist 的增益；额外验证相对学生增益，防止两位教师都不如学生 |
| 不确定性 | TrustMOPD 做尺度归一化；VG-OPD verifier abstention 不授权 | 留出校准、误激活率、可靠性曲线；预测误差或有限样本置信下界进入理论 |
| fallback | ExpertAlign 全不通过时 skip；VG-OPD 无授权时无蒸馏 | 默认 generalist，并证明 fallback 是否改善广域保留；不得把下限本身当性能保证 |
| 多模态内容 | LEGO-OPD 控制视觉因素；SF-MOPD 保持通用能力 | 混合感知／推理前缀上的条件价值变化；可验证的 grounding 改进与 reasoning 改进分开报告 |

需要诚实面对两个技术限制：

1. 从完整专家续写得到的 \(V_i(h)>V_g(h)\)，并不自动证明只替换当前一步或混合 logits 会提升终局奖励。必须明确理论比较对象是续写选择、局部 surrogate，还是具有额外假设的学生性能；不可偷换。
2. 轻量 router 的区间只有在统计条件成立时才是置信保证。对固定 prefix 的 Monte Carlo 误差界不等于对任意新 prefix 的预测误差界；on-policy 分布变化还需要刷新校准或显式分布偏移假设。

上述限制来自本次对候选方法的分析，而非任一论文已经替本提案证明的结论。

## 5. 通用能力保持相关工作

这三篇已核对摘要，机制可引用；本次未逐式复核全部附录证明。

| 论文 | 日期／机制 | 本提案应如何区分 |
|---|---|---|
| [CaMOPD: Counteraction-Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain Preservation](https://arxiv.org/abs/2605.27115) | v2，2026-09-14；将 general recovery 与 domain preservation 交替解耦，按 teacher–student gap 选样本 | 不能说此前没有 general/domain 保持；比较交替训练与同一 prefix 的锚定是否更好 |
| [UCMOPD: Recovering General Capabilities via Uncertainty-Calibrated Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2608.26735) | v2，2026-09-28；高温探索对照常温 anchor、trajectory selection、centered log-likelihood token endorsement | 其“anchor”包含采样参照含义，不应混同为本文 shared teacher；可靠性 gating 也不是新概念 |
| [SF-MOPD: Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation](https://arxiv.org/abs/2610.02324) | v1，2026-10-01；EMA slow model 作动态能力参考，移除使 fast model 进一步远离 slow model 的 teacher update 分量 | 是 VLM capability preservation 的直接 baseline；本文固定 generalist 必须检验是否过度限制新能力 |

## 6. 对原骨架应立即修改的表述

- 删除“首次 shared/routed 教师角色划分”；用“研究 generalist-relative 增益识别与保守启用”作为候选定位。
- 删除“ChemOPD 的残差参考不是 generalist”；原论文本来就是 generalist-relative correction。
- 将 LEGO-OPD 的“固定因子／固定粒度”改为“prefix 自适应 source-KL 校准”。
- 将所有未做实验的“超过”“一致收益”“把退化压到更低”改成研究目标、待检验假设或结果占位。
- MOPD-Router 的 1.18× 是特定八卡 H20、同规模有标签配置的 token-normalized 测量；不能直接当作本文续写 verifier 或新增 router 的成本预测。[原成本表](https://arxiv.org/html/2609.30837v2)
- 主 baseline 至少包含 ChemOPD-style warm-up、ExpertAlign、TrustMOPD、VG-OPD-style verifier gate、LEGO-style budget、SF-MOPD；在统一教师池和总算力预算下比较。
- 本次未核验 VGS 的角度数字、VA-OPD 的 93% 数字及其适用条件。它们不能直接证明 perception 与 reasoning residual 正交，也不能在未经逐表核对前用作定理假设的事实依据。

## 7. 本轮检索边界

除上述原论文外，还对 prefix verifier router、multi-teacher confidence bound、generalist value 等组合进行了检索；没有找到比 VG-OPD 更直接的同类方案，但搜索结果不构成不存在证明。原骨架其余条目（例如 DN-MOPD、D³-MOPD、PMOPD、UP-MOPD、Road MOPD、C-MOPD、Latent-MOPD 等）本轮未逐篇全文核验，不应把本报告解释为全部参考文献的背书。

## 8. 候选 D 的经典理论来源与边界（补检）

以下三篇原始论文足以建立 D 的理论出处。它们并未直接研究 VLM 多教师 OPD；D 的策略改进式应写成经典理论的实例化，而不是新发现的策略改进定理。

| 原始文献 | 核验内容与 D 的关系 | 不可省略的边界 |
|---|---|---|
| Kakade & Langford, ICML 2002, [Approximately Optimal Approximate Reinforcement Learning（作者原文）](https://www.cs.cmu.edu/~jcl/papers/aoarl/Final.pdf)；[Berkeley 镜像](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) | Conservative Policy Iteration 用当前策略与近似 greedy policy 的概率混合。§4.1 利用相对当前策略的 advantage 说明保守步长下的性能改进；理论设计依赖 restart distribution 和 approximate greedy policy chooser。D 的共同后续策略正是标准 Q/advantage 定义。 | 平均 advantage 为正不允许任意大混合步长；原文控制步长及状态分布影响。D 使用“所有相关状态的一步 advantage 非负”这一更强条件时，可直接由 policy improvement 得非退化；不能只从少量学生前缀的平均正值推出它。 |
| Laroche et al., ICML 2019, [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)；[全文](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf) | SPIBB 在样本不足的 state–action pairs 保持 baseline 的动作概率，在其余部分优化；Theorem 2 给出有限 MDP 下带误差余量的高概率改进界。支持“不确定时回退 baseline”的经典来源。 | 要有 baseline、样本计数／模型误差控制和明确策略约束；不是随意设 shared weight floor 就可继承保证。\(\Pi_{\le b}\) 实用变体不保留原 SPI 保证；深度 pseudo-count 实现也不能直接享有完整有限表格情形保证。 |
| Thomas et al., ICML 2015, [High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.html)；[全文](https://proceedings.mlr.press/v37/thomas15.pdf) | 利用独立轨迹和 off-policy performance lower bound 对候选策略做性能检验，允许返回无法确认改进的结果；训练／检验数据独立性是避免选择偏差的关键。适合作为 C/D 的有效置信下界与保留评估集的依据。 | 严格 finite-sample 保证来自相应 concentration-inequality 版本；t-test 与 bootstrap 版本作者明确区分为 semi-safe。高方差、数据需求和长轨迹是现实限制；不能把少量重复续写的 bootstrap 区间直接称为全状态有效保证。 |

对《候选方案与修订骨架》§6 的快速检查：当前没有明显“首次”误称，且已正确区分组合策略、学生拟合误差与相对 MOPD 的附加条件。建议把 §6.3 标为“经典 policy improvement 的多教师实例化”，并在 Related Work 补入上述三篇。

D 真正待验证的研究贡献应是：如何在 VLM 前缀上低成本、可靠地估计共同续写 advantage；如何校准并覆盖组合策略可能访问的新前缀；以及该监督能否被学生有效吸收。概率混合的线性优势公式、非负 advantage 推出改进、置信不足时回退本身都有明确经典先例。
