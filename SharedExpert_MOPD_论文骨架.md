# Shared-Expert MOPD 论文骨架

> 主题：**Generalist-Anchored Gated Residual Distillation for Multimodal Capability Integration**
> 一句话：把 MoE 的「shared expert + routed experts」搬到**教师空间**——一个 always-on 的通用多模态教师做公共先验与能力锚，感知/推理两个 specialist 只提供 token 级门控的**残差增量**。
> 说明：第 5 节的理论是**带假设的分析框架（Proposition + proof sketch）**，不是已发表定理；写作时以「我们给出界与可证伪预测，并用实验验证」呈现。

---

## 0. 标题与定位

**标题候选**
1. *Shared-Expert MOPD: Generalist-Anchored Residual Distillation for Multimodal Capability Integration*
2. *Anchor and Route: Rethinking Teacher Roles in Multi-Teacher On-Policy Distillation for VLMs*
3. *Common-Mode Rejection in Multimodal Distillation: A Shared-Expert Framework for MOPD*

**定位**：不是「新的 MoE 架构」，而是**教师角色的重新划分**（shared vs routed）＋**残差组合与门控**＋**VLM 特定的理论解释**。

---

## 1. Abstract 骨架（5 句）

1. 多教师 on-policy 蒸馏（MOPD）是把多个域专家整合进一个学生的有效范式，但现有做法把教师当**平级 peers**（prompt 硬路由或等权平均）。
2. 这带来三个问题：公共语言先验被重复学习、教师噪声直接进入梯度、无锚点导致通用能力税（capability tax）。
3. 我们提出 **Shared-Expert MOPD**：一个在宽数据上 RL 的通用多模态教师作为 **always-on anchor**，感知与推理两个 specialist 只提供**相对 generalist 的、token 级门控的残差**。
4. 我们从理论上刻画了该设计的四个性质：共模抑制（方差下降）、误路由鲁棒性（误差上界只与残差幅度有关）、能力税上界（由锚权重与残差预算控制）、以及与 reverse-KL 几何聚合目标的一致性。
5. 在图像/视频多模态基准上，该方法在感知与推理维度上超过 generalist-only 与 flat MOPD，同时把通用能力退化压到更低；消融验证了四个理论预测。

---

## 2. Introduction（5 段式）

**P1（背景）**：VLM 后训练中，不同能力（细粒度感知/grounding、数学/STEM 推理、通用指令）常由独立 RL 专家获得；MOPD 把它们蒸馏进一个学生，避免 Mix-RL 的域冲突与参数合并的能力破坏（MOPD 2606.30406）。

**P2（Gap）**：现有 MOPD 把教师视为**平级**：要么按 prompt 域标签硬路由（MOPD、Video-MOPD、UI-MOPD），要么等权/标量加权聚合（Open-MOPD、DN-MOPD 的动机）。后果：
- (i) 每个 token 只看到一个教师，公共语言先验被反复学习，稀疏的专家信号被稀释；
- (ii) 域标签错误或缺失时，误路由代价是「教师全分布距离」；
- (iii) 没有能力锚，通用能力随多域蒸馏漂移（SF-MOPD/UCMOPD/CaMOPD 的共同问题）。

**P3（Key insight）**：MoE 的 shared expert 吸收了公共模式、routed experts 只处理特化，从而降低冗余与路由不稳定。教师空间存在同样的结构：VLM 教师共享一个很强的**语言先验/通用描述分量**（VGS 证明语言先验与视觉 grounding 梯度近乎正交；LEGO 证明 VLM 完整分布中二者纠缠），而真正的能力差异是**相对这个公共分量的残差**。

**P4（Method）**：Shared-Expert MOPD：
- generalist 教师 $T_g$（宽数据 RL）**always-on**，提供公共先验与能力锚；
- 感知专家 $T_p$、推理专家 $T_r$ 只贡献残差 $\Delta_i=\log p_i-\log p_g$；
- token 级 gate $g_{i,t}$ 决定残差是否/以多强加入：$z_{\text{eff}}=z_g+\sum_i\lambda_i g_{i,t}\Delta_i$；
- gate 复用 ExpertAlign 思想，但参考从「pre-RL base」换成 **generalist**；配合域归一化控制尺度。

**P5（Contributions）**：
1. **问题重述**：首次把 MOPD 的教师角色划分为 shared（公共先验/能力锚）与 routed（能力残差），并给出 VLM 下的形式化。
2. **方法**：generalist-anchored gated residual，包含 specialist gate、域归一化、与 reverse-KL 几何聚合的一致性设计。
3. **理论**：共模抑制（Prop 1）、误路由鲁棒性（Prop 2）、能力税上界（Prop 3）、聚合一致性（Prop 4）＋ VLM 特定的正交/稀疏性分析（Prop 5），每一条都有可测预测。
4. **实证**：在感知/推理/通用三类基准上相对 generalist-only、flat MOPD、ChemOPD 式 anchor-residual、Δ-MOPD、LEGO 式因子化均有一致收益；消融验证理论预测。

---

## 3. Related Work（5 小节 + 定位表）

**3.1 Multi-Teacher On-Policy Distillation**：MOPD 范式（2606.30406）、路由（MOPD-Router 2609.30837、TrustMOPD 2609.23697、VG-OPD 2609.15404、Compatibility Matters 2610.21234）、聚合/预算（Open-MOPD 2608.19098、DN-MOPD 2609.35347、D³-MOPD 2608.24987）、优化几何（PMOPD 2609.34605、UP-MOPD 2610.08398、Road MOPD 2610.13521）、目标/表征（Δ-MOPD 2610.10460、Latent-MOPD 2610.02381、C-MOPD 2609.33791）。

**3.2 Generalist + Specialist 的已有形态**：ChemOPD（2609.33838，generalist + 部分重叠专家组 + anchor-residual）、CaMOPD（2605.27115，通用恢复与域保持解耦）、UCMOPD（2608.26735，通用教师 + 域教师）、SF-MOPD（2610.02324，slow EMA 作为能力参考）。→ 本文与它们的区别在**能力轴（感知/推理）、token 级门控残差、以及理论刻画**。

**3.3 残差/位移目标**：Δ-MOPD（teacher−base shift）、VAD（反事实目标重建，OPSD 侧）、FP-OPD（Fisher 投影）、LEGO-OPD（语言先验 × 视觉似然）。→ 本文的参考是 **generalist 教师**而非 base，且残差被 token 级门控。

**3.4 VLM 的视觉证据与语言先验**：VA-OPD（视觉 advantage，top 10% token 承载约 93%）、VGS（loss 分解、梯度近正交）、VC-OPD/S-OPD（视觉依赖缺口、学生侧感知）。→ 本文把这些诊断用作 **VLM-specific 理论假设的经验支撑**。

**3.5 MoE 的 shared/routed 思想**：DeepSeekMoE 等 shared-expert 设计证明 shared 吸收公共模式、routed 处理特化并提升路由稳定性。→ 本文是**教师空间的功能类比**，不是架构复现。

**定位表（写作时放 Related Work 末尾）**

| 维度 | ChemOPD | Δ-MOPD | LEGO-OPD | 本文 |
|---|---|---|---|---|
| generalist | 保留、专家增补 | 无（base 参考） | 无（语言专家做先验） | **always-on anchor** |
| specialist 残差 | anchor-residual（标量 schedule） | teacher−base shift | 因子化组合 | **generalist 参考的 token 级门控残差** |
| 粒度 | 任务级 | 样本/教师级 | 固定因子 | **token 级 gate** |
| 理论 | 经验 | 机制观察 | 贝叶斯分解 | **共模/误路由/能力税/聚合** |
| VLM 特定性 | 无 | 无 | 有（语言先验×视觉似然） | **有（正交子空间 + 稀疏证据 + 能力税）** |

---

## 4. Preliminaries & Problem Setup

### 4.1 记号

- 学生 $\pi_\theta$；共享 pre-RL base $\pi_0$；generalist $T_g=\pi_{\phi_g}$；感知专家 $T_p$、推理专家 $T_r$。
- 学生 rollout $y\sim\pi_\theta(\cdot|x)$；前缀 $h_t=(x,y_{<t})$；教师下一 token 分布 $p_{i,t}=\pi_{\phi_i}(\cdot|h_t)$，logits $z_{i,t}$；学生 logits $z_{\theta,t}$。
- MOPD 基础目标（reverse KL，token 平均）：
$$\mathcal{L}_{\text{MOPD}}(\theta)=\mathbb{E}_{x}\mathbb{E}_{y\sim\pi_\theta}\frac{1}{|y|}\sum_t \mathrm{KL}\big(\pi_\theta(\cdot|h_t)\,\big\|\,\pi_{\text{route}(x)}(\cdot|h_t)\big)$$

### 4.2 教师 logit 的加性分解（分析假设 A1）

假设每个教师的 logits 可写成
$$z_{g,t}=c_t+\varepsilon_{g,t},\qquad z_{p,t}=c_t+s_{p,t}+\varepsilon_{p,t},\qquad z_{r,t}=c_t+s_{r,t}+\varepsilon_{r,t}$$
其中 $c_t$ 是**教师共享的公共分量**（VLM 的语言先验/通用描述能力），$s_{i,t}$ 是 specialist 的**能力分量**，$\varepsilon_{i,t}$ 是教师特有噪声（风格/采样/校准），假设跨教师独立、零均值、方差 $\sigma^2$。
> 这不是「教师真的这样分解」，而是分析用的工作假设；第 5 节会明确假设与可证伪预测。

### 4.3 方法目标（提前写出，便于理论对照）

$$z_{\text{eff},t}=z_{g,t}+\sum_{i\in\{p,r\}}\lambda_i\,g_{i,t}\,\Delta_{i,t},\qquad \Delta_{i,t}=z_{i,t}-z_{g,t}$$
$$g_{i,t}=\mathbb{1}\big[\langle e_{i,t},d_{i,t}\rangle>\delta\big]\cdot c_{i,t},\quad e_{i,t}=\Delta_{i,t},\ d_{i,t}=z_{i,t}-z_{\theta,t},\ c_{i,t}=\max(\cos(e_{i,t},d_{i,t}),0)$$
$$\mathcal{L}=\mathbb{E}\Big[\mathrm{KL}\big(\pi_\theta\,\big\|\,\mathrm{softmax}(z_{\text{eff}})\big)\Big]+\mu\,\mathrm{KL}\big(\pi_\theta\,\|\,T_g\big)$$
（$\mu$ 为能力锚系数；实现中可由 generalist 项主导，$\mu$ 用于理论上的 trade-off 分析。）

---

## 5. Theoretical Analysis（5 个 Proposition + 可测预测）

> 写作建议：正文给「假设 + 陈述 + 直觉 + 可测预测」，完整证明放附录；每个 Proposition 后紧跟一句「这一预测在 §7.x 被验证」。

### Proposition 1（共模抑制 / 方差下降）

**假设**：A1（加性分解、跨教师独立噪声）。
**陈述**：令 $\Lambda=\sum_i\lambda_i\le 1$。残差目标的噪声为
$$\varepsilon_{\text{eff}}=(1-\Lambda)\varepsilon_g+\sum_i\lambda_i\varepsilon_i,\qquad \mathrm{Var}(\varepsilon_{\text{eff}})=\Big[(1-\Lambda)^2+\sum_i\lambda_i^2\Big]\sigma^2\le\sigma^2$$
等权 $\lambda_i=\Lambda/k$ 时最小值为 $\big[(1-\Lambda)^2+\Lambda^2/k\big]\sigma^2$（$\Lambda=0.5,k=2$ 时约 $0.375\sigma^2$）；而 flat 硬路由的噪声方差为 $\sigma^2$。
**证明草图**：直接由独立性展开；凸组合方差不超过最大值。
**可测预测**：
- 固定专家信号时，残差目标的梯度噪声/token 级方差低于 flat MOPD；
- 公共分量占比越大（VLM 语言先验越强），收益越大 → 在「语言先验更强的教师对」上差距更明显。

### Proposition 2（误路由鲁棒性）

**假设**：A1；路由/门控有错误概率；flat MOPD 的误路由代价为教师间全分布距离。
**陈述**：
$$\text{err}_{\text{flat}}\le \max_{i,j}\lVert z_i-z_j\rVert,\qquad \text{err}_{\text{shared+residual}}\le \Lambda\max_i\lVert\Delta_i\rVert$$
当公共分量占主导（$\lVert c\rVert\gg\lVert s_i\rVert$）时，$\lVert\Delta_i\rVert\ll\lVert z_i-z_j\rVert$，残差方案的误路由误差小一个量级；期望误差分别为 $P_{\text{route}}^{\text{err}}\cdot\max\lVert z_i-z_j\rVert$ 与 $P_{\text{gate}}^{\text{err}}\cdot\Lambda\max\lVert\Delta_i\rVert$。
**证明草图**：误激活对目标的扰动 = 被错误加入/去掉的残差项；flat 路由错则整条分布换成另一教师。
**可测预测**：向训练数据注入域标签噪声/路由噪声，flat MOPD 退化显著快于本方法；在无标签混合数据上优势更大（对应 MOPD-Router 的无标签场景）。

### Proposition 3（能力税上界）

**假设**：目标含锚项 $\mu\,\mathrm{KL}(\pi_\theta\|T_g)$；残差预算 $\lVert\lambda_i\Delta_i\rVert\le B$；评测损失有界 $L$。
**陈述**：最优学生满足
$$\mathrm{KL}(\pi_\theta\|T_g)\le \frac{1}{\mu}\cdot\big(\text{特化收益项}\big)$$
由 Pinsker + performance difference lemma，通用能力退化
$$\big|J(T_g)-J(\pi_\theta)\big|\le L\sqrt{2\,\mathrm{KL}(\pi_\theta\|T_g)}\le L\sqrt{\frac{2}{\mu}\cdot(\text{特化收益项})}$$
即能力税由 $\mu$ 与残差预算控制，**与教师数量无直接关系**；flat MOPD 无锚项，能力税随教师冲突增大。
**可测预测**：能力税–特化增益曲线随 $\mu$ 单调变化；存在最优 $\mu^\star$；教师数从 2→3 时 flat MOPD 的能力税上升更快，本方法更平。

### Proposition 4（与 reverse-KL 几何聚合的一致性）

**假设**：reverse KL 多教师最优解为归一化几何聚合（2609.38666）。
**陈述**：本方法的残差 logit 目标对应
$$p_{\text{eff}}\propto p_g^{\,1-\Lambda}\prod_i p_i^{\lambda_i}$$
的一阶近似；flat 硬路由不对应任何单一聚合目标（条件选择），因此目标随 token 跳变、目标方差更大。
**证明草图**：$\log p_{\text{eff}}=(1-\Lambda)\log p_g+\sum_i\lambda_i\log p_i$，在 $z_g$ 附近一阶展开即残差式；硬路由的指示函数不是任何固定权重的聚合。
**可测预测**：残差目标的 token 间目标方差低于 flat MOPD；训练曲线更稳（loss 方差、梯度范数）。

### Proposition 5（VLM 特定：正交子空间 + 稀疏证据）

**假设**：A1；(a) 感知残差与推理残差近似正交（VGS：语言先验与视觉 grounding 梯度夹角 60–92°）；(b) 只有比例 $\rho$ 的 token 承载视觉/时序证据（VA-OPD：top 10% token 承载约 93% 的视觉信号）。
**陈述**：
- (a) 若 $\Delta_p\perp\Delta_r$ 且与 generalist 更新方向近似正交，则总更新范数近似 $\sqrt{\lVert g_g\rVert^2+\lVert g_p\rVert^2+\lVert g_r\rVert^2}$，无破坏性干涉 → 能力税更低；
- (b) token 级稀疏门控把残差噪声方差按激活比例 $\rho$ 缩放，语言脚手架 token 不被专家残差污染。
**证明草图**：余弦展开 + 方差按激活比例缩放。
**可测预测**：
- 感知残差与推理残差的余弦相似度显著低于 flat MOPD 中两教师校正的余弦；
- gate 激活 token 与视觉证据 token（VA-OPD/VC-OPD 式诊断）显著重合；
- $\rho$ 越低，通用能力税越低。

---

## 6. Method

### 6.1 总览（Fig. 1）

```
                ┌───────────────┐
   prompt  ───► │ Student π_θ   │──► rollout y
                └──────┬────────┘
                       │ 每个前缀 h_t
        ┌──────────────┼──────────────────────────┐
        ▼              ▼                          ▼
  ┌──────────┐   ┌────────────┐            ┌────────────┐
  │Generalist│   │Perception  │            │Reasoning   │
  │  T_g     │   │ specialist │            │ specialist │
  └────┬─────┘   └─────┬──────┘            └─────┬──────┘
       │ z_g           │ Δ_p                     │ Δ_r
       │               ▼                         ▼
       │        ┌──────────────┐          ┌──────────────┐
       │        │ Gate g_{p,t} │          │ Gate g_{r,t} │
       │        └──────┬───────┘          └──────┬───────┘
       └───────────────┴──────────┬───────────────┘
                                  ▼
                    z_eff = z_g + Σ λ_i g_{i,t} Δ_i
                                  ▼
                    reverse-KL to softmax(z_eff) + μ·KL to T_g
```

### 6.2 三个教师

- **Generalist $T_g$**：宽数据、同源 base、RL 训练（不是 SFT）；覆盖通用 VQA/视频/文档/格式（数据见 §7.1）。
- **Perception specialist $T_p$**：非文本细粒度视觉 + grounding（计数/空间/属性/小目标/bbox/point），用可验证奖励 RL。
- **Reasoning specialist $T_r$**：数学/STEM/逻辑，用可验证答案 RL。
- **同源约束**：三者都从同一个 pre-RL/SFT checkpoint 出发（MOPD 2606.30406：跨族更强教师会退化；Δ-MOPD：base pull 可能超过 post-training shift）。

### 6.3 Specialist gate（关键设计）

对每个 token、每个 specialist：
$$e_{i,t}=\Delta_{i,t}=z_{i,t}-z_{g,t},\qquad d_{i,t}=z_{i,t}-z_{\theta,t}$$
$$c_{i,t}=\max\big(\cos(e_{i,t},d_{i,t}),0\big),\qquad g_{i,t}=\mathbb{1}\big[\langle e_{i,t},d_{i,t}\rangle>\delta\big]\cdot c_{i,t}$$
**直觉**：$e$ 是「specialist 相对 generalist 的能力增量方向」，$d$ 是「它现在想把学生往哪推」；两者正对齐 → specialist 是在用自己的专长教，而不是泛泛地与 generalist 意见不同。这与 MOPD-Router 的 ExpertAlign 同构，但**参考从 pre-RL base 换成了 generalist**（因此 gate 直接度量「超出通用能力的增量」）。

### 6.4 归一化与聚合

- **域尺度归一化**（DN-MOPD）：$w_d=\mathrm{clip}(\sigma_{all}/\sigma_d,0.25,4)$ 乘在残差项上，防止某个 specialist 主导（Open-MOPD：域间长度差可达 25×）。
- **token 预算平衡**（Open-MOPD 的 token-share balancing）作为可选组件。
- **λ 的约束**：$\sum_i\lambda_i\le 1$（Prop 1 的凸组合条件）；默认 $\lambda_p=\lambda_r=0.25$，做 λ sweep。

### 6.5 训练流程（Algorithm 1）

1. 共享 base → 分别 RL 出 $T_g,T_p,T_r$（同源）；
2. 学生从同一 base 初始化；每个 step 采样 rollout；
3. 在每前缀上算 $z_g,z_p,z_r,z_\theta$；
4. 算 gate $g_{i,t}$、残差 $\Delta_i$、$z_{\text{eff}}$；
5. 用 reverse-KL to softmax($z_{\text{eff}}$) + 锚项更新学生；
6. 周期性验证（归一化恢复率、能力税），可选 IM-MOPD 式初始化/欠恢复域残差补强。

### 6.6 与 ChemOPD 的区别（正文一段 + 表格）

ChemOPD 是「generalist + 部分重叠专家组 + anchor-residual（标量 schedule）」解决化学多任务；本文是「**能力轴**（感知/推理）+ **token 级门控残差** + **VLM 特定理论**」。相同点是 generalist 保留、专家增补；不同点在残差参考（generalist 而非任务亲和）、粒度（token）、分析（共模/误路由/能力税/聚合）与对象（VLM）。

### 6.7 复杂度

每 token 多一次 generalist 前向与 teacher 前向（若 MOPD 本来就要全教师评分，则几乎无额外成本）；多 $\Delta$ 计算与 gate（O(k·|V|) 或 top-k 近似）。参照 MOPD-Router：端到端 token 成本约 1.18×。

---

## 7. Experiments

### 7.1 训练数据与教师

| 教师 | 数据 | 奖励 |
|---|---|---|
| Generalist | 宽 SFT（LLaVA-OneVision-Data/The Cauldron/Cambrian 选 2–3）+ 视频（LLaVA-Video-178K 等）+ 文档图表 10–20% + 格式 | 可验证短答案 + 规则格式 |
| Perception | 非文本细粒度 VQA + grounding（RefCOCO 系/point/count/space） | exact-match / IoU / point-in-box |
| Reasoning | MathVista/MathVision/Geometry3K 等数学与 STEM | 可验证答案 |

> 留出所有评测 benchmark，不进入训练。

### 7.2 评测（三类，缺一不可）

- **感知/grounding**：RefCOCO/+/g、PointBench、CountBench、CV-Bench、RealWorldQA（选 3–4 个）
- **推理**：MathVista、MathVision、MMMU-Pro、LogicVista
- **通用/能力税**：MMBench、MMStar、SEED、OCRBench（作为留出通用集）
- 可选视频：TempCompass、MVBench、Video-MMMU

### 7.3 Baselines

1. 学生初始模型（base）
2. **Generalist-only**（$T_g$ 单教师 OPD）
3. **Flat 3-teacher MOPD**（等权/域标签硬路由，MOPD 原配方）
4. Flat + DN-MOPD 归一化
5. **ChemOPD 式** generalist + specialist + anchor-residual（最接近 prior）
6. **Δ-MOPD 式** teacher−base 残差
7. **LEGO 式** 因子化（若可复现）
8. MOPD-Router / TrustMOPD（token 级路由对照）
9. 强 SFT on teacher rollouts（2610.04272 对照）、最强单教师

### 7.4 主结果表（骨架）

| 方法 | 感知 avg | 推理 avg | 通用 avg | 能力税 ↓ | 相对最强单教师 |
|---|---|---|---|---|---|
| base | | | | | |
| Generalist-only | | | | | |
| Flat MOPD | | | | | |
| ChemOPD-style | | | | | |
| Δ-MOPD-style | | | | | |
| **Ours** | | | | | |

指标定义：
- 归一化恢复率 $\tilde r_d=(S_d-S_d^{\text{init}})/(S_d^{\max}-S_d^{\text{init}})$（0=初始学生，1=该域最强教师）
- **特化增益** $=S_d^{\text{ours}}-S_d^{\text{generalist-only}}$
- **能力税** $=\frac{1}{|G|}\sum_{g\in G}(S_g^{\text{generalist-only}}-S_g^{\text{ours}})$

### 7.5 消融（必须）

| 消融 | 验证 |
|---|---|
| generalist on/off; λ sweep | H1/H2，Prop 1/3 |
| 残差 vs 等权 vs 几何聚合 | 组合方式 |
| gate on/off; gate 准则（ExpertAlign/置信度/Novelty） | 门控必要性 |
| token 级 vs sample 级 gate | 粒度 |
| 路由噪声注入（标签 flip） | Prop 2 |
| μ 锚系数 sweep；教师数 2→3 | Prop 3 |
| gate 激活 vs 视觉证据 token 重合 | Prop 5(b) |
| 残差余弦/正交性测量 | Prop 5(a) |
| 教师规模（generalist 8B vs 32B；specialist 4B/8B） | 规模非单调 |

### 7.6 理论验证实验（让审稿人看到预测被打勾）

- **Prop 1**：token 级梯度方差/噪声（用重复采样估计）——ours < flat；
- **Prop 2**：标签噪声 x 轴 → 性能退化曲线——flat 更陡；
- **Prop 3**：能力税 vs 特化增益曲线随 μ/λ 移动；
- **Prop 4**：target 的 token 间方差、训练 loss 方差；
- **Prop 5**：残差间余弦、gate 与证据 token 的 IoU、ρ 与能力税关系。

### 7.7 定性分析

- gate 在一条 rollout 上的热图：感知 token（小字/计数/空间）激活 perception gate，数学步骤激活 reasoning gate，格式 token 不激活；
- 失败案例：generalist 本身不会的样本（specialist 也救不了）。

---

## 8. Discussion & Limitations

1. **同源假设**：跨家族/更强教师会破坏同源前提；理论分析也假设公共分量可被 generalist 代表。
2. **generalist 的强度**：若 generalist 在 specialist 域不弱，specialist 残差可能趋于 0（H1 失败）；需要确保 specialist 在域内严格更强（用归一化恢复率验证）。
3. **理论假设**：logit 加性分解、噪声独立、凸 λ 只是分析假设；正文明确「这是解释性框架，不是普适定理」。
4. **数据重叠**：perception 与 OCR 数据要分开，否则 gate 无法区分。
5. **成本**：多教师 RL 成本；推理时只保留学生，无额外成本。

---

## 9. Conclusion（3 句）

1. 教师不应被当作平级 peers：shared generalist 吸收公共先验，routed specialists 提供能力残差。
2. 我们给出 generalist-anchored gated residual 的方法与四个理论刻画，并在多模态基准上验证。
3. 该框架可推广到视频/文档等更多能力轴，也为 MOPD 的「共享 vs 路由」提供了一个可检验的设计原则。

---

## 10. Appendix 计划

- A. 证明：Prop 1–5 完整推导与假设讨论；
- B. 教师训练细节（数据、RL 超参、同源 base）；
- C. 额外 benchmark 与视频结果；
- D. gate 可视化与路由统计；
- E. 与 ChemOPD/Δ-MOPD/LEGO 的实现细节对比；
- F. 计算成本（GPU-hours、token 归一化成本）。

---

## 11. 图表计划

| 图/表 | 内容 |
|---|---|
| Fig. 1 | 架构图（generalist anchor + 两个 gated residual specialists） |
| Fig. 2 | gate 热图（token × teacher，对应感知/推理/格式 token） |
| Fig. 3 | 能力税 vs 特化增益曲线（μ/λ sweep） |
| Fig. 4 | 路由噪声鲁棒性曲线（Prop 2） |
| Fig. 5 | 残差余弦/正交性分析（Prop 5） |
| Table 1 | 主结果（三类 benchmark + 能力税） |
| Table 2 | 消融 |
| Table 3 | 与 closest prior（ChemOPD/Δ-MOPD/LEGO）的对照 |

---

## 12. Reviewer 攻击与回应（写作时埋好）

| 攻击 | 回应 |
|---|---|
| 「这不就是 ChemOPD 的 generalist+specialist？」 | 我们是**能力轴**分解 + token 级门控残差 + VLM 理论；把 ChemOPD 作为 baseline 并展示差异（感知/推理维度、能力税、门控覆盖率）。 |
| 「shared expert 只是 MoE 类比，没有架构创新」 | 明确是**教师空间的功能类比**；贡献在 formalization + 理论 + 实证，不声称架构新颖。 |
| 「收益来自多训了一个教师」 | 对照 generalist-only、最强单教师、强 SFT；做教师数消融与成本核算（2610.04272 教训）。 |
| 「理论假设太强」 | 假设写清楚 + 每个 Proposition 给可测预测并做实验验证 + 附录讨论失效条件。 |
| 「token 门控是否稳定/是否只是变相路由」 | gate 对照（无门控/置信度/Novelty）、覆盖率统计、与 MOPD-Router 对比。 |
| 「VLM 特定性在哪？」 | Prop 5 + 语言先验共模（VGS/LEGO）+ 视觉证据稀疏（VA-OPD）+ 异构教师输入的统一参考。 |

---

## 13. 关键引用（写作时直接取用）

- MOPD：https://arxiv.org/abs/2606.30406
- Open-MOPD：https://arxiv.org/abs/2608.19098
- DN-MOPD：https://arxiv.org/abs/2609.35347
- MOPD-Router：https://arxiv.org/abs/2609.30837
- TrustMOPD：https://arxiv.org/abs/2609.23697
- VG-OPD：https://arxiv.org/abs/2609.15404
- Δ-MOPD：https://arxiv.org/abs/2610.10460
- Latent-MOPD：https://arxiv.org/abs/2610.02381
- ChemOPD：https://arxiv.org/abs/2609.33838
- LEGO-OPD：https://arxiv.org/abs/2610.00333
- CaMOPD：https://arxiv.org/abs/2605.27115
- UCMOPD：https://arxiv.org/abs/2608.26735
- SF-MOPD：https://arxiv.org/abs/2610.02324
- VA-OPD：https://arxiv.org/abs/2605.21924
- VGS：https://arxiv.org/abs/2606.00564
- Video-MOPD：https://arxiv.org/abs/2609.09300
- 2610.04272（负结果/成本）：https://arxiv.org/abs/2610.04272
- 2609.38666（reverse-KL 几何聚合）：https://arxiv.org/abs/2609.38666

