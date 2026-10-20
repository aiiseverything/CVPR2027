# Shared-Expert MOPD：数学审查与可证明的替代结论

> 本报告审查原始《SharedExpert_MOPD_论文骨架.md》的五个 Proposition、gate 和优化目标。不修改原文件。本文中的新命题是本次推导，不能当作已发表结论引用；涉及相关文献时单独给出来源。核心结论是：可以证明几何聚合恒等式、锚点偏离界、权重扰动界，以及带明确统计假设的风险改善条件；不能证明这种方法在所有任务上优于朴素 MOPD。

## 1. 首先必须修正的结论

| 原文位置 | 问题 | 应如何修改 |
|---|---|---|
| Prop 1：共模抑制导致更低方差 | 固定凸权重下的独立噪声方差公式正确，但它同样适用于普通加权 logit 平均。残差本身的噪声反而是两份噪声之差。门控依赖同一份噪声时，原独立性公式不能直接套用。 | 称为“带偏差的锚点收缩”，同时分析偏差、方差和相关性。比较对象必须写明是单教师硬路由还是同权重软聚合。 |
| Prop 1：“公共分量越大，收益越大” | 不由公式推出。公共分量的大小不出现在该方差表达式中。 | 删除理论预测；若有兴趣，作为尚待定义和验证的经验假说。 |
| Prop 2：“公共分量大，所以教师间距离大而残差小” | 公共项在两个教师相减时也消失：\(z_i-z_j=s_i-s_j+\epsilon_i-\epsilon_j\)。 | 改成权重扰动的 Lipschitz 界；只能说限制残差预算限制了错误决策的影响。 |
| Prop 3：能力税直接由一个 token KL 控制 | 必须区分状态分布、完整轨迹 KL、序列长度、奖励范围和学生拟合误差。原“特化收益项”没有定义。 | 使用完整轨迹 KL 的链式法则，或明确只讨论固定状态分布上的 surrogate。 |
| Prop 3：“随 μ 单调”“教师数无直接关系” | 精确非参数最优目标到锚点的 KL 可以单调；下游分数和神经网络训练结果没有该保证。教师数也可能通过残差幅度和总预算影响上界。 | 把可证明单调性限制在固定残差方向的精确指数倾斜路径上。 |
| Prop 4：几何聚合是一阶近似 | 实际是精确恒等式。 | 作为方法定位和实现一致性定理，不能包装为新的聚合机制。 |
| Prop 4：“硬路由没有聚合目标” | 硬路由就是权重为 one-hot 的状态条件几何聚合。 | 删除；比较重点应是权重生成机制和预算约束。 |
| Prop 5：logit 正交意味着梯度无冲突 | logits、概率、参数梯度对应不同空间和不同度量。即使参数梯度正交，也只能消除一阶交叉项，不能消除二阶损失。 | 以实际参数梯度测量冲突；给出带曲率项的局部界。 |
| Prop 5：稀疏门控方差必然乘以 ρ | 只有 oracle 稀疏支持、无漏检、无选择偏差等条件下成立。 | 明确写出漏检信号损失、误检噪声和选择偏差。 |
| 原 gate：正向对齐表示专家“有用” | 学生等于 generalist 时，任意非零 specialist 残差都有 \(e=d\)，因此全部满激活。对齐并不验证正确性。 | 将质量判断与几何新颖性分开；质量需要验证器、带标签校准集或经校准的代理。 |

还有一个容易被审稿人指出的事实：

\[
z_g+\sum_i a_i(z_i-z_g)
=(1-\sum_i a_i)z_g+\sum_i a_i z_i.
\]

因此，在相同权重、相同教师、相同温度下，“残差聚合”和“加权 logit 聚合”是同一种计算，不能作为两个不同算法做消融。真正可比较的是固定权重与自适应权重、硬路由与软权重、是否保留锚点预算、是否使用质量信号。

## 2. 统一且不依赖 logit 任意平移的定义

在固定前缀 \(h\) 上，设所有教师具有相同词表，分布严格为正：

\[
g(v)=p_g(v\mid h),\qquad p_i(v)=p_i(v\mid h),\qquad
r_i(v)=\log p_i(v)-\log g(v).
\]

设

\[
a_i\ge 0,\quad A=\sum_i a_i\le\beta_{\max}<1,
\quad s_a(v)=\sum_i a_i r_i(v),
\]

\[
q_a(v)=\frac{g(v)e^{s_a(v)}}{Z_a},
\qquad Z_a=\sum_v g(v)e^{s_a(v)}.
\]

其中 \(1-A\) 是 generalist 的**有效几何聚合权重**。把 generalist 写在公式最前面，并不代表它始终具有权重 1；若 \(A=1\)，generalist 的聚合指数已经降到 0。若要声称 always-on，需约束 \(1-A\ge\alpha_{\min}>0\)。

定义与 logit 平移无关的振幅：

\[
\operatorname{osc}(f)=\max_v f(v)-\min_v f(v).
\]

对需要内积、余弦或平方误差的分析，应先去除常数方向。例如固定参考分布 \(m\)，定义

\[
\bar r_i=r_i-\mathbb E_m r_i,\qquad
\langle u,v\rangle_m=\sum_xm(x)(u(x)-\mathbb E_m u)(v(x)-\mathbb E_m v).
\]

这等价于使用分类分布的 Fisher 矩阵 \(\operatorname{diag}(m)-mm^\top\) 对 logits 度量。若用原始 logits 的普通余弦，给任一教师 logits 加一个常数便能改变 gate，而教师概率分布完全不变。

## 3. 定理 A：精确几何池恒等式；μ 与锚权重的关系

**假设。** 状态 \(h\)、教师分布和权重均固定；\(w_j\ge0\)，\(\sum_jw_j=1\)。令

\[
q_w(v)=\frac{\prod_jp_j(v)^{w_j}}{Z_w}.
\]

**结论。** 对任意学生分布 \(p\)，有

\[
\boxed{\sum_jw_j\operatorname{KL}(p\Vert p_j)
=\operatorname{KL}(p\Vert q_w)-\log Z_w.}
\]

因此，\(q_w\) 是加权 reverse-KL 目标在概率单纯形上的唯一最优解。残差形式与几何池**完全相同**，不需要线性化。

**证明。** 由

\[
\log q_w=\sum_jw_j\log p_j-\log Z_w
\]

直接代入 KL 定义即可。严格正分布保证 KL 非负，且 \(\operatorname{KL}(p\Vert q_w)=0\) 当且仅当 \(p=q_w\)。Hölder 不等式另给出 \(Z_w\le1\)。

### A.1 额外 anchor KL 的实际作用

原目标为

\[
\operatorname{KL}(p\Vert q_a)+\mu\operatorname{KL}(p\Vert g).
\]

令

\[
\widetilde q(v)\propto q_a(v)^{1/(1+\mu)}g(v)^{\mu/(1+\mu)}.
\]

则该目标等于

\[
(1+\mu)\operatorname{KL}(p\Vert\widetilde q)+C,
\]

其中 \(C\) 与学生分布 \(p\) 无关，且

\[
\boxed{\widetilde q(v)\propto
g(v)\exp\!\left(\sum_i\frac{a_i}{1+\mu}r_i(v)\right).}
\]

因此，固定状态和权重时，额外 anchor KL 与缩小全部 specialist 权重等价，同时把 loss 整体乘以 \(1+\mu\)。后者会影响有效学习率。\(\mu\) 和总 specialist 权重高度冗余，不宜同时做无约束超参数搜索。

**适用边界。** 这是固定状态的精确恒等式。如果状态分布取决于学生，\(C(h)\) 对状态的期望可能依赖学生；不能把它从完整 trajectory objective 中随意删除。若 gate 依赖学生，也必须在该次更新中冻结/stop-gradient，才能把 \(C\) 当作对当前优化变量不变。

**实验含义。** “同权重 residual vs geometric pooling”应给出数值一致性检查，而不是独立创新对照。主要对照应包含完全相同有效权重的平级加权 baseline。

## 4. 定理 B：路由错误的预算界

**假设。** 比较同一状态上的权重 \(a,b\)，保持教师固定。令

\[
\delta(v)=\sum_i(b_i-a_i)r_i(v),\qquad D=\operatorname{osc}(\delta).
\]

**结论。**

\[
\boxed{\operatorname{KL}(q_a\Vert q_b)\le D^2/8,\qquad
\operatorname{KL}(q_b\Vert q_a)\le D^2/8,}
\]

\[
\boxed{\operatorname{TV}(q_a,q_b)\le\min\{1,D/4\},}
\]

且

\[
D\le\sum_i|b_i-a_i|\operatorname{osc}(r_i).
\]

对固定学生分布 \(p\)，还有

\[
\boxed{\left|\operatorname{KL}(p\Vert q_b)-\operatorname{KL}(p\Vert q_a)\right|\le D.}
\]

**证明。** 因为 \(q_b=q_ae^\delta/\mathbb E_{q_a}e^\delta\)，

\[
\operatorname{KL}(q_a\Vert q_b)
=\log\mathbb E_{q_a}e^\delta-\mathbb E_{q_a}\delta
\le D^2/8,
\]

最后一步为有界随机变量的 Hoeffding 引理。交换 \(a,b\) 同理。Pinsker 不等式给出 TV 结论。最后一个不等式来自

\[
\operatorname{KL}(p\Vert q_b)-\operatorname{KL}(p\Vert q_a)
=\log\mathbb E_{q_a}e^\delta-\mathbb E_p\delta,
\]

右边两个量都处于 \([\min\delta,\max\delta]\)。

这里的 Hoeffding 引理也可直接证明：设 \(X\) 的取值区间宽度为 \(D\)，\(\phi(t)=\log\mathbb E e^{tX}\)，则 \(\phi''(t)=\operatorname{Var}_{p_t}(X)\le D^2/4\)。积分得到

\[
\log\mathbb E e^X-\mathbb E X
=\int_0^1(1-t)\phi''(t)\,dt\le D^2/8.
\]

有界变量的方差上界来自 \(\operatorname{Var}(X)\le\mathbb E(X-\text{区间中点})^2\le D^2/4\)。

### B.1 可以怎样比较硬路由

若正确教师为 \(i\)，错误教师为 \(j\)，无锚点目标由 \(p_i\) 换为 \(p_j\)。若采用固定残差预算 \(\beta\)，则目标由

\[
q_i\propto g^{1-\beta}p_i^\beta
\quad\text{换成}\quad
q_j\propto g^{1-\beta}p_j^\beta.
\]

两者的扰动为 \(\delta=\beta\log(p_j/p_i)\)，所以相同形式的 KL **上界**缩小为 \(\beta^2\) 倍。这没有保证实际 KL 的比值恰好是 \(\beta^2\)，更没有保证最终任务精度更高：正确路由时的 specialist 信号也同时被缩小了。

若误路由发生概率为 \(\varepsilon\)，正确决策时两目标一致，错误时 \(D\le D_{\max}\)，则

\[
\mathbb E\operatorname{KL}(q_{\rm correct}\Vert q_{\rm noisy})
\le\varepsilon D_{\max}^2/8.
\]

不同方法的误路由率必须单独测量，不能默认相同。

**可验证预测。** 固定同一权重方向和同一注入错误，逐步提高 \(\beta\)，目标扰动上界按 \(\beta\) 或 \(\beta^2\) 增长。这是可直接检查的数学性质。下游性能曲线只是额外经验结果。

## 5. 定理 C：锚点 KL、单调性与完整轨迹能力界

### C.1 目标分布到锚点的偏离

令 \(B=\operatorname{osc}(s_a)\)。由定理 B 取一个权重向量为 0，有

\[
\boxed{\operatorname{KL}(q_a\Vert g),\operatorname{KL}(g\Vert q_a)\le B^2/8,}
\]

其中

\[
B\le\sum_i a_i\operatorname{osc}(r_i).
\]

加入原始 \(\mu\) 项后，折合目标满足

\[
\operatorname{KL}(\widetilde q\Vert g)
\le\frac{B^2}{8(1+\mu)^2}.
\]

若每个 specialist 的 \(\operatorname{osc}(r_i)\le R\)，并且始终固定总预算 \(\sum_i a_i\le\beta_{\max}\)，则该界为 \(\beta_{\max}^2R^2/8\)，不显式含教师个数。该结论需要同时固定残差幅度上界和总预算；增加教师后不能默认二者仍不变。

### C.2 什么随锚点强度单调

固定一个残差方向 \(s\)，定义

\[
q_\tau(v)=g(v)e^{\tau s(v)-\psi(\tau)},\qquad
\psi(\tau)=\log\mathbb E_ge^{\tau s},\quad\tau\ge0.
\]

则

\[
\operatorname{KL}(q_\tau\Vert g)=\tau\psi'(\tau)-\psi(\tau),
\]

\[
\boxed{\frac{d}{d\tau}\operatorname{KL}(q_\tau\Vert g)
=\tau\operatorname{Var}_{q_\tau}(s)\ge0.}
\]

令 \(\tau=1/(1+\mu)\)，得到目标分布到 generalist 的 KL 随 \(\mu\) 非增。这不蕴含任何特定 benchmark 分数单调，也不保证有限容量学生训练后的 KL 单调。

### C.3 固定状态分布下的正则化最优解

固定状态分布 \(d\) 和目标 \(q\)，假设学生类包含 \(g\)，并且得到一个 \(\xi\)-近似最优学生 \(p_*\)：

\[
\mathbb E_d[\operatorname{KL}(p_*\Vert q)+\mu\operatorname{KL}(p_*\Vert g)]
\le\mathbb E_d\operatorname{KL}(g\Vert q)+\xi.
\]

利用 KL 非负，立即得到

\[
\boxed{\mathbb E_d\operatorname{KL}(p_*\Vert g)
\le\frac{\mathbb E_d\operatorname{KL}(g\Vert q)+\xi}{\mu}.}
\]

这给原文未定义的“特化收益项”一个可检验替代：右端实际是锚点到特化目标的距离加优化误差，并非已证明的任务收益。若学生类不包含 \(g\)，必须换成可实现的 reference student，并计入其逼近误差。

### C.4 从 token KL 到序列任务能力

假设相同 prompt 分布、有限长度 \(T\)；可将提前 EOS 后的位置补成吸收状态。设 \(P_\pi,P_g\) 为完整回答的轨迹分布。KL 链式法则给出

\[
\boxed{\operatorname{KL}(P_\pi\Vert P_g)
=\sum_{t=1}^{T}\mathbb E_{h_t\sim d_{\pi,t}}
\operatorname{KL}(\pi(\cdot\mid h_t)\Vert g(\cdot\mid h_t)).}
\]

若右侧按时间平均为 \(\bar\varepsilon\)，且序列评分 \(R(y,x)\) 的范围为 \([R_{\min},R_{\max}]\)，则

\[
\boxed{|J(\pi)-J(g)|
\le (R_{\max}-R_{\min})\min\{1,\sqrt{T\bar\varepsilon/2}\}.}
\]

**证明。** 对有界函数的期望差不超过函数振幅乘以 TV，再应用 Pinsker。这里不需要另外援引 performance difference lemma。

若学生恰好实现 \(q_a\)，并且每个可达状态都满足 \(\operatorname{osc}(s_a)\le B\)，则

\[
\boxed{|J(q_a)-J(g)|
\le (R_{\max}-R_{\min})\min\{1,B\sqrt T/4\}.}
\]

如果学生只有近似拟合，并且

\[
\frac1T\sum_t\mathbb E_{d_{\pi,t}}
\operatorname{KL}(\pi(\cdot\mid h_t)\Vert q_a(\cdot\mid h_t))\le\varepsilon_{\rm fit},
\]

则在上述 **uniform** 目标振幅约束下，用轨迹 TV 的三角不等式可得

\[
\boxed{|J(\pi)-J(g)|\le (R_{\max}-R_{\min})
\min\left\{1,\sqrt{T\varepsilon_{\rm fit}/2}+B\sqrt T/4\right\}.}
\]

注意：KL 本身没有三角不等式；这里先分别控制轨迹 TV，再相加。

**适用边界。** 训练 rollout 上测得的平均 KL，不自动控制新测试 prompt 或其他策略访问的状态。若要迁移界，需要明确 coverage/concentrability 条件，或直接在目标评估分布下估计上述量。\(1/|y|\) 的随机长度归一化也不能无条件替换轨迹 KL 的求和；可使用统一上限 \(T\) 得到保守界。

## 6. 定理 D：真正可用于“优于硬路由”的条件性风险结论

### D.1 一般偏差—方差分解

在固定状态、固定中心化坐标下，假设

\[
z_g=c+\epsilon_g,\quad z_i=c+s_i+\epsilon_i,\quad
z^*=c+s^*.
\]

假设噪声零均值，相互独立，权重与这些噪声独立；令 \(V_j=\mathbb E\|\epsilon_j\|^2\)。则

\[
\boxed{\mathbb E\|z_a-z^*\|^2
=\left\|\sum_i a_is_i-s^*\right\|^2
 +(1-A)^2V_g+\sum_i a_i^2V_i.}
\]

**证明。** 将误差拆成确定性偏差与均值为零的噪声；展开平方，独立性使交叉项期望为零。

若噪声相关，需加入

\[
2(1-A)\sum_i a_i\mathbb E\langle\epsilon_g,\epsilon_i\rangle
+2\sum_{i<j}a_ia_j\mathbb E\langle\epsilon_i,\epsilon_j\rangle.
\]

同源教师并不保证噪声独立，反而常会共享错误。若 gate 由同一组带噪 logits 计算，权重与噪声相关，需额外分析选择偏差，不能直接使用本式。

### D.2 单 specialist：一个严格改善条件

考虑当前状态确实需要 specialist \(i\)，即 \(s^*=s_i=s\)。令

\[
z_\beta=(1-\beta)z_g+\beta z_i,\quad
A_0=\|s\|^2+V_g.
\]

则

\[
\mathcal R(\beta)=(1-\beta)^2A_0+\beta^2V_i,
\qquad\mathcal R_{\rm hard}=V_i.
\]

无约束最优权重为

\[
\boxed{\beta^*=\frac{A_0}{A_0+V_i},\qquad
\mathcal R(\beta^*)=\frac{A_0V_i}{A_0+V_i}.}
\]

在 \(V_i>0\)、\(A_0\) 有限时，最优收缩风险严格小于硬路由风险。对于任意给定 \(\beta<1\)，改善的充要条件为

\[
\boxed{(1-\beta)A_0<(1+\beta)V_i.}
\]

**含义。** specialist 噪声越大，适当锚点收缩越有价值；specialist 的真实增量越大，所需 specialist 权重越接近 1。若强制锚权重过高，就可能不满足改善条件。若 specialist 完全正确且无噪声，\(V_i=0\)，硬路由已经达到零风险，任何有偏收缩都无法严格改善。

**不能直接写成准确率定理。** 这里的风险是中心化 logit 的均方误差。若要转为分类准确率，需要额外的 margin 假设；若要转为序列奖励，还要处理状态分布和长时信用分配。有限扰动下可用 softmax 的平滑性建立 KL 上界，但不能把某个风险上界更小直接等同于真实 KL 或准确率更好。

### D.2a 允许 generalist 与 specialist 共享错误的精确条件

同源教师更合理的情形是噪声相关。令

\[
C=\mathbb E\langle\epsilon_g,\epsilon_i\rangle,
\qquad D=A_0+V_i-2C
=\|s\|^2+\mathbb E\|\epsilon_g-\epsilon_i\|^2\ge0.
\]

此时

\[
\mathcal R(\beta)=(1-\beta)^2A_0+\beta^2V_i+2\beta(1-\beta)C,
\]

\[
\boxed{\mathcal R(\beta)-\mathcal R_{\rm hard}
=(1-\beta)\left[(1-\beta)D-2(V_i-C)\right].}
\]

所以对于 \(\beta<1\)，严格改善的充要条件为

\[
\boxed{(1-\beta)D<2(V_i-C).}
\]

当 \(D>0\)，限制到 \([0,1]\) 的最优权重为

\[
\beta^*=\operatorname{clip}_{[0,1]}\left(\frac{A_0-C}{D}\right).
\]

若未裁剪解位于内部，其最优风险为 \((A_0V_i-C^2)/D\)。若还要求 \(\beta\le\beta_{\max}<1\)，应进一步把最优解投影到该允许区间，不能声称总能达到无约束改善。如果 \(D=0\)，则信号为零且两者噪声几乎处处相同，所有权重的风险相同。

特别地，若两者完全共享同一误差，\(\epsilon_g=\epsilon_i\)，则 \(C=V_i\)，任何收缩都不会减少这份共同误差；若 \(s\ne0\)，反而引入额外偏差。这比“共同分量越大，收益越大”更准确地体现了相关误差的影响。

### D.3 两个 specialist 的组合偏差

如果真实目标是 \(s^*=s_p+s_r\)，而强制 \(a_p+a_r\le1\)，一般不能同时取 \(a_p=a_r=1\)，因此可能存在不可避免的收缩偏差。不能因为感知与推理残差正交，就忽略这一问题。

可考虑两种可验证情况：一是大部分状态只需要一个主专家，适合稀疏条件选择；二是两个专家的增量有明显冗余，凸组合足以表达目标。若两者都不成立，需允许更强的组合、训练复合专家，或者接受保守锚点带来的性能权衡。允许权重和超过 1 会使 generalist 指数为负，不能继续套用凸组合方差结论。

## 7. 定理 E：稀疏门控何时减少风险

令真实残差信号仅在事件 \(A\) 上非零，\(P(A)=\rho\)。观察

\[
r=s+\epsilon,\quad s=0\text{ on }A^c,
\quad\mathbb E[\epsilon\mid h]=0,\quad
\mathbb E\|\epsilon\|^2=V.
\]

令二值 gate 为 \(G\in\{0,1\}\)。假设给定状态与信号后，gate 的决策不选择性依赖同一次噪声，且条件噪声二阶矩恒为 \(V\)。定义

\[
f_n=P(G=0\mid A),\qquad f_p=P(G=1\mid A^c).
\]

则

\[
\boxed{\mathbb E\|Gr-s\|^2
=\mathbb E[\mathbf1_{A,G=0}\|s\|^2]
+V[\rho(1-f_n)+(1-\rho)f_p].}
\]

**证明。** 展开 \(Gr-s=(G-1)s+G\epsilon\)，在条件独立和零均值下交叉项为零。

始终激活的风险为 \(V\)。因此 gate 改善风险的充要条件为

\[
\boxed{\mathbb E[\mathbf1_{A,G=0}\|s\|^2]
<V\,[\rho f_n+(1-\rho)(1-f_p)].}
\]

左边是漏掉必要信号的代价，右边是关闭 gate 所节省的噪声。只有完美支持识别 \(f_n=f_p=0\) 时，风险才恰好从 \(V\) 降为 \(\rho V\)。若仅有 \(\|s\|^2\le M^2\)，可用 \(\rho f_nM^2\) 给左边做上界。

**重要限制。** 若 residual 是 \(z_i-z_g\)，且 generalist 本身也带噪，则 residual 噪声与锚点噪声相关。本定理描述残差估计的风险，不能直接当成完整混合 logits 的风险；完整风险应回到定理 D。若 gate 使用当前 noisy residual 自己决定开关，则必须估计选择偏差或改为交叉拟合。

**实验应测什么。** 除激活率外，至少报告 gate 的漏检、误检、漏检信号强度、校准误差，以及相同 token 预算下的最终任务收益。仅报告稀疏率低或热图好看不足以验证本定理。

## 8. 原 gate 的反例与可辩护替代

原式是

\[
e_i=z_i-z_g,\quad d_i=z_i-z_\theta,\quad
g_i=\mathbf1[\langle e_i,d_i\rangle>\delta]\max\{\cos(e_i,d_i),0\}.
\]

当学生等于 generalist 时，\(e_i=d_i\)。只要 \(\|e_i\|^2>\delta\)，每个 specialist 的 gate 都是 1，包括纯粹犯错或只改变风格的 specialist。由此可见，该 gate 测的是几何关系，不是条件质量。

还存在不可识别性：两个世界可以具有完全相同的教师 logits 和学生 logits，但正确答案不同。在这两个世界中，任何只读取 logits 的确定性 gate 都会作出相同选择，因此不可能在两个世界中都保证选中正确专家。质量保证必须引入标签、验证器、环境反馈或明确可检验的质量假设。

建议将两个判断分开：

1. **资格/质量信号**：估计在该状态或状态类型上，使用 specialist 相对 generalist 的条件收益，例如 \(\widehat U_i(h)\)。可来自带验证器的候选续写、人工/规则监督，或由这些数据校准的小 router。完整逐 token 候选续写成本很高，可离线采样校准，再在线预测。
2. **迁移与预算信号**：只在有正质量证据的 specialist 中，结合学生能力缺口、残差幅度、冲突和成本分配权重。Fisher 对齐可以作为辅助特征，但不能独自充当质量证明。

一个可实现、便于理论描述的决策为

\[
U_i^{\rm LCB}(h)=\widehat U_i(h)-\kappa\widehat\sigma_i(h)-c_i,
\]

保留 \(U_i^{\rm LCB}>0\) 的候选，并对它们做 softmax 或 top-1/top-2 归一化得到 \(v_i(h)\)，\(\sum_i v_i=1\)。没有正候选时设 \(a_i=0\)，退回 generalist。

LCB 真具有置信下界意义，需要具体的独立采样、样本量和统计模型；否则应称为“保守评分”，不要滥用概率保证。

例如，若每个 reward 在 \([0,1]\)，同一前缀下为专家 \(i\) 收集 \(n_i\) 对独立续写，令每对的 reward 差为 \(D_{i,j}\in[-1,1]\)，则 \(\widehat U_i=n_i^{-1}\sum_jD_{i,j}\)，取

\[
b_i=\sqrt{\frac{2\log(2K/\delta)}{n_i}}
\]

可由 Hoeffding 不等式和对 \(K\) 个专家的 union bound 保证：以至少 \(1-\delta\) 的概率，所有 \(|\widehat U_i-U_i|\le b_i\)。这里要求每个固定专家的样本对在采样维度上独立；不同专家之间共享 generalist 样本不会破坏 union bound。如果使用离线训练的 router 对新前缀输出 \(\widehat U_i\)，则此 Monte Carlo 区间不再自动适用，必须另外校准预测误差。

### 8.0 必须承认的效用代理断层

例如，可以在同一个学生前缀 \(h\) 后让教师 \(i\) 与 generalist 分别续写，并通过验证器估计

\[
U_i(h)=\mathbb E[R(h,Y_i)\mid Y_i\sim P_i(\cdot\mid h)]
-\mathbb E[R(h,Y_g)\mid Y_g\sim P_g(\cdot\mid h)].
\]

这个量回答“若后续整段由该教师接管，它比 generalist 好多少”。它不等于：该教师在当前 token 的因果贡献、student 一步更新的收益、或几何聚合 \(q_a\) 的序列收益。由 \(U_i>0\) 不能推出 \(J(q_a)>J(g)\)：几何池会重新归一化各 token 概率，且后续状态分布变化，任务奖励通常还对整条推理链高度非线性。

\(\sum_iw_iU_i\) 精确对应的是“在该前缀抽取一个教师索引，并让该教师完成剩余整段”的随机策略相对 generalist 的期望收益；它不精确对应 token 级切换教师的策略，更不对应逐 token 几何池。

因此，应把 continuation verifier advantage 称为**目标质量/路由效用的可校准代理**，用独立实验检查它和最终混合目标或学生收益之间的关系。若希望直接检验这道断层，可以对少量前缀实际从候选聚合目标续写，并比较 verifier 收益；这会增加教师调用成本，但能提供重要机制证据。

质量 proxy 还有数据要求：任务验证器必须能正确处理已经包含错误的学生前缀、多个合理答案与长链推理，且 prefix selection 和标签分配不能泄漏测试答案。

### 8.1 用一个可解释预算决定 shared-expert 权重

固定路由方向 \(s_v=\sum_iv_ir_i\)，令

\[
q_\beta(v)\propto g(v)e^{\beta s_v(v)}.
\]

选择

\[
\boxed{\beta(h)=\max\{b\in[0,\beta_{\max}]:
\operatorname{KL}(q_b\Vert g)\le\varepsilon(h)\},}
\]

再设

\[
a_i(h)=\beta(h)v_i(h),\qquad w_g(h)=1-\beta(h).
\]

由定理 C.2，约束中的 KL 随 \(b\) 单调，因此可以做稳定的一维二分；教师前向完成后，只需复用缓存的分布。若预算不紧，取 \(\beta_{\max}\)；若残差大，自动降低其权重。保守解析替代为

\[
\beta(h)\le \min\left\{\beta_{\max},
\frac{\sqrt{8\varepsilon(h)}}{\operatorname{osc}(s_v)}\right\}.
\]

若残差方向恒为零，直接保持 generalist。实际 full-vocabulary KL 通常比振幅界更紧；若只计算 top-k，必须说明 tail 如何处理，否则不能声称有完整分布 KL 保证。

这是“每轮固定预算参数、每个状态自适应实际权重”的明确设计：在同一轮 rollout 内保持 router 参数、温度、\(\beta_{\max}\) 和 KL 预算规则固定，但允许 \(v_i(h_t),\beta(h_t)\) 随前缀变化。每次学生更新时对已经得到的教师目标和 gate 做 stop-gradient。不同训练轮再依据验证集与预算误差调整全局参数。

## 9. 正交性究竟能保证什么

设通用能力 surrogate 为 \(L_g(\theta)\)，其梯度 \(u=\nabla L_g(\theta)\)，残差更新方向为 \(v\)。若 \(L_g\) 是 \(L\)-smooth，采用

\[
\theta'=\theta-\eta(u+v),
\]

则

\[
\boxed{L_g(\theta')\le L_g(\theta)
-\eta\|u\|^2-\eta\langle u,v\rangle
+\frac{L\eta^2}{2}\|u+v\|^2.}
\]

即使 \(u\perp v\)，仍有 \(L\eta^2\|v\|^2/2\) 形式的二阶代价。在 \(u=0\) 的 generalist 局部最优点，任意非零残差更新都可能增加通用损失；例子是 \(L_g(\theta)=\|\theta\|^2/2\)，从 \(\theta=0\) 沿任意非零方向更新都会产生正损失。

此外，\(r_p\perp r_r\) 在普通 logit 欧氏空间中，并不意味着参数梯度正交：参数梯度还经过 softmax 曲率与学生 Jacobian。应测真实参数梯度余弦，或清楚说明所用的 Fisher/Jacobian 诱导度量。视觉/语言梯度的近正交证据，也不能自动替代“感知/推理两个 specialist 残差近正交”这一不同假设。

## 10. On-policy rollout 与真实 on-policy 梯度必须分开

令

\[
\ell_\theta(h)=\operatorname{KL}(\pi_\theta(\cdot\mid h)\Vert q(\cdot\mid h)).
\]

在固定状态、固定教师目标上，

\[
\nabla_\theta\ell_\theta(h)
=\mathbb E_{a\sim\pi_\theta}
[(\log\pi_\theta(a\mid h)-\log q(a\mid h))\nabla_\theta\log\pi_\theta(a\mid h)].
\]

这可以使用完整词表计算，或构造正确的 score-function 估计。不能只对采样到的 \(\log\pi_\theta(a)-\log q(a)\) 做普通反向传播并期望自动得到该梯度；离散采样的概率权重也需要处理。

若完整目标是

\[
F(\theta)=\mathbb E_{\tau\sim P_\theta}\sum_t\ell_\theta(h_t),
\]

则其梯度包含两部分：

\[
\boxed{\nabla F(\theta)=
\mathbb E_{P_\theta}\sum_t\nabla\ell_\theta(h_t)
+\mathbb E_{P_\theta}\left[
\left(\sum_t\ell_\theta(h_t)\right)
\nabla\log P_\theta(\tau)\right].}
\]

第二项是访问状态分布变化产生的项；可用因果性化简为各动作对其后续状态损失的贡献。若目标/gate 也依赖学生而不冻结，第一项里还会出现教师目标的导数。

实际可采用清晰的外内循环：

\[
\tau\sim\pi_{\theta_k},\quad
\min_\theta\mathbb E_{h\sim d_{\theta_k}}
\operatorname{KL}(\pi_\theta(\cdot\mid h)\Vert q_k(\cdot\mid h)),
\]

其中 \(d_{\theta_k}\) 和 \(q_k\) 在当前更新内冻结。这是使用 on-policy 数据的固定状态 surrogate；本报告定理 A、C.3 等适用于这个层面。不能据此宣称已经证明完整非凸 on-policy 目标的收敛性或全局奖励最优性。

关于访问分布、policy gradient 与 performance difference 的标准处理，可参考原始研究论文 [Agarwal et al., On the Theory of Policy Gradient Methods](https://jmlr.csail.mit.edu/papers/volume22/19-736/19-736.pdf)。本报告选择直接使用轨迹 KL 链式法则，使有限长度生成任务的假设与长度依赖更明确。

## 11. 可选：router regret 能提供什么，不能提供什么

### 11.1 LCB router：同一可行集上的单步线性代理界

把 generalist 的相对效用定义为 \(U_g=0\)。设预算可行集为

\[
\mathcal W=\{w\ge0:\sum_jw_j=1,\ w_g\ge\alpha_{\min}\},
\]

并设在一个同时置信事件上，所有候选满足

\[
|\widehat U_i-U_i|\le b_i,\qquad L_i=\widehat U_i-b_i,
\quad b_g=0.
\]

令 \(w^*\) 最大化 \(\langle w,U\rangle\)，\(\widehat w\) 最大化 \(\langle w,L\rangle\)，两者都在同一 \(\mathcal W\) 内，则

\[
\boxed{\langle w^*,U\rangle-\langle\widehat w,U\rangle
\le 2\sum_iw_i^*b_i
\le2(1-\alpha_{\min})\max_i b_i.}
\]

**证明。** 将差值拆为

\[
\langle w^*,U-L\rangle
+\langle w^*-\widehat w,L\rangle
+\langle\widehat w,L-U\rangle.
\]

第二项由最优性非正，第三项由 \(L\le U\) 非正，第一项由 \(U-L\le2b\) 得到上界。若路由器只达到 \(\xi\)-近似最优，再加 \(\xi\)。softmax 的熵正则会改变目标，必须计入相应近似/正则化差值，不能直接当作精确最大化线性 LCB。

**比较对象。** 这是相同状态上、相同预算集合内的最佳线性代理分配。若 \(\alpha_{\min}>0\)，纯 specialist hard routing 不属于 \(\mathcal W\)，因此该定理完全不能推出“相对于原始 MOPD 的 regret”。另外，上述线性效用是第 8.0 节的整段教师随机选择代理，不是几何池真实效用。

### 11.2 多轮 full-information 代理 regret

若希望加入在线权重学习理论，最稳妥的是先限定一个 full-information 的代理问题：在每轮 \(t\)，对所有专家都能获得 \(\ell_{t,i}\in[0,1]\)，按指数权重

\[
w_{t,i}\propto\exp\left(-\eta\sum_{s<t}\ell_{s,i}\right)
\]

更新，则标准 Hedge 证明给出

\[
\sum_{t=1}^N\langle w_t,\ell_t\rangle
-\min_i\sum_{t=1}^N\ell_{t,i}
\le\frac{\log K}{\eta}+\frac{\eta N}{8}.
\]

取 \(\eta=\sqrt{8\log K/N}\)，右端为 \(\sqrt{N\log K/2}\)。

**证明要点。** 设 \(W_t=\sum_i\exp(-\eta\sum_{s<t}\ell_{s,i})\)。Hoeffding 引理给出

\[
\log(W_{t+1}/W_t)\le-\eta\langle w_t,\ell_t\rangle+\eta^2/8.
\]

求和并用 \(W_{N+1}\ge\exp(-\eta\min_i\sum_t\ell_{t,i})\)、\(W_1=K\)，即可整理得到结论。

这个保证的比较对象是**同一已发生状态序列上的最佳固定专家**，不是另一个策略生成的完整轨迹。若只查询部分专家，需要 bandit estimator，并计入方差/探索成本；若以 student-teacher KL 为代理，小 KL 可能只是老师和学生都错，不能解释为质量保证。

若有可靠监督 token 标签，使用有界或经过明确概率下界处理的 log loss，可利用几何池的

\[
-\log q_w(y)=\sum_iw_i[-\log p_i(y)]+\log Z_w
\le\sum_iw_i[-\log p_i(y)]
\]

把线性代理 regret 转为目标分布的累计 log-loss 上界。但这种 token 监督在任意错误学生前缀上未必有定义；不能直接把 gold answer 的第 \(t\) 个 token 当成任意 rollout 前缀的正确下一 token。

该部分适合作为扩展，不建议为增加“理论数量”强行塞入主方法。

指数权重学习的经典原始来源是 [Freund and Schapire, A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X)。上述具体常数使用本文给出的有界 loss 指数权重证明；不应把经典 Hedge 结论包装为本方法原创理论。

## 12. 为什么不存在无条件优于 MOPD 的证明

考虑二分类任务，正确答案为 1。令 generalist 对正确答案的概率为 \(g(1)=0.5\)，正确 specialist 为 \(p_i(1)=0.99\)，朴素 MOPD 的路由完全正确。若学生可以精确拟合目标，则朴素 MOPD 的期望正确率为 \(0.99\)。

任何严格保留锚点的几何组合 \(0\le\beta<1\) 都满足

\[
q_\beta(1)=\operatorname{sigmoid}(\beta\operatorname{logit}(0.99))<0.99.
\]

因此，强制锚点的方案在这个例子上严格更差。若把奖励定义为抽样回答是否正确，这就是直接的下游奖励反例。

能够成立的论文主张应是：

> 在专家噪声、路由不确定性和稀疏能力增量满足明确条件时，预算受控的 generalist 锚点可以降低目标估计风险和路由扰动；在完整轨迹层面，锚点 KL 约束可控制最坏情形的能力偏离。实际的多能力收益由实验检验。

建议正文只保留三个主结论：**精确聚合与权重解释、预算约束下的扰动/锚点界、带漏检误检项的条件风险改善**。完整轨迹说明、正交反例和门控偏差放附录，但必须在正文避免作出它们所否定的强结论。

## 13. 对实验与论文措辞的具体修正

- 将“证明方法比 MOPD 更好”改为“给出优于硬路由目标估计风险的充分/充要条件，并用实验测量这些条件是否近似满足”。
- 将“共模抑制”改为“以 generalist 为参考的受控收缩”，除非能够定义并独立测出真正被消除的共同误差分量。
- 将“通用能力税上界”中的 reference 明确为教师 generalist、generalist-only 学生或初始学生；三者不相同。若实验比较的是 generalist-only 学生而理论比较教师，需报告两者的拟合差异，或把理论 reference 改成可实际部署的 baseline student。
- 将“残差 vs 几何聚合”消融换成“固定权重 vs 自适应权重”，并保留同权重精确一致性检查。
- 将“正交性无冲突”改为“参数梯度的一阶交叉项较小”，同时报告步长、残差范数和真实通用损失变化。
- 对 gate 验证至少报告效用校准、误选率、漏选率；单纯 teacher disagreement、低熵或正向余弦不是专长证据。
- “何时增加专家”可用候选新专家相对当前池的留出集边际效用衡量，并计入查询成本、方差与新冲突。专家数量本身不能替代能力覆盖证据。
- 所有“已验证”“一致收益”“首次”措辞应等待真实实验和相关工作核验后再写。原稿列出的许多新近论文标识需要单独核验，本报告不替其真实性背书。

## 14. 候选路线 D：用算术混合与公共续写获得任务收益改进保证

> 本节是另一条方法路线，替换了前文的几何聚合目标与整段 specialist 续写代理。它的理论更直接，但条件优势估计成本更高。它不推翻前面的反例：这里新增了可验证优势与可行性假设，并没有声称任意 anchored mixture 都更好。

### 14.1 与其他路线的关键区别

路线 C 的效用是“从学生前缀出发，后续整段由某个 specialist 生成”的 reward；它和当前 token 的组合目标之间有代理断层。路线 D 改为：**当前只让候选 expert 选择一个 token，剩余整段始终交给同一个 reference policy**。不同候选之间仅当前动作分布不同，因此能够用 performance difference identity 把局部优势与最终策略的真实任务收益连接起来。

同时，将目标由 logit／几何池改成**概率分布的算术混合**。训练仍可使用 reverse-KL \(\operatorname{KL}(\pi_\theta\Vert q)\)；“reverse-KL 作为学生损失”并不要求教师目标必须是几何池。需要区别的是：最小化各教师 reverse-KL 的加权和，才会得到几何池；本路线是先显式构造算术目标 \(q\)，再蒸馏该目标。

### 14.2 有限长度设定与 common-continuation advantage

令回答最长为 \(T\)，提前结束的序列使用吸收 EOS 补齐。状态 \(h_t=(x,y_{<t},t)\) 包含完整前缀；prompt 分布固定；最终序列 reward 在 \([R_{\min},R_{\max}]\)，记范围 \(R_{\rm span}=R_{\max}-R_{\min}\)。所有策略的词表和环境转移相同。

选择固定 reference policy \(g\)。定义

\[
Q_t^g(h,a)=\mathbb E[R(x,Y)\mid h_t=h,\ y_t=a,
\text{所有剩余 token 由 }g\text{ 生成}],
\]

\[
V_t^g(h)=\mathbb E_{a\sim g(\cdot\mid h)}Q_t^g(h,a),
\]

\[
A_{i,t}^g(h)=\mathbb E_{a\sim p_i(\cdot\mid h)}Q_t^g(h,a)-V_t^g(h).
\]

因此，\(A_{i,t}^g\) 衡量 specialist 的**下一 token 分布**在统一 generalist 续写下的优势；它不是 \(J(p_i\mid h)-J(g\mid h)\)。对 generalist 自己，有 \(A_{g,t}^g(h)=0\)。时间下标也可吸收入状态后省略。

定义算术混合策略

\[
\boxed{q(\cdot\mid h)=(1-\beta(h))g(\cdot\mid h)
+\beta(h)\sum_i\alpha_i(h)p_i(\cdot\mid h),}
\]

其中 \(0\le\beta(h)\le1-\alpha_{\min}\)，\(\alpha_i(h)\ge0\)，有活跃专家时 \(\sum_i\alpha_i(h)=1\)；没有活跃专家时取 \(\beta(h)=0\)。这样 generalist 的实际概率混合权重至少是 \(\alpha_{\min}\)。

### 14.3 定理 D1：相对 generalist 的精确收益差与安全改进

设 \(d_{q,t}\) 为 \(q\) 在时间 \(t\) 的状态访问分布，则

\[
\boxed{J(q)-J(g)
=\sum_{t=1}^{T}\mathbb E_{h\sim d_{q,t}}
\left[\beta(h)\sum_i\alpha_i(h)A_{i,t}^g(h)\right].}
\]

因此，只要在 \(q\) 的所有可达状态上，活跃 expert 都满足 \(A_{i,t}^g(h)\ge0\)，便有

\[
\boxed{J(q)\ge J(g).}
\]

“每个活跃 expert 非负”是充分条件；更一般地，只要求其加权和 \(\sum_i\alpha_iA_i^g\ge0\) 即可。若该加权优势在一组具有正访问概率的状态上严格为正，且 \(\beta>0\)，则 \(J(q)>J(g)\)。

**证明。** 对任意策略 \(q\)，有限长度 performance difference identity 为

\[
J(q)-J(g)=\sum_t\mathbb E_{h\sim d_{q,t},a\sim q}
[Q_t^g(h,a)-V_t^g(h)].
\]

可直接用望远镜求和证明：给定当前动作后，下一状态的 reference value 的条件期望等于 \(Q_t^g\)，终止状态 value 等于已确定的最终 reward，因此 \(\sum_t[V_{t+1}^g(h_{t+1})-V_t^g(h_t)]\) 的期望为 \(J(q)-J(g)\)。把算术混合 \(q\) 代入对当前动作的期望，generalist 项为零，剩余项恰为 \(\beta\sum_i\alpha_iA_i^g\)。非负性立即给出结论。

这里即使每次局部 \(Q_g\) 都假设“后续交给 \(g\)”，最终目标 \(q\) 仍然可以在每一个 token 再次选择 specialist；performance difference identity 已精确计入这种多次决策，无需假设目标真的只偏离 reference 一步。关键是所有状态上的共同 reference 和 \(q\) 自己的访问分布。

**为什么几何池没有这一结论。** 几何池下，\(\mathbb E_{a\sim q}Q_g\) 一般不是各 expert 的 \(\mathbb E_{p_i}Q_g\) 的加权和，所以以上关键等式不成立。若想保留几何池，需要直接验证它自己的 \(\mathbb E_qQ_g-V_g\ge0\)，不能仅凭每个 teacher 的优势为正。

### 14.4 LCB 证书、可达状态覆盖与正 margin

假设在同时置信事件 \(\mathcal E\) 上，对所有需要保证的状态–expert 对都有

\[
A_i^g(h)\ge L_i(h).
\]

只允许 \(L_i(h)\ge0\) 的 expert 激活，并令

\[
c(h)=\beta(h)\sum_i\alpha_i(h)L_i(h)\ge0,
\qquad
\Gamma=\sum_t\mathbb E_{h\sim d_{q,t}}c(h).
\]

则在事件 \(\mathcal E\) 上，

\[
\boxed{J(q)-J(g)\ge\Gamma\ge0.}
\]

若每个激活的 expert 都要求 \(L_i(h)\ge m(h)>0\)，则还可用 \(\sum_t\mathbb E_{d_q}\beta(h)m(h)\) 下界 \(\Gamma\)。仅有非负优势给出不退化；严格改善需要正 margin 和实际访问概率。

一种具体估计方法是在同一前缀 \(h\) 上做独立成对实验：一侧采样 \(a\sim p_i\)，再从 \(h+a\) 用 \(g\) 续写；另一侧从 \(h\) 全程用 \(g\) 续写。记 reward 差为 \(D_j\in[-R_{\rm span},R_{\rm span}]\)，则其期望恰为 \(A_i^g(h)\)。若总共要同时覆盖 \(M\) 个已确定的状态–expert 对，每对独立采样 \(n_{h,i}\) 次，可取

\[
\widehat A_i^g(h)=\frac1{n_{h,i}}\sum_jD_j,
\quad
b_i(h)=R_{\rm span}\sqrt{\frac{2\log(2M/\delta)}{n_{h,i}}},
\quad
L_i(h)=\widehat A_i^g(h)-b_i(h).
\]

Hoeffding 与 union bound 给出 \(P(\mathcal E)\ge1-\delta\)。

**覆盖条件不可省略。** 有限训练前缀上的同时区间，不自动覆盖目标策略的所有可达前缀；冻结一个从样本学出来的 router，也不自动满足全状态安全条件。主定理要么明确假设可达状态全覆盖或有效的统一泛化证书，要么只对已覆盖分布给出局部保证，并额外计入未覆盖状态的负优势。若逐次在线查询，可以分配总和不超过 \(\delta\) 的置信失败预算处理自适应查询，但仅对单条已采样轨迹成立的事件仍不足以直接保证完整策略期望；需要策略在所有会被调用的状态上都执行同样的有效认证规则，或另做分布层面的误差分析。

一个明确的放宽是：若真实混合优势在未认证状态上最坏不低于 \(-\nu_t(h)\)，则

\[
J(q)-J(g)\ge\Gamma_{\rm certified}
-\sum_t\mathbb E_{d_{q,t}}[\mathbf1_{\rm uncertified}(h)\nu_t(h)].
\]

由于单个 reference advantage 的绝对值不超过 \(R_{\rm span}\)，无额外信息时可保守取 \(\nu_t(h)=\beta(h)R_{\rm span}\)。该修正使“认证覆盖率”成为必须报告的实验量，而不是隐含的全局假设。

### 14.5 定理 D2：加入 student 蒸馏误差后的收益保证

设最终学生为 \(\pi\)，最终目标策略 \(q\) 固定。如果

\[
\varepsilon_{\rm fit}
=\frac1T\sum_t\mathbb E_{h\sim d_{\pi,t}}
\operatorname{KL}(\pi(\cdot\mid h)\Vert q(\cdot\mid h)),
\]

则轨迹 KL 链式法则与 Pinsker 给出

\[
|J(\pi)-J(q)|
\le R_{\rm span}\min\{1,\sqrt{T\varepsilon_{\rm fit}/2}\}.
\]

与 D1 的 margin 结合，在同一置信事件上得到

\[
\boxed{J(\pi)-J(g)
\ge\Gamma-R_{\rm span}\min\{1,\sqrt{T\varepsilon_{\rm fit}/2}\}.}
\]

所以，要把“目标策略非退化”升级成“学生严格超过 generalist”，需要 \(\Gamma\) 大于拟合误差惩罚；仅有 \(\Gamma=0\) 不够。训练 minibatch 上的小 KL 也不能直接当作这个式子中的测试分布、学生访问状态下的真实平均 KL。

此结论比较的是固定评估 reward；若同时要求感知、推理、通用三个 reward 全都不退化，需要分别定义三个 \(Q\) 和 advantage，并保证每个 reward 的相应局部优势非负。对一个标量综合 reward 的改进，不自动保证所有分项指标改进。

### 14.6 定理 D3：相对朴素 MOPD baseline 的版本及其可行性

如果真正希望与朴素 MOPD 比较，先明确一个**固定 baseline policy** \(b\)：它可以是朴素 MOPD 的固定教师目标 \(q_{\rm M}\)，也可以是训练完成的 baseline 学生 \(\pi_{\rm M}\)。二者是不同的分布，不能混称。随后将 common continuation reference 从 \(g\) 换成 \(b\)：

\[
Q_t^b(h,a)=\mathbb E[R\mid h,a,\text{之后始终跟随 }b],
\qquad
A_j^b(h)=\mathbb E_{a\sim p_j}Q_t^b(h,a)-V_t^b(h),
\]

其中候选集合包括 \(p_g=g\) 与 specialists。取

\[
q(\cdot\mid h)=\sum_jw_j(h)p_j(\cdot\mid h),
\quad w(h)\in\mathcal W,
\quad\mathcal W=\{w\ge0:\sum_jw_j=1,w_g\ge\alpha_{\min}\}.
\]

如果在所有 \(q\)-可达状态上存在并选取满足

\[
\boxed{\sum_jw_j(h)A_j^b(h)\ge0}
\]

的可行权重，则相同 PDL 推导给出

\[
\boxed{J(q)\ge J(b).}
\]

正 margin、LCB 认证与 student 拟合误差的处理完全类似。这里应对全部候选的 \(A_j^b\) 估计 LCB，**包括 generalist**；相对 \(b\)，generalist 的优势不再自动为 0。

**必须增加的可行性假设。** 不能默认上述安全权重总存在。当 \(A_g^b<0\) 时，退回 generalist 本身就是负优势动作。仅有 shared floor 约束时，每个状态上的最优可行线性优势为

\[
\max_{w\in\mathcal W}\sum_jw_jA_j^b
=\max\left\{A_g^b,
\alpha_{\min}A_g^b+(1-\alpha_{\min})\max_i A_i^b\right\}.
\]

因而，当 \(A_g^b<0\)、\(0<\alpha_{\min}<1\) 时，非退化可行性要求

\[
\max_i A_i^b\ge\frac{\alpha_{\min}}{1-\alpha_{\min}}(-A_g^b).
\]

这就是 specialist 必须提供足够正优势以抵消强制锚点代价的条件。如果 baseline \(b\) 已经在当前状态选择了最好的动作分布，所有其他候选优势都非正，而 generalist 严格为负，则没有任何满足严格 shared floor 的非退化混合。即使将 \(b\) 自己作为一个额外候选，它的优势也只是 0，不能抵消负的 generalist 强制权重。

可以让 \(b\) 本身作为失败时的 fallback 来恢复“至少与 baseline 一样”的可行性，但当 \(b\) 不含 generalist 时，这会放弃 every-state shared floor；不能同时保留两个互相冲突的保证。若安全可行集为空，应如实报告不可认证或放松 floor，不能默认回退 \(g\) 是安全的。

### 14.7 教师目标比较与最终学生比较的误差项

如果 D3 的 reference 是已经训练好的 baseline 学生 \(b=\pi_{\rm M}\)，那么 D2 直接给出

\[
J(\pi)-J(\pi_{\rm M})
\ge\Gamma_b-R_{\rm span}\sqrt{T\varepsilon_{\rm fit}/2},
\]

其中同样可对 TV 上界截断到 1。条件是用该 baseline 学生构造 \(Q_b\)，且存在经过认证的 shared-floor 可行权重。

如果 reference 是固定教师目标 \(b=q_{\rm M}\)，而实验比较的 baseline 是其蒸馏学生 \(\pi_{\rm M}\)，定义其在自身访问状态下的拟合误差

\[
\varepsilon_{\rm M}=\frac1T\sum_t\mathbb E_{d_{\pi_{\rm M},t}}
\operatorname{KL}(\pi_{\rm M}(\cdot\mid h)\Vert q_{\rm M}(\cdot\mid h)).
\]

则只能给出

\[
\boxed{J(\pi)-J(\pi_{\rm M})
\ge\Gamma_b-R_{\rm span}
\left[\sqrt{T\varepsilon_{\rm fit}/2}
+\sqrt{T\varepsilon_{\rm M}/2}\right],}
\]

其中每个 TV 贡献可分别截断到 1。它清楚显示：比较两个最终训练学生，需要处理双方的拟合误差；不能把“相对于 naive teacher target 的保证”直接写成“最终优于 naive MOPD student”。

### 14.8 路线 D 的价值与成本

这条路线给出了与真实 verifier reward 直接相连的条件改进定理，适合“理论优先”的论文版本。代价是它改变了聚合方式，且需要昂贵的 one-token intervention + common-continuation 估计；在长序列和大词表下，优势信号可能弱、方差大、覆盖困难。可以先在短回答、离散可验证决策或小规模受控多模态环境验证，再尝试用校准 router 近似，但近似后必须显式计入估计、覆盖和蒸馏误差。

正确的论文措辞是：“在共同参考续写下的非负局部优势、可达状态覆盖和足够小的蒸馏误差条件下，我们证明算术混合目标相对 reference 的性能不退化，并给出 margin 大于拟合误差时的学生改进保证。”这比无条件“shared expert 优于 MOPD”更窄，但是真正可证明、可实验检验的主张。
