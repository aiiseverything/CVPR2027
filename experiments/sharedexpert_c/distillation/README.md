# 按 domain 路由的三教师 OPD

本目录实现最朴素的三教师 on-policy distillation：原始 Qwen3.5-4B
生成回答，根据 **prompt 的固定 domain 标签**选择一位冻结教师，对学生实际生成的
token 计算条件 log-probability，再更新学生。教师不生成训练答案，不按正确率挑选
样本；未实现 C 方案的前缀探测、校准或自适应路由。

代码作为项目已有 VERL / Vision-OPD fork 的独立插件运行，不修改第三方源码或教师训练入口。
复用现有 `.venv-train`、Qwen3.5 多模态 rollout、FSDP2 和 checkpoint 保存机制。
上游 [VERL OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)解释了学生轨迹上
蒸馏的基本流程；本目录的配置接口针对本项目固定 commit，并非最新版上游接口。

## 已确定的实验协议

| 项目 | 设置 |
|---|---|
| 学生初始化 | 原始 Qwen3.5-4B，全部参数训练，含视觉部分 |
| 教师选择 | perception → 感知教师；reasoning → 推理教师；generalist → 通用教师 |
| 数据 | 三域约 1:1:1，6,400 条，每条在一次训练中使用一次 |
| 训练 | 200 optimizer steps × prompt batch 32 × 每题 1 条回答 |
| prompt / response | 2048 / 8192，保留 thinking |
| 采样 | temperature=1、top_p=1、top_k=-1 |
| 优化 | LR 1e-6，10 steps warmup，之后恒定；每轮一个完整 minibatch、一次更新 |
| 损失 | sampled-token reverse-KL score-function surrogate，sequence-mean token-mean |
| 额外目标 | 无 GRPO、外部答案奖励、reference KL 或 entropy 正则 |
| 数值修正 | 沿用学生 vLLM/HF log-prob 重算及 token importance correction，上限 2 |
| 存储 | 每 25 steps 和最后一步保存，保留最近两个 checkpoint |
| 默认资源 | 学生 FSDP/rollout 1 GPU；冻结教师评分单独 1 GPU |

学生预算、每题采样数、初始化和三域比例由用户指定；LR、长度、warmup 和视觉处理沿用项目设置。
缺失或未知 domain 报错。“其余数据”在数据准备阶段明确标为 generalist，运行时不猜标签。
两卡由隔离的本地 Ray 实例分配角色，不保证输入列表中的某一物理卡固定承担学生角色。

## 数据

`prepare_data.py` 从已有三个 prepared 训练 parquet 采样，保留原始图片、消息、来源和答案元数据，
新增 `domain` 与 `opd_index`。答案元数据不参与蒸馏损失。

| domain | 数据源与实际条数 | 合计 |
|---|---|---:|
| perception | TallyQA 854、CLEVR 853、RefCOCO 427 | 2134 |
| reasoning | Geometry3K 853、MMK12 427、ViRL39K 853 | 2133 |
| generalist | VQAv2 1422、DocVQA 178、ChartQA 178、IF-format 355 | 2133 |

感知和推理域分别沿用 4:4:2 与 4:2:4 的来源配比；通用域剔除原 generalist 训练池中的
感知/推理回放，保留 VQA:文档图表:IF=4:1:1。
按 perception、reasoning、generalist 循环排列，不 shuffle。每连续 96 条（3 steps）严格等比例；
32 和 6400 都不能被 3 整除，因此单步配额轮换 11/11/10，总量相差一条。

本次产物有 6,400 个独立 record ID，没有重复曝光。训练域之间可能有同图不同问题；
这不是独立的 teacher holdout。本次复用教师训练池用于基础蒸馏。
构建时用现有评测清单的图片/问题/记录标识排除重叠，不使用评测答案；该检查不等于语义级去污染。
清单记录来源 SHA-256、每来源配额、重复曝光及交集。原始三个 parquet 不被修改。

```bash
experiments/sharedexpert_c/.venv-train/bin/python -m experiments.sharedexpert_c.distillation.prepare_data
```

输出位于 `data_pipeline/prepared/domain_opd_6400.parquet` 和同名 `.manifest.json`。
parquet 按项目约定不提交 Git，可用上述命令重建。

## 损失和实现边界

对于学生前缀 h、学生实际采样的动作 a，取 q 为 domain 对应的冻结教师分布。
教师输入直接复用 rollout 的 input IDs、mRoPE position IDs 和图像 patch tensors，
没有解码后重编码、重新套模板或替换图片。评分一次一条序列，去除外侧 padding，
防止 Qwen3.5 recurrent attention 受补齐 token 干扰。

实现使用：

```text
rho = exp(log p_theta(a|h) - log p_old(a|h))
loss_token = stopgrad(rho * (log p_theta(a|h) - log q(a|h))) * log p_theta(a|h)
```

另外乘已冻结的 rollout importance weight。在固定前缀上对动作分布取期望，其梯度等于
reverse KL 的梯度（忽略状态分布导数）。有限采样和被截断的 vLLM/HF correction 引入近似。
这是采样梯度估计，不是完整词表 KL 的逐项求和，也不是 top-k KL；
日志中的 surrogate loss 和 sampled log-ratio 可以为负，不能标为精确 KL 或任务准确率。
EOS 纳入有效响应，padding 不参与；达到长度上限的有效 token 仍用于 OPD，不使用教师 RL
中“截断答案奖励置零”的规则。

`reward_manager.py` 仅借用 VERL reward tensor 传输冻结教师 log-prob，
`mopd_targets` 原样转交这些值，`mopd_reverse_kl` 实现上述目标。旧 VERL 日志中的
`critic/score`、`critic/rewards` 等通用字段因而不是正确率。阅读 `mopd/*` 和实际评测结果。
`agent_loop.py` 显式携带 domain，避免现有 VERL 异步生成边界丢失自定义 parquet 列。

教师在单独 GPU 上按域分组评分，默认只驻留一个模型，切换时先卸载旧模型。
`teacher_backend.max_resident_teachers=3` 可以省加载时间，但需重新验证显存。
教师完整 backbone 前向后分块计算 LM head 和词表归一化，不传输完整 logits。
学生使用原有 VERL 前向，仍可能产生较大的 logits 中间张量；本机 96GB 卡的完整训练显存
必须由 GPU smoke 验证，CPU 测试无法保证可运行。

只支持当前 dense Qwen3.5 多模态架构和一致的词表 ID / 视觉 patch 几何。
导出工具增加的末尾 special-token 声明可以兼容，前提是已有 token ID 不变、
新增 ID 不冲突且处于原模型已有的输出词表内；不修改任何 tokenizer 文件或重排权重。

## 检查和运行

从项目根目录执行。默认配置在 [configs/domain_opd.json](configs/domain_opd.json)。
感知/推理默认绑定已导出的 step100 教师；通用教师绑定当前训练预计输出的
`runs/generalist/hf/global_step_100`。只有导出完成并通过文件检查后才允许启动。

```bash
# 无需 GPU；打印解析后的协议、资源布局及 overrides
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/distillation/launch.py --dry-run

# 无需 GPU；检查三教师/学生文件、数据 domain、图片路径、Hydra 配置
CUDA_VISIBLE_DEVICES='' experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/distillation/launch.py --check

# 等教师训练完成、两张 GPU 空闲后执行。独立目录、三域各一题、一步非零 LR 更新。
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/distillation/launch.py --gpus 0,1 --smoke

# 要求相同模型、数据、代码和 GPU 数量的 smoke 完成凭据
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/distillation/launch.py --gpus 0,1 --execute
```

启动器拒绝繁忙 GPU，不终止已有进程，不接入已有 Ray 集群。运行产物位于
`runs/domain_opd[_smoke]/`，包括配置、模型权重/数据/代码哈希、训练日志、TensorBoard 和 checkpoint。
重复 `--execute` 恢复同一协议；任何哈希变化要求使用新的 `--run-name` 并重新 smoke。
已完成 200 步不会继续到 201。smoke checkpoint 不用于初始化正式训练。

最终 FSDP checkpoint 可用现有 merger 转为 HF 供统一评测：

```bash
PYTHONPATH=pre-lab/repo:. CUDA_VISIBLE_DEVICES='' experiments/sharedexpert_c/.venv-train/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir experiments/sharedexpert_c/runs/domain_opd/checkpoints/global_step_200/actor \
  --target_dir experiments/sharedexpert_c/runs/domain_opd/hf/global_step_200 \
  --use_cpu_initialization
```

## 验证状态

CPU 测试覆盖路由与未知标签、6,400 条配额/去重/评测标识排除、真实 Hydra 配置和插件加载、
采样损失与解析 reverse-KL 梯度等价性、teacher stop-gradient、padding、importance weights、
多模态张量传递、教师评分的因果位移及 chunk 等价性。使用真实随机小型 HF Qwen3.5 的
CPU 前向补充验证 scorer 与完整 logits 的数值一致性。
本次 **47 项测试全部通过**，环境版本、源码哈希、数据检查和当前缺失的通用教师导出
记录在 [verification_cpu.json](verification_cpu.json)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=pre-lab/repo:. \
experiments/sharedexpert_c/.venv-train/bin/python -m unittest discover \
  -s experiments/sharedexpert_c/distillation/tests -v
```

搭建时两张 GPU 正在训练 generalist，因此没有启动 GPU smoke 或学生训练。
CPU 通过不代表三教师 4B 模型的 GPU 训练已验收；当前通用教师最终导出仍待完成。
