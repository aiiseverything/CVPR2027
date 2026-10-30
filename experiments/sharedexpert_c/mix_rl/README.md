# 同数据 Mix-RL 对照

用原始 Qwen3.5-4B 在 OPD 的同一份混合数据上执行普通 GRPO：
**200 steps × 每步 32 个 prompt × 每题 8 条响应**。
同一个策略学习三个领域，以数据集提供的可验证答案计算 outcome reward。
本方案不加载蒸馏教师，不使用 OPD 的 token log-prob 或自适应教师路由。

## 与 OPD 对齐的实验设定

| 项目 | domain OPD | Mix-RL |
|---|---|---|
| 初始化 | 原始 Qwen3.5-4B | 同一模型路径 |
| 训练数据 | `domain_opd_6400.parquet` | 同一文件、相同 SHA-256、相同顺序 |
| 感知/推理/通用 | 2134 / 2133 / 2133 | 相同 |
| 优化步数 | 200 | 200 |
| 每步 prompt | 32 | 32 |
| 每题响应 | 1 | 8 |
| 每步响应 | 32 | 256 |
| 总 prompt 曝光 | 6400 | 6400 |
| 总采样响应预算 | 6400 | 51200 |
| 监督 | domain 对应教师的 token log-prob | 现有任务可验证奖励 |
| 目标 | sampled reverse-KL | clipped GRPO，按每题 8 条响应标准化 advantage |
| 默认两卡分工 | 一卡学生，一卡教师评分 | 两卡 FSDP 训练同一个混合策略 |

这是**同数据、同 prompt 曝光、同更新步数**的比较；响应采样量和计算成本不同。
Mix-RL 生成响应数为 OPD 的 8 倍，OPD 另有教师评分成本。比较效率时应同时报告
真实生成 token 数、耗时、GPU-hours 和最终统一评测结果。

两组都沿用 LR 1e-6、10-step warmup 后恒定、prompt/response=2048/8192、thinking、
temperature=1、top_p=1、top_k=-1、全参数（含视觉）训练、单序列 microbatch=1、
gradient clip=1、weight decay=0 和 sequence-mean token-mean loss 聚合。
每次 rollout 只做一次完整 minibatch 更新，KL 正则和 entropy 正则关闭。
FSDP2、optimizer CPU offload、rollout 显存比例 0.4、max_num_seqs=8 沿用 OPD 的设置。

`ppo_mini_batch_size=32` 表示 prompt 数；当前 VERL worker 内部会乘 `rollout.n=8`，
因此实际每次更新使用 256 条响应。不能把该配置直接改成 256，否则会再乘一次 8。
每个 prompt 的 8 条响应单独组成 GRPO group，不按 domain 混合归一化；
全对/全错且奖励相同的组保留在预算中，advantage 为零，不重采样。

## 数据与奖励

复用已经构建的 `data_pipeline/prepared/domain_opd_6400.parquet`，不重新抽样或 shuffle。
来源及配比见 [OPD 数据说明](../distillation/README.md)。
配置固定文件 SHA-256，并在启动前核对 `opd_index`、record ID、三域循环顺序、图片路径、
训练 split 与 reward schema。数据/初始化/主要超参数偏离 OPD 时拒绝作为这个对照启动。

直接复用 `rewards.py` 与 `training/reward_manager.py`：精确答案、数学答案、
VQA 共识、bbox IoU、ANLS、宽容数值匹配、带内容校验的 JSON 格式任务。
需要有效 final boxed/JSON 答案；截断答案奖励为 0。没有单独的纯格式奖励。
reward audit 检查参考答案能获得合理正奖励，不代表模型准确率；
低共识 VQA 即使输出最佳参考答案也可能得不到 1 分。

## 运行

所有命令从项目根目录执行，复用现有训练虚拟环境，不需要安装或升级依赖。
配置为 [configs/mix_rl.json](configs/mix_rl.json)，输出在 `runs/mix_rl/`。

```bash
# 只打印配置，不创建 Ray 集群或访问 GPU
python3 experiments/sharedexpert_c/mix_rl/launch.py --dry-run

# CPU 检查：模型、完整数据、奖励和 OPD 对齐
CUDA_VISIBLE_DEVICES='' experiments/sharedexpert_c/.venv-train/bin/python \
  experiments/sharedexpert_c/mix_rl/launch.py --check

# GPU 空闲后：独立 smoke，4 prompts × 8 responses，一步非零学习率更新
experiments/sharedexpert_c/.venv-train/bin/python \
  experiments/sharedexpert_c/mix_rl/launch.py --gpus 0,1 --smoke

# 相同模型/数据/代码/GPU数量的 smoke 完成后：正式 200 steps
experiments/sharedexpert_c/.venv-train/bin/python \
  experiments/sharedexpert_c/mix_rl/launch.py --gpus 0,1 --execute

# 训练完成后，CPU 导出供统一评测；不会启动训练
experiments/sharedexpert_c/.venv-train/bin/python \
  experiments/sharedexpert_c/mix_rl/launch.py --merge-only
```

可选 `--gpus 0` 使用单卡学生训练；显存是否足够需单独 smoke。
`--run-name mix_rl_<名称>` 可创建独立运行。默认 smoke 保存在 `runs/mix_rl_smoke/`，
不会被正式训练恢复。正式运行保存 protocol、源码/模型/数据哈希、TensorBoard 和日志，
每 25 steps 及最后一步保存 checkpoint，保留最近两个；重复执行恢复同一协议，
完成 200 steps 后不会继续训练。修改模型/代码/数据必须换 run name 并重新 smoke。
如果 smoke 在 checkpoint 保存后被中断、没有成功凭据，启动器会明确报错；请换新 run name 重做 smoke。

启动器检查 GPU 空闲，并与 OPD 共用设备启动锁；不停止已有进程，不接入已有 Ray 集群。
Mix-RL 不依赖三教师最终导出，当前尚未启动 GPU smoke/正式训练是因为两张卡仍用于教师训练。

## CPU 验证

测试覆盖真实 Hydra 配置、OPD 对齐、32×8 有效 batch、按题 GRPO grouping、
uniform-group 零 advantage、smoke 设置、数据漂移/顺序/重复检测和全部七类奖励格式。
实际 6,400 行的奖励参考检查与学生权重文件完整性检查均已通过；
结果与环境版本记录在 [verification_cpu.json](verification_cpu.json)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=pre-lab/repo:. \
experiments/sharedexpert_c/.venv-train/bin/python -m unittest discover \
  -s experiments/sharedexpert_c/mix_rl/tests -v
```

GPU smoke 和 200-step 稳定性尚未验证；CPU 成功不是训练完成或性能提升的声明。
