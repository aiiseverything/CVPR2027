# SharedExpert / CVPR2027

围绕 SharedExpert 与多教师 on-policy distillation（MOPD）的论文设计和多模态训练实验。当前阶段是用 Qwen3.5-4B 分别训练感知、推理、通用三个教师，再统一评测；C 方案的学生蒸馏及 MOPD 对比尚未开始。

## 项目入口

- [初始论文骨架](SharedExpert_MOPD_论文骨架.md)
- [候选方案与修订骨架](SharedExpert_MOPD_候选方案与修订骨架.md)
- [数学审查](SharedExpert_MOPD_数学审查.md)
- [文献核验](SharedExpert_MOPD_文献核验.md)
- [方法与实验审查](SharedExpert_MOPD_方法与实验审查.md)
- [训练配置、数据与可验证奖励](experiments/sharedexpert_c/docs/训练配置与可验证奖励.md)
- [搭建问题复盘](experiments/sharedexpert_c/docs/搭建问题复盘.md)
- [评测配置与问题记录](experiments/sharedexpert_c/docs/评测配置与问题记录.md)

## 当前训练协议

| 项目 | 设置 |
|---|---|
| 教师 / 后续学生基座 | Qwen3.5-4B |
| 感知教师 | TallyQA : CLEVR : RefCOCO = 4 : 4 : 2 |
| 推理教师 | Geometry3K train : MMK12 : ViRL39K = 4 : 2 : 4 |
| 通用教师 | VQA : 文档图表 : 推理 : 感知 : IF = 40 : 10 : 20 : 20 : 10 |
| 每教师预算 | 300 steps，prompt batch 30，group 8 |
| 长度上限 | prompt 2048，response 8192 |
| RL | GRPO，KL / entropy 关闭，LR 1e-6 |
| 分布式 | 4 GPU，FSDP2，单序列 microbatch 1 |
| checkpoint | 每 25 steps 保存，保留最近 2 个 |

保留每个教师 12,000 行 prepared 数据，当前 batch 30 协议依顺序使用前 9,000 行，共 72,000 次响应采样。多序列 packing 尚未通过梯度等价性验证，正式训练不启用。完整、权威参数见 [common.json](experiments/sharedexpert_c/training/config/common.json) 和实验文档。

## 目录

```text
SharedExpert_MOPD_*.md          论文设计与审查
experiments/sharedexpert_c/
  training/                    训练入口、环境构建、验证与配置
  data_pipeline/               数据获取、构建、清洗与审计
  evaluation/                  固定评测套件构建、推理和评分
  docs/                        训练、搭建和评测文档
  tests/                       奖励测试
  logs/                        小型配置、环境版本、审计记录和辅助脚本
  runtime/                     仅跟踪明确列出的历史验收凭据
  monitor_training.py          OOM 监控和训练曲线
  run_experiment.py            三教师训练与评测编排
pre-lab/                       前期模型、评测与感知探针脚本
  repo/                        Vision-OPD 子模块
  vlmevalkit/                   VLMEvalKit 子模块
reproducibility/               依赖版本、源码补丁和历史指标快照
```

## 获取完整源码和依赖修改

```bash
git clone --recurse-submodules https://github.com/aiiseverything/CVPR2027.git
cd CVPR2027
python3 reproducibility/restore_dependencies.py
python3 reproducibility/restore_dependencies.py --check
```

两个第三方仓库均锁定到精确 commit；本地修改完整保存在 [patches](reproducibility/patches)，版本与补丁 SHA-256 见 [dependencies.json](reproducibility/dependencies.json)。恢复脚本拒绝覆盖其他未提交修改。子模块设置 `ignore=dirty`，因为应用这些记录在案的补丁后，其工作树会有预期修改；请用上述 `--check` 核验，而非仅看顶层 `git status`。

## 环境、数据和运行

依次阅读：

1. [训练环境与入口](experiments/sharedexpert_c/training/README.md)：独立虚拟环境、FlashAttention 构建、CPU/GPU 检查及启动。
2. [数据准备](experiments/sharedexpert_c/data_pipeline/README.md)：来源、配比、去重与评测集泄漏检查。
3. [评测入口](experiments/sharedexpert_c/evaluation/README.md)：固定套件与严格评分规则。
4. [监控入口](experiments/sharedexpert_c/monitoring/README.md)：独立后台监控、OOM 告警、TensorBoard 指标和图表。

原实验工作区为 `/home/ml-user/workdir/CVPR2027`。部分脚本、配置和审计凭据记录了该绝对路径以及服务器 GPU/端口布局；换机器时须按文档重新配置、生成数据及执行 smoke/完整性检查。历史成功凭据不能充当新机器的验证结果。这里不提供跳过训练门禁的一键启动命令。

## 数据与运行产物

Git 跟踪项目代码、论文、配置、数据准备与评测脚本、小型审计记录以及历史结果快照。以下保留在训练服务器，由 `.gitignore` 排除：

- 模型权重、checkpoint、优化器状态和虚拟环境；
- 下载数据、图片、parquet / TSV、评测题目与逐样本生成内容；
- 原始服务日志、进程环境、PID、运行锁、Ray/vLLM 缓存和持续增长的监控采样。

数据和模型请通过项目脚本从原始来源获取，并遵循各自使用条款。第三方源码遵循对应子模块中的许可；本仓库未额外指定项目许可证。

[2026-10-20 历史快照](reproducibility/snapshot_20261020/snapshot.json) 包含已完成步数、指标文件哈希、训练曲线和 baseline 汇总；它不是实时状态，也不是训练完成声明。当前教师训练尚未结束，不能把早期 reward 或 baseline 成绩写成最终教师结果。旧故障和未通过的尝试仍保留在复盘文档。

![训练指标历史快照](reproducibility/snapshot_20261020/training_metrics.png)
