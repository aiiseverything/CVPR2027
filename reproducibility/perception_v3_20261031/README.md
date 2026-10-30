# 感知 v3：训练完成与高精度感知结果

本快照日期为 **2026-10-31（Asia/Shanghai）**。感知 v3 已完成 100 步正式训练和最终 HF 模型导出，并完成高精度感知的 1,722 道题。随后用户要求停止后续评测：OCR 服务启动后被停止，成功预测为 **0/997**；主套件 **0/3,423，未启动**。这两套没有 v3 最终成绩。

导出完成于 2026-10-30 17:31:09 UTC（北京时间 10 月 31 日 01:31:09）；高精度感知完成于 18:45:41 UTC。停止请求、进程退出核验和队列终态分别保存在 [stop.json](execution/stop.json) 与 [queue_snapshot.json](execution/queue_snapshot.json)。这些文件是静态记录，不是运行中的队列。

## 高精度感知对比

所有列采用相同的套件、样本选择、生成协议和评分协议哈希，且均为 1,722 题完整结果，缺失和 API 错误均为 0。分数为百分比，差值为百分点。

| 基准 | Baseline | 感知 v1 | 推理教师 | 感知 v2 | 感知 v3 | v3 − v2 |
|---|---:|---:|---:|---:|---:|---:|
| VStar（115） | 80.87 | 81.74 | 82.61 | 80.87 | 80.00 | −0.87 |
| Zoom（807） | 47.46 | 50.68 | 52.79 | 50.81 | 46.84 | −3.97 |
| HRBench 4K（400） | 86.00 | 89.25 | 90.75 | 90.50 | 87.50 | −3.00 |
| HRBench 8K（400） | 81.75 | 84.75 | 86.00 | 84.75 | 82.50 | −2.25 |

感知 v3 四项均低于感知 v2，Zoom 降幅最大。v3 有 232 条长度截断回答并按规则记零，v2 为 115 条；该现象本身不能证明退步的原因。

- 可直接制表的数据：[comparison.csv](highres/comparison.csv)、[comparison.json](highres/comparison.json)。
- v3 的完整分项、错误及输出长度统计：[summary.json](highres/summary.json)、[summary.csv](highres/summary.csv)。
- 旧四模型的原始汇总：[historical_four_model_summary.json](highres/historical_four_model_summary.json)。新 v3 汇总中的旧模型为 `not_started`，只表示它们没有在 v3 的独立输出目录重新运行；历史成绩来自上述旧汇总。

这些是项目选定的感知子集成绩，不是各基准完整官方成绩；本次不包含 MME-RealWorld。VStar 采用 direct attributes，Zoom 使用冻结的感知题目选择，HRBench 使用 single/FSP 子集。每个 HRBench 分辨率的 400 行是 100 个原题的四轮选项排列，报告 cycle-average accuracy。Zoom 使用本地确定性匹配，没有调用官方 LLM judge。

所有模型读取完整原图，不提供目标特权裁剪。处理器像素面积上限为 33,554,432，下限 4,096；prompt 上限 40,960 token，输出上限 8,192，总上下文 49,152。thinking 开启，temperature=0、top_p=1、seed=20261019；空回答、未完成 thinking 和长度截断按既定规则记零。不同基准的分母与含义不同，本表不另合成一个跨基准总准确率。

## 训练与导出记录

最终模型来自 Qwen3.5-4B 的感知 v3 GRPO 训练，使用 3,000 条通过固定基座难度筛选的数据。每条候选有一组 8 个回答，接受条件为 2–6 个回答正确；这不保证训练时重新采样的正确数不变。六类配比为 20:20:20:15:15:10：

| 类别 | 条数 |
|---|---:|
| 物体类别和属性 | 600 |
| 定位 | 600 |
| OCR / 文档 | 600 |
| 空间关系和局部细节 | 450 |
| 目标存在性 | 450 |
| 通用视觉问答回放 | 300 |

正式预算为 100 optimizer steps、prompt batch 30、group size 8、学习率 1e-6，thinking 开启，prompt/response 上限为 2,048/8,192。24,000 是该预算的回答采样数，不包含失败或恢复期间被重做的采样。最终续跑从 checkpoint 50 恢复，在单张 RTX 6000D 上完成；实际内存、offload 和 eager 设置以 [resolved_config.yaml](training/resolved_config.yaml) 为准，不能用原四卡模板推断这次运行。

[completion.json](training/completion.json) 和 [export_complete.json](training/export_complete.json) 记录 step100 正常完成及导出。导出权重保留在原实验存储，不随此 Git 快照发布。[checkpoint.json](highres/checkpoint.json) 记录权重文件名、大小和修改时间，以及元数据内容哈希；它没有声称完成了整个权重文件的内容哈希。

CPU 数据、模型和依赖检查见 [dataset_validation.json](training/dataset_validation.json)、[resume_data_cpu_checks.json](training/resume_data_cpu_checks.json)、[resume_model_cpu_checks.json](training/resume_model_cpu_checks.json) 和 [import_checks.json](training/import_checks.json)。快速续跑入口复用了这些独立检查，没有重复执行入口 preflight，详见 [launch_receipt.json](training/launch_receipt.json)。没有额外运行独立 GPU smoke。

## 环境与复现范围

训练依赖记录为 torch 2.10.0、vLLM 0.18.0、transformers 5.5.0、Ray 2.53.0、FlashAttention 2.8.3；sm120 构建记录见 [flash_sm120_build.json](training/flash_sm120_build.json)。评测使用另一个环境：torch 2.13.0+cu130、vLLM 0.30.0、transformers 5.18.0、flashinfer 0.6.18.post1。环境和输入检查的准确边界见 [environment_highres_cpu_checks.json](checks/environment_highres_cpu_checks.json)：启动前检查了 CPU 导入、冻结 token 数和图片存在性/长度，并未进行独立 GPU kernel 测试；之后已执行完整正式高精度评测。

本快照包含配置、哈希、检查凭据和聚合结果，不包含模型权重、训练 parquet、冻结题目 JSONL、图片数据库或逐样本生成。重跑需要恢复这些原始输入：

1. 按 [数据管线](../../experiments/sharedexpert_c/data_pipeline/README.md) 和 v3 源码准备训练数据，并核验 [dataset_manifest.json](training/dataset_manifest.json) 的记录/分组/文件哈希。训练入口与各次恢复记录中的绝对路径必须按机器检查。
2. 恢复最终 step100 HF 模型、冻结高精度套件及 `images.sqlite`，并保留或正确映射冻结元数据中的基座模型路径。原始套件的 SHA-256 为 `28886d7f5496dbeb250147c36d715fff08d29da51f06bfa8820cc477331ca95c`；选择 SHA-256 为 `ad2d075d517f9dec50d78f8775d47b2a2993919300fb67b1f0545740b052ed18`。在新位置重建输入时应产生新的可审计凭据，不应伪造旧哈希或复用旧的完成状态作为新运行证明。
3. 使用 [register_model.py](../../experiments/sharedexpert_c/evaluation/highres/register_model.py) 为新导出建立私有元数据视图，保留源冻结文件；它核验所选 1,722 题的模板、分词、prompt/视觉 token 数，并检查代表图片的实际 CPU 预处理。执行设置和实际注册结果见 [run.json](highres/run.json)、[status_snapshot.json](highres/status_snapshot.json) 与 [registered_suite.meta.json](highres/registered_suite.meta.json)。

恢复完整输入后，高精度评测的命令结构如下；变量需指向经过检查的本地环境与模型。此命令不包括已取消的 OCR 和主套件。

```bash
EVAL_PY=/path/to/eval-environment/bin/python
EXP=experiments/sharedexpert_c
FROZEN=$EXP/evaluation/data/highres_perception_v1
MODEL=$EXP/runs/perception_v3/hf/global_step_100
VIEW=$EXP/runtime/eval_perception_v3_inputs/highres
RESULTS=$EXP/evaluation/results_perception_v3/highres

"$EVAL_PY" "$EXP/evaluation/highres/register_model.py" \
  --source-root "$FROZEN" --root "$VIEW" --model "$MODEL" \
  --name perception-v3-step100 --benchmarks vstar zoom hrbench_4k hrbench_8k
"$EVAL_PY" "$EXP/evaluation/highres/run.py" \
  --root "$VIEW" --model "$MODEL" --name perception-v3-step100 \
  --output-root "$RESULTS" --gpu 0 \
  --benchmarks vstar zoom hrbench_4k hrbench_8k \
  --concurrency 64 --throughput-mode --enforce-eager
```

汇总器读取输出目录 `status.json` 中的 `config.benchmarks` 确定四项范围；正式队列会写入它。独立重跑完成后，对此次新建的输出目录记录同一范围再汇总，不可将四项预测按六项分母汇总：

```bash
cp reproducibility/perception_v3_20261031/highres/scope.json "$RESULTS/status.json"
"$EVAL_PY" "$EXP/evaluation/highres/scoring.py" --root "$VIEW" --results "$RESULTS"
```

所有复制文件的原始相对路径、字节数和 SHA-256 见 [provenance.json](provenance.json)。记录中的 PID、服务器地址和绝对路径属于历史审计字段，不是可直接跨机器复用的服务配置。
