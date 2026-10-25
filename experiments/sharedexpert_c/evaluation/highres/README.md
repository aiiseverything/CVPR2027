# 高分辨率感知子集评测

评测基座、推理教师、感知 v1、感知 v2 四个 4B 模型。V⋆ 115、Zoom 807、HR-Bench 4K/8K 各400、MME-RW EN 19,129、CN 3,610，每模型24,461行，共97,844次推理。HR-Bench每项是100个基本问题的四轮选项排列。

[资料快照](../../../../reproducibility/evaluation_20261026/README.md) 保存旧三套结果、诊断和本次冻结清单。本轮为感知子集，不是官方完整六基准成绩。

## 推理与评分

完整原图最多33,554,432像素，不使用提供的框或裁剪图。输入上限40,960 tokens，输出（含思考）8,192，总上下文49,152；thinking开启、temperature=0、top_p=1。四模型使用同一题目和处理协议。

显存预算95%，服务端最多64序列、每轮32,768 tokens。客户端按本次启动日志的KV容量预留prompt+8192+4096，总量不超过容量的90%，最多64在途；容量日志不可读时上限32。明确的本次CUDA OOM才依次降为32/16384/95%和32/16384/92%。不按分辨率硬锁2并发，不缩减单题预算。

Zoom选择是AI根据题干及选项完成的静态感知范围审查，未进行人工图像审阅；使用本地确定性选项/数值评分，不是官方LLM judge。所有11条精确识别的源标签异常均保留原文、选项和gold，主分母不变；另报排除11条的敏感性分析。精确匹配夹具见annotation_issues.py。

## 准备和启动

需要本地四份完整HF模型、可用CUDA/vLLM环境，以及requests、pyarrow、Pillow、transformers、torch。模型相对位置见protocol.py的MODEL_PATHS；权重和虚拟环境不随Git发布。当前环境记录见RUNNING.txt。

从仓库根目录执行，EVAL_PY按本机环境修改：

```bash
EVAL_PY=/root/.venv-eval/bin/python
DATA=experiments/sharedexpert_c/evaluation/data/highres_perception_v1
RESULTS=experiments/sharedexpert_c/evaluation/results_highres_perception_v1
mkdir -p "$DATA" "$RESULTS"
cp reproducibility/evaluation_20261026/highres/source_manifest.json "$DATA/"
cp reproducibility/evaluation_20261026/highres/selection.json "$DATA/"
"$EVAL_PY" experiments/sharedexpert_c/evaluation/highres/download.py \
  --root "$DATA" --source-root /path/to/source_scratch \
  --endpoint https://hf-mirror.com --workers 4 --retries 12
nohup setsid "$EVAL_PY" -u experiments/sharedexpert_c/evaluation/highres/launch_queue.py \
  --root "$DATA" --output-root "$RESULTS" --python "$EVAL_PY" \
  --gpu 0 --concurrency 64 --throughput-mode >"$RESULTS/queue.log" 2>&1 </dev/null &
```

上面使用已审查冻结清单；finalize_selection.py用于重新审查后的提交，正常复现无需重新筛选。原始Parquet约40.39GiB，可放临时盘；selected图片以SHA去重保存为挂载盘上的SQLite，避免大量小文件，并保留15GiB空间保护。完成整理后推理不依赖临时源或电脑代理。

队列依次下载就绪检查、CPU准备、CPU处理器与计数校验、正式模型推理；不额外安排GPU smoke。首条正式题校验API实际token计数，随后并发；逐题保存并支持同一清单/模型/策略的断点恢复。不要同时启动两个队列。

```bash
cat "$RESULTS/status.json"
cat "$DATA/prepare_state.json"
cat "$DATA/freeze_state.json"
cat "$RESULTS/full/baseline/status.json"
"$EVAL_PY" -m unittest discover -s experiments/sharedexpert_c/evaluation/highres -p 'test_*.py' -q
```

当前49项CPU测试通过。正式GPU容量与吞吐以本机实际运行日志为准；静态快照不代表实时进度。
