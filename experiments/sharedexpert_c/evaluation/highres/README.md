# 高分辨率感知子集评测

本目录实现四个 Qwen3.5-4B 模型的原图评测：基座 `baseline`、推理教师 `reasoning-step100`、感知教师 `perception-step100` 和感知教师 v2 `perception-v2-step100`。数据选择、生成协议和运行资源配置分别记录，支持按已完成预测恢复运行。

**静态状态：2026-10-25 19:20:36 UTC / 2026-10-26 03:20:36 Asia/Shanghai。** 当时 87 个源文件已下载并校验，CPU 数据准备完成 78 个分片，队列仍处于 `preparing`，模型任务列表为空。该快照没有高分辨率 GPU 评测成绩。发布后的进度请查看实际运行目录；本文不是实时监控。

已完成的旧五模型评测及本次清单副本见[评测资料快照](../../../../reproducibility/evaluation_20261026/README.md)。旧评测采用不同图像预算，不能作为本协议的结果。

## 数据与筛选范围

| 子集 | 每模型题数 | 固定选择规则 |
|---|---:|---|
| VStar | 115 | `direct_attributes`；排除相对位置任务 |
| ZoomBench | 807 | 对全部 845 条题干及边界选项进行 AI 文本审阅后固定源 ID |
| HR-Bench4K | 400 | `category=single`；100 个基础问题的四个循环版本 |
| HR-Bench8K | 400 | 同上 |
| MME-RealWorld EN | 19,129 | 官方 `Perception`，排除 `Attribute_Motion_*` |
| MME-RealWorld CN | 3,610 | 同上 |
| **合计** | **24,461** | 四模型共 **97,844** 次推理 |

这些是项目选择的低推理负担感知子集，**不是官方统一的 pure-perception split**。Zoom 保留直接 OCR、颜色、材质、形状结构、普通物体／存在判断及直接计数，排除外部品牌或人物身份、国旗与地标身份、钟针读时及推导类问题；明确的数字显示保留为 OCR。选择依据题目文本，不依据 gold 或模型成绩。审阅没有逐图人工验证，普通物体命名、材质判断、计数和复杂场景定位仍可能需要语义知识或推理。

MME 使用官方任务分类进一步筛选，分类名称不保证每题完全不含推理。HR-Bench 的 400 行包含循环变体，不应解释为 400 个独立原题。完整规则和固定 ID 位于快照中的 [selection.json](../../../../reproducibility/evaluation_20261026/highres/selection.json)。

## 输入、生成与评分

[protocol.py](protocol.py) 固定 `sharedexpert-highres-perception-subsets-v1`：每题使用一张完整原图，不提供特权裁剪或标框。处理器的像素面积上限为 **33,554,432 pixels（32 Mi-pixels）**，并非 32,768 像素的边长；图像仍会按处理器规则缩放到预算内。输入最多 40,960 tokens，输出最多 8,192 tokens，总上下文 49,152；开启 thinking，temperature=0、top_p=1、seed=20261019。

[scoring.py](scoring.py) 使用明确标注的本地确定性评分。空输出、错误和截断输出不能当作正确答案；按最终答案抽取与任务类型进行字母、数值或规范化匹配。Zoom 的本地规则不等同于官方 LLM judge。HR-Bench 保留循环组与类别报告；MME 保留原始类别分组。部分完成状态不会冒充完整数据集成绩。

固定选择中发现 **11 条精确匹配的上游标注异常**，其中 10 条的 gold 不在展示选项标签中；包括缺失 gold、重复标签和非拉丁标签问题。[annotation_issues.py](annotation_issues.py) 仅允许这些固定身份与内容匹配的异常，不修改题目、选项或 gold。主结果保留全部 24,461 行；另报排除这 11 行的敏感性结果。精简的[异常摘要](../../../../reproducibility/evaluation_20261026/highres/annotation_issues_summary.json)保留原始审计哈希，不能代替原始审计文件。

## 吞吐与恢复

[throughput.py](throughput.py) 的 `profiled-kv-admission-v1` 初始档为 GPU 显存比例 **0.95**、服务器最多 **64** 个序列、每批 **32,768** tokens，客户端最多 **64** 个在途请求。这里的 64 是上限，实际并发由当前启动日志报告的 KV 容量进一步限制：

```text
token_budget = floor(0.9 × measured_GPU_KV_capacity)
request_reservation = prompt_tokens + 8192 + 4096
sum(inflight request_reservation) <= token_budget
```

未解析到本次启动的容量时退至最多 32 个在途请求；日志偏移保证不会读取旧启动的容量。单题超出已知总预算会明确报错，避免队列永久等待。vLLM 继续负责实际 KV 分配、视觉编码器预算和分块 prefill；客户端不再固定为“两张大图”的预算。

仅在当前所属服务器日志出现明确 CUDA OOM 证据时，有限降档到 `32 sequences / 16384 batch tokens / 0.95`，随后为同序列和批量、显存比例 `0.92`。配置和重试历史随运行保存；普通 API 错误、模型或数据哈希不匹配不能当作 OOM 降档依据。降档不降低原图、上下文或输出预算。显存预分配比例不是吞吐保证；本快照尚未实测该档位的正式 GPU 速度。

## 在新服务器复现

从仓库根目录执行以下步骤。仓库包含代码、汇总和清单副本；模型权重、源数据、图片数据库及逐条预测不随仓库发布，需要另外取得。四个 HF 模型的预期位置见 [protocol.py](protocol.py) 的 `MODEL_PATHS`；在冻结之前放置模型或建立对应的有效链接。

原运行环境使用 Python 3.12、PyTorch 2.13.0、Transformers 5.18.0、vLLM 0.30.0，并依赖 NumPy、PyArrow、Pillow 等。应使用与目标 CUDA/GPU 相容且固定版本的环境。历史 [LOCAL_ENV.md](../LOCAL_ENV.md) 记录旧服务器迁移，不能将其中的旧硬件参数或 smoke 步骤当成本轮要求。

先运行 CPU 单元检查；这些命令不启动 GPU，也不产生 smoke 检查点：

```bash
EVAL_PYTHON=/root/.venv-eval/bin/python  # 改为本机的已准备解释器
"$EVAL_PYTHON" -m unittest discover \
  -s experiments/sharedexpert_c/evaluation/highres -p 'test_*.py'
```

将发布快照中的两份冻结清单复制到**本机运行目录**。快照目录用于归档，不能直接作为可写运行目录：

```bash
EVAL_DIR=experiments/sharedexpert_c/evaluation
SNAPSHOT_DIR=reproducibility/evaluation_20261026
DATA_ROOT="$EVAL_DIR/data/highres_perception_v1"
mkdir -p "$DATA_ROOT"
cp "$SNAPSHOT_DIR/highres/source_manifest.json" "$DATA_ROOT/source_manifest.json"
cp "$SNAPSHOT_DIR/highres/selection.json" "$DATA_ROOT/selection.json"
```

`source_manifest.json` 和 `selection.json` 是原冻结文件的逐字节副本。`zoom_review_summary.json` 与 `annotation_issues_summary.json` 是剔除原题／gold 后的派生审阅摘要，保留原审计 SHA，不冒充清单引用的原始 audit 文件。正常复现已有选择只需两份冻结清单；**不要为此重新运行 `finalize_selection.py`**。该脚本用于重新建立完整审阅记录时的提交步骤，需要未打包的原注释和分片审阅文件。

选择容量足够的本地缓存盘下载原始分片；`--source-root` 必须指向本机实际目录。下载器在 `DATA_ROOT/sources` 创建链接，校验固定 revision、大小和 SHA，并复用已存在且校验通过的文件。已有缓存可直接指定同一路径；不会依赖旧服务器的 `/root/autodl-tmp` 路径。

```bash
SOURCE_CACHE=/path/to/local-cache/highres_perception_sources  # 替换为本机目录
"$EVAL_PYTHON" "$EVAL_DIR/highres/download.py" \
  --root "$DATA_ROOT" --source-root "$SOURCE_CACHE" \
  --workers 3 --endpoint https://huggingface.co
```

原始分片和后续图片 SQLite 是两份存储，另需预留日志、临时事务和结果空间；源文件总大小记录在 manifest。下载器支持断点和固定哈希验证，不要编辑 manifest 来适配未验证的文件。

下载完成后启动持久队列。队列先完成 CPU 数据准备、逐题 token 检查、模型／数据冻结，再直接运行四个正式模型；**没有额外 GPU smoke**。首次正式请求的 token 校验属于正式运行的一部分。

```bash
nohup "$EVAL_PYTHON" "$EVAL_DIR/highres/launch_queue.py" \
  --python "$EVAL_PYTHON" --root "$DATA_ROOT" \
  --output-root "$EVAL_DIR/results_highres_perception_v1" \
  --gpu 0 --concurrency 64 --throughput-mode \
  > "$EVAL_DIR/highres_queue.log" 2>&1 < /dev/null &
```

当前进度写入结果目录的 `status.json` 及数据目录的 `prepare_state.json`、`freeze_state.json`。重启时使用同一参数和目录；代码校验冻结身份并复用成功预测，最后一条重试记录生效。不要改运行中的源文件或冻结清单。`launch_queue.py` 在各模型完成后汇总，最终报告应保留评分版本、实际运行档位、异常敏感性结果与全部完成状态。

## 按明确请求跳过离线预检

2026-10-26，用户要求停止继续校验。运行数据根目录的 `skip_preflight.json` 设为 `{"enabled": true, "reason": "explicit user request"}` 后，跳过图片库重复扫描、四模型 processor 重放和实际 CPU 大图预处理，仅用基座 processor 一次构造正式输入及 token 预算。`suite.meta.json` 的 `preflight_policy` 明确记录跳过内容，不记作检查通过；正式响应仍记录实际 token 使用量。已完成的数据下载和准备凭据保留。
