# 三教师评测执行说明

本目录独立于旧 `pre-lab/bench`。它不会读取 `.env`，不会调用外部 judge，不会停止既有服务。入口 Python 为仓库的 `pre-lab/venv/bin/python`。所有四个模型使用同一冻结清单、相同 prompt、相同图像处理、相同生成预算和同一评分器。

## 固定协议

- 模型：未训练 Qwen3.5-4B baseline，以及 generalist / perception / reasoning 的 step 100 HF 导出。
- thinking=True，temperature=0，top_p=1，seed=20261019，max_tokens=8192；无额外 force-answering pass。
- 与训练统一：`finish_reason=length` 一律记零，即使截断前已有完整正确 final。答案抽取和惩罚前得分保留用于审计。
- prompt 上限 2048，按当前 AutoProcessor 的模板及展开后的视觉 token 精确计数。只在制表阶段统一排除超长样本，不截断视觉占位符或末尾问题。循环选项组/配对样本一起保留或排除。
- 图像 processor size：每图 longest_edge=min(786432,1572864//图像数) pixels，shortest_edge=4096 pixels，与训练一致。名称是 Hugging Face 的配置字段，值为像素面积，不是边长。每个请求显式设置，避免复用旧服务时继承其大图设置。
- 每个 API 响应的实际 prompt_tokens 必须和 CPU 预计算一致，否则停止；同一 suite/protocol 的哈希固定写入结果。
- 严格解析 `<answer>`，兼容明确的 boxed / 最终行答案。未闭合 thinking、空回答、无法解析都记错并单独统计。数值容许仅 1e-8 浮点误差；分数与小数可以等价。没有 LLM 猜测答案。符号等价表达可能被保守判错，报告中明确为本地严格匹配分数。
- RefCOCO 用 xyxy、0–1000 归一化坐标，IoU≥0.5；JSON 数组以外/越界/退化框判错。

## 评测集与解释边界

`core_suite.jsonl` 在训练前冻结，候选 **2924 行**，现行 v3 保留 **2923 行**（1 条 MMMU 多图问题超出 prompt 2048 预算）；完整排除清单以 `core_suite.meta.json` 为准：

| 维度 | 数据 | 样本 |
|---|---|---:|
| 推理 | MathVista_MINI，按 category 固定分层抽样 | 500 / 1000 |
| 推理 | MathVision_MINI | 全部 304 |
| 推理 | MMMU_DEV_VAL，仅 validation multiple-choice | 选 450，保留 449 |
| 感知 | MME count / position / color / existence | 全部 240 |
| 通用 | MMStar，固定分层抽样 | 500 / 1500 |
| 通用 | MMBench_DEV_EN_V11，250 个 base question 全部循环版本 | 930 行 / 250 组 |

RefCOCO heldout 500（UNC val，439 张图像）已经加入独立最终 `suite.jsonl`，总计 **3423 行**；`core_suite` 与最终套件采用相同图像预算。最终四模型均使用 `suite.jsonl`。已有 core baseline 结果逐行核对后导入，只补 RefCOCO。

MMBench 主指标是同一个问题所有循环选项都回答正确的比例。MME这里只报告四项 perception 子集的严格准确率，不声称是完整 MME 官方总分。MathVista/MathVision 的本地抽取协议与旧 DeepSeek judge 不同，历史成绩不进入新主表。三维总分是各基准归一化准确率的等权均值，逐基准分数仍保留。

最终回答互补性用于判断 C 路线是否值得启动，不等于已经验证 prefix continuation advantage，也不能用评测标签训练 C router。

## 生成与去污染

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/prepare_suite.py \
  --refcoco-jsonl /absolute/path/refcoco_heldout.jsonl --require-grounding
```

RefCOCO 输入每行需要 `id`, `question`, `images`（本地绝对路径列表）, `split`, `bbox_xyxy_1000`。split 不允许包含 train。默认从 heldout 确定性选择 500。

每行 `image_sha256` 始终为列表，定义为 `SHA256(struct.pack('>II', width, height) + RGB pixels)`，另存原始图像文件 SHA256。训练数据准备应排除与 eval 图像相同的样本，再检查题目/来源 ID。`question_sha256` 为 NFKC、casefold、合并空格后的题目哈希；不要只凭通用短问句独立判断污染。精确像素 hash 不覆盖所有 JPEG 重编码、裁剪或改写重复，需要保留 dataset split/source 的审计边界。

## 运行

对已存在、已获使用授权的本地 4B 服务运行 baseline（8112 已配置最多 8 图；8110/8111 仅单图，不能跑完整 suite）：

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/run_api.py \
  --suite experiments/sharedexpert_c/evaluation/core_suite.jsonl \
  --base-url http://127.0.0.1:8112/v1 --model Qwen3.5-4B \
  --name baseline --concurrency 4 \
  --output-root experiments/sharedexpert_c/evaluation/core_results
```

RefCOCO 加入最终 suite 后导入完全一致的已跑记录：

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/run_api.py \
  --suite experiments/sharedexpert_c/evaluation/suite.jsonl \
  --base-url http://127.0.0.1:8112/v1 --model Qwen3.5-4B \
  --name baseline --concurrency 4 \
  --import-results experiments/sharedexpert_c/evaluation/core_results/full/baseline
```

实际已启动的持久入口会等待 core 客户端退出后自动完成上述导入、补跑和汇总：

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/baseline_final.py
```

此命令使用独占锁；训练 orchestrator 在结尾再次调用会等待已有入口完成，并检查/续跑最终 baseline，避免重复推理。日志为 `baseline_core.log` / `baseline_final.log`，PID 为同名 `.pid` 文件。core 结果暂存 `core_results`，最终结果统一在 `results/full/baseline`。

训练进程退出并完成 HF 导出后，在空闲 GPU 上依次运行三个教师：

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/run_checkpoint.py \
  --model experiments/sharedexpert_c/runs/generalist/hf/global_step_100 \
  --name generalist --gpu 3
```

替换 name/path 为 perception、reasoning。launcher 自动选择本地端口，检查 GPU 无已有 compute process；仅退出自己启动的 vLLM 进程组。可加 `--dry-run` 查看命令而不启动 GPU。训练 shard 合并由训练流水线负责；该 launcher 要求完整 HF 模型目录，不修改 checkpoint。

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/evaluation/summarize.py
```

生成 `results/summary.md`, `summary.json`, `per_sample.csv`。不完整或 API 失败的模型不报告最终准确率。完整模型报告按图像/循环组 bootstrap 的 95% 区间、对 baseline 的配对差值、格式抽取失败及截断率，并给出 specialist 独立答对的案例计数。`run_api.py` 支持断点续跑：已成功样本不重跑，API 错误样本可重试。

推理协议与评分协议分开保存：`protocol_sha256` / `suite_sha256` / 每条 `row_sha256` 固定不变；`scoring_protocol` 明确记录截断判零。汇总每次从原始回答重计，输出独立 `predictions.scored.jsonl` 和 `scoring.json`，不修改仍在写入的原始缓存。旧回答导入/续跑会使用当前规则重计，单纯评分更新不触发 GPU 重推理。当前旧客户端运行期间，其内嵌旧 `correct` 字段仅作历史记录，以新 scored 文件和汇总为准。

## 已执行的验证

源码 CPU 编译检查；评分与断点续跑的 9 项 CPU 测试通过；v2 真实 4B 共 8 题、跨 6 个基准 smoke（包含 MMMU 多图）已完成，精确 prompt token 计数一致、最终无 API 错误，2 条 8192-token 截断按协议记错。最初单图服务拒绝的多图请求已在 8112 正确补跑。GPU 教师评测必须等待实际 step 100 模型导出。

初版 image max262144 的短暂 smoke/core 记录已整体归档到 `old_protocol/image262144`。现行 v2 与训练统一为 max786432，禁止混入旧协议预测。

2026-10-19：现行 v3 加入总图像像素预算 1572864。3423 行中只有 MMMU_DEV_VAL:10719 的视觉网格改变，prompt 从 1720 变为 1545 token；其余 3422 行原始记录 hash 不变。旧 v2 套件及所有已有生成保留在 `old_protocol/migration_v2_to_v3_total_image_budget`，`migration.json` 记录逐行核对和迁移依据。已有 329 条成功生成得到保留，173 条服务连接错误从成功缓存排除并自动重试。长度截断判零规则不变。

`run_checkpoint.py` 会在 GPU 空闲检查前验证 checkpoint provenance、suite/protocol、全部逐行 hash 和 API 成功状态；若完整则直接重新汇总，无需启动 GPU 服务。SIGTERM/SIGINT 也会清理该 launcher 自己启动的 vLLM 进程组。

OOM 后发现 EPC 旧 flock 租约会阻塞已退出进程的续跑，因此三个评测入口使用 `/tmp/sharedexpert-evaluation-locks-<uid>` 下、绑定绝对资源路径和用途的本机锁；本地服务和 GPU 仍受互斥保护。baseline finalizer 对纯 API 错误自动重试，协议/provenance 不一致立即失败。
