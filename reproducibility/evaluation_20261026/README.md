# 教师评测与诊断资料快照

本目录保存已完成的五个 Qwen3.5-4B 模型评测、推理教师感知表现诊断，以及新高分辨率感知评测的冻结清单。目录日期按上海时间命名。

**本说明的状态截点：2026-10-25 19:20:36 UTC / 2026-10-26 03:20:36 Asia/Shanghai。** 旧评测已经完成；高分辨率评测当时仍在 CPU 准备阶段，尚无 GPU 评测成绩。这是静态归档，不是运行状态接口。[status_snapshot.json](highres/status_snapshot.json) 记录其自身采集时间和精简状态。

## 已完成的五模型结果

五个 4B 模型均完成 **7,162 题**，包括主套件 3,423 题、BLINK＋MMVP 2,201 题、IFEval＋OCRBench 1,538 题；每模型缺失数和 API 错误数均为 **0**。截断或解析失败按评分协议处理，不等于 API 错误。本表不混入仅完成部分套件的 9B 模型或旧名称占位记录。

数值为百分数，保留两位小数；更精确值、置信区间及评分身份见[汇总 JSON](completed/summary.json) 和[完整分数 CSV](completed/all_model_scores.csv)。

| 评测项 | 行数 | 基座 | 通用教师 | 感知 v1 | 推理教师 | 感知 v2 |
|---|---:|---:|---:|---:|---:|---:|
| BLINK_val | 1,901 | 59.18 | 67.18 | 66.86 | 67.54 | 62.49 |
| IFEval_strict | 541 | 39.19 | 64.14 | 56.01 | 66.17 | 53.79 |
| MMBench_DEV_EN_V11 | 930 | 85.20 | 88.00 | 87.20 | 88.40 | 86.40 |
| MME_perception | 240 | 92.92 | 95.42 | 95.00 | 95.00 | 95.00 |
| MMMU_DEV_VAL | 449 | 57.02 | 67.26 | 63.47 | 68.37 | 60.13 |
| MMStar | 500 | 67.60 | 72.40 | 71.60 | 73.80 | 68.80 |
| MMVP | 300 | 79.00 | 81.67 | 81.33 | 80.33 | 82.00 |
| MathVision_MINI | 304 | 35.86 | 45.07 | 44.41 | 48.68 | 35.53 |
| MathVista_MINI | 500 | 71.60 | 79.60 | 78.00 | 81.20 | 75.40 |
| OCRBench_heldout | 997 | 81.54 | 82.95 | 83.75 | 83.55 | 83.25 |
| RefCOCO_heldout | 500 | 91.80 | 91.20 | 92.00 | 92.00 | 93.00 |

模型名依次对应 `baseline`、`generalist`、`perception-step100`、`reasoning-step100`、`perception-v2-step100`。

各评测的统计单位不同，不能把上表所有百分数平均后称为官方总分。MMBench 使用循环组全部答对的准确率，930 是行数；RefCOCO 使用 IoU≥0.5 的准确率。MME 是 count／color／existence／position 的 240 行子集，不是完整官方 MME。MMVP 沿用本地逐题准确率，**不等同于官方成对题全部答对的指标**。OCRBench 是过滤后的 997 行、规范化 substring 评分，不是官方 1,000 分总分。IFEval 主表使用项目的 strict prompt accuracy；补充敏感性口径见 [ifeval_sensitivity.json](completed/ifeval_sensitivity.json)。

旧评测开启 thinking、输出上限为 8,192 tokens，`finish_reason=length` 按冻结严格协议计零，不追加第二次作答。旧图像预算单图最多 786,432 pixels，多图总预算 1,572,864 pixels；这与新高分辨率协议不同。

## 推理教师的感知表现：观察与边界

[diagnostic/analysis.json](diagnostic/analysis.json) 保存输入 SHA、逐基准配对胜负、长度统计、2,000 次配对 bootstrap（seed=20261019）和类别分析。[comparison.csv](diagnostic/comparison.csv) 提供便于查看的比较表；[analyze.py](diagnostic/analyze.py) 是对应分析脚本，复算需要本地保留或重新生成逐题预测，归档不含这些原始输出。

在 BLINK 上，推理教师比基座高 **8.36 个百分点**，配对 95% 区间为 **[6.58, 10.21]**；比感知 v1 高 **0.68 个百分点**，区间为 **[−1.21, 2.47]**，因此这组数据不足以证明它优于感知 v1。推理教师在 MME 和 RefCOCO 上与感知 v1 同分，OCRBench 略低（83.55% 对 83.75%），也不能据此称其全面优于感知教师。

BLINK 的推理教师相对基座净多答对 **159 题**。按两模型截断状态分解，截断状态不同的行合计净差为 **+164 题**，双方均未截断的行净差为 **−5 题**。这是观察到的分数差的**算术分解，不是截断减少造成提升的因果贡献**。共同未截断／共同正常结束的子集是在模型输出之后筛选的，不能当作控制推理预算后的因果消融，也不能推断取消输出上限后谁会更好。多项探索性区间没有做多重比较校正。

[training_audit.json](diagnostic/training_audit.json) 说明了训练数据构成与来源证明的边界：重建数据的类别和确定性构建记录可以审计，但迁移后的 parquet 字节哈希不同，不能声称已证明其与不可得的原始训练文件逐字节相同。[exact_overlap_audit.json](diagnostic/exact_overlap_audit.json) 记录精确匹配检查；这类检查不排除裁剪、重新编码、语义近重复或基座预训练暴露。现有结果支持讨论任务表现和输出稳定性，不能单独识别能力迁移的机制。

## 新高分辨率评测：仅清单与准备状态

新协议使用原图、最高 33,554,432 pixels 的处理器面积预算、40,960 输入 tokens 与 8,192 输出 tokens。计划评测基座、推理教师、感知 v1、感知 v2 四个模型；本轮不包含通用教师。

| 固定子集 | 每模型题数 |
|---|---:|
| VStar 直接属性 | 115 |
| ZoomBench 文本审阅选择 | 807 |
| HR-Bench4K 单实例任务 | 400 |
| HR-Bench8K 单实例任务 | 400 |
| MME-RealWorld EN 感知、排除运动属性 | 19,129 |
| MME-RealWorld CN 感知、排除运动属性 | 3,610 |
| **合计** | **24,461** |

四模型总计 **97,844 次计划推理**。状态截点时 87 个源文件已下载并完成 SHA 校验，CPU 准备进度为 78 个分片；尚未开始模型评测。95% 显存、最多 64 序列、32,768 batch tokens 和按实测 KV 容量准入是待正式执行的运行设置，不能作为已经验证的速度或成绩。

“仅感知”是项目的范围选择，不是官方统一划分。Zoom 依据题干和选项做 AI 文本审阅，没有逐图人工验证；MME 的官方 Perception 类别也不保证零推理负担。HR-Bench 的循环版本不是独立原题。11 条已确认上游标注异常保留原 gold 和主分母，另报排除这些行的敏感性结果。

复现命令、环境、评分和恢复说明见[高分辨率代码 README](../../experiments/sharedexpert_c/evaluation/highres/README.md)。CPU 配置和数据检查完成后直接执行正式评测，不另行安排 GPU smoke。

## 文件索引与复现范围

| 路径 | 内容 |
|---|---|
| [completed/summary.json](completed/summary.json) | 已完成评测的机器可读汇总 |
| [completed/all_model_scores.csv](completed/all_model_scores.csv) | 五模型各基准分数 |
| [completed/ifeval_sensitivity.json](completed/ifeval_sensitivity.json) | IFEval 补充评分口径 |
| [diagnostic/analyze.py](diagnostic/analyze.py) | 从冻结预测生成诊断统计的脚本 |
| [diagnostic/analysis.json](diagnostic/analysis.json) | 配对分析、分层统计和输入身份 |
| [diagnostic/comparison.csv](diagnostic/comparison.csv) | 诊断比较表 |
| [diagnostic/training_audit.json](diagnostic/training_audit.json) | 训练数据和来源审计边界 |
| [diagnostic/exact_overlap_audit.json](diagnostic/exact_overlap_audit.json) | 精确重叠检查记录 |
| [highres/source_manifest.json](highres/source_manifest.json) | 原始分片的固定源 revision、大小和 SHA |
| [highres/selection.json](highres/selection.json) | 冻结的筛选规则和 Zoom ID 名单 |
| [highres/zoom_review_summary.json](highres/zoom_review_summary.json) | 精简审阅记录与原审计 SHA |
| [highres/annotation_issues_summary.json](highres/annotation_issues_summary.json) | 已确认标注异常的精简摘要与原审计 SHA |
| [highres/status_snapshot.json](highres/status_snapshot.json) | 带采集时间的准备状态快照 |

两份 manifest／selection 按原文件逐字节归档；review／issues summary 是供审阅的派生文件，不能代替原始 audit 或伪装成其哈希。新服务器将 `source_manifest.json` 和 `selection.json` 复制到运行数据目录，再按固定来源下载数据即可复用已有选择；只有重做选择审阅时才需要 `finalize_selection.py` 的完整原始输入。

本目录不打包图片、parquet、SQLite、模型权重、完整逐题预测或包含原题／gold 的审计原件。因此它提供协议、来源身份、选择和统计证据；实际重跑仍需取得相应模型与数据，并遵守上游使用条件。代码中的少量异常匹配样例用于检测固定的上游标签问题，不构成整套评测数据发布。
