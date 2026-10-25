# 教师评测结果与高分辨率评测资料

五个 4B 模型均完成主套件 3,423 题、BLINK/MMVP 2,201 题、OCR/IF 1,538 题，每模型合计 7,162 题，缺失和 API 错误均为 0。9B 仅完成主套件，不进入下面的完整五模型比较。

| 基准 | 题数 | 基座 | 推理教师 | 感知 v1 | 感知 v2 | 通用教师 |
|---|---:|---:|---:|---:|---:|---:|
| BLINK_val | 1901 | 59.18 | 67.54 | 66.86 | 62.49 | 67.18 |
| IFEval_strict | 541 | 39.19 | 66.17 | 56.01 | 53.79 | 64.14 |
| MMBench_DEV_EN_V11 | 930 | 85.20 | 88.40 | 87.20 | 86.40 | 88.00 |
| MME_perception | 240 | 92.92 | 95.00 | 95.00 | 95.00 | 95.42 |
| MMMU_DEV_VAL | 449 | 57.02 | 68.37 | 63.47 | 60.13 | 67.26 |
| MMStar | 500 | 67.60 | 73.80 | 71.60 | 68.80 | 72.40 |
| MMVP | 300 | 79.00 | 80.33 | 81.33 | 82.00 | 81.67 |
| MathVision_MINI | 304 | 35.86 | 48.68 | 44.41 | 35.53 | 45.07 |
| MathVista_MINI | 500 | 71.60 | 81.20 | 78.00 | 75.40 | 79.60 |
| OCRBench_heldout | 997 | 81.54 | 83.55 | 83.75 | 83.25 | 82.95 |
| RefCOCO_heldout | 500 | 91.80 | 92.00 | 92.00 | 93.00 | 91.20 |

分数单位为百分比。MMBench 使用循环选项整组全对指标，RefCOCO 使用 IoU≥0.5；MME 是四项感知子集。其他指标及置信区间见 [完整汇总](completed/summary.json)；[CSV](completed/all_model_scores.csv) 适合进一步分析。

## 推理教师感知能力诊断

推理教师 BLINK 得分为 67.54%，基座为 59.18%。配对分析显示，改善主要伴随长度截断减少；这不能直接证明纯视觉表征增强。两者均正常结束的子集属于事后条件筛选，不是因果对照。推理教师相对感知 v1 的 BLINK 差值约 +0.68 个百分点，95% 区间约为 [-1.21, +2.47]。

统计与方法见 [analysis.json](diagnostic/analysis.json)、[比较表](diagnostic/comparison.csv)、[训练审计](diagnostic/training_audit.json)、[精确重叠审计](diagnostic/exact_overlap_audit.json) 和 [分析脚本](diagnostic/analyze.py)。脚本需要服务器上未打包的冻结套件与原始预测。精确重叠为零不排除语义或变换后的重复。

IFEval 对固定检查器的随机回退使用逐题确定种子；两道符号计数异常题保留在完整分数，并另报 [敏感性分析](completed/ifeval_sensitivity.json)。

## 高分辨率感知评测

四模型为基座、推理教师、感知 v1、感知 v2。六项选择为 V⋆ 115、Zoom 807、HR-Bench 4K 400、HR-Bench 8K 400、MME-RW EN 19,129、MME-RW CN 3,610；每模型 24,461 行，共 97,844 次推理。HR-Bench 每项为 100 个基本问题的四轮选项排列。

采用完整原图，上限 33,554,432 像素；输入上限 40,960 tokens，思考与答案上限 8,192 tokens。显存预算 95%，最多 64 个在途请求，每轮 32,768 tokens；实际并发受本次 KV 容量控制。

这些是单独冻结的感知子集，不能当作六项官方完整成绩，也不能与旧低分辨率套件合并。Zoom 为基于题干与选项的 AI 文本审阅子集，未进行图像人工标注；评分使用本地确定性抽取而非官方 LLM judge。11 条精确识别的源标签异常保留原文与 gold，另报排除这些条目的敏感性分析。

运行方式见 [高分辨率评测说明](../../experiments/sharedexpert_c/evaluation/highres/README.md)。本次发布的 [状态快照](highres/status_snapshot.json) 带采集时间，不是实时状态；采集时仍在准备数据，未提供 GPU 推理成绩。

## 文件索引

- [固定数据源与 SHA256](highres/source_manifest.json)
- [冻结筛选清单](highres/selection.json)
- [Zoom 逐题 ID 审查摘要](highres/zoom_review_summary.json)
- [源标签异常摘要](highres/annotation_issues_summary.json)
- [来源与发布副本校验](provenance.json)

本快照不打包模型、图片、Parquet、SQLite、完整套件题目、新的逐题模型输出或活动日志。审查摘要是派生文件，不冒充 selection.json 引用的原始审查文件；11 条异常原文只作为精确匹配代码夹具保留。
