# SharedExpert 三教师统一评测

固定协议：thinking=True，temperature=0，response≤8192；实际多模态 prompt≤2048。超过 prompt 预算的样本对四模型统一排除。finish_reason=length 一律记零，即使截断前已有完整正确 final；四模型统一。

本表为本地严格抽取及匹配协议，与既有 32768-token / DeepSeek judge 的历史成绩不能直接比较。

| 模型 | 状态 | 缺失 | API 错误 | 抽取失败 | 长度截断 |
|---|---|---:|---:|---:|---:|
| baseline | complete | 0 | 0 | 531 | 530 |
| generalist | not_started | 3423 | — | — | — |
| perception | not_started | 3423 | — | — | — |
| reasoning | not_started | 3423 | — | — | — |
| perception-step100 | complete | 0 | 0 | 302 | 302 |
| reasoning-step100 | complete | 0 | 0 | 225 | 229 |

只报告完整模型的成绩；准确率单位为百分比，括号为按图像／循环选项组 bootstrap 的 95% 区间。

| 评测集 | 实际行数 | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMBench_DEV_EN_V11 | 930 | 85.20 (80.80–89.60) | — | — | — | 87.20 (82.80–91.20) | 88.40 (84.40–92.40) |
| MME_perception | 240 | 92.92 (89.43–96.15) | — | — | — | 95.00 (91.52–98.31) | 95.00 (91.59–97.90) |
| MMMU_DEV_VAL | 449 | 57.02 (52.34–61.47) | — | — | — | 63.47 (59.01–67.93) | 68.37 (63.92–72.83) |
| MMStar | 500 | 67.60 (63.60–71.80) | — | — | — | 71.60 (67.40–75.40) | 73.80 (70.00–77.60) |
| MathVision_MINI | 304 | 35.86 (30.59–41.45) | — | — | — | 44.41 (38.82–50.00) | 48.68 (42.76–54.28) |
| MathVista_MINI | 500 | 71.60 (68.00–75.40) | — | — | — | 78.00 (74.40–81.40) | 81.20 (77.80–84.80) |
| RefCOCO_heldout | 500 | 91.80 (89.32–94.15) | — | — | — | 92.00 (89.56–94.47) | 92.00 (89.61–94.33) |

MMBench 使用所选 base question 的全部循环选项版本，主指标为一组全对；其余是严格答案准确率，RefCOCO 为 IoU≥0.5。MME 仅 count/position/color/existence 子集，非完整 MME 总分。

| 维度（各评测集等权） | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---:|---:|---:|---:|---:|---:|
| reasoning | 54.82 | — | — | — | 61.96 | 66.09 |
| perception | 92.36 | — | — | — | 93.50 | 93.50 |
| general | 76.40 | — | — | — | 79.40 | 81.10 |

固定样本总数：3423；预先排除条目：1；grounding 包含状态：True。

逐题结果与配对差值详见 summary.json / per_sample.csv；未完成的任务不产生最终准确率。
