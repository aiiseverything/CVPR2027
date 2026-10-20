# SharedExpert 三教师统一评测

固定协议：thinking=True，temperature=0，response≤8192；实际多模态 prompt≤2048。超过 prompt 预算的样本对四模型统一排除。finish_reason=length 一律记零，即使截断前已有完整正确 final；四模型统一。

本表为本地严格抽取及匹配协议，与既有 32768-token / DeepSeek judge 的历史成绩不能直接比较。

| 模型 | 状态 | 缺失 | API 错误 | 抽取失败 | 长度截断 |
|---|---|---:|---:|---:|---:|
| baseline | complete | 0 | 0 | 531 | 530 |
| generalist | not_started | 3423 | — | — | — |
| perception | not_started | 3423 | — | — | — |
| reasoning | not_started | 3423 | — | — | — |

只报告完整模型的成绩；准确率单位为百分比，括号为按图像／循环选项组 bootstrap 的 95% 区间。

| 评测集 | 实际行数 | baseline | generalist | perception | reasoning |
|---|---:|---:|---:|---:|---:|
| MMBench_DEV_EN_V11 | 930 | 85.20 (80.80–89.60) | — | — | — |
| MME_perception | 240 | 92.92 (89.43–96.15) | — | — | — |
| MMMU_DEV_VAL | 449 | 57.02 (52.34–61.47) | — | — | — |
| MMStar | 500 | 67.60 (63.60–71.80) | — | — | — |
| MathVision_MINI | 304 | 35.86 (30.59–41.45) | — | — | — |
| MathVista_MINI | 500 | 71.60 (68.00–75.40) | — | — | — |
| RefCOCO_heldout | 500 | 91.80 (89.32–94.15) | — | — | — |

MMBench 使用所选 base question 的全部循环选项版本，主指标为一组全对；其余是严格答案准确率，RefCOCO 为 IoU≥0.5。MME 仅 count/position/color/existence 子集，非完整 MME 总分。

| 维度（各评测集等权） | baseline | generalist | perception | reasoning |
|---|---:|---:|---:|---:|
| reasoning | 54.82 | — | — | — |
| perception | 92.36 | — | — | — |
| general | 76.40 | — | — | — |

固定样本总数：3423；预先排除条目：1；grounding 包含状态：True。

逐题结果与配对差值详见 summary.json / per_sample.csv；未完成的任务不产生最终准确率。
