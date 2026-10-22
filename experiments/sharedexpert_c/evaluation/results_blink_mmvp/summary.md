# SharedExpert 三教师统一评测

固定协议：thinking=True，temperature=0，response≤8192；实际多模态 prompt≤2048。超过 prompt 预算的样本对四模型统一排除。finish_reason=length 一律记零，即使截断前已有完整正确 final；四模型统一。

本表为本地严格抽取及匹配协议，与既有 32768-token / DeepSeek judge 的历史成绩不能直接比较。

| 模型 | 状态 | 缺失 | API 错误 | 抽取失败 | 长度截断 |
|---|---|---:|---:|---:|---:|
| baseline | complete | 0 | 0 | 454 | 455 |
| generalist | not_started | 2201 | — | — | — |
| perception | not_started | 2201 | — | — | — |
| reasoning | not_started | 2201 | — | — | — |
| perception-step100 | complete | 0 | 0 | 187 | 187 |
| reasoning-step100 | complete | 0 | 0 | 105 | 105 |

只报告完整模型的成绩；准确率单位为百分比，括号为按图像／循环选项组 bootstrap 的 95% 区间。

| 评测集 | 实际行数 | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLINK_val | 1901 | 59.18 (56.97–61.39) | — | — | — | 66.86 (64.81–69.02) | 67.54 (65.33–69.59) |
| MMVP | 300 | 79.00 (74.00–83.67) | — | — | — | 81.33 (76.67–85.67) | 80.33 (75.67–84.67) |

MMBench 使用所选 base question 的全部循环选项版本，主指标为一组全对；其余是严格答案准确率，RefCOCO 为 IoU≥0.5。MME 仅 count/position/color/existence 子集，非完整 MME 总分。

| 维度（各评测集等权） | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---:|---:|---:|---:|---:|---:|
| reasoning | nan | nan | nan | nan | nan | nan |
| perception | 69.09 | — | — | — | 74.10 | 73.94 |
| general | nan | nan | nan | nan | nan | nan |

固定样本总数：2201；预先排除条目：0；grounding 包含状态：False。

逐题结果与配对差值详见 summary.json / per_sample.csv；未完成的任务不产生最终准确率。
