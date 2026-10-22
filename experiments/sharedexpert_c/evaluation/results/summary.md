# SharedExpert 三教师统一评测（主套件 + BLINK/MMVP）

两个套件使用完全相同的冻结协议：thinking=True，temperature=0，response≤8192；多模态 prompt≤2048；finish_reason=length 一律记零。主套件 3423 行，BLINK/MMVP 增量 2201 行（BLINK val 1901、MMVP 300），合计 5624 行。

| 模型 | 主套件 3423 | BLINK/MMVP 2201 | 缺失 | API 错误 | 抽取失败 | 长度截断 |
|---|---|---|---:|---:|---:|---:|
| baseline | not_started | complete (2201/2201) | 3423 | 0 | 454 | 455 |
| generalist | not_started | not_started | 5624 | 0 | 0 | 0 |
| perception | not_started | not_started | 5624 | 0 | 0 | 0 |
| reasoning | not_started | not_started | 5624 | 0 | 0 | 0 |
| perception-step100 | complete (3423/3423) | complete (2201/2201) | 0 | 0 | 489 | 489 |
| reasoning-step100 | complete (3423/3423) | complete (2201/2201) | 0 | 0 | 330 | 334 |

只报告该基准所有行都成功预测的模型；baseline 本机只跑了 BLINK/MMVP，主套件旧基准列保持 —。

| 评测集 | 行数 | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLINK_val | 1901 | 59.18 (56.97–61.39) | — | — | — | 66.86 (64.81–69.02) | 67.54 (65.33–69.59) |
| MMBench_DEV_EN_V11 | 930 | — | — | — | — | 87.20 (82.80–91.20) | 88.40 (84.40–92.40) |
| MME_perception | 240 | — | — | — | — | 95.00 (91.52–98.31) | 95.00 (91.59–97.90) |
| MMMU_DEV_VAL | 449 | — | — | — | — | 63.47 (59.01–67.93) | 68.37 (63.92–72.83) |
| MMStar | 500 | — | — | — | — | 71.60 (67.40–75.40) | 73.80 (70.00–77.60) |
| MMVP | 300 | 79.00 (74.00–83.67) | — | — | — | 81.33 (76.67–85.67) | 80.33 (75.67–84.67) |
| MathVision_MINI | 304 | — | — | — | — | 44.41 (38.82–50.00) | 48.68 (42.76–54.28) |
| MathVista_MINI | 500 | — | — | — | — | 78.00 (74.40–81.40) | 81.20 (77.80–84.80) |
| RefCOCO_heldout | 500 | — | — | — | — | 92.00 (89.56–94.47) | 92.00 (89.61–94.33) |

MMBench 使用循环选项组全对；BLINK/MMVP 为严格 MCQ；RefCOCO 为 IoU≥0.5；其余为严格答案准确率。

| 维度宏平均 | 基准集 | baseline | generalist | perception | reasoning | perception-step100 | reasoning-step100 |
|---|---|---:|---:|---:|---:|---:|---:|
| reasoning | MMMU_DEV_VAL, MathVision_MINI, MathVista_MINI | — | — | — | — | 61.96 | 66.09 |
| perception | BLINK_val, MME_perception, MMVP, RefCOCO_heldout | 69.09 | — | — | — | 83.80 | 83.72 |
| general | MMBench_DEV_EN_V11, MMStar | — | — | — | — | 79.40 | 81.10 |
| perception（共同基准） | BLINK_val, MMVP | 69.09 | — | — | — | 74.10 | 73.94 |

维度宏平均是该模型已完整覆盖基准的等权均值；baseline 的 perception 宏平均只含 BLINK/MMVP，与 step100 的四项 perception 不可直接比较，因此另列“共同基准”行。

对 baseline 的配对差值与 step100 之间的配对差值见 summary.json。

固定样本总数：5624；协议与评分器 sha256 见 summary.json。
逐题结果详见 per_sample.csv；未完整覆盖的基准不产生准确率。
