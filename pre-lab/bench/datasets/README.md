# mathversemini —— MathVerse 的 1000 条固定子集

## 是什么

`MathVerse_MINI_1K.tsv` = 官方 `MathVerse_MINI` 的**前 1000 行**（测量顺序的前 1000 条）。

命名为 **mathversemini**，从 MathVerse 中单独切出，以便后续实验固定复用同一批样本、结果可比。

## 为什么是「前 1000 行」

VLMEvalKit 按 TSV 行序逐条处理，所以「测量顺序的前 1000 条」等价于「TSV 的前 1000 行」，
等价于 `index` 列 1–1000。

前 1000 行的构成恰好是 **200 道题 × 5 个 problem_version**，分布完全均衡：

| problem_version | 条数 |
|---|---|
| Text Dominant | 200 |
| Text Lite | 200 |
| Vision Intensive | 200 |
| Vision Dominant | 200 |
| Vision Only | 200 |

这是个干净的切法，各难度层级都覆盖到了。

## 文件与校验

| 项 | 值 |
|---|---|
| 行数 | 1000（原 `MathVerse_MINI` 为 3940） |
| index 范围 | 1 – 1000 |
| 大小 | 28.3 MB |
| MD5 | `44e050b46085b864a4c5abde2053f444` |

文件位置：
- `bench/datasets/MathVerse_MINI_1K.tsv` —— 归档副本（本文件同目录）
- `<LMUData>/MathVerse_MINI_1K.tsv` —— VLMEvalKit 实际读取的位置

## 怎么用（下次复现）

数据集已在 VLMEvalKit 注册为 `MathVerse_MINI_1K`
（见 `vlmevalkit/vlmeval/dataset/image_vqa.py` 中 `MathVerse` 类的 `DATASET_URL` / `DATASET_MD5`）。

```bash
# 先把 TSV 放到 LMUData（MD5 匹配则 VLMEvalKit 不会尝试下载）
export LMUData=/home/ml-user/workdir/CVPR2027/pre-lab/lmudata
cp bench/datasets/MathVerse_MINI_1K.tsv "$LMUData/"

# 然后正常下发
python vlmevalkit/run.py --model Qwen3.5-4B --model-class GPT4V \
  --base-url http://localhost:8110/v1 --key EMPTY \
  --data MathVerse_MINI_1K --work-dir ./vlmeval_out \
  --mode infer --api-nproc 28 --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20
```

> 注意：该子集是**按行号静态切分**的，不是随机抽样 —— 这样才可复现、可跨模型对齐。
> 若今后 `MathVerse_MINI` 源文件更新，前 1000 行的内容可能变化，但本目录的 TSV 已固化，
> 以本文件的 MD5 为准。

## 本次跑法的说明

2026-10-04 那次评测是按完整 `MathVerse_MINI`（3940 条）下发的，跑到 index 1000 时由
`bench/stop_mathverse_at_1k.sh` 自动停止（**没有重跑**），已完成的前 1000 条即为
mathversemini 的结果。判分时预测文件里也就只有这 1000 条。

生成方式：

```python
import csv
csv.field_size_limit(10**9)
rows = list(csv.reader(open('lmudata/MathVerse_MINI.tsv'), delimiter='\t'))
with open('lmudata/MathVerse_MINI_1K.tsv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(rows[0]); w.writerows(rows[1:1001])
```
