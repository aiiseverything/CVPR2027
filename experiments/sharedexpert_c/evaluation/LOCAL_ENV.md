# 当前服务器上的评测运行配置（2026-10-21）

本文件记录把冻结评测流程迁移到新服务器时**仅涉及运行环境**的调整。生成协议、prompt 预算、图像预算、评分规则与 `suite.meta.json` 均未修改。

## 环境差异

| 项目 | 原服务器（文档记录） | 当前服务器 |
|---|---|---|
| GPU | 多卡，评测使用 GPU 3 / 服务用 GPU 2 | 单卡 RTX 6000D，85 GB（sm_120） |
| CPU / 内存 | 8 核容器配额，64 GiB cgroup | 25 核，754 GiB |
| Python | `pre-lab/venv/bin/python` | `/root/.venv-eval/bin/python` |
| vLLM / Transformers / PyTorch | 0.30.0 / 5.18.0 / 2.13.0 | 0.30.0 / 5.18.0 / 2.13.0（固定同版本） |

评测虚拟环境放在**本地磁盘** `/root/.venv-eval`。最初装在工作区（AutoFS）时，vLLM 导入和 humming kernel 编译极慢（多次启动卡在 1–2 MB/s 的文件读取上），迁到本地后启动时间从 ~20 分钟降到 ~1 分钟（权重页缓存命中时）。所有命令都用该解释器启动；`run_checkpoint.py` 以 `sys.executable` 派生 vLLM 服务进程。

## 代码调整（不改变评测协议）

- `common.py`：新增 `MODEL_NAMES`，加入 `perception-step100`、`reasoning-step100`；`PROTOCOL` / `SCORING_PROTOCOL` 未改。
- `run_api.py`、`run_checkpoint.py`：`--name` 接受上述名称；`run_checkpoint.py` 改用 `sys.executable`，不再写死 `pre-lab/venv`。
- `local_server.json`：只放环境相关的 vLLM 参数（显存比例、并发序列数、dtype、环境变量），由 `run_checkpoint.py --server-config` 读取。
- `summarize.py`：模型列由 `MODEL_NAMES` 生成，未改任何统计口径。
- `stage_checkpoint.py`：新增。把训练导出的 tar 展开为完整 HF 目录并校验 tar 未截断；归档内已有的 tokenizer/processor 文件优先，缺失时才从基座补齐。

## 数据重建（已按冻结哈希验证）

冻结清单本身不随仓库提交（`.gitignore` 忽略 `*.jsonl` / `assets/`）。本机重建结果：

- 六个 VLMEvalKit TSV 从 OpenXLab OSS 通过 HTTP 下载（当前网络 HTTPS 不可用），逐个与 `suite.meta.json` 的 `source_sha256` 完全一致。
- RefCOCO heldout 由固定 revision `jxu124/refcoco@9fba820…` 的 parquet 重建：按 `prepare_refcoco.py` 相同的随机序列（先 shuffle 42,404 行的 train image ids，再 shuffle val）取 500 条，得到 **439 张图像**，与冻结文档一致。
- 在 `/home/ml-user/workdir/CVPR2027` 建立指向 `/autodl-fs/data/CVPR2027` 的符号链接，并把 `suite.jsonl` 的资产前缀改写回原服务器路径，使行内容可与冻结哈希逐字节比较。
- 重建时发现：冻结套件是从 v2 迁移到 v3 的，v2 阶段排除的 `MMMU_DEV_VAL:6635`（4 图，v2 计数 2526）被原样保留；按 v3 规则重算为 1563 token，会重新纳入。去掉该行后得到 **3423 行**，`sha256 = 1494d14bab22fa73ed92f6fffd9df6a48bfc1268db8c56fe29a5f3e7f74f9100`，与 `suite.meta.json` 完全一致；全部图像文件哈希复核通过。评测沿用冻结套件的 3423 行口径。

## 检查点

- 完整 tar：`experiments/sharedexpert_c/models/Qwen3.5-4B-perception-step100.tar`（10,370,037,760 字节，含 config、generation_config、chat_template、model.safetensors、processor_config、tokenizer）。
- 已展开到 `experiments/sharedexpert_c/models/Qwen3.5-4B-perception-step100`；`model.safetensors` 头部声明 724 个张量，数据区完整。
- 截断的旧文件保留为 `Qwen3.5-4B-perception-step100.tar.old-20261021`（7.81 GB），不要使用。
- 注意：训练导出 `config.json` 顶层 `dtype=float32`；服务端显式 `--dtype bfloat16`，与冻结协议一致。
- reasoning-step100：`Qwen3.5-4B-reasoning-step100.tar`（10,370,037,760 字节）sha256 校验通过，已展开到 `Qwen3.5-4B-reasoning-step100`；`model.safetensors` 为 10,350,019,328 字节、724 个张量。其 config / chat template / generation config / processor / tokenizer 与 perception-step100 逐字节一致，仅权重不同；抽样 13 条（含多图）重算 prompt_tokens 与冻结清单零偏差。

## 运行方式

```bash
cd /autodl-fs/data/CVPR2027

# 单卡全量评测（3423 题，客户端并发 4）
/root/.venv-eval/bin/python experiments/sharedexpert_c/evaluation/run_checkpoint.py \
  --model experiments/sharedexpert_c/models/Qwen3.5-4B-perception-step100 \
  --name perception-step100 --gpu 0 \
  --output-root experiments/sharedexpert_c/evaluation/results --concurrency 4

# reasoning-step100 全量评测（客户端并发与服务端 max_num_seqs 对齐为 16）
/root/.venv-eval/bin/python experiments/sharedexpert_c/evaluation/run_checkpoint.py \
  --model experiments/sharedexpert_c/models/Qwen3.5-4B-reasoning-step100 \
  --name reasoning-step100 --gpu 0 \
  --output-root experiments/sharedexpert_c/evaluation/results --concurrency 16

# 汇总（baseline 缓存不在本机，表中 baseline 显示 not_started）
/root/.venv-eval/bin/python experiments/sharedexpert_c/evaluation/summarize.py
```

已完成的 8 题 smoke（单独写入 `evaluation/results_smoke/`，不进最终结果）：0 API 错误，prompt token 与冻结逐题完全一致，1 条 MathVista 在 8192 token 截断，按冻结评分规则记零。

## 正式评测结果（2026-10-21）

全量 3423 题完成，缺失 0、API 错误 0；302 条 `finish_reason=length` 全部按协议记零（抽取失败数与截断数相同）。结果目录 `evaluation/results/full/perception-step100/`，汇总在 `evaluation/results/summary.{md,json}`。

| 维度 | perception-step100 | baseline 参考值（文档） | 差值 |
|---|---:|---:|---:|
| reasoning | 61.96 | 54.82 | +7.14 |
| perception | 93.50 | 92.36 | +1.14 |
| general | 79.40 | 76.40 | +3.00 |

逐基准：MathVista 78.00、MathVision 44.41、MMMU 63.47、MME perception 子集 95.00、RefCOCO 92.00、MMStar 71.60、MMBench 循环组 87.20。

注意：baseline 本机没有缓存，参考值来自冻结文档（相同 suite/protocol），不是本机重跑，因而没有配对 bootstrap 差值；step100 是中间检查点，不是 step300 教师。运行中按用户要求在保留前 1431 条后把客户端并发由 4 改为 16（服务端 `max_num_seqs=16`）；temperature=0 的逐题请求不变，但并发不属于冻结生成参数，报告时应保留这一环境说明。

## 正式评测结果：reasoning-step100（2026-10-22）

全量 3423 题完成，缺失 0、API 错误 0；229 条 `finish_reason=length` 全部按协议记零，抽取失败 225 条。结果目录 `evaluation/results/full/reasoning-step100/`。客户端并发全程 16（服务端 `max_num_seqs=16`），单卡 GPU 0；与 perception-step100 使用完全相同的 suite / protocol / 评分器。

| 维度 | reasoning-step100 | perception-step100 | 差值 |
|---|---:|---:|---:|
| reasoning | 66.09 | 61.96 | +4.13 |
| perception | 93.50 | 93.50 | 0.00 |
| general | 81.10 | 79.40 | +1.70 |

逐基准（括号为对 perception-step100 的聚类 bootstrap 配对差值 95% 区间）：

| 评测集 | reasoning-step100 | perception-step100 | 差值 |
|---|---:|---:|---:|
| MathVista_MINI | 81.20 | 78.00 | +3.20 (+0.39 ~ +6.20) |
| MathVision_MINI | 48.68 | 44.41 | +4.28 (−0.66 ~ +9.21) |
| MMMU_DEV_VAL | 68.37 | 63.47 | +4.90 (+1.34 ~ +8.46) |
| MME_perception | 95.00 | 95.00 | 0.00 (−2.44 ~ +2.40) |
| RefCOCO_heldout | 92.00 | 92.00 | 0.00 (−1.96 ~ +1.83) |
| MMStar | 73.80 | 71.60 | +2.20 (−0.60 ~ +5.20) |
| MMBench_DEV_EN_V11 | 88.40 | 87.20 | +1.20 (−1.60 ~ +4.00) |

全 3423 行等权配对差值为 **+2.78（95% CI +1.21 ~ +4.35）**。与 specialist 身份一致：reasoning 维度提升最大，perception 维度保持不变；MMMU、MathVista 的配对差值区间不含 0。截断与抽取失败均少于 perception-step100（229 vs 302、225 vs 302），主要差异集中在 MathVision/MMMU 的长思考题。baseline 仍无本机缓存，以上只做两个 step100 中间检查点之间的同协议配对比较。
