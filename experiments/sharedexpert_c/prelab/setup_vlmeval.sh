#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
cd "$ROOT"
source venv/bin/activate
PIP="pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo "[$(date '+%T')] 安装 VLMEvalKit（--no-deps，避免它动已有的 torch/vllm/transformers）"
cd "$ROOT/vlmevalkit" && $PIP --no-deps -e . 2>&1 | tail -5
cd "$ROOT"

echo "[$(date '+%T')] 补齐缺失的运行时依赖"
$PIP pandas numpy pillow tqdm requests matplotlib seaborn openpyxl timm torchvision \
     decord word2number mathruler nltk openai 2>&1 | tail -8

echo "[$(date '+%T')] 验证"
python -c "
import torch, transformers, vllm
print('  torch        ', torch.__version__)
print('  transformers ', transformers.__version__)
print('  vllm         ', vllm.__version__)
import vlmeval
from vlmeval.dataset import SUPPORTED_DATASETS
need=['MME','MathVista_MINI','MathVerse_MINI','MMStar','MMBench_DEV_EN_V11','MathVision_MINI','MMMU_DEV_VAL']
for n in need:
    print(f'    {n:<22}', 'OK' if n in SUPPORTED_DATASETS else '✗ 不在支持列表')
from vlmeval.vlm import Qwen3VLChat
print('  Qwen3VLChat  OK')
" 2>&1 | tail -22
echo "[$(date '+%T')] 完成"
