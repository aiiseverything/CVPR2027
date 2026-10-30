#!/usr/bin/env bash
# 建独立 venv 装 vLLM（不动 /opt/venv/main 里现有的 torch 2.4.1）
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
VENV="$ROOT/venv"
PIP="pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo "[$(date '+%T')] 创建 venv $VENV"
python3 -m venv "$VENV" || { echo "✗ venv 创建失败"; exit 1; }
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "[$(date '+%T')] 安装 vllm (含配套 torch/transformers)"
$PIP vllm 2>&1 | tail -20

echo "[$(date '+%T')] 安装 qwen-vl-utils / pillow / datasets"
$PIP qwen-vl-utils pillow datasets 2>&1 | tail -10

echo "[$(date '+%T')] 版本确认"
python -c "
import torch, transformers, vllm
print('  torch       ', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  vllm        ', vllm.__version__)
print('  GPU 数      ', torch.cuda.device_count())
"
echo "[$(date '+%T')] 完成"
