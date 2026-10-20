#!/usr/bin/env bash
# 等待下载完成 -> 校验 -> 解压 Vision-OPD-6K -> 转 parquet
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
cd "$ROOT"

# 等待具体的下载脚本 PID，而不是用 pgrep 匹配进程名 —— 后者会在
# 两次 hf download 交接的空档（前一个已退出、后一个未启动）误判为「全部完成」。
if [ $# -ge 1 ]; then
    echo "[$(date '+%F %T')] 等待 PID: $*"
    for pid in "$@"; do
        while kill -0 "$pid" 2>/dev/null; do sleep 15; done
        echo "[$(date '+%F %T')] PID $pid 已结束"
    done
else
    echo "[$(date '+%F %T')] 未指定 PID，回退到进程名匹配"
    while pgrep -f "hf download (Qwen/Qwen3.5-4B|Qwen/Qwen3.5-9B|yuanqianhao/Vision-OPD-6K)" >/dev/null 2>&1; do
        sleep 30
    done
fi
echo "[$(date '+%F %T')] 下载进程已结束"

echo
echo "========== 下载日志尾部 =========="
grep -vE "FutureWarning|warnings.warn|Still waiting" logs/download_models.log | tail -4
grep -vE "FutureWarning|warnings.warn|Still waiting" logs/download_vision_opd.log | tail -4

echo
echo "========== 模型校验 =========="
for m in Qwen3.5-4B Qwen3.5-9B; do
    d="models/$m"
    echo "--- $m ---"
    if [ -f "$d/model.safetensors.index.json" ] || ls "$d"/model.safetensors* >/dev/null 2>&1; then
        ls -la "$d" | grep -E "safetensors|json|jinja" | awk '{printf "  %10s  %s\n", $5, $9}'
        echo "  大小: $(du -sh "$d" | cut -f1)"
    else
        echo "  ✗ 权重缺失"
    fi
done

echo
echo "========== Vision-OPD-6K 解压 =========="
D="$ROOT/data/Vision-OPD-6K"
if [ ! -f "$D/train.jsonl" ]; then
    echo "✗ train.jsonl 缺失，跳过解压"; exit 1
fi
echo "train.jsonl 行数: $(wc -l < "$D/train.jsonl")"

# 学生图: 6 个分片按序合并后解压（tar 内部为 ./<name>.png 扁平结构）
if [ -d "$D/images" ] && ls "$D/images"/images.tar.gz* >/dev/null 2>&1; then
    echo "解压 images ..."
    # 必须带 -z: 从管道读时 tar 无法自动识别 gzip(仅从文件读才行)
    ( cd "$D/images" && cat images.tar.gz* | tar -xzf - -C . ) && echo "  ✓ 完成"
else
    echo "  (images 已解压或分片不存在)"
fi

if [ -f "$D/teacher_images/teacher_images.tar.gz" ]; then
    echo "解压 teacher_images ..."
    ( cd "$D/teacher_images" && tar -xf teacher_images.tar.gz -C . ) && echo "  ✓ 完成"
else
    echo "  (teacher_images 已解压或不存在)"
fi

if [ -f "$D/original_images/original_images.tar.gz" ]; then
    echo "解压 original_images ..."
    ( cd "$D/original_images" && tar -xf original_images.tar.gz -C . ) && echo "  ✓ 完成"
else
    echo "  (original_images 已解压或不存在)"
fi

echo
echo "--- 解压后各目录图片数 ---"
for sub in images teacher_images original_images; do
    printf "  %-18s %s 张\n" "$sub" "$(find "$D/$sub" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)"
done

echo
echo "========== 转 parquet =========="
cd "$ROOT"
python3 make_parquet.py --data-dir "$D" 2>&1 | tail -6

echo
echo "========== 最终结果 =========="
ls -la "$D"/train.parquet 2>/dev/null
du -sh "$D" models 2>/dev/null
echo "[$(date '+%F %T')] 全部完成"
