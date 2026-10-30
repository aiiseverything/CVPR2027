#!/usr/bin/env bash
# 下载 Vision-OPD-6K (官方 yuanqianhao/Vision-OPD-6K, tar 布局)
# 用法:
#   ./download_vision_opd.sh            # 只下载
#   ./download_vision_opd.sh extract    # 下载后自动解压
set -uo pipefail

export HF_HOME=/home/ml-user/workdir/cache/hf
export HF_XET_HIGH_PERFORMANCE=1

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
DEST="$ROOT/data/Vision-OPD-6K"
mkdir -p "$DEST" "$ROOT/logs"

echo "[$(date '+%F %T')] 下载 yuanqianhao/Vision-OPD-6K -> data/Vision-OPD-6K (约 37.5 GB)"
hf download yuanqianhao/Vision-OPD-6K --repo-type dataset --local-dir "$DEST" || { echo "✗ 下载失败"; exit 1; }
echo "[$(date '+%F %T')] ✓ 下载完成"
du -sh "$DEST"

if [ "${1:-}" = "extract" ]; then
    echo "[$(date '+%F %T')] 开始解压"
    # 学生图: 6 个分片先合并再解压
    ( cd "$DEST/images" && cat images.tar.gz* | tar -xzf - -C ./ ) && echo "  ✓ images"
    ( cd "$DEST/teacher_images" && tar -xzf teacher_images.tar.gz -C ./ ) && echo "  ✓ teacher_images"
    ( cd "$DEST/original_images" && tar -xzf original_images.tar.gz -C ./ ) && echo "  ✓ original_images"
    echo "[$(date '+%F %T')] ✓ 解压完成"
    echo "--- 各目录文件数 ---"
    for d in images teacher_images original_images; do
        printf "  %-16s %s 张\n" "$d" "$(ls "$DEST/$d" 2>/dev/null | wc -l)"
    done
fi
echo "[$(date '+%F %T')] 全部结束"
