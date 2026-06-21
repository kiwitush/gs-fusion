#!/usr/bin/env bash
# 物体A: 多视角重建（lego 小车 → COLMAP + 2DGS）
#
# 用法: bash scripts/run_object_a.sh [图像目录] [输出目录] [最大分辨率]
# 示例: bash scripts/run_object_a.sh data/object_a/lego outputs/object_a 1200
set -euo pipefail

IMAGE_DIR="${1:-data/object_a/lego}"
OUTPUT_DIR="${2:-outputs/object_a}"
MAX_SIZE="${3:-1200}"

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "错误: 图像目录不存在: ${IMAGE_DIR}"
    exit 1
fi

echo "物体A 多视角重建 | 图像: ${IMAGE_DIR} | 输出: ${OUTPUT_DIR} | 最大边长: ${MAX_SIZE}"

# 清理旧的 COLMAP 数据，避免和新图冲突
rm -f "${OUTPUT_DIR}/database.db"
rm -rf "${OUTPUT_DIR}/sparse"
rm -rf "${OUTPUT_DIR}/images"

# --- 可选: 降分辨率（如果原图太大，COLMAP 会非常慢） ---
NEEDS_RESIZE=false
for img in "${IMAGE_DIR}"/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
    [ -f "$img" ] || continue
    size=$(python3 -c "from PIL import Image; s=Image.open('${img}').size; print(max(s))" 2>/dev/null || echo 0)
    if [ "$size" -gt "${MAX_SIZE}" ] 2>/dev/null; then
        NEEDS_RESIZE=true
        break
    fi
done

if $NEEDS_RESIZE; then
    echo "[0] 降分辨率至 ${MAX_SIZE}px（原图过大）..."
    RESIZED_DIR="${OUTPUT_DIR}/images_resized"
    mkdir -p "${RESIZED_DIR}"
    python3 -c "
import os
from PIL import Image
from pathlib import Path
max_size = ${MAX_SIZE}
src = Path('${IMAGE_DIR}')
dst = Path('${RESIZED_DIR}')
for f in sorted(src.iterdir()):
    if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        continue
    img = Image.open(f).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    img.save(dst / f.name, quality=95)
    print(f'  {f.name}: {w}x{h} -> {img.size[0]}x{img.size[1]}')
print(f'完成，已保存至 {dst}')
"
    IMAGE_DIR="${RESIZED_DIR}"
fi

COLMAP_OUT="${OUTPUT_DIR}/sparse"
mkdir -p "${OUTPUT_DIR}" "${COLMAP_OUT}"

# 将图像重命名为补零格式，使字母序 = 数值序，确保 sequential_matcher 正确匹配相邻帧
echo "[*] 重命名图像为零填充格式（以确保时序匹配正确）..."
python3 -c "
import re
from pathlib import Path
d = Path('${IMAGE_DIR}')
pattern = re.compile(r'^(.+?)(\d+)(\.\w+)$')
pairs, max_digits = [], 0
for f in d.iterdir():
    m = pattern.match(f.name)
    if m:
        n = int(m.group(2))
        pairs.append((f, m.group(1), n, m.group(3)))
        max_digits = max(max_digits, len(str(n)))
if max_digits > 0:
    for f, prefix, n, ext in pairs:
        new = d / f'{prefix}{n:0{max_digits}d}{ext}'
        if new != f:
            f.rename(new)
    print(f'  已重命名 {len(pairs)} 个文件 ({max_digits} 位补零)')
"

echo "[1/6] COLMAP 特征提取 ..."
colmap feature_extractor \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model PINHOLE

echo "[2/6] 时序特征匹配（前后各5帧）..."
colmap sequential_matcher \
    --database_path "${OUTPUT_DIR}/database.db" \
    --SequentialMatching.overlap 5

echo "[3/6] 稀疏重建 ..."
colmap mapper \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${COLMAP_OUT}"

echo "[4/6] 准备训练图像 ..."
cp -r "${IMAGE_DIR}" "${OUTPUT_DIR}/images"

echo "[5/6] 2DGS 训练 ..."
python src/reconstruction/train.py \
    --source_path "${OUTPUT_DIR}" \
    --model_path "${OUTPUT_DIR}" \
    --iterations 30000 \
    --max_sh_degree 3 \
    --sh_degree_interval 1000 \
    --densify_grad_threshold 2e-5 \
    --opacity_reset_interval 2000

echo "[6/6] 导出高斯 PLY ..."
BEST_CHECKPOINT="${OUTPUT_DIR}/best.pth"
if [ -f "${BEST_CHECKPOINT}" ]; then
    CKPT="${BEST_CHECKPOINT}"
    echo "  使用最优模型(best.pth)"
else
    CKPT="${OUTPUT_DIR}/chkpnt_final.pth"
    echo "  使用最终模型(chkpnt_final.pth)"
fi
python src/reconstruction/export.py \
    --checkpoint "${CKPT}" \
    --output "${OUTPUT_DIR}/object_a.ply"

echo "完成。高斯 PLY: ${OUTPUT_DIR}/object_a.ply"
