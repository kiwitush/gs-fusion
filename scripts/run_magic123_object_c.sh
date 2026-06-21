#!/usr/bin/env bash
# 单图到3D —— 物体C（Magic123）
#
# 用法: bash scripts/run_magic123_object_c.sh <输入图像> <输出目录>
# 前置: Magic123 已克隆安装，rembg 已安装

set -euo pipefail

INPUT_IMAGE="${1:?用法: $0 <输入图像> <输出目录>}"
OUTPUT_DIR="${2:?用法: $0 <输入图像> <输出目录>}"

echo "Magic123 —— 物体C | 输入: ${INPUT_IMAGE} | 输出: ${OUTPUT_DIR}"

echo "[1/3] 预处理：去背景 & 中心裁剪 ..."
python -c "
from src.generation.image_to_3d import ImageTo3DGenerator
import os
gen = ImageTo3DGenerator(output_dir='${OUTPUT_DIR}')
os.makedirs('${OUTPUT_DIR}/processed', exist_ok=True)
ImageTo3DGenerator.remove_background('${INPUT_IMAGE}', '${OUTPUT_DIR}/processed/input_rgba.png')
ImageTo3DGenerator.center_crop_to_square('${OUTPUT_DIR}/processed/input_rgba.png', '${OUTPUT_DIR}/processed/input_square.png', size=512)
"

echo "[2/3] Magic123 重建 ..."
python -c "
from src.generation.image_to_3d import ImageTo3DGenerator
gen = ImageTo3DGenerator(output_dir='${OUTPUT_DIR}')
gen.generate('${OUTPUT_DIR}/processed/input_square.png', elevation=30.0)
"

echo "[3/3] Mesh → 高斯面片 ..."
MESH_PATH="${OUTPUT_DIR}/mesh/model.obj"
if [ -f "${MESH_PATH}" ]; then
    python src/utils/mesh_to_gs.py \
        -i "${MESH_PATH}" \
        -o "${OUTPUT_DIR}/object_c.ply" \
        --num_samples 100000
else
    echo "警告: 未找到 Mesh (${MESH_PATH})，请检查 Magic123 输出。"
fi

echo "完成。高斯 PLY: ${OUTPUT_DIR}/object_c.ply"
