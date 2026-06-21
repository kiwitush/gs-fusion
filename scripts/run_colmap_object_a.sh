#!/usr/bin/env bash
# 物体A: 多视角重建（COLMAP + 2DGS）
#
# 用法: bash scripts/run_colmap_object_a.sh <图像目录> <输出目录>
# 前置: COLMAP 已安装，相邻图像重叠 ≥ 70%
set -euo pipefail

IMAGE_DIR="${1:?用法: $0 <图像目录> <输出目录>}"
OUTPUT_DIR="${2:?用法: $0 <图像目录> <输出目录>}"
mkdir -p "${OUTPUT_DIR}/sparse"

echo "物体A 多视角重建 | 图像: ${IMAGE_DIR} | 输出: ${OUTPUT_DIR}"

echo "[1/6] 提取特征 ..."
colmap feature_extractor \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model PINHOLE

echo "[2/6] 穷举特征匹配 ..."
colmap exhaustive_matcher \
    --database_path "${OUTPUT_DIR}/database.db"

echo "[3/6] 稀疏重建 ..."
colmap mapper \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${OUTPUT_DIR}/sparse"

echo "[4/6] 准备训练图像 ..."
if [ ! -d "${OUTPUT_DIR}/images" ]; then
    cp -r "${IMAGE_DIR}" "${OUTPUT_DIR}/images"
fi

echo "[5/6] 2DGS 训练 ..."
python src/reconstruction/train.py \
    --source_path "${OUTPUT_DIR}" \
    --model_path "${OUTPUT_DIR}" \
    --iterations 15000

echo "[6/6] 导出高斯 PLY ..."
python src/reconstruction/export.py \
    --checkpoint "${OUTPUT_DIR}/chkpnt_final.pth" \
    --output "${OUTPUT_DIR}/object_a.ply"

echo "完成。高斯 PLY: ${OUTPUT_DIR}/object_a.ply"
