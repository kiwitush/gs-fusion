#!/usr/bin/env bash
# 背景场景 2DGS 训练
#
# 用法: bash scripts/run_background.sh <场景路径> [场景名称]
# 示例: bash scripts/run_background.sh data/mipnerf360/garden garden

set -euo pipefail

SCENE_PATH="${1:?用法: $0 <场景路径> [场景名称]}"
SCENE_NAME="${2:-$(basename "${SCENE_PATH}")}"
MODEL_PATH="outputs/background/${SCENE_NAME}"

echo "2DGS 背景重建 | 场景: ${SCENE_NAME} | 数据: ${SCENE_PATH} | 输出: ${MODEL_PATH}"

python src/reconstruction/train.py \
    --source_path "${SCENE_PATH}" \
    --model_path "${MODEL_PATH}" \
    --iterations 30000 \
    --lambda_dssim 0.2

python src/reconstruction/export.py \
    --checkpoint "${MODEL_PATH}/chkpnt_final.pth" \
    --output "${MODEL_PATH}/point_cloud_final.ply"

echo "完成。模型: ${MODEL_PATH}/point_cloud_final.ply"
