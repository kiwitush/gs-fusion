#!/usr/bin/env bash
# 物体B: 文本到3D（threestudio + SDS Loss）
#
# 用法: bash scripts/run_object_b.sh "<Prompt>" [输出目录]
# 示例: bash scripts/run_object_b.sh "a ceramic teapot, photorealistic" outputs/object_b
# 前置: threestudio 已克隆并 pip install -e
# 2D先验: runwayml/stable-diffusion-v1-5 (国内可直接下载，sd-2-1 有权限墙)

set -euo pipefail

PROMPT="${1:-A single, standard white ceramic coffee mug with an intricate, yet simple, classic blue and white botanical and geometric pattern wrapping around its external surface, glossy glaze finish, simple 3D model, isolated on a seamless pure white background, three-quarter product shot}"
OUTPUT_DIR="${2:-outputs/object_b}"

echo "物体B 文本到3D | Prompt: ${PROMPT} | 输出: ${OUTPUT_DIR}"

echo "[1/2] threestudio 生成 ..."
python src/generation/text_to_3d.py \
    --prompt "${PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --iters 10000

echo "[2/2] Mesh → 高斯面片 ..."
python src/utils/mesh_to_gs.py \
    -i "${OUTPUT_DIR}/model.obj" \
    -o "${OUTPUT_DIR}/object_b.ply" \
    --num_samples 100000

echo "完成。高斯 PLY: ${OUTPUT_DIR}/object_b.ply"
