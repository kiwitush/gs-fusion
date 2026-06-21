#!/usr/bin/env bash
# 全链路执行：背景 → 物体A → 物体B → 物体C → 融合 → 视频
# 用法: bash scripts/full_pipeline.sh

set -euo pipefail

BG_SCENE_PATH="data/garden"
BG_SCENE_NAME="garden"

# 若 garden 场景不存在则自动下载
if [ ! -d "${BG_SCENE_PATH}/sparse" ] || [ ! -d "${BG_SCENE_PATH}/images" ]; then
    echo "[0/6] 下载 Mip-NeRF 360 garden 场景 ..."
    python scripts/download_garden.py --output data/mipnerf360
fi

OBJ_A_IMAGES="data/object_a/lego"
OBJ_A_OUTPUT="outputs/object_a"
OBJ_A_MAX_SIZE="1200"
OBJ_B_PROMPT="A single, standard white ceramic coffee mug with an intricate, yet simple, classic blue and white botanical and geometric pattern wrapping around its external surface, glossy glaze finish, simple 3D model, isolated on a seamless pure white background, three-quarter product shot"
OBJ_B_OUTPUT="outputs/object_b"
OBJ_C_IMAGE="data/object_c/photo.jpg"
OBJ_C_OUTPUT="outputs/object_c"
PLACEMENT_CONFIG="configs/placements.json"
FUSED_SCENE="outputs/fused_scene.ply"
VIDEO_OUTPUT="outputs/videos/flythrough.mp4"

echo "3d_fusion 全链路开始"

echo "[1/6] 背景场景重建 ..."
bash scripts/run_background.sh "${BG_SCENE_PATH}" "${BG_SCENE_NAME}"

echo "[2/6] 物体A: COLMAP + 2DGS ..."
bash scripts/run_object_a.sh "${OBJ_A_IMAGES}" "${OBJ_A_OUTPUT}" "${OBJ_A_MAX_SIZE}"

echo "[3/6] 物体B: 文本到3D (threestudio) ..."
bash scripts/run_object_b.sh "${OBJ_B_PROMPT}" "${OBJ_B_OUTPUT}"

echo "[4/6] 物体C: 单图到3D (Magic123) ..."
bash scripts/run_magic123_object_c.sh "${OBJ_C_IMAGE}" "${OBJ_C_OUTPUT}"

echo "[5/6] 场景融合 ..."
python src/fusion/fuse.py \
    --config "${PLACEMENT_CONFIG}" \
    --output "${FUSED_SCENE}"

echo "[6/6] 渲染漫游视频 ..."
python src/fusion/render_video.py \
    --scene "${FUSED_SCENE}" \
    --output "${VIDEO_OUTPUT}" \
    --trajectory spiral \
    --start_radius 4.0 \
    --end_radius 1.5 \
    --start_height 3.0 \
    --end_height 0.5 \
    --turns 2.5 \
    --num_frames 180 \
    --fps 30

echo "全链路完成。视频: ${VIDEO_OUTPUT}  场景: ${FUSED_SCENE}"
