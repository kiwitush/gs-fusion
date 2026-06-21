# 3d-fusion

**基于 2DGS 与 AIGC 的多源 3D 资产生成与真实场景融合**

本项目使用 2D Gaussian Splatting 重建真实世界背景场景，通过三种不同的 AIGC 技术路线
（多视角重建、文本到 3D、单图到 3D）生成虚拟物体，最终将它们统一为高斯面片格式并
融合到同一 3D 场景中进行多视角漫游渲染。


## 项目结构

```
gs-fusion/
├── data/                              # 原始数据集 & 拍摄图像
│   ├── mipnerf360/                    # Mip-NeRF 360 背景场景
│   ├── object_a/images/               # 物体A 多视角照片
│   ├── object_b/                      # 物体B 文本 Prompt（运行时指定）
│   └── object_c/photo.jpg             # 物体C 单张照片
├── outputs/                           
│   ├── background/                    # 背景场景 2DGS 模型
│   ├── object_a/                      # 物体A 2DGS 模型
│   ├── object_b/                      # 物体B Mesh & 高斯 PLY
│   ├── object_c/                      # 物体C Mesh & 高斯 PLY
│   ├── fused_scene.ply                # 合并后的场景
│   └── videos/                        # 渲染视频
├── configs/
│   └── placements.json                # 物体放置参数配置
├── src/
│   ├── reconstruction/             # 2DGS 训练、渲染、评估、导出
│   │   ├── model.py                   # 2DGS 模型定义
│   │   ├── train.py                   # 训练循环
│   │   ├── render.py                  # 新视角渲染器
│   │   ├── eval.py                    # 定量评估（PSNR/SSIM/LPIPS）
│   │   └── export.py                  # 模型导出 & 统计分析
│   ├── generation/               # AIGC 生成管线
│   │   ├── text_to_3d.py              # threestudio 文本→3D
│   │   └── image_to_3d.py             # Magic123 单图→3D
│   ├── fusion/                 # 场景融合 & 渲染
│   │   ├── camera.py                  # 相机轨迹生成
│   │   ├── fuse.py                    # 高斯场景拼接
│   │   └── render_video.py            # 视频渲染
│   └── utils/                         # 工具模块
│       └── mesh_to_gs.py              # Mesh → 2DGS 高斯面片转换
├── scripts/                           # Shell 执行脚本
│   ├── run_background.sh              # 背景场景 2DGS 训练
│   ├── run_colmap_object_a.sh         # COLMAP —— 物体A
│   ├── run_object_b.sh                # threestudio —— 物体B
│   ├── run_magic123_object_c.sh       # Magic123 —— 物体C
│   └── full_pipeline.sh               # 全链路一键执行
├── environment.yml                    # Conda 环境配置文件
└── README.md
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate 3d_fusion
```

### 2. 安装 PyTorch

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Object A 依赖

Object A 需要 COLMAP（SfM 位姿提取）和 2DGS 训练环境：

```bash
# COLMAP
conda install -c conda-forge colmap

# 2DGS 光栅化器
git clone https://github.com/hbb1/diff-surfel-rasterization
cd diff-surfel-rasterization && pip install . && cd ..

# simple-knn
git clone https://github.com/graphdeco-inria/simple-knn
cd simple-knn && pip install . && cd ..
```

### 4. Object B 依赖

```bash
git clone https://github.com/threestudio-project/threestudio
cd threestudio
pip install -r requirements.txt
cd ..
```

### 5. Object C 依赖（单图到 3D，Magic123）

```bash
git clone https://github.com/guochengqian/Magic123
cd Magic123
pip install -r requirements.txt
cd ..
```

### 6. 额外依赖

其他共享工具包，运行时显示缺的包再手动安装即可

```bash
pip install trimesh opencv-python rembg imageio[ffmpeg] swanlab
```

---

## 数据准备

### 背景场景

从 [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) 下载一个场景（如 `garden`），
解压后放置为：

```
data/mipnerf360/garden/
├── images/       # 所有训练图像
├── sparse/0/     # COLMAP 稀疏重建结果
└── ...
```

### 物体A —— 多视角照片

1. 用手机/相机对真实物体拍摄环绕视频或多视角照片（建议 30–60 张，相邻照片重叠 ≥ 70%）。
2. 将照片放入 `data/object_a/images/`。
3. 运行 COLMAP。

### 物体B —— 文本 Prompt

直接在命令行中指定 Prompt，无需准备数据文件。

### 物体C —— 单张照片

1. 对真实物体拍摄一张 2D 照片。
2. 将照片放置为 `data/object_c/photo.jpg`。
3. 可使用 `rembg` 自动去背景，或手动处理。

---

## 各模块使用说明

### 1. 背景场景重建（2DGS）

**概述：** 在 Mip-NeRF 360 数据集上使用 2DGS 重建背景场景。

```bash
# 训练
python src/reconstruction/train.py \
    --source_path data/mipnerf360/garden \
    --model_path outputs/background/garden \
    --iterations 30000

# 导出最终 PLY
python src/reconstruction/export.py \
    --checkpoint outputs/background/garden/chkpnt_final.pth \
    --output outputs/background/garden/point_cloud_final.ply

# 查看统计信息
python src/reconstruction/export.py \
    --checkpoint outputs/background/garden/point_cloud_final.ply \
    --output outputs/background/garden/point_cloud_final.ply \
    --stats
```

**参数：**

| 参数 | 说明 | 
|------|------|--------|
| `--source_path` | COLMAP 场景路径（含 images/ 和 sparse/） | 
| `--model_path` | 检查点输出目录 | 
| `--iterations` | 训练迭代次数 | 
| `--lambda_dssim` | SSIM 损失权重 | 

也可以使用 Shell 脚本：

```bash
bash scripts/run_background.sh data/mipnerf360/garden garden
```

---

### 2. 物体A — 多视角重建（COLMAP + 2DGS）

**概述：** 对真实物体拍摄多视角照片，使用 COLMAP 提取位姿，再用 2DGS 重建。

#### 第一步：运行 COLMAP SfM

```bash
bash scripts/run_colmap_object_a.sh data/object_a/images outputs/object_a
```

分步调试：

```bash
# 特征提取
colmap feature_extractor \
    --database_path outputs/object_a/database.db \
    --image_path data/object_a/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1

# 穷举特征匹配
colmap exhaustive_matcher \
    --database_path outputs/object_a/database.db \
    --SiftMatching.use_gpu 1

# 稀疏重建
colmap mapper \
    --database_path outputs/object_a/database.db \
    --image_path data/object_a/images \
    --output_path outputs/object_a/sparse
```

#### 第二步：训练 2DGS

```bash
python src/reconstruction/train.py \
    --source_path outputs/object_a \
    --model_path outputs/object_a \
    --iterations 15000
```

#### 第三步：导出高斯 PLY

```bash
python src/reconstruction/export.py \
    --checkpoint outputs/object_a/chkpnt_final.pth \
    --output outputs/object_a/object_a.ply
```

---

### 3. 物体B — 文本到3D（threestudio ）

仅通过一段文本 Prompt 生成 3D 虚拟物体。底层基于 threestudio 的DreamFusion 管线，使用 Stable Diffusion v1.5 作为 2D 先验，SDS Loss 引导 NeRF 优化。

```bash
python src/generation/text_to_3d.py \
    --prompt "A ceramic teapot with floral patterns, photorealistic" \
    --output_dir outputs/object_b \
    --iters 10000 \
    --method dreamfusion
```

也可以使用 Shell 脚本：

```bash
bash scripts/run_object_b.sh
```

**关键参数：**

| 参数 | 说明 |
|------|------|
| `--prompt` | 描述 3D 物体的文本 Prompt |
| `--negative_prompt` | 负向提示词 |
| `--output_dir` | 输出目录 |
| `--iters` | 训练迭代次数 |
| `--method` | threestudio 方法 |
| `--guidance_model` | 扩散先验模型（默认 runwayml/stable-diffusion-v1-5） |

默认 Prompt：

```
A single, standard white ceramic coffee mug with an intricate, yet simple, classic
blue and white botanical and geometric pattern wrapping around its external surface,
glossy glaze finish, simple 3D model, isolated on a seamless pure white background,
three-quarter product shot
```

**输出：** 训练完成后在 `outputs/object_b/model.obj` 生成带纹理 Mesh。

> 因 threestudio 依赖的 nerfacc CUDA 扩展在 CUDA 12.8 / RTX 5090 下编译失败，实际实验中改用 [Tripo3D] 文生 3D 工具生成 OBJ，再经 `mesh_to_gs.py` 转换为 PLY。

---

### 4. 物体C — 单图到3D（Magic123）

拍摄一张真实物体的 2D 照片，通过 `rembg` 去背景，输入 Magic123 生成 3D Model。

#### 完整流程

```bash
bash scripts/run_magic123_object_c.sh data/object_c/photo.jpg outputs/object_c
```

#### 分步执行

```bash
# 步骤1: 去背景 + 中心裁剪
python -c "
from src.generation.image_to_3d import ImageTo3DGenerator
gen = ImageTo3DGenerator(output_dir='outputs/object_c')
gen.run_pipeline('data/object_c/photo.jpg', size=512, elevation=30.0)
"

# 步骤2 
python src/generation/image_to_3d.py \
    --image outputs/object_c/processed/input_square.png \
    --output_dir outputs/object_c \
    --no_bg_removal \
    --elevation 30.0
```

**参数：**

| 参数 | 说明 |
|------|------|
| `--image` | 输入图像路径 |
| `--output_dir` | 输出目录（默认 ./outputs/object_c） |
| `--size` | 预处理正方形边长（默认 512） |
| `--elevation` | 假设相机仰角（默认 30.0） |
| `--no_bg_removal` | 跳过自动去背景|

**输出：** 生成完成后在 `outputs/object_c/mesh/model.obj` 生成 Mesh。

> 因 Magic123（2023）与当前环境存在多项不兼容，实际实验中改用 [Tripo3D](https://www.tripo3d.ai/) 图生 3D 工具生成 OBJ，经 `mesh_to_gs.py` 转换为 PLY。

---

### 5. Mesh 转高斯面片

将物体 B / C 生成的 Mesh（OBJ/PLY/GLB）转换为 2DGS 可渲染的高斯面片格式。

这是连接 AIGC 生成管线与融合渲染的关键桥梁——threestudio 和 Magic123 输出的是传统 Mesh 或隐式场表示，而 2DGS 背景和渲染器工作在显式高斯面片表示下。
本模块将 Mesh 表面采样为带朝向的平面高斯（surfel），实现不同表示之间的格式统一。

```bash
# 转换物体B
python src/utils/mesh_to_gs.py \
    --input outputs/object_b/model.obj \
    --output outputs/object_b/object_b.ply \
    --num_samples 100000

# 转换物体C
python src/utils/mesh_to_gs.py \
    --input outputs/object_c/mesh/model.obj \
    --output outputs/object_c/object_c.ply \
    --num_samples 100000
```

**关键参数：**

| 参数 | 说明 | 
|------|------|
| `--input` | 输入 Mesh 路径 (.obj/.ply/.glb) | 
| `--output` | 输出 2DGS PLY 路径 | 
| `--num_samples` | 采样的 surfel 数量（默认 100000） |
| `--surfel_scale` | surfel 盘面半径（默认 0.008） | 
| `--min_opacity` | 最小透明度（默认 0.1）| 
| `--max_opacity` | 最大透明度（默认 0.9）| 

---

### 6. 场景融合

将物体 A、B、C 的高斯 PLY 与背景场景拼接，施加独立的缩放、旋转、平移变换，输出一个统一的合并场景 PLY 文件。

#### 方法一：使用 JSON 配置文件

创建 `configs/placements.json`：

```json
{
  "background": "outputs/background/garden/point_cloud_final.ply",
  "objects": {
    "object_a": {
      "ply": "outputs/object_a/object_a.ply",
      "scale": 0.3,
      "rotation_deg": [0, 45, 0],
      "translation": [1.2, 0.5, 0.3]
    },
    "object_b": {
      "ply": "outputs/object_b/object_b.ply",
      "scale": 0.5,
      "rotation_deg": [10, 0, 0],
      "translation": [-0.8, -0.3, 0.5]
    },
    "object_c": {
      "ply": "outputs/object_c/object_c.ply",
      "scale": 0.4,
      "rotation_deg": [0, -30, 0],
      "translation": [0.2, -0.8, 0.2]
    }
  }
}
```

```bash
python src/fusion/fuse.py \
    --config configs/placements.json \
    --output outputs/fused_scene.ply
```

#### 方法二：命令行直接指定

```bash
python src/fusion/fuse.py \
    --background outputs/background/garden/point_cloud_final.ply \
    --objects outputs/object_a/object_a.ply outputs/object_b/object_b.ply outputs/object_c/object_c.ply \
    --names obj_a obj_b obj_c \
    --placements 0.3 0 45 0 1.2 0.5 0.3 \
                 0.5 10 0 0 -0.8 -0.3 0.5 \
                 0.4 0 -30 0 0.2 -0.8 0.2 \
    --output outputs/fused_scene.ply
```

> `--placements` 格式：每个物体 7 个数 (scale, rx, ry, rz, tx, ty, tz)，按物体顺序拼接。旋转角度单位为度，施加顺序为 Z→Y→X。

---

### 7. 漫游视频渲染

加载合并后的场景，沿预设相机轨迹渲染多视角帧并编码为 MP4 视频。

```bash
python src/fusion/render_video.py \
    --scene outputs/fused_scene.ply \
    --output outputs/videos/flythrough.mp4 \
    --trajectory spiral \
    --center_x 0.0 --center_y 0.0 --center_z 0.5 \
    --radius 4.0 \
    --orbit_height 2.0 \
    --num_frames 180 \
    --fps 30 \
    --res_width 1920 --res_height 1080
```

#### 内置轨迹类型

| 轨迹 | 说明 | 适用场景 |
|------|------|----------|
| `circular` | 绕中心点水平旋转 | 全景展示，物体环绕 |
| `spiral` | 从远到近螺旋逼近 | 电影感揭露镜头 |
| `keyframe` | 关键帧插值 | 自定义复杂路径 |

#### 使用自定义 JSON 轨迹

```bash
# 生成轨迹 JSON
python src/fusion/render_video.py --scene ... --output ... -t circular

# 或用 camera.py 模块编程生成
python -c "
from src.fusion.camera import spiral_trajectory, save_trajectory
cams = spiral_trajectory(center=(0,0,0), num_frames=120)
save_trajectory(cams, 'outputs/my_trajectory.json')
"

# 使用 JSON 文件渲染
python src/fusion/render_video.py \
    --scene outputs/fused_scene.ply \
    --output outputs/videos/custom.mp4 \
    --trajectory outputs/my_trajectory.json
```


---

## 全链路一键运行

配置好数据路径后，执行：

```bash
bash scripts/full_pipeline.sh
```

该脚本从零开始执行全部六个阶段：
1. 背景场景 2DGS 重建
2. 物体A COLMAP SfM + 2DGS 训练
3. 物体B threestudio 文本到 3D + Mesh 转高斯
4. 物体C Magic123 单图到 3D + Mesh 转高斯
5. 场景融合（需要先编辑 `configs/placements.json`）
6. 漫游视频渲染

---

### Tripo3D 替代流程

由于 threestudio 和 Magic123 与当前 CUDA/PyTorch 生态存在兼容性问题，实际实验中物体 B 和 C 改用 Tripo3D 生成 OBJ，后续步骤与全管线一致。
整体主流程仍以 `full_pipeline.sh` 为参考，记录实际运行命令。

```bash
# Step 1: 使用 Tripo3D 分别生成物体 B（文本→3D）和物体 C（图像→3D）的 OBJ
将下载的 OBJ 分别保存为:
`outputs/object_b/objectb.obj`
`outputs/object_c/objectc.obj`

# Step 2: OBJ → PLY 转换
python src/utils/mesh_to_gs.py -i outputs/object_b/objectb.obj -o outputs/object_b/objectb.ply
python src/utils/mesh_to_gs.py -i outputs/object_c/objectc.obj -o outputs/object_c/objectc.ply

# Step 3: 场景融合
python src/fusion/fuse.py --config configs/placements.json --output outputs/fused_scene.ply

# Step 4: 渲染漫游视频
python src/fusion/render_video.py \
    --scene outputs/fused_scene.ply \
    --output outputs/videos/flythrough.mp4 \
    --trajectory spiral
```

---

## 指标与日志

训练过程中通过 Swanlab 实时记录损失曲线。

---

## 引用

本项目基于以下开源工作构建：

- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) 
- [threestudio](https://github.com/threestudio-project/threestudio) —
- [Magic123](https://github.com/guochengqian/Magic123) 
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) —
- [COLMAP](https://colmap.github.io/) 
