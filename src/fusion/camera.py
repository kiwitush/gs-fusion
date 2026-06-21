"""
相机轨迹生成工具，用于多视角漫游渲染。

提供圆形、螺旋和关键帧插值三种轨迹类型，统一输出为
Renderer2DGS.render_view 兼容的相机字典列表。
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import torch


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """计算 4×4 的世界→相机（Look-At）视图矩阵（OpenCV 约定，与 3DGS 兼容）。

    相机坐标系: +X 右, +Y 下, +Z 前（指向注视点）。

    Args:
        eye: (3,) 世界空间中的相机位置。
        center: (3,) 目标注视点。
        up: (3,) 世界空间"上方"参考方向（默认为 +Z）。

    Returns:
        4×4 视图矩阵（numpy row-major，调用方负责 .T 转 column-major）。
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-10)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-10)
    u = np.cross(s, f)

    R = np.eye(4, dtype=np.float32)
    R[0, :3] = s     # 相机 +X = 右
    R[1, :3] = -u    # 相机 +Y = 下（OpenCV 图像约定）
    R[2, :3] = f     # 相机 +Z = 前（OpenCV 注视方向）

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    return R @ T


def focal2fov(focal: float, pixels: float) -> float:
    """将焦距（像素单位）转换为视场角（弧度）。

    匹配 3DGS 仓库中的 ``focal2fov`` 公式。
    """
    return 2.0 * float(np.arctan(pixels / (2.0 * focal)))


def projection_from_fov(
    fov_x: float, fov_y: float,
    znear: float = 0.01, zfar: float = 100.0,
) -> np.ndarray:
    """3DGS 风格的 OpenCV 投影矩阵（Z 轴向前）。

    与 3DGS 仓库 ``getProjectionMatrix`` 完全一致，
    供 diff-surfel-rasterization 直接使用。

    Args:
        fov_x: 水平视场角（弧度）。
        fov_y: 垂直视场角（弧度）。
        znear, zfar: 近/远裁剪平面。

    Returns:
        4×4 投影矩阵（numpy row-major，调用方负责 .T 转 column-major）。
    """
    tan_half_x = float(np.tan(fov_x / 2.0))
    tan_half_y = float(np.tan(fov_y / 2.0))

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tan_half_x
    P[1, 1] = 1.0 / tan_half_y
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def projection_from_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    znear: float = 0.01, zfar: float = 100.0,
) -> np.ndarray:
    """从针孔内参构建 3DGS/OpenCV 投影矩阵。

    直接通过 fx,fy,cx,cy,width,height 构建，无需先转换 FOV。
    等同于 ``projection_from_fov(focal2fov(fx,width), focal2fov(fy,height), ...)``。

    Args:
        fx, fy: 焦距（像素单位）。
        cx, cy: 主点偏移（像素单位）。
        width, height: 图像分辨率。
        znear, zfar: 近/远裁剪平面。

    Returns:
        4×4 投影矩阵（numpy row-major，调用方负责 .T 转 column-major）。
    """
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[2, 0] = 2.0 * (cx / width) - 1.0
    P[2, 1] = 2.0 * (cy / height) - 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def get_world_to_view(
    R_c2w: np.ndarray, t_vec: np.ndarray,
    translate: np.ndarray = None, scale: float = 1.0,
) -> np.ndarray:
    """从 COLMAP 风格外参构建世界→相机视图矩阵。

    与 3DGS ``getWorld2View2`` 等价。

    Args:
        R_c2w: 3×3 相机→世界旋转矩阵（即 R_w2c 的转置）。
        t_vec: (3,) COLMAP tvec（相机原点在世界坐标中的位置）。
        translate: (3,) 可选世界空间平移，用于场景缩放调整。
        scale: 可选世界空间缩放因子。

    Returns:
        4×4 世界→相机视图矩阵（numpy row-major，调用方负责 .T 转 column-major）。
    """
    if translate is None:
        translate = np.zeros(3, dtype=np.float32)

    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R_c2w.transpose()  # R_w2c
    Rt[:3, 3] = t_vec
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt


def circular_trajectory(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 3.0,
    height: float = 1.5,
    num_frames: int = 120,
    fov_y: float = 0.539,
    aspect: float = 16.0 / 9.0,
    height_res: int = 1080,
    width_res: int = 1920,
    device: str = "cuda",
) -> List[dict]:
    """生成绕中心点旋转的圆形轨迹。

    Returns:
        相机字典列表（与 Renderer2DGS.render_view 兼容）。
    """
    cameras = []
    for i in range(num_frames):
        angle = 2.0 * np.pi * i / num_frames
        eye = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height,
        ], dtype=np.float32)
        center_pt = np.array(center, dtype=np.float32)

        view = look_at(eye, center_pt)
        fov_x = 2.0 * np.arctan(np.tan(fov_y / 2.0) * aspect)
        proj = projection_from_fov(fov_x, fov_y)

        cameras.append({
            "height": height_res,
            "width": width_res,
            "viewmatrix": torch.tensor(view.T, dtype=torch.float32, device=device),
            "projmatrix": torch.tensor(proj.T, dtype=torch.float32, device=device),
            "fovx": fov_x,
            "fovy": fov_y,
            "campos": torch.tensor(eye, dtype=torch.float32, device=device),
        })

    return cameras


def spiral_trajectory(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_radius: float = 5.0,
    end_radius: float = 1.5,
    start_height: float = 3.0,
    end_height: float = 0.5,
    num_frames: int = 180,
    turns: float = 2.5,
    fov_y: float = 0.539,
    aspect: float = 16.0 / 9.0,
    height_res: int = 1080,
    width_res: int = 1920,
    device: str = "cuda",
) -> List[dict]:
    """生成向中心螺旋逼近的轨迹。

    适合作为渲染视频中的"揭露镜头"。
    """
    cameras = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        angle = 2.0 * np.pi * t * turns
        radius = start_radius + (end_radius - start_radius) * t
        height = start_height + (end_height - start_height) * t

        eye = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height,
        ], dtype=np.float32)
        center_pt = np.array(center, dtype=np.float32)

        view = look_at(eye, center_pt)
        fov_x = 2.0 * np.arctan(np.tan(fov_y / 2.0) * aspect)
        proj = projection_from_fov(fov_x, fov_y)

        cameras.append({
            "height": height_res,
            "width": width_res,
            "viewmatrix": torch.tensor(view.T, dtype=torch.float32, device=device),
            "projmatrix": torch.tensor(proj.T, dtype=torch.float32, device=device),
            "fovx": fov_x,
            "fovy": fov_y,
            "campos": torch.tensor(eye, dtype=torch.float32, device=device),
        })

    return cameras


def keyframe_trajectory(
    keyframes: List[Tuple[np.ndarray, np.ndarray]],
    num_frames: int = 120,
    fov_y: float = 0.539,
    aspect: float = 16.0 / 9.0,
    height_res: int = 1080,
    width_res: int = 1920,
    device: str = "cuda",
) -> List[dict]:
    """在关键帧 (eye, center) 对之间进行平滑插值。

    对相机位置使用线性插值，对注视中心使用线性插值。

    Args:
        keyframes: (eye_xyz, center_xyz) 元组列表，长度至少为 2。
        num_frames: 输出帧数。
        fov_y, aspect: 相机内参。
        height_res, width_res: 分辨率。
        device: PyTorch 设备。

    Returns:
        相机字典列表。
    """
    if len(keyframes) < 2:
        raise ValueError("至少需要 2 个关键帧。")

    n_segments = len(keyframes) - 1
    cameras = []

    for i in range(num_frames):
        t = i / (num_frames - 1)
        seg = min(int(t * n_segments), n_segments - 1)
        local_t = (t - seg / n_segments) * n_segments
        local_t = np.clip(local_t, 0.0, 1.0)

        eye0, center0 = keyframes[seg]
        eye1, center1 = keyframes[seg + 1]

        eye = eye0 + local_t * (eye1 - eye0)
        center_pt = center0 + local_t * (center1 - center0)

        view = look_at(eye, center_pt)
        fov_x = 2.0 * np.arctan(np.tan(fov_y / 2.0) * aspect)
        proj = projection_from_fov(fov_x, fov_y)

        cameras.append({
            "height": height_res,
            "width": width_res,
            "viewmatrix": torch.tensor(view.T, dtype=torch.float32, device=device),
            "projmatrix": torch.tensor(proj.T, dtype=torch.float32, device=device),
            "fovx": fov_x,
            "fovy": fov_y,
            "campos": torch.tensor(eye, dtype=torch.float32, device=device),
        })

    return cameras


def save_trajectory(cameras: List[dict], path: str) -> None:
    """将相机轨迹序列化为 JSON 文件（Tensor → list）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    serialisable = []
    for cam in cameras:
        entry = {}
        for k, v in cam.items():
            if isinstance(v, torch.Tensor):
                entry[k] = v.cpu().tolist()
            else:
                entry[k] = v
        serialisable.append(entry)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[相机] 轨迹已保存 → {path} ({len(cameras)} 帧)")


def load_trajectory(path: str, device: str = "cuda") -> List[dict]:
    """从 JSON 文件加载相机轨迹（list → Tensor）。"""
    with open(path, "r") as f:
        data = json.load(f)
    cameras = []
    for entry in data:
        cam = {}
        for k, v in entry.items():
            if k in ("viewmatrix", "projmatrix", "campos"):
                cam[k] = torch.tensor(v, dtype=torch.float32, device=device)
            else:
                cam[k] = v
        cameras.append(cam)
    print(f"[相机] 轨迹已加载 ← {path} ({len(cameras)} 帧)")
    return cameras
