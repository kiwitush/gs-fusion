"""
场景融合引擎：将多个 2DGS PLY 资产合并为统一场景。

加载背景场景和多个物体高斯点云，对每个物体施加独立的相似变换
（缩放、旋转、平移），最后拼接为一个可用于渲染的合并 PLY 文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.model import GaussianModel2D, write_2dgs_ply


class SceneFusion:
    """将多个 2D 高斯点云合并为单个场景。

    使用方法::

        fusion = SceneFusion()
        fusion.add_background("outputs/bg/point_cloud_final.ply")
        fusion.add_object("outputs/a.ply", name="obj_a", scale=0.5, translation=(1,0,0))
        fusion.fuse("outputs/fused.ply")
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._models: Dict[str, GaussianModel2D] = {}
        self._placements: Dict[str, dict] = {}

    def add_background(self, ply_path: str, name: str = "background") -> None:
        """加载背景场景（不施加任何变换）。"""
        model = GaussianModel2D(max_sh_degree=3, device=self.device)
        model.load_ply(ply_path)
        self._models[name] = model
        self._placements[name] = {
            "scale": 1.0,
            "rotation_deg": [0, 0, 0],
            "translation": [0, 0, 0],
        }
        print(f"[融合] 背景 '{name}' 已加载: {model.num_gaussians} 个高斯")

    def add_object(
        self,
        ply_path: str,
        name: str,
        scale: float = 1.0,
        rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """加载物体 PLY 并注册其放置变换。

        Args:
            ply_path: 2DGS PLY 文件路径。
            name: 唯一物体名称。
            scale: 均匀缩放因子。
            rotation_deg: (rx, ry, rz) 欧拉角（度），按 Z→Y→X 顺序施加。
            translation: (tx, ty, tz) 世界空间平移（缩放+旋转之后施加）。
        """
        model = GaussianModel2D(max_sh_degree=3, device=self.device)
        model.load_ply(ply_path)
        self._models[name] = model
        self._placements[name] = {
            "scale": scale,
            "rotation_deg": list(rotation_deg),
            "translation": list(translation),
        }
        print(f"[融合] 物体 '{name}' 已加载: {model.num_gaussians} 个高斯")

    def fuse(self, output_path: str) -> str:
        """施加所有变换并写入合并后的 PLY。

        Returns:
            输出文件的绝对路径。
        """
        all_xyz, all_f_dc, all_f_rest, all_scale, all_rot, all_opacity, all_normal = (
            [], [], [], [], [], [], [],
        )

        for name, model in self._models.items():
            p = self._placements[name]
            s = p["scale"]
            rx, ry, rz = [np.deg2rad(a) for a in p["rotation_deg"]]
            tx, ty, tz = p["translation"]

            R, t_vec = self._build_rotation_translation(rx, ry, rz, tx, ty, tz)

            xyz = model._xyz.detach().cpu().numpy()
            centroid = xyz.mean(axis=0)  # 物体中心化，确保旋转围绕自身中心
            normals = model._normal.detach().cpu().numpy()
            f_dc = model._features_dc.detach().cpu().numpy().reshape(-1, 3)
            f_rest = model._features_rest.detach().cpu().numpy().reshape(-1, 45)
            log_scale = model._scaling.detach().cpu().numpy()
            rot = model._rotation.detach().cpu().numpy()
            opacity = model._opacity.detach().cpu().numpy().reshape(-1)

            xyz_t = s * (R @ (xyz - centroid).T).T + t_vec

            normals_t = (R @ normals.T).T
            norm_len = np.linalg.norm(normals_t, axis=-1, keepdims=True) + 1e-10
            normals_t = normals_t / norm_len

            rot_t = self._rotate_quaternions(rot, R)

            if s <= 0:
                raise ValueError(f"物体 '{name}' 的缩放因子必须为正数，当前值: {s}")
            log_scale_t = log_scale + np.log(s)

            all_xyz.append(xyz_t)
            all_f_dc.append(f_dc)
            all_f_rest.append(f_rest)
            all_scale.append(log_scale_t)
            all_rot.append(rot_t)
            all_opacity.append(opacity)
            all_normal.append(normals_t)

            print(
                f"[融合] 已变换 '{name}': s={s}, "
                f"R=({p['rotation_deg']}), t=({tx:.2f},{ty:.2f},{tz:.2f})"
            )

        merged = {
            "xyz": np.concatenate(all_xyz, axis=0).astype(np.float32),
            "normals": np.concatenate(all_normal, axis=0).astype(np.float32),
            "f_dc": np.concatenate(all_f_dc, axis=0).astype(np.float32),
            "f_rest": np.concatenate(all_f_rest, axis=0).astype(np.float32),
            "scale": np.concatenate(all_scale, axis=0).astype(np.float32),
            "rot": np.concatenate(all_rot, axis=0).astype(np.float32),
            "opacity": np.concatenate(all_opacity, axis=0).astype(np.float32),
        }

        total = merged["xyz"].shape[0]
        print(f"[融合] 合并场景总高斯数: {total}")

        self._write_ply(output_path, merged)
        return os.path.abspath(output_path)

    @staticmethod
    def _build_rotation_translation(
        rx: float, ry: float, rz: float,
        tx: float, ty: float, tz: float,
    ) -> tuple:
        """构建 3×3 旋转矩阵与 3D 平移向量（不含缩放）。"""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

        R = Rz @ Ry @ Rx
        t = np.array([tx, ty, tz], dtype=np.float32)
        return R, t

    @staticmethod
    def _rotate_quaternions(quats: np.ndarray, R: np.ndarray) -> np.ndarray:
        """将 3×3 旋转矩阵 R 施加到 N 个四元数（w,x,y,z 顺序）上。"""
        from scipy.spatial.transform import Rotation as R_scipy

        q_scipy = np.stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]], axis=1)
        rot_orig = R_scipy.from_quat(q_scipy)
        rot_matrices = rot_orig.as_matrix()
        rotated = R @ rot_matrices
        rot_new = R_scipy.from_matrix(rotated)
        q_new = rot_new.as_quat()
        return np.stack([q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]], axis=1)

    @staticmethod
    def _write_ply(path: str, merged: dict) -> None:
        """将合并后的高斯云写入 2DGS PLY 文件（委托给共享的 write_2dgs_ply）。"""
        xyz = merged["xyz"]
        n = merged["normals"]
        f_dc = merged["f_dc"]
        f_rest = merged["f_rest"]
        sc = merged["scale"]
        ro = merged["rot"]
        op = merged["opacity"]

        write_2dgs_ply(
            path, xyz, n, f_dc, f_rest,
            op.reshape(-1, 1) if op.ndim == 1 else op,
            sc, ro,
        )


def load_placements_from_json(json_path: str) -> dict:
    """从 JSON 文件加载物体放置配置。

    期望格式::

        {
          "background": "outputs/background/point_cloud_final.ply",
          "objects": {
            "object_a": {
              "ply": "outputs/object_a/object_a.ply",
              "scale": 0.5,
              "rotation_deg": [0, 45, 0],
              "translation": [1.0, 0.0, 0.5]
            },
            "object_b": { ... },
            "object_c": { ... }
          }
        }
    """
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="将多个 2DGS 资产融合为单个场景")
    parser.add_argument("--config", "-c", help="JSON 放置配置文件路径")
    parser.add_argument("--background", help="背景场景 PLY 路径")
    parser.add_argument("--objects", nargs="*", help="物体 PLY 路径列表")
    parser.add_argument(
        "--placements", nargs="*", type=float,
        help="扁平放置参数: s1 rx1 ry1 rz1 tx1 ty1 tz1 s2 rx2 ...",
    )
    parser.add_argument("--names", nargs="*", help="物体名称列表（与 --objects 配合使用）")
    parser.add_argument("--output", "-o", default="outputs/fused_scene.ply", help="输出合并 PLY 路径")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fusion = SceneFusion(device=args.device)

    if args.config:
        cfg = load_placements_from_json(args.config)
        fusion.add_background(cfg["background"])
        for obj_name, obj_cfg in cfg["objects"].items():
            fusion.add_object(
                ply_path=obj_cfg["ply"],
                name=obj_name,
                scale=obj_cfg.get("scale", 1.0),
                rotation_deg=tuple(obj_cfg.get("rotation_deg", [0, 0, 0])),
                translation=tuple(obj_cfg.get("translation", [0, 0, 0])),
            )
    elif args.background and args.objects:
        fusion.add_background(args.background)
        names = args.names or [f"object_{i}" for i in range(len(args.objects))]
        if args.placements:
            chunk = 7
            for i, name in enumerate(names):
                p = args.placements[i * chunk : (i + 1) * chunk]
                if len(p) == chunk:
                    fusion.add_object(args.objects[i], name, p[0], tuple(p[1:4]), tuple(p[4:7]))
                else:
                    fusion.add_object(args.objects[i], name)
        else:
            for i, (obj_path, name) in enumerate(zip(args.objects, names)):
                fusion.add_object(obj_path, name)
    else:
        print("请提供 --config 或 --background + --objects。参见 --help。")
        return

    fusion.fuse(args.output)


if __name__ == "__main__":
    main()
