"""
2D Gaussian Splatting 训练循环。

完整训练流程：SfM 点云初始化 → 迭代优化（L1 + SSIM 损失）→
自适应密度控制 → SH 阶数逐步提升 → 定期检查点保存。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fusion.camera import focal2fov, projection_from_intrinsics, get_world_to_view
from src.reconstruction.model import GaussianModel2D, _normals_to_quaternions
from src.utils.image_quality import psnr, ssim

try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_RASTERIZER = True
except ImportError:
    HAS_RASTERIZER = False
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


class Trainer2DGS:
    """端到端的 2D Gaussian Splatting 训练器。"""

    def __init__(
        self,
        model: GaussianModel2D,
        source_path: str,
        model_path: str,
        iterations: int = 30_000,
        position_lr_init: float = 1.6e-4,
        position_lr_final: float = 1.6e-6,
        scaling_lr: float = 5e-3,
        rotation_lr: float = 1e-3,
        opacity_lr: float = 5e-2,
        feature_lr: float = 2.5e-3,
        lambda_dssim: float = 0.2,
        densify_from_iter: int = 500,
        densify_until_iter: int = 25_000,
        densification_interval: int = 100,
        opacity_reset_interval: int = 3_000,
        densify_grad_threshold: float = 1e-5,
        resolution: int = 1600,
        sh_degree_interval: int = 10_000,
        save_interval: int = 5_000,
        device: str = "cuda",
    ):
        self.model = model
        self.source_path = Path(source_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.iterations = iterations

        self.lambda_dssim = lambda_dssim
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.resolution = resolution
        self.sh_degree_interval = sh_degree_interval
        self.save_interval = save_interval
        self.device = device

        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.opacity_lr = opacity_lr
        self.feature_lr = feature_lr

        self._init_optimizer(
            position_lr_init, position_lr_final,
            scaling_lr, rotation_lr, opacity_lr, feature_lr,
        )

        self._dataset = None
        self._test_dataset = None

    def _init_optimizer(self, pos_init, pos_final, s_lr, r_lr, o_lr, f_lr):
        """构建 Adam 优化器，学习率按指数衰减。"""
        param_groups = [
            {"params": [self.model._xyz], "lr": pos_init, "name": "xyz"},
            {"params": [self.model._features_dc], "lr": f_lr, "name": "f_dc"},
            {"params": [self.model._features_rest], "lr": f_lr / 20.0, "name": "f_rest"},
            {"params": [self.model._opacity], "lr": o_lr, "name": "opacity"},
            {"params": [self.model._scaling], "lr": s_lr, "name": "scaling"},
            {"params": [self.model._rotation], "lr": r_lr, "name": "rotation"},
            {"params": [self.model._normal], "lr": pos_init * 0.1, "name": "normal"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def _update_lr(self, iteration: int):
        """位置学习率按多项式衰减 (原版 3DGS 公式: (1 - t/T)^0.9)。"""
        progress = iteration / max(self.iterations, 1)
        new_lr = self.position_lr_init * ((1.0 - progress) ** 0.9)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = max(new_lr, self.position_lr_final)

    def train(self):
        """执行完整训练循环。"""
        if not HAS_RASTERIZER:
            raise RuntimeError(
                "diff-surfel-rasterization 未安装。"
                "请运行: pip install git+https://github.com/hbb1/diff-surfel-rasterization"
            )

        pcd = self._load_colmap_sparse()
        self.model.create_from_pcd(pcd)
        self._load_dataset()

        # 用相机平均距离计算 spatial_lr_scale（原版 3DGS 做法）
        cam_positions = torch.stack([v["campos"] for v in self._dataset])
        centroid = cam_positions.mean(dim=0)
        camera_extent = torch.mean(torch.norm(cam_positions - centroid, dim=1)).item()
        self.model.spatial_lr_scale = max(camera_extent, 1.0)
        print(f"[训练] 相机范围={camera_extent:.2f}, spatial_lr_scale={self.model.spatial_lr_scale:.2f}")

        # 将 spatial_lr_scale 乘入 position_lr，确保 _update_lr 与 _init_optimizer 一致
        self.position_lr_init = self.position_lr_init * self.model.spatial_lr_scale
        self.position_lr_final = self.position_lr_final * self.model.spatial_lr_scale

        # create_from_pcd 生成了新的 Parameter，必须重建优化器使其指向正确的参数
        self._init_optimizer(
            self.position_lr_init,
            self.position_lr_final,
            self.scaling_lr, self.rotation_lr, self.opacity_lr, self.feature_lr,
        )

        import swanlab
        swanlab.init(
            project="3d_fusion",
            experiment_name=f"2dgs-{self.source_path.name}",
        )

        ema_loss = 0.0
        best_val_psnr = 0.0
        first_iter = 0

        for iteration in range(first_iter, self.iterations):
            viewpoint = self._sample_viewpoint()
            gt_image = viewpoint["image"].to(self.device)

            render_pkg = self._render(viewpoint)
            image = render_pkg["render"]
            viewspace_points = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            Ll1 = F.l1_loss(image, gt_image)
            ssim_val = ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_val)
            loss.backward()

            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            # 在 optimizer.zero_grad 清空梯度前，保存视空间点梯度用于密度控制
            needs_densify = (
                self.densify_from_iter <= iteration <= self.densify_until_iter
                and iteration % self.densification_interval == 0
            )
            viewspace_grad = viewspace_points.grad.clone() if needs_densify else None

            with torch.no_grad():
                self._update_lr(iteration)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if iteration % self.sh_degree_interval == 0:
                    self.model.oneup_SH_degree()

                if viewspace_grad is not None:
                    self._densify(viewspace_grad, visibility_filter, radii)

                if iteration % self.opacity_reset_interval == 0 and iteration < self.densify_until_iter:
                    self._reset_opacity()

                if iteration % self.save_interval == 0 or iteration == self.iterations - 1:
                    self.model.save_checkpoint(str(self.model_path / f"chkpnt{iteration}.pth"))
                    self.model.save_ply(str(self.model_path / f"point_cloud_{iteration}.ply"))

            psnr_val = psnr(image, gt_image)

            if iteration % 100 == 0:
                log_dict = {"loss": loss.item(), "l1": Ll1.item(), "psnr": psnr_val}
                if self._test_dataset and iteration % 1000 == 0 and iteration > 0:
                    val_metrics = self._validate(max_views=4)
                    if val_metrics:
                        log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                        if val_metrics["psnr"] > best_val_psnr:
                            best_val_psnr = val_metrics["psnr"]
                            self.model.save_checkpoint(str(self.model_path / "best.pth"))
                            self.model.save_ply(str(self.model_path / "best.ply"))
                            log_dict["val/best"] = 1
                swanlab.log(log_dict, step=iteration)
                print(
                    f"[迭代 {iteration:05d}] Loss: {loss.item():.4f}  "
                    f"PSNR: {psnr_val:.2f}  高斯数: {self.model.num_gaussians}"
                )

        self.model.save_checkpoint(str(self.model_path / "chkpnt_final.pth"))
        self.model.save_ply(str(self.model_path / "point_cloud_final.ply"))
        if best_val_psnr > 0:
            print(f"[训练] 最佳验证 PSNR: {best_val_psnr:.2f} → {self.model_path / 'best.ply'}")
        swanlab.finish()
        print("[训练] 训练完成。")

    @torch.no_grad()
    def _validate(self, max_views: int = 4) -> dict:
        """在测试集上评估 PSNR/SSIM（最多评估 max_views 张以控制开销）。"""
        views = random.sample(self._test_dataset, min(max_views, len(self._test_dataset)))
        psnr_vals, ssim_vals = [], []
        for v in views:
            gt = v["image"].to(self.device)
            pkg = self._render(v)
            pred = pkg["render"].clamp(0, 1)
            psnr_vals.append(psnr(pred, gt))
            ssim_vals.append(ssim(pred, gt))
        return {"psnr": float(np.mean(psnr_vals)), "ssim": float(np.mean(ssim_vals))}

    def _load_colmap_sparse(self) -> Dict[str, np.ndarray]:
        """加载 COLMAP 稀疏重建结果。优先尝试 binary 格式，其次 text 格式。"""
        sparse_dir = self.source_path / "sparse" / "0"
        if not sparse_dir.exists():
            raise FileNotFoundError(f"COLMAP 稀疏目录不存在: {sparse_dir}")

        if (sparse_dir / "points3D.bin").exists():
            return self._read_colmap_binary(sparse_dir)
        if (sparse_dir / "points3D.txt").exists():
            return self._read_colmap_text(sparse_dir)

        raise FileNotFoundError(f"在 {sparse_dir} 中未找到 COLMAP 模型")

    def _read_colmap_binary(self, sparse_dir):
        """解析 COLMAP 二进制格式（points3D.bin）。"""
        points_path = sparse_dir / "points3D.bin"

        points = {}
        with open(points_path, "rb") as f:
            n_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_points):
                pid = struct.unpack("<Q", f.read(8))[0]
                x, y, z = struct.unpack("<ddd", f.read(24))
                r, g, b = struct.unpack("<BBB", f.read(3))
                _error = struct.unpack("<d", f.read(8))[0]
                n_tracks = struct.unpack("<Q", f.read(8))[0]
                f.read(8 * n_tracks)  # 每条 track: image_id(uint32=4B) + point2D_idx(uint32=4B)
                points[pid] = {
                    "xyz": np.array([x, y, z], dtype=np.float32),
                    "rgb": np.array([r, g, b], dtype=np.float32) / 255.0,
                }

        xyz_arr = np.stack([p["xyz"] for p in points.values()], axis=0)
        rgb_arr = np.stack([p["rgb"] for p in points.values()], axis=0)
        normals = np.zeros_like(xyz_arr)
        return {"points": xyz_arr, "colors": rgb_arr, "normals": normals}

    def _read_colmap_text(self, sparse_dir):
        """解析 COLMAP 文本格式（points3D.txt）。"""
        pts_file = sparse_dir / "points3D.txt"
        xyz_list, rgb_list = [], []
        with open(pts_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                xyz_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb_list.append([int(parts[4]), int(parts[5]), int(parts[6])])

        xyz_arr = np.array(xyz_list, dtype=np.float32)
        rgb_arr = np.array(rgb_list, dtype=np.float32) / 255.0
        return {"points": xyz_arr, "colors": rgb_arr, "normals": np.zeros_like(xyz_arr)}

    def _load_dataset(self):
        """加载训练图像并解析 COLMAP 相机位姿与内参。

        解析 cameras.bin / images.bin，为每个视角构建与 3DGS rasterizer
        兼容的 viewmatrix、projmatrix、fovx、fovy、campos。
        """
        sparse_dir = self.source_path / "sparse" / "0"
        images_dir = self.source_path / "images"

        if not (sparse_dir / "cameras.bin").exists():
            raise FileNotFoundError(f"缺少 cameras.bin: {sparse_dir}")
        if not (sparse_dir / "images.bin").exists():
            raise FileNotFoundError(f"缺少 images.bin: {sparse_dir}")

        # COLMAP 4.0 二进制格式不兼容 → 统一转为文本格式解析
        cameras_txt = sparse_dir / "cameras.txt"
        images_txt = sparse_dir / "images.txt"
        points3D_txt = sparse_dir / "points3D.txt"
        if not (cameras_txt.exists() and images_txt.exists()):
            import subprocess
            print("[训练] 转换 COLMAP 二进制 → 文本格式 ...")
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(sparse_dir),
                "--output_type", "TXT",
            ], check=True)

        # 解析 cameras.txt
        cameras = {}
        with open(cameras_txt, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                w = int(parts[2])
                h = int(parts[3])
                params = [float(x) for x in parts[4:]]
                if model == "PINHOLE":
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                elif model == "SIMPLE_PINHOLE":
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                elif model == "SIMPLE_RADIAL":
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                else:
                    raise ValueError(f"不支持的相机模型: {model}")
                cameras[cam_id] = {
                    "width": w, "height": h,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                }

        # 解析 images.txt
        colmap_images = {}
        with open(images_txt, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            # 跳过 points2D 行
            i += 1
            colmap_images[name] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": np.array([tx, ty, tz], dtype=np.float32),
                "camera_id": cam_id,
            }

        # 加载图像并与 COLMAP 位姿匹配
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.JPEG")) + sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
        if not image_files:
            raise FileNotFoundError(f"在 {images_dir} 中未找到图像")

        self._dataset = []
        matched, unmatched = 0, 0
        for img_path in image_files:
            img_name = img_path.name
            if img_name not in colmap_images:
                unmatched += 1
                continue

            colmap = colmap_images[img_name]
            cam = cameras[colmap["camera_id"]]

            # 外参: COLMAP quaternion → R_w2c, R_c2w, tvec
            qw, qx, qy, qz = colmap["qvec"]
            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            R_c2w = R_w2c.T
            t_vec = colmap["tvec"]

            view = get_world_to_view(R_c2w, t_vec)
            proj = projection_from_intrinsics(
                cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                cam["width"], cam["height"],
            )

            fov_x = focal2fov(cam["fx"], cam["width"])
            fov_y = focal2fov(cam["fy"], cam["height"])
            campos = t_vec.copy()

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = cam["width"], cam["height"]
            if self.resolution > 0 and max(img_w, img_h) > self.resolution:
                scale = self.resolution / max(img_w, img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img_w, img_h = new_w, new_h
            img_tensor = torch.tensor(
                np.array(img) / 255.0, dtype=torch.float32
            ).permute(2, 0, 1)

            # 缩放后重新计算投影矩阵
            proj = projection_from_intrinsics(
                cam["fx"] * scale if self.resolution > 0 and max(cam["width"], cam["height"]) > self.resolution else cam["fx"],
                cam["fy"] * scale if self.resolution > 0 and max(cam["width"], cam["height"]) > self.resolution else cam["fy"],
                cam["cx"] * scale if self.resolution > 0 and max(cam["width"], cam["height"]) > self.resolution else cam["cx"],
                cam["cy"] * scale if self.resolution > 0 and max(cam["width"], cam["height"]) > self.resolution else cam["cy"],
                img_w, img_h,
            )

            self._dataset.append({
                "image": img_tensor,
                "path": str(img_path),
                "height": img_h,
                "width": img_w,
                "viewmatrix": torch.tensor(view.T, dtype=torch.float32, device=self.device),
                "projmatrix": torch.tensor(proj.T, dtype=torch.float32, device=self.device),
                "fovx": fov_x,
                "fovy": fov_y,
                "campos": torch.tensor(campos, dtype=torch.float32, device=self.device),
            })
            matched += 1

        if unmatched > 0:
            print(f"[训练] 警告: {unmatched} 张图像在 COLMAP 位姿目录中未找到匹配")
        if matched == 0:
            raise RuntimeError("没有任何图像与 COLMAP 位姿匹配，请检查 images/ 与 sparse/0/ 目录")

        # 训练/测试划分: 每 8 张留 1 张作为测试集
        test_interval = 8
        train_list, test_list = [], []
        for i, v in enumerate(self._dataset):
            if i % test_interval == 0:
                test_list.append(v)
            else:
                train_list.append(v)

        self._dataset = train_list
        self._test_dataset = test_list

        # 保存测试视角 JSON（不含图像张量，仅相机参数 + 路径）
        test_views_json = []
        for v in test_list:
            entry = {k: v[k] for k in v if k != "image"}
            entry["image_path"] = v["path"]
            # viewmatrix/projmatrix 转 list 以便 JSON 序列化
            entry["viewmatrix"] = v["viewmatrix"].cpu().tolist()
            entry["projmatrix"] = v["projmatrix"].cpu().tolist()
            entry["campos"] = v["campos"].cpu().tolist()
            test_views_json.append(entry)

        json_path = self.model_path / "test_views.json"
        with open(json_path, "w") as f:
            json.dump(test_views_json, f, indent=2)
        print(f"[训练] 训练集: {len(train_list)} 视角, 测试集: {len(test_list)} 视角 → {json_path}")

    def _sample_viewpoint(self) -> dict:
        """随机选择一个训练视角。"""
        idx = torch.randint(0, len(self._dataset), (1,)).item()
        return self._dataset[idx]

    @staticmethod
    def _eval_sh(dirs: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
        """Python SH 求值：将 (N,3) 方向 + (N,3,16) 系数 → (N,3) RGB。"""
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
              -1.0925484305920792, 0.5462742152960396]
        C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
              0.3731763325901154, -0.4570457994644658, 1.445305721320277,
              -0.5900435899266435]

        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = C0 * sh[:, :, 0]
        if sh.shape[-1] >= 4:
            result = result + C1 * (-sh[:, :, 1] * y + sh[:, :, 2] * z - sh[:, :, 3] * x)
        if sh.shape[-1] >= 9:
            xx, yy, zz = x * x, y * y, z * z
            result = result + (
                C2[0] * sh[:, :, 4] * x * y + C2[1] * sh[:, :, 5] * y * z
                + C2[2] * sh[:, :, 6] * (2 * zz - xx - yy) + C2[3] * sh[:, :, 7] * x * z
                + C2[4] * sh[:, :, 8] * (xx - yy))
        if sh.shape[-1] >= 16:
            xx, yy, zz = x * x, y * y, z * z
            result = result + (
                C3[0] * sh[:, :, 9] * y * (3 * xx - yy)
                + C3[1] * sh[:, :, 10] * x * y * z
                + C3[2] * sh[:, :, 11] * y * (4 * zz - xx - yy)
                + C3[3] * sh[:, :, 12] * z * (2 * zz - 3 * xx - 3 * yy)
                + C3[4] * sh[:, :, 13] * x * (4 * zz - xx - yy)
                + C3[5] * sh[:, :, 14] * z * (xx - yy)
                + C3[6] * sh[:, :, 15] * x * (xx - 3 * yy))
        return result + 0.5

    def _render(self, viewpoint: dict) -> dict:
        """从单个视角光栅化当前高斯云。

        返回包含 render、viewspace_points、visibility_filter、radii、depth、alpha 的字典。
        """
        height, width = viewpoint["height"], viewpoint["width"]
        fovx = viewpoint["fovx"]
        fovy = viewpoint["fovy"]
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=torch.tan(torch.tensor(fovx * 0.5)),
            tanfovy=torch.tan(torch.tensor(fovy * 0.5)),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint["viewmatrix"],
            projmatrix=viewpoint["projmatrix"],
            sh_degree=self.model.active_sh_degree,
            campos=viewpoint["campos"],
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.model.get_xyz
        means2D = torch.zeros_like(means3D, requires_grad=True)

        C0 = 0.28209479177387814
        features = self.model.get_features  # (N, 3, 16)
        viewdir = means3D.detach() - viewpoint["campos"].detach()
        viewdir = viewdir / (torch.norm(viewdir, dim=-1, keepdim=True) + 1e-10)
        colors_precomp = self._eval_sh(viewdir, features) if self.model.active_sh_degree > 0 else (
            features[:, :, 0] * C0 + 0.5
        )

        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=self.model.get_opacity,
            scales=self.model.get_scaling[:, :2],
            rotations=self.model.get_rotation,
            cov3D_precomp=None,
        )

        return {
            "render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": depth,
        }

    @torch.no_grad()
    def _densify(self, viewspace_grad, visibility_filter, radii):
        """基于视空间梯度幅值执行克隆/分裂自适应密度控制。

        Args:
            viewspace_grad: 预先保存的视空间点梯度（已在 optimizer.zero_grad 前克隆）。
            visibility_filter: 可见性掩码。
            radii: 各高斯的屏幕空间半径。
        """
        grad_norm = torch.norm(viewspace_grad[:, :2], dim=-1, keepdim=True)

        clone_mask = torch.logical_and(
            grad_norm > self.densify_grad_threshold,
            visibility_filter.unsqueeze(-1),
        ).squeeze(-1) & (self.model.get_scaling.max(dim=1).values < 0.02)

        split_mask = torch.logical_and(
            grad_norm > self.densify_grad_threshold,
            visibility_filter.unsqueeze(-1),
        ).squeeze(-1) & (self.model.get_scaling.max(dim=1).values >= 0.02)

        if clone_mask.any():
            self._clone_gaussians(clone_mask)
        if split_mask.any():
            self._split_gaussians(split_mask)

        prune_mask = (self.model._opacity < 0.005).squeeze(-1)
        if prune_mask.any():
            self._prune_gaussians(prune_mask)

    def _clone_gaussians(self, mask):
        """克隆梯度大但尺度小的高斯。"""
        self._dup_by_mask(mask, clone=True)

    def _split_gaussians(self, mask):
        """分裂梯度大且尺度大的高斯。"""
        self._dup_by_mask(mask, clone=False)

    def _dup_by_mask(self, mask, clone: bool):
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if idx.numel() == 0:
            return

        new_xyz = self.model._xyz[idx]
        new_rot = self.model._rotation[idx]
        new_scaling = self.model._scaling[idx]
        new_opacity = self.model._opacity[idx]
        new_f_dc = self.model._features_dc[idx]
        new_f_rest = self.model._features_rest[idx]
        new_normal = self.model._normal[idx]

        if not clone:
            # 在局部坐标系中采样偏移量（按各向异性尺度加权）
            std = self.model._scaling[idx].exp() * 0.5
            offset_local = torch.randn(idx.numel(), 3, device=std.device) * std
            # 用高斯四元数将局部偏移旋转到世界空间
            q = self.model._rotation[idx]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            R = torch.zeros(idx.numel(), 3, 3, device=q.device)
            R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
            R[:, 0, 1] = 2 * x * y - 2 * w * z
            R[:, 0, 2] = 2 * x * z + 2 * w * y
            R[:, 1, 0] = 2 * x * y + 2 * w * z
            R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
            R[:, 1, 2] = 2 * y * z - 2 * w * x
            R[:, 2, 0] = 2 * x * z - 2 * w * y
            R[:, 2, 1] = 2 * y * z + 2 * w * x
            R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
            offset_world = (R @ offset_local.unsqueeze(-1)).squeeze(-1)
            new_xyz = new_xyz + offset_world
            new_scaling = new_scaling - math.log(1.6)
            new_opacity = new_opacity * 0.5

        self._cat_gaussians(new_xyz, new_f_dc, new_f_rest, new_scaling, new_rot, new_opacity, new_normal)

    def _replace_params(self, old_params: dict, new_params: dict):
        """将优化器参数引用从旧 tensor 迁移到新 tensor。

        仅在形状未变时保留 Adam 状态（如 clone），形状变化时（prune/split）
        丢弃旧状态使优化器重新初始化，避免 shape mismatch 崩溃。
        """
        opt = self.optimizer
        for group in opt.param_groups:
            stored = group["params"][0]
            for name in old_params:
                if stored is old_params[name]:
                    group["params"][0] = new_params[name]
                    if stored in opt.state:
                        old_state = opt.state.pop(stored)
                        if old_state.get("exp_avg", stored).shape == new_params[name].shape:
                            opt.state[new_params[name]] = old_state
                    break

    _PARAM_NAMES = [
        "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_normal",
    ]

    def _cat_gaussians(self, xyz, f_dc, f_rest, scaling, rotation, opacity, normal):
        """将新高斯追加到当前模型中，并同步更新优化器参数引用。"""
        old_params = {name: getattr(self.model, name) for name in self._PARAM_NAMES}

        self.model._xyz = nn.Parameter(torch.cat([old_params["_xyz"], xyz], dim=0))
        self.model._features_dc = nn.Parameter(torch.cat([old_params["_features_dc"], f_dc], dim=0))
        self.model._features_rest = nn.Parameter(torch.cat([old_params["_features_rest"], f_rest], dim=0))
        self.model._scaling = nn.Parameter(torch.cat([old_params["_scaling"], scaling], dim=0))
        self.model._rotation = nn.Parameter(torch.cat([old_params["_rotation"], rotation], dim=0))
        self.model._opacity = nn.Parameter(torch.cat([old_params["_opacity"], opacity], dim=0))
        self.model._normal = nn.Parameter(torch.cat([old_params["_normal"], normal], dim=0))

        new_params = {name: getattr(self.model, name) for name in self._PARAM_NAMES}
        self._replace_params(old_params, new_params)

    def _prune_gaussians(self, mask):
        """删除透明度低于阈值的高斯，并同步更新优化器参数引用。"""
        keep = ~mask
        old_params = {name: getattr(self.model, name) for name in self._PARAM_NAMES}

        self.model._xyz = nn.Parameter(old_params["_xyz"][keep])
        self.model._features_dc = nn.Parameter(old_params["_features_dc"][keep])
        self.model._features_rest = nn.Parameter(old_params["_features_rest"][keep])
        self.model._scaling = nn.Parameter(old_params["_scaling"][keep])
        self.model._rotation = nn.Parameter(old_params["_rotation"][keep])
        self.model._opacity = nn.Parameter(old_params["_opacity"][keep])
        self.model._normal = nn.Parameter(old_params["_normal"][keep])

        new_params = {name: getattr(self.model, name) for name in self._PARAM_NAMES}
        self._replace_params(old_params, new_params)

    @torch.no_grad()
    def _reset_opacity(self):
        """周期性重置透明度以控制高斯数量。"""
        self.model._opacity.data.clamp_(max=0.01)


def main():
    parser = argparse.ArgumentParser(description="在 COLMAP 场景上训练 2D Gaussian Splatting")
    parser.add_argument("--source_path", "-s", required=True, help="COLMAP 场景路径（含 images/ 和 sparse/）")
    parser.add_argument("--model_path", "-m", required=True, help="模型检查点输出目录")
    parser.add_argument("--iterations", type=int, default=30_000, help="训练迭代次数")
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="SSIM 损失权重")
    parser.add_argument("--resolution", type=int, default=1600, help="训练图像长边最大像素（0=不缩放）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--position_lr_init", type=float, default=1.6e-4)
    parser.add_argument("--position_lr_final", type=float, default=1.6e-6)
    parser.add_argument("--scaling_lr", type=float, default=5e-3)
    parser.add_argument("--rotation_lr", type=float, default=1e-3)
    parser.add_argument("--opacity_lr", type=float, default=5e-2)
    parser.add_argument("--feature_lr", type=float, default=2.5e-3)
    parser.add_argument("--max_sh_degree", type=int, default=3, help="最大球谐阶数（0=无视角依赖）")
    parser.add_argument("--sh_degree_interval", type=int, default=1000, help="SH 阶数提升间隔（步）")
    parser.add_argument("--densify_grad_threshold", type=float, default=2e-5, help="密度控制梯度阈值（越小越多分裂）")
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=25_000)
    parser.add_argument("--opacity_reset_interval", type=int, default=2_000, help="不透明度周期性修剪间隔")
    args = parser.parse_args()

    model = GaussianModel2D(max_sh_degree=args.max_sh_degree, device=args.device)
    trainer = Trainer2DGS(
        model=model,
        source_path=args.source_path,
        model_path=args.model_path,
        iterations=args.iterations,
        position_lr_init=args.position_lr_init,
        position_lr_final=args.position_lr_final,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        opacity_lr=args.opacity_lr,
        feature_lr=args.feature_lr,
        lambda_dssim=args.lambda_dssim,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densify_grad_threshold=args.densify_grad_threshold,
        opacity_reset_interval=args.opacity_reset_interval,
        sh_degree_interval=args.sh_degree_interval,
        resolution=args.resolution,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
