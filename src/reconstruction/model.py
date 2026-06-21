"""
2D Gaussian Splatting 模型定义。

封装 diff-surfel-rasterization CUDA 内核并提供 PyTorch Parameter
对象用于所有可优化属性（位置、透明度、缩放、旋转、球谐系数）。
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement

SH_DEGREE = 3
SH_COEFFS = (SH_DEGREE + 1) ** 2
FEATURES_PER_GAUSSIAN = SH_COEFFS * 3

PLY_ATTRS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{k}" for k in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def write_2dgs_ply(path: str, xyz, normals, f_dc, f_rest, opacity, scale, rotation) -> None:
    """将 2DGS 属性数组写入二进制 PLY 文件（供 model / fuse / mesh_to_gs 共用）。"""
    N = xyz.shape[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    dtype = [(name, "f4") for name in PLY_ATTRS]
    data = np.zeros(N, dtype=dtype)

    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["nx"], data["ny"], data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for k in range(45):
        data[f"f_rest_{k}"] = f_rest[:, k]
    data["opacity"] = opacity.reshape(-1)
    data["scale_0"], data["scale_1"], data["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    PlyData([PlyElement.describe(data, "vertex")]).write(path)
    print(f"[PLY] 已保存 {N} 个高斯 → {path}")


def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """计算将世界 Z 轴旋转到每个法线方向的四元数 (w,x,y,z)，向量化实现。"""
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    N = normals.shape[0]
    quats = np.zeros((N, 4), dtype=np.float32)

    n = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10)
    d = n @ world_z  # (N,) 点积

    aligned = d > 0.99999
    quats[aligned, 0] = 1.0  # w=1, 单位四元数 (1,0,0,0)

    opposed = d < -0.99999
    quats[opposed, 1] = 1.0  # x=1, 绕 X 轴 180° (0,1,0,0)

    general = ~(aligned | opposed)
    if general.any():
        axis = np.cross(world_z, n[general])
        axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-10)
        angle = np.arccos(np.clip(d[general], -1.0, 1.0))
        half = angle / 2.0
        s = np.sin(half)
        quats[general, 0] = np.cos(half)     # w
        quats[general, 1] = axis[:, 0] * s   # x
        quats[general, 2] = axis[:, 1] * s   # y
        quats[general, 3] = axis[:, 2] * s   # z

    return quats


class GaussianModel2D(nn.Module):
    """可微的 2D 高斯面片云。

    所有属性都以 ``nn.Parameter`` 形式存储，可被任意 PyTorch 优化器直接优化。
    模型同时维护密度控制（adaptive density control）所需的辅助张量。
    """

    def __init__(self, max_sh_degree: int = 3, device: str = "cuda"):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        self._device = device

        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._features_dc = nn.Parameter(torch.empty(0, 3, 1, device=device))
        self._features_rest = nn.Parameter(torch.empty(0, 3, 15, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=device))
        self._normal = nn.Parameter(torch.empty(0, 3, device=device))

        self.max_radii2D = torch.empty(0, device=device)
        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=2)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    def create_from_pcd(self, pcd: Dict[str, np.ndarray], spatial_lr_scale: float = 1.0):
        """从 COLMAP SfM 稀疏点云初始化高斯。

        当 COLMAP 未提供法线（全零向量）时，通过局部 PCA 自动估计表面法线方向。
        2D surfel 缩放初始化为各向异性：切平面内 (s_xy, s_xy) 与极薄的法线方向 (s_z)。

        Args:
            pcd: 包含 ``points`` (N,3)、``colors`` (N,3)、``normals`` (N,3) 的字典。
            spatial_lr_scale: 学习率缩放因子。
        """
        points = torch.tensor(pcd["points"], dtype=torch.float32, device=self._device)
        colors = torch.tensor(pcd["colors"], dtype=torch.float32, device=self._device)
        normals = torch.tensor(pcd["normals"], dtype=torch.float32, device=self._device)

        N = points.shape[0]
        print(f"[2DGS 模型] 从 {N} 个点云初始化, spatial_lr_scale={spatial_lr_scale}")

        chunk_size = 4096
        dist2_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_dists = torch.cdist(points[start:end], points)
            chunk_dists[chunk_dists < 1e-8] = float("inf")
            dist2_list.append(torch.min(chunk_dists, dim=1).values)
        dist2_nearest = torch.cat(dist2_list, dim=0)

        avg_dist = max(dist2_nearest.mean().item(), 1e-6)

        # 如果 COLMAP 未提供法线（全零），用局部 PCA 估计
        if normals.abs().max().item() < 1e-8:
            points_np = points.cpu().numpy()
            from scipy.spatial import KDTree
            tree = KDTree(points_np)
            k = min(16, N - 1)
            _, idx = tree.query(points_np, k=k + 1)
            idx = idx[:, 1:]  # 移除自身
            neighbors = points_np[idx]
            centered = neighbors - neighbors.mean(axis=1, keepdims=True)
            cov = np.einsum("nki,nkj->nij", centered, centered) / k
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals_est = eigvecs[:, :, 0]  # 最小特征值方向 = 法线
            normals = torch.tensor(normals_est, dtype=torch.float32, device=self._device)
            # 确保法线朝向一致性（指向相机平均方向）
            centroid = points.mean(dim=0)
            to_centroid = centroid - points
            flip = (normals * to_centroid).sum(dim=-1, keepdim=True) < 0
            normals = torch.where(flip, -normals, normals)

        # 2D surfel 的各向异性缩放初始值：切平面半径为 avg_dist*0.25，法线方向极薄
        surfel_radii = torch.full((N, 1), avg_dist * 0.25, device=self._device)
        surfel_thick = torch.full((N, 1), avg_dist * 0.001, device=self._device)
        init_scale = torch.log(torch.cat([surfel_radii, surfel_radii, surfel_thick], dim=1))

        # 用法线初始化旋转四元数
        init_rot = _normals_to_quaternions(normals.cpu().numpy())
        init_rot = torch.tensor(init_rot, dtype=torch.float32, device=self._device)
        # quaternion order: model uses (w, x, y, z) → [0]=w, [1]=x, [2]=y, [3]=z

        self._xyz = nn.Parameter(points.contiguous())
        self._features_dc = nn.Parameter(colors.unsqueeze(2).contiguous())
        self._features_rest = nn.Parameter(torch.zeros(N, 3, 15, device=self._device))
        self._scaling = nn.Parameter(init_scale.contiguous())
        self._rotation = nn.Parameter(init_rot.contiguous())
        self._opacity = nn.Parameter(0.1 * torch.ones(N, 1, device=self._device))
        self._normal = nn.Parameter(normals.contiguous())
        self.spatial_lr_scale = spatial_lr_scale

    def oneup_SH_degree(self):
        """逐渐提升球谐阶数以减小高频伪影。"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"[2DGS 模型] SH 阶数提升至 {self.active_sh_degree}")

    def save_ply(self, path: str) -> None:
        """将高斯导出为 2DGS 格式的 PLY 文件。"""
        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        f_rest = self._features_rest.detach().cpu().numpy().reshape(-1, 45)
        opacities = self._opacity.detach().cpu().numpy().reshape(-1)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        write_2dgs_ply(
            path, xyz, normals, f_dc, f_rest,
            opacities.reshape(-1, 1) if opacities.ndim == 1 else opacities,
            scale, rotation,
        )

    def load_ply(self, path: str) -> None:
        """从 2DGS PLY 文件加载高斯。"""
        plydata = PlyData.read(path)
        raw = plydata["vertex"]
        verts = raw.data if hasattr(raw, "data") else raw

        xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)
        if "nx" in verts.dtype.names and "ny" in verts.dtype.names and "nz" in verts.dtype.names:
            normals = np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=1)
        else:
            normals = np.zeros_like(xyz)

        f_dc = np.stack([verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"]], axis=1)
        f_rest = np.zeros((xyz.shape[0], 45), dtype=np.float32)
        for k in range(45):
            key = f"f_rest_{k}"
            if key in verts.dtype.names:
                f_rest[:, k] = verts[key]

        opacity = verts["opacity"]
        scale = np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=1)
        rotation = np.stack([verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"]], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=self._device))
        self._features_dc = nn.Parameter(
            torch.tensor(f_dc, dtype=torch.float32, device=self._device).unsqueeze(2)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(f_rest, dtype=torch.float32, device=self._device).reshape(-1, 3, 15)
        )
        self._scaling = nn.Parameter(torch.tensor(scale, dtype=torch.float32, device=self._device))
        self._rotation = nn.Parameter(torch.tensor(rotation, dtype=torch.float32, device=self._device))
        self._opacity = nn.Parameter(
            torch.tensor(opacity, dtype=torch.float32, device=self._device).unsqueeze(1)
        )
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float32, device=self._device))

        self.active_sh_degree = self.max_sh_degree

        print(f"[2DGS 模型] 已加载 {xyz.shape[0]} 个高斯 ← {path}")

    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点（PyTorch 格式）。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "xyz": self._xyz,
                "features_dc": self._features_dc,
                "features_rest": self._features_rest,
                "scaling": self._scaling,
                "rotation": self._rotation,
                "opacity": self._opacity,
                "normal": self._normal,
                "active_sh_degree": self.active_sh_degree,
            },
            path,
        )
        print(f"[2DGS 模型] 检查点已保存 → {path}")

    def load_checkpoint(self, path: str) -> None:
        """从 PyTorch 检查点恢复模型。"""
        ckpt = torch.load(path, map_location=self._device)
        self._xyz = nn.Parameter(ckpt["xyz"])
        self._features_dc = nn.Parameter(ckpt["features_dc"])
        self._features_rest = nn.Parameter(ckpt["features_rest"])
        self._scaling = nn.Parameter(ckpt["scaling"])
        self._rotation = nn.Parameter(ckpt["rotation"])
        self._opacity = nn.Parameter(ckpt["opacity"])
        self._normal = nn.Parameter(ckpt["normal"])
        self.active_sh_degree = ckpt.get("active_sh_degree", self.max_sh_degree)
        print(f"[2DGS 模型] 检查点已加载 ← {path}")
