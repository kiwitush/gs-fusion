"""
Mesh / 隐式场 → 2D Gaussian Splatting 点云转换器。

将有纹理的三角网格（OBJ / PLY / GLB）或 NeRF 风格的隐式场转换为
2DGS 兼容的 PLY 文件。每个输出点是一个 2D 平面高斯面片（surfel），
包含位置、法线、球谐颜色系数、透明度、各向异性缩放和四元数旋转。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.model import write_2dgs_ply

# 球谐 DC 分量常数: 1/(2*sqrt(pi))
C0 = 0.28209479177387814


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """将 (N, 3) RGB [0,1] 数组转换为 3 阶球谐系数 (N, 48)。

    仅填充 DC（0 阶）分量，高阶分量保持为零。
    2DGS 渲染器会在需要时对高阶分量进行插值。
    """
    N = rgb.shape[0]
    sh = np.zeros((N, 48), dtype=np.float32)
    sh[:, 0] = rgb[:, 0] / C0
    sh[:, 1] = rgb[:, 1] / C0
    sh[:, 2] = rgb[:, 2] / C0
    return sh


class MeshToGSConverter:
    """将带纹理的三角网格转换为 2DGS 高斯点云。

    对每个三角面片进行采样，生成法线与面片法线对齐、主轴位于面片平面内的
    定向平面面片（surfel）。
    """

    def __init__(
        self,
        num_samples: int = 100_000,
        min_opacity: float = 0.1,
        max_opacity: float = 0.9,
        surfel_scale_xy: float = 0.008,
        surfel_scale_z_factor: float = 0.001,
    ):
        self.num_samples = num_samples
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.surfel_scale_xy = surfel_scale_xy
        self.surfel_scale_z_factor = surfel_scale_z_factor

    def convert(self, mesh_path: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """执行完整转换流程。

        Returns:
            (points, colors) — 各为 (N,3) 数组，供检查使用。
        """
        mesh = self._load_mesh(mesh_path)
        verts, faces, face_normals, vert_colors = self._extract_geometry(mesh)
        points, normals, colors = self._sample_surfels(
            verts, faces, face_normals, vert_colors, self.num_samples
        )
        gaussian_attrs = self._build_2dgs_attributes(points, normals, colors)
        self._save_ply(output_path, gaussian_attrs)
        print(f"[mesh_to_gs] 已保存 {points.shape[0]} 个 surfel → {output_path}")
        return points, colors

    @staticmethod
    def _load_mesh(path: str):
        """根据文件扩展名分派到对应的加载器。"""
        ext = Path(path).suffix.lower()
        if ext == ".obj":
            return MeshToGSConverter._load_obj(path)
        if ext == ".ply":
            return MeshToGSConverter._load_ply(path)
        if ext in (".glb", ".gltf"):
            import trimesh
            return trimesh.load(path, force="mesh")
        raise ValueError(f"不支持的网格格式: {ext}")

    @staticmethod
    def _load_obj(path: str) -> dict:
        """轻量级 OBJ 加载器，无需 trimesh 依赖。"""
        verts, faces, colors = [], [], []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    verts.append([float(x) for x in parts[1:4]])
                    if len(parts) >= 7:
                        colors.append([float(x) for x in parts[4:7]])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    idx = [int(p.split("/")[0]) - 1 for p in parts]
                    if len(idx) == 3:
                        faces.append(idx)
                    elif len(idx) == 4:
                        faces.append([idx[0], idx[1], idx[2]])
                        faces.append([idx[0], idx[2], idx[3]])

        verts = np.array(verts, dtype=np.float32)
        faces = np.array(faces, dtype=np.int64).reshape(-1, 3)
        if colors:
            colors = np.array(colors, dtype=np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.full_like(verts, 0.7)
        return {"vertices": verts, "faces": faces, "vertex_colors": colors}

    @staticmethod
    def _load_ply(path: str):
        """通过 trimesh 加载 PLY 文件。"""
        import trimesh
        return trimesh.load(path)

    @staticmethod
    def _extract_geometry(mesh):
        """将不同网格表示统一为原生数组格式。

        Returns:
            verts (N,3), faces (M,3), face_normals (M,3), vert_colors (N,3)
        """
        import trimesh

        if isinstance(mesh, trimesh.Trimesh):
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            normals = np.asarray(mesh.face_normals, dtype=np.float32)
            if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
                colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = np.full_like(verts, 0.7)
        elif isinstance(mesh, dict):
            verts = mesh["vertices"].astype(np.float32)
            faces = mesh["faces"].astype(np.int64)
            if faces.ndim == 1:
                faces = faces.reshape(-1, 3)
            v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
            normals = normals / norm_len
            colors = mesh.get("vertex_colors", np.full_like(verts, 0.7))
            if isinstance(colors, np.ndarray) and colors.max() > 1.0:
                colors = colors / 255.0
        else:
            raise TypeError(f"不支持的网格类型: {type(mesh)}")

        return verts, faces, normals, colors

    @staticmethod
    def _sample_surfels(verts, faces, face_normals, vert_colors, num_samples):
        """在网格表面均匀采样 surfel。

        使用按面积加权 + 重心坐标插值的方式在三角面片内采样。
        """
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
        probs = areas / areas.sum()

        face_counts = np.random.multinomial(num_samples, probs)
        all_points, all_normals, all_colors = [], [], []

        for fidx, count in enumerate(face_counts):
            if count == 0:
                continue
            r1 = np.random.rand(count, 1)
            r2 = np.random.rand(count, 1)
            mask = r1 + r2 > 1.0
            r1[mask] = 1.0 - r1[mask]
            r2[mask] = 1.0 - r2[mask]

            v0f, v1f, v2f = v0[fidx], v1[fidx], v2[fidx]
            pts = v0f + r1 * (v1f - v0f) + r2 * (v2f - v0f)
            nrm = np.tile(face_normals[fidx], (count, 1))

            if vert_colors is not None:
                c0 = vert_colors[faces[fidx, 0]]
                c1 = vert_colors[faces[fidx, 1]]
                c2 = vert_colors[faces[fidx, 2]]
                cols = c0 + r1 * (c1 - c0) + r2 * (c2 - c0)
            else:
                cols = np.full_like(pts, 0.7)

            all_points.append(pts)
            all_normals.append(nrm)
            all_colors.append(cols)

        return (
            np.concatenate(all_points, axis=0).astype(np.float32),
            np.concatenate(all_normals, axis=0).astype(np.float32),
            np.concatenate(all_colors, axis=0).astype(np.float32),
        )

    def _build_2dgs_attributes(self, points, normals, colors):
        """从原始 surfel 数据构建完整的 2DGS 属性字典。"""
        N = points.shape[0]

        opacity = np.full((N, 1), 0.7, dtype=np.float32)
        opacity += np.random.randn(N, 1).astype(np.float32) * 0.1
        opacity = np.clip(opacity, self.min_opacity, self.max_opacity)

        # 各向异性缩放（log-space）：XY 方向为盘面、Z 方向极薄
        scales = np.zeros((N, 3), dtype=np.float32)
        scales[:, 0] = np.log(self.surfel_scale_xy)
        scales[:, 1] = np.log(self.surfel_scale_xy)
        scales[:, 2] = np.log(max(self.surfel_scale_xy * self.surfel_scale_z_factor, 1e-10))

        rot = self._normals_to_quaternions(normals)
        sh = rgb_to_sh(colors)

        return dict(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            nx=normals[:, 0], ny=normals[:, 1], nz=normals[:, 2],
            f_dc_0=sh[:, 0], f_dc_1=sh[:, 1], f_dc_2=sh[:, 2],
            **{f"f_rest_{k}": sh[:, 3 + k] for k in range(45)},
            opacity=opacity[:, 0],
            scale_0=scales[:, 0], scale_1=scales[:, 1], scale_2=scales[:, 2],
            rot_0=rot[:, 0], rot_1=rot[:, 1], rot_2=rot[:, 2], rot_3=rot[:, 3],
        )

    @staticmethod
    def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
        """计算将世界 Z 轴 (0,0,1) 旋转到每个法线方向的四元数（向量化）。"""
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        N = normals.shape[0]
        quats = np.zeros((N, 4), dtype=np.float32)

        n = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10)
        d = np.dot(n, world_z)

        # 几乎对齐：返回单位四元数 (w=1, x=0, y=0, z=0)
        aligned = d > 0.99999
        quats[aligned] = [1.0, 0.0, 0.0, 0.0]

        # 几乎反向：绕 X 轴旋转 180° (w=0, x=1, y=0, z=0)
        opposed = d < -0.99999
        quats[opposed] = [0.0, 1.0, 0.0, 0.0]

        # 一般情况
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

    @staticmethod
    def _save_ply(path: str, attrs: dict) -> None:
        """将 2DGS 属性写入二进制 PLY 文件（委托给共享的 write_2dgs_ply）。"""
        N = attrs["x"].shape[0]
        xyz = np.stack([attrs["x"], attrs["y"], attrs["z"]], axis=1)
        normals = np.stack([attrs["nx"], attrs["ny"], attrs["nz"]], axis=1)
        f_dc = np.stack([attrs["f_dc_0"], attrs["f_dc_1"], attrs["f_dc_2"]], axis=1)
        f_rest = np.stack([attrs[f"f_rest_{k}"] for k in range(45)], axis=1)
        opacity = np.asarray(attrs["opacity"], dtype=np.float32).reshape(-1, 1)
        scale = np.stack([attrs["scale_0"], attrs["scale_1"], attrs["scale_2"]], axis=1)
        rotation = np.stack([attrs["rot_0"], attrs["rot_1"], attrs["rot_2"], attrs["rot_3"]], axis=1)
        write_2dgs_ply(path, xyz, normals, f_dc, f_rest, opacity, scale, rotation)


def main():
    parser = argparse.ArgumentParser(description="将有纹理的 Mesh 转换为 2DGS PLY 格式")
    parser.add_argument("--input", "-i", required=True, help="输入 Mesh 路径 (.obj/.ply/.glb)")
    parser.add_argument("--output", "-o", required=True, help="输出 .ply 文件路径")
    parser.add_argument("--num_samples", type=int, default=100_000, help="采样的 surfel 数量")
    parser.add_argument("--surfel_scale", type=float, default=0.008, help="surfel 盘面半径（局部 XY 方向）")
    parser.add_argument("--min_opacity", type=float, default=0.1)
    parser.add_argument("--max_opacity", type=float, default=0.9)
    args = parser.parse_args()

    converter = MeshToGSConverter(
        num_samples=args.num_samples,
        surfel_scale_xy=args.surfel_scale,
        min_opacity=args.min_opacity,
        max_opacity=args.max_opacity,
    )
    converter.convert(args.input, args.output)


if __name__ == "__main__":
    main()
