"""
使用训练好的 2DGS 模型进行新视角渲染。

支持单张图像渲染以及沿预设相机轨迹的批量渲染和视频编码。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.model import GaussianModel2D

try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_RASTERIZER = True
except ImportError:
    HAS_RASTERIZER = False


class Renderer2DGS:
    """训练好的 2DGS 模型离线渲染器。"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        if not HAS_RASTERIZER:
            raise RuntimeError("diff-surfel-rasterization 未安装。")

        self.device = device
        self.model = GaussianModel2D(max_sh_degree=3, device=device)

        if checkpoint_path.endswith(".ply"):
            self.model.load_ply(checkpoint_path)
        else:
            self.model.load_checkpoint(checkpoint_path)
        self.model.eval()

    def render_view(self, camera: dict, bg_color: tuple = (0.0, 0.0, 0.0)) -> dict:
        """渲染单个视角。

        Args:
            camera: 相机参数字典，包含:
                - height, width (int): 图像分辨率
                - viewmatrix (4×4 Tensor): 世界→相机变换
                - projmatrix (4×4 Tensor): 相机→NDC 投影
                - fovx, fovy (float): 视场角（弧度）
                - campos (3, Tensor): 世界空间相机位置
            bg_color: RGB 背景颜色三元组。

        Returns:
            包含 render (C,H,W)、depth、alpha、radii 的字典。
        """
        bg = torch.tensor(bg_color, device=self.device)

        raster_settings = GaussianRasterizationSettings(
            image_height=camera["height"],
            image_width=camera["width"],
            tanfovx=torch.tan(torch.tensor(camera["fovx"] * 0.5)),
            tanfovy=torch.tan(torch.tensor(camera["fovy"] * 0.5)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=camera["viewmatrix"],
            projmatrix=camera["projmatrix"],
            sh_degree=self.model.active_sh_degree,
            campos=camera["campos"],
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        C0 = 0.28209479177387814
        features = self.model.get_features
        colors_precomp = features[:, :, 0] * C0 + 0.5

        rendered, radii, depth = rasterizer(
            means3D=self.model.get_xyz,
            means2D=torch.zeros_like(self.model.get_xyz, requires_grad=True),
            shs=None,
            colors_precomp=colors_precomp,
            opacities=self.model.get_opacity,
            scales=self.model.get_scaling[:, :2],
            rotations=self.model.get_rotation,
            cov3D_precomp=None,
        )

        return {"render": rendered, "depth": depth, "alpha": depth, "radii": radii}

    @staticmethod
    def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
        """将 (C,H,W) 渲染张量转换为 (H,W,C) uint8 numpy 数组。"""
        img = tensor.clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8).transpose(1, 2, 0)

    def render_trajectory(
        self,
        cameras: list,
        output_dir: str,
        bg_color: tuple = (0.0, 0.0, 0.0),
        save_depth: bool = False,
    ):
        """沿轨迹渲染一组图像并保存为 PNG。

        Args:
            cameras: 相机字典列表（见 render_view）。
            output_dir: 输出目录。
            bg_color: 背景色。
            save_depth: 是否同时导出深度图。
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, cam in enumerate(tqdm(cameras, desc="渲染中")):
            with torch.no_grad():
                result = self.render_view(cam, bg_color)

            img_np = self._to_numpy_image(result["render"])
            Image.fromarray(img_np).save(os.path.join(output_dir, f"frame_{idx:05d}.png"))

            if save_depth:
                depth = result["depth"]
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)
                depth_np = depth.cpu().numpy()
                d_min, d_max = depth_np.min(), depth_np.max()
                depth_np = (depth_np - d_min) / (d_max - d_min + 1e-10)
                depth_np = (depth_np * 255).astype(np.uint8).transpose(1, 2, 0).squeeze()
                Image.fromarray(depth_np).save(os.path.join(output_dir, f"depth_{idx:05d}.png"))

        print(f"[渲染] 已保存 {len(cameras)} 帧 → {output_dir}")

    def render_to_video(
        self,
        cameras: list,
        output_path: str,
        fps: int = 30,
        bg_color: tuple = (0.0, 0.0, 0.0),
    ):
        """沿轨迹渲染并编码为 MP4 视频。"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 先渲染所有帧到临时目录，再用 imageio-ffmpeg 编码
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="frames_")

        for idx, cam in enumerate(tqdm(cameras, desc="渲染帧")):
            with torch.no_grad():
                result = self.render_view(cam, bg_color)
            img_np = self._to_numpy_image(result["render"])
            Image.fromarray(img_np).save(os.path.join(tmpdir, f"frame_{idx:05d}.png"))

        # 尝试用 imageio-ffmpeg 编码（自带 ffmpeg 二进制）
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            import subprocess
            subprocess.run(
                [exe, "-y", "-r", str(fps), "-i", f"{tmpdir}/frame_%05d.png",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            print(f"[渲染] 视频编码失败，帧已保存至 {tmpdir}，可手动: ffmpeg -r {fps} -i frame_%05d.png ...")
            return

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"[渲染] 视频已保存 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从训练好的 2DGS 模型渲染新视角")
    parser.add_argument("--checkpoint", "-c", required=True, help=".pth 或 .ply 检查点路径")
    parser.add_argument("--output_dir", "-o", required=True, help="渲染帧输出目录")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    renderer = Renderer2DGS(checkpoint_path=args.checkpoint, device=args.device)
    dummy_camera = {
        "height": 1080,
        "width": 1920,
        "viewmatrix": torch.eye(4, device=args.device),
        "projmatrix": torch.eye(4, device=args.device),
        "fovx": 0.691,
        "fovy": 0.539,
        "campos": torch.tensor([0.0, 0.0, 3.0], device=args.device),
    }
    renderer.render_trajectory([dummy_camera], output_dir=args.output_dir)


if __name__ == "__main__":
    main()
