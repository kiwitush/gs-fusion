"""
单张图像到 3D 生成：通过 Magic123 将单张 RGB 照片转换为 3D 资产。

两阶段流程: (1) 使用 SDS Loss 进行粗 NeRF 优化,
(2) 使用可微光栅化进行 Mesh 精细化。
内置背景去除和中心裁剪预处理。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAGIC123_DIR = PROJECT_ROOT / "Magic123"


class ImageTo3DGenerator:
    """单张图像 → 3D Mesh 生成器，内部调用 Magic123。

    使用方法::

        gen = ImageTo3DGenerator(output_dir="./outputs/object_c")
        mesh_path = gen.run_pipeline("data/object_c/photo.jpg")
    """

    def __init__(self, output_dir: str = "./outputs/object_c", device: str = "cuda"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    @staticmethod
    def remove_background(image_path: str, output_path: str, method: str = "rembg") -> str:
        """从输入图像中去除背景，输出 RGBA 图像。

        Args:
            image_path: 输入 RGB 图像路径。
            output_path: RGBA 输出路径。
            method: ``rembg``（基于 ML 的轻量方案）或 ``manual``（直通）。

        Returns:
            输出文件路径。
        """
        img = Image.open(image_path).convert("RGBA")

        if method == "rembg":
            try:
                from rembg import remove
                img_rgba = remove(img)
                img_rgba.save(output_path)
                print(f"[单图到3D] 已通过 rembg 去背景 → {output_path}")
                return output_path
            except ImportError:
                print("[单图到3D] rembg 未安装，退回直通模式。")

        img.save(output_path)
        print(f"[单图到3D] 图像已保存（未去背景）: {output_path}")
        return output_path

    @staticmethod
    def center_crop_to_square(image_path: str, output_path: str, size: int = 512) -> str:
        """中心裁剪并缩放为正方形。

        Args:
            image_path: 输入图像路径。
            output_path: 输出路径。
            size: 目标边长（像素）。

        Returns:
            输出文件路径。
        """
        img = Image.open(image_path).convert("RGBA")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), Image.LANCZOS)
        img.save(output_path)
        print(f"[单图到3D] 已裁剪并缩放至 {size}x{size} → {output_path}")
        return output_path

    def generate(
        self,
        image_path: str,
        elevation: float = 30.0,
        iters_coarse: int = 5_000,
        iters_fine: int = 2_000,
    ) -> str:
        """执行 Magic123 生成流程。

        Args:
            image_path: 预处理后的 RGBA 输入图像路径。
            elevation: 假设的相机仰角（度）。
            iters_coarse: NeRF 优化迭代次数。
            iters_fine: Mesh 精细化迭代次数。

        Returns:
            导出 OBJ Mesh 的路径。
        """
        if not MAGIC123_DIR.exists():
            raise FileNotFoundError(
                f"Magic123 未找到于 {MAGIC123_DIR}。"
                "请先克隆: git clone https://github.com/guochengqian/Magic123"
            )

        dest_img = self.output_dir / "input_rgba.png"
        Image.open(image_path).save(dest_img)

        config = {
            "image_path": str(dest_img),
            "output_dir": str(self.output_dir),
            "elevation": elevation,
            "iters_coarse": iters_coarse,
            "iters_fine": iters_fine,
        }
        config_path = self.output_dir / "magic123_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = [
            sys.executable, str(MAGIC123_DIR / "main.py"),
            "--config", str(config_path),
            "--workspace", str(self.output_dir),
        ]
        print(f"[单图到3D] 运行 Magic123: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(MAGIC123_DIR), env=env)

        mesh_path = self.output_dir / "mesh" / "model.obj"
        if mesh_path.exists():
            print(f"[单图到3D] Mesh 已导出 → {mesh_path}")
        else:
            print("[单图到3D] 警告: 未找到导出 Mesh，请检查 Magic123 日志。")

        return str(mesh_path)

    def run_pipeline(
        self,
        raw_image_path: str,
        background_method: str = "rembg",
        size: int = 512,
        elevation: float = 30.0,
    ) -> str:
        """端到端流程：去背景 → 裁剪 → Magic123 → Mesh。

        Args:
            raw_image_path: 原始 RGB 照片路径。
            background_method: 背景去除方法。
            size: 输出正方形边长。
            elevation: 相机仰角（度）。

        Returns:
            导出 Mesh 路径。
        """
        processed_dir = self.output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        rgba_path = str(processed_dir / "input_rgba.png")
        self.remove_background(raw_image_path, rgba_path, method=background_method)

        square_path = str(processed_dir / "input_square.png")
        self.center_crop_to_square(rgba_path, square_path, size=size)

        return self.generate(square_path, elevation=elevation)


def main():
    parser = argparse.ArgumentParser(description="通过 Magic123 从单张图像生成 3D 资产")
    parser.add_argument("--image", "-i", required=True, help="输入 RGB/RGBA 图像路径")
    parser.add_argument("--output_dir", "-o", default="./outputs/object_c")
    parser.add_argument("--no_bg_removal", action="store_true", help="跳过自动背景去除")
    parser.add_argument("--size", type=int, default=512, help="输出正方形边长（像素）")
    parser.add_argument("--elevation", type=float, default=30.0, help="假设的相机仰角（度）")
    parser.add_argument("--iters_coarse", type=int, default=5_000)
    parser.add_argument("--iters_fine", type=int, default=2_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generator = ImageTo3DGenerator(output_dir=args.output_dir, device=args.device)

    if args.no_bg_removal:
        mesh_path = generator.generate(args.image, elevation=args.elevation)
    else:
        mesh_path = generator.run_pipeline(
            args.image,
            background_method="rembg",
            size=args.size,
            elevation=args.elevation,
        )

    print(f"\n生成完成。Mesh: {mesh_path}")
    print(f"下一步: 将 Mesh 转换为 2DGS →")
    print(f"  python src/utils/mesh_to_gs.py -i {mesh_path} -o {args.output_dir}/object_c.ply")


if __name__ == "__main__":
    main()
