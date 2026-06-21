"""
多视角漫游视频渲染器。

加载合并后的 2DGS 场景，沿相机轨迹渲染并输出 MP4 视频。
支持内置轨迹（圆形/螺旋/关键帧）和自定义 JSON 轨迹文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fusion.camera import (
    circular_trajectory,
    spiral_trajectory,
    keyframe_trajectory,
    load_trajectory,
    save_trajectory,
)
from src.reconstruction.render import Renderer2DGS


class FusionVideoRenderer:
    """从合并后的 2DGS 场景渲染漫游视频。

    使用方法::

        renderer = FusionVideoRenderer("outputs/fused_scene.ply")
        renderer.render("outputs/video.mp4", trajectory="spiral")
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.renderer = Renderer2DGS(checkpoint_path, device=device)
        self.device = device

    def render(
        self,
        output_path: str,
        trajectory: str = "circular",
        trajectory_config: Optional[dict] = None,
        fps: int = 30,
        bg_color: tuple = (0.0, 0.0, 0.0),
    ) -> str:
        """生成最终视频。

        Args:
            output_path: 输出 MP4 文件路径。
            trajectory: 轨迹类型，可选 ``circular``、``spiral``、``keyframe``
                        或 JSON 文件路径。
            trajectory_config: 轨迹构建器的额外参数。
            fps: 视频帧率。
            bg_color: RGB 背景颜色。

        Returns:
            输出文件的绝对路径。
        """
        cameras = self._build_trajectory(trajectory, trajectory_config or {})

        traj_json = Path(output_path).with_suffix(".json")
        save_trajectory(cameras, str(traj_json))

        self.renderer.render_to_video(cameras, output_path, fps=fps, bg_color=bg_color)
        return os.path.abspath(output_path)

    def _build_trajectory(self, trajectory: str, config: dict):
        """根据轨迹类型分派构建器。"""
        if trajectory.endswith(".json"):
            return load_trajectory(trajectory, device=self.device)

        cfg = {
            "center": config.get("center", (0.0, 0.0, 0.0)),
            "fov_y": config.get("fov_y", 0.539),
            "aspect": config.get("aspect", 16.0 / 9.0),
            "height_res": config.get("height_res", 1080),
            "width_res": config.get("width_res", 1920),
            "device": self.device,
        }

        if trajectory == "circular":
            cfg["num_frames"] = config.get("num_frames", 120)
            cfg["radius"] = config.get("radius", 3.0)
            cfg["height"] = config.get("height", 1.5)
            return circular_trajectory(**cfg)

        elif trajectory == "spiral":
            cfg["num_frames"] = config.get("num_frames", 180)
            cfg["start_radius"] = config.get("start_radius", 5.0)
            cfg["end_radius"] = config.get("end_radius", 1.5)
            cfg["start_height"] = config.get("start_height", 3.0)
            cfg["end_height"] = config.get("end_height", 0.5)
            cfg["turns"] = config.get("turns", 2.5)
            return spiral_trajectory(**cfg)

        elif trajectory == "keyframe":
            keyframes = config.get("keyframes", [
                ([-4, 0, 2], [0, 0, 0]),
                ([4, 0, 2], [0, 0, 0]),
                ([0, 4, 2], [0, 0, 0]),
            ])
            keyframes = [
                (np.array(eye, dtype=np.float32), np.array(center, dtype=np.float32))
                for eye, center in keyframes
            ]
            cfg["num_frames"] = config.get("num_frames", 120)
            cfg["keyframes"] = keyframes
            return keyframe_trajectory(**cfg)

        else:
            raise ValueError(f"未知的轨迹类型: {trajectory}")


def main():
    parser = argparse.ArgumentParser(description="从合并后的 2DGS 场景渲染漫游视频")
    parser.add_argument("--scene", "-s", required=True, help="合并后的场景 PLY 或检查点路径")
    parser.add_argument("--output", "-o", required=True, help="输出 MP4 路径")
    parser.add_argument(
        "--trajectory", "-t", default="circular",
        help="轨迹类型 (circular/spiral/keyframe) 或 JSON 文件路径",
    )
    parser.add_argument("--center_x", type=float, default=0.0)
    parser.add_argument("--center_y", type=float, default=0.0)
    parser.add_argument("--center_z", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=3.0, help="圆形轨迹半径")
    parser.add_argument("--orbit_height", type=float, default=1.5, help="圆形轨迹相机高度")
    parser.add_argument("--start_radius", type=float, default=5.0, help="螺旋轨迹起始半径")
    parser.add_argument("--end_radius", type=float, default=1.5, help="螺旋轨迹终止半径")
    parser.add_argument("--start_height", type=float, default=3.0, help="螺旋轨迹起始高度")
    parser.add_argument("--end_height", type=float, default=0.5, help="螺旋轨迹终止高度")
    parser.add_argument("--turns", type=float, default=2.5, help="螺旋轨迹圈数")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res_width", type=int, default=1920, help="渲染图像宽度")
    parser.add_argument("--res_height", type=int, default=1080, help="渲染图像高度")
    parser.add_argument("--bg_r", type=float, default=0.0)
    parser.add_argument("--bg_g", type=float, default=0.0)
    parser.add_argument("--bg_b", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = {
        "center": (args.center_x, args.center_y, args.center_z),
        "radius": args.radius,
        "height": args.orbit_height,
        "start_radius": args.start_radius,
        "end_radius": args.end_radius,
        "start_height": args.start_height,
        "end_height": args.end_height,
        "turns": args.turns,
        "num_frames": args.num_frames,
        "aspect": args.res_width / args.res_height,
        "height_res": args.res_height,
        "width_res": args.res_width,
    }

    renderer = FusionVideoRenderer(checkpoint_path=args.scene, device=args.device)
    output = renderer.render(
        output_path=args.output,
        trajectory=args.trajectory,
        trajectory_config=config,
        fps=args.fps,
        bg_color=(args.bg_r, args.bg_g, args.bg_b),
    )

    print(f"\n视频已渲染: {output}")


if __name__ == "__main__":
    main()
