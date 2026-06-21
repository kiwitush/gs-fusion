"""
2DGS 模型导出工具。

将训练好的检查点转换为独立的 PLY 文件，支持降采样和统计分析。
适用于下游融合、可视化或 Blender/MeshLab 中的手动编辑。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.model import GaussianModel2D


def export_to_ply(checkpoint_path: str, output_path: str, device: str = "cuda") -> str:
    """加载 2DGS 检查点并导出为 2DGS PLY 文件。

    Args:
        checkpoint_path: .pth 或 .ply 路径。
        output_path: 目标 .ply 路径。
        device: PyTorch 设备。

    Returns:
        输出文件的绝对路径。
    """
    model = GaussianModel2D(max_sh_degree=3, device=device)

    if checkpoint_path.endswith(".ply"):
        model.load_ply(checkpoint_path)
    else:
        model.load_checkpoint(checkpoint_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save_ply(output_path)
    return os.path.abspath(output_path)


def export_downsampled(
    checkpoint_path: str,
    output_path: str,
    target_count: int = 50_000,
    device: str = "cuda",
) -> str:
    """通过均匀随机采样导出降采样版本。

    Args:
        checkpoint_path: .pth 或 .ply 路径。
        output_path: 目标 .ply 路径。
        target_count: 目标高斯数量。
        device: PyTorch 设备。

    Returns:
        输出文件的绝对路径。
    """
    model = GaussianModel2D(max_sh_degree=3, device=device)

    if checkpoint_path.endswith(".ply"):
        model.load_ply(checkpoint_path)
    else:
        model.load_checkpoint(checkpoint_path)

    N = model.num_gaussians
    if N <= target_count:
        print(f"[导出] {N} 个高斯 ≤ {target_count}，导出全部。")
        model.save_ply(output_path)
        return os.path.abspath(output_path)

    indices = torch.randperm(N, device=device)[:target_count]
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_normal"]:
        tensor = getattr(model, attr)
        setattr(model, attr, torch.nn.Parameter(tensor[indices]))

    model.save_ply(output_path)
    print(f"[导出] 降采样: {N} → {target_count} 个高斯。")
    return os.path.abspath(output_path)


def export_statistics(checkpoint_path: str, device: str = "cuda") -> dict:
    """打印并返回 2DGS 模型的汇总统计信息。

    Args:
        checkpoint_path: .pth 或 .ply 路径。
        device: PyTorch 设备。

    Returns:
        统计信息字典。
    """
    model = GaussianModel2D(max_sh_degree=3, device=device)

    if checkpoint_path.endswith(".ply"):
        model.load_ply(checkpoint_path)
    else:
        model.load_checkpoint(checkpoint_path)

    xyz = model._xyz.detach()
    scale = model._scaling.detach().exp()
    opacity = model._opacity.detach()

    stats = {
        "num_gaussians": int(xyz.shape[0]),
        "xyz_center": xyz.mean(dim=0).tolist(),
        "xyz_extent": (xyz.max(dim=0).values - xyz.min(dim=0).values).tolist(),
        "scale_mean": scale.mean(dim=0).tolist(),
        "scale_std": scale.std(dim=0).tolist(),
        "opacity_mean": opacity.mean().item(),
        "opacity_std": opacity.std().item(),
        "opacity_min": opacity.min().item(),
        "opacity_max": opacity.max().item(),
    }

    print("\n" + "=" * 50)
    print(" 模型统计信息")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="将 2DGS 模型导出为 PLY 格式")
    parser.add_argument("--checkpoint", "-c", required=True, help=".pth 或 .ply 检查点路径")
    parser.add_argument("--output", "-o", required=True, help="输出 .ply 路径")
    parser.add_argument("--downsample", type=int, default=0, help="降采样至 N 个高斯（0 = 不降采样）")
    parser.add_argument("--stats", action="store_true", help="打印模型统计信息")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.stats:
        export_statistics(args.checkpoint, device=args.device)

    if args.downsample > 0:
        export_downsampled(args.checkpoint, args.output, target_count=args.downsample, device=args.device)
    else:
        export_to_ply(args.checkpoint, args.output, device=args.device)


if __name__ == "__main__":
    main()
