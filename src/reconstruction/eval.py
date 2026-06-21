"""
2DGS 重建质量评估套件。

在留存的测试视角上计算 PSNR、SSIM 和 LPIPS 指标，
并通过统一的日志接口记录结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reconstruction.model import GaussianModel2D
from src.reconstruction.render import Renderer2DGS
from src.utils.image_quality import psnr, ssim

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class Evaluator2DGS:
    """2DGS 模型定量评估器。"""

    def __init__(
        self,
        checkpoint_path: str,
        test_views: List[Dict],
        device: str = "cuda",
        use_lpips: bool = True,
    ):
        self.renderer = Renderer2DGS(checkpoint_path, device=device)
        self.test_views = test_views
        self.device = device
        self.use_lpips = use_lpips and HAS_LPIPS
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device) if self.use_lpips else None

    def evaluate(self) -> Dict[str, float]:
        """在所有测试视角上运行评估并返回汇总指标。"""
        from PIL import Image
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        for view in tqdm(self.test_views, desc="评估中"):
            # 从磁盘加载 GT 图像
            img_path = view.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                print(f"[评估] 警告: 找不到图像 {img_path}，跳过")
                continue
            gt = torch.tensor(
                np.array(Image.open(img_path).convert("RGB")) / 255.0,
                dtype=torch.float32,
            ).permute(2, 0, 1).to(self.device)

            with torch.no_grad():
                result = self.renderer.render_view(view)
                pred = result["render"].clamp(0, 1)

            metrics["psnr"].append(psnr(pred, gt))
            metrics["ssim"].append(ssim(pred, gt))

            if self.lpips_fn is not None:
                metrics["lpips"].append(self._compute_lpips(pred, gt))

        if not metrics["psnr"]:
            return {"psnr_mean": 0.0, "ssim_mean": 0.0, "lpips_mean": 0.0, "error": "no valid views"}

        return {
            "psnr_mean": float(np.mean(metrics["psnr"])),
            "psnr_std": float(np.std(metrics["psnr"])),
            "ssim_mean": float(np.mean(metrics["ssim"])),
            "ssim_std": float(np.std(metrics["ssim"])),
            "lpips_mean": float(np.mean(metrics["lpips"])) if metrics["lpips"] else 0.0,
            "lpips_std": float(np.std(metrics["lpips"])) if metrics["lpips"] else 0.0,
            "num_views": len(self.test_views),
        }

    def _compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        return self.lpips_fn(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()


def main():
    parser = argparse.ArgumentParser(description="评估训练好的 2DGS 模型")
    parser.add_argument("--checkpoint", "-c", required=True, help=".pth 或 .ply 检查点路径")
    parser.add_argument("--test_views", required=True, help="测试视角 JSON 文件路径")
    parser.add_argument("--output_metrics", "-o", default="metrics.json", help="指标 JSON 输出路径")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.test_views, "r") as f:
        test_views = json.load(f)

    for v in test_views:
        for key in ("viewmatrix", "projmatrix", "campos"):
            if key in v and not isinstance(v[key], torch.Tensor):
                v[key] = torch.tensor(v[key], device=args.device)
        v.setdefault("fovx", 0.691)
        v.setdefault("fovy", 0.539)
        if "campos" not in v:
            v["campos"] = torch.tensor([0.0, 0.0, 3.0], device=args.device)

    evaluator = Evaluator2DGS(
        checkpoint_path=args.checkpoint,
        test_views=test_views,
        device=args.device,
        use_lpips=not args.no_lpips,
    )
    results = evaluator.evaluate()

    print("\n" + "=" * 50)
    print(" 评估结果")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(args.output_metrics, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n指标已保存 → {args.output_metrics}")


if __name__ == "__main__":
    main()
