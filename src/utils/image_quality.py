"""图像质量评估指标（PSNR、SSIM），消除 train.py / eval.py 之间的重复实现。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算两张 (C,H,W) 图像之间的 PSNR（峰值信噪比）。"""
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """计算两张 (C,H,W) 图像之间的 SSIM（结构相似性）。"""
    C = pred.shape[0]

    def _gaussian(ws: int, sigma: float) -> torch.Tensor:
        gauss = torch.tensor(
            [math.exp(-(x - ws // 2) ** 2 / (2 * sigma ** 2)) for x in range(ws)],
            dtype=torch.float32,
        )
        return gauss / gauss.sum()

    _window = _gaussian(window_size, 1.5).unsqueeze(1)
    window = _window.mm(_window.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).to(pred.device)

    mu1 = F.conv2d(pred.unsqueeze(0), window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(gt.unsqueeze(0), window, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred.unsqueeze(0) * pred.unsqueeze(0), window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt.unsqueeze(0) * gt.unsqueeze(0), window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred.unsqueeze(0) * gt.unsqueeze(0), window, padding=window_size // 2, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
