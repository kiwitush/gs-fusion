"""
下载 Mip-NeRF 360 garden 场景并提取到 data/garden/。
仅下载完整 360_v2.zip（约 2.7 GB），但只提取 garden 场景以节省磁盘空间。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
ZIP_NAME = "360_v2.zip"


def _progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        bar_len = 40
        filled = int(bar_len * pct / 100.0)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  下载中 [{bar}] {pct:.0f}%  ({downloaded / 1024**2:.0f}/{total_size / 1024**2:.0f} MB)")
        sys.stdout.flush()


def download_and_extract_garden(target_dir: str, keep_zip: bool = False):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target.parent / ZIP_NAME

    # 下载
    if zip_path.exists():
        print(f"  ZIP 已存在: {zip_path}，跳过下载。")
    else:
        print(f"  下载 {URL} ...")
        urlretrieve(URL, zip_path, reporthook=_progress)
        print()  # 换行

    # 只提取 garden
    print(f"  提取 garden 场景 → {target} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        garden_members = [m for m in zf.namelist() if m.startswith("garden/") or m.startswith("360_v2/garden/")]
        if not garden_members:
            # 尝试带前缀
            prefix = "360_v2/"
            garden_members = [m for m in zf.namelist() if m.startswith(f"{prefix}garden/")]
        else:
            prefix = ""

        for member in garden_members:
            rel_path = member[len(prefix):]  # 去掉 360_v2/ 前缀
            dest = target.parent / rel_path
            if member.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    # 确保 images/ 存在（部分版本用 images_4/ images_8/）
    images_dir = target / "images"
    sparse_dir = target / "sparse"
    if (not images_dir.exists() or not list(images_dir.glob("*"))):
        for alt in ("images_4", "images_2", "images_8"):
            alt_dir = target / alt
            if alt_dir.exists() and list(alt_dir.glob("*")):
                print(f"  图像目录 '{alt}' → 链接为 'images/'")
                alt_dir.rename(images_dir)
                break

    if not images_dir.exists() or not list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")):
        print("  警告: 未能定位 garden 场景的图像文件，请手动检查。")

    # 清理
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  已删除临时 ZIP 文件。")

    # 验证
    img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    has_sparse = sparse_dir.exists() and (sparse_dir / "0").exists()
    print(f"\n  完成。图像: {img_count} 张 | COLMAP sparse 模型: {'已包含' if has_sparse else '缺失'}")
    print(f"  场景目录: {target.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="下载 Mip-NeRF 360 garden 场景")
    parser.add_argument("--output", "-o", default="data", help="输出目录 (默认: data)")
    parser.add_argument("--keep-zip", action="store_true", help="下载后保留 ZIP 文件")
    args = parser.parse_args()

    target_dir = os.path.join(args.output, "garden")
    download_and_extract_garden(target_dir, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
