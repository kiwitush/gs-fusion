"""
文本到 3D 生成：通过 threestudio + SDS Loss 从文本 Prompt 生成 3D 资产。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREESTUDIO_DIR = PROJECT_ROOT / "threestudio"
TEMPLATE_CONFIG = THREESTUDIO_DIR / "configs" / "dreamfusion-sd.yaml"

# 公开模型，不需要 HuggingFace 授权
DEFAULT_GUIDANCE_MODEL = "runwayml/stable-diffusion-v1-5"


class TextTo3DGenerator:

    def __init__(
        self,
        output_dir: str = "./outputs/object_b",
        guidance_model: str = DEFAULT_GUIDANCE_MODEL,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.guidance_model = guidance_model

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        iters: int = 10_000,
        resolution: int = 64,
    ) -> str:
        config_path = self._write_config(prompt, negative_prompt, iters, resolution)
        print(f"[文本到3D] 使用 prompt: '{prompt}'")
        print(f"[文本到3D] 配置文件: {config_path}")

        # 确保 custom/ 目录存在
        (THREESTUDIO_DIR / "custom").mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(THREESTUDIO_DIR / "launch.py"),
            "--config", config_path,
            "--train",
        ]
        env = os.environ.copy()
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        subprocess.run(cmd, check=True, cwd=str(THREESTUDIO_DIR), env=env)

        mesh_path = self._find_mesh()
        if mesh_path:
            print(f"[文本到3D] Mesh 已导出 → {mesh_path}")
        return mesh_path

    def _write_config(
        self, prompt: str, negative_prompt: str, iters: int, resolution: int
    ) -> str:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(TEMPLATE_CONFIG)

        cfg.seed = 42
        cfg.name = self.output_dir.name
        cfg.exp_root_dir = str(self.output_dir.parent)
        cfg.tag = "default"

        cfg.data.width = resolution
        cfg.data.height = resolution

        cfg.system.prompt_processor.prompt = prompt
        cfg.system.prompt_processor.negative_prompt = (
            negative_prompt or "ugly, blurry, low quality, distorted"
        )
        cfg.system.prompt_processor.pretrained_model_name_or_path = self.guidance_model
        cfg.system.prompt_processor.spawn = False
        cfg.system.guidance.pretrained_model_name_or_path = self.guidance_model

        cfg.system.loggers.wandb.enable = False

        cfg.trainer.max_steps = iters

        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        return str(config_path)

    def _find_mesh(self) -> str:
        """训练完成后在 trial_dir 中寻找导出的 mesh 文件。"""
        trial_base = self.output_dir.parent / self.output_dir.name
        if not trial_base.exists():
            print(f"[文本到3D] 警告: 未找到 trial 目录于 {trial_base}")
            return ""

        subdirs = sorted(
            [d for d in trial_base.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not subdirs:
            return ""

        trial_dir = subdirs[0]

        # dreamfusion-system 导出到 save/ 目录
        save_dir = trial_dir / "save"
        for obj in save_dir.rglob("*.obj") if save_dir.exists() else []:
            mesh_path = self.output_dir / "model.obj"
            shutil.copy(obj, mesh_path)
            return str(mesh_path)

        # 备用：搜索整个 trial_dir
        for obj in trial_dir.rglob("*.obj"):
            mesh_path = self.output_dir / "model.obj"
            shutil.copy(obj, mesh_path)
            return str(mesh_path)

        print("[文本到3D] 警告: 未找到导出的 mesh 文件")
        return ""


def main():
    parser = argparse.ArgumentParser(description="通过 threestudio 从文本 Prompt 生成 3D 资产")
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output_dir", "-o", default="./outputs/object_b")
    parser.add_argument("--guidance_model", default=DEFAULT_GUIDANCE_MODEL)
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    generator = TextTo3DGenerator(
        output_dir=args.output_dir,
        guidance_model=args.guidance_model,
    )
    mesh_path = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        iters=args.iters,
        resolution=args.resolution,
    )

    print(f"\n生成完成。Mesh: {mesh_path}")
    print(f"下一步: 将 Mesh 转换为 2DGS →")
    print(f"  python src/utils/mesh_to_gs.py -i {mesh_path} -o {args.output_dir}/object_b.ply")


if __name__ == "__main__":
    main()
