"""阶段1 MVP 冒烟测试（无需 ComfyUI 运行）。

用法：python tests/smoke_test.py
验证：模块可导入、配置读写、工作流占位符替换、客户端离线探测、Conda 环境识别。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.comfyui_client import ComfyUIClient, build_workflow, random_seed
from pixelforge.config import Settings
from pixelforge.conda_manager import find_conda, list_envs

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # 1. 配置读写
    s = Settings.load()
    assert s.comfyui_port == 8188, "默认端口错误"
    print(f"[OK] 配置加载: port={s.comfyui_port}, output={s.output_path()}")

    # 2. 工作流占位符替换
    template = json.loads(
        (ROOT / "pixelforge" / "workflows" / "txt2img_basic.json").read_text("utf-8")
    )
    wf = build_workflow(
        template,
        checkpoint="pixel_v1.safetensors",
        prompt="pixel art knight",
        negative_prompt="blurry",
        seed=123456,
        width=512, height=512, steps=20, cfg=7.0,
    )
    ks = wf["5"]["inputs"]
    assert ks["seed"] == 123456 and ks["cfg"] == 7.0
    assert wf["1"]["inputs"]["ckpt_name"] == "pixel_v1.safetensors"
    assert wf["2"]["inputs"]["clip"] == ["1", 1], "clip 应引用 Checkpoint 的第 1 输出"
    assert wf["6"]["inputs"]["vae"] == ["1", 2], "vae 应引用 Checkpoint 的第 2 输出"
    print("[OK] 工作流占位符替换与节点连线正确")

    # 3. 离线探测（不抛异常即通过）
    client = ComfyUIClient("127.0.0.1", 59999)  # 必然离线的端口
    assert client.is_online() is False
    print("[OK] ComfyUI 离线探测正常返回 False")

    # 4. Conda 识别
    conda = find_conda()
    print(f"[OK] Conda 可执行文件: {conda}")
    if conda:
        envs = list_envs(conda)
        print(f"[OK] 识别到 {len(envs)} 个环境: {[n for n, _ in envs]}")

    # 5. GUI 模块可导入
    from pixelforge.ui.image_lab import ImageLab  # noqa: F401
    from pixelforge.ui.review_panel import ReviewWidget  # noqa: F401
    print("[OK] UI 模块导入正常")
    seeds = {random_seed() for _ in range(20)}
    assert len(seeds) > 10
    print("[OK] 随机种子生成正常")

    print("\n全部冒烟测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
