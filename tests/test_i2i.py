"""img2img 跨帧一致性系列生成测试（mock，不依赖真实 ComfyUI）。

用法：python tests/test_i2i.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.comfyui_client import ComfyUIClient
from pixelforge.config import PROJECT_ROOT, Settings

WF = PROJECT_ROOT / "pixelforge" / "workflows" / "img2img_frame.json"


def main() -> int:
    # ---------- 1. 工作流模板：img2img 改线契约 ----------
    tpl = json.loads(WF.read_text("utf-8"))
    cls = {n["class_type"] for n in tpl.values()}
    assert {"LoadImage", "VAEEncode", "SetLatentNoiseMask", "KSampler"} <= cls
    ks = next(n for n in tpl.values() if n["class_type"] == "KSampler")
    latent_ref = ks["inputs"]["latent_image"]
    encode_id = next(k for k, n in tpl.items()
                     if n["class_type"] == "VAEEncode")
    mask_id = next(k for k, n in tpl.items()
                   if n["class_type"] == "SetLatentNoiseMask")
    assert latent_ref == [mask_id, 0], "latent 应经 SetLatentNoiseMask"
    mask_samples = next(n for k, n in tpl.items() if k == mask_id) \
        ["inputs"]["samples"]
    assert mask_samples == [encode_id, 0], "mask 采样应来自 VAEEncode"
    assert ks["inputs"]["denoise"] == "__DENOISE__"
    li = next(n for n in tpl.values() if n["class_type"] == "LoadImage")
    assert li["inputs"]["image"] == "__IMAGE__"
    print("[OK] 1. img2img 模板: LoadImage->VAEEncode->Mask->KSampler(denoise)")

    # ---------- 2. generate_frame_series：mock img2img ----------
    client = ComfyUIClient.__new__(ComfyUIClient)  # 跳过网络初始化
    calls: list[dict] = []
    fake_dir = Path(tempfile.mkdtemp(prefix="pf_i2i_"))

    def fake_img2img(anchor, **kw):
        calls.append({"anchor_shape": anchor.shape, **kw})
        p = fake_dir / f"frame_{len(calls):03d}.png"
        p.write_bytes(b"fake")
        return [p]

    client.img2img = fake_img2img  # monkeypatch
    anchor = np.zeros((32, 32, 4), np.uint8)
    anchor[..., 3] = 255
    out = client.generate_frame_series(
        anchor, 5, checkpoint="a.safetensors", prompt="pixel knight",
        negative_prompt="blurry", denoise=0.4, steps=15, cfg=6.0,
        out_dir=fake_dir, progress=lambda i, t: None)
    assert len(out) == 4, f"应生成 4 个变体帧: {len(out)}"
    assert len(calls) == 4
    assert all(c["anchor_shape"] == (32, 32, 4) for c in calls), \
        "每帧都应以同一 anchor 为底图（跨帧一致性核心）"
    assert all(c["denoise"] == 0.4 and c["steps"] == 15 for c in calls)
    assert all(c["seed"] is None for c in calls), "每帧应独立随机 seed"
    assert [str(p) for p in out] == [c["out_dir"] and str(fake_dir / f"frame_{i:03d}.png")
                                     for i, c in zip(range(1, 5), calls)]
    # n_frames=1 -> 无变体
    assert client.generate_frame_series(anchor, 1, checkpoint="a",
                                        prompt="", negative_prompt="",
                                        out_dir=fake_dir) == []
    print("[OK] 2. 系列生成: anchor 恒定 / 独立 seed / 参数透传 / 边界 n=1")

    # ---------- 3. Settings i2i 字段往返 + 旧 JSON 兼容 ----------
    old_cfg = cfgmod.CONFIG_FILE
    try:
        tmp = Path(tempfile.mkdtemp(prefix="pf_cfg_"))
        cfgmod.CONFIG_FILE = tmp / "settings.json"
        s = Settings(i2i_frames=6, i2i_denoise=0.5)
        s.save()
        s2 = Settings.load()
        assert s2.i2i_frames == 6 and abs(s2.i2i_denoise - 0.5) < 1e-6
        (tmp / "settings.json").write_text(
            json.dumps({"prompt": "old"}), "utf-8")  # 旧版无 i2i 字段
        s3 = Settings.load()
        assert s3.i2i_frames == 0 and s3.prompt == "old"
    finally:
        cfgmod.CONFIG_FILE = old_cfg
    print("[OK] 3. Settings: i2i 字段持久化往返 + 旧 JSON 兼容")

    print("\nimg2img 跨帧一致性测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
