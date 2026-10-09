"""像素风化/帧间补间/角色档案/unitypackage 测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_pixel_tween_profiles.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.config import Settings
from pixelforge.processing.editing import pixelate, tween_x2
from pixelforge.processing.image_processor import ImageProcessor

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp" / "ptp"


def _grad_frame(size: int = 64) -> np.ndarray:
    """水平渐变帧（像素化前颜色连续，像素化后应呈块状）。"""
    a = np.zeros((size, size, 4), np.uint8)
    a[:, :, 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
    a[:, :, 1] = a[:, :, 0]
    a[:, :, 2] = a[:, :, 0]
    a[:, :, 3] = 255
    return a


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. pixelate：块状恒定 ----------
    g = _grad_frame(64)
    px = pixelate(g, 16)
    assert px.shape == g.shape, "尺寸应不变"
    block = px[0:4, 0:4]  # scale=4 -> 4x4 像素块颜色恒定
    assert (block == block[0, 0]).all(), "像素块内颜色应恒定"
    assert pixelate(g, 64) is g or (pixelate(g, 64) == g).all(), \
        "target >= 边长时应原样返回"
    px_alpha = pixelate(g.copy(), 16)
    assert (px_alpha[:, :, 3] == 255).all()
    print("[OK] 1. pixelate: 4x4 块恒定 / 尺寸不变 / 边界 target>=边长")

    # ---------- 2. 管线集成：pixel_size ----------
    sheet = np.full((48, 96, 3), (0, 255, 0), np.uint8)
    sheet[12:36, 12:30] = (90, 50, 180)
    sheet[12:36, 50:84] = (40, 200, 60)
    sp = TMP / "pipe_px.png"
    cv2.imwrite(str(sp), sheet)
    img = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 3
    proc = ImageProcessor(pixel_size=16)
    bgra = proc.remove_background(img)  # 绿幕 -> BGRA
    frames = proc.run_pipeline(bgra)
    assert len(frames) == 2
    h, w = frames[0].image.shape[:2]
    scale = max(1, round(max(h, w) / 16))
    blk = frames[0].image[2:2 + scale, 2:2 + scale, :3]
    assert (blk == blk[0, 0]).all(), "管线帧内应呈像素块"
    print("[OK] 2. run_pipeline: pixel_size 像素风化生效")

    # ---------- 3. tween_x2 ----------
    f1 = np.zeros((8, 8, 4), np.uint8); f1[:] = (0, 0, 0, 255)
    f2 = np.zeros((8, 8, 4), np.uint8); f2[:] = (100, 100, 100, 255)
    f3 = np.zeros((8, 8, 4), np.uint8); f3[:] = (200, 200, 200, 255)
    tw = tween_x2([f1, f2, f3])
    assert len(tw) == 5 and (tw[0] == f1).all() and (tw[2] == f2).all() \
        and (tw[4] == f3).all()
    expect = np.zeros((8, 8, 4), np.uint8); expect[:] = (50, 50, 50, 255)
    assert (tw[1] == expect).all(), "过渡帧应为两端 50% 混合"
    assert tween_x2([f1]) == [f1]
    print("[OK] 3. tween_x2: 3 帧 -> 5 帧 / 中点 50% 混合 / 单帧不变")

    # ---------- 4. 角色档案：存取往返 + MainWindow 应用 ----------
    old_cfg, old_char = cfgmod.CONFIG_FILE, cfgmod.CHARACTERS_FILE
    try:
        tdir = Path(tempfile.mkdtemp(prefix="pf_prof_"))
        cfgmod.CONFIG_FILE = tdir / "settings.json"
        cfgmod.CHARACTERS_FILE = tdir / "characters.json"
        assert cfgmod.load_profiles() == {}
        profs = {"骑士": {"checkpoint": "a.safetensors", "prompt": "pixel knight",
                     "width": 512, "cfg": 7.5}}
        cfgmod.save_profiles(profs)
        assert cfgmod.load_profiles() == profs

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from pixelforge.ui.main_window import MainWindow
        win = MainWindow()
        win._apply_profile(profs["骑士"])
        assert win.ed_prompt.toPlainText() == "pixel knight"
        assert win.spin_cfg.value() == 7.5
        assert win.cmb_ckpt.currentText() == "a.safetensors"
        data = win._profile_data()
        assert data["checkpoint"] == "a.safetensors" and data["cfg"] == 7.5 \
            and data["width"] == 512
        assert "骑士" in win.cmb_profile.itemText(1), "档案下拉应包含已存档案"
        win.close()
        print("[OK] 4. 角色档案: JSON 往返 + 主窗口应用/抽取/下拉同步")
    finally:
        cfgmod.CONFIG_FILE, cfgmod.CHARACTERS_FILE = old_cfg, old_char

    # ---------- 5. unitypackage：结构校验 + 播放器集成 ----------
    from pixelforge.unity_package import build_unity_package
    from PySide6.QtWidgets import QApplication as _QA
    app = QApplication.instance() or _QA(sys.argv)
    from pixelforge.ui.anim_player import AnimPlayer
    p = AnimPlayer()
    f1 = np.zeros((8, 8, 4), np.uint8); f1[2:6, 2:6] = (255, 0, 0, 255)
    f2 = np.zeros((8, 8, 4), np.uint8); f2[2:6, 2:6] = (0, 255, 0, 255)
    p.set_frames([f1, f2], name="hero")
    entries = p._unity_package_entries()
    assert any(e[1].endswith("hero.atlas.json") for e in entries)
    assert sum(1 for _, up in entries if up.endswith(".png")) == 2
    out = build_unity_package(entries, TMP / "hero.unitypackage")
    assert out.is_file() and out.stat().st_size > 0
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
        guids = {n.split("/")[0] for n in names}
        assert len(guids) == len(entries), "每资产一个 GUID 目录"
        for g in guids:
            assert f"{g}/pathname" in names
            assert f"{g}/asset" in names
            assert f"{g}/asset.meta" in names
        pn = [n for n in names if n.endswith("/pathname")]
        paths = {tar.extractfile(n).read().decode("utf-8") for n in pn}
        assert any(x == "Assets/Textures/Generated/hero.atlas.json" for x in paths)
        png_meta = next(tar.extractfile(n).read().decode("utf-8")
                        for n in names if n.endswith("/asset.meta")
                        and b"TextureImporter" in tar.extractfile(n).read())
        assert "textureType: 8" in png_meta and "guid:" in png_meta
    print(f"[OK] 5. unitypackage: {len(entries)} 资产 / pathname+meta 结构 / Sprite 配置")

    print("\n像素风化/补间/档案/unitypackage 全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
