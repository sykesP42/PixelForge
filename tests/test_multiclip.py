"""Aseprite 多动画（frameTags）+ 透明 GIF 测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_multiclip.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.aseprite_io import export_aseprite, frames_by_tag
from pixelforge.ui.anim_player import AnimPlayer

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp" / "multiclip"


def _frame(w: int, h: int, color) -> np.ndarray:
    a = np.zeros((h, w, 4), np.uint8)
    a[2:h - 2, 2:w - 2] = (*color, 255)
    return a


def main() -> int:
    app = QApplication(sys.argv)
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. frames_by_tag：hash 格式分组 ----------
    frames = [_frame(8, 8, (255, 0, 0)), _frame(8, 8, (0, 255, 0)),
              _frame(8, 8, (0, 0, 255)), _frame(8, 8, (255, 255, 0)),
              _frame(8, 8, (0, 255, 255)), _frame(8, 8, (255, 0, 255))]
    png_p, json_p = export_aseprite(frames, TMP / "hero", fps=8, name="hero")
    data = json.loads(json_p.read_text("utf-8"))
    data["meta"]["frameTags"] = [
        {"name": "idle", "from": 0, "to": 1, "direction": "forward"},
        {"name": "walk", "from": 2, "to": 5, "direction": "forward"},
    ]
    json_p.write_text(json.dumps(data), "utf-8")
    clips = frames_by_tag(json_p)
    assert set(clips) == {"idle", "walk"}
    assert len(clips["idle"]) == 2 and len(clips["walk"]) == 4
    assert (clips["idle"][0][4, 4] == (255, 0, 0, 255)).all()
    assert (clips["walk"][-1][4, 4] == (255, 0, 255, 255)).all()
    # array 格式 + 无 frameTags -> default 全量
    data["frames"] = list(data["frames"].values())
    del data["meta"]["frameTags"]
    arr_json = TMP / "hero_arr.json"
    arr_json.write_text(json.dumps(data), "utf-8")
    clips2 = frames_by_tag(arr_json)
    assert set(clips2) == {"default"} and len(clips2["default"]) == 6
    print("[OK] 1. frames_by_tag: hash 分组 / array+无 tags 回落 default")

    # ---------- 2. AnimPlayer 多动画组：切换播放 ----------
    p = AnimPlayer()
    p._clips = {"idle": [frames[0], frames[1]],
                "walk": frames[2:6]}
    p.set_frames(p._clips["idle"], name="idle", keep_clips=True)
    assert p.cmb_clips.isEnabled() and p.cmb_clips.count() == 2
    p._on_clip_selected("walk")
    assert len(p._frames) == 4 and p._name == "walk"
    assert len(p._clips) == 2, "切换组不应清掉其他动画组"
    # 普通载入（不 keep_clips）应重置为单组
    p.set_frames(frames[:2], name="solo")
    assert list(p._clips) == ["solo"] and not p.cmb_clips.isEnabled()
    print("[OK] 2. 动画组切换: cmb 同步 / keep_clips / 单动画重置")

    # ---------- 3. Godot 多动画 .tres ----------
    p._clips = {"idle": [frames[0], frames[1]], "walk": frames[2:6]}
    p.set_frames(p._clips["idle"], name="idle", keep_clips=True)
    exp_g = TMP / "godot_out"
    tp, total = p.export_godot(exp_g)
    assert total == 6 and tp.is_file()
    tres = tp.read_text("utf-8")
    assert tres.count("[ext_resource") == 6
    assert '"name": &"idle"' in tres and '"name": &"walk"' in tres
    assert tres.count("animations = [") == 1
    for i in range(6):
        assert list(exp_g.glob(f"*_{i:03d}.png")), f"帧 {i} 未写出"
    # 单动画 .tres 结构不变（回归）
    p.set_frames(frames[:3], name="solo")
    exp_g2 = TMP / "godot_solo"
    tp2, n2 = p.export_godot(exp_g2)
    tres2 = tp2.read_text("utf-8")
    assert tres2.count("[ext_resource") == 3 and '"name": &"solo"' in tres2
    print(f"[OK] 3. Godot 导出: 多动画 {total} 帧 / 单动画结构回归")

    # ---------- 4. Unity 多动画：每动画组一套 atlas ----------
    p._clips = {"idle": [frames[0], frames[1]], "walk": frames[2:6]}
    p.set_frames(p._clips["idle"], name="idle", keep_clips=True)
    exp_u = TMP / "unity_out"
    atlases = p.export_unity_clips(exp_u)
    assert sorted(a.name for a in atlases) == ["idle.atlas.json", "walk.atlas.json"]
    for name, expect_n in (("idle", 2), ("walk", 4)):
        meta = json.loads((exp_u / f"{name}.atlas.json").read_text("utf-8"))
        assert meta["name"] == name and meta["frame_count"] == expect_n
        for fr in meta["frames"]:
            assert set(fr) >= {"file", "x", "y", "width", "height",
                               "pivotX", "pivotY", "hull"}
            assert (exp_u / fr["file"]).is_file()
    print("[OK] 4. Unity 多动画: 每组独立 atlas（含 hull 契约）")

    # ---------- 5. 透明 GIF ----------
    p.set_frames([_frame(16, 16, (255, 0, 0)), _frame(16, 16, (0, 0, 255))],
                 name="gif_test")
    p.cmb_bg.setCurrentText("透明")
    from PIL import Image
    gp = TMP / "alpha.gif"
    p.export_gif(gp)
    im = Image.open(gp)
    assert getattr(im, "n_frames", 1) == 2, "应为 2 帧"
    assert im.info.get("transparency") is not None, "应带透明索引"
    rgba = im.convert("RGBA")
    assert rgba.getpixel((0, 0))[3] == 0, "四角应为全透明"
    # 非透明路径回归
    p.cmb_bg.setCurrentText("深灰")
    gp2 = TMP / "bg.gif"
    p.export_gif(gp2)
    im2 = Image.open(gp2)
    assert im2.info.get("transparency") is None
    print("[OK] 5. 透明 GIF: 透明索引/2 帧/角落 alpha=0；非透明路径回归")

    print("\nAseprite 多动画 + 透明 GIF 全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
