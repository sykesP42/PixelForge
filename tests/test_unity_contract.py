"""阶段4 Unity 导入契约测试：验证 Python 导出的 atlas.json 与 C# 端约定严格一致。

用法：QT_QPA_PLATFORM=offscreen python tests/test_unity_contract.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from pixelforge.processing.image_processor import ImageProcessor
from pixelforge.ui.review_panel import ReviewWidget

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"

# C# JsonUtility 模型字段（unity/PixelForgeImporter.cs 中 AtlasMeta/FrameMeta）
ATLAS_KEYS = {"name", "fps", "frame_count", "frames"}
FRAME_KEYS = {"file", "x", "y", "width", "height", "pivotX", "pivotY", "hull"}


def unity_pivot(tex_w: int, tex_h: int, m: dict) -> tuple[float, float]:
    """复刻 C# ComputePivot 公式（pivot 来自 atlas 每帧记录）。"""
    px = (m["x"] + m.get("pivotX", 0.5) * m["width"]) / tex_w
    py = (tex_h - m["y"] - m.get("pivotY", 1.0) * m["height"]) / tex_h
    return px, py


def main() -> int:
    app = QApplication(sys.argv)

    sheet = TMP / "synth_sheet.png"
    if not sheet.is_file():
        sys.path.insert(0, str(ROOT / "tests"))
        from test_processing import make_sheet
        make_sheet(sheet)
    out = TMP / "contract_out"
    files = [f for f in ImageProcessor().process(sheet, out) if f.suffix == ".png"]

    w = ReviewWidget()
    w.set_frames(files, {"name": "knight_idle", "fps": 10})
    w.chk_reviewed.setChecked(True)
    exp = TMP / "unity_export"
    w.export_to(exp)

    atlas = json.loads((exp / "knight_idle.atlas.json").read_text("utf-8"))

    # 1. 字段契约：C# JsonUtility 严格按字段名映射
    assert set(atlas.keys()) == ATLAS_KEYS, f"atlas 字段: {set(atlas.keys())}"
    for fr in atlas["frames"]:
        assert set(fr.keys()) == FRAME_KEYS, f"frame 字段: {set(fr.keys())}"
        assert isinstance(fr["x"], int) and isinstance(fr["y"], int)
        assert isinstance(fr["pivotX"], float) and isinstance(fr["pivotY"], float)
    assert atlas["frame_count"] == len(atlas["frames"]) == 4
    print(f"[OK] 字段契约: atlas/frame 字段与 C# 模型一致, {atlas['frame_count']} 帧")

    # 2. 帧文件唯一且有序
    names = [fr["file"] for fr in atlas["frames"]]
    assert len(set(names)) == len(names)
    assert names == sorted(names)
    print(f"[OK] 帧文件命名唯一有序: {names}")

    # 3. bbox 与贴图实际内容一致（Unity 切片依赖此精度）
    for fr in atlas["frames"]:
        tex = cv2.imread(str(exp / fr["file"]), cv2.IMREAD_UNCHANGED)
        th, tw = tex.shape[:2]
        ys, xs = np.where(tex[:, :, 3] > 0)
        assert (xs.min(), ys.min()) == (fr["x"], fr["y"])
        assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == \
               (fr["width"], fr["height"])
        assert 0 <= fr["x"] and fr["x"] + fr["width"] <= tw
        assert 0 <= fr["y"] and fr["y"] + fr["height"] <= th
    print("[OK] bbox 与贴图内容像素级一致, 且在画布边界内")

    # 3.5 hull 契约：扁平坐标、偶数长度、点全部落在帧画布内
    for fr in atlas["frames"]:
        hull = fr["hull"]
        assert isinstance(hull, list) and len(hull) >= 6 and len(hull) % 2 == 0, \
            f"hull 应为扁平偶数长度列表: {len(hull)}"
        assert all(isinstance(v, (int, float)) for v in hull)
        tex = cv2.imread(str(exp / fr["file"]), cv2.IMREAD_UNCHANGED)
        th, tw = tex.shape[:2]
        hx, hy = hull[0::2], hull[1::2]
        assert 0 <= min(hx) and max(hx) <= tw and 0 <= min(hy) and max(hy) <= th, \
            "hull 顶点应落在帧画布内（C# 侧按 pivot 换算本地坐标）"
    print("[OK] hull 契约: 扁平坐标 / 偶数长度 / 顶点在画布内")

    # 4. pivot y 翻转语义验证（基线对齐 => 内容贴画布底 => pivotY=1.0 应得 py=0）
    for fr in atlas["frames"]:
        tex = cv2.imread(str(exp / fr["file"]), cv2.IMREAD_UNCHANGED)
        th, tw = tex.shape[:2]
        px, py = unity_pivot(tw, th, fr)
        assert 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0, f"pivot 越界: ({px},{py})"
        assert fr["y"] + fr["height"] == th, "基线对齐后内容应贴画布底"
        assert abs(py) < 1e-6, f"脚底 pivot 应落在 Unity 底边, 实际 py={py}"
        assert abs(px - 0.5) < 0.02, \
            f"锚点应落在画布中心（feet_center 对齐）, 实际 px={px}"
    print("[OK] pivot 换算: 脚底 -> Unity 底边(py=0), 水平居中(px=0.5)")

    # 5. C# 公式语义抽查（非贴底内容）
    m = {"x": 10, "y": 20, "width": 30, "height": 40, "pivotX": 0.5, "pivotY": 0.0}
    _, py_top = unity_pivot(64, 64, m)   # pivotY=0 => 内容顶
    assert abs(py_top - (64 - 20) / 64) < 1e-6
    m["pivotY"] = 1.0
    _, py_bot = unity_pivot(64, 64, m)   # pivotY=1 => 内容底
    assert abs(py_bot - (64 - 60) / 64) < 1e-6
    assert py_top > py_bot
    print("[OK] C# 公式语义: pivotY=0 为内容顶, pivotY=1 为内容底")

    print("\n阶段4 契约测试全部通过 ✔（C# 运行时行为需在 Unity 编辑器内人工确认）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
