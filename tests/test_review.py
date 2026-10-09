"""阶段3 审核面板测试（QT 离屏，无需显示器）。

用法：QT_QPA_PLATFORM=offscreen python tests/test_review.py
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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pixelforge.processing.image_processor import ImageProcessor
from pixelforge.ui.review_panel import ReviewWidget

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def main() -> int:
    app = QApplication(sys.argv)

    # 用阶段2 的合成精灵表跑出帧
    sheet = TMP / "synth_sheet.png"
    if not sheet.is_file():
        sys.path.insert(0, str(ROOT / "tests"))
        from test_processing import make_sheet
        make_sheet(sheet)
    out = TMP / "review_out"
    files = ImageProcessor().process(sheet, out)
    frames = [f for f in files if f.suffix == ".png"]
    atlas = json.loads((out / "synth_sheet.atlas.json").read_text("utf-8"))
    assert len(frames) == 4

    w = ReviewWidget()
    w.set_frames(frames, atlas)
    assert w.lst.count() == 4, f"帧网格应有 4 项, 实际 {w.lst.count()}"
    assert w.checked_count() == 4
    print("[OK] 帧网格装载: 4 帧, 全部勾选")

    # 洋葱皮渲染 + 播放推进
    w.chk_onion.setChecked(True)
    w._render_current()
    w._advance()
    assert w._pos == 1
    w.toggle_play()
    assert w._timer.isActive()
    w.toggle_play()
    print("[OK] 洋葱皮渲染与播放器启停正常")

    # 未审核 -> 导出被门禁拦截
    try:
        w.export_to(TMP / "export_no")
        raise AssertionError("未审核导出应被拒绝")
    except PermissionError:
        print("[OK] 审核门禁: 未审核导出被拒绝")

    # 单帧操作：删除 + 排序 + 手动裁剪
    w.delete_frame(0)
    assert w.lst.count() == 3
    first_name = w.lst.item(0).text()
    assert w.move_frame(0, 1)
    assert w.lst.item(1).text() == first_name
    assert not w.move_frame(5, -1)          # 越界保护
    assert not w.move_frame(0, -1)
    def content_h(a):
        ys, _ = np.where(a[:, :, 3] > 0)
        return int(ys.max() - ys.min() + 1)

    arr_before = w.lst.item(0).data(Qt.UserRole)["arr"]
    h_before = content_h(arr_before)
    assert w.crop_frame(0, 2, 2, 2, 2)
    arr_after = w.lst.item(0).data(Qt.UserRole)["arr"]
    # 画布会被重归一化为统一尺寸，但内容 bbox 高度应因底部裁剪减少 2px
    assert content_h(arr_after) == h_before - 2, \
        f"内容高度 {h_before} -> {content_h(arr_after)}"
    assert arr_after.shape == arr_before.shape  # 画布尺寸保持统一
    assert not w.crop_frame(0, 999, 0, 0, 0)  # 越界保护
    print("[OK] 删除/排序/手动裁剪 + 越界保护")

    # 审核通过 -> 导出
    w.chk_reviewed.setChecked(True)
    w._sync_buttons()
    assert w.btn_export.isEnabled()
    exp = TMP / "export_ok"
    exported = w.export_to(exp)
    assert (exp / "synth_sheet.atlas.json").is_file()
    meta = json.loads((exp / "synth_sheet.atlas.json").read_text("utf-8"))
    assert meta["frame_count"] == 3
    pngs = [p for p in exported if p.suffix == ".png"]
    assert len(pngs) == 3
    for p in pngs:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert im is not None and im.shape[2] == 4
    print(f"[OK] 导出: {len(pngs)} 帧 PNG + atlas.json（fps={meta['fps']}）")

    print("\n阶段3 全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
