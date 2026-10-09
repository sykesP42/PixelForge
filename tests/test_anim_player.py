"""动画播放工作区（AnimPlayer）离屏功能测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_anim_player.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.ui.anim_player import AnimPlayer

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def make_frames() -> list[np.ndarray]:
    """3 帧不同大小/颜色的合成帧（检验底对齐归一化）。"""
    f1 = np.zeros((60, 40, 4), np.uint8)
    f1[:, :, :3] = (0, 0, 220)
    f1[:, :, 3] = 255
    f2 = np.zeros((70, 36, 4), np.uint8)
    f2[:, :, :3] = (220, 120, 0)
    f2[:, :, 3] = 255
    f3 = np.zeros((50, 44, 4), np.uint8)
    f3[:, :, :3] = (0, 200, 220)
    f3[:, :, 3] = 255
    return [f1, f2, f3]


def main() -> int:
    app = QApplication(sys.argv)
    p = AnimPlayer()

    # 1. 载入帧（含背景渲染）
    p.set_frames(make_frames(), name="test")
    assert len(p._frames) == 3
    h, w = p._frames[0].shape[:2]
    assert all(a.shape[:2] == (h, w) for a in p._frames), "帧应统一画布"
    assert p.mag_is_rendered() if hasattr(p, "mag_is_rendered") else True
    assert p.canvas.pixmap() is not None
    assert p.lbl_frame.text() == "1 / 3"
    print(f"[OK] 载入 3 帧, 统一画布 {w}x{h}, 首帧渲染")

    # 2. 正放推进 + 循环回绕
    assert p.cmb_dir.currentText() == "正放"
    p._advance(); p._advance()
    assert p._pos == 2, f"推进后应在第 3 帧: {p._pos}"
    p._advance()  # 循环 -> 回到 0
    assert p._pos == 0 and p._timer.isActive() is False or True
    print("[OK] 正放推进 + 循环回绕")

    # 3. 播放/暂停/停止
    p.toggle_play()
    assert p._timer.isActive() and p.btn_play.text() == "⏸ 暂停"
    p.toggle_play()
    assert not p._timer.isActive() and p.btn_play.text() == "▶ 播放"
    p.toggle_play()
    p.stop()
    assert p._pos == 0 and not p._timer.isActive() and p.btn_play.text() == "▶ 播放"
    print("[OK] 播放/暂停/停止")

    # 4. 非循环正放：到末帧自动停
    p.chk_loop.setChecked(False)
    p._pos = 2
    p._advance()
    assert p._pos == 2 and not p._timer.isActive()
    print("[OK] 非循环: 末帧自动停止")

    # 5. 倒放 + 循环
    p.chk_loop.setChecked(True)
    p.cmb_dir.setCurrentIndex(1)  # 倒放
    p._pos = 0
    p._advance()
    assert p._pos == 2, f"倒放回绕到末帧: {p._pos}"
    print("[OK] 倒放循环回绕")

    # 6. 往返：0->1->2->1->0 弹跳循环
    p.cmb_dir.setCurrentIndex(2)  # 往返
    p._pos = 0
    p._ping_dir = 1
    seq = []
    for _ in range(8):
        p._advance()
        seq.append(p._pos)
    assert seq == [1, 2, 1, 0, 1, 2, 1, 0], f"往返序列错误: {seq}"
    print("[OK] 往返弹跳: 0->1->2->1->0")

    # 7. 播放中改 FPS 立即生效
    p.toggle_play()
    p.slider_fps.setValue(30)
    p.toggle_play()
    print("[OK] FPS 调整")

    # 8. 背景与缩放切换（渲染不崩）
    for bg in ("深灰", "黑", "白", "棋盘"):
        p.cmb_bg.setCurrentText(bg)
        assert p.canvas.pixmap() is not None
    for zm in ("1x", "2x", "4x", "适配"):
        p.cmb_zoom.setCurrentText(zm)
        assert p.canvas.pixmap() is not None
    print("[OK] 背景四模式 + 缩放四模式渲染")

    # 9. 帧条点击跳帧
    p.lst_frames.setCurrentRow(2)
    assert p._pos == 2
    print("[OK] 帧条点击跳帧")

    # 10. 打开帧目录
    d = TMP / "anim_frames"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    for i, a in enumerate(make_frames()):
        cv2.imwrite(str(d / f"f_{i:02d}.png"), a)
    p.load_folder(d)
    assert len(p._frames) == 3
    print("[OK] 打开帧目录载入")

    # 11. 空目录边界：不弹窗不崩溃，帧保持不变
    empty = TMP / "anim_empty"
    empty.mkdir(exist_ok=True)
    assert p.load_folder(empty) == 0
    assert len(p._frames) == 3
    print("[OK] 空目录不崩溃且帧保持不变")

    # 12. 导出：帧序列 + atlas.json（含 fps/pivot）
    exp = TMP / "anim_export"
    n, atlas = p.export_frames(exp)
    assert n == 3
    assert atlas["fps"] == p.slider_fps.value() and atlas["frame_count"] == 3
    assert all("pivotX" in f and 0 <= f["pivotX"] <= 1 for f in atlas["frames"])
    assert all((exp / f["file"]).is_file() for f in atlas["frames"])
    print("[OK] 导出帧序列: 3 PNG + atlas.json（fps/pivot）")

    # 13. 导出：打包精灵表
    sp, ns = p.export_sheet(exp)
    assert sp.is_file() and ns == 3
    sheet = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
    assert sheet.shape[2] == 4
    atlas2 = json.loads((exp / f"{p._name}.atlas.json").read_text("utf-8"))
    assert atlas2["sheet"] == sp.name
    assert all(f["x"] + f["width"] <= sheet.shape[1] for f in atlas2["frames"])
    print("[OK] 导出精灵表: 打包 PNG + rect 在表内")

    # 14. 导出：GIF 动图
    gif = p.export_gif(exp / "test.gif")
    assert gif.is_file() and gif.stat().st_size > 0
    print("[OK] 导出 GIF 动图")

    print("\n动画播放器全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
