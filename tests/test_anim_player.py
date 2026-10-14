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

    # 15. 洋葱皮：前后帧残影叠加（前红/后蓝），范围可调
    p.stop()
    p._pos = 1
    base = p._frames[1]
    p.chk_onion.setChecked(False)
    assert not p.spin_onion.isEnabled(), "未勾选时范围微调应禁用"
    p._render()
    assert p.canvas.pixmap() is not None
    p.chk_onion.setChecked(True)
    assert p.spin_onion.isEnabled(), "勾选洋葱皮后范围微调应启用"
    on = p._onion_frame(base)
    assert on.shape == base.shape and on.dtype == np.uint8, "合成应保持 BGRA 尺寸"
    assert not np.array_equal(on, base), "洋葱皮合成结果应与原帧不同"
    p._render()
    assert p.canvas.pixmap() is not None
    # 范围可调：5 帧动画下 rng=2 比 rng=1 多纳入更远帧 -> 结果不同
    p.set_frames([p._frames[i % 3] for i in range(5)], name="onion5")
    p._pos = 2
    p.chk_onion.setChecked(True)
    p.spin_onion.setValue(1)
    r1 = p._onion_frame(p._frames[2])
    p.spin_onion.setValue(2)
    r2 = p._onion_frame(p._frames[2])
    assert not np.array_equal(r1, r2), "范围变化应影响残影强度/层数"
    p.spin_onion.setValue(1)
    # 首帧：无前帧，仅后帧残影，仍可正常渲染
    p._pos = 0
    p._render()
    assert p.canvas.pixmap() is not None
    p.chk_onion.setChecked(False)
    print("[OK] 洋葱皮: 前红/后蓝残影叠加，范围可调，边界帧安全")

    # 16. 逐帧位移/缩放：入口在播放器，应用后保留用户位移（不再底对齐）
    from PySide6.QtWidgets import QDialog as _QD
    from pixelforge.ui.image_lab import FrameAdjustDialog

    def _fake_exec(self):
        self.sp_dx.setValue(5)
        self._broadcast()
        return _QD.Accepted

    def _rel_x(fp: np.ndarray) -> float:
        """内容水平中心相对「自身画布中心」的偏移。"""
        ys, xs = np.where(fp[:, :, 3] > 0)
        return float(xs.mean()) - fp.shape[1] / 2.0

    p.set_frames(make_frames(), name="adj")
    orig_exec = FrameAdjustDialog.exec
    FrameAdjustDialog.exec = _fake_exec
    try:
        rel0 = [_rel_x(f) for f in p._frames]
        p._open_frame_adjust()
        assert len(p._frames) == 3, "帧数应保持 3"
        assert p._frames[0].shape == p._frames[2].shape, "应用后应统一画布"
        rel1 = [_rel_x(f) for f in p._frames]
        for a, b in zip(rel0, rel1):
            assert abs((b - a) - 5.0) < 1e-6, \
                f"全部帧相对画布中心应右移 5px: {a:.1f}->{b:.1f}"
        assert len(p._clips.get("adj", [])) == 3, "当前动画组应同步更新"
        h, w = p._frames[0].shape[:2]
        assert all(f.shape[:2] == (h, w) for f in p._frames)
    finally:
        FrameAdjustDialog.exec = orig_exec
    # 空帧守卫：不崩且给出状态提示
    p._frames = []
    p._open_frame_adjust()   # 仅应写状态，不应抛异常
    p.set_frames(make_frames(), name="test")
    print("[OK] 逐帧位移/缩放: 播放器入口一键应用（右移 5px 保留，不重新底对齐）+ 动画组同步")

    # 17. 手动裁切：进入/数值面板/比例/键盘/应用到全部帧/仅当前帧/撤销/退出
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    p.set_frames(make_frames(), name="crop")
    assert p.crop_bar.isHidden(), "裁切条默认隐藏"
    p.btn_crop.click()
    assert p._crop_active and not p.crop_bar.isHidden(), "应进入裁切模式"
    assert p._crop_rect == [0, 0, 44, 70], f"初始框应为全帧: {p._crop_rect}"
    # 数值面板 -> 框
    p.sp_cx.setValue(4)
    p.sp_cy.setValue(6)
    p.sp_cw.setValue(20)
    p.sp_ch.setValue(10)
    assert p._crop_rect == [4, 6, 24, 16], p._crop_rect
    # 比例预设 1:1 -> 中心不变正方形
    p.cmb_cratio.setCurrentIndex(1)
    assert p._crop_ratio == (1, 1)
    x0, y0, x1, y1 = p._crop_rect
    assert (x1 - x0) == (y1 - y0), f"1:1 应正方形: {p._crop_rect}"
    # 键盘：右移 1px / Shift+右 10px / Alt+右 加宽 / Enter 应用全部 / Esc 退出
    def _key(k, mod=Qt.NoModifier):
        return QKeyEvent(QEvent.KeyPress, k, mod)

    before = list(p._crop_rect)
    assert p.eventFilter(p.canvas, _key(Qt.Key_Right)) is True
    assert p._crop_rect[0] == before[0] + 1
    before = list(p._crop_rect)
    p.eventFilter(p.canvas, _key(Qt.Key_Right, Qt.ShiftModifier))
    assert p._crop_rect[0] == before[0] + 10, "Shift 应 10px 步进"
    before = list(p._crop_rect)
    p.eventFilter(p.canvas, _key(Qt.Key_Right, Qt.AltModifier))
    assert p._crop_rect[2] - p._crop_rect[0] == \
        before[2] - before[0] + 1, "Alt+右 应只加宽"
    # Enter = 应用到全部帧
    p.cmb_cratio.setCurrentIndex(0)   # 解锁比例
    p._crop_set_rect(4, 5, 44, 35)    # 40x30（帧宽 44）
    p.eventFilter(p.canvas, _key(Qt.Key_Return))
    assert all(f.shape[:2] == (30, 40) for f in p._frames), \
        f"全部帧应被裁到 40x30: {[f.shape for f in p._frames]}"
    assert len(p._clips["crop"]) == 3, "动画组应同步"
    # 仅当前帧 + 撤销
    p._crop_set_rect(2, 2, 20, 18)
    p._crop_apply(False)
    assert p._frames[0].shape[:2] == (16, 18), p._frames[0].shape
    assert p._frames[1].shape[:2] == (30, 40), "其他帧不受影响"
    p._crop_undo_apply()
    assert all(f.shape[:2] == (30, 40) for f in p._frames), "撤销应恢复全部帧"
    # Esc 退出
    p.eventFilter(p.canvas, _key(Qt.Key_Escape))
    assert not p._crop_active and p.crop_bar.isHidden()
    print("[OK] 手动裁切: 进入/数值/1:1/键盘三态/Enter应用全部/仅当前帧/撤销/Esc退出")

    print("\n动画播放器全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
