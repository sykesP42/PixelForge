"""专业看图器测试（离屏）：加载/缩放/平移/放大镜/鸟瞰图。

用法：QT_QPA_PLATFORM=offscreen python tests/test_image_viewer.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.ui.image_viewer import ProImageViewer


def main() -> int:
    app = QApplication(sys.argv)

    tmp = Path(tempfile.mkdtemp(prefix="pf_view_"))
    img = tmp / "grid.png"
    a = np.zeros((320, 640, 3), np.uint8)          # 640x320 BGR
    a[:, :320] = (255, 0, 0)                       # 左半蓝
    a[:, 320:] = (0, 0, 255)                       # 右半红
    cv2.imwrite(str(img), a)

    v = ProImageViewer()
    v.resize(800, 500)
    v.set_image(str(img))
    assert v.current_path() == str(img)
    assert v.image_size() == (640, 320)
    print("[OK] 1. 加载图片 + 尺寸")

    # 适配窗口：scale < 1
    s0 = v._scale()
    assert s0 < 1.0, f"适配后应缩小: {s0}"
    # 滚轮放大（光标锚点）
    e = QWheelEvent(QPointF(400, 250), QPointF(400, 250),
                    QPoint(0, 120), QPoint(0, 120), Qt.NoButton,
                    Qt.NoModifier, Qt.ScrollUpdate, False)
    v.view.wheelEvent(e)
    assert v._scale() > s0
    e2 = QWheelEvent(QPointF(400, 250), QPointF(400, 250),
                     QPoint(0, -120), QPoint(0, -120), Qt.NoButton,
                     Qt.NoModifier, Qt.ScrollUpdate, False)
    v.view.wheelEvent(e2)
    assert abs(v._scale() - s0) < 0.02
    print("[OK] 2. 滚轮缩放往返")

    # 1:1 与适配
    v.zoom_1to1()
    assert abs(v._scale() - 1.0) < 1e-6
    v.fit_window()
    print("[OK] 3. 1:1 与适配")

    # 放大镜开关
    v.set_loupe(True)
    assert v.btn_loupe.isChecked()
    v._mouse_pos = QPointF(400, 250)
    v._mouse_inside = True
    from PySide6.QtGui import QPixmap, QPainter
    pm = QPixmap(800, 500)
    p = QPainter(pm)
    v._draw_loupe(p)     # 不崩即通过
    p.end()
    v.set_loupe(False)
    assert not v._loupe
    print("[OK] 4. 放大镜开关 + 绘制不崩")

    # 鸟瞰图：视口矩形 + 定位
    vp = v.viewport_scene_rect()
    assert vp.width() > 0 and vp.height() > 0
    v.center_on_scene(QPointF(320, 160))
    assert v.viewport_scene_rect().contains(QPointF(320, 160))
    print("[OK] 5. 鸟瞰图视口矩形 + 定位")

    # Alt 按住临时放大镜：按下开、松开关；放大镜时滚轮调倍率
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    kp = QKeyEvent(QEvent.KeyPress, Qt.Key_Alt, Qt.NoModifier)
    kr = QKeyEvent(QEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier)
    v.view.keyPressEvent(kp)
    assert v._loupe and v._alt_held, "Alt 按下应临时开启放大镜"
    z0 = v.loupe_zoom
    v.view.wheelEvent(QWheelEvent(
        QPointF(400, 250), QPointF(400, 250), QPoint(0, 120),
        QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
    assert v.loupe_zoom == z0 + 1, "放大镜模式滚轮应调倍率"
    v.view.keyReleaseEvent(kr)
    assert not v._loupe and not v._alt_held, "Alt 松开应关闭放大镜"
    assert not v.btn_loupe.isChecked()
    assert hasattr(v, "lbl_tips") and "Alt" in v.lbl_tips.text()
    print("[OK] 6. Alt 按住临时放大镜 + 滚轮倍率 + 底部提示行")

    # 7. waifu2x：同步跑完（打桩 start）→ 自动打开结果图 + 进度/错误走日志
    from pixelforge.ui.w2x_worker import Waifu2xWorker
    small = tmp / "tiny.png"
    cv2.imwrite(str(small), np.full((12, 12, 4), (90, 120, 60, 255), np.uint8))
    v.set_image(str(small))
    msgs = []
    v.state_message.connect(msgs.append)
    orig_start = Waifu2xWorker.start
    Waifu2xWorker.start = lambda self: self.run()
    try:
        v.btn_w2x.click()
    finally:
        Waifu2xWorker.start = orig_start
    out_path = tmp / "tiny_waifu2x2x.png"
    assert out_path.is_file(), f"结果应写盘: {out_path}"
    assert v.current_path() == str(out_path), "完成后应自动打开结果图"
    assert v.image_size() == (24, 24), f"应为 2x: {v.image_size()}"
    assert any("waifu2x 完成" in m for m in msgs), f"应有完成日志: {msgs}"
    assert v.btn_w2x.isEnabled() and v.btn_w2x.text().endswith("waifu2x"), \
        "按钮应恢复"
    print("[OK] 7. waifu2x: 看图器一键放大（写盘+自动打开+日志）")

    # 8. 过大图守卫（不发推理，直接拒绝）
    big = np.zeros((1500, 1400, 4), np.uint8)          # 210 万像素 > 200 万
    wk = Waifu2xWorker(big, scale=2, noise_level=-1)
    errs = []
    wk.failed.connect(errs.append)
    wk.run()
    assert errs and "图片过大" in errs[0], f"应拒绝过大图: {errs}"
    print("[OK] 8. waifu2x: 超过 200 万像素守卫生效")

    print("\n看图器测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
