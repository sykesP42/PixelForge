"""像素编辑器测试（离屏）：画布绘制/撤销/工具/保存。

用法：QT_QPA_PLATFORM=offscreen python tests/test_pixel_editor.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.ui.pixel_editor import PixelEditorDialog


def _click(canvas, x: int, y: int) -> None:
    for etype, btn in ((QEvent.MouseButtonPress, Qt.LeftButton),
                       (QEvent.MouseButtonRelease, Qt.LeftButton)):
        e = QMouseEvent(etype, QPointF(x, y), btn, btn, Qt.NoModifier)
        (canvas.mousePressEvent if etype == QEvent.MouseButtonPress
         else canvas.mouseReleaseEvent)(e)


def main() -> int:
    app = QApplication(sys.argv)

    tmp = Path(tempfile.mkdtemp(prefix="pf_edit_"))
    img = tmp / "sprite.png"
    a = np.zeros((32, 32, 4), np.uint8)
    a[10:14, 10:14] = (255, 0, 0, 255)   # BGR 蓝点
    cv2.imwrite(str(img), a)

    dlg = PixelEditorDialog(str(img))
    cv = dlg.canvas
    assert cv.bgra.shape[:2] == (32, 32)
    print("[OK] 1. 打开 32x32 图像")

    # 画笔点击 (4,4) -> cell(0,0) -> 入撤销栈
    n0 = cv.undo_stack.count()
    _click(cv, 4, 4)
    assert cv.undo_stack.count() == n0 + 1, "一笔应入撤销栈"
    assert tuple(cv.bgra[0, 0]) == (0, 0, 0, 255)
    print("[OK] 2. 画笔绘制 + 撤销栈入栈")

    # 撤销恢复透明
    cv.undo_stack.undo()
    assert int(cv.bgra[0, 0, 3]) == 0
    print("[OK] 3. 撤销恢复原像素")

    # 吸管：取蓝点颜色（BGR 255,0,0）
    cv.tool = "picker"
    e = QMouseEvent(QEvent.MouseButtonPress, QPointF(12 * 12, 12 * 12),
                    Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    cv.mousePressEvent(e)
    assert tuple(cv.color) == (255, 0, 0, 255), f"吸管取色错: {cv.color}"
    print("[OK] 4. 吸管取色")

    # 橡皮 + 缩放
    cv.tool = "eraser"
    cv.set_zoom(24)
    assert cv.zoom == 24
    _click(cv, 4, 4)
    assert int(cv.bgra[0, 0, 3]) == 0
    print("[OK] 5. 橡皮 + 缩放档位")

    # 保存覆盖
    dlg._save()
    b = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    assert b is not None and b.shape[:2] == (32, 32)
    got = []
    dlg.edit_done.connect(lambda p: got.append(p))
    dlg._save()
    assert got and Path(got[0]) == img
    print("[OK] 6. 保存覆盖 + edit_done 信号")

    print("\n像素编辑器测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
