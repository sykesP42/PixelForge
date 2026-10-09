"""像素图片人工编辑器（独立模块）。

功能：画笔 / 橡皮 / 吸管 / 油漆填充 / 笔刷大小 / 撤销重做 / 缩放网格 /
调色板（从图取色 + 常用色）/ 保存覆盖 / 另存为。
由资产库右键「✏ 编辑」打开。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QFileDialog, QHBoxLayout, QLabel, QInputDialog,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from PySide6.QtGui import QUndoCommand, QUndoStack

PALETTE_STD = [
    "#000000", "#ffffff", "#7f7f7f", "#c3c3c3",
    "#880015", "#ed1c24", "#ff7f27", "#fff200",
    "#22b14c", "#00a2e8", "#3f48cc", "#a349a4",
    "#b97a57", "#ffaec9", "#ffc90e", "#efe4b0",
    "#b5e61d", "#99d9ea", "#7092be", "#c8bfe7",
]


def _load_bgra(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), np.uint8)      # 中文路径安全
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"无法读取图片: {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


class PaintCommand(QUndoCommand):
    """一笔 = 一个撤销单元（保存绘制前后的像素补丁）。"""

    def __init__(self, canvas: "PixelCanvas", before: dict, after: dict,
                 text: str = "paint"):
        super().__init__(text)
        self.canvas = canvas
        self.before, self.after = before, after

    def undo(self) -> None:
        self.canvas.apply_patch(self.before)

    def redo(self) -> None:
        self.canvas.apply_patch(self.after)


class PixelCanvas(QWidget):
    """缩放画布：网格线 + 像素级绘制。"""

    def __init__(self, bgra: np.ndarray, parent=None):
        super().__init__(parent)
        self.bgra = bgra
        self.zoom = 12
        self.tool = "pen"          # pen | eraser | picker | fill
        self.brush = 1
        self.color = (0, 0, 0, 255)      # BGR primary
        self._drawing = False
        self._patch: dict = {}
        self.undo_stack = QUndoStack(self)
        self.setMouseTracking(True)
        self._update_geom()

    # ---- 几何 ----
    def _update_geom(self) -> None:
        h, w = self.bgra.shape[:2]
        self.setFixedSize(w * self.zoom, h * self.zoom)
        self.update()

    def set_zoom(self, z: int) -> None:
        self.zoom = max(2, min(40, z))
        self._update_geom()

    def image_size(self) -> tuple[int, int]:
        h, w = self.bgra.shape[:2]
        return w, h

    # ---- 坐标 ----
    def _cell(self, pos: QPoint) -> tuple[int, int] | None:
        x, y = pos.x() // self.zoom, pos.y() // self.zoom
        h, w = self.bgra.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    # ---- 绘制 ----
    def apply_patch(self, patch: dict) -> None:
        for (x, y), v in patch.items():
            self.bgra[y, x] = v
        self.update()

    def _stamp(self, x: int, y: int, color) -> None:
        r = self.brush - 1
        h, w = self.bgra.shape[:2]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    key = (xx, yy)
                    if key not in self._patch:
                        self._patch[key] = self.bgra[yy, xx].copy()
                    self.bgra[yy, xx] = color

    def _flood(self, x: int, y: int, color) -> None:
        h, w = self.bgra.shape[:2]
        target = self.bgra[y, x].copy()
        if tuple(target) == tuple(color):
            return
        from collections import deque
        dq = deque([(x, y)])
        seen = {(x, y)}
        while dq:
            cx, cy = dq.popleft()
            key = (cx, cy)
            if key not in self._patch:
                self._patch[key] = self.bgra[cy, cx].copy()
            self.bgra[cy, cx] = color
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen
                        and tuple(self.bgra[ny, nx]) == tuple(target)):
                    seen.add((nx, ny))
                    dq.append((nx, ny))

    def _commit(self) -> None:
        if not self._patch:
            return
        after = {k: self.bgra[k[1], k[0]].copy() for k in self._patch}
        self.undo_stack.push(PaintCommand(self, self._patch, after))
        self._patch = {}
        self.update()

    # ---- 事件 ----
    def mousePressEvent(self, e) -> None:
        c = self._cell(e.position().toPoint())
        if c is None:
            return
        x, y = c
        if e.button() == Qt.LeftButton:
            if self.tool == "picker":
                self.color = tuple(int(v) for v in self.bgra[y, x])
                self.window().sync_palette_selection(self.color)
                return
            self._drawing = True
            self._patch = {}
            if self.tool == "fill":
                self._flood(x, y, self.color)
                self._commit()
            else:
                col = (0, 0, 0, 0) if self.tool == "eraser" else self.color
                self._stamp(x, y, col)
                self.update()
        elif e.button() == Qt.RightButton and self.tool == "pen":
            self.color = tuple(int(v) for v in self.bgra[y, x])
            self.window().sync_palette_selection(self.color)

    def mouseMoveEvent(self, e) -> None:
        if not self._drawing:
            return
        c = self._cell(e.position().toPoint())
        if c is None:
            return
        col = (0, 0, 0, 0) if self.tool == "eraser" else self.color
        self._stamp(*c, col)
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._drawing:
            self._drawing = False
            self._commit()

    def wheelEvent(self, e) -> None:
        if e.modifiers() & Qt.ControlModifier:
            self.set_zoom(self.zoom + (2 if e.angleDelta().y() > 0 else -2))
        else:
            super().wheelEvent(e)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        h, w = self.bgra.shape[:2]
        rgba = cv2.cvtColor(self.bgra, cv2.COLOR_BGRA2RGBA)
        img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        # 棋盘透明底
        p.fillRect(self.rect(), QColor("#808080"))
        for yy in range(0, self.height(), self.zoom * 4):
            for xx in range(0, self.width(), self.zoom * 4):
                p.fillRect(xx, yy, self.zoom * 2, self.zoom * 2, QColor("#a0a0a0"))
                p.fillRect(xx + self.zoom * 2, yy + self.zoom * 2,
                           self.zoom * 2, self.zoom * 2, QColor("#a0a0a0"))
        p.drawImage(self.rect(), img)
        if self.zoom >= 10:      # 高倍率画网格线
            p.setPen(QPen(QColor(0, 0, 0, 40)))
            for xx in range(0, self.width() + 1, self.zoom):
                p.drawLine(xx, 0, xx, self.height())
            for yy in range(0, self.height() + 1, self.zoom):
                p.drawLine(0, yy, self.width(), yy)


class PixelEditorDialog(QWidget):
    """像素编辑器窗口。edit_done = Signal(str) 保存路径（外部刷新资产库）。"""

    edit_done = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"像素编辑器 — {Path(path).name}")
        self.resize(1000, 720)
        self.path = Path(path)
        self.bgra = _load_bgra(self.path)
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._tool_btns: list[QPushButton] = []
        for label, tool in (("✏ 画笔", "pen"), ("🧽 橡皮", "eraser"),
                            ("💧 填充", "fill"), ("🎯 吸管", "picker")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _, t=tool: self._pick_tool(t, b))
            b.setProperty("toolBtn", tool)
            self._tool_btns.append(b)
            bar.addWidget(b)
        self._tool_btns[0].setChecked(True)
        bar.addWidget(QLabel("笔刷"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 6)
        bar.addWidget(self.spin_brush)
        b_color = QPushButton("🎨 颜色…")
        b_color.clicked.connect(self._pick_color)
        bar.addWidget(b_color)
        bar.addStretch(1)
        b_undo = QPushButton("↩ 撤销")
        b_undo.clicked.connect(lambda: self.canvas.undo_stack.undo())
        bar.addWidget(b_undo)
        b_redo = QPushButton("↪ 重做")
        b_redo.clicked.connect(lambda: self.canvas.undo_stack.redo())
        bar.addWidget(b_redo)
        v.addLayout(bar)

        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("调色板"))
        self._pal_btns: list[QPushButton] = []
        for hexc in self._image_palette() + PALETTE_STD:
            b = QPushButton()
            b.setFixedSize(20, 20)
            b.setToolTip(hexc)
            b.setStyleSheet(f"background:{hexc}; border:1px solid #666;")
            col = QColor(hexc)
            bgr = (col.blue(), col.green(), col.red(), 255)
            b.clicked.connect(lambda _, c=bgr: self._set_color(c))
            pal_row.addWidget(b)
            self._pal_btns.append(b)
        pal_row.addStretch(1)
        v.addLayout(pal_row)

        self.canvas = PixelCanvas(self.bgra, self)
        self.spin_brush.valueChanged.connect(
            lambda v: setattr(self.canvas, "brush", v))
        sa = QScrollArea()
        sa.setWidgetResizable(False)
        sa.setWidget(self.canvas)
        sa.setAlignment(Qt.AlignCenter)
        v.addWidget(sa, 1)

        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("缩放"))
        self.cmb_zoom = QComboBox()
        for z in (4, 8, 12, 16, 24, 32):
            self.cmb_zoom.addItem(f"{z}x", z)
        self.cmb_zoom.setCurrentText("12x")
        self.cmb_zoom.currentIndexChanged.connect(
            lambda _: self.canvas.set_zoom(self.cmb_zoom.currentData()))
        bar2.addWidget(self.cmb_zoom)
        self.lbl_size = QLabel(
            f"{self.canvas.image_size()[0]}×{self.canvas.image_size()[1]} px")
        bar2.addWidget(self.lbl_size)
        bar2.addStretch(1)
        b_save = QPushButton("💾 保存")
        b_save.setProperty("accent", True)
        b_save.clicked.connect(self._save)
        bar2.addWidget(b_save)
        b_saveas = QPushButton("另存为…")
        b_saveas.clicked.connect(self._save_as)
        bar2.addWidget(b_saveas)
        v.addLayout(bar2)

    def _image_palette(self) -> list[str]:
        px = self.bgra[self.bgra[:, :, 3] > 0][:, :3]
        if not px.size:
            return []
        uniq, cnt = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-cnt)[:20]
        out = []
        for b, g, r in uniq[order]:
            out.append(f"#{r:02x}{g:02x}{b:02x}")
        return out

    def _pick_tool(self, tool: str, btn: QPushButton) -> None:
        self.canvas.tool = tool
        for b in self._tool_btns:
            b.setChecked(b is btn)

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(
            self.canvas.color[2], self.canvas.color[1], self.canvas.color[0]),
            self, "选择颜色")
        if c.isValid():
            self._set_color((c.blue(), c.green(), c.red(), 255))

    def _set_color(self, bgr) -> None:
        self.canvas.color = tuple(bgr)

    def sync_palette_selection(self, color) -> None:
        """吸管取色后高亮对应调色板（简单实现：仅更新画笔色）。"""
        pass

    def _save(self) -> None:
        ok, enc = cv2.imencode(".png", self.bgra)
        if ok:
            enc.tofile(str(self.path))     # 中文路径安全
            self.edit_done.emit(str(self.path))

    def _save_as(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(
            self, "另存为", str(self.path.parent / "edited.png"),
            "PNG (*.png)")
        if fn:
            ok, enc = cv2.imencode(".png", self.bgra)
            if ok:
                enc.tofile(fn)
                self.edit_done.emit(fn)


if __name__ == "__main__":  # 独立调试
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    p = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    w = PixelEditorDialog(p)
    w.show()
    sys.exit(app.exec())
