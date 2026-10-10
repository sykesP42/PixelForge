"""专业看图器：滚轮缩放 / 拖拽平移 / 放大镜 / 鸟瞰图（minimap）。

由「生成与审核」中央面板使用，替代原 QLabel 预览：
- 滚轮缩放（以光标为锚点，8% ~ 40x）
- 左键拖拽平移；双击或按 F 适配窗口；按 1 回 1:1
- 放大镜：点按钮开启（或按 M / 按住 Alt 临时开启），跟随鼠标，滚轮调倍率
- 鸟瞰图：右下角 minimap 显示全图 + 当前视口框，可在其上点击/拖动定位
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout, QWidget,
)

ZOOM_MIN = 0.08
ZOOM_MAX = 40.0
LOUPE_RADIUS = 70


class _Minimap(QWidget):
    """鸟瞰图：整图缩略 + 视口矩形，可拖动定位。"""

    def __init__(self, viewer: "ProImageViewer", parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setFixedSize(150, 100)
        self._dragging = False

    def _scene_rect(self) -> QRectF:
        w = self.viewer.image_size()[0] or 1
        h = self.viewer.image_size()[1] or 1
        return QRectF(0, 0, w, h)

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 22, 26, 220))
        pm = self.viewer.pixmap_full()
        if pm.isNull():
            return
        scaled = pm.scaled(self.size(), Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        p.drawPixmap(ox, oy, scaled)
        # 视口矩形（映射到 minimap 坐标）
        img_r = self._scene_rect()
        sx = scaled.width() / img_r.width()
        sy = scaled.height() / img_r.height()
        vp = self.viewer.viewport_scene_rect()
        r = QRectF(ox + vp.left() * sx, oy + vp.top() * sy,
                   vp.width() * sx, vp.height() * sy)
        p.setPen(QPen(QColor("#2f6fdb"), 1.5))
        p.drawRect(r)

    def _center_on(self, pos: QPoint) -> None:
        pm = self.viewer.pixmap_full()
        if pm.isNull():
            return
        scaled = pm.scaled(self.size(), Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
        ox = (self.width() - scaled.width()) / 2
        oy = (self.height() - scaled.height()) / 2
        img_r = self._scene_rect()
        sx = img_r.width() / max(1, scaled.width())
        sy = img_r.height() / max(1, scaled.height())
        self.viewer.center_on_scene(QPointF((pos.x() - ox) * sx,
                                            (pos.y() - oy) * sy))

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._center_on(e.position().toPoint())

    def mouseMoveEvent(self, e) -> None:
        if self._dragging:
            self._center_on(e.position().toPoint())

    def mouseReleaseEvent(self, e) -> None:
        self._dragging = False


class ProImageViewer(QWidget):
    """看图器主体：QGraphicsView + loupe 覆盖层 + minimap。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: str | None = None
        self._pm = QPixmap()
        self._loupe = False
        self._alt_held = False       # Alt 按住的临时放大镜
        self.loupe_zoom = 4          # 放大镜倍率（滚轮在放大镜模式下调节）
        self._mouse_pos = QPointF(-1, -1)
        self._mouse_inside = False

        self.view = _View(self)
        self.scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self.scene.addItem(self._item)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)

        self.minimap = _Minimap(self)
        self.minimap.setParent(self)

        self.btn_loupe = QPushButton("🔍 放大镜 (M)")
        self.btn_loupe.setCheckable(True)
        self.btn_loupe.setFixedHeight(24)
        self.btn_loupe.setToolTip(
            "开启后放大镜跟随鼠标移动；滚轮调节放大倍率（2x~20x）；按 M 切换")
        self.btn_loupe.toggled.connect(self.set_loupe)
        self.btn_fit = QPushButton("⤢ 适配 (F)")
        self.btn_fit.setFixedHeight(24)
        self.btn_fit.clicked.connect(self.fit_window)
        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.setFixedHeight(24)
        self.btn_1to1.clicked.connect(self.zoom_1to1)
        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        brow = QHBoxLayout()
        brow.addWidget(self.btn_loupe)
        brow.addWidget(self.btn_fit)
        brow.addWidget(self.btn_1to1)
        brow.addStretch(1)
        rl.addLayout(brow)
        rl.addWidget(self.view, 1)
        # 工业软件式操作提示行（看图区底部）
        self.lbl_tips = QLabel(
            "滚轮=缩放｜拖拽=平移｜双击/F=适配｜1=1:1｜M=放大镜｜"
            "Alt 按住=临时放大镜｜放大镜时滚轮=倍率｜右下角=鸟瞰图定位")
        self.lbl_tips.setStyleSheet("color:#8a8f98; font-size:11px;")
        rl.addWidget(self.lbl_tips)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(row, 1)

        self.view.setMouseTracking(True)
        self.setMouseTracking(True)

    # ---- 对外接口 ----

    def set_image(self, path: str | None) -> None:
        """加载图片；path 为空则清空。"""
        if not path or not Path(path).is_file():
            return
        self._path = str(path)
        data = np.fromfile(path, np.uint8)          # 中文路径安全
        im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if im is None:
            return
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        rgba = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._pm = QPixmap.fromImage(img.copy())
        self._item.setPixmap(self._pm)
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        self.fit_window()
        self.minimap.update()

    def current_path(self) -> str | None:
        return self._path

    def pixmap_full(self) -> QPixmap:
        return self._pm

    def image_size(self) -> tuple[int, int]:
        return int(self._pm.width()), int(self._pm.height())

    def viewport_scene_rect(self) -> QRectF:
        return self.view.mapToScene(self.view.viewport().rect()).boundingRect()

    def center_on_scene(self, sp: QPointF) -> None:
        self.view.centerOn(sp)

    # ---- 缩放/视图 ----

    def _scale(self) -> float:
        return self.view.transform().m11()

    def fit_window(self) -> None:
        if self._pm.isNull():
            return
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_1to1(self) -> None:
        if self._pm.isNull():
            return
        self.view.resetTransform()

    def _zoom_at(self, factor: float, view_pos: QPointF) -> None:
        cur = self._scale()
        new = max(ZOOM_MIN, min(ZOOM_MAX, cur * factor))
        factor = new / cur
        before = self.view.mapToScene(view_pos.toPoint())
        self.view.scale(factor, factor)
        after = self.view.mapToScene(view_pos.toPoint())
        delta = after - before
        self.view.translate(delta.x(), delta.y())
        self.minimap.update()

    def set_loupe_zoom(self, z: int) -> None:
        self.loupe_zoom = max(2, min(20, z))
        self.btn_loupe.setText(f"🔍 {self.loupe_zoom}x (M)")
        self.view.viewport().update()

    # ---- 放大镜 ----

    def set_loupe(self, on: bool) -> None:
        self._loupe = on
        self.btn_loupe.setChecked(on)
        self.btn_loupe.setText(
            f"🔍 {self.loupe_zoom}x (M)" if on else "🔍 放大镜 (M)")
        self.view.viewport().update()

    def toggle_loupe(self) -> None:
        self.set_loupe(not self._loupe)

    def _draw_loupe(self, p: QPainter) -> None:
        if not (self._loupe and self._mouse_inside and not self._pm.isNull()):
            return
        vp = self.view.viewport().rect()
        pos = self._mouse_pos.toPoint()
        if not vp.contains(pos):
            return
        sp = self.view.mapToScene(pos)
        src_scale = self.loupe_zoom          # 放大镜倍率（相对原始像素）
        r = LOUPE_RADIUS
        # 圆形 loupe：以场景点为中心取源区域，放大绘制
        src_w = 2 * r / src_scale
        src = QRectF(sp.x() - src_w / 2, sp.y() - src_w / 2, src_w, src_w)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addEllipse(QPointF(pos.x(), pos.y()), r, r)
        p.setClipPath(path)
        p.drawPixmap(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r).toRect(),
                     self._pm, src.toRect())
        p.restore()
        p.setPen(QPen(QColor("#2f6fdb"), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(pos.x(), pos.y()), r, r)


class _View(QGraphicsView):
    """滚轮缩放（光标锚点）+ M 键放大镜 + loupe 绘制。"""

    def __init__(self, owner: ProImageViewer):
        super().__init__()
        self.owner = owner
        self.setStyleSheet("border: none; background: #1a1b1f;")
        self.setFocusPolicy(Qt.StrongFocus)   # 接收 M/Alt/F/1 快捷键

    def wheelEvent(self, e) -> None:
        if self.owner._pm.isNull():
            return
        if self.owner._loupe:
            # 放大镜模式下：滚轮调节放大倍率
            self.owner.set_loupe_zoom(
                self.owner.loupe_zoom + (1 if e.angleDelta().y() > 0 else -1))
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.owner._zoom_at(factor, e.position())

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Alt:
            # 按住 Alt：临时开启放大镜（若按钮未开）；松开恢复
            if not self.owner._loupe:
                self.owner._alt_held = True
                self.owner.set_loupe(True)
            e.accept()
            return
        if e.key() == Qt.Key_M:
            self.owner.toggle_loupe()
        elif e.key() == Qt.Key_F:
            self.owner.fit_window()
        elif e.key() == Qt.Key_1:
            self.owner.zoom_1to1()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Alt and self.owner._alt_held:
            self.owner._alt_held = False
            self.owner.set_loupe(False)
            e.accept()
            return
        super().keyReleaseEvent(e)

    def mouseMoveEvent(self, e) -> None:
        self.owner._mouse_pos = e.position()
        self.owner._mouse_inside = True
        self.owner.minimap.update()
        if self.owner._loupe:
            self.viewport().update()   # 放大镜实时跟随鼠标
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self.owner._mouse_inside = False
        self.owner.view.viewport().update()
        super().leaveEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        self.owner.fit_window()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self.viewport())
        self.owner._draw_loupe(p)
        p.end()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        mm = self.owner.minimap
        vp = self.viewport()
        mm.move(self.owner.width() - mm.width() - 12,
                max(0, vp.mapTo(self.owner, vp.rect().topLeft()).y()
                    + self.height() - mm.height() - 12))
        self.owner.minimap.update()
