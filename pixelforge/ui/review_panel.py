"""质量审核面板（阶段3）：帧网格、洋葱皮、循环播放、单帧操作、导出门禁。

独立可测：set_frames() 喂入帧 PNG + atlas 元数据；delete_frame / move_frame /
crop_frame / export_to 为纯逻辑方法，GUI 事件只是薄封装。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMenu, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from pixelforge.processing.image_processor import frame_hull
from pixelforge.processing.motion_metrics import coherence_score


def _bgra_to_pixmap(arr: np.ndarray) -> QPixmap:
    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


def _onion_composite(frames: list[np.ndarray], weights=(0.3, 1.0, 0.3)) -> np.ndarray:
    """前后帧半透明叠加到当前帧（自动统一画布尺寸）。"""
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    canvas = np.zeros((h, w, 4), np.float32)
    for arr, a in zip(frames, weights):
        if arr.shape[:2] != (h, w):
            pad = np.zeros((h, w, 4), np.uint8)
            pad[:arr.shape[0], :arr.shape[1]] = arr
            arr = pad
        over = arr.astype(np.float32)
        over[:, :, 3] *= a
        af = over[:, :, 3:4] / 255.0
        canvas = over * af + canvas * (1.0 - af)
    return canvas.astype(np.uint8)


class CropDialog(QDialog):
    def __init__(self, parent=None, max_w: int = 512, max_h: int = 512):
        super().__init__(parent)
        self.setWindowTitle("手动裁剪边缘")
        form = QFormLayout(self)
        self.sp_l = QSpinBox(); self.sp_l.setRange(0, max_w)
        self.sp_t = QSpinBox(); self.sp_t.setRange(0, max_h)
        self.sp_r = QSpinBox(); self.sp_r.setRange(0, max_w)
        self.sp_b = QSpinBox(); self.sp_b.setRange(0, max_h)
        form.addRow("左", self.sp_l)
        form.addRow("上", self.sp_t)
        form.addRow("右", self.sp_r)
        form.addRow("下", self.sp_b)
        btn = QPushButton("应用")
        btn.clicked.connect(self.accept)
        form.addRow(btn)


class ReviewWidget(QWidget):
    state_changed = Signal(str)   # 日志消息
    frames_changed = Signal()
    reroll_requested = Signal(int)  # 请求 AI 重抽指定行
    edit_requested = Signal(int)    # 请求送工具坊编辑指定行

    ICON_SIZE = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = "frames"
        self._pos = -1
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ---------- UI ----------

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        self.preview = QLabel("审核区：生成后自动裁切显示帧")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setStyleSheet("background:#2b2d31; color:#9aa0a6;")
        lay.addWidget(self.preview, 1)

        bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(8)
        self.spin_fps.setSuffix(" FPS")
        self.chk_onion = QCheckBox("洋葱皮")
        self.chk_onion.toggled.connect(lambda _: self._render_current())
        bar.addWidget(self.btn_play)
        bar.addWidget(self.spin_fps)
        bar.addWidget(self.chk_onion)
        bar.addStretch(1)

        self.chk_reviewed = QCheckBox("审核通过")
        self.btn_export = QPushButton("导出到 Unity")
        self.btn_export.setProperty("accent", True)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_clicked)
        bar.addWidget(self.chk_reviewed)
        bar.addWidget(self.btn_export)
        lay.addLayout(bar)

        self.lst = QListWidget()
        self.lst.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.lst.setViewMode(QListWidget.IconMode)
        self.lst.setResizeMode(QListWidget.Adjust)
        self.lst.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._context_menu)
        self.lst.currentRowChanged.connect(lambda _: self._render_current())
        self.lst.itemChanged.connect(lambda _: self._render_current())
        lay.addWidget(self.lst, 2)

    # ---------- 数据装载 ----------

    def set_frames(self, files: list[Path], atlas: dict) -> None:
        self._timer.stop()
        self.btn_play.setText("▶ 播放")
        self.lst.blockSignals(True)  # 防止装载过程触发 itemChanged 重入渲染
        try:
            self.lst.clear()
            self._name = atlas.get("name", "frames")
            self.spin_fps.setValue(int(atlas.get("fps", 8)))
            arrs = [self._load_norm(Path(f)) for f in files]
            arrs = self._normalize(arrs)
            for f, a in zip(files, arrs):
                self._append_item(Path(f), a)
            self._pos = 0 if self.lst.count() else -1
        finally:
            self.lst.blockSignals(False)
        self._sync_buttons()
        self._render_current()
        self.frames_changed.emit()

    @staticmethod
    def _load_norm(path: Path) -> np.ndarray:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        return im

    @staticmethod
    def _normalize(arrs: list[np.ndarray]) -> list[np.ndarray]:
        """统一画布：底对齐（基线）+ 水平居中。"""
        if not arrs:
            return []
        h = max(a.shape[0] for a in arrs)
        w = max(a.shape[1] for a in arrs)
        out = []
        for a in arrs:
            canvas = np.zeros((h, w, 4), np.uint8)
            dy = h - a.shape[0]
            dx = (w - a.shape[1]) // 2
            canvas[dy:dy + a.shape[0], dx:dx + a.shape[1]] = a
            out.append(canvas)
        return out

    def _append_item(self, path: Path, arr: np.ndarray) -> None:
        item = QListWidgetItem(path.name)
        item.setData(Qt.UserRole, {"path": path, "arr": arr})
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setIcon(_icon_from(arr, self.ICON_SIZE))
        self.lst.addItem(item)

    # ---------- 单帧操作（纯逻辑，供测试） ----------

    def delete_frame(self, row: int) -> None:
        if 0 <= row < self.lst.count():
            self.lst.takeItem(row)
            self._pos = min(self._pos, self.lst.count() - 1)
            self._sync_buttons()
            self._render_current()
            self.frames_changed.emit()

    def move_frame(self, row: int, delta: int) -> bool:
        n = self.lst.count()
        new = row + delta
        if not (0 <= row < n) or not (0 <= new < n):
            return False
        item = self.lst.takeItem(row)
        self.lst.insertItem(new, item)
        self.lst.setCurrentRow(new)
        self.frames_changed.emit()
        return True

    def frame_arr(self, row: int) -> np.ndarray | None:
        """取指定行当前帧数组（供单帧重抽等外部流程使用）。"""
        item = self.lst.item(row)
        return item.data(Qt.UserRole)["arr"] if item else None

    def replace_frame(self, row: int, arr: np.ndarray) -> bool:
        """替换指定行帧内容（单帧重抽完成后回调）。"""
        item = self.lst.item(row)
        if item is None:
            return False
        self._set_arr(item, arr)
        it2 = self.lst.item(row)
        if it2 is not None:
            it2.setIcon(_icon_from(arr, self.ICON_SIZE))
        self._renormalize()
        self._render_current()
        self.frames_changed.emit()
        return True

    def coherence_report(self) -> list[tuple[int, int, float]]:
        """全序列相邻帧连贯性报告：[(帧i, 帧i+1, 得分)]，纯逻辑可测。"""
        out = []
        for i in range(self.lst.count() - 1):
            a = self.frame_arr(i)
            b = self.frame_arr(i + 1)
            if a is not None and b is not None:
                out.append((i, i + 1, coherence_score(a, b)))
        return out

    def _show_coherence_report(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        pairs = self.coherence_report()
        if not pairs:
            QMessageBox.information(self, "连贯性分析", "帧数不足（至少 2 帧）")
            return
        lines = []
        worst = None
        for i, j, s in pairs:
            flag = " ⚠" if s < 0.45 else ""
            lines.append(f"帧 {i} ↔ {j}:  {s}{flag}")
            if worst is None or s < worst[2]:
                worst = (i, j, s)
        avg = sum(s for _, _, s in pairs) / len(pairs)
        lines.insert(0, f"平均连贯性: {avg:.3f}")
        if worst is not None and worst[2] < 0.45:
            lines.append(f"\n建议优先重抽帧 {worst[1]}（与帧 {worst[0]} 差异最大）")
        QMessageBox.information(self, "连贯性分析", "\n".join(lines))

    def _adjust_frame(self, row: int, fn, label: str) -> None:
        """就地调整：fn(帧数组) -> 新帧，直接回填（纯逻辑可测）。"""
        arr = self.frame_arr(row)
        if arr is None:
            return
        try:
            new = fn(arr)
        except Exception as e:  # noqa: BLE001
            self.state_changed.emit(f"调整失败: {e}")
            return
        if new.shape[:2] != arr.shape[:2]:
            self._set_arr(self.lst.item(row), new)
            self._renormalize()
        else:
            self._set_arr(self.lst.item(row), new)
        it = self.lst.item(row)
        if it is not None:
            it.setIcon(_icon_from(new, self.ICON_SIZE))
        self._render_current()
        self.frames_changed.emit()
        self.state_changed.emit(f"帧 #{row} 已{label}")

    def crop_frame(self, row: int, left: int, top: int, right: int, bottom: int) -> bool:
        item = self.lst.item(row)
        if item is None:
            return False
        arr = item.data(Qt.UserRole)["arr"]
        h, w = arr.shape[:2]
        if left + right >= w or top + bottom >= h:
            return False
        self._set_arr(item, arr[top:h - bottom, left:w - right].copy())
        self._renormalize()
        self._render_current()
        self.frames_changed.emit()
        return True

    @staticmethod
    def _set_arr(item: QListWidgetItem, arr: np.ndarray) -> None:
        d = item.data(Qt.UserRole)
        d["arr"] = arr
        item.setData(Qt.UserRole, d)  # data() 返回副本，必须显式写回

    def _renormalize(self) -> None:
        items = [self.lst.item(i) for i in range(self.lst.count())]
        arrs = [it.data(Qt.UserRole)["arr"] for it in items]
        self.lst.blockSignals(True)  # setData 会触发 itemChanged，阻塞防重入
        try:
            for it, a in zip(items, self._normalize(arrs)):
                self._set_arr(it, a)
                it.setIcon(_icon_from(a, self.ICON_SIZE))
        finally:
            self.lst.blockSignals(False)

    def checked_count(self) -> int:
        return sum(1 for i in range(self.lst.count())
                   if self.lst.item(i).checkState() == Qt.Checked)

    def _ordered_frames(self) -> list[np.ndarray]:
        idxs = [i for i in range(self.lst.count())
                if self.lst.item(i).checkState() == Qt.Checked]
        if not idxs:  # 全未勾选时按全部帧播放
            idxs = list(range(self.lst.count()))
        return [self.lst.item(i).data(Qt.UserRole)["arr"] for i in idxs]

    # ---------- 播放 / 渲染 ----------

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            if self.lst.count() == 0:
                return
            self._timer.start(max(16, 1000 // max(1, self.spin_fps.value())))
            self.btn_play.setText("⏸ 暂停")

    def _advance(self) -> None:
        n = max(1, len(self._ordered_frames()))
        self._pos = (self._pos + 1) % n
        self._render_current()

    def _render_current(self) -> None:
        frames = self._ordered_frames()
        if not frames or self._pos < 0:
            self.preview.setText("审核区：生成后自动裁切显示帧")
            return
        self._pos %= len(frames)
        if self.chk_onion.isChecked() and len(frames) > 1:
            trio = [frames[(self._pos - 1) % len(frames)], frames[self._pos],
                    frames[(self._pos + 1) % len(frames)]]
            arr = _onion_composite(trio)
        else:
            arr = frames[self._pos]
        pm = _bgra_to_pixmap(arr)
        self.preview.setPixmap(pm.scaled(
            self.preview.width(), self.preview.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _sync_buttons(self) -> None:
        self.btn_export.setEnabled(
            self.chk_reviewed.isChecked() and self.lst.count() > 0)

    # ---------- 右键菜单 ----------

    def _context_menu(self, pos) -> None:
        item = self.lst.itemAt(pos)
        if item is None:
            return
        row = self.lst.row(item)
        menu = QMenu(self)
        act_reroll = menu.addAction("🔄 AI 重抽本帧...")
        act_edit = menu.addAction("✏ 在工具坊中编辑")
        act_coherence = menu.addAction("📊 连贯性分析（全序列）")
        act_adj = menu.addMenu("就地调整")
        act_flip = act_adj.addAction("水平翻转")
        act_bright = act_adj.addAction("亮度 +15")
        act_dark = act_adj.addAction("亮度 -15")
        act_rot = act_adj.addAction("旋转 90°")
        act_crop = act_adj.addAction("裁边 4px")
        menu.addSeparator()
        act_cropdlg = menu.addAction("手动裁剪边缘")
        act_up = menu.addAction("上移")
        act_down = menu.addAction("下移")
        menu.addSeparator()
        act_del = menu.addAction("删除帧")
        chosen = menu.exec(self.lst.mapToGlobal(pos))
        if chosen == act_del:
            self.state_changed.emit(f"已删除帧 #{row}")
            self.delete_frame(row)
        elif chosen == act_reroll:
            self.reroll_requested.emit(row)
        elif chosen == act_edit:
            self.edit_requested.emit(row)
        elif chosen == act_coherence:
            self._show_coherence_report()
        elif chosen == act_flip:
            self._adjust_frame(row, lambda a: a[:, ::-1].copy(), "水平翻转")
        elif chosen == act_bright:
            self._adjust_frame(
                row, lambda a: _shift_brightness(a, 15), "亮度 +15")
        elif chosen == act_dark:
            self._adjust_frame(
                row, lambda a: _shift_brightness(a, -15), "亮度 -15")
        elif chosen == act_rot:
            self._adjust_frame(
                row, lambda a: np.ascontiguousarray(np.rot90(a)), "旋转 90°")
        elif chosen == act_crop:
            self._adjust_frame(
                row, lambda a: a[4:-4, 4:-4].copy(), "裁边 4px")
        elif chosen == act_cropdlg:
            self.move_frame(row, -1)
        elif chosen == act_down:
            self.move_frame(row, 1)
        elif chosen == act_crop:
            arr = item.data(Qt.UserRole)["arr"]
            dlg = CropDialog(self, arr.shape[1], arr.shape[0])
            if dlg.exec():
                self.crop_frame(row, dlg.sp_l.value(), dlg.sp_t.value(),
                                dlg.sp_r.value(), dlg.sp_b.value())
                self.state_changed.emit(f"帧 #{row} 已手动裁剪")

    # ---------- 导出 ----------

    def _on_export_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if not self.chk_reviewed.isChecked():
            QMessageBox.warning(self, "PixelForge", "请先勾选“审核通过”，才允许导出到 Unity")
            return
        d = QFileDialog.getExistingDirectory(self, "选择 Unity 项目 Assets/Textures/Generated 目录")
        if not d:
            return
        files = self.export_to(Path(d))
        QMessageBox.information(self, "PixelForge", f"已导出 {len(files)} 个文件到:\n{d}")

    def export_to(self, out_dir: Path) -> list[Path]:
        """导出勾选帧（按顺序）+ atlas.json，供 Unity 侧消费。"""
        if not self.chk_reviewed.isChecked():
            raise PermissionError("未通过人工审核，禁止导出")
        idxs = [i for i in range(self.lst.count())
                if self.lst.item(i).checkState() == Qt.Checked]
        if not idxs:
            idxs = list(range(self.lst.count()))
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_meta = []
        files: list[Path] = []
        for i, row in enumerate(idxs):
            arr = self.lst.item(row).data(Qt.UserRole)["arr"]
            fname = f"{self._name}_{i:03d}.png"
            p = out_dir / fname
            cv2.imwrite(str(p), arr)
            files.append(p)
            ys, xs = np.where(arr[:, :, 3] > 0)
            fx = 0.5
            if ys.size:
                x0, y0 = int(xs.min()), int(ys.min())
                rects = {"x": x0, "y": y0,
                         "width": int(xs.max() - x0 + 1), "height": int(ys.max() - y0 + 1)}
                # 锚点 = 脚底带（底部 15%）不透明像素中心，保证播放对齐
                bottom = int(ys.max())
                band_top = max(0, bottom - max(2, int((bottom + 1) * 0.15)))
                bxs = np.where(arr[band_top:bottom + 1, :, 3] > 0)[1]
                if bxs.size:
                    fx = float(bxs.mean()) / arr.shape[1]
            else:
                rects = {"x": 0, "y": 0, "width": arr.shape[1], "height": arr.shape[0]}
            frames_meta.append(rects | {"file": fname,
                                        "pivotX": round(fx, 4), "pivotY": 1.0,
                                        "hull": frame_hull(arr)})
        atlas = {"name": self._name, "fps": self.spin_fps.value(),
                 "frame_count": len(frames_meta), "frames": frames_meta}
        jp = out_dir / f"{self._name}.atlas.json"
        jp.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        files.append(jp)
        self.state_changed.emit(f"已导出 {len(frames_meta)} 帧到 {out_dir}")
        return files


def _shift_brightness(arr: np.ndarray, delta: int) -> np.ndarray:
    """亮度整体平移（含 alpha 通道安全：只动 RGB）。"""
    out = arr.copy()
    rgb = out[:, :, :3].astype(np.int16) + delta
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def _icon_from(arr: np.ndarray, size: int):
    from PySide6.QtGui import QIcon
    pm = _bgra_to_pixmap(arr).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pm)
