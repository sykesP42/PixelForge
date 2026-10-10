"""动画播放工作区：裁切帧的循环预览，带基本播放设置。

播放设置：播放/暂停/停止、FPS（1-60，播放中可调）、方向（正放/倒放/往返）、
循环开关、缩放（适配/1x/2x/4x）、背景（深灰/黑/白/棋盘格）、帧条逐帧浏览、
洋葱皮（前红/后蓝半透明残影叠加，前后各 N 帧）。
数据源：工具坊"发送到动画播放器"，或"打开帧目录"（按文件名排序的 PNG）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from pixelforge.processing.aseprite_io import export_aseprite, frames_by_tag
from pixelforge.processing.editing import bleed_edges, tween_x2
from pixelforge.processing.image_processor import frame_hull


def _to_pixmap(arr: np.ndarray) -> QPixmap:
    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


_BG_COLORS = {"深灰": (43, 45, 49), "黑": (16, 16, 16), "白": (238, 238, 238)}


class AnimPlayer(QWidget):
    state_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list[np.ndarray] = []
        self._clips: dict[str, list[np.ndarray]] = {}  # 多动画组（Aseprite tag）
        self._name = "anim"
        self._pos = 0
        self._ping_dir = 1        # 往返模式当前方向
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        self.canvas = QLabel("裁切帧后点“发送到动画播放器”，或打开帧目录")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(420, 420)
        self.canvas.setStyleSheet("background:#202225; color:#9aa0a6;")
        lay.addWidget(self.canvas, 1)

        bar1 = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        bar1.addWidget(self.btn_play)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop)
        bar1.addWidget(self.btn_stop)
        bar1.addWidget(QLabel("FPS"))
        self.slider_fps = QSlider(Qt.Horizontal)
        self.slider_fps.setRange(1, 60)
        self.slider_fps.setValue(8)
        self.slider_fps.setMinimumWidth(160)
        self.slider_fps.valueChanged.connect(self._on_fps)
        bar1.addWidget(self.slider_fps)
        self.lbl_fps = QLabel("8")
        self.lbl_fps.setMinimumWidth(24)
        self.slider_fps.valueChanged.connect(self.lbl_fps.setNum)
        bar1.addWidget(self.lbl_fps)
        bar1.addStretch(1)
        lay.addLayout(bar1)

        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("方向"))
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(["正放", "倒放", "往返"])
        bar2.addWidget(self.cmb_dir)
        self.chk_loop = QCheckBox("循环")
        self.chk_loop.setChecked(True)
        bar2.addWidget(self.chk_loop)
        bar2.addWidget(QLabel("缩放"))
        self.cmb_zoom = QComboBox()
        self.cmb_zoom.addItems(["适配", "1x", "2x", "4x"])
        self.cmb_zoom.currentIndexChanged.connect(lambda _: self._render())
        bar2.addWidget(self.cmb_zoom)
        bar2.addWidget(QLabel("背景"))
        self.cmb_bg = QComboBox()
        self.cmb_bg.addItems(["深灰", "黑", "白", "棋盘", "透明"])
        self.cmb_bg.setItemData(4, "预览沿用深灰底；导出 GIF 时保留透明通道",
                                Qt.ToolTipRole)
        self.cmb_bg.currentIndexChanged.connect(lambda _: self._render())
        bar2.addWidget(self.cmb_bg)
        self.chk_bleed = QCheckBox("边缘外扩")
        self.chk_bleed.setChecked(True)
        self.chk_bleed.setToolTip("导出前向透明区外渗 RGB（防引擎图集采样串色）；\n"
                                  "勾选时 rect 覆盖整帧画布且帧间加 2px 间距")
        bar2.addWidget(self.chk_bleed)
        self.chk_tween = QCheckBox("补间 x2")
        self.chk_tween.setToolTip("导出时在相邻帧之间插入 50% 混合过渡帧（ghost frame），\n"
                                  "序列长度近似翻倍，播放更顺滑")
        bar2.addWidget(self.chk_tween)
        self.chk_onion = QCheckBox("洋葱皮")
        self.chk_onion.setToolTip(
            "叠加显示当前帧前(红)/后(蓝)若干帧的半透明残影，便于检查动作连贯与对位")
        self.chk_onion.toggled.connect(lambda _: self._render())
        bar2.addWidget(self.chk_onion)
        self.spin_onion = QSpinBox()
        self.spin_onion.setRange(1, 4)
        self.spin_onion.setValue(1)
        self.spin_onion.setToolTip("洋葱皮范围：前后各叠加 N 帧")
        self.spin_onion.setEnabled(False)
        self.spin_onion.valueChanged.connect(lambda _: self._render())
        self.chk_onion.toggled.connect(self.spin_onion.setEnabled)
        bar2.addWidget(self.spin_onion)
        self.lbl_frame = QLabel("0 / 0")
        bar2.addWidget(self.lbl_frame)
        bar2.addWidget(QLabel("动画组"))
        self.cmb_clips = QComboBox()
        self.cmb_clips.setEnabled(False)
        self.cmb_clips.setToolTip("Aseprite frameTags 导入后的动画组；切换即播放")
        self.cmb_clips.currentTextChanged.connect(self._on_clip_selected)
        bar2.addWidget(self.cmb_clips)
        btn_multi = QPushButton("导入多动画...")
        btn_multi.clicked.connect(self.load_aseprite_dialog)
        bar2.addWidget(btn_multi)
        btn_open = QPushButton("打开帧目录...")
        btn_open.clicked.connect(self.open_folder_dialog)
        bar2.addWidget(btn_open)
        btn_sess = QPushButton("保存会话...")
        btn_sess.clicked.connect(self.save_session_dialog)
        bar2.addWidget(btn_sess)
        btn_load = QPushButton("打开会话...")
        btn_load.clicked.connect(self.load_session_dialog)
        bar2.addWidget(btn_load)
        self.btn_adjust = QPushButton("⇕ 逐帧位移/缩放...")
        self.btn_adjust.setToolTip(
            "逐帧调节位移与缩放（锚点：中心/脚底中心/顶部/左上），"
            "一键应用到全部帧、一键脚底对齐，消掉帧间抖动")
        self.btn_adjust.clicked.connect(self._open_frame_adjust)
        bar2.addWidget(self.btn_adjust)
        self.btn_export = QPushButton("导出...")
        self.btn_export.setProperty("accent", True)
        menu = QMenu(self)
        menu.addAction("帧序列 + atlas.json...", lambda: self._do_export("frames"))
        menu.addAction("打包精灵表 + atlas.json...", lambda: self._do_export("sheet"))
        menu.addAction("Aseprite JSON（横向表）...", lambda: self._do_export("aseprite"))
        menu.addAction("Unity 多动画（每动画组一套 atlas）...",
                       lambda: self._do_export("unity_clips"))
        menu.addAction("Unity .unitypackage（拖入即用）...",
                       lambda: self._do_export("unity_pkg"))
        menu.addAction("GIF 动图...", lambda: self._do_export("gif"))
        menu.addAction("Godot SpriteFrames (.tres)...",
                       lambda: self._do_export("godot"))
        self.btn_export.setMenu(menu)
        bar2.addWidget(self.btn_export)
        bar2.addStretch(1)
        lay.addLayout(bar2)

        self.lst_frames = QListWidget()
        self.lst_frames.setObjectName("frameStrip")
        self.lst_frames.setIconSize(QSize(64, 64))
        self.lst_frames.setViewMode(QListWidget.IconMode)
        self.lst_frames.setMaximumHeight(110)
        self.lst_frames.currentRowChanged.connect(self._on_strip_click)
        lay.addWidget(self.lst_frames)

    # ---------------- 数据 ----------------

    def set_frames(self, frames: list, name: str = "anim",
                   keep_clips: bool = False) -> None:
        """接收帧（np.ndarray BGRA 或图片路径），统一画布后待播。"""
        arrs = []
        for f in frames:
            if isinstance(f, (str, Path)):
                a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            else:
                a = f
            if a is None:
                continue
            if a.ndim == 2:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGRA)
            elif a.shape[2] == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2BGRA)
            arrs.append(a)
        if not arrs:
            self.status("没有可播放的帧")
            return
        self._name = name
        self._frames = self._normalize(arrs)
        self._pos = 0
        self._ping_dir = 1
        self._timer.stop()
        self.btn_play.setText("▶ 播放")
        self._rebuild_strip()
        self._render()
        if not keep_clips:  # 单动画模式：clip 表随载入重置
            self._clips = {name: list(self._frames)}
        self._sync_clips_combo(current=name)
        self.status(f"已载入 {len(self._frames)} 帧（{name}），FPS {self.slider_fps.value()}")

    # ---------------- 多动画组 ----------------

    def _open_frame_adjust(self) -> None:
        """⇕ 逐帧位移/缩放：对当前动画帧统一画布调节（不重新底对齐）。"""
        from PySide6.QtWidgets import QDialog
        from pixelforge.ui.image_lab import FrameAdjustDialog
        if not self._frames:
            self.status("没有可调节的帧——先拖入 GIF/帧目录或从工具坊发送")
            return
        dlg = FrameAdjustDialog(self._frames, self)
        if dlg.exec() != QDialog.Accepted:
            return
        frames = dlg.result_frames()
        if not frames:
            return
        self._frames = [f.copy() for f in frames]   # 不走 _normalize：保留用户位移
        self._pos = 0
        self._ping_dir = 1
        self._timer.stop()
        self.btn_play.setText("▶ 播放")
        cur = self.cmb_clips.currentText()
        if cur in self._clips:
            self._clips[cur] = list(self._frames)   # 同步当前动画组
        self._rebuild_strip()
        self._render()
        h, w = frames[0].shape[:2]
        self.status(f"已应用逐帧位移/缩放：{len(frames)} 帧 → 统一画布 {w}x{h}")

    def _sync_clips_combo(self, current: str) -> None:
        self.cmb_clips.blockSignals(True)
        self.cmb_clips.clear()
        for clip in self._clips:
            self.cmb_clips.addItem(clip)
        if current in self._clips:
            self.cmb_clips.setCurrentText(current)
        self.cmb_clips.setEnabled(len(self._clips) > 1)
        self.cmb_clips.blockSignals(False)

    def load_aseprite_dialog(self) -> None:
        """导入带 frameTags 的 Aseprite JSON，每个标签成为一组动画。"""
        f, _ = QFileDialog.getOpenFileName(
            self, "导入 Aseprite 多动画", "", "Aseprite JSON (*.json)")
        if not f:
            return
        try:
            clips = frames_by_tag(f)
        except Exception as e:
            QMessageBox.warning(self, "动画播放", f"导入失败: {e}")
            return
        self._clips = {name: self._normalize(list(arrs))
                       for name, arrs in clips.items()}
        first = next(iter(self._clips))
        self.set_frames(self._clips[first], name=first, keep_clips=True)
        names = ", ".join(
            f"{n}({len(v)})" for n, v in self._clips.items())
        self.status(f"已导入 {len(self._clips)} 个动画组: {names}")

    def _on_clip_selected(self, clip: str) -> None:
        if clip and clip in self._clips and \
                self._frames is not self._clips[clip]:
            self.set_frames(self._clips[clip], name=clip, keep_clips=True)

    def load_folder(self, path: Path) -> int:
        """加载目录中的 PNG 帧（按文件名排序），返回帧数。"""
        files = sorted(Path(path).glob("*.png"))
        if not files:
            self.status(f"目录中没有 PNG 帧: {path}")
            return 0
        self.set_frames(files, name=Path(path).stem or Path(path).name)
        return len(self._frames)

    def open_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择帧目录（PNG 按文件名排序播放）")
        if not d:
            return
        if self.load_folder(Path(d)) == 0:
            QMessageBox.information(self, "动画播放", "目录中没有 PNG 帧")

    @staticmethod
    def _normalize(arrs: list[np.ndarray]) -> list[np.ndarray]:
        """统一画布：底对齐（脚底基线）+ 水平居中，播放时角色不漂移。"""
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

    # ---------------- 播放控制 ----------------

    def toggle_play(self) -> None:
        if not self._frames:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            self._timer.start(max(16, 1000 // max(1, self.slider_fps.value())))
            self.btn_play.setText("⏸ 暂停")

    def stop(self) -> None:
        self._timer.stop()
        self._pos = 0
        self._ping_dir = 1
        self.btn_play.setText("▶ 播放")
        self._render()

    def _on_fps(self, _v: int) -> None:
        if self._timer.isActive():  # 播放中改 FPS 立即生效
            self._timer.start(max(16, 1000 // max(1, self.slider_fps.value())))

    def _advance(self) -> None:
        n = len(self._frames)
        if n == 0:
            self._timer.stop()
            return
        if n == 1:
            self._pos = 0
            self._render()
            return
        loop = self.chk_loop.isChecked()
        d = self.cmb_dir.currentText()
        if d == "正放":
            self._pos += 1
            if self._pos >= n:
                if loop:
                    self._pos = 0
                else:
                    self._pos = n - 1
                    self._stop("播放结束")
                    return
        elif d == "倒放":
            self._pos -= 1
            if self._pos < 0:
                if loop:
                    self._pos = n - 1
                else:
                    self._pos = 0
                    self._stop("播放结束")
                    return
        else:  # 往返
            nxt = self._pos + self._ping_dir
            if 0 <= nxt < n:
                self._pos = nxt
            elif loop:
                self._ping_dir = -self._ping_dir
                self._pos = self._pos + self._ping_dir
            else:
                self._pos = min(max(self._pos, 0), n - 1)
                self._stop("播放结束")
                return
        self._render()

    def _stop(self, reason: str) -> None:
        self._timer.stop()
        self.btn_play.setText("▶ 播放")
        self.status(reason)

    # ---------------- 渲染 ----------------

    def _rebuild_strip(self) -> None:
        self.lst_frames.blockSignals(True)
        self.lst_frames.clear()
        for i, a in enumerate(self._frames):
            it = QListWidgetItem(str(i))
            pm = _to_pixmap(a).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            it.setIcon(QIcon(pm))
            self.lst_frames.addItem(it)
        if self._frames:
            self.lst_frames.setCurrentRow(0)
        self.lst_frames.blockSignals(False)

    def _on_strip_click(self, row: int) -> None:
        if row < 0 or not self._frames:
            return
        self._pos = row
        self._render()

    def _composite(self, a: np.ndarray) -> np.ndarray:
        """帧透明区合成到当前背景设置（导出 GIF 复用同一视觉）。

        「透明」模式原样返回 BGRA（预览用 alpha 显示，导出走透明 GIF）。
        """
        h, w = a.shape[:2]
        bg_name = self.cmb_bg.currentText()
        if bg_name == "透明":
            return a
        if bg_name == "棋盘":
            cell = 8
            bg = np.empty((h, w, 3), np.uint8)
            ys, xs = np.mgrid[0:h, 0:w]
            dark = ((ys // cell + xs // cell) % 2 == 0)
            bg[dark] = (52, 52, 56)
            bg[~dark] = (82, 82, 88)
        else:
            bg = np.full((h, w, 3), _BG_COLORS.get(bg_name, (43, 45, 49)), np.uint8)
        af = a[:, :, 3:4].astype(np.float32) / 255.0
        return (a[:, :, :3].astype(np.float32) * af
                + bg.astype(np.float32) * (1 - af)).astype(np.uint8)

    def _onion_frame(self, cur: np.ndarray) -> np.ndarray:
        """洋葱皮：把当前帧前(红)/后(蓝)各 N 帧以半透明残影垫在底层。

        返回合成后的 BGRA 帧，再交给 _composite 处理背景，保证各背景模式一致。
        """
        n = len(self._frames)
        rng = self.spin_onion.value() if hasattr(self, "spin_onion") else 1
        h, w = cur.shape[:2]
        out = np.zeros((h, w, 4), np.float32)
        # 由远及近铺设残影（远的先画，近的后画覆盖在上）
        for off in range(rng, 0, -1):
            for delta, tint in ((-off, (0, 80, 255)), (off, (255, 120, 0))):
                j = self._pos + delta
                if not (0 <= j < n):
                    continue
                fr = self._frames[j]
                if fr.shape[:2] != (h, w):
                    fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST)
                al = fr[:, :, 3:4].astype(np.float32) / 255.0 * (0.35 / off)
                rgb = (fr[:, :, :3].astype(np.float32) * 0.5
                       + np.array(tint, np.float32) * 0.5)
                a1 = al[:, :, 0]
                out[:, :, :3] = rgb * al + out[:, :, :3] * (1 - al)
                out[:, :, 3] = np.clip(a1 * 255 + out[:, :, 3] * (1 - a1),
                                       0, 255)
        # 当前帧置顶（不透明部分盖住残影）
        ca = cur[:, :, 3:4].astype(np.float32) / 255.0
        c1 = ca[:, :, 0]
        out[:, :, :3] = cur[:, :, :3].astype(np.float32) * ca + out[:, :, :3] * (1 - ca)
        out[:, :, 3] = np.clip(c1 * 255 + out[:, :, 3] * (1 - c1), 0, 255)
        return out.astype(np.uint8)

    def _render(self) -> None:
        if not self._frames:
            return
        self._pos = min(max(self._pos, 0), len(self._frames) - 1)
        a = self._frames[self._pos]
        if (hasattr(self, "chk_onion") and self.chk_onion.isChecked()
                and len(self._frames) > 1):
            a = self._onion_frame(a)
        h, w = a.shape[:2]
        pm = _to_pixmap(self._composite(a))

        zm = self.cmb_zoom.currentText()
        if zm == "适配":
            pm = pm.scaled(self.canvas.width(), self.canvas.height(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        elif zm != "1x":
            k = int(zm[0])
            pm = pm.scaled(w * k, h * k, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.canvas.setPixmap(pm)

        self.lbl_frame.setText(f"{self._pos + 1} / {len(self._frames)}")
        self.lst_frames.blockSignals(True)
        self.lst_frames.setCurrentRow(self._pos)
        self.lst_frames.blockSignals(False)

    # ---------------- 导出 ----------------

    @staticmethod
    def _feet_pivot_x(a: np.ndarray) -> float:
        """脚底带（底部 15%）不透明像素中心 / 帧宽 —— 与裁切管线锚点一致。"""
        alpha = a[:, :, 3]
        ys, _ = np.where(alpha > 0)
        if ys.size == 0:
            return 0.5
        bottom = int(ys.max())
        band_top = max(0, bottom - max(2, int((bottom + 1) * 0.15)))
        bxs = np.where(alpha[band_top:bottom + 1] > 0)[1]
        return float(bxs.mean()) / a.shape[1] if bxs.size else 0.5

    def _do_export(self, kind: str) -> None:
        if not self._frames:
            QMessageBox.information(self, "动画播放", "请先载入帧")
            return
        if kind == "gif":
            f, _ = QFileDialog.getSaveFileName(
                self, "导出 GIF", f"{self._name}.gif", "GIF (*.gif)")
            if f:
                self.export_gif(Path(f))
                self.status(f"已导出 GIF: {f}")
            return
        if kind == "aseprite":
            f, _ = QFileDialog.getSaveFileName(
                self, "导出 Aseprite", f"{self._name}", "Aseprite JSON (*.json)")
            if f:
                base = Path(f).with_suffix("")
                export_aseprite(self._export_frames(), base,
                                fps=self.slider_fps.value(), name=self._name)
                self.status(f"已导出 Aseprite: {base.name}.png / .json"
                            f"（{len(self._frames)} 帧）")
            return
        if kind == "godot":
            d = QFileDialog.getExistingDirectory(self, "选择 Godot 导出目录")
            if d:
                tp, ng = self.export_godot(Path(d))
                self.status(f"已导出 Godot SpriteFrames（{ng} 帧）到 {tp}")
            return
        d = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not d:
            return
        out_dir = Path(d)
        if kind == "frames":
            n, _ = self.export_frames(out_dir)
            self.status(f"已导出 {n} 帧 + atlas.json 到 {out_dir}")
        elif kind == "unity_pkg":
            f, _ = QFileDialog.getSaveFileName(
                self, "导出 .unitypackage", f"{self._name}.unitypackage",
                "Unity Package (*.unitypackage)")
            if f:
                entries = self._unity_package_entries()
                from pixelforge.unity_package import build_unity_package
                out = build_unity_package(entries, Path(f))
                self.status(f"已导出 {out.name}（{len(entries)} 个资产），"
                            "拖入 Unity 项目窗口即自动导入")
            return
        elif kind == "unity_clips":
            atlases = self.export_unity_clips(out_dir)
            clips = self._clips or {self._name: self._frames}
            total = sum(len(v) for v in clips.values())
            self.status(f"已导出 {len(atlases)} 个动画组（{total} 帧）到 {out_dir}")
        else:
            sp, n = self.export_sheet(out_dir)
            self.status(f"已打包精灵表（{n} 帧）到 {sp}")

    def _export_frames(self) -> list[np.ndarray]:
        """按 bleed / 补间选项返回导出用帧（不改动播放数据本身）。"""
        frames = list(self._frames)
        if self.chk_tween.isChecked():
            frames = tween_x2(frames)
        if self.chk_bleed.isChecked():
            frames = [bleed_edges(a, 1) for a in frames]
        return frames

    def _meta_entry(self, a: np.ndarray, bleed_on: bool, fn: str) -> dict:
        """单帧 atlas 条目（rect/pivot/hull），供帧序列与多动画导出共用。"""
        ys, xs = np.where(a[:, :, 3] > 0)
        fx = self._feet_pivot_x(a)
        if ys.size and not bleed_on:
            x0, y0 = int(xs.min()), int(ys.min())
            rect = {"x": x0, "y": y0, "width": int(xs.max() - x0 + 1),
                    "height": int(ys.max() - y0 + 1)}
        else:  # bleed 时 rect 覆盖整帧（把外渗色包进采样范围）
            rect = {"x": 0, "y": 0, "width": a.shape[1], "height": a.shape[0]}
        return rect | {"file": fn,
                       "pivotX": round(fx, 4), "pivotY": 1.0,
                       "hull": frame_hull(a)}

    def export_frames(self, out_dir: Path) -> tuple[int, dict]:
        """帧 PNG + atlas.json（含 fps/pivot/hull），返回 (帧数, atlas)。

        补间勾选时导出补间后序列（帧数近似翻倍）。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bleed_on = self.chk_bleed.isChecked()
        frames_meta = []
        for i, a in enumerate(self._export_frames()):
            fn = f"{self._name}_{i:03d}.png"
            cv2.imwrite(str(out_dir / fn), a)
            frames_meta.append(self._meta_entry(a, bleed_on, fn))
        atlas = {"name": self._name, "fps": self.slider_fps.value(),
                 "frame_count": len(frames_meta), "frames": frames_meta}
        (out_dir / f"{self._name}.atlas.json").write_text(
            json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        return len(frames_meta), atlas

    def export_sheet(self, out_dir: Path) -> tuple[Path, int]:
        """打包精灵表 PNG + atlas.json；bleed 时帧间留 2px 间距防串色。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(self._frames)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        h, w = self._frames[0].shape[:2]
        pad = 2 if self.chk_bleed.isChecked() else 0
        cell_w, cell_h = w + 2 * pad, h + 2 * pad
        sheet = np.zeros((rows * cell_h, cols * cell_w, 4), np.uint8)
        frames_meta = []
        for i, src in enumerate(self._frames):
            a = bleed_edges(src, 1) if self.chk_bleed.isChecked() else src
            r, c = divmod(i, cols)
            x = c * cell_w + pad
            y = r * cell_h + pad
            sheet[y:y + h, x:x + w] = a
            fx = self._feet_pivot_x(a)
            if self.chk_bleed.isChecked():
                rect = {"x": x, "y": y, "width": w, "height": h}
            else:
                ys, xs = np.where(a[:, :, 3] > 0)
                if ys.size:
                    x0, y0 = int(xs.min()), int(ys.min())
                    rect = {"x": x + x0, "y": y + y0,
                            "width": int(xs.max() - x0 + 1),
                            "height": int(ys.max() - y0 + 1)}
                else:
                    rect = {"x": x, "y": y, "width": w, "height": h}
            frames_meta.append(rect | {"file": f"{self._name}.png", "index": i,
                                       "pivotX": round(fx, 4), "pivotY": 1.0,
                                       "hull": frame_hull(a)})
        atlas = {"name": self._name, "fps": self.slider_fps.value(),
                 "frame_count": n, "sheet": f"{self._name}.png",
                 "frames": frames_meta}
        sp = out_dir / f"{self._name}.png"
        cv2.imwrite(str(sp), sheet)
        (out_dir / f"{self._name}.atlas.json").write_text(
            json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        return sp, n

    def export_gif(self, path: Path) -> Path:
        """导出 GIF 动图；背景选「透明」时保留 alpha 通道。"""
        if self.cmb_bg.currentText() == "透明":
            return self._export_gif_transparent(path)
        from PIL import Image
        frames_rgb = [Image.fromarray(cv2.cvtColor(self._composite(a), cv2.COLOR_BGR2RGB))
                      for a in self._frames]
        duration = max(20, int(1000 / max(1, self.slider_fps.value())))
        frames_rgb[0].save(
            path, save_all=True, append_images=frames_rgb[1:],
            duration=duration, loop=0 if self.chk_loop.isChecked() else 1,
            disposal=2)
        return path

    def _export_gif_transparent(self, path: Path) -> Path:
        """透明 GIF：量化到 255 色，透明区写入保留索引 255。"""
        from PIL import Image
        ims = []
        for a in self._frames:
            rgba = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            im = Image.fromarray(rgba, "RGBA")
            alpha = im.getchannel("A")
            p = im.convert("RGB").quantize(colors=255)
            mask = alpha.point(lambda v: 255 if v < 128 else 0)  # 透明 -> 索引 255
            p.paste(255, mask)
            ims.append(p)
        duration = max(20, int(1000 / max(1, self.slider_fps.value())))
        ims[0].save(
            path, save_all=True, append_images=ims[1:], duration=duration,
            loop=0 if self.chk_loop.isChecked() else 1,
            transparency=255, disposal=2)
        return path

    def export_godot(self, out_dir: Path) -> tuple[Path, int]:
        """导出 Godot 4 SpriteFrames 资源（.tres）+ 帧序列 PNG。

        多动画组（Aseprite frameTags）时每组生成一个 animation。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = self._clips or {self._name: self._frames}
        fps = max(1, self.slider_fps.value())
        loop = "true" if self.chk_loop.isChecked() else "false"
        ext_lines: list[str] = []
        animations: list[str] = []
        fid = 0
        total = 0
        for clip_name, arrs in clips.items():
            if not arrs:
                continue
            keyframes = []
            for a in arrs:
                fn = f"{clip_name}_{total:03d}.png"  # 全局序号保证文件名唯一
                cv2.imwrite(str(out_dir / fn), a)
                fid += 1
                ext_lines.append(
                    f'[ext_resource type="Texture2D" '
                    f'path="res://{out_dir.name}/{fn}" id="{fid}"]')
                keyframes.append(
                    '{"duration": 1.0, "texture": ExtResource("' + str(fid) + '")}')
                total += 1
            animations.append(
                '{\n"frames": [' + ", ".join(keyframes) + '],\n'
                f'"loop": {loop},\n"name": &"{clip_name}",\n"speed": {fps}\n}}')
        tres = (
            f'[gd_resource type="SpriteFrames" load_steps={fid + 1} format=3]\n\n'
            + "\n".join(ext_lines)
            + "\n\n[resource]\nanimations = [" + ", ".join(animations) + "]\n")
        tp = out_dir / f"{self._name}.tres"
        tp.write_text(tres, "utf-8")
        return tp, total

    def export_unity_clips(self, out_dir: Path) -> list[Path]:
        """多动画导出：每个动画组一套 帧序列 PNG + <clip>.atlas.json。

        Unity 导入器监听 *.atlas.json，每组自动生成独立的
        AnimationClip + AnimatorController。返回生成的 atlas 文件列表。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = self._clips or {self._name: self._frames}
        bleed_on = self.chk_bleed.isChecked()
        produced: list[Path] = []
        for clip_name, arrs in clips.items():
            if not arrs:
                continue
            srcs = tween_x2(list(arrs)) if self.chk_tween.isChecked() \
                else list(arrs)
            frames_meta = []
            for i, src in enumerate(srcs):
                a = bleed_edges(src, 1) if bleed_on else src
                fn = f"{clip_name}_{i:03d}.png"
                cv2.imwrite(str(out_dir / fn), a)
                frames_meta.append(self._meta_entry(a, bleed_on, fn))
            atlas = {"name": clip_name, "fps": self.slider_fps.value(),
                     "frame_count": len(frames_meta), "frames": frames_meta}
            jp = out_dir / f"{clip_name}.atlas.json"
            jp.write_text(json.dumps(atlas, ensure_ascii=False, indent=2),
                          "utf-8")
            produced.append(jp)
        return produced

    def _unity_package_entries(self) -> list[tuple[Path, str]]:
        """生成临时导出产物并映射为 (本地文件, Unity 项目内路径)。

        atlas JSON 放 Assets/Textures/Generated/（与导入器监听目录一致），
        帧贴图平铺同目录（atlas 的 file 字段即文件名，导入器按 dir/file 寻址）。
        """
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="pf_upkg_"))
        atlases = self.export_unity_clips(tmp)
        entries: list[tuple[Path, str]] = []
        for jp in atlases:
            meta = json.loads(jp.read_text("utf-8"))
            entries.append((jp, f"Assets/Textures/Generated/{jp.name}"))
            for fr in meta["frames"]:
                entries.append((tmp / fr["file"],
                                f"Assets/Textures/Generated/{fr['file']}"))
        return entries

    def save_session(self, out_dir: Path) -> Path:
        """帧 PNG + session.json（fps/方向/循环/背景/缩放）。"""
        out_dir = Path(out_dir)
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(self._frames):
            cv2.imwrite(str(frames_dir / f"{self._name}_{i:03d}.png"), a)
        meta = {"name": self._name, "fps": self.slider_fps.value(),
                "loop": self.chk_loop.isChecked(),
                "direction": self.cmb_dir.currentText(),
                "bg": self.cmb_bg.currentText(),
                "zoom": self.cmb_zoom.currentText(),
                "bleed": self.chk_bleed.isChecked(),
                "frame_count": len(self._frames)}
        sp = out_dir / "session.json"
        sp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        return sp

    def load_session(self, session_dir: Path) -> bool:
        sp = Path(session_dir) / "session.json"
        if not sp.is_file():
            return False
        meta = json.loads(sp.read_text("utf-8"))
        files = sorted((Path(session_dir) / "frames").glob("*.png"))
        self.set_frames(files, name=meta.get("name", "anim"))
        self.slider_fps.setValue(int(meta.get("fps", 8)))
        self.chk_loop.setChecked(bool(meta.get("loop", True)))
        idx = self.cmb_dir.findText(meta.get("direction", "正放"))
        if idx >= 0:
            self.cmb_dir.setCurrentIndex(idx)
        idx = self.cmb_bg.findText(meta.get("bg", "深灰"))
        if idx >= 0:
            self.cmb_bg.setCurrentIndex(idx)
        idx = self.cmb_zoom.findText(meta.get("zoom", "适配"))
        if idx >= 0:
            self.cmb_zoom.setCurrentIndex(idx)
        self.chk_bleed.setChecked(bool(meta.get("bleed", True)))
        return True

    def save_session_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择会话保存目录")
        if d:
            sp = self.save_session(Path(d))
            self.status(f"会话已保存: {sp}")

    def load_session_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择会话目录（含 session.json）")
        if d and not self.load_session(Path(d)):
            QMessageBox.information(self, "动画播放",
                                    "该目录不是有效会话（缺少 session.json）")

    def status(self, msg: str) -> None:
        self.state_changed.emit(msg)
