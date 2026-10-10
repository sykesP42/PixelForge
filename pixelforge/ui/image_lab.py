"""图像工具坊（独立工作区）：打开任意图片即可去背景/裁切/编辑，不依赖生成流程。

能力：
- 打开/拖拽图片，撤销（20步），重置，保存/另存为
- 一键去背景：四角自动取色 + 容差滑块实时预览（防抖 120ms），应用/取消两段式提交
- 画笔修边：擦除残留 / 恢复误删（笔刷大小可调，映射缩放坐标），按住查看原图
- A/B 对比：弹出对比窗口，左=原图(A) 右=当前(B)，黄色分割线可拖动比对
- 智能单帧：整体内容紧致化输出 1 帧（单角色图不再被网格切碎）
- 自动裁切双模式：连通域 / 网格切分（行 x 列）
- 自由裁切：画布叠加裁切框，拖四边微调 / 框内整体移动 / 框外拖动新建框
- 放大镜：鼠标位置实时放大预览（可开关），配合裁切线/画笔精确操作
- 常用编辑：亮度/对比度/饱和度（带数值显示）、旋转90°、水平/垂直翻转、手动裁边
- 旋转任意角度（最近邻保持像素感）、边缘外扩 bleed（防图集串色）
- 投影拆分：行列占用带分析，支持带背景的外部精灵表
- 换色引擎：色板条选源色 -> 拾取目标色 -> 一键替换
- 动作宏：录制操作步骤 -> 回放到帧 / 保存 / 加载 / 批量回放文件夹
- GIF/APNG 逐帧导入；Aseprite JSON 导入
- 裁切结果帧列表，可保存帧 PNG + atlas.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QEvent, QPoint, QRect, QSize, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import (
    QColor, QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap, QRegion,
)
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from pixelforge.processing import editing
from pixelforge.processing.aseprite_io import import_aseprite
from pixelforge.processing.image_processor import (
    ImageProcessor,
    chroma_key,
    detect_background_color,
    remap_palette_colors,
)
from pixelforge.processing.macro import (
    apply_macro,
    load_macro,
    replay_folder,
    save_macro,
)


def _to_pixmap(arr: np.ndarray) -> QPixmap:
    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


def _norm_bgra(arr: np.ndarray) -> np.ndarray:
    """统一为 BGRA（灰度/三通道输入自动升维）。"""
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return arr


class _ABCompare(QWidget):
    """A/B 对比画布：左 = 基准 A（原图），右 = 当前 B，分割线可拖动。

    双击画面复位到 50%；鼠标移动时光标提示可拖动。
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, parent=None):
        super().__init__(parent)
        self._pa = _to_pixmap(_norm_bgra(a))
        self._pb = _to_pixmap(_norm_bgra(b))
        self._ratio = 0.5
        self._drag = False
        self.setMinimumSize(600, 480)
        self.setMouseTracking(True)
        self.setCursor(Qt.SplitHCursor)

    # 显示区几何：等比缩放居中 + 底部留出标签行
    def _geo(self) -> tuple[int, int, int, int, QSize]:
        r = self.rect().adjusted(10, 10, -10, -30)
        pm = self._pb.scaled(r.size(), Qt.KeepAspectRatio,
                             Qt.FastTransformation)
        x = r.x() + (r.width() - pm.width()) // 2
        y = r.y() + (r.height() - pm.height()) // 2
        return x, y, pm.width(), pm.height(), pm.size()

    def paintEvent(self, _e) -> None:  # noqa: N802 Qt 约定
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 34, 37))
        x, y, w, h, size = self._geo()
        if w <= 0 or h <= 0:
            p.end()
            return
        pa = self._pa.scaled(size, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        pb = self._pb.scaled(size, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        split = x + int(w * self._ratio)
        p.save()
        p.setClipRect(QRect(x, y, split - x, h))
        p.drawPixmap(x, y, pa)
        p.restore()
        p.save()
        p.setClipRect(QRect(split, y, x + w - split, h))
        p.drawPixmap(x, y, pb)
        p.restore()
        # 分割线 + 手柄
        p.setPen(QPen(QColor(255, 208, 0), 2))
        p.drawLine(split, y, split, y + h)
        p.setBrush(QColor(255, 208, 0))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPoint(split, y + h // 2), 7, 7)
        # 角标
        p.setPen(QColor(220, 224, 230))
        p.drawText(QRect(x, y + h + 4, w // 2, 22), Qt.AlignLeft,
                   "A 基准（原图）")
        p.drawText(QRect(x + w // 2, y + h + 4, w // 2, 22), Qt.AlignRight,
                   "B 当前")
        p.end()

    def _set_ratio(self, px: float) -> None:
        _x, _y, w, _h, _s = self._geo()
        if w <= 0:
            return
        self._ratio = min(1.0, max(0.0, (px - _x) / w))
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._drag = True
        self._set_ratio(e.position().x())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self.setCursor(Qt.SizeHorCursor if self._drag else Qt.SplitHCursor)
        if self._drag:
            self._set_ratio(e.position().x())

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._drag = False

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._ratio = 0.5
        self.update()


class _MacroWorker(QThread):
    """后台线程执行宏批量回放，避免阻塞 UI。"""
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, macro: str, in_dir: str, out_dir: str, parent=None):
        super().__init__(parent)
        self._macro, self._in_dir, self._out_dir = macro, in_dir, out_dir

    def run(self) -> None:
        try:
            produced = replay_folder(self._macro, self._in_dir, self._out_dir)
            self.done.emit(len(produced))
        except Exception as e:  # 后台线程兜底，错误回传 UI
            self.failed.emit(str(e))


class ImageLab(QWidget):
    state_changed = Signal(str)
    frames_to_player = Signal(list)  # 发送裁切帧到动画播放器
    frame_back = Signal(object)      # 编辑完成回填审核区帧

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.arr: np.ndarray | None = None
        self._orig: np.ndarray | None = None      # 打开时的原图（恢复/对比用）
        self.path: Path | None = None
        self._undo: list[np.ndarray] = []
        self._zoom = 1.0
        self._frames: list[np.ndarray] = []
        self._preview: np.ndarray | None = None   # 色键实时预览（未提交）
        self._brush_mode = "browse"               # browse | erase | restore | crop
        self._locked_palette: list | None = None  # 锁定调色板
        self._brush_size = 12
        self._painting = False
        self._crop_rect: list[int] | None = None  # 自由裁切框 [x0, y0, x1, y1]
        self._drag: str | None = None             # left/right/top/bottom/move/new
        self._drag_anchor: tuple[int, int] = (0, 0)
        self._drag_orig: list[int] | None = None
        self._edit_row: int | None = None         # 回填目标帧行（None=非回填模式）
        self._map: dict[int, tuple[int, int, int]] = {}   # 换色映射 {源索引: (B,G,R)}
        self._macro_steps: list[dict] = []                # 录制的宏步骤
        self._macro_worker = None
        # 定时器必须先于 _build_ui() 创建（_build_ui 会引用它）
        self._tol_timer = QTimer(self)            # 容差滑块防抖
        self._tol_timer.setSingleShot(True)
        self._tol_timer.setInterval(120)
        self._tol_timer.timeout.connect(self._update_preview)
        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        lay = QHBoxLayout(self)

        # ---- 左侧工具面板（可滚动，滑块带数值显示，不再被截断） ----
        tools = QVBoxLayout()
        g_file = QGroupBox("文件")
        fl = QVBoxLayout(g_file)
        for text, fn in (("打开图片...", self.open_file),
                         ("导入 Aseprite JSON...", self.import_aseprite_file),
                         ("保存", self.save),
                         ("另存为...", self.save_as),
                         ("撤销 (Ctrl+Z)", self.undo),
                         ("重置为原图", self.reset)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            fl.addWidget(b)
        self.b_back = QPushButton("↩ 回填到审核区帧")
        self.b_back.setProperty("accent", True)
        self.b_back.setVisible(False)
        self.b_back.clicked.connect(self._back_to_review)
        fl.addWidget(self.b_back)
        tools.addWidget(g_file)

        g_bg = QGroupBox("一键去背景（实时预览）")
        bl = QVBoxLayout(g_bg)
        bl.addWidget(QLabel("色键容差（拖动即时预览）"))
        self.slider_tol = QSlider(Qt.Horizontal)
        self.slider_tol.setRange(10, 150)
        self.slider_tol.setValue(60)
        # lambda 惰性解引用，防抖由 _tol_timer 处理
        self.slider_tol.valueChanged.connect(lambda _: self._tol_timer.start())
        bl.addLayout(self._with_value(self.slider_tol))
        self.chk_rembg = QCheckBox("无纯色背景时用 rembg（需安装）")
        bl.addWidget(self.chk_rembg)
        row_bg = QHBoxLayout()
        b_apply = QPushButton("应用去背景")
        b_apply.clicked.connect(self.apply_key)
        b_cancel = QPushButton("取消预览")
        b_cancel.clicked.connect(self.cancel_preview)
        row_bg.addWidget(b_apply)
        row_bg.addWidget(b_cancel)
        bl.addLayout(row_bg)
        tools.addWidget(g_bg)

        g_brush = QGroupBox("操作模式")
        br = QVBoxLayout(g_brush)
        self.cmb_brush = QComboBox()
        self.cmb_brush.addItems(["浏览（不涂抹）", "擦除残留", "恢复误删",
                                 "裁切框（自由裁切）"])
        self.cmb_brush.currentIndexChanged.connect(self._set_brush_mode)
        br.addWidget(self.cmb_brush)
        br.addWidget(QLabel("笔刷大小"))
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(2, 64)
        self.slider_brush.setValue(12)
        self.slider_brush.valueChanged.connect(
            lambda v: setattr(self, "_brush_size", v))
        br.addLayout(self._with_value(self.slider_brush))
        b_cmp = QPushButton("按住查看原图")
        b_cmp.pressed.connect(lambda: self._render(self._orig))
        b_cmp.released.connect(lambda: self._render())
        br.addWidget(b_cmp)
        b_ab = QPushButton("⇋ A/B 对比")
        b_ab.setToolTip("弹出对比窗口：左=打开时的原图(A)，右=当前结果(B)，拖动中缝比对")
        b_ab.clicked.connect(self._open_ab_compare)
        br.addWidget(b_ab)
        tools.addWidget(g_brush)

        g_crop = QGroupBox("自动裁切")
        cl = QVBoxLayout(g_crop)
        b_single = QPushButton("智能单帧（整体紧致化）")
        b_single.clicked.connect(self.smart_single_frame)
        cl.addWidget(b_single)
        b_proj = QPushButton("投影拆分（外部精灵表）")
        b_proj.clicked.connect(self.proj_split)
        cl.addWidget(b_proj)
        row_pal = QHBoxLayout()
        b_pl = QPushButton("锁定调色板...")
        b_pl.clicked.connect(self._load_palette_dialog)
        row_pal.addWidget(b_pl)
        b_ps = QPushButton("保存当前调色板")
        b_ps.clicked.connect(self._save_palette_dialog)
        row_pal.addWidget(b_ps)
        cl.addLayout(row_pal)
        self.lbl_palette = QLabel("调色板：未锁定")
        cl.addWidget(self.lbl_palette)
        b1 = QPushButton("连通域裁切（自动识别帧）")
        b1.clicked.connect(self.auto_crop)
        cl.addWidget(b1)
        cl.addWidget(QLabel("对齐方式（解决中心漂移）"))
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["脚底中心（推荐）", "质心", "仅基线（旧行为）"])
        cl.addWidget(self.cmb_align)
        row_opt = QHBoxLayout()
        self.chk_harden = QCheckBox("边缘硬化")
        self.chk_harden.setToolTip("Alpha 二值化 + 边界像素吸附调色板色（去光晕）")
        self.chk_dither = QCheckBox("F-S 抖动")
        self.chk_dither.setToolTip("Floyd–Steinberg 抖动量化（复古风）")
        row_opt.addWidget(self.chk_harden)
        row_opt.addWidget(self.chk_dither)
        cl.addLayout(row_opt)
        row_bleed = QHBoxLayout()
        row_bleed.addWidget(QLabel("边缘外扩(px)"))
        self.spin_bleed = QSpinBox()
        self.spin_bleed.setRange(0, 4)
        self.spin_bleed.setValue(0)
        self.spin_bleed.setToolTip("向透明区外渗 RGB，防图集采样串色")
        row_bleed.addWidget(self.spin_bleed)
        row_bleed.addStretch(1)
        cl.addLayout(row_bleed)
        row_px = QHBoxLayout()
        row_px.addWidget(QLabel("像素密度(px)"))
        self.spin_pixel = QSpinBox()
        self.spin_pixel.setRange(0, 256)
        self.spin_pixel.setValue(0)
        self.spin_pixel.setSpecialValueText("关闭")
        self.spin_pixel.setToolTip("目标最长边像素数：box 下采样后最近邻放大，\n"
                                   "产出颜色块恒定的真像素画（0=关闭）")
        row_px.addWidget(self.spin_pixel)
        row_px.addStretch(1)
        cl.addLayout(row_px)
        cl.addWidget(QLabel("网格切分（帧间距均匀时用）"))
        grid = QHBoxLayout()
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(1, 16); self.spin_rows.setValue(2)
        self.spin_cols = QSpinBox(); self.spin_cols.setRange(1, 16); self.spin_cols.setValue(2)
        grid.addWidget(self.spin_rows)
        grid.addWidget(QLabel("x"))
        grid.addWidget(self.spin_cols)
        b2 = QPushButton("切分")
        b2.clicked.connect(self.grid_crop)
        grid.addWidget(b2)
        cl.addLayout(grid)
        tools.addWidget(g_crop)

        g_fc = QGroupBox("自由裁切")
        fc = QVBoxLayout(g_fc)
        fc.addWidget(QLabel("先选“裁切框”模式：拖黄线四边微调，\n框内拖动整体移动，框外拖动新建框"))
        row_fc = QHBoxLayout()
        b_fa = QPushButton("应用裁切")
        b_fa.clicked.connect(self.apply_crop)
        b_fr = QPushButton("重置裁切框")
        b_fr.clicked.connect(self.reset_crop)
        row_fc.addWidget(b_fa)
        row_fc.addWidget(b_fr)
        fc.addLayout(row_fc)
        tools.addWidget(g_fc)

        g_edit = QGroupBox("编辑")
        el = QVBoxLayout(g_edit)
        row = QHBoxLayout()
        b_rot = QPushButton("旋转90°")
        b_rot.clicked.connect(self._rot90)
        b_fh = QPushButton("水平翻转")
        b_fh.clicked.connect(self._flip_h)
        b_fv = QPushButton("垂直翻转")
        b_fv.clicked.connect(self._flip_v)
        row.addWidget(b_rot)
        row.addWidget(b_fh)
        row.addWidget(b_fv)
        el.addLayout(row)
        b3 = QPushButton("手动裁边...")
        b3.clicked.connect(self._trim_dialog)
        el.addWidget(b3)
        row_rot2 = QHBoxLayout()
        row_rot2.addWidget(QLabel("角度"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.spin_angle.setValue(45)
        row_rot2.addWidget(self.spin_angle)
        b_any = QPushButton("旋转任意角")
        b_any.clicked.connect(self.rotate_any)
        row_rot2.addWidget(b_any)
        el.addLayout(row_rot2)
        b_bleed = QPushButton("边缘外扩 (bleed)")
        b_bleed.clicked.connect(self.bleed_now)
        el.addWidget(b_bleed)
        tools.addWidget(g_edit)

        g_adj = QGroupBox("色彩调整")
        al = QFormLayout(g_adj)
        self.sl_b = self._slider(-100, 100, 0)
        self.sl_c = self._slider(-50, 50, 0)   # 显示值，实际 = 1 + v/100
        self.sl_s = self._slider(0, 200, 100)  # 实际 = v/100
        al.addRow("亮度", self._with_value(self.sl_b))
        al.addRow("对比度", self._with_value(self.sl_c))
        al.addRow("饱和度", self._with_value(self.sl_s))
        ba = QPushButton("应用调整")
        ba.clicked.connect(self.apply_adjust)
        al.addRow(ba)
        tools.addWidget(g_adj)

        g_remap = QGroupBox("换色引擎（Palette Swap）")
        rl = QVBoxLayout(g_remap)
        self.lst_swatch = QListWidget()
        self.lst_swatch.setViewMode(QListWidget.IconMode)
        self.lst_swatch.setFlow(QListWidget.LeftToRight)
        self.lst_swatch.setFixedHeight(56)
        self.lst_swatch.setToolTip("自动裁切生成的调色板色板：选中一项，再拾取目标色")
        rl.addWidget(self.lst_swatch)
        row_rm = QHBoxLayout()
        b_pick = QPushButton("目标色...")
        b_pick.clicked.connect(self._pick_target)
        row_rm.addWidget(b_pick)
        b_clr = QPushButton("清除映射")
        b_clr.clicked.connect(self._clear_map)
        row_rm.addWidget(b_clr)
        rl.addLayout(row_rm)
        self.lst_map = QListWidget()
        self.lst_map.setMaximumHeight(96)
        rl.addWidget(self.lst_map)
        b_remap = QPushButton("应用换色")
        b_remap.clicked.connect(self.apply_remap)
        rl.addWidget(b_remap)
        tools.addWidget(g_remap)

        g_macro = QGroupBox("动作宏")
        ml = QVBoxLayout(g_macro)
        self.chk_record = QCheckBox("录制操作（去背景/裁切/编辑…）")
        self.chk_record.toggled.connect(self._toggle_record)
        ml.addWidget(self.chk_record)
        row_m1 = QHBoxLayout()
        b_rp = QPushButton("回放到帧")
        b_rp.clicked.connect(self.macro_replay)
        row_m1.addWidget(b_rp)
        b_ms = QPushButton("保存宏...")
        b_ms.clicked.connect(self.macro_save)
        row_m1.addWidget(b_ms)
        ml.addLayout(row_m1)
        row_m2 = QHBoxLayout()
        b_ml = QPushButton("加载宏并回放...")
        b_ml.clicked.connect(self.macro_load_replay)
        row_m2.addWidget(b_ml)
        b_mb = QPushButton("批量回放文件夹...")
        b_mb.clicked.connect(self.macro_batch)
        row_m2.addWidget(b_mb)
        ml.addLayout(row_m2)
        tools.addWidget(g_macro)
        tools.addStretch(1)

        left = QWidget()
        left.setLayout(tools)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setFixedWidth(310)
        lay.addWidget(left_scroll)

        # ---- 中间画布 + 缩放 + 帧面板 ----
        canvas_col = QVBoxLayout()
        zbar = QHBoxLayout()
        for text, z in (("25%", 0.25), ("50%", 0.5), ("100%", 1.0), ("200%", 2.0), ("400%", 4.0)):
            bz = QPushButton(text)
            bz.clicked.connect(lambda _, v=z: self._set_zoom(v))
            zbar.addWidget(bz)
        zbar.addStretch(1)
        canvas_col.addLayout(zbar)
        self.canvas = QLabel("打开或拖入一张图片开始（PNG/JPG）")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(420, 420)
        self.canvas.setStyleSheet("background:#202225; color:#9aa0a6;")
        self.canvas.setMouseTracking(True)          # 无需按下也有 move 事件
        self.canvas.installEventFilter(self)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.canvas)
        canvas_col.addWidget(sc, 1)
        canvas_col.addWidget(self._build_frame_panel())

        # ---- 放大镜列 ----
        mag_col = QVBoxLayout()
        mag_col.addWidget(QLabel("放大镜"))
        self.chk_mag = QCheckBox("启用")
        self.chk_mag.setChecked(True)
        mag_col.addWidget(self.chk_mag)
        self.mag = QLabel("鼠标移到画布\n查看放大区域")
        self.mag.setFixedSize(160, 160)
        self.mag.setAlignment(Qt.AlignCenter)
        self.mag.setStyleSheet("background:#111111; color:#888888; border:1px solid #444444;")
        mag_col.addWidget(self.mag)
        mag_col.addStretch(1)

        mid = QHBoxLayout()
        mid.addLayout(canvas_col, 1)
        mid.addLayout(mag_col)
        lay.addLayout(mid, 1)

    @staticmethod
    def _slider(lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    @staticmethod
    def _with_value(slider: QSlider) -> QHBoxLayout:
        """滑块 + 实时数值显示（解决参数被面板截断看不清的问题）。"""
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        lb = QLabel(str(slider.value()))
        lb.setMinimumWidth(36)
        lb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lb.setNum)
        h.addWidget(slider, 1)
        h.addWidget(lb)
        return h

    def _build_frame_panel(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        self.lst_frames = QListWidget()
        self.lst_frames.setObjectName("frameStrip")
        self.lst_frames.setIconSize(QSize(72, 72))
        self.lst_frames.setViewMode(QListWidget.IconMode)
        self.lst_frames.setMaximumHeight(120)
        h.addWidget(self.lst_frames, 1)
        vb = QVBoxLayout()
        bp = QPushButton("发送到动画播放器")
        bp.clicked.connect(self.send_to_player)
        vb.addWidget(bp)
        bf = QPushButton("保存全部帧...")
        bf.clicked.connect(self.save_frames)
        vb.addWidget(bf)
        vb.addStretch(1)
        h.addLayout(vb)
        return w

    # ---------------- 文件 ----------------

    def open_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.apng)")
        if f:
            self.load_path(Path(f))

    def load_path(self, path: Path) -> None:
        if path.suffix.lower() in (".gif", ".apng"):  # 动图逐帧导入
            try:
                frames = editing.load_gif_frames(path)
            except Exception as e:
                QMessageBox.warning(self, "图像工具坊", f"GIF/APNG 读取失败: {e}")
                return
            if not frames:
                QMessageBox.warning(self, "图像工具坊", "动图中没有帧")
                return
            self._undo.clear()
            self.arr = frames[0].copy()
            self._orig = self.arr.copy()
            self._preview = None
            self._crop_rect = None
            self._drag = None
            self.path = path
            self._set_frames(frames)
            self._set_zoom(1.0)
            self.status(f"已打开动图 {path.name}（{len(frames)} 帧，可发送到播放器）")
            return
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            QMessageBox.warning(self, "图像工具坊", f"无法读取: {path}")
            return
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        self._undo.clear()
        self.arr = im
        self._orig = im.copy()
        self._preview = None
        self._crop_rect = None
        self._drag = None
        self.path = path
        self._frames = []
        self.lst_frames.clear()
        self._set_zoom(1.0)
        self.status(f"已打开 {path.name} ({im.shape[1]}x{im.shape[0]})")

    def load_arr(self, arr: np.ndarray, name: str = "frame") -> None:
        """直接载入数组（审核区帧送编辑场景），不依赖磁盘文件。"""
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        self._undo.clear()
        self.arr = arr.copy()
        self._orig = self.arr.copy()
        self._preview = None
        self._crop_rect = None
        self._drag = None
        self.path = None
        self._frames = []
        self.lst_frames.clear()
        self._set_zoom(1.0)
        self.status(f"已载入 {name}（编辑后点「回填」按钮返回审核区）")

    def set_edit_target(self, row: int | None) -> None:
        """设置回填目标行；None 退出回填模式。"""
        self._edit_row = row
        self.b_back.setVisible(row is not None)
        if row is not None:
            self.b_back.setText(f"↩ 回填到审核区帧 #{row}")

    def _back_to_review(self) -> None:
        if self.arr is not None:
            self.frame_back.emit(self.arr.copy())

    def save(self) -> None:
        if self.arr is None:
            return
        if self.path is None:
            self.save_as()
            return
        cv2.imwrite(str(self.path), self.arr)
        self.status(f"已保存 {self.path.name}")

    def save_as(self) -> None:
        if self.arr is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "另存为", "untitled.png", "PNG (*.png)")
        if f:
            cv2.imwrite(f, self.arr)
            self.path = Path(f)
            self.status(f"已另存为 {f}")

    # ---------------- 撤销 ----------------

    def _push_undo(self) -> None:
        if self.arr is not None:
            self._undo.append(self.arr.copy())
            if len(self._undo) > 20:
                self._undo.pop(0)

    def undo(self) -> None:
        if self._undo:
            self.arr = self._undo.pop()
            self._preview = None
            self._crop_rect = None
            self._render()
            self.status("已撤销")

    def reset(self) -> None:
        if self._orig is None:
            return
        self._push_undo()
        self.arr = self._orig.copy()
        self._preview = None
        self._crop_rect = None
        self._render()
        self.status("已重置为打开时的原图")

    # ---------------- 变换 ----------------

    def _mutate(self, fn, *args) -> None:
        if self.arr is None:
            return
        self._push_undo()
        self.arr = fn(*args)
        self._preview = None
        self._crop_rect = None
        self._render()

    # ---------------- 色键实时预览 ----------------

    def _update_preview(self) -> None:
        """容差滑块拖动 -> 防抖后即时计算色键并显示（不改动 self.arr）。"""
        if self.arr is None or self._orig is None:
            return
        bg = detect_background_color(self._orig[:, :, :3])
        if bg is None:
            self._preview = None
            self._render()
            self.status("四角颜色不一致，无纯色背景可预览（可用画笔修边或 rembg）")
            return
        self._preview = chroma_key(self._orig[:, :, :3], bg, self.slider_tol.value())
        self._render()
        self.status(f"实时预览：容差 {self.slider_tol.value()}（应用以生效 / 取消放弃）")

    def apply_key(self) -> None:
        if self.arr is None:
            return
        if self._preview is None:
            self._update_preview()
        if self._preview is None:
            if self.chk_rembg.isChecked():
                self._push_undo()
                try:
                    self.arr = ImageProcessor()._remove_with_rembg(self.arr[:, :, :3])
                except RuntimeError as e:
                    self._undo.pop()
                    QMessageBox.warning(self, "图像工具坊", str(e))
                    return
                self._render()
                self._record("remove_bg", bg_mode="rembg")
                self.status("rembg 去背景完成")
                return
            QMessageBox.information(
                self, "图像工具坊",
                "四角颜色不一致，未检测到纯色背景。\n"
                "可勾选 rembg 兜底，或用画笔修边。")
            return
        self._push_undo()
        self.arr = self._preview
        self._preview = None
        self._crop_rect = None
        self._render()
        self._record("remove_bg", tolerance=self.slider_tol.value())
        self.status("已应用去背景（画笔修边/裁切可继续）")

    def cancel_preview(self) -> None:
        if self._preview is not None:
            self._preview = None
            self._render()
            self.status("已取消预览")

    # ---------------- 智能单帧 ----------------

    def smart_single_frame(self) -> None:
        """整体内容紧致化输出 1 帧（单角色图不再被网格切碎）。"""
        if self.arr is None:
            return
        ys, xs = np.where(self.arr[:, :, 3] > 0)
        if ys.size == 0:
            QMessageBox.information(self, "图像工具坊", "画面全空，没有可裁切的内容")
            return
        pad = 2
        h, w = self.arr.shape[:2]
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
        self._record("smart_single")
        self._set_frames([self.arr[y0:y1, x0:x1].copy()])
        self.status(f"智能单帧: {x1 - x0}x{y1 - y0}（整体内容紧致化）")

    # ---------------- 操作模式 / 画笔 ----------------

    def _set_brush_mode(self, idx: int) -> None:
        self._brush_mode = ("browse", "erase", "restore", "crop")[idx]
        self.canvas.setCursor(Qt.CrossCursor if idx else Qt.ArrowCursor)
        if self._brush_mode == "crop":
            if self.arr is None:
                self.status("先打开图片再使用裁切框")
                return
            h, w = self.arr.shape[:2]
            if self._crop_rect is None:
                self._crop_rect = [0, 0, w, h]
            self.status("裁切模式：拖黄线四边微调，框内拖动整体移动，框外拖动新建框")
        elif self._brush_mode == "erase":
            self.status("擦除模式：左键涂抹抹掉残留")
        elif self._brush_mode == "restore":
            self.status("恢复模式：左键涂抹找回原图像素")
        else:
            self.status("浏览模式")
        self._render()

    def _img_pos(self, event: QMouseEvent, clamp: bool = False) -> tuple[int, int] | None:
        """画布坐标 -> 图像像素坐标（考虑居中偏移与缩放）。"""
        pm = self.canvas.pixmap()
        if pm is None or self.arr is None:
            return None
        ox = max(0, (self.canvas.width() - pm.width()) // 2)
        oy = max(0, (self.canvas.height() - pm.height()) // 2)
        x = (event.position().x() - ox) / self._zoom
        y = (event.position().y() - oy) / self._zoom
        h, w = self.arr.shape[:2]
        if clamp:
            return int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
        if not (0 <= x < w and 0 <= y < h):
            return None
        return int(x), int(y)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.canvas and self.arr is not None:
            t = event.type()
            if t == QEvent.MouseMove:
                pos = self._img_pos(event)
                if pos is not None and self.chk_mag.isChecked():
                    self._update_magnifier(*pos)
            if self._brush_mode == "crop":
                if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._crop_press(self._img_pos(event, clamp=True))
                elif t == QEvent.MouseMove and self._drag:
                    self._crop_move(self._img_pos(event, clamp=True))
                elif t == QEvent.MouseButtonRelease:
                    self._drag = None
            elif self._brush_mode != "browse":
                if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._push_undo()
                    self._painting = True
                    self._paint_at(event)
                elif t == QEvent.MouseMove and self._painting:
                    self._paint_at(event)
                elif t == QEvent.MouseButtonRelease and self._painting:
                    self._painting = False
                    self.status("笔刷修改完成（Ctrl+Z 可撤销）")
        return super().eventFilter(obj, event)

    def _paint_at(self, event: QMouseEvent) -> None:
        pos = self._img_pos(event)
        if pos is None:
            return
        x, y = pos
        r = self._brush_size
        if self._brush_mode == "erase":
            alpha = np.ascontiguousarray(self.arr[:, :, 3])
            cv2.circle(alpha, (x, y), r, 0, -1)
            self.arr[:, :, 3] = alpha
        else:  # restore：从原图找回该圆内的不透明像素
            h, w = self.arr.shape[:2]
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            src = self._orig[y0:y1, x0:x1]
            mask = src[:, :, 3] > 0
            self.arr[y0:y1, x0:x1][mask] = src[mask]
        self._render()

    # ---------------- 自由裁切 ----------------

    def _crop_hit(self, x: int, y: int) -> str | None:
        x0, y0, x1, y1 = self._crop_rect
        tol = max(3, int(6 / self._zoom))
        if abs(x - x0) <= tol:
            return "left"
        if abs(x - x1) <= tol:
            return "right"
        if abs(y - y0) <= tol:
            return "top"
        if abs(y - y1) <= tol:
            return "bottom"
        if x0 < x < x1 and y0 < y < y1:
            return "move"
        return None

    def _crop_press(self, pos: tuple[int, int] | None) -> None:
        if pos is None or self._crop_rect is None:
            return
        x, y = pos
        hit = self._crop_hit(x, y)
        self._drag_anchor = (x, y)
        if hit:
            self._drag = hit
            self._drag_orig = list(self._crop_rect)
        else:
            self._drag = "new"
            self._crop_rect = [x, y, x, y]
        self._render()

    def _crop_move(self, pos: tuple[int, int] | None) -> None:
        if pos is None or not self._drag or self._crop_rect is None:
            return
        x, y = pos
        h, w = self.arr.shape[:2]
        r = self._crop_rect
        if self._drag == "left":
            r[0] = int(min(max(x, 0), r[2] - 2))
        elif self._drag == "right":
            r[2] = int(max(min(x, w), r[0] + 2))
        elif self._drag == "top":
            r[1] = int(min(max(y, 0), r[3] - 2))
        elif self._drag == "bottom":
            r[3] = int(max(min(y, h), r[1] + 2))
        elif self._drag == "move":
            dx = x - self._drag_anchor[0]
            dy = y - self._drag_anchor[1]
            ox0, oy0, ox1, oy1 = self._drag_orig
            bw, bh = ox1 - ox0, oy1 - oy0
            nx0 = min(max(0, ox0 + dx), w - bw)
            ny0 = min(max(0, oy0 + dy), h - bh)
            r[0], r[2] = nx0, nx0 + bw
            r[1], r[3] = ny0, ny0 + bh
        else:  # new
            ax, ay = self._drag_anchor
            r[0], r[2] = min(ax, x), max(ax, x)
            r[1], r[3] = min(ay, y), max(ay, y)
        self._render()

    def apply_crop(self) -> None:
        if self.arr is None or not self._crop_rect:
            QMessageBox.information(
                self, "图像工具坊",
                "请先在操作模式中选择“裁切框”，调整好黄线后再应用。")
            return
        x0, y0, x1, y1 = [int(v) for v in self._crop_rect]
        x0, x1 = sorted((max(0, x0), min(self.arr.shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(self.arr.shape[0], y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            QMessageBox.warning(self, "图像工具坊", "裁切区域太小")
            return
        self._push_undo()
        self.arr = self.arr[y0:y1, x0:x1].copy()
        self._crop_rect = None
        self._render()
        self.status(f"已裁切: {x1 - x0}x{y1 - y0}")

    def reset_crop(self) -> None:
        if self.arr is not None:
            h, w = self.arr.shape[:2]
            self._crop_rect = [0, 0, w, h]
            self._render()
            self.status("裁切框已重置为全图")

    # ---------------- 裁切与导出 ----------------

    def auto_crop(self) -> None:
        if self.arr is None:
            return
        mode = ("feet_center", "centroid", "baseline")[self.cmb_align.currentIndex()]
        proc = ImageProcessor(
            align_mode=mode,
            harden_alpha=self.chk_harden.isChecked(),
            bleed_px=self.spin_bleed.value(),
            dither=self.chk_dither.isChecked(),
            pixel_size=self.spin_pixel.value(),
        )
        if self._locked_palette:
            proc.set_locked_palette(self._locked_palette)
        frames = proc.run_pipeline(self.arr)
        if not frames:
            QMessageBox.information(self, "图像工具坊", "未检测到帧（画面全空？）")
            return
        if proc.last_palette and not self._locked_palette:
            self._locked_palette = proc.last_palette
            self.lbl_palette.setText(f"调色板：已记录 {len(self._locked_palette)} 色")
            self._refresh_swatch()
        self._record("auto_crop", align=mode,
                     harden_alpha=self.chk_harden.isChecked(),
                     bleed_px=self.spin_bleed.value(),
                     dither=self.chk_dither.isChecked(),
                     pixel_size=self.spin_pixel.value())
        self._set_frames([f.image for f in frames])
        self.status(f"连通域裁切: {len(frames)} 帧（对齐: {mode}，"
                    f"调色板: {len(proc.last_palette or []) or "锁定"} 色）")

    def _load_palette_dialog(self) -> None:
        from pixelforge.processing.image_processor import load_palette
        f, _ = QFileDialog.getOpenFileName(self, "锁定调色板", "",
                                           "调色板 (*.json)")
        if not f:
            return
        self._locked_palette = load_palette(Path(f))
        self.lbl_palette.setText(f"调色板：已锁定 {len(self._locked_palette)} 色")
        self._refresh_swatch()
        self.status(f"已锁定调色板 {len(self._locked_palette)} 色（来自 {Path(f).name}）")

    def _save_palette_dialog(self) -> None:
        if not self._locked_palette:
            QMessageBox.information(self, "图像工具坊",
                                    "当前没有可保存的调色板（先裁切一次或锁定）")
            return
        from pixelforge.processing.image_processor import save_palette
        f, _ = QFileDialog.getSaveFileName(self, "保存调色板",
                                           "palette.json", "调色板 (*.json)")
        if not f:
            return
        save_palette(Path(f), self._locked_palette)
        self.status(f"调色板已保存: {f}")

    def grid_crop(self) -> None:
        if self.arr is None:
            return
        try:
            frames = editing.grid_split(self.arr, self.spin_rows.value(),
                                        self.spin_cols.value())
        except ValueError as e:
            QMessageBox.warning(self, "图像工具坊", str(e))
            return
        self._record("grid_split", rows=self.spin_rows.value(),
                     cols=self.spin_cols.value())
        self._set_frames(frames)
        self.status(f"网格切分: {len(frames)} 帧")

    def _set_frames(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self.lst_frames.clear()
        for i, a in enumerate(frames):
            it = QListWidgetItem(f"#{i} {a.shape[1]}x{a.shape[0]}")
            pm = _to_pixmap(a).scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            it.setIcon(QIcon(pm))
            self.lst_frames.addItem(it)
        if not frames:
            QMessageBox.information(self, "图像工具坊", "未检测到帧（画面全空？）")

    def send_to_player(self) -> None:
        if not self._frames:
            QMessageBox.information(self, "图像工具坊", "请先裁切出帧再发送")
            return
        self.frames_to_player.emit(list(self._frames))

    def save_frames(self) -> None:
        if not self._frames:
            QMessageBox.information(self, "图像工具坊", "请先执行自动裁切或网格切分")
            return
        d = QFileDialog.getExistingDirectory(self, "选择帧输出目录")
        if not d:
            return
        out = Path(d)
        name = self.path.stem if self.path else "frames"
        meta = []
        for i, a in enumerate(self._frames):
            fn = f"{name}_{i:03d}.png"
            cv2.imwrite(str(out / fn), a)
            ys, xs = np.where(a[:, :, 3] > 0)
            fx = 0.5
            if ys.size:
                x0, y0 = int(xs.min()), int(ys.min())
                rect = {"x": x0, "y": y0, "width": int(xs.max() - x0 + 1),
                        "height": int(ys.max() - y0 + 1)}
                bottom = int(ys.max())
                band_top = max(0, bottom - max(2, int((bottom + 1) * 0.15)))
                bxs = np.where(a[band_top:bottom + 1, :, 3] > 0)[1]
                if bxs.size:
                    fx = float(bxs.mean()) / a.shape[1]
            else:
                rect = {"x": 0, "y": 0, "width": a.shape[1], "height": a.shape[0]}
            meta.append(rect | {"file": fn, "pivotX": round(fx, 4), "pivotY": 1.0})
        atlas = {"name": name, "fps": 8, "frame_count": len(meta), "frames": meta}
        (out / f"{name}.atlas.json").write_text(
            json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        self.status(f"已保存 {len(meta)} 帧 + atlas.json 到 {out}")

    # ---------------- 新增图形能力 ----------------

    def _rot90(self) -> None:
        self._mutate(editing.rotate90, self.arr)
        self._record("rotate90", k=1)

    def _flip_h(self) -> None:
        self._mutate(editing.flip, self.arr, True)
        self._record("flip", horizontal=True)

    def _flip_v(self) -> None:
        self._mutate(editing.flip, self.arr, False)
        self._record("flip", horizontal=False)

    def rotate_any(self) -> None:
        """任意角度旋转（最近邻采样保持像素感）。"""
        if self.arr is None:
            return
        self._mutate(editing.rotate_arbitrary, self.arr,
                     float(self.spin_angle.value()))
        self._record("rotate_arbitrary", angle=self.spin_angle.value())

    def bleed_now(self) -> None:
        """对当前画布做边缘外扩（RGB 外渗、alpha 不变）。"""
        if self.arr is None:
            return
        self._push_undo()
        self.arr = editing.bleed_edges(self.arr, 1)
        self._preview = None
        self._render()
        self._record("bleed", px=1)
        self.status("边缘外扩完成（防图集采样串色）")

    def proj_split(self) -> None:
        """投影分析拆分：适用于带背景的外部精灵表。"""
        if self.arr is None:
            return
        frames = ImageProcessor().split_sheet_projection(self.arr)
        if not frames:
            QMessageBox.information(self, "图像工具坊", "投影分析未发现内容")
            return
        self._record("projection_split")
        self._set_frames([f.image for f in frames])
        self.status(f"投影拆分: {len(frames)} 帧")

    def import_aseprite_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "导入 Aseprite", "",
                                           "Aseprite JSON (*.json)")
        if not f:
            return
        try:
            frames, fps = import_aseprite(Path(f))
        except Exception as e:
            QMessageBox.warning(self, "图像工具坊", str(e))
            return
        self._undo.clear()
        self.arr = frames[0].copy()
        self._orig = self.arr.copy()
        self._preview = None
        self._crop_rect = None
        self.path = Path(f)
        self._set_frames(frames)
        self._set_zoom(1.0)
        self.status(f"已导入 Aseprite: {Path(f).name}（{len(frames)} 帧, {fps} fps）")

    # ---------------- 换色引擎 ----------------

    def _refresh_swatch(self) -> None:
        self.lst_swatch.clear()
        for i, c in enumerate(self._locked_palette or []):
            b, g, r = int(c[0]), int(c[1]), int(c[2])
            pm = QPixmap(22, 22)
            pm.fill(QColor(r, g, b))
            it = QListWidgetItem(f"{i}")
            it.setIcon(QIcon(pm))
            it.setData(Qt.UserRole, i)
            it.setToolTip(f"#{i}  RGB({r},{g},{b})")
            self.lst_swatch.addItem(it)

    def _pick_target(self) -> None:
        it = self.lst_swatch.currentItem()
        if it is None:
            QMessageBox.information(self, "换色引擎", "先在色板条中选中一个源颜色")
            return
        src = int(it.data(Qt.UserRole))
        c = QColorDialog.getColor(parent=self, title=f"色 #{src} 的目标颜色")
        if not c.isValid():
            return
        self._map[src] = (c.blue(), c.green(), c.red())
        self._refresh_map()

    def _refresh_map(self) -> None:
        self.lst_map.clear()
        for src, (b, g, r) in sorted(self._map.items()):
            pm = QPixmap(14, 14)
            pm.fill(QColor(r, g, b))
            self.lst_map.addItem(
                QListWidgetItem(QIcon(pm), f"色 #{src} -> #{r:02X}{g:02X}{b:02X}"))

    def _clear_map(self) -> None:
        self._map.clear()
        self._refresh_map()
        self.status("换色映射已清空")

    def apply_remap(self) -> None:
        if not self._map:
            QMessageBox.information(self, "换色引擎",
                                    "没有映射（选中色板项后拾取目标色）")
            return
        if not self._locked_palette:
            QMessageBox.information(self, "换色引擎", "需要先自动裁切生成/锁定调色板")
            return
        mapping = {str(k): list(v) for k, v in self._map.items()}
        if self._frames:
            self._frames = [remap_palette_colors(f, self._locked_palette, mapping)
                            for f in self._frames]
            self._set_frames(self._frames)
        else:
            self._push_undo()
            self.arr = remap_palette_colors(self.arr, self._locked_palette, mapping)
            self._preview = None
            self._render()
        self._record("remap",
                     palette=[list(map(int, c)) for c in self._locked_palette],
                     mapping=mapping)
        self.status(f"换色完成：{len(mapping)} 个颜色被替换")

    # ---------------- 动作宏 ----------------

    def _record(self, op: str, **args) -> None:
        """录制钩子：勾选录制时把操作追加为宏步骤。"""
        if self.chk_record.isChecked():
            self._macro_steps.append({"op": op, "args": args})
            self.status(f"宏录制：+ {op}（共 {len(self._macro_steps)} 步）")

    def _toggle_record(self, on: bool) -> None:
        if on:
            self._macro_steps = []
            self.status("宏录制已开启：接下来执行的每步操作都会被记录")
        else:
            self.status(f"宏录制已关闭（共 {len(self._macro_steps)} 步）")

    def macro_replay(self) -> None:
        if self.arr is None:
            return
        if not self._macro_steps:
            QMessageBox.information(self, "动作宏",
                                    "没有已录制的步骤（先勾选录制并执行操作）")
            return
        try:
            frames = apply_macro(self._macro_steps, self.arr)
        except Exception as e:
            QMessageBox.warning(self, "动作宏", f"回放失败: {e}")
            return
        self._set_frames(frames)
        self.status(f"宏回放完成（{len(self._macro_steps)} 步 -> {len(frames)} 帧）")

    def macro_save(self) -> None:
        if not self._macro_steps:
            QMessageBox.information(self, "动作宏", "没有已录制的步骤")
            return
        f, _ = QFileDialog.getSaveFileName(self, "保存宏", "macro.json",
                                           "动作宏 (*.json)")
        if not f:
            return
        try:
            save_macro(Path(f), self._macro_steps)
        except ValueError as e:
            QMessageBox.warning(self, "动作宏", str(e))
            return
        self.status(f"宏已保存: {f}")

    def macro_load_replay(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "加载宏", "", "动作宏 (*.json)")
        if not f:
            return
        try:
            self._macro_steps = load_macro(f)
        except Exception as e:
            QMessageBox.warning(self, "动作宏", f"加载失败: {e}")
            return
        self.status(f"宏已加载（{len(self._macro_steps)} 步）")
        self.macro_replay()

    def macro_batch(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "选择宏", "", "动作宏 (*.json)")
        if not f:
            return
        src = QFileDialog.getExistingDirectory(self, "选择输入目录（批量图片）")
        if not src:
            return
        dst = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not dst:
            return
        self._macro_worker = _MacroWorker(f, src, dst, self)
        self._macro_worker.done.connect(self._on_macro_done)
        self._macro_worker.failed.connect(self._on_macro_failed)
        self._macro_worker.start()
        self.status("批量回放中……")

    def _on_macro_done(self, n: int) -> None:
        self.status(f"批量回放完成：输出 {n} 帧")

    def _on_macro_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "动作宏", f"批量回放失败: {msg}")

    def apply_adjust(self) -> None:
        self._mutate(
            editing.adjust_bcs, self.arr,
            float(self.sl_b.value()),
            1.0 + self.sl_c.value() / 100.0,
            self.sl_s.value() / 100.0,
        )
        self._record("adjust", brightness=float(self.sl_b.value()),
                     contrast=1.0 + self.sl_c.value() / 100.0,
                     saturation=self.sl_s.value() / 100.0)
        self.status("色彩调整已应用（撤销可回退）")

    def _trim_dialog(self) -> None:
        if self.arr is None:
            return
        from pixelforge.ui.review_panel import CropDialog
        h, w = self.arr.shape[:2]
        dlg = CropDialog(self, w, h)
        if dlg.exec():
            self._mutate(editing.trim_edges, self.arr,
                         dlg.sp_l.value(), dlg.sp_t.value(),
                         dlg.sp_r.value(), dlg.sp_b.value())
            self._record("trim", left=dlg.sp_l.value(), top=dlg.sp_t.value(),
                         right=dlg.sp_r.value(), bottom=dlg.sp_b.value())

    # ---------------- 渲染 ----------------

    def _set_zoom(self, v: float) -> None:
        self._zoom = v
        self._render()

    def _open_ab_compare(self) -> None:
        """⇋ A/B 对比：左=打开时的原图(A)，右=当前结果(B)，中缝可拖动。

        A/B 尺寸不同时把 B 就近缩放到 A 尺寸，保证分割对比对齐。
        """
        a = self._orig
        b = self._preview if self._preview is not None else self.arr
        if a is None or b is None:
            self.state_changed.emit("A/B 对比：请先打开一张图片")
            return
        note = ""
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            note = "（尺寸不同：B 已就近缩放到 A 尺寸）"
        dlg = QDialog(self)
        dlg.setWindowTitle("A/B 对比 —— 左：原图(A)  右：当前(B)")
        dlg.resize(860, 680)
        lay = QVBoxLayout(dlg)
        view = _ABCompare(a, b)
        lay.addWidget(view, 1)
        tip = QLabel("拖动中间黄色分割线自由对比｜双击画面复位 50%｜Esc 关闭　"
                     + note)
        tip.setStyleSheet("color:#8a8f98; font-size:11px;")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        row = QHBoxLayout()
        row.addStretch(1)
        b_close = QPushButton("关闭")
        b_close.setProperty("accent", True)
        b_close.clicked.connect(dlg.accept)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.state_changed.emit("A/B 对比：拖动分割线比对原图与当前结果")
        dlg.exec()

    def _render(self, arr: np.ndarray | None = None) -> None:
        a = arr if arr is not None else (
            self._preview if self._preview is not None else self.arr)
        if a is None:
            return
        pm = _to_pixmap(a)
        if self._zoom != 1.0:
            pm = pm.scaled(int(pm.width() * self._zoom), int(pm.height() * self._zoom),
                           Qt.KeepAspectRatio,
                           Qt.FastTransformation if self._zoom > 1 else Qt.SmoothTransformation)
        if arr is None and self._brush_mode == "crop" and self._crop_rect \
                and a is self.arr:
            self._paint_crop_overlay(pm)
        self.canvas.setPixmap(pm)

    def _paint_crop_overlay(self, pm: QPixmap) -> None:
        """在画布上叠加裁切框：外部半透明遮罩 + 黄色边线 + 四边中点手柄。"""
        p = QPainter(pm)
        x0, y0, x1, y1 = [int(round(v * self._zoom)) for v in self._crop_rect]
        rect = QRect(x0, y0, x1 - x0, y1 - y0)
        p.setClipRegion(QRegion(pm.rect()) - QRegion(rect))
        p.fillRect(pm.rect(), QColor(0, 0, 0, 130))
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 208, 0), 2))
        p.drawRect(rect)
        p.fillRect(x0 - 3, (y0 + y1) // 2 - 3, 6, 6, QColor(255, 208, 0))
        p.fillRect(x1 - 3, (y0 + y1) // 2 - 3, 6, 6, QColor(255, 208, 0))
        p.fillRect((x0 + x1) // 2 - 3, y0 - 3, 6, 6, QColor(255, 208, 0))
        p.fillRect((x0 + x1) // 2 - 3, y1 - 3, 6, 6, QColor(255, 208, 0))
        p.end()

    def _update_magnifier(self, ix: int, iy: int) -> None:
        """放大镜：鼠标附近 80x80 像素区域放大显示 + 中心十字线。"""
        if self.arr is None:
            return
        r = 40
        h, w = self.arr.shape[:2]
        x0, x1 = max(0, ix - r), min(w, ix + r)
        y0, y1 = max(0, iy - r), min(h, iy + r)
        tile = self.arr[y0:y1, x0:x1]
        if tile.size == 0:
            return
        pm = _to_pixmap(tile).scaled(self.mag.width(), self.mag.height(),
                                     Qt.KeepAspectRatioByExpanding,
                                     Qt.FastTransformation)
        p = QPainter(pm)
        p.setPen(QPen(QColor(255, 60, 60), 1))
        p.drawLine(0, pm.height() // 2, pm.width(), pm.height() // 2)
        p.drawLine(pm.width() // 2, 0, pm.width() // 2, pm.height())
        p.end()
        self.mag.setPixmap(pm)

    def status(self, msg: str) -> None:
        self.state_changed.emit(msg)

    # ---------------- 拖拽 ----------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp",
                                   ".gif", ".apng")):
                self.load_path(Path(p))
                break
