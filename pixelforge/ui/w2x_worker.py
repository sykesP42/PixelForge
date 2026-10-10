"""waifu2x 后台线程封装：大图推理不阻塞 UI，工具坊与看图器共用。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from pixelforge.processing import waifu2x

# 单图上限（像素总数）：纯 CPU 推理，2M 像素约需 1~3 分钟
MAX_INPUT_PIXELS = 2_000_000


class Waifu2xWorker(QThread):
    """对一张图执行 waifu2x 放大，结果写盘并回传数组。

    src 可传图片路径，也可直接传 BGRA ndarray（工具坊处理中的画布）。
    """

    progress = Signal(float)
    done = Signal(object, str)      # (BGRA ndarray, 输出文件路径)
    failed = Signal(str)

    def __init__(self, src, scale: int = 2, noise_level: int = -1,
                 out_dir: str | None = None, parent=None):
        super().__init__(parent)
        self._src = src
        self._scale = scale
        self._noise = noise_level
        self._out_dir = out_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if isinstance(self._src, np.ndarray):
                img = self._src
                src = Path(getattr(self, "_src_name", "") or "canvas.png")
            else:
                # np.fromfile + imdecode：Windows 中文路径安全（cv2.imread 不支持）
                data = np.fromfile(str(self._src), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is None:
                    self.failed.emit(f"无法读取图片: {self._src}")
                    return
                src = Path(str(self._src))
            h, w = img.shape[:2]
            if h * w > MAX_INPUT_PIXELS:
                self.failed.emit(
                    f"图片过大（{w}x{h} = {h * w // 10000} 万像素），"
                    f"waifu2x 建议 ≤ {MAX_INPUT_PIXELS // 10000} 万像素；"
                    "请先缩小或裁切")
                return
            out = waifu2x.upscale(
                img, scale=self._scale, noise_level=self._noise,
                progress=lambda v: self.progress.emit(min(0.99, float(v))))
            if self._cancelled:
                return
            base = Path(self._out_dir) if self._out_dir else src.parent
            path = base / f"{src.stem}_waifu2x{self._scale}x.png"
            try:
                base.mkdir(parents=True, exist_ok=True)
            except OSError:
                path = Path(tempfile.gettempdir()) / path.name
            cv2.imwrite(str(path), out)
            self.done.emit(out, str(path))
        except Exception as e:  # noqa: BLE001 后台线程兜底，错误回传 UI
            self.failed.emit(f"{type(e).__name__}: {e}")
