"""动作宏：把一次完整的处理流程录制为 JSON 步骤序列，可单帧重放或批量回放。

步骤格式: {"op": <名称>, "args": {...}}
源操作（从原图生成帧列表）:
    remove_bg          色键去背景（需 ImageProcessor）
    auto_crop          完整管线：去背景->裁切->对齐->调色板->硬化->外扩
    smart_single       智能单帧：去背景 + 紧致裁切
    projection_split   投影分析拆分外部精灵表
    grid_split         均匀网格切分（args: rows/cols/margin）
变换操作（优先作用于已生成的帧列表，否则作用于原图）:
    rotate90 / rotate_arbitrary / flip / adjust / trim / bleed / remap

remap 的 args 中直接内嵌 palette（BGR 列表），宏自包含、跨会话可重放。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pixelforge.processing import editing
from pixelforge.processing.image_processor import ImageProcessor

SOURCE_OPS = ("remove_bg", "auto_crop", "smart_single",
              "projection_split", "grid_split")
TRANSFORM_OPS = ("rotate90", "rotate_arbitrary", "flip", "adjust",
                 "trim", "bleed", "remap")


def save_macro(path, steps: list[dict]) -> None:
    """步骤序列存为 JSON；含未知 op 时拒绝保存。"""
    for s in steps:
        if s.get("op") not in SOURCE_OPS + TRANSFORM_OPS:
            raise ValueError(f"未知宏操作: {s.get('op')!r}")
    Path(path).write_text(
        json.dumps({"version": 1, "steps": steps},
                   ensure_ascii=False, indent=2), "utf-8")


def load_macro(path) -> list[dict]:
    """读取宏 JSON 并校验 op 合法性，返回步骤列表。"""
    data = json.loads(Path(path).read_text("utf-8"))
    steps = data.get("steps", [])
    for s in steps:
        if s.get("op") not in SOURCE_OPS + TRANSFORM_OPS:
            raise ValueError(f"未知宏操作: {s.get('op')!r}")
    return steps


def apply_macro(steps: list[dict], arr: np.ndarray,
                proc: ImageProcessor | None = None) -> list[np.ndarray]:
    """对 BGRA 原图按步骤序列处理，返回帧列表（至少一帧）。

    源操作重建帧列表；其后遇到的变换操作逐帧作用。
    若步骤全是变换操作，则视为对单张原图的编辑，返回单帧列表。
    """
    if proc is None:
        proc = ImageProcessor()
    frames: list[np.ndarray] | None = None
    for step in steps:
        op = step["op"]
        args = dict(step.get("args") or {})
        if op in SOURCE_OPS:
            frames = _SOURCE_HANDLERS[op](arr, proc, args)
        else:
            fn = _TRANSFORM_HANDLERS[op]
            if frames is None:
                arr = fn(arr, args)
            else:
                frames = [fn(f, args) for f in frames]
    return frames if frames is not None else [_ensure_bgra(arr)]


def replay_folder(macro_path, in_dir, out_dir,
                  proc: ImageProcessor | None = None) -> list[Path]:
    """批量回放：对 in_dir 下每张图片应用同一宏，帧写入 out_dir。

    输出命名 <图名>_###.png；返回生成的文件列表。
    """
    steps = load_macro(macro_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for p in sorted(Path(in_dir).iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            continue
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            continue
        frames = apply_macro(steps, _ensure_bgra(arr), proc)
        for i, f in enumerate(frames):
            fp = out_dir / f"{p.stem}_{i:03d}.png"
            cv2.imwrite(str(fp), f)
            produced.append(fp)
    return produced


# ------------------------------------------------ 内部工具

def _ensure_bgra(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return arr


# ------------------------------------------------ 源操作处理器

def _op_remove_bg(arr, proc: ImageProcessor, args):
    proc.bg_mode = args.get("bg_mode", proc.bg_mode)
    proc.bg_tolerance = int(args.get("tolerance", proc.bg_tolerance))
    return [proc.remove_background(arr)]


def _op_auto_crop(arr, proc: ImageProcessor, args):
    proc.align_mode = args.get("align", proc.align_mode)
    proc.harden_alpha = bool(args.get("harden_alpha", proc.harden_alpha))
    proc.bleed_px = int(args.get("bleed_px", proc.bleed_px))
    proc.dither = bool(args.get("dither", proc.dither))
    proc.pixel_size = int(args.get("pixel_size", proc.pixel_size))
    bgra = proc.remove_background(arr)
    return [f.image for f in proc.run_pipeline(bgra)]


def _op_smart_single(arr, proc: ImageProcessor, args):
    bgra = proc.remove_background(arr)
    return [editing.tighten(bgra, int(args.get("pad", 2)))]


def _op_projection_split(arr, proc: ImageProcessor, args):
    return [f.image for f in proc.split_sheet_projection(
        _ensure_bgra(arr), float(args.get("bg_tol", 32.0)))]


def _op_grid_split(arr, proc: ImageProcessor, args):
    return editing.grid_split(_ensure_bgra(arr),
                              int(args.get("rows", 1)),
                              int(args.get("cols", 1)),
                              int(args.get("margin", 0)))


# ------------------------------------------------ 变换操作处理器

def _op_rotate90(img, args):
    return editing.rotate90(img, int(args.get("k", 1)))


def _op_rotate_arbitrary(img, args):
    return editing.rotate_arbitrary(img, float(args.get("angle", 0.0)))


def _op_flip(img, args):
    return editing.flip(img, bool(args.get("horizontal", True)))


def _op_adjust(img, args):
    return editing.adjust_bcs(img,
                              float(args.get("brightness", 0.0)),
                              float(args.get("contrast", 1.0)),
                              float(args.get("saturation", 1.0)))


def _op_trim(img, args):
    return editing.trim_edges(img,
                              int(args.get("left", 0)),
                              int(args.get("top", 0)),
                              int(args.get("right", 0)),
                              int(args.get("bottom", 0)))


def _op_bleed(img, args):
    return editing.bleed_edges(img, int(args.get("px", 1)))


def _op_remap(img, args):
    palette = args.get("palette")
    if not palette:
        raise ValueError("remap 需要调色板（录制时会自动内嵌）")
    return remap_palette_colors(img, palette,
                                dict(args.get("mapping") or {}))


_SOURCE_HANDLERS = {
    "remove_bg": _op_remove_bg,
    "auto_crop": _op_auto_crop,
    "smart_single": _op_smart_single,
    "projection_split": _op_projection_split,
    "grid_split": _op_grid_split,
}

_TRANSFORM_HANDLERS = {
    "rotate90": _op_rotate90,
    "rotate_arbitrary": _op_rotate_arbitrary,
    "flip": _op_flip,
    "adjust": _op_adjust,
    "trim": _op_trim,
    "bleed": _op_bleed,
    "remap": _op_remap,
}

# remap 复用 image_processor 的吸附实现（避免循环导入放在底部引入）
from pixelforge.processing.image_processor import remap_palette_colors  # noqa: E402
