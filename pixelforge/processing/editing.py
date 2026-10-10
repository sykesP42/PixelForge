"""常用图像编辑核心（纯函数，可独立测试）。

所有函数输入/输出均为 BGRA uint8 ndarray，不改变尺寸（trim 除外）。
"""
from __future__ import annotations

import cv2
import numpy as np


def adjust_bcs(bgra: np.ndarray, brightness: float = 0.0,
               contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
    """亮度(-100~100) / 对比度(0.5~1.5) / 饱和度(0~2) 调整，仅作用于内容像素。"""
    out = bgra.copy()
    bgr = out[:, :, :3].astype(np.float32)
    bgr = (bgr - 128.0) * contrast + 128.0 + brightness
    out[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    if abs(saturation - 1.0) > 1e-3:
        mask = out[:, :, 3] > 0
        if mask.any():
            hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            bgr2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            out[:, :, :3] = np.where(mask[:, :, None], bgr2, out[:, :, :3])
    return out


def rotate90(bgra: np.ndarray, k: int = 1) -> np.ndarray:
    """逆时针旋转 90° * k。"""
    return np.ascontiguousarray(np.rot90(bgra, k))


def flip(bgra: np.ndarray, horizontal: bool = True) -> np.ndarray:
    return cv2.flip(bgra, 1 if horizontal else 0)


def trim_edges(bgra: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """裁掉四边像素；越界时返回原图。"""
    h, w = bgra.shape[:2]
    if left + right >= w or top + bottom >= h or min(left, top, right, bottom) < 0:
        return bgra
    return bgra[top:h - bottom, left:w - right].copy()


def grid_split(bgra: np.ndarray, rows: int, cols: int,
               margin: int = 0) -> list[np.ndarray]:
    """按均匀网格切分为 rows x cols 帧（AI 输出规范网格时的快速路径）。"""
    h, w = bgra.shape[:2]
    if rows < 1 or cols < 1:
        raise ValueError("rows/cols 必须 >= 1")
    ch, cw = h // rows, w // cols
    frames = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * ch + margin
            x0 = c * cw + margin
            y1 = (r + 1) * ch - margin if r < rows - 1 else h - margin
            x1 = (c + 1) * cw - margin if c < cols - 1 else w - margin
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue
            cell = bgra[y0:y1, x0:x1].copy()
            # 紧致化：裁到内容 bbox（全空则跳过）
            ys, xs = np.where(cell[:, :, 3] > 0)
            if ys.size == 0:
                continue
            cell = cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            frames.append(cell)
    return frames


def tighten(bgra: np.ndarray, pad: int = 2) -> np.ndarray:
    """紧致裁切到内容 bbox（pad 像素余量）；画面全空时原样返回。"""
    ys, xs = np.where(bgra[:, :, 3] > 0)
    if ys.size == 0:
        return bgra
    h, w = bgra.shape[:2]
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    return bgra[y0:y1, x0:x1].copy()


def bleed_edges(bgra: np.ndarray, px: int = 1) -> np.ndarray:
    """把不透明边缘的 RGB 向透明区外渗 px 像素（alpha 保持 0）。

    防止引擎图集采样在 rect 边界吃到相邻帧颜色（黑边/彩边闪烁）。
    """
    if px < 1 or not (bgra[:, :, 3] > 0).any():
        return bgra
    out = bgra.copy()
    color = out[:, :, :3].astype(np.float32)
    known = (out[:, :, 3] > 0).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(px):
        # 注意：cv2.boxFilter 会把单通道 (h,w,1) 压成 (h,w)，需手动恢复维度
        wsum = cv2.boxFilter(known, -1, (3, 3), normalize=False)[:, :, None]
        ssum = cv2.boxFilter(color * known[:, :, None],
                             -1, (3, 3), normalize=False)
        avg = ssum / np.maximum(wsum, 1e-6)
        dil = cv2.dilate((known > 0).astype(np.uint8), kernel) > 0
        ring = dil & (known == 0)
        color[ring] = avg[ring]
        known[ring] = 1.0
    out[:, :, :3] = np.clip(color, 0, 255).astype(np.uint8)
    return out


def rotate_arbitrary(bgra: np.ndarray, angle: float) -> np.ndarray:
    """任意角度旋转（最近邻采样保持像素感），画布自动扩大，空隙透明。"""
    h, w = bgra.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_v, sin_v = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin_v + w * cos_v + 0.5)
    nh = int(h * cos_v + w * sin_v + 0.5)
    m[0, 2] += nw / 2.0 - center[0]
    m[1, 2] += nh / 2.0 - center[1]
    return cv2.warpAffine(bgra, m, (nw, nh), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def pixelate(bgra: np.ndarray, target: int) -> np.ndarray:
    """像素风化：下采样到目标像素密度（box 平均），再最近邻整数倍放大回原尺寸。

    target = 内容最长边的目标像素数；产出「每个像素块颜色恒定」的真像素画，
    消除 AI 生成图缩小显示时的混色假像素。半透明边缘交给后续边缘硬化处理。
    """
    h, w = bgra.shape[:2]
    target = max(4, int(target))
    scale = max(1, round(max(h, w) / target))
    if scale <= 1:
        return bgra
    small = cv2.resize(bgra, (max(1, w // scale), max(1, h // scale)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def tween_x2(frames: list[np.ndarray]) -> list[np.ndarray]:
    """相邻帧之间插入 50% 混合过渡帧（ghost frame），序列长度近似翻倍。

    输入帧须已统一画布尺寸；两端帧保持原样，过渡帧为半透明混合效果。
    """
    if len(frames) < 2:
        return list(frames)
    out: list[np.ndarray] = []
    for i, f in enumerate(frames):
        out.append(f)
        if i + 1 < len(frames):
            mix = ((f.astype(np.float32) + frames[i + 1].astype(np.float32))
                   / 2.0).astype(np.uint8)
            out.append(mix)
    return out


def adjust_frames(frames: list[np.ndarray],
                  offsets: list[tuple[int, int]] | None = None,
                  scales: list[float] | None = None,
                  anchor: tuple[float, float] = (0.5, 0.5),
                  ) -> list[np.ndarray]:
    """逐帧调节位移与缩放，输出统一画布的帧序列。

    参数
    ----
    frames  : BGRA 帧列表（各帧尺寸可不同）
    offsets : 每帧 (dx, dy) 像素位移，可正可负；None = 全 0
    scales  : 每帧缩放系数（1.0 原样）；None = 全 1.0
    anchor  : (ax, ay) 归一化锚点，缩放与画布内对位的基准。
              常用 (0.5, 0.5)=中心 / (0.5, 1.0)=脚底中心 / (0, 0)=左上

    画布尺寸 = 各帧缩放后的最大宽高，四周再加 pad 容纳最大位移量，
    因此任何合法的位移/缩放都不会裁掉内容。
    """
    if not frames:
        return []
    n = len(frames)
    if offsets is None:
        offsets = [(0, 0)] * n
    if scales is None:
        scales = [1.0] * n
    if len(offsets) != n or len(scales) != n:
        raise ValueError("offsets/scales 长度必须与 frames 一致")

    ax, ay = float(anchor[0]), float(anchor[1])
    scaled: list[np.ndarray] = []
    for a, s in zip(frames, scales):
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGRA)
        elif a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2BGRA)
        s = max(0.01, float(s))
        nw = max(1, int(round(a.shape[1] * s)))
        nh = max(1, int(round(a.shape[0] * s)))
        if (nw, nh) == (a.shape[1], a.shape[0]):
            scaled.append(a)
        else:
            interp = cv2.INTER_NEAREST if s >= 1.0 else cv2.INTER_AREA
            scaled.append(cv2.resize(a, (nw, nh), interpolation=interp))

    max_w = max(f.shape[1] for f in scaled)
    max_h = max(f.shape[0] for f in scaled)
    pad_x = max(abs(int(dx)) for dx, _ in offsets)
    pad_y = max(abs(int(dy)) for _, dy in offsets)
    canvas_w = max_w + 2 * pad_x
    canvas_h = max_h + 2 * pad_y

    out: list[np.ndarray] = []
    for f, (dx, dy) in zip(scaled, offsets):
        canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)
        # 先按锚点在 max_w x max_h 区域内对位，再叠加位移
        bx = int(round(ax * max_w - ax * f.shape[1]))
        by = int(round(ay * max_h - ay * f.shape[0]))
        x0 = pad_x + bx + int(dx)
        y0 = pad_y + by + int(dy)
        fh, fw = f.shape[:2]
        # 与画布求交（数学上恒在内，仍做一次保护）
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(fw - sx0, canvas_w - dx0)
        h = min(fh - sy0, canvas_h - dy0)
        if w <= 0 or h <= 0:
            out.append(canvas)
            continue
        src = f[sy0:sy0 + h, sx0:sx0 + w]
        region = canvas[dy0:dy0 + h, dx0:dx0 + w]
        al = src[:, :, 3:4].astype(np.float32) / 255.0
        region[:] = (src.astype(np.float32) * al
                     + region.astype(np.float32) * (1 - al)).astype(np.uint8)
        out.append(canvas)
    return out


def align_frames_baseline(frames: list[np.ndarray]) -> list[np.ndarray]:
    """一键基线对齐：统一画布到最大尺寸，各帧底边+水平中心对齐（脚底不漂移）。"""
    return adjust_frames(frames, anchor=(0.5, 1.0))


def load_gif_frames(path) -> list[np.ndarray]:
    """GIF/APNG 逐帧导入为 BGRA 列表（PIL 处理帧合成与处置）。"""
    from PIL import Image, ImageSequence
    im = Image.open(str(path))
    out = []
    for fr in ImageSequence.Iterator(im):
        rgba = np.array(fr.convert("RGBA"))
        out.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    im.close()
    return out


def apply_dither(bgra: np.ndarray, palette_bgr: list) -> np.ndarray:
    """Floyd–Steinberg 抖动量化到给定调色板（保留原 alpha）；失败时原样返回。"""
    try:
        from PIL import Image
    except ImportError:
        return bgra
    flat = []
    for c in palette_bgr:
        b, g, r = int(c[0]), int(c[1]), int(c[2])
        flat += [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]
    while len(flat) < 768:
        flat += [0, 0, 0]
    pal_img = Image.new("P", (max(1, len(palette_bgr)), 1))
    pal_img.putpalette(flat)
    rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb, "RGB").quantize(
        palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG)
    dithered = np.array(im.convert("RGB"))
    out = bgra.copy()
    out[:, :, :3] = cv2.cvtColor(dithered, cv2.COLOR_RGB2BGR)
    return out
