"""waifu2x 超分辨率（UpConv7 架构，纯 numpy/cv2 推理，不依赖深度学习框架）。

权重来源：PyPI 包 waifu2x（FHPythonUtils/Waifu2x，MIT）内嵌的
models/upconv7/anime_style_*_rgb.npz —— 源自 nagadomi/waifu2x 与
tsurumeso/waifu2x-chainer（均为 MIT）。随包分发于 pixelforge/models/waifu2x/。

架构（与上游 chainer 实现逐位对齐）：
    conv1..conv6 : 3x3 卷积，pad=0（valid），LeakyReLU(0.1)，输出尺寸逐层 -2
    conv7        : 反卷积 ksize=4, stride=2, pad=3, 无偏置 -> out = 2*in - 4
    整网: in -> (in - 12) -> 2*(in - 12) - 4 = 2*in - 28
因此输入四周以 edge 模式填充 7px 后，输出恰为 2 倍尺寸，无需再裁剪。
上游 blockwise 分块与整图推理在内部区域完全等价，这里按块推理以限制内存。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "waifu2x"

NOISE_NAMES = {-1: "不降噪", 0: "降噪 0（几乎无）", 1: "降噪 1（轻）",
               2: "降噪 2（中）", 3: "降噪 3（强）"}

# 分块尺寸（inner tile）；78x78 输入块 -> 156x156 输出块，内存约 60MB/块
_TILE_INNER = 64


class Waifu2xError(RuntimeError):
    """waifu2x 处理失败（模型缺失 / 图片过大等）。"""


def available_models() -> list[str]:
    """列出已随包分发的模型文件名。"""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.npz"))


def _model_file(noise_level: int) -> Path:
    name = ("anime_style_scale_rgb.npz" if noise_level < 0
            else f"anime_style_noise{int(noise_level)}_scale_rgb.npz")
    p = MODELS_DIR / name
    if not p.is_file():
        raise Waifu2xError(
            f"缺少 waifu2x 模型权重: {p}\n"
            "请重新安装/解包（pixelforge/models/waifu2x/ 应含 5 个 .npz）")
    return p


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k].astype(np.float32) for k in d.files}


def _conv_valid(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """3x3 valid 卷积（im2col + BLAS），x: (Cin, H, W) float32。"""
    cin, h, wd = x.shape
    cout, _, kh, kw = w.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
    # (Cin, oh, ow, kh, kw) -> (oh*ow, Cin*kh*kw)
    cols = win.transpose(1, 2, 0, 3, 4).reshape(
        (h - kh + 1) * (wd - kw + 1), cin * kh * kw)
    out = cols @ w.reshape(cout, -1).T
    if b is not None:
        out += b
    oh, ow = h - kh + 1, wd - kw + 1
    return out.reshape(oh, ow, cout).transpose(2, 0, 1)


def _deconv2d(x: np.ndarray, w: np.ndarray, stride: int = 2,
              pad: int = 3) -> np.ndarray:
    """转置卷积（无偏置）：x (Cin,H,W) -> (Cout, s*(H-1)-2p+k, s*(W-1)-2p+k)。

    按卷积核的 16 个 (dy,dx) 偏移做 strided scatter-add，内存与速度都稳定。
    """
    cin, h, wd = x.shape
    cin, cout, kh, kw = w.shape      # chainer Deconv W: (in, out, kh, kw)
    oh = stride * (h - 1) - 2 * pad + kh
    ow = stride * (wd - 1) - 2 * pad + kw
    out = np.zeros((cout, oh, ow), np.float32)
    for dy in range(kh):
        ys = dy - pad
        i0 = max(0, math.ceil(-ys / stride))
        i1 = min(h, math.ceil((oh - ys) / stride))
        if i0 >= i1:
            continue
        for dx in range(kw):
            xs = dx - pad
            j0 = max(0, math.ceil(-xs / stride))
            j1 = min(wd, math.ceil((ow - xs) / stride))
            if j0 >= j1:
                continue
            patch = x[:, i0:i1, j0:j1]                     # (Cin, n, m)
            # w 形如 (Cin, Cout, kh, kw)（chainer Deconvolution2D 约定）
            contrib = np.tensordot(w[:, :, dy, dx], patch, axes=([0], [0]))
            out[:, stride * i0 + ys: stride * (i1 - 1) + ys + 1: stride,
                stride * j0 + xs: stride * (j1 - 1) + xs + 1: stride] += contrib
    return out


def _forward_block(x: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """单个输入块 (3, in_block, in_block) 0..1 -> (3, block, block) 0..1。"""
    h = x
    for i in range(1, 7):
        h = _conv_valid(h, w[f"conv{i}/W"], w[f"conv{i}/b"])
        np.maximum(h, 0.1 * h, out=h)          # LeakyReLU(0.1)，就地省内存
    return _deconv2d(h, w["conv7/W"], stride=2, pad=3)


def _blockwise(rgb01: np.ndarray, w: dict[str, np.ndarray],
               progress=None, tile_inner: int = _TILE_INNER) -> np.ndarray:
    """整图分块推理：rgb01 (H,W,3) float32 0..1 -> (2H,2W,3) float32。"""
    hh, ww, _ = rgb01.shape
    in_off = 7                                  # 输入侧 edge 填充（=offset/2）
    in_block = tile_inner + in_off * 2          # 78

    def _pad(size: int) -> int:
        return in_off if size % tile_inner == 0 else \
            tile_inner - size % tile_inner + in_off

    psrc = np.pad(rgb01.transpose(2, 0, 1),
                  ((0, 0), (in_off, _pad(hh)), (in_off, _pad(ww))), "edge")
    nh = (psrc.shape[1] - in_off * 2) // tile_inner
    nw = (psrc.shape[2] - in_off * 2) // tile_inner
    block = 2 * tile_inner
    dst = np.zeros((3, nh * block, nw * block), np.float32)
    total = nh * nw
    for i in range(nh):
        ih = i * tile_inner
        for j in range(nw):
            jw = j * tile_inner
            tile = psrc[:, ih:ih + in_block, jw:jw + in_block]
            dst[:, i * block:(i + 1) * block,
                j * block:(j + 1) * block] = _forward_block(tile, w)
            if progress:
                progress((i * nw + j + 1) / total)
    return dst[:, :hh * 2, :ww * 2].transpose(1, 2, 0)


def _fill_transparent(bgra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把透明区的杂色向不透明色内扩（防网络把杂色晕染进可见边缘）。

    返回 (修复后的 RGB 0..1 float32, alpha uint8)。
    """
    a = bgra[:, :, 3]
    rgb = bgra[:, :, :3].astype(np.float32)
    holes = a == 0
    if holes.any() and (~holes).any():
        filled = cv2.inpaint(bgra[:, :, :3], holes.astype(np.uint8), 3,
                             cv2.INPAINT_TELEA)
        rgb = filled.astype(np.float32)
    return rgb / 255.0, a


def _upscale_alpha(a: np.ndarray, scale: int) -> np.ndarray:
    """alpha 用最近邻放大：像素画硬边蒙版保持锐利（与 RGB 的 2x 步长一致）。"""
    return cv2.resize(a, (a.shape[1] * scale, a.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def upscale(bgra: np.ndarray, scale: int = 2, noise_level: int = -1,
            progress=None) -> np.ndarray:
    """waifu2x 放大 BGRA 图。

    参数
    ----
    bgra        : BGRA uint8（BGR 三通道输入会自动补 alpha）
    scale       : 2 或 4（4 = 连跑两次 2x）
    noise_level : -1 不降噪 / 0~3 降噪强度（对应不同权重文件）
    progress    : 可选回调 fn(0..1)，跨多次 2x 累计

    返回 BGRA uint8，尺寸为 (H*scale, W*scale)。
    """
    if scale not in (2, 4):
        raise Waifu2xError(f"不支持的倍率: {scale}（仅 2x / 4x）")
    if noise_level not in NOISE_NAMES:
        raise Waifu2xError(f"不支持的降噪等级: {noise_level}")
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    elif bgra.shape[2] == 3:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)

    w = _load_weights(_model_file(noise_level))
    rgb, a = _fill_transparent(bgra)
    passes = scale // 2
    out_rgb, out_a = rgb, a
    for p in range(passes):
        base = p / passes

        def cb(v, base=base, passes=passes, progress=progress):
            if progress:
                progress((base + v / passes))

        out_rgb = np.clip(_blockwise(out_rgb, w, cb), 0.0, 1.0)
        out_a = _upscale_alpha(out_a, 2)
    out = np.empty((*out_a.shape, 4), np.uint8)
    out[:, :, :3] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 3] = out_a
    return out


def estimate_pixels(src_w: int, src_h: int, scale: int = 2) -> int:
    """估算输出像素总数（UI 预检用）。"""
    return (src_h * scale) * (src_w * scale)
