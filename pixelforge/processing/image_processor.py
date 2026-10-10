"""图像后处理管线（阶段2）。

管线顺序：去背景 -> 连通域裁切 -> 基线对齐 -> 统一调色板 -> 元数据 JSON。
可独立运行：
    python -m pixelforge.processing.image_processor <精灵表.png> [-o 输出目录]
输入：AI 生成的精灵表 PNG（绿幕/纯色背景）；
输出：帧序列 PNG（带 Alpha）+ `<名称>.atlas.json` 元数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pixelforge.processing.editing import apply_dither, bleed_edges


@dataclass
class FrameRect:
    """单帧在原精灵表中的位置与轴心点（y 以图像左上角为原点）。"""
    x: int
    y: int
    width: int
    height: int
    pivot_x: float = 0.5   # 归一化轴心（0~1）
    pivot_y: float = 1.0   # 默认底部中心（脚底基线）


@dataclass
class Frame:
    rect: FrameRect
    image: np.ndarray      # BGRA，紧致裁切
    index: int


class ImageProcessor:
    def __init__(
        self,
        bg_mode: str = "auto",        # auto | chroma | rembg | none
        bg_tolerance: int = 60,       # 色键容差
        palette_size: int = 24,       # 统一调色板颜色数（16~24）
        min_area_ratio: float = 0.005,  # 最小帧面积占比（过滤噪点）
        rembg_fallback: bool = False,   # 色键失败时是否尝试 rembg
        align_mode: str = "feet_center",  # baseline | feet_center | centroid
        harden_alpha: bool = False,       # 边缘硬化（alpha 二值化+去光晕）
        bleed_px: int = 0,                # 边缘外扩像素（防图集采样串色，0=关）
        dither: bool = False,             # Floyd–Steinberg 抖动量化（复古风）
        pixel_size: int = 0,              # 目标像素密度（像素风化，0=关）
        bg_color=None,                    # 手动指定背景色 BGR（绿幕等，None=自动）
        key_method: str = "hsv",          # 色键算法: rgb | hsv | exg
    ):
        self.bg_mode = bg_mode
        self.bg_tolerance = bg_tolerance
        self.key_method = key_method
        self.palette_size = max(2, min(24, palette_size))
        self.min_area_ratio = min_area_ratio
        self.rembg_fallback = rembg_fallback
        self.align_mode = align_mode    # 帧对齐锚点策略
        self.harden_alpha = harden_alpha
        self.bleed_px = max(0, int(bleed_px))
        self.dither = dither
        self.pixel_size = int(pixel_size)
        self.bg_color = list(bg_color) if bg_color else None
        self._locked_palette: list | None = None  # 锁定调色板（BGR 颜色列表）
        self.last_palette: list | None = None     # 最近一次量化结果

    # ------------------------------------------------ 对外主入口

    def process(self, sprite_sheet: Path, out_dir: Path, fps: int = 8) -> list[Path]:
        """处理精灵表，输出帧 PNG + atlas.json，返回生成的文件列表。"""
        sheet = cv2.imread(str(sprite_sheet), cv2.IMREAD_UNCHANGED)
        if sheet is None:
            raise FileNotFoundError(f"无法读取图片: {sprite_sheet}")
        if sheet.ndim == 2:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        if sheet.shape[2] == 4:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_BGRA2BGR)

        bgra = self.remove_background(sheet)
        frames = self.run_pipeline(bgra)
        if not frames:
            raise ValueError("未能从图像中检测到任何帧（请检查背景是否为纯色/绿幕）")

        out_dir.mkdir(parents=True, exist_ok=True)
        name = sprite_sheet.stem
        produced: list[Path] = []
        for f in frames:
            p = out_dir / f"{name}_{f.index:03d}.png"
            cv2.imwrite(str(p), f.image)
            produced.append(p)
        atlas = {
            "name": name,
            "fps": fps,
            "frame_count": len(frames),
            "frames": [asdict(f.rect)
                       | {"file": p.name,
                          "pivotX": round(getattr(f.rect, "pivot_x", 0.5), 4),
                          "pivotY": 1.0,
                          "hull": frame_hull(f.image)}
                       for f, p in zip(frames, produced)],
        }
        jp = out_dir / f"{name}.atlas.json"
        jp.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        produced.append(jp)
        return produced

    # ------------------------------------------------ 完整管线（供 GUI/宏复用）

    def run_pipeline(self, bgra: np.ndarray) -> list[Frame]:
        """裁切 -> 对齐 -> 像素风化 -> 调色板 -> 边缘硬化 -> 外扩；空图返回空列表。"""
        frames = self.auto_crop_frames(bgra)
        if not frames:
            return frames
        self.unify_frames(frames)
        if self.pixel_size > 0:  # 像素风化须在量化前：先收敛颜色块再 KMeans
            from pixelforge.processing.editing import pixelate
            for f in frames:
                f.image = pixelate(f.image, self.pixel_size)
        self.quantize_palette(frames)
        if self.harden_alpha:
            self.harden_alpha_frames(frames)
        if self.bleed_px > 0:
            for f in frames:
                f.image = bleed_edges(f.image, self.bleed_px)
        return frames

    def harden_alpha_frames(self, frames: list[Frame], threshold: int = 128) -> None:
        """边缘硬化：alpha 二值化 + 边界像素颜色吸附最近调色板色（去光晕）。"""
        palette = self._locked_palette or self.last_palette
        for f in frames:
            f.image = harden_alpha_image(f.image, palette, threshold)

    # ------------------------------------------------ 1. 去背景

    def remove_background(self, sheet: np.ndarray) -> np.ndarray:
        """返回 BGRA。auto 模式：四角颜色一致则色键，否则按需 rembg / 保留。"""
        if sheet.ndim == 2:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        if sheet.ndim == 3 and sheet.shape[2] == 4:  # 兼容 BGRA 输入
            sheet = cv2.cvtColor(sheet, cv2.COLOR_BGRA2BGR)
        if self.bg_color:  # 手动指定背景色：跳过四角检测（绿幕等可靠场景）
            return chroma_key(sheet, np.array(self.bg_color, np.float32),
                              self.bg_tolerance, self.key_method)
        mode = self.bg_mode
        if mode == "auto":
            bg = detect_background_color(sheet)
            if bg is not None:
                return chroma_key(sheet, bg, self.bg_tolerance,
                              self.key_method)
            if self.rembg_fallback:
                return self._remove_with_rembg(sheet)
            return cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)  # 背景不规则且不启用 rembg
        if mode == "chroma":
            bg = detect_background_color(sheet)
            if bg is None:
                raise ValueError("未检测到纯色背景，无法色键抠图")
            return chroma_key(sheet, bg, self.bg_tolerance, self.key_method)
        if mode == "rembg":
            return self._remove_with_rembg(sheet)
        return cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)

    @staticmethod
    def _remove_with_rembg(sheet: np.ndarray) -> np.ndarray:
        try:
            from rembg import remove
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("rembg 未安装：pip install rembg") from e
        pil = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        out = remove(pil)
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGRA)

    # ------------------------------------------------ 2. 连通域裁切

    def auto_crop_frames(self, bgra: np.ndarray) -> list[Frame]:
        alpha = bgra[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        min_area = max(16, int(mask.size * self.min_area_ratio))
        labels, stats = self._separate_components(mask, min_area)

        frames: list[Frame] = []
        for i in range(1, stats.shape[0]):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            crop = bgra[y:y + h, x:x + w].copy()
            # 只保留本连通域的像素，清除 bbox 内的邻域噪声
            keep = (labels[y:y + h, x:x + w] == i)
            crop[:, :, 3] = np.where(keep, crop[:, :, 3], 0)
            rect = FrameRect(x=int(x), y=int(y), width=int(w), height=int(h))
            frames.append(Frame(rect=rect, image=crop, index=0))

        frames = self._sort_reading_order(frames)
        for idx, f in enumerate(frames):
            f.index = idx
        return frames

    def _separate_components(self, mask: np.ndarray, min_area: int):
        """对粘连帧渐进式腐蚀分离，返回 (labels, stats)（labels 覆盖原始 mask 像素）。"""
        _, labels0, stats0, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_labels, best_n = labels0, stats0.shape[0]
        best_count = self._valid_count(stats0, min_area)
        kernel = np.ones((3, 3), np.uint8)
        work = mask.copy()
        for _ in range(4):  # 最多腐蚀 4 次
            work = cv2.erode(work, kernel)
            if cv2.countNonZero(work) < min_area * 2:
                break
            _, wl, ws, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
            count = self._valid_count(ws, min_area)
            if count > best_count:
                best_labels, best_n = wl, ws.shape[0]
                best_count = count
            else:
                break
        # 用腐蚀得到的分割 + 原始 mask 重建标签（恢复被腐蚀的形状细节）
        labels = np.zeros_like(mask)
        for i in range(1, best_n):
            labels[(best_labels == i) & (mask > 0)] = i
        return labels, self._recompute_stats(labels, best_n)

    @staticmethod
    def _valid_count(stats: np.ndarray, min_area: int) -> int:
        return int((stats[1:, 4] >= min_area).sum())

    @staticmethod
    def _recompute_stats(labels: np.ndarray, n: int) -> np.ndarray:
        stats = np.zeros((n, 5), dtype=np.int32)
        for i in range(1, n):
            ys, xs = np.where(labels == i)
            if ys.size == 0:
                continue
            stats[i] = [int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                        int(ys.size)]
        return stats

    @staticmethod
    def _sort_reading_order(frames: list[Frame]) -> list[Frame]:
        """按阅读顺序排序：先按行分组（行容差=中位高度的一半），行内按 x。"""
        if not frames:
            return []
        cys = [f.rect.y + f.rect.height / 2 for f in frames]
        med_h = float(np.median([f.rect.height for f in frames]))
        order = np.argsort(cys)
        rows: list[list[Frame]] = []
        for i in order:
            f = frames[i]
            if rows and abs(cys[i] - np.mean([g.rect.y + g.rect.height / 2 for g in rows[-1]])) < med_h / 2:
                rows[-1].append(f)
            else:
                rows.append([f])
        out: list[Frame] = []
        for row in rows:
            out.extend(sorted(row, key=lambda f: f.rect.x))
        return out

    # ------------------------------------------------ 2.5 投影拆分（外部精灵表）

    def split_sheet_projection(self, sheet: np.ndarray,
                               bg_tol: float = 32.0) -> list[Frame]:
        """投影分析拆分：自动检测内容行/列带，支持带背景的外部精灵表。"""
        if sheet.ndim == 2:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        if sheet.shape[2] == 4 and (sheet[:, :, 3] > 0).any():
            mask = sheet[:, :, 3] > 10
        else:
            bg = detect_background_color(sheet[:, :, :3])
            if bg is None:
                bg = sheet[0, 0, :3].astype(np.float32)
            mask = np.linalg.norm(
                sheet[:, :, :3].astype(np.float32) - bg, axis=2) > bg_tol
        if not mask.any():
            return []
        col_bands = _bands(mask.any(axis=0))
        row_bands = _bands(mask.any(axis=1))
        frames: list[Frame] = []
        for ry0, ry1 in row_bands:
            for cx0, cx1 in col_bands:
                sub = mask[ry0:ry1, cx0:cx1]
                if not sub.any():
                    continue
                ys, xs = np.where(sub)
                x0, x1 = cx0 + int(xs.min()), cx0 + int(xs.max()) + 1
                y0, y1 = ry0 + int(ys.min()), ry0 + int(ys.max()) + 1
                frames.append(Frame(
                    rect=FrameRect(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                    image=sheet[y0:y1, x0:x1].copy(), index=0))
        frames = self._sort_reading_order(frames)
        for idx, f in enumerate(frames):
            f.index = idx
        return frames

    # ------------------------------------------------ 3. 基线对齐

    def unify_frames(self, frames: list[Frame], mode: str | None = None,
                     pad: int = 2) -> list[Frame]:
        """把所有帧放到统一尺寸画布上，按锚点对齐，解决角色中心漂移。

        mode: feet_center（推荐：脚底带中心，行走时脚不动）
              centroid（alpha 质心）| baseline（bbox 中心，旧行为）
        帧脚底统一落在同一行，水平按锚点对齐；输出帧尺寸全部一致，
        可直接用于播放与 Unity 导入（等尺寸帧不会因 bbox 差异重新居中）。
        """
        if not frames:
            return frames
        mode = mode or self.align_mode
        infos = []
        for f in frames:
            alpha = f.image[:, :, 3]
            ys, xs = np.where(alpha > 0)
            h, w = f.image.shape[:2]
            bottom = int(ys.max()) if ys.size else h - 1
            if mode == "centroid":
                ax = float(xs.mean()) if xs.size else w / 2
            elif mode == "feet_center":
                # 脚底带（内容底部 15%）的不透明像素中心 —— 行走时脚基本不动
                band_top = max(0, bottom - max(2, int((bottom + 1) * 0.15)))
                bxs = np.where(alpha[band_top:bottom + 1] > 0)[1]
                ax = float(bxs.mean()) if bxs.size else w / 2
            else:  # baseline：bbox 中心（旧行为，受道具/姿态影响）
                ax = w / 2
            infos.append((ax, bottom, h, w))
        W = max(int(2 * max(ax, w - ax)) for ax, _, _, w in infos) + 2 * pad
        feet_line = max(b for _, b, _, _ in infos)
        drop_max = max(h - 1 - b for _, b, h, _ in infos)
        H = feet_line + 1 + drop_max
        out = []
        for f, (ax, bottom, h, w) in zip(frames, infos):
            canvas = np.zeros((H, W, 4), np.uint8)
            dx = int(round(W / 2 - ax))
            dy = feet_line - bottom
            canvas[dy:dy + h, dx:dx + w] = f.image
            f.image = canvas
            f.rect = FrameRect(x=dx, y=dy, width=w, height=h,
                               pivot_x=round((ax + 0.5) / w, 4), pivot_y=1.0)
            out.append(f)
        return out

    @staticmethod
    def align_baseline(frames: list[Frame]) -> int:
        """所有帧底部对齐（脚底基线 = 统一画布最后一行），返回画布高度。"""
        bottoms = []
        for f in frames:
            ys, _ = np.where(f.image[:, :, 3] > 0)
            bottoms.append(int(ys.max()) if ys.size else f.image.shape[0] - 1)
        lowest = max(bottoms)  # 全体最靠下的内容底
        canvas_h = max(f.image.shape[0] + (lowest - b)
                       for f, b in zip(frames, bottoms))
        for f, b in zip(frames, bottoms):
            pad_top = canvas_h - 1 - b  # 顶部下垫，使脚底落在画布最后一行
            f.image = np.pad(f.image, ((pad_top, 0), (0, 0), (0, 0))) \
                if pad_top > 0 else f.image
            if f.image.shape[0] < canvas_h:  # 不足则底部补齐到统一高度
                f.image = np.pad(f.image,
                                 ((0, canvas_h - f.image.shape[0]), (0, 0), (0, 0)))
            f.rect.height = int(f.image.shape[0])
            f.rect.width = int(f.image.shape[1])
        return canvas_h

    # ------------------------------------------------ 4. 统一调色板

    def set_locked_palette(self, colors: list) -> None:
        """锁定调色板（BGR 颜色列表），quantize 时不再 KMeans。"""
        self._locked_palette = [list(map(int, c)) for c in colors] or None

    def _map_locked(self, frames: list[Frame]) -> None:
        centers = np.array(self._locked_palette, dtype=np.float32)
        for f in frames:
            h, w = f.image.shape[:2]
            alpha = f.image[:, :, 3]
            flat = f.image[:, :, :3].reshape(-1, 3).astype(np.float32)
            idx = _nearest_centers(flat, centers)
            f.image = np.dstack(
                [centers[idx].reshape(h, w, 3).astype(np.uint8), alpha])
        self.last_palette = [list(map(int, c)) for c in centers]

    def quantize_palette(self, frames: list[Frame]) -> None:
        """统计所有帧颜色，KMeans 生成统一调色板并映射（锁定时直接映射）。"""
        if self._locked_palette:
            self._map_locked(frames)
            if self.dither and self.last_palette:
                for f in frames:
                    f.image = apply_dither(f.image, self.last_palette)
            return
        samples = []
        for f in frames:
            px = f.image[f.image[:, :, 3] > 0][:, :3]
            if px.size:
                samples.append(px)
        if not samples:
            return
        data = np.concatenate(samples).astype(np.float32)
        if data.shape[0] > 20000:  # 采样加速
            data = data[np.random.default_rng(0).choice(
                data.shape[0], 20000, replace=False)]
        k = min(self.palette_size, int(np.unique(data, axis=0).shape[0]))
        if k <= 1:
            return
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.int32)
        self.last_palette = [list(map(int, c)) for c in centers]

        for f in frames:
            h, w = f.image.shape[:2]
            alpha = f.image[:, :, 3]
            flat = f.image[:, :, :3].reshape(-1, 3).astype(np.float32)
            idx = _nearest_centers(flat, centers)
            f.image = np.dstack([
                centers[idx].reshape(h, w, 3).astype(np.uint8),
                alpha,
            ])
        if self.dither and self.last_palette:
            for f in frames:  # 可选：F-S 抖动变体（复古风）
                f.image = apply_dither(f.image, self.last_palette)

    # ------------------------------------------------ 5. 元数据

    @staticmethod
    def export_atlas_json(frames: list[Frame], path: Path, name: str, fps: int = 8) -> None:
        atlas = {
            "name": name,
            "fps": fps,
            "frame_count": len(frames),
            "frames": [asdict(f.rect) | {"file": f"{name}_{f.index:03d}.png"}
                       for f in frames],
        }
        path.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")


def save_palette(path: Path, colors: list) -> None:
    """调色板存为 JSON（BGR 颜色列表），供跨批次锁定复用。"""
    Path(path).write_text(
        json.dumps({"colors": [list(map(int, c)) for c in colors]},
                   ensure_ascii=False, indent=2), "utf-8")


def load_palette(path: Path) -> list:
    data = json.loads(Path(path).read_text("utf-8"))
    return [list(map(int, c)) for c in data["colors"]]


def _nearest_centers(flat: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty(flat.shape[0], dtype=np.int32)
    step = 1 << 20
    for i in range(0, flat.shape[0], step):
        chunk = flat[i:i + step]
        d = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[i:i + step] = d.argmin(axis=1)
    return out


def detect_background_color(sheet: np.ndarray, sample: int = 10) -> np.ndarray | None:
    """取四角中位数颜色；四角颜色一致（std<12）才视为纯色背景。"""
    h, w = sheet.shape[:2]
    s = min(sample, h // 4, w // 4)
    corners = np.concatenate([
        sheet[:s, :s].reshape(-1, 3), sheet[:s, w - s:].reshape(-1, 3),
        sheet[h - s:, :s].reshape(-1, 3), sheet[h - s:, w - s:].reshape(-1, 3),
    ]).astype(np.float32)
    if corners.std(axis=0).max() > 12.0:
        return None
    return np.median(corners, axis=0)


def despill_green(bgra: np.ndarray, zone_dilate: int = 2,
                  bg: np.ndarray | None = None) -> np.ndarray:
    """去绿镶边（despill），两级压制：

    1) halo 带：前景边缘外扩 zone_dilate 像素内，绿通道一律压回 max(R,B)；
    2) 全图色相带：绿主导（G > max(R,B)+4）且色相接近背景色相的像素
       同样压制——治理人物内部残留的小绿块；远离该色相带的合法绿色
       （如深绿服装且色相偏移）不受影响。
    """
    out = bgra.copy()
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    g_cap = np.maximum(b, r)

    # 1) halo 带（原逻辑）
    solid = (out[:, :, 3] >= 250).astype(np.uint8)
    zone = cv2.dilate(solid, np.ones((zone_dilate * 2 + 1,) * 2, np.uint8))
    halo = (zone > 0) & (out[:, :, 3] > 0)
    out[:, :, 1] = np.where(halo & (g > g_cap), g_cap, g)

    # 2) 全图绿主导 + 色相带限定
    if bg is not None:
        hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV)
        bg_hsv = cv2.cvtColor(
            np.uint8([[bg]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        dh = np.abs(hsv[:, :, 0].astype(np.float32) - bg_hsv[0])
        dh = np.minimum(dh, 180 - dh)
        near_bg_hue = dh < 25
        g2 = out[:, :, 1]
        green_dom = g2 > (np.maximum(out[:, :, 0], out[:, :, 2]) + 4)
        sat_ok = hsv[:, :, 1] > 40
        hit = near_bg_hue & green_dom & sat_ok
        out[:, :, 1] = np.where(hit, g_cap, g2)

        # 3) 内部绿斑清除：色相命中背景且饱和度极高的孤点直接判为背景
        strong_bg = near_bg_hue & (hsv[:, :, 1] > 120) & \
            (np.abs(hsv[:, :, 2].astype(np.float32) - bg_hsv[2]) < 60)
        out[:, :, 3] = np.where(strong_bg, 0, out[:, :, 3]).astype(np.uint8)
    return out


def clean_alpha_specks(bgra: np.ndarray, min_area: int = 6) -> np.ndarray:
    """清除 alpha 中的孤立小噪点（残留半透明绿斑等）：小于 min_area 的
    不透明连通域置为全透明。"""
    out = bgra.copy()
    mask = (out[:, :, 3] > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 0
    return out


def chroma_key_hsv(sheet: np.ndarray, bg: np.ndarray,
                   tolerance: int = 60) -> np.ndarray:
    """HSV 色相键：以背景色相为锚，色相+饱和度双条件判定。

    对绿幕明暗不均、渐变背景比 RGB 欧氏距离稳健得多；despill 同步进行。
    """
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(
        np.uint8([[bg]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    dh = np.abs(hsv[:, :, 0] - bg_hsv[0])
    dh = np.minimum(dh, 180 - dh)                       # 色相环距离
    ds = hsv[:, :, 1] - bg_hsv[1]
    score = np.sqrt(dh ** 2 + (0.6 * ds) ** 2)          # 加权色差
    alpha = np.clip((score - tolerance / 2) /
                    max(tolerance / 2, 1) * 255, 0, 255).astype(np.uint8)
    out = np.dstack([sheet, alpha])
    return clean_alpha_specks(despill_green(out, bg=bg))


def chroma_key_exg(sheet: np.ndarray, bg: np.ndarray,
                   tolerance: int = 60) -> np.ndarray:
    """过绿检测键（Excess Green）：ExG = 2G - R - B。

    对「绿幕比人物亮/暗很多」的场景鲁棒；对背景为非绿纯色自动退化为
    普通色键（ExG 与背景距离联合判定）。
    """
    f = sheet.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    exg_bg = 2.0 * bg[1] - bg[0] - bg[2]
    dist = np.linalg.norm(f - bg.astype(np.float32), axis=2)
    score = np.maximum(dist, (exg - exg_bg) * 1.2 + dist * 0.25)
    alpha = np.clip((score - tolerance / 2) /
                    max(tolerance / 2, 1) * 255, 0, 255).astype(np.uint8)
    out = np.dstack([sheet, alpha])
    return clean_alpha_specks(despill_green(out, bg=bg))


def chroma_key(sheet: np.ndarray, bg: np.ndarray, tolerance: int = 60,
               method: str = "rgb") -> np.ndarray:
    """色键抠图（多方案）：rgb=欧氏距离（快） | hsv=色相键（绿幕首选） |
    exg=过绿检测（明暗不均背景）。

    附带去绿边（despill）：半透明边缘像素绿通道压到 R/B 最大值。
    """
    if method == "hsv":
        return chroma_key_hsv(sheet, bg, tolerance)
    if method == "exg":
        return chroma_key_exg(sheet, bg, tolerance)
    dist = np.linalg.norm(sheet.astype(np.float32) - bg.astype(np.float32), axis=2)
    alpha = np.clip((dist - tolerance / 2) / max(tolerance / 2, 1) * 255, 0, 255)
    alpha = alpha.astype(np.uint8)

    out = sheet.copy()
    spill = (alpha > 0) & (alpha < 230)
    if spill.any():
        b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        g_cap = np.maximum(b, r)  # 绿通道上限
        out[:, :, 1] = np.where(spill & (g > g_cap), g_cap, g)
    return clean_alpha_specks(despill_green(np.dstack([out, alpha]), bg=bg))


# ------------------------------------------------ 图形增强工具

def harden_alpha_image(bgra: np.ndarray, palette: list | None = None,
                       threshold: int = 128) -> np.ndarray:
    """边缘硬化：alpha 二值化（<threshold 全透明）；边界像素颜色吸附最近调色板色。"""
    out = bgra.copy()
    out[:, :, 3] = np.where(out[:, :, 3] >= threshold, 255, 0).astype(np.uint8)
    hard = out[:, :, 3]
    if palette:
        edge = (hard > 0) & (cv2.erode(hard, np.ones((3, 3), np.uint8)) == 0)
        if edge.any():
            centers = np.array(palette, dtype=np.float32)
            flat = out[:, :, :3][edge].astype(np.float32)
            idx = _nearest_centers(flat, centers)
            out[:, :, :3][edge] = centers[idx].reshape(-1, 3).astype(np.uint8)
    return out


def unify_palette_frames(frames: list[np.ndarray], k: int = 24) -> list[np.ndarray]:
    """跨帧统一调色板：全部帧的不透明像素合并 KMeans，再逐帧映射。

    治理逐帧独立抠图导致的「颜色闪烁」（同色像素帧间微差）。
    """
    if not frames:
        return frames
    samples = []
    for f in frames:
        px = f[f[:, :, 3] > 0][:, :3]
        if px.size:
            samples.append(px)
    if not samples:
        return frames
    data = np.concatenate(samples).astype(np.float32)
    k = min(k, int(np.unique(data, axis=0).shape[0]))
    if k <= 1:
        return [f.copy() for f in frames]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        alpha = f[:, :, 3]
        flat = f[:, :, :3].reshape(-1, 3).astype(np.float32)
        idx = _nearest_centers(flat, centers)
        out.append(np.dstack([
            centers[idx].reshape(h, w, 3).astype(np.uint8), alpha]))
    return out


def align_ground_line(frames: list[np.ndarray], max_shift: int = 6
                      ) -> list[np.ndarray]:
    """接地基准对齐：以「最靠下的内容底边」为地面线，各帧底边向地面线
    对齐，但单帧平移量限幅（max_shift），保留腾空帧的垂直位移语义。

    治理帧间脚底位置随机微差造成的垂直抖动。
    """
    if not frames:
        return frames
    bottoms = []
    for f in frames:
        ys, _ = np.where(f[:, :, 3] > 0)
        bottoms.append(int(ys.max()) if ys.size else f.shape[0] - 1)
    ground = max(bottoms)  # 地面线 = 最靠下的底边
    out = []
    for f, b in zip(frames, bottoms):
        shift = min(max_shift, ground - b)  # 只允许上移对齐（限幅）
        if shift <= 0:
            out.append(f)
            continue
        h, w = f.shape[:2]
        canvas = np.zeros_like(f)
        canvas[:h - shift] = f[shift:]  # 上移 shift 像素
        out.append(canvas)
    return out


def smooth_horizontal(frames: list[np.ndarray], max_shift: int = 4
                      ) -> list[np.ndarray]:
    """水平质心限幅平滑：帧间内容水平中心微差收敛（限幅保留大位移）。

    治理逐帧水平抖动；跑动等大水平位移（>max_shift）不受影响。
    """
    if len(frames) < 2:
        return list(frames)
    centers = []
    for f in frames:
        ys, xs = np.where(f[:, :, 3] > 0)
        centers.append(float((xs.min() + xs.max()) / 2) if xs.size
                       else f.shape[1] / 2)
    ref = centers[0]
    out = []
    for f, cx in zip(frames, centers):
        shift = cx - ref
        if abs(shift) < 0.5:
            out.append(f)
            continue
        shift = int(round(np.clip(shift, -max_shift, max_shift)))
        h, w = f.shape[:2]
        canvas = np.zeros_like(f)
        if shift > 0:   # 内容左移 shift
            canvas[:, :w - shift] = f[:, shift:]
        else:           # 内容右移
            canvas[:, -shift:] = f[:, :w + shift]
        out.append(canvas)
    return out


def strip_ground_lines(bgra: np.ndarray, coverage: float = 0.72,
                       max_thick: int = 14) -> np.ndarray:
    """清除横贯网格的地面线/阴影线（AI 精灵表常见副产物）。

    判定：某行不透明像素横向占比 > coverage（角色不会横跨大半张图，
    地面线会），且连续满足的行带厚度 <= max_thick。尘埃等小团不受影响。
    """
    out = bgra.copy()
    alpha = out[:, :, 3]
    h = alpha.shape[0]
    row_hit = (alpha > 0).mean(axis=1) > coverage
    y = 0
    while y < h:
        if row_hit[y]:
            y2 = y
            while y2 + 1 < h and row_hit[y2 + 1]:
                y2 += 1
            if y2 - y + 1 <= max_thick:
                out[y:y2 + 1, :, 3] = 0
            y = y2 + 1
        else:
            y += 1
    return out


def split_grid_even(bgra: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    """把网格图等分为 rows x cols 个格子（不紧致裁切，保留帧内位置）。"""
    h, w = bgra.shape[:2]
    ch, cw = h // rows, w // cols
    return [bgra[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy()
            for r in range(rows) for c in range(cols)]


def normalize_grid_cells(cells: list[np.ndarray], align: str = "free",
                         pad: int = 14) -> list[np.ndarray]:
    """网格等分帧的归一化：保留 AI 帧内相对位置，仅水平居中 + 统一画布。

    align="free"   垂直原样保留（跳跃等腾空动作：帧内垂直位移即动画语义）
    align="bottom" 内容底对齐（贴地动作：idle/walk）
    水平统一居中（原地动作）；所有帧输出同一尺寸（格尺寸 + 2*pad），
    不做逐帧缩放——帧间相对大小关系严格保留。
    """
    if not cells:
        return []
    H, W = cells[0].shape[:2]
    FH, FW = H + 2 * pad, W + 2 * pad
    out = []
    for cell in cells:
        canvas = np.zeros((FH, FW, 4), np.uint8)
        ys, xs = np.where(cell[:, :, 3] > 0)
        dy, dx = pad, pad
        if ys.size:
            if align == "bottom":
                dy = (FH - pad) - (ys.max() + 1)   # 内容底贴到画布底-pad
            cx = (xs.min() + xs.max() + 1) / 2
            dx = int(round(FW / 2 - cx))           # 内容水平中心 -> 画布中心
        y0d, x0d = max(0, dy), max(0, dx)
        y0s, x0s = max(0, -dy), max(0, -dx)
        hh = min(FH - y0d, H - y0s)
        ww = min(FW - x0d, W - x0s)
        if hh > 0 and ww > 0:
            canvas[y0d:y0d + hh, x0d:x0d + ww] = \
                cell[y0s:y0s + hh, x0s:x0s + ww]
        out.append(canvas)
    return out


def normalize_height(frames: list[np.ndarray],
                     target_h: int | None = None) -> list[np.ndarray]:
    """角色高度归一化：裁内容 bbox -> 等比缩放到统一高度 -> 底对齐居中。

    解决 AI 逐帧生成「角色大小/位置随机」问题（动作序列必备）。
    target_h 缺省时取所有帧的最大内容高度。
    """
    if not frames:
        return []
    contents = []
    for f in frames:
        ys, xs = np.where(f[:, :, 3] > 0)
        if ys.size:
            f = f[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        contents.append(f)
    target = max(8, target_h or max(c.shape[0] for c in contents))
    norm = []
    for c in contents:
        s = target / max(1, c.shape[0])
        nw = max(1, int(c.shape[1] * s))
        r = cv2.resize(c, (nw, target), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target, max(target // 3, nw), 4), np.uint8)
        dx = (canvas.shape[1] - nw) // 2
        canvas[:, dx:dx + nw] = r
        norm.append(canvas)
    W = max(f.shape[1] for f in norm)
    out = []
    for f in norm:
        canvas = np.zeros((f.shape[0], W, 4), np.uint8)
        dx = (W - f.shape[1]) // 2
        canvas[:, dx:dx + f.shape[1]] = f
        out.append(canvas)
    return out


def frame_hull(image: np.ndarray, epsilon: float = 1.5) -> list[float]:
    """帧轮廓多边形，扁平 [x0,y0,x1,y1,...]（帧画布内、顶左原点）。

    扁平格式兼容 Unity JsonUtility（不支持嵌套数组），导入器据此
    生成 PolygonCollider2D 物理形状。
    """
    mask = (image[:, :, 3] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    pts = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    return [float(v) for p in pts for v in p]


def extract_palette(image: np.ndarray, k: int = 16) -> list:
    """从 BGRA/BGR 图提取主色板（忽略透明像素），返回 BGR 颜色列表。

    k-means 聚类主色，采样上限 20000 像素加速；唯一色数少于 k 时直接返回全部唯一色。
    """
    if image is None or image.ndim < 3:
        raise ValueError("需要彩色图像（BGR/BGRA）")
    bgr = image[:, :, :3]
    if image.shape[2] == 4:
        mask = image[:, :, 3] > 0
    else:
        mask = np.ones(bgr.shape[:2], dtype=bool)
    px = bgr[mask]
    if px.size == 0:
        raise ValueError("画面全透明，无可提取的颜色")
    data = px.astype(np.float32)
    if data.shape[0] > 20000:  # 采样加速
        data = data[np.random.default_rng(0).choice(
            data.shape[0], 20000, replace=False)]
    uniq = np.unique(data.astype(np.uint8), axis=0)
    if uniq.shape[0] <= max(1, k):
        return [list(map(int, c)) for c in uniq]
    k = max(2, min(int(k), 32))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return [list(map(int, c)) for c in centers.astype(np.int32)]


def remap_palette_colors(image: np.ndarray, palette: list,
                         mapping: dict) -> np.ndarray:
    """换色引擎：像素吸附到最近调色板色，再按 {源索引: (B,G,R)} 替换目标色。

    已按调色板量化的帧上，未映射颜色保持不变（恒等映射）。
    """
    centers = np.array(palette, dtype=np.float32)
    targets = centers.copy()
    for src, dst in mapping.items():
        targets[int(src)] = np.array(dst, dtype=np.float32)
    out = image.copy()
    h, w = out.shape[:2]
    flat = out[:, :, :3].reshape(-1, 3).astype(np.float32)
    idx = _nearest_centers(flat, targets)
    out[:, :, :3] = targets[idx].reshape(h, w, 3).astype(np.uint8)
    return out


def _bands(occ: np.ndarray) -> list[tuple[int, int]]:
    """布尔序列 -> 连续 True 段 [(start, end)...]。"""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(bool(x) for x in occ):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(occ)))
    return out


# ------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PixelForge 精灵表后处理（阶段2）")
    ap.add_argument("input", help="精灵表 PNG 路径")
    ap.add_argument("-o", "--out", default="outputs/processed", help="输出目录")
    ap.add_argument("--bg", default="auto", choices=["auto", "chroma", "rembg", "none"])
    ap.add_argument("--palette", type=int, default=24, help="调色板颜色数 (16~24)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--rembg-fallback", action="store_true")
    ap.add_argument("--harden", action="store_true", help="边缘硬化（去光晕）")
    ap.add_argument("--bleed", type=int, default=0, help="边缘外扩像素（防图集串色）")
    ap.add_argument("--dither", action="store_true", help="F-S 抖动量化（复古风）")
    ap.add_argument("--pixel", type=int, default=0,
                    help="目标像素密度（像素风化，0=关闭）")
    args = ap.parse_args(argv)

    proc = ImageProcessor(bg_mode=args.bg, palette_size=args.palette,
                          rembg_fallback=args.rembg_fallback,
                          harden_alpha=args.harden, bleed_px=args.bleed,
                          dither=args.dither, pixel_size=args.pixel)
    files = proc.process(Path(args.input), Path(args.out), fps=args.fps)
    print(f"处理完成，输出 {len(files)} 个文件:")
    for p in files:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
