"""调色板锁定测试：跨批次颜色一致性。

场景：图 A 量化后锁定调色板；图 B（不同颜色构成）用锁定调色板量化
→ B 的所有输出颜色必须是 A 调色板中的颜色（精确最近邻映射）。

用法：python tests/test_palette_lock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.image_processor import (
    ImageProcessor,
    load_palette,
    save_palette,
)

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def make_sheet_a(path: Path) -> None:
    """绿幕 256x128：红 + 黄绿 两个小人（与 test_alignment 同布局）。"""
    sheet = np.full((128, 256, 3), (60, 180, 60), np.uint8)
    cv2.rectangle(sheet, (20, 40), (80, 100), (0, 0, 220), -1)
    cv2.rectangle(sheet, (80, 55), (160, 70), (0, 0, 220), -1)
    cv2.rectangle(sheet, (170, 40), (230, 100), (0, 200, 220), -1)
    cv2.imwrite(str(path), sheet)


def make_sheet_b(path: Path) -> None:
    """另一张：蓝 + 品红 小人（颜色构成与 A 完全不同）。"""
    sheet = np.full((128, 256, 3), (60, 180, 60), np.uint8)
    cv2.rectangle(sheet, (20, 40), (80, 100), (220, 120, 0), -1)
    cv2.rectangle(sheet, (80, 55), (160, 70), (220, 120, 0), -1)
    cv2.rectangle(sheet, (170, 40), (230, 100), (200, 0, 200), -1)
    cv2.imwrite(str(path), sheet)


def output_colors(frames) -> set:
    colors = set()
    for f in frames:
        px = f.image[f.image[:, :, 3] > 0][:, :3]
        colors |= {tuple(p) for p in px}
    return colors


def run_pipeline(proc: ImageProcessor, sheet: Path):
    bgra = proc.remove_background(cv2.imread(str(sheet)))
    frames = proc.auto_crop_frames(bgra)
    frames = proc.unify_frames(frames, proc.align_mode)
    proc.quantize_palette(frames)
    return frames


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    a_path, b_path = TMP / "pal_a.png", TMP / "pal_b.png"
    make_sheet_a(a_path)
    make_sheet_b(b_path)

    # 1. 图 A：常规量化 -> 记录调色板
    proc_a = ImageProcessor()
    frames_a = run_pipeline(proc_a, a_path)
    pal = proc_a.last_palette
    assert pal and 2 <= len(pal) <= 24, f"应记录调色板: {len(pal or [])} 色"
    print(f"[OK] 图 A 量化: 记录调色板 {len(pal)} 色")

    # 2. 保存/加载 调色板文件
    pf = TMP / "locked.json"
    save_palette(pf, pal)
    loaded = load_palette(pf)
    assert loaded == pal, "调色板文件往返应无损"
    print(f"[OK] 调色板文件保存/加载无损: {pf.name}")

    # 3. 图 B：锁定 A 的调色板量化 -> 输出颜色 ⊆ A 调色板
    proc_b = ImageProcessor()
    proc_b.set_locked_palette(loaded)
    assert proc_b._locked_palette == loaded
    frames_b = run_pipeline(proc_b, b_path)
    colors_b = output_colors(frames_b)
    pal_set = {tuple(p) for p in loaded}
    off = [c for c in colors_b if c not in pal_set]
    assert not off, f"锁定量化后出现调色板外颜色: {off[:4]}"
    print(f"[OK] 锁定量化: 图 B 输出 {len(colors_b)} 色全部 ∈ 图 A 调色板（跨批次一致）")

    # 4. 对照：不锁定时图 B 会产生自己的 KMeans 颜色（含调色板外色）
    proc_c = ImageProcessor()
    frames_c = run_pipeline(proc_c, b_path)
    colors_c = output_colors(frames_c)
    off_c = [c for c in colors_c if c not in pal_set]
    assert off_c, "未锁定的对照应有调色板外颜色"
    print(f"[OK] 对照: 未锁定时出现 {len(off_c)} 个调色板外颜色（证明锁定的必要性）")

    # 5. 边界：空锁定列表等效未锁定（set_locked_palette 空 -> None）
    proc_d = ImageProcessor()
    proc_d.set_locked_palette([])
    assert proc_d._locked_palette is None
    print("[OK] 边界: 空调色板列表等效未锁定")

    print("\n调色板锁定全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
