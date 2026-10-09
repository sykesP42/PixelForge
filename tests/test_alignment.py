"""帧对齐策略测试：验证脚底中心对齐解决"角色中心不一致"。

合成场景：两个同身体小人，A 带向右伸出 80px 的长枪（道具导致 bbox 不对称）。
- feet_center 模式：两帧身体在统一画布上应完全对齐（差异 <= 1px）
- baseline 模式（旧行为）：身体中心漂移 >= 20px（证明问题存在）

用法：python tests/test_alignment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.image_processor import ImageProcessor

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def make_asymmetric_sheet(path: Path) -> None:
    """绿幕 256x128：A(红, 带长枪) x 20..160，B(黄绿) x 150..210，同高 40..100。"""
    sheet = np.full((128, 256, 3), (60, 180, 60), np.uint8)
    cv2.rectangle(sheet, (20, 40), (80, 100), (0, 0, 220), -1)     # A 身体
    cv2.rectangle(sheet, (80, 55), (160, 70), (0, 0, 220), -1)     # A 长枪（向右）
    cv2.rectangle(sheet, (170, 40), (230, 100), (0, 200, 220), -1)  # B 身体
    cv2.imwrite(str(path), sheet)


def band_center(img: np.ndarray) -> float:
    """脚底带（底部 15%）不透明像素的 x 中心 —— 播放时的视觉锚点。"""
    a = img[:, :, 3]
    ys, xs = np.where(a > 0)
    bottom = int(ys.max())
    band_top = max(0, bottom - max(2, int((bottom + 1) * 0.15)))
    bxs = np.where(a[band_top:bottom + 1] > 0)[1]
    return float(bxs.mean()) if bxs.size else -1.0


def main() -> int:
    sheet = TMP / "align_sheet.png"
    TMP.mkdir(parents=True, exist_ok=True)
    make_asymmetric_sheet(sheet)

    proc = ImageProcessor()
    bgra = proc.remove_background(cv2.imread(str(sheet)))
    frames = proc.auto_crop_frames(bgra)
    assert len(frames) == 2, f"应裁出 2 帧, 实际 {len(frames)}"

    # 1. feet_center：统一画布 + 身体对齐
    uni = proc.unify_frames(frames, "feet_center")
    assert len({f.image.shape for f in uni}) == 1, "统一画布后帧尺寸应一致"
    xs = [band_center(f.image) for f in uni]
    assert abs(xs[0] - xs[1]) <= 1.0, f"脚底中心对齐后应一致: {xs}"
    W = uni[0].image.shape[1]
    assert abs(xs[0] - W / 2) <= 1.0, "锚点应位于统一画布水平中心"
    print(f"[OK] 脚底中心对齐: 两帧锚点 {xs} (画布宽 {W}, 居中)")

    # 2. baseline（旧行为）：证明漂移真实存在
    frames2 = proc.auto_crop_frames(bgra)
    base = proc.unify_frames(frames2, "baseline")
    xs2 = [band_center(f.image) for f in base]
    drift = abs(xs2[0] - xs2[1])
    assert drift >= 20, f"基线模式应存在 >=20px 漂移, 实际 {drift}"
    print(f"[OK] 旧行为对照: bbox 中心对齐导致身体漂移 {drift:.1f}px（已被新模式消除）")

    # 3. centroid 模式：受道具影响应有偏移（介于两者之间）
    frames3 = proc.auto_crop_frames(bgra)
    cen = proc.unify_frames(frames3, "centroid")
    xs3 = [band_center(f.image) for f in cen]
    drift3 = abs(xs3[0] - xs3[1])
    assert drift3 >= 5, f"质心模式受长枪影响应有偏移: {drift3}"
    print(f"[OK] 质心模式对照: 漂移 {drift3:.1f}px（不对称道具下的已知局限）")

    # 4. 端到端 process()：输出帧同尺寸 + atlas rect 落在统一画布内
    out = TMP / "align_out"
    files = proc.process(sheet, out)
    pngs = [f for f in files if f.suffix == ".png"]
    shapes = {cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[:2] for p in pngs}
    assert len(shapes) == 1, f"process 输出帧应同尺寸: {shapes}"
    atlas = json.loads((out / "align_sheet.atlas.json").read_text("utf-8"))
    for fr in atlas["frames"]:
        assert 0 <= fr["x"] and fr["x"] + fr["width"] <= W
        assert 0 <= fr["y"] and fr["y"] + fr["height"] <= uni[0].image.shape[0]
    print("[OK] process 端到端: 等尺寸帧 + atlas rect 在画布内")

    print("\n对齐策略全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
