"""阶段2 后处理管线测试（合成精灵表，无需 ComfyUI）。

用法：python tests/test_processing.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.image_processor import ImageProcessor

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp"


def make_sheet(path: Path) -> None:
    """造一张 2x2 绿幕精灵表：4 个不同高度/位置的纯色小人，帧间距不均、底部不齐。"""
    cell = 128
    sheet = np.full((cell * 2, cell * 2, 3), (60, 180, 60), np.uint8)  # 绿幕

    def draw(cx, base_y, body_w, body_h, color):
        head_r = body_w // 2
        cv2.circle(sheet, (cx, base_y - body_h), head_r, color, -1)
        cv2.rectangle(sheet, (cx - body_w // 2, base_y - body_h + head_r),
                      (cx + body_w // 2, base_y), color, -1)

    # 顺序：左上、右上、左下、右下 —— 底部 y 各不相同（测试基线对齐）
    draw(40, 96, 24, 40, (0, 0, 220))      # 红人
    draw(180, 88, 20, 55, (220, 120, 0))   # 蓝人(BGR: 橙)
    draw(30, 230, 22, 48, (0, 200, 220))   # 黄人
    draw(170, 240, 26, 30, (200, 0, 200))  # 紫人
    cv2.imwrite(str(path), sheet)
    return sheet


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    sheet_path = TMP / "synth_sheet.png"
    make_sheet(sheet_path)

    proc = ImageProcessor(bg_mode="auto", palette_size=24)
    files = proc.process(sheet_path, TMP / "out")

    out = TMP / "out"
    pngs = sorted(out.glob("synth_sheet_*.png"))
    assert len(pngs) == 4, f"应裁出 4 帧，实际 {len(pngs)}"
    print(f"[OK] 连通域裁切: {len(pngs)} 帧, 顺序 {[p.name for p in pngs]}")

    # 抠图检查：不透明像素中不应残留接近绿幕的颜色
    bg = np.array([60, 180, 60], np.float32)
    for p in pngs:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert im.shape[2] == 4
        opaque = im[im[:, :, 3] > 128][:, :3].astype(np.float32)
        assert opaque.size, f"{p.name} 没有不透明像素"
        near_bg = (np.linalg.norm(opaque - bg, axis=1) < 45).mean()
        assert near_bg < 0.01, f"{p.name} 残留绿幕像素 {near_bg:.2%}"
    print(f"[OK] 绿幕抠除: 不透明像素中无绿幕残留")

    # 基线对齐：所有帧最底行应有像素（脚底在最后一行）
    for p in pngs:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert im[:, :, 3].max() > 0
        assert (im[-1, :, 3] > 0).any(), f"{p.name} 底部未对齐"
    print("[OK] 基线对齐: 所有帧脚底位于画布底部")

    # 调色板：内容色不超过 24 种
    for p in pngs:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        colors = np.unique(im[im[:, :, 3] > 0][:, :3].reshape(-1, 3), axis=0)
        assert len(colors) <= 24, f"{p.name} 颜色数 {len(colors)} 超限"
    print("[OK] 统一调色板: 每帧内容色 <= 24")

    # atlas JSON
    atlas = json.loads((out / "synth_sheet.atlas.json").read_text("utf-8"))
    assert atlas["frame_count"] == 4 and len(atlas["frames"]) == 4
    fr = atlas["frames"]
    assert fr[0]["pivot_y"] == 1.0 and fr[0]["pivot_x"] == 0.5
    # 统一画布：PNG 尺寸一致；rect 为内容 bbox（各帧高度可不同），都在画布内
    png_shapes = {cv2.imread(str(out / f["file"]), cv2.IMREAD_UNCHANGED).shape[:2]
                  for f in fr}
    assert len(png_shapes) == 1, f"输出帧应同尺寸: {png_shapes}"
    H, W = next(iter(png_shapes))
    for f in fr:
        assert 0 <= f["x"] and f["x"] + f["width"] <= W
        assert 0 <= f["y"] and f["y"] + f["height"] <= H

    def has_color(i: int, bgr) -> bool:
        im = cv2.imread(str(out / fr[i]["file"]), cv2.IMREAD_UNCHANGED)
        c = np.array(bgr, dtype=int)
        return bool(((np.abs(im[:, :, :3].astype(int) - c).sum(axis=2) < 90)
                     & (im[:, :, 3] > 0)).any())

    assert has_color(0, (0, 0, 220)), "#0 应含红色小人"
    assert has_color(3, (200, 0, 200)), "#3 应含紫色小人"
    print(f"[OK] atlas.json: 统一画布 {W}x{H}, 阅读顺序保持")

    # CLI 端到端
    r = subprocess.run(
        [sys.executable, "-m", "pixelforge.processing.image_processor",
         str(sheet_path), "-o", str(TMP / "cli_out"), "--bg", "chroma"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, f"CLI 失败: {r.stderr}"
    assert (TMP / "cli_out" / "synth_sheet.atlas.json").is_file()
    print("[OK] CLI 端到端运行成功")

    print("\n阶段2 全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
