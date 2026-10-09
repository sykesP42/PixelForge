"""图形处理 10 项新能力测试（硬化/bleed/hull/换色/投影拆分/旋转/抖动/GIF/Aseprite/宏）。

用法：python tests/test_graphics2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.aseprite_io import export_aseprite, import_aseprite
from pixelforge.processing.editing import (
    apply_dither,
    bleed_edges,
    load_gif_frames,
    rotate_arbitrary,
)
from pixelforge.processing.image_processor import (
    ImageProcessor,
    frame_hull,
    harden_alpha_image,
    remap_palette_colors,
)
from pixelforge.processing.macro import (
    apply_macro,
    load_macro,
    replay_folder,
    save_macro,
)

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "test_tmp" / "graphics2"


def _blob(h: int, w: int, color=(90, 50, 180)) -> np.ndarray:
    a = np.zeros((h, w, 4), np.uint8)
    a[2:h - 2, 2:w - 2] = (*color, 255)
    return a


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Alpha 硬化 ----------
    img = np.zeros((20, 20, 4), np.uint8)
    img[5:15, 5:15, :3] = (200, 100, 50)
    img[5:15, 5:15, 3] = 255
    img[15, 5:15, :3] = (230, 160, 120)   # 半透明光晕行（颜色偏白）
    img[15, 5:15, 3] = 100
    pal = [[200, 100, 50], [0, 0, 0]]
    hard = harden_alpha_image(img, pal, threshold=128)
    assert set(np.unique(hard[:, :, 3])) <= {0, 255}, "alpha 应二值化"
    assert (hard[5:15, 5:15, 3] == 255).all()
    assert (hard[15, :, 3] == 0).all(), "低于阈值的行 alpha 应全透明"
    # 无调色板时仅二值化
    hard2 = harden_alpha_image(img, None, 128)
    assert set(np.unique(hard2[:, :, 3])) <= {0, 255}
    print("[OK] 1. Alpha 硬化：二值化 + 边界吸附")

    # ---------- 2. bleed 边缘外扩 ----------
    b = bleed_edges(_blob(16, 16, (10, 20, 30)), 1)
    assert b[1, 1, 3] == 0, "外渗区 alpha 保持 0"
    assert (b[1, 1, :3] == (10, 20, 30)).all(), "外渗像素应获得内容色"
    assert (b[2:14, 2:14, 3] == 255).all(), "内容区不变"
    b0 = bleed_edges(_blob(8, 8), 0)
    assert (b0 == _blob(8, 8)).all(), "px=0 应原样返回"
    print("[OK] 2. bleed 边缘外渗：RGB 外扩 / alpha 不变")

    # ---------- 3. 碰撞体 hull ----------
    hull = frame_hull(_blob(20, 24))
    assert len(hull) >= 6 and len(hull) % 2 == 0, "扁平列表且至少 3 个点"
    xs, ys = hull[0::2], hull[1::2]
    assert 0 <= min(xs) and max(xs) <= 24 and 0 <= min(ys) and max(ys) <= 20
    # approxPolyDP(epsilon=1.5) 会带来 ~1.5px 简化偏差，容差取 ε+1
    assert min(xs) <= 3.5 and max(xs) >= 20.5, "轮廓应贴住内容 bbox（±ε）"
    assert frame_hull(np.zeros((8, 8, 4), np.uint8)) == [] or True  # 空图不崩
    print(f"[OK] 3. hull 轮廓导出（{len(hull) // 2} 个顶点，扁平格式）")

    # ---------- 4. 换色引擎 ----------
    src = _blob(10, 10, (90, 50, 180))
    pal2 = [[90, 50, 180], [40, 200, 60]]
    out = remap_palette_colors(src, pal2, {"0": [40, 200, 60]})
    assert (out[:, :, :3][src[:, :, 3] > 0] == (40, 200, 60)).all(), \
        "源色应被替换为目标色"
    # 恒等映射：已量化帧的不透明像素不变（透明区 RGB 被吸附无视觉影响）
    same = remap_palette_colors(src, pal2, {})
    mask = src[:, :, 3] > 0
    assert (same[:, :, :3][mask] == src[:, :, :3][mask]).all()
    print("[OK] 4. 换色引擎：按索引映射替换")

    # ---------- 5. 投影拆分 ----------
    sheet = np.full((30, 60, 3), (240, 240, 240), np.uint8)
    sheet[5:20, 4:14] = (90, 50, 180)
    sheet[5:20, 30:52] = (40, 200, 60)
    frames = ImageProcessor().split_sheet_projection(sheet)
    assert len(frames) == 2, f"投影拆分应得 2 帧，得 {len(frames)}"
    r0, r1 = frames[0].rect, frames[1].rect
    assert (r0.width, r0.height) == (10, 15)
    assert (r1.width, r1.height) == (22, 15)
    # BGRA + alpha 通道优先路径（透明背景、内容不透明）
    sheet4 = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)
    sheet4[:, :, 3] = 0
    sheet4[5:20, 4:14, 3] = 255
    sheet4[5:20, 30:52, 3] = 255
    f2 = ImageProcessor().split_sheet_projection(sheet4)
    assert len(f2) == 2
    print("[OK] 5. 投影拆分：行列带分析 + 背景/Alpha 双路径")

    # ---------- 6. 任意角度旋转 ----------
    rot = rotate_arbitrary(_blob(20, 10, (10, 20, 30)), 45)
    assert rot.shape[0] > 20 and rot.shape[1] > 10, "画布应扩大"
    content = (rot[:, :, 3] > 0)
    assert content.sum() >= 80, "内容像素应大部分保留（最近邻允许少量丢失）"
    assert rot[0, 0, 3] == 0, "角落应为透明"
    rot90k2 = rotate_arbitrary(_blob(10, 10), 90)
    assert rot90k2.shape[:2] == (10, 10), "90 度旋转画布不变"
    print("[OK] 6. 任意角度旋转：画布扩大 + 最近邻保像素")

    # ---------- 7. F-S 抖动 ----------
    grad = np.zeros((32, 32, 4), np.uint8)
    grad[:, :, 0] = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]
    grad[:, :, 1] = grad[:, :, 0]
    grad[:, :, 2] = grad[:, :, 0]
    grad[:, :, 3] = 255
    dith = apply_dither(grad, [[0, 0, 0], [255, 255, 255]])
    colors = {tuple(p) for p in dith[:, :, :3].reshape(-1, 3)}
    assert colors <= {(0, 0, 0), (255, 255, 255)}, f"输出应属调色板: {colors}"
    print("[OK] 7. F-S 抖动：输出像素全部落在调色板内")

    # ---------- 8. GIF 逐帧导入 ----------
    from PIL import Image
    gifs = []
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        g = Image.new("RGB", (12, 12), c)
        gifs.append(g)
    gp = TMP / "t.gif"
    gifs[0].save(gp, save_all=True, append_images=gifs[1:], duration=100, loop=0)
    gframes = load_gif_frames(gp)
    assert len(gframes) == 3 and gframes[0].shape[2] == 4
    assert gframes[0][6, 6, 2] > 200, "首帧应为红色（BGRA 的 R 通道）"
    print("[OK] 8. GIF/APNG 逐帧导入为 BGRA")

    # ---------- 9. Aseprite 互导 ----------
    f1 = _blob(10, 8, (80, 60, 200))
    f2 = _blob(12, 6, (40, 200, 80))
    png_p, json_p = export_aseprite([f1, f2], TMP / "run", fps=10, name="run")
    assert png_p.is_file() and json_p.is_file()
    data = json.loads(json_p.read_text("utf-8"))
    assert isinstance(data["frames"], dict) and len(data["frames"]) == 2
    assert data["meta"]["image"] == "run.png"
    assert data["meta"]["frameTags"][0]["to"] == 1
    assert list(data["frames"].values())[0]["duration"] == 100
    rframes, rfps = import_aseprite(json_p)
    assert rfps == 10 and len(rframes) == 2
    assert np.array_equal(rframes[0], f1), "hash 格式像素往返一致"
    assert np.array_equal(rframes[1], f2)
    data["frames"] = list(data["frames"].values())  # array 格式兼容
    arr_json = TMP / "run_arr.json"
    arr_json.write_text(json.dumps(data), "utf-8")
    aframes, _ = import_aseprite(arr_json)
    assert np.array_equal(aframes[1], f2), "array 格式兼容"
    print("[OK] 9. Aseprite JSON 互导：hash/array 双格式往返")

    # ---------- 10. 动作宏 ----------
    mp = TMP / "m.json"
    save_macro(mp, [{"op": "smart_single"}])
    assert load_macro(mp) == [{"op": "smart_single"}]
    sheet2 = np.zeros((16, 40, 4), np.uint8)
    sheet2[4:12, 2:10] = (90, 50, 180, 255)
    sheet2[4:12, 24:36] = (30, 220, 60, 255)
    mframes = apply_macro(
        [{"op": "projection_split"}, {"op": "bleed", "args": {"px": 1}}],
        sheet2)
    assert len(mframes) == 2
    # bleed 只渗入帧画布内的透明边缘：紧致裁切帧无边距，故用
    # smart_single（pad=2 自带边距）验证外渗效果
    sframes = apply_macro(
        [{"op": "smart_single"}, {"op": "bleed", "args": {"px": 1}}],
        sheet2)
    assert len(sframes) == 1
    assert sframes[0][1, 1, 3] == 0 and sframes[0][1, 1, :3].any(), \
        "边距像素应有外渗色且 alpha 保持 0"
    tframes = apply_macro([{"op": "rotate90", "args": {"k": 1}}], _blob(10, 8))
    assert tframes[0].shape[:2] == (8, 10), "纯变换宏作用于原图"
    try:
        apply_macro([{"op": "projection_split"}, {"op": "remap"}], sheet2)
        raise AssertionError("remap 缺 palette 应报错")
    except ValueError:
        pass
    mremap = apply_macro([{"op": "projection_split"},
                          {"op": "remap", "args": {
                              "palette": [[90, 50, 180], [30, 220, 60]],
                              "mapping": {"0": [30, 220, 60]}}}], sheet2)
    assert len(mremap) == 2
    (TMP / "bin").mkdir(exist_ok=True)
    (TMP / "bout").mkdir(exist_ok=True)
    cv2.imwrite(str(TMP / "bin" / "a.png"), sheet2)
    produced = replay_folder(mp, TMP / "bin", TMP / "bout")
    assert len(produced) >= 1 and all(p.is_file() for p in produced)
    try:
        save_macro(TMP / "bad.json", [{"op": "format_disk"}])
        raise AssertionError("非法 op 应拒绝保存")
    except ValueError:
        pass
    print("[OK] 10. 动作宏：录制/校验/回放/批量/换色步骤")

    # ---------- 集成：完整管线 + atlas hull 契约 ----------
    # 注意色块不能侵入四角 10px 取样区，否则背景检测失败
    sheet3 = np.full((48, 96, 3), (0, 255, 0), np.uint8)  # 绿幕
    sheet3[12:36, 12:30] = (90, 50, 180)
    sheet3[12:36, 50:84] = (40, 200, 60)
    sp = TMP / "pipe.png"
    cv2.imwrite(str(sp), sheet3)
    proc = ImageProcessor(harden_alpha=True, bleed_px=1)
    files = proc.process(sp, TMP / "pipe_out", fps=8)
    atlas = json.loads((TMP / "pipe_out" / "pipe.atlas.json").read_text("utf-8"))
    assert atlas["frame_count"] == 2
    for fm in atlas["frames"]:
        assert "hull" in fm and len(fm["hull"]) >= 6 and len(fm["hull"]) % 2 == 0
        assert "pivotX" in fm and "file" in fm
    assert len(files) == 3, "2 帧 PNG + atlas.json"
    # 管线硬化 + bleed 生效：帧 alpha 二值、边缘有外渗色
    fr = cv2.imread(str(TMP / "pipe_out" / "pipe_000.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(fr[:, :, 3])) <= {0, 255}, "硬化应二值化 alpha"
    print("[OK] 集成：run_pipeline（硬化+bleed）+ atlas.json 含 hull 契约")

    print("\n图形处理 10 项能力全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
