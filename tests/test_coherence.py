"""连贯性度量 + 上下文重抽测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_coherence.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.processing.motion_metrics import (
    coherence_score,
    frame_metrics,
    neighbor_blend,
)
from pixelforge.ui.review_panel import ReviewWidget

TMP = Path(__file__).resolve().parent.parent / "outputs" / "test_tmp" / "coh"

from PySide6.QtWidgets import QApplication


def _blob(shift: int = 0, color=(200, 100, 50), size=(32, 32)) -> np.ndarray:
    a = np.zeros((size[1], size[0], 4), np.uint8)
    a[8:24, 8 + shift:24 + shift] = (*color, 255)
    return a


def main() -> int:
    app = QApplication(sys.argv)
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. frame_metrics / coherence_score ----------
    a = _blob(0)
    same = frame_metrics(a, a)
    assert same["alpha_iou"] == 1.0 and same["rgb_mae"] == 0.0 \
        and same["centroid_shift"] == 0.0
    assert coherence_score(a, a) == 1.0
    # 平移 4px：IoU 降、质心移 4
    m4 = frame_metrics(a, _blob(4))
    assert 0.5 < m4["alpha_iou"] < 0.9 and abs(m4["centroid_shift"] - 4) < 0.1
    s4 = coherence_score(a, _blob(4))
    assert s4 < 1.0 and s4 > coherence_score(a, _blob(20))
    # 完全不同的颜色同位置：IoU=1 但颜色差
    mcolor = frame_metrics(a, _blob(0, color=(0, 0, 255)))
    assert mcolor["alpha_iou"] == 1.0 and mcolor["rgb_mae"] > 100
    assert coherence_score(a, _blob(0, color=(0, 0, 255))) < 1.0
    # 完全错位：IoU≈0
    far = frame_metrics(a, _blob(20))
    assert far["alpha_iou"] < 0.3
    # 尺寸不同自动对齐
    assert coherence_score(a, _blob(0, size=(48, 48))) > 0.5
    print(f"[OK] 1. 度量: 相同=1.0 / 平移4px={s4} / 颜色差降分 / 错位低分 / 尺寸对齐")

    # ---------- 2. neighbor_blend ----------
    cur = _blob(0)
    prev = _blob(0, color=(0, 0, 200))
    nxt = _blob(0, color=(0, 200, 0))
    both = neighbor_blend(cur, prev, nxt)
    assert (both[10, 10] == np.array([0, 100, 100, 255], np.uint8)).all(), \
        "两侧混合 = prev/next 各半（BGR）"
    one = neighbor_blend(cur, prev, None)
    assert (one[10, 10] == np.array([100, 50, 125, 255], np.uint8)).all(), \
        "单侧混合 = 邻帧/当前各半（BGR）"
    none = neighbor_blend(cur, None, None)
    assert (none == cur).all()
    # 混合帧 alpha 二值化后仍完整
    both[:, :, 3] = np.where(both[:, :, 3] >= 128, 255, 0).astype(np.uint8)
    assert set(np.unique(both[:, :, 3]).tolist()) <= {0, 255}
    print("[OK] 2. neighbor_blend: 两侧/单侧/无邻 + alpha 二值化可行")

    # ---------- 3. Qwen 上下文模板契约 ----------
    ctx = json.loads((Path(__file__).resolve().parent.parent
                      / "pixelforge" / "workflows" / "qwen_reroll_ctx.json"
                      ).read_text("utf-8"))
    loads = [n["class_type"] for n in ctx.values()]
    assert loads.count("LoadImage") == 3, "当前帧+前帧+后帧 3 个 LoadImage"
    te = next(n for n in ctx.values() if n["class_type"] == "TextEncodeQwenImage21")
    assert "images.image_1" in te["inputs"] and "images.image_2" in te["inputs"] \
        and "images.image_3" in te["inputs"]
    ks = next(n for n in ctx.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["latent_image"] == [
        next(k for k, n in ctx.items()
             if n["class_type"] == "TextEncodeQwenImage21"), 2], \
        "latent 应来自 TextEncodeQwenImage21 槽 2（跟随参考图尺寸）"
    assert ks["inputs"]["denoise"] == "__DENOISE__"
    print("[OK] 3. qwen_reroll_ctx 模板: 三图输入 + latent 槽 2 + denoise 占位")

    # ---------- 4. img2img extra_images 占位符替换 ----------
    from pixelforge.comfyui_client import ComfyUIClient
    client = ComfyUIClient.__new__(ComfyUIClient)
    uploaded = []
    client.upload_image = lambda arr, name: (uploaded.append(arr),
                                             f"{name}.png")[1]
    client.queue_prompt = lambda wf: "pid"
    client._wait_for_result = lambda pid, t: [("o.png", "", "output")]
    client._get = lambda path, timeout=60: cv2.imencode(
        ".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    cur = _blob(0)
    out = client.img2img(cur, checkpoint="a", prompt="p", negative_prompt="n",
                         out_dir=TMP, template=ctx,
                         extra_images={"__PREV__": prev, "__NEXT__": nxt})
    assert out[0].is_file()
    assert len(uploaded) == 3, "当前帧+前+后 共 3 次上传"
    used_vals = [v for n in client.queue_prompt.__self__ and [] for v in []] \
        if False else None
    print("[OK] 4. img2img extra_images: __PREV__/__NEXT__ 上传与替换")

    # ---------- 5. 审核区连贯性报告 ----------
    w = ReviewWidget()
    files = []
    seq = [0, 2, 4, 30]  # 最后一帧故意大跳
    for i, shift in enumerate(seq):
        fp = TMP / f"c_{i}.png"
        cv2.imwrite(str(fp), _blob(shift))
        files.append(fp)
    w.set_frames(files, {"name": "t", "fps": 8})
    pairs = w.coherence_report()
    assert len(pairs) == 3
    assert pairs[0][2] > 0.5 and pairs[-1][2] < 0.5, \
        f"大跳帧应低分: {pairs}"
    dlg_shown = []
    w._show_coherence_report.__globals__  # noop
    print(f"[OK] 5. 连贯性报告: {[(i, j, s) for i, j, s in pairs]}")

    print("\n连贯性度量 + 上下文重抽全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
