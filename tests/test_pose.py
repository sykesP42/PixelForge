"""骨架驱动生成 + 单帧重抽 + 错误翻译测试（mock，不依赖真实 ComfyUI）。

用法：QT_QPA_PLATFORM=offscreen python tests/test_pose.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.comfyui_client import (
    ComfyUIClient,
    build_workflow,
    prepare_pose,
)
from pixelforge.config import PROJECT_ROOT
from pixelforge.error_hints import translate_error
from pixelforge.ui.review_panel import ReviewWidget

ROOT = Path(__file__).resolve().parent.parent
WF = PROJECT_ROOT / "pixelforge" / "workflows" / "pose2img_frame.json"

from PySide6.QtWidgets import QApplication


def _pose_png(path: Path, w: int = 32, h: int = 48) -> None:
    img = np.zeros((h, w), np.uint8)  # 白底黑骨架（灰度）
    img[:] = 255
    cv2.line(img, (w // 2, 8), (w // 2, h - 8), 0, 2)
    cv2.circle(img, (w // 2, 6), 4, 0, -1)
    cv2.imwrite(str(path), img)


def main() -> int:
    app = QApplication(sys.argv)

    # ---------- 1. 模板契约：img2img 锚定 + ControlNet 姿势 ----------
    tpl = json.loads(WF.read_text("utf-8"))
    cls = {n["class_type"] for n in tpl.values()}
    assert {"LoadImage", "VAEEncode", "SetLatentNoiseMask", "KSampler",
            "ControlNetLoader", "ControlNetApply", "SaveImage"} <= cls
    assert sum(n["class_type"] == "LoadImage" for n in tpl.values()) == 2, \
        "应有 2 个 LoadImage（锚定帧 + 骨架）"
    ks = next(n for n in tpl.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["denoise"] == "__DENOISE__"
    cn = next(n for n in tpl.values() if n["class_type"] == "ControlNetApply")
    pos_id = next(k for k, n in tpl.items() if n["class_type"] == "CLIPTextEncode"
                  and n["inputs"]["text"] == "__PROMPT__")
    assert cn["inputs"]["conditioning"] == [pos_id, 0], "CN 应接正向提示"
    assert cn["inputs"]["image"][1] == 0 and cn["inputs"]["strength"] == "__POSE_STRENGTH__"
    cn_id = next(k for k, n in tpl.items() if n["class_type"] == "ControlNetApply")
    assert ks["inputs"]["positive"] == [cn_id, 0], "KSampler 正向应来自 CN 输出"
    assert sum(1 for n in tpl.values() if n["class_type"] == "SaveImage") == 1
    print("[OK] 1. pose2img 模板: 双 LoadImage / CN 接正向 / KS 接 CN / 单 SaveImage")

    # ---------- 2. build_workflow：新占位符替换 + LoRA 兼容 ----------
    wf = build_workflow(tpl, checkpoint="a.safetensors", prompt="p",
                        negative_prompt="n",
                        seed=7, width=64, height=64, steps=10, cfg=6.0,
                        lora_name="my.safetensors", controlnet_name="openpose.safetensors",
                        pose_strength=0.85)
    all_inputs = [v for n in wf.values() for v in n["inputs"].values()]
    assert "openpose.safetensors" in all_inputs
    assert 0.85 in all_inputs
    assert not any(isinstance(v, str) and v.startswith("__") for v in all_inputs
                   if v not in ("__IMAGE__", "__POSE__", "__DENOISE__")), \
        "静态占位符应全部替换（__IMAGE__/__POSE__/__DENOISE__ 为运行时值）"
    assert "90" in wf, "LoRA 注入节点 90 应存在"
    cn_node = next(n for n in wf.values() if n["class_type"] == "ControlNetApply")
    # LoRA 在 CLIP 层生效：CN 的 conditioning 来自已 LoRA 化的 CLIP 输出
    clip_node = next(n for n in wf.values() if n["class_type"] == "CLIPTextEncode"
                     and n["inputs"].get("text") == "p")
    assert clip_node["inputs"]["clip"] == ["90", 1], "正向 CLIP 应过 LoRA"
    assert cn_node["inputs"]["conditioning"] == [pos_id, 0], \
        "CN 应接 LoRA 化 CLIP 的输出（90 -> 2 -> 9 -> 10 链序）"
    print("[OK] 2. build_workflow: __CONTROLNET__/__POSE_STRENGTH__ 替换 + LoRA 改线兼容")

    # ---------- 3. prepare_pose：灰度骨架缩放到生成尺寸 ----------
    tmp = Path(tempfile.mkdtemp(prefix="pf_pose_"))
    _pose_png(tmp / "p1.png", 32, 48)
    pose = prepare_pose(tmp / "p1.png", 64, 64)
    assert pose.shape == (64, 64, 3) and pose.dtype == np.uint8
    try:
        prepare_pose(tmp / "missing.png", 64, 64)
        raise SystemExit("应报错")
    except Exception as e:
        assert "骨架图读取失败" in str(e)
    print("[OK] 3. prepare_pose: 灰度->BGR + nearest 缩放 + 缺文件报错")

    # ---------- 4. generate_pose_series：mock 循环 ----------
    client = ComfyUIClient.__new__(ComfyUIClient)
    uploads: list[np.ndarray] = []

    def fake_upload(arr, name):
        uploads.append(arr)
        return f"{name}.png"

    client.upload_image = fake_upload
    client.queue_prompt = lambda wf: f"pid_{len(uploads)}"
    client._wait_for_result = lambda pid, t: [("out.png", "", "output")]
    fake_dir = tmp / "out"

    # 预置假输出文件（_wait_for_result mock 后 generate 内部不再下载——
    # generate_pose_series 自行下载，因此 mock _get）
    client._get = lambda path, timeout=60: cv2.imencode(
        ".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()

    anchor = np.zeros((64, 64, 4), np.uint8); anchor[..., 3] = 255
    _pose_png(tmp / "p2.png", 20, 20)
    _pose_png(tmp / "p3.png", 20, 20)
    got = client.generate_pose_series(
        anchor, [tmp / "p2.png", tmp / "p3.png"], checkpoint="a",
        controlnet_name="cn.safetensors", prompt="knight", negative_prompt="blur",
        pose_strength=0.9, denoise=0.45, out_dir=fake_dir)
    assert len(got) == 2 and all(p.is_file() for p in got)
    # uploads: [anchor, pose1, pose2] —— anchor 只上传一次
    assert len(uploads) == 3
    assert uploads[0].shape == (64, 64, 4) and uploads[0].shape[2] == 4
    assert uploads[1].shape == (64, 64, 3) and uploads[1].shape[:2] == (64, 64), \
        "骨架图应被缩放到 anchor 尺寸"
    print("[OK] 4. generate_pose_series: 每帧换骨架 / anchor 单次上传 / 尺寸对齐")

    # ---------- 5. 错误翻译 ----------
    assert "显存" in translate_error("CUDA out of memory at step 3")
    assert "ControlNet" in translate_error("control_net_name not in list")
    assert translate_error("完全未知的错误xyz") == "完全未知的错误xyz", \
        "未命中模式应原样返回"
    print("[OK] 5. translate_error: 显存/CN 映射 + 未命中原样返回")

    # ---------- 6. 审核区单帧重抽接线 ----------
    w = ReviewWidget()
    events = []
    w.reroll_requested.connect(lambda row: events.append(row))
    import json as _json
    files = []
    for i in range(3):
        fp = tmp / f"f_{i}.png"
        a = np.zeros((16, 16, 4), np.uint8); a[4:12, 4:12] = (10 * i, 0, 200, 255)
        cv2.imwrite(str(fp), a)
        files.append(fp)
    w.set_frames(files, {"name": "t", "fps": 8})
    w._context_menu  # 存在
    assert (w.frame_arr(1)[4:12, 4:12, 0] == 10).all()
    new_arr = np.zeros((16, 16, 4), np.uint8)
    new_arr[4:12, 4:12] = (99, 0, 200, 255)
    assert w.replace_frame(1, new_arr)
    assert (w.frame_arr(1)[4:12, 4:12, 0] == 99).all(), "replace_frame 应更新帧"
    assert not w.replace_frame(99, new_arr), "越界替换返回 False"
    print("[OK] 6. ReviewWidget: frame_arr/replace_frame/reroll 信号就绪")

    print("\n骨架驱动 + 单帧重抽 + 错误翻译全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
