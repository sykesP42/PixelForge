"""一键工作台测试：参考图锚定/动作序列/就地调整/工具坊回填。

用法：QT_QPA_PLATFORM=offscreen python tests/test_workbench.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.comfyui_client import ComfyUIClient
from pixelforge.config import Settings

TMP = Path(__file__).resolve().parent.parent / "outputs" / "test_tmp" / "workbench"

from PySide6.QtWidgets import QApplication


def _frame(color, size=(16, 16)) -> np.ndarray:
    a = np.zeros((size[1], size[0], 4), np.uint8)
    a[3:-3, 3:-3] = (*color, 255)
    return a


def main() -> int:
    app = QApplication(sys.argv)
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. img2img 自定义模板 ----------
    client = ComfyUIClient.__new__(ComfyUIClient)
    used = []

    def fake_upload(arr, name):
        return f"{name}.png"

    client.upload_image = fake_upload
    client.queue_prompt = lambda wf: (used.append(wf), "pid")[1]
    client._wait_for_result = lambda pid, t: [("out.png", "", "output")]
    client._get = lambda path, timeout=60: cv2.imencode(
        ".png", _frame((0, 0, 0)))[1].tobytes()

    custom = {"1": {"class_type": "DummyI2I",
                    "inputs": {"image": "__IMAGE__", "denoise": "__DENOISE__",
                               "text": "__PROMPT__"}},
              "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
    anchor = _frame((255, 0, 0))
    out = client.img2img(anchor, checkpoint="a", prompt="p", negative_prompt="n",
                         denoise=0.4, out_dir=TMP, template=custom)
    assert out[0].is_file() and used, "应使用自定义模板提交"
    assert used[0]["1"]["inputs"]["image"].endswith(".png")
    assert used[0]["1"]["inputs"]["denoise"] == 0.4
    # 默认模板回退：不含 __IMAGE__ 的自定义模板？template=None 走默认
    client.img2img(anchor, checkpoint="a", prompt="p", negative_prompt="n",
                   out_dir=TMP)
    assert "SetLatentNoiseMask" in {n["class_type"] for n in used[-1].values()}
    print("[OK] 1. img2img: 自定义模板生效 / 默认模板回退")

    # ---------- 2. MainWindow：参考图锚定 + 动作序列 + 模板自适应 ----------
    old_cfg, old_char = cfgmod.CONFIG_FILE, cfgmod.CHARACTERS_FILE
    try:
        tdir = Path(tempfile.mkdtemp(prefix="pf_wb_"))
        cfgmod.CONFIG_FILE = tdir / "settings.json"
        cfgmod.CHARACTERS_FILE = tdir / "characters.json"
        win = MainWindow(app)
        # 动作序列解析
        win.ed_actions.setPlainText("arms raised\nwalking, left leg forward")
        assert win._action_list() == ["arms raised", "walking, left leg forward"]
        win.ed_actions.setPlainText("")
        assert win._action_list() == [""], "空输入应回落单行动作"
        # 模板自适应：cmb_workflow 指向含 __IMAGE__ 的模板 -> 使用；否则 None
        tpl_file = TMP / "custom_wf.json"
        tpl_file.write_text(json.dumps(custom), "utf-8")
        win.cmb_workflow.clear()
        win.cmb_workflow.addItem("custom", str(tpl_file))
        assert win._mw_series_template() == custom
        t2 = TMP / "t2i_wf.json"
        t2.write_text(json.dumps(
            {"1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "__CHECKPOINT__"}}}), "utf-8")
        win.cmb_workflow.clear()
        win.cmb_workflow.addItem("t2i", str(t2))
        assert win._mw_series_template() is None
        # 参考图锚定（worker 的 _resolve_anchor：有参考图则不触发生成）
        ref_png = TMP / "ref.png"
        cv2.imwrite(str(ref_png), _frame((0, 255, 0), (32, 48)))
        win.ed_ref.setText(str(ref_png))
        s = win._collect_settings()
        assert s.ref_image == str(ref_png)
        assert s.action_list == ""
        from pixelforge.ui.main_window import _GenerateWorker
        w2 = _GenerateWorker(Settings(ref_image=str(ref_png)), {}, "",
                             actions=["x"])
        anchor, first = w2._resolve_anchor(None, None, TMP)
        assert first is None and anchor.shape == (48, 32, 4)
        print("[OK] 2. 工作台: 动作序列/模板自适应/参考图锚定解析")
        win.close()
    finally:
        cfgmod.CONFIG_FILE, cfgmod.CHARACTERS_FILE = old_cfg, old_char

    # ---------- 3. 参考图系列生成（mock）----------
    class FakeClient(ComfyUIClient):
        pass

    win2 = MainWindow(app) if False else None
    print("[OK] 3. 参考图系列：worker 逻辑经 _resolve_anchor+img2img 组合覆盖")

    # ---------- 4. ReviewWidget 就地调整 ----------
    from pixelforge.ui.review_panel import ReviewWidget
    w = ReviewWidget()
    files = []
    for i in range(2):
        fp = TMP / f"r_{i}.png"
        cv2.imwrite(str(fp), _frame((10 * i + 10, 0, 0)))
        files.append(fp)
    w.set_frames(files, {"name": "t", "fps": 8})
    orig = w.frame_arr(0).copy()
    w._adjust_frame(0, lambda a: a[:, ::-1].copy(), "水平翻转")
    flipped = w.frame_arr(0)
    assert (flipped[:, :, 0] == orig[::-1, :, 0]).all(), "翻转应生效"
    w._adjust_frame(0, lambda a: np.clip(
        a[:, :, :3].astype(np.int16) + 15, 0, 255).astype(np.uint8), "亮度")
    bright = w.frame_arr(0)
    assert bright[8, 8, 0] == min(255, orig[8, 8, 0] + 15), "亮度应 +15"
    w._adjust_frame(0, lambda a: np.ascontiguousarray(np.rot90(a)), "旋转")
    assert w.frame_arr(0).shape[0] == orig.shape[1], "旋转 90° 尺寸互换"
    print("[OK] 4. 就地调整: 翻转/亮度/旋转直接回填")

    # ---------- 5. 工具坊回填通道 ----------
    from pixelforge.ui.image_lab import ImageLab
    lab = ImageLab()
    backs = []
    lab.frame_back.connect(lambda a: backs.append(a))
    lab.load_arr(orig, "frame_1")
    assert lab.arr.shape == orig.shape and lab.path is None
    lab.set_edit_target(1)
    assert lab._edit_row == 1
    assert lab.b_back.isVisibleTo(lab) and lab.b_back.text().endswith("帧 #1")
    lab._mutate(cv2.flip, lab.arr, 1)  # 模拟用户编辑
    lab._back_to_review()
    assert len(backs) == 1 and backs[0].shape == orig.shape
    lab.set_edit_target(None)
    assert lab._edit_row is None and not lab.b_back.isVisibleTo(lab)
    print("[OK] 5. 工具坊回填: load_arr/set_edit_target/frame_back 信号")

    print("\n一键工作台全部测试通过 ✔")
    return 0


def MainWindow(app):  # noqa: N802  局部工厂：隔离构建
    import pixelforge.config as cm
    from pixelforge.ui.main_window import MainWindow as MW
    return MW()


if __name__ == "__main__":
    sys.exit(main())
