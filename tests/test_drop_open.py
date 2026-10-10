"""拖拽打开测试（离屏）：真实 QDropEvent 灌入文件/文件夹/JSON。

用法：QT_QPA_PLATFORM=offscreen python tests/test_drop_open.py
验证：acceptDrops 生效、单图→看图器、多图→动画帧、目录→帧目录、
      Aseprite JSON→多动画组、无效 JSON/空目录不崩且给出日志。
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
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod


def _png(path: Path, color) -> Path:
    a = np.zeros((16, 16, 4), np.uint8)
    a[2:14, 2:14] = (*color, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), a)
    return path


def _drop(win, paths: list[Path]) -> None:
    """构造真实 QDropEvent 走 dropEvent 入口（而非直接调内部函数）。"""
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    ev = QDropEvent(QPointF(20, 20), Qt.CopyAction, md,
                    Qt.LeftButton, Qt.NoModifier)
    win.dropEvent(ev)
    assert ev.isAccepted(), "dropEvent 应接受该拖放"


def main() -> int:
    app = QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp(prefix="pf_drop_"))
    old_cfg, old_q = cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE
    cfgmod.CONFIG_FILE = tmp / "settings.json"
    cfgmod.GEN_QUEUE_FILE = tmp / "gen_queue.json"
    (tmp / "settings.json").write_text(
        json.dumps({"output_dir": str(tmp / "out")}), "utf-8")
    try:
        from pixelforge.ui.main_window import MainWindow
        win = MainWindow()
        win.assets_panel._root = tmp / "assets"
        assert win.acceptDrops(), "主窗口应开启 acceptDrops"

        # 1. 单张图片 -> 看图器（标签 0）
        one = _png(tmp / "solo.png", (255, 0, 0))
        _drop(win, [one])
        assert win.tabs.currentIndex() == 0, f"应切到看图器: {win.tabs.currentIndex()}"
        assert win.viewer.current_path() and \
            Path(win.viewer.current_path()).name == "solo.png"
        print("[OK] 1. 拖入单图 → 看图器预览")

        # 2. 多张图片 -> 动画播放器（帧数 = 图数）
        multi = [_png(tmp / f"seq_{i}.png", (0, 200, i * 4)) for i in range(4)]
        _drop(win, multi)
        assert win.tabs.currentWidget() is win.anim_player, "应切到动画播放"
        assert len(win.anim_player._frames) == 4, \
            f"应为 4 帧: {len(win.anim_player._frames)}"
        print("[OK] 2. 拖入多图 → 动画帧播放（4 帧）")

        # 3. 帧目录 -> 动画播放器
        d = tmp / "frames_dir"
        for i in range(3):
            _png(d / f"f_{i:03d}.png", (10, 10, 220))
        _drop(win, [d])
        assert len(win.anim_player._frames) == 3
        print("[OK] 3. 拖入帧目录 → 动画帧播放（3 帧）")

        # 4. Aseprite JSON -> 多动画组
        sheet = np.zeros((16, 48, 4), np.uint8)
        for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            sheet[2:14, i * 16 + 2:i * 16 + 14] = (*c, 255)
        cv2.imwrite(str(tmp / "anim.png"), sheet)
        aj = {"frames": [
            {"frame": {"x": i * 16, "y": 0, "w": 16, "h": 16},
             "duration": 100} for i in range(3)],
            "meta": {"image": "anim.png", "frameTags": [
                {"name": "idle", "from": 0, "to": 1},
                {"name": "run", "from": 2, "to": 2}]}}
        (tmp / "anim.json").write_text(json.dumps(aj), "utf-8")
        _drop(win, [tmp / "anim.json"])
        assert set(win.anim_player._clips) == {"idle", "run"}, \
            f"应导入 2 个动画组: {list(win.anim_player._clips)}"
        assert len(win.anim_player._frames) == 2, "当前应播放 idle（2 帧）"
        print("[OK] 4. 拖入 Aseprite JSON → 多动画组（idle/run）")

        # 5. 无关键型：无效 JSON + 空目录 -> 不崩，仍接受拖放
        bad = tmp / "bad.json"
        bad.write_text('{"hello": 1}', "utf-8")
        empty = tmp / "empty_dir"
        empty.mkdir(exist_ok=True)
        before = len(win.anim_player._frames)
        _drop(win, [bad, empty])
        assert len(win.anim_player._frames) == before, "无效拖入不应改变帧"
        print("[OK] 5. 无效 JSON / 空目录 → 安全忽略且日志提示")

        # 6. 动图 GIF -> 拆帧进动画播放器
        from PIL import Image
        gp = tmp / "walk.gif"
        cols = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
        frs = []
        for c in cols:
            im = Image.new("RGBA", (16, 16), c)
            frs.append(im)
        frs[0].save(gp, save_all=True, append_images=frs[1:], duration=100,
                    loop=0)
        _drop(win, [gp])
        assert win.tabs.currentWidget() is win.anim_player, "GIF 应切到动画播放"
        assert len(win.anim_player._frames) == 3, \
            f"GIF 应拆出 3 帧: {len(win.anim_player._frames)}"
        # GIF + 普通图混合：GIF 进动画，普通图进看图器
        png2 = _png(tmp / "mix.png", (255, 255, 0))
        _drop(win, [gp, png2])
        assert len(win.anim_player._frames) == 3, "GIF 帧数应保持 3"
        assert win.tabs.currentIndex() == 0 and \
            Path(win.viewer.current_path()).name == "mix.png", \
            "混合拖入时静态图应进看图器"
        print("[OK] 6. 拖入 GIF → 拆帧动画（3 帧）+ GIF/静态图混合分流")

        win.close()
    finally:
        cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE = old_cfg, old_q

    print("\n拖拽打开测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
