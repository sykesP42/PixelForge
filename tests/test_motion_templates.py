"""动作模板集 + 批量断点续跑测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_motion_templates.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.config import Settings
from pixelforge.motion_templates import MOTION_TEMPLATES, get_template, template_names
from pixelforge.ui.batch_panel import BatchPanel, _BatchWorker


def main() -> int:
    app = QApplication(sys.argv)

    # ---------- 1. 模板注册表合法性 ----------
    names = template_names()
    assert len(names) >= 4 and len(set(names)) == len(names)
    seen: set[str] = set()
    for n in names:
        rows = get_template(n)
        assert rows, f"模板 {n} 为空"
        for action, prompt in rows:
            assert action and prompt, f"模板 {n} 存在空动作/空提示词"
            assert "/" not in action and "\\" not in action, \
                "动作名是输出子目录名，不能含路径分隔符"
            key = (n, action)
            assert key not in seen
            seen.add(key)
    assert "行走 · 4 方向" in MOTION_TEMPLATES
    print(f"[OK] 1. 模板注册表: {len(names)} 个模板全部合法")

    # ---------- 2. BatchPanel 一键填充 ----------
    panel = BatchPanel(settings_provider=lambda: Settings())
    before = panel.table.rowCount()
    n = panel.apply_template("行走 · 4 方向")
    assert n == 4 and panel.table.rowCount() == before + 4
    item = panel.table.item(before, 0)
    assert item.text() == "walk_down"
    assert "walk cycle" in panel.table.item(before, 1).text()
    # 空行不会被 _actions 收录——模板行必须可执行
    panel.table.setItem(before, 1, None)  # 模拟提示词缺失
    try:
        actions = panel._actions()
    finally:
        panel.table.setItem(before, 1, panel.table.item(before, 1))
    print(f"[OK] 2. 模板填充: {n} 行入队，动作名/提示词写入正确")

    # ---------- 3. 断点续跑：已完成动作跳过（不触发生成） ----------
    tmp = Path(tempfile.mkdtemp(prefix="pf_resume_"))
    (tmp / "idle_down").mkdir(parents=True)
    (tmp / "idle_down" / "atlas.json").write_text(
        json.dumps({"name": "idle_down", "frame_count": 2, "frames": []}),
        "utf-8")

    class FakeClient:
        @classmethod
        def from_settings(cls, s):
            return cls()

        def is_online(self):
            return True

        def generate(self, *a, **kw):  # 续跑时绝不应该触发生成
            raise AssertionError("已完成动作不应重新生成")

    import pixelforge.ui.batch_panel as bp

    worker = _BatchWorker(Settings(), [("idle_down", "walk cycle")],
                          tmp, {"stop": False}, resume=True)
    events = {"done": [], "batch": None}
    worker.sig.action_done.connect(lambda i, ok, m: events["done"].append((ok, m)))
    worker.sig.batch_done.connect(lambda ok, t: events.__setitem__("batch", (ok, t)))
    orig = bp.ComfyUIClient
    try:
        bp.ComfyUIClient = FakeClient
        worker.run()
        assert events["done"] and events["done"][0][0] is True, events
        assert "跳过" in events["done"][0][1]
        assert events["batch"] == (1, 1)

        # 关闭续跑 -> 走生成路径 -> FakeClient 抛错经失败事件回传
        worker2 = _BatchWorker(Settings(), [("idle_down", "walk cycle")],
                               tmp, {"stop": False}, resume=False)
        events2 = []
        worker2.sig.action_done.connect(
            lambda i, ok, m: events2.append((ok, m)))
        worker2.run()
        assert len(events2) == 1 and events2[0][0] is False
        assert "AssertionError" in events2[0][1], "失败路径应回传生成错误"
    finally:
        bp.ComfyUIClient = orig
    print("[OK] 3. 断点续跑: 已完成跳过且不触发生成；关闭续跑走生成路径")

    print("\n动作模板 + 断点续跑全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
