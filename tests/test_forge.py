"""Batch Forge 多角色批量生成引擎测试。

用法：QT_QPA_PLATFORM=offscreen python tests/test_forge.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.batch_forge import (
    ForgeRole,
    ForgeTask,
    build_tasks,
    inject_prompt,
    resolve_action,
    archive_path,
    sanitize_code,
)
from pixelforge.config import Settings

TMP = Path(__file__).resolve().parent.parent / "outputs" / "test_tmp" / "forge"

from PySide6.QtWidgets import QApplication


def _save_ref(path: Path, color) -> None:
    img = np.zeros((32, 32, 3), np.uint8)
    img[:] = color
    cv2.imwrite(str(path), img)


def main() -> int:
    app = QApplication(sys.argv)
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------- 1. resolve_action：字典/冒号/自由文本 ----------
    assert resolve_action("walk") == ("walk", "walking cycle, side view")
    assert resolve_action("stomp") == ("stomp", "violent stomping down, impact frame")
    assert resolve_action("wave: waving hand gently") == ("wave", "waving hand gently")
    assert resolve_action("dancing happily") == ("custom", "dancing happily")
    assert resolve_action("") == ("default", "")
    print("[OK] 1. resolve_action: 内置字典 / name: 文本 / 自由文本 / 空")

    # ---------- 2. inject_prompt：占位符替换与拼接 ----------
    m1 = inject_prompt("master, [动作], [角色描述], green bg",
                       "walking cycle", "black hair, red jacket")
    assert m1 == "master, walking cycle, black hair, red jacket, green bg", m1
    # 英文占位符
    m2 = inject_prompt("master, [Action], [Character]",
                       "attack pose", "white hair")
    assert m2 == "master, attack pose, white hair"
    # 无占位符：尾部拼接
    m3 = inject_prompt("master, ultra detailed", "attack pose", "white hair")
    assert m3 == "master, ultra detailed, attack pose, white hair", m3
    # 空段不产生多余逗号
    m4 = inject_prompt("master", "", "desc only")
    assert m4 == "master, desc only"
    print("[OK] 2. inject_prompt: 中英占位符 / 尾部拼接 / 空段清理")

    # ---------- 3. build_tasks：矩阵与 seed 三模式 ----------
    roles = [ForgeRole("Hero_A", "a.png", "black hair", 2),
             ForgeRole("Villain_B", "b.png", "white hair", 1)]
    t_fixed = build_tasks(roles, ["walk", "attack"], "fixed", 1024,
                          "master, [动作], [角色描述]", TMP, "proj")
    # 2 角色 × 2 动作 × 数量(2+1) = 6
    assert len(t_fixed) == 6, len(t_fixed)
    assert all(t.seed == 1024 for t in t_fixed), "fixed 模式同种子"
    hero_walk = [t for t in t_fixed if t.role_code == "Hero_A"
                 and t.action_name == "walk"]
    assert len(hero_walk) == 2 and hero_walk[0].index == 1 \
        and hero_walk[1].index == 2
    assert "walking cycle" in hero_walk[0].prompt
    assert "black hair" in hero_walk[0].prompt
    # 归档路径
    assert hero_walk[0].out_path == TMP / "proj" / "Hero_A" / \
        "Hero_A_walk_1024.png"
    t_inc = build_tasks(roles, ["walk"], "increment", 500,
                        "m", TMP, "proj")
    seeds = [t.seed for t in t_inc]
    assert seeds == sorted(seeds) and len(set(seeds)) == len(seeds), \
        "increment 应递增且不重复"
    t_rand = build_tasks(roles, ["walk"], "random", 0, "m", TMP, "proj")
    assert len(t_rand) == 3
    print(f"[OK] 3. build_tasks: 6 任务矩阵 / fixed 同种子 / increment 递增 "
          f"/ random（seeds={sorted(t.seed for t in t_rand)}）")

    # ---------- 4. sanitize_code ----------
    used: set[str] = set()
    assert sanitize_code("Hero A!", used) == "Hero_A"
    assert sanitize_code("Hero A!", used) == "Hero_A_2", "重名应加序号"
    assert sanitize_code("", used).startswith("char")
    print("[OK] 4. sanitize_code: 非法字符替换 + 去重")

    # ---------- 5. worker：串行 + /free + 失败重试（mock 网络）----------
    import pixelforge.ui.forge_panel as fp
    calls = {"free": 0, "order": [], "fail_first": {}}

    class FakeClient:
        @classmethod
        def from_settings(cls, s):
            return cls()

        def is_online(self):
            return True

        def free_memory(self):
            calls["free"] += 1
            return True

        def upload_image(self, arr, name):
            return f"{name}.png"

        def queue_prompt(self, wf):
            # Hero_A_walk 任务首次失败一次（重试验证）
            tag = wf["99"]["inputs"]["filename_prefix"] if "99" in wf else ""
            calls["order"].append(wf["1"]["inputs"]["image"] if "1" in wf else "?")
            return f"pid_{len(calls['order'])}"

        def _wait_for_result(self, pid, timeout=900):
            idx = int(pid.split("_")[1])
            # 约定：第 2 个提交的任务首次失败
            if idx == 2 and calls["fail_first"].get(idx) is None:
                calls["fail_first"][idx] = True
                raise RuntimeError("boom")
            return [("out.png", "", "output")]

        def _get(self, path, timeout=120):
            return cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()

    # 构造 2 任务：role_a 1 张、role_b 1 张，模板里用 filename_prefix 区分？
    # 简化：直接给 worker 两个任务，mock 队列按序返回
    ref_a = TMP / "a.png"; _save_ref(ref_a, (255, 0, 0))
    ref_b = TMP / "b.png"; _save_ref(ref_b, (0, 0, 255))
    roles2 = [ForgeRole("Hero_A", str(ref_a), "black hair", 1),
              ForgeRole("Villain_B", str(ref_b), "white hair", 1)]
    tasks2 = build_tasks(roles2, ["walk"], "fixed", 100, "m, [动作], [角色描述]",
                         TMP, "wproj")
    assert len(tasks2) == 2
    orig_client = fp.ComfyUIClient
    fp.ComfyUIClient = FakeClient
    events = {"done": None, "failed": [], "thumbs": []}
    try:
        worker = fp._ForgeWorker(Settings(), tasks2,
                                 {r.code: r for r in roles2}, None,
                                 {"stop": False}, retries=1)
        # 注入自定义模板（含 __IMAGE__ 与 filename_prefix 可区分）
        worker.template = {
            "1": {"class_type": "DummyI2I",
                  "inputs": {"image": "__IMAGE__", "text": "__PROMPT__"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        worker.sig.task_failed.connect(
            lambda tag, e: events["failed"].append((tag, e)))
        worker.sig.done.connect(lambda ok, total: events.__setitem__("done", (ok, total)))
        worker.sig.thumb.connect(lambda p: events["thumbs"].append(p))
        worker.run()
    finally:
        fp.ComfyUIClient = orig_client
    assert events["done"] == (2, 2), events
    assert calls["free"] == 2, f"每任务前应调 /free: {calls['free']}"
    assert len(events["thumbs"]) == 2
    # 归档文件存在且命名正确
    for t in tasks2:
        assert t.out_path.is_file(), f"归档缺失: {t.out_path}"
        assert t.out_path.name.startswith(t.role_code)
    print(f"[OK] 5. worker: 串行 2 任务 / /free x{calls['free']} / 归档命名正确")

    # ---------- 6. 失败重试：首次失败 → 同 seed 重试成功 ----------
    # 永久失败场景：重试上限后进失败列表（独立的 AlwaysFail 类）
    class AlwaysFailClient(FakeClient):
        def _wait_for_result(self, pid, timeout=900):
            raise RuntimeError("permanent boom")

    fail_events = []
    tasks3 = [ForgeTask("Hero_A", "walk", "m", 777, 1,
                        TMP / "wproj" / "Hero_A" / "Hero_A_walk_777.png")]
    orig_client = fp.ComfyUIClient
    try:
        fp.ComfyUIClient = AlwaysFailClient
        worker3 = fp._ForgeWorker(Settings(), tasks3,
                                  {"Hero_A": ForgeRole("Hero_A", str(ref_a))},
                                  None, {"stop": False}, retries=1)
        worker3.template = {"1": {"class_type": "DummyI2I",
                                  "inputs": {"image": "__IMAGE__"}},
                            "2": {"class_type": "SaveImage",
                                  "inputs": {"images": ["1", 0]}}}
        worker3.sig.task_failed.connect(
            lambda t, e: fail_events.append((t, e)))
        worker3.sig.done.connect(
            lambda ok, total: events.__setitem__("done3", (ok, total)))
        worker3.run()
    finally:
        fp.ComfyUIClient = orig_client
    assert events.get("done3") == (0, 1), events.get("done3")
    assert len(fail_events) == 1 and "permanent boom" in fail_events[0][1]
    print("[OK] 6. 永久失败: 重试一次后进失败列表并携带原因")

    # 重试成功场景：FailOnce 客户端（同 seed 第一次失败，第二次成功）
    class FailOnceClient(FakeClient):
        calls = 0

        def _wait_for_result(self, pid, timeout=900):
            FailOnceClient.calls += 1
            if FailOnceClient.calls == 1:
                raise RuntimeError("transient boom")
            return [("out.png", "", "output")]

    tasks4 = [ForgeTask("Hero_A", "walk", "m", 999, 1,
                        TMP / "wproj" / "Hero_A" / "Hero_A_walk_999.png")]
    events["done4"] = None
    try:
        fp.ComfyUIClient = FailOnceClient
        worker4 = fp._ForgeWorker(Settings(), tasks4,
                                  {"Hero_A": ForgeRole("Hero_A", str(ref_a))},
                                  None, {"stop": False}, retries=1)
        worker4.template = {"1": {"class_type": "DummyI2I",
                                  "inputs": {"image": "__IMAGE__"}},
                            "2": {"class_type": "SaveImage",
                                  "inputs": {"images": ["1", 0]}}}
        worker4.sig.done.connect(
            lambda ok, total: events.__setitem__("done4", (ok, total)))
        worker4.run()
    finally:
        fp.ComfyUIClient = orig_client
    assert events.get("done4") == (1, 1), events.get("done4")
    assert tasks4[0].attempts == 1, "应恰好重试一次"
    assert tasks4[0].out_path.is_file()
    print("[OK] 6b. 瞬时失败: 同 seed 重试一次后成功并归档")

    # ---------- 7. UI 面板冒烟 + ZIP ----------
    from pixelforge.ui.forge_panel import ForgePanel
    panel = ForgePanel(lambda: Settings())
    assert panel.table.columnCount() == 4
    assert panel.cmb_tpl.count() >= 3, "模板下拉应列出全部工作流"
    # 模拟添加角色（绕过文件对话框）
    _save_ref(TMP / "r1.png", (255, 128, 0))
    used2: set[str] = set()
    code = sanitize_code("Hero_A", used2)
    r = panel.table.rowCount()
    panel.table.insertRow(r)
    from PySide6.QtWidgets import QTableWidgetItem
    it0 = QTableWidgetItem("")
    it0.setToolTip(str(TMP / "r1.png"))
    panel.table.setItem(r, 0, it0)
    panel.table.setItem(r, 1, QTableWidgetItem(code))
    panel.table.setItem(r, 2, QTableWidgetItem("black hair, red jacket"))
    panel.table.setItem(r, 3, QTableWidgetItem("1"))
    roles_ui = panel._roles()
    assert len(roles_ui) == 1 and roles_ui[0].code == "Hero_A"
    assert roles_ui[0].desc == "black hair, red jacket"
    # ZIP 导出（直接调打包逻辑）
    zout = TMP / "proj.zip"
    with zipfile.ZipFile(zout, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((TMP / "wproj").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(TMP / "wproj").parent.as_posix() + "/" + p.name)
    with zipfile.ZipFile(zout) as z:
        assert len(z.namelist()) >= 2
    print("[OK] 7. UI 冒烟: 角色表/模板下拉/roles 解析 + ZIP 打包")

    print("\nBatch Forge 全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
