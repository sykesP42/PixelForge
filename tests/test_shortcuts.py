"""快捷键面板 + 全局快捷键绑定测试（离屏）。

用法：QT_QPA_PLATFORM=offscreen python tests/test_shortcuts.py
验证：快捷键一览数据结构、QShortcut 注册数量与键位、
      面板可渲染（打桩 QDialog.exec 避免模态阻塞）、
      实际触发 Ctrl+R / Ctrl+1 生效。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QLabel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp(prefix="pf_shortcut_"))
    old_cfg, old_q = cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE
    cfgmod.CONFIG_FILE = tmp / "settings.json"
    cfgmod.GEN_QUEUE_FILE = tmp / "gen_queue.json"
    (tmp / "settings.json").write_text(
        json.dumps({"output_dir": str(tmp / "out")}), "utf-8")
    try:
        from pixelforge.ui.main_window import MainWindow
        win = MainWindow()
        win.assets_panel._root = tmp / "assets"

        # 1. 一览表数据结构完整
        groups = win.SHORTCUT_GROUPS
        assert groups, "SHORTCUT_GROUPS 不应为空"
        n_items = 0
        for group, items in groups:
            assert isinstance(group, str) and group, "分组名应为非空字符串"
            assert items, f"分组「{group}」不应为空"
            for keys, desc in items:
                assert isinstance(keys, str) and keys.strip()
                assert isinstance(desc, str) and desc.strip()
                n_items += 1
        print(f"[OK] 1. 快捷键一览数据结构：{len(groups)} 组 / {n_items} 条")

        # 2. QShortcut 注册数量与键位
        seqs = [sc.key().toString() for sc in win._shortcuts]
        expect = ["F1", "Ctrl+,", "Ctrl+G", "Ctrl+Shift+S", "Ctrl+Shift+O",
                  "Ctrl+R", "Ctrl+P", "Ctrl+1", "Ctrl+2", "Ctrl+3"]
        for e in expect:
            assert e in seqs, f"缺少快捷键绑定 {e}；已绑定={seqs}"
        assert len(win._shortcuts) == len(expect), f"绑定数={len(seqs)}"
        print(f"[OK] 2. 已注册 {len(seqs)} 个全局快捷键：{', '.join(seqs)}")

        # 3. 面板可渲染（打桩 exec，防离屏模态挂起）
        captured = {}
        orig_exec = QDialog.exec
        QDialog.exec = lambda self: captured.setdefault("dlg", self) and 0
        try:
            win._show_shortcuts()
        finally:
            QDialog.exec = orig_exec
        dlg = captured.get("dlg")
        assert dlg is not None, "应构建并 exec 快捷键对话框"
        texts = " ".join(l.text() for l in dlg.findChildren(QLabel))
        assert "F1" in texts and "Ctrl+G" in texts, "面板应列出快捷键内容"
        assert "看图器" in texts and "洋葱皮" in texts, "面板应含分组标题"
        print("[OK] 3. 面板渲染：分组标题 + 键位说明齐全")

        # 4. 实际触发：Ctrl+R 刷新资产库
        hits = {"n": 0}
        win.assets_panel.refresh = lambda: hits.__setitem__("n", hits["n"] + 1)
        sc_r = next(s for s in win._shortcuts if s.key().toString() == "Ctrl+R")
        sc_r.activated.emit()
        assert hits["n"] == 1, f"Ctrl+R 应触发一次刷新: {hits}"
        # Ctrl+1 -> 笨蛋模式
        sc_1 = next(s for s in win._shortcuts if s.key().toString() == "Ctrl+1")
        sc_1.activated.emit()
        assert win.cmb_ui_mode.currentIndex() == 0
        print("[OK] 4. 实际触发：Ctrl+R 刷新资产库 / Ctrl+1 切换笨蛋模式")

        win.close()
    finally:
        cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE = old_cfg, old_q

    print("\n快捷键面板测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
