"""持久化生成队列测试（离屏）：config 往返 / 启动恢复 / 载入设置 /
写入更新 / 移除 / 永久删除（二次确认 + 索引清理 + 删除日志）。

用法：QT_QPA_PLATFORM=offscreen python tests/test_gen_queue.py
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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod


def _png(path: Path, color=(255, 0, 0)) -> Path:
    a = np.zeros((16, 16, 4), np.uint8)
    a[3:-3, 3:-3] = (*color, 255)
    cv2.imwrite(str(path), a)
    return path


def main() -> int:
    app = QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp(prefix="pf_queue_"))
    old_cfg, old_q = cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE
    cfgmod.CONFIG_FILE = tmp / "settings.json"
    cfgmod.GEN_QUEUE_FILE = tmp / "gen_queue.json"
    (tmp / "settings.json").write_text(
        json.dumps({"output_dir": str(tmp / "out")}), "utf-8")
    try:
        # ---------- 1. config 层：upsert / 最新在前 / 更新 ----------
        cfgmod.upsert_queue_record({
            "id": "a1", "time": "10-01 12:00", "status": "done",
            "engine": "GPT", "prompt": "knight idle breathing",
            "negative_prompt": "blurry", "model": "gpt-image-2",
            "size": "1k", "refs": [], "results": []})
        cfgmod.upsert_queue_record({
            "id": "a2", "time": "10-01 12:05", "status": "running",
            "engine": "GPT", "prompt": "mage cast", "results": []})
        recs = cfgmod.load_gen_queue()
        assert [r["id"] for r in recs] == ["a2", "a1"], "新记录应在最前"
        cfgmod.upsert_queue_record({"id": "a1", "status": "failed",
                                    "error": "boom"})
        again = {r["id"]: r for r in cfgmod.load_gen_queue()}
        assert again["a1"]["status"] == "failed" and len(again) == 2
        assert again["a1"]["prompt"] == "knight idle breathing"
        print("[OK] 1. config 队列往返：插入/最新在前/按 id 更新")

        # ---------- 2. 启动恢复 + running 归一化 ----------
        from pixelforge.ui.main_window import MainWindow
        win = MainWindow()
        win.GALLERY_INDEX = tmp / "gallery_index.json"
        win.PURGE_LOG = tmp / "purge_log.jsonl"
        win.assets_panel._root = tmp / "assets"
        assert win.lst_queue.count() == 2, f"应恢复 2 条: {win.lst_queue.count()}"
        top = win.lst_queue.item(0)
        assert "a2" in top.text(), f"最新应在最上: {top.text()}"
        assert "⏹" in top.text(), "上次未结束的 running 应标记为已停止"
        assert cfgmod.load_gen_queue()[0]["status"] == "stopped"
        print("[OK] 2. 启动恢复历史队列（running→stopped，最新在前）")

        # ---------- 3. 双击载入设置复用 ----------
        win.lst_queue.setCurrentItem(win.lst_queue.item(1))   # a1
        win._queue_load_settings()
        ed_p, ed_n = win._active_prompt_editors()
        assert ed_p.toPlainText() == "knight idle breathing"
        assert ed_n.toPlainText() == "blurry"
        assert win.cmb_gpt_model.currentText() == "gpt-image-2"
        assert win.cmb_gpt_size.currentText() == "1k"
        print("[OK] 3. 载入历史任务设置（提示词/负面/模型/尺寸回填）")

        # ---------- 4. 新任务写入与状态更新 ----------
        png1 = _png(tmp / "out" / "g1.png")
        png1.parent.mkdir(parents=True, exist_ok=True)
        _png(png1)
        win._qid_of_tn[7] = "b1"
        cfgmod.upsert_queue_record({
            "id": "b1", "time": "10-02 09:00", "status": "running",
            "engine": "GPT", "prompt": "run cycle", "results": []})
        it = win._queue_add_item(cfgmod.load_gen_queue()[0], 7)
        assert it.data(Qt.UserRole) == 7 and it.data(Qt.UserRole + 2) == "b1"
        win._queue_update("b1", status="done", results=[str(png1)])
        win._set_queue_done(7, "✔")
        assert win._queue_record_of(it)["status"] == "done"
        assert win._queue_record_of(it)["results"] == [str(png1)]
        assert it.text().endswith("✔"), f"应带完成标记: {it.text()}"
        print("[OK] 4. 运行中任务：写入记录 + 完成回填产物/标记")

        # ---------- 5. 从队列移除（保留文件） ----------
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        win._queue_delete(it, purge=False)
        assert not png1.is_file() is False and png1.is_file()
        assert "b1" not in {r["id"] for r in cfgmod.load_gen_queue()}
        assert win.lst_queue.count() == 2
        print("[OK] 5. 移除队列项：记录删除、本地文件保留")

        # ---------- 6. 永久删除（二次确认 + 索引清理 + 日志） ----------
        png2 = _png(tmp / "out" / "g2.png")
        (tmp / "gallery_index.json").write_text(
            json.dumps([str(png1), str(png2)], ensure_ascii=False), "utf-8")
        lib = {"favorites": [png2.name], "folders": {png2.name: "角色"},
               "categories": {png2.name: "特效"},
               "prompts": {png2.name: {"prompt": "x"}}}
        (tmp / "assets").mkdir(exist_ok=True)
        (tmp / "assets" / "library.json").write_text(
            json.dumps(lib, ensure_ascii=False), "utf-8")
        win._refresh_gallery_wall()      # 先建画廊墙（验证该方法可用）
        assert win.lst_gallery_wall.count() == 2
        win.assets_panel.refresh()
        cfgmod.upsert_queue_record({
            "id": "c1", "status": "done", "engine": "GPT",
            "prompt": "purge me", "results": [str(png2)]})
        it2 = win._queue_add_item(cfgmod.load_gen_queue()[0])
        QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Yes)
        win._queue_delete(it2, purge=True)
        assert not png2.is_file(), "永久删除应删掉本地文件"
        assert json.loads((tmp / "gallery_index.json").read_text("utf-8")) == \
            [str(png1)], "画廊索引应剔除已删文件"
        lib_after = json.loads(
            (tmp / "assets" / "library.json").read_text("utf-8"))
        assert png2.name not in lib_after["favorites"]
        assert png2.name not in lib_after["folders"]
        assert png2.name not in lib_after["categories"]
        assert png2.name not in lib_after["prompts"]
        assert win.lst_gallery_wall.count() == 1, "画廊墙应重建为 1 项"
        log_lines = [x for x in (tmp / "purge_log.jsonl").read_text("utf-8")
                     .splitlines() if x.strip()]
        assert len(log_lines) == 1 and png2.name in log_lines[0]
        print("[OK] 6. 永久删除：文件/队列/画廊/资产库全清 + 删除日志留存")
        win.close()
    finally:
        cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE = old_cfg, old_q

    print("\n持久化生成队列全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
