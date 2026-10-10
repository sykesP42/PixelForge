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

        # ---------- 7. 队列筛选：状态 + 关键词模糊过滤 ----------
        n = win.lst_queue.count()
        assert n == 2, f"筛选用例前置应为 2 条: {n}"
        win.cmb_qfilter.setCurrentText("全部")
        win.ed_qsearch.setText("")
        assert all(not win.lst_queue.item(i).isHidden() for i in range(n))
        # 状态过滤：失败 -> 只剩 a1
        win.cmb_qfilter.setCurrentText("失败")
        vis = [win.lst_queue.item(i).isHidden() for i in range(n)]
        assert vis.count(False) == 1, f"失败状态应只剩 1 项可见: {vis}"
        # 关键词：knight 命中 a1
        win.cmb_qfilter.setCurrentText("全部")
        win.ed_qsearch.setText("knight")
        vis = [win.lst_queue.item(i).isHidden() for i in range(n)]
        assert vis.count(False) == 1, f"knight 应命中 1 项: {vis}"
        # 顺序子序列：knigt -> knight
        win.ed_qsearch.setText("knigt")
        vis = [win.lst_queue.item(i).isHidden() for i in range(n)]
        assert vis.count(False) == 1, "子序列应命中 knight"
        # 无命中
        win.ed_qsearch.setText("zzz")
        vis = [win.lst_queue.item(i).isHidden() for i in range(n)]
        assert vis.count(False) == 0
        # 清空复原 + 计数标签
        win.ed_qsearch.setText("")
        win.cmb_qfilter.setCurrentText("全部")
        assert all(not win.lst_queue.item(i).isHidden() for i in range(n))
        assert "2 / 共 2" in win.lbl_qcount.text(), win.lbl_qcount.text()
        print("[OK] 7. 队列筛选：状态过滤 + 关键词模糊 + 计数标签")

        # ---------- 8. 打开输出目录（打桩 os.startfile） ----------
        from pixelforge.ui import main_window as mw
        target = tmp / "outdir_probe"
        win.settings.output_dir = str(target)
        calls = []
        orig_startfile = getattr(mw.os, "startfile", None)
        mw.os.startfile = lambda p: calls.append(p)
        win._open_output_dir()
        assert calls and Path(calls[0]) == target, f"应打开 {target}: {calls}"
        assert target.is_dir(), "输出目录应被自动创建"
        if orig_startfile is not None:
            mw.os.startfile = orig_startfile
        print("[OK] 8. 打开输出目录：自动创建目录并调起系统文件管理器")

        # ---------- 9. 上传素材收编：外部图 → 本地资产 uploads/ ----------
        ext = Path(tempfile.mkdtemp(prefix="pf_up_")) / "hero.png"
        _png(ext)
        n_assets = win.assets_panel.lst.count()
        win._on_basic_refs_dropped([str(ext)])
        assert win._basic_ref_paths, "拖入后参考图列表应有内容"
        ref0 = Path(win._basic_ref_paths[0])
        assert ref0.parent.name == "uploads" and ref0.is_file(), \
            f"参考图应指向收编副本: {ref0}"
        assert ref0 != ext, "不应直接引用库外原文件"
        win.assets_panel.refresh()
        assert win.assets_panel.lst.count() == n_assets + 1, \
            f"资产库应新增 1 项: {win.assets_panel.lst.count()}"
        assert win.assets_panel._load_index()["categories"][
            win.assets_panel._rel(ref0)] == "local", "应登记为本地资产"
        print("[OK] 9. 上传素材收编：拖入外部图 → 本地资产 uploads/ + 参考图指向副本")

        # ---------- 10. 生成前参数摘要确认 ----------
        from PySide6.QtWidgets import QCheckBox, QDialog
        s = win._collect_settings()
        rows_g = win._gen_summary_rows(s, "gpt", refs=2, actions=3)
        keys_g = dict(rows_g)
        assert keys_g["引擎"] == "云端 GPT-Image" and keys_g["参考图"] == "2 张" \
            and keys_g["动作数"] == "3" and "画质" in keys_g and "线路" in keys_g, \
            f"GPT 摘要行缺失: {keys_g}"
        rows_l = win._gen_summary_rows(s, "local", refs=1, actions=8)
        keys_l = dict(rows_l)
        assert keys_l["引擎"] == "本地 ComfyUI" and "Checkpoint" in keys_l \
            and "种子" in keys_l and keys_l["动作数"] == "8", \
            f"本地摘要行缺失: {keys_l}"
        # 跳过标志 → 不弹窗直接放行
        win._skip_gen_confirm = True
        assert win._confirm_generate_summary(s, "gpt") is True
        # 弹窗：取消 → False；确认+勾选 → True 且之后永久跳过
        win._skip_gen_confirm = False
        results = {"n": 0}
        orig_exec = QDialog.exec
        def _exec_reject(self):
            results["n"] += 1
            return 0
        QDialog.exec = _exec_reject
        try:
            assert win._confirm_generate_summary(s, "gpt") is False
            def _exec_accept(self):
                results["n"] += 1
                self.findChildren(QCheckBox)[0].setChecked(True)
                return 1
            QDialog.exec = _exec_accept
            assert win._confirm_generate_summary(s, "gpt") is True
            assert results["n"] == 2, f"应弹窗 2 次: {results['n']}"
            assert win._confirm_generate_summary(s, "local") is True
            assert results["n"] == 2, "勾选不再提示后不应再弹窗"
        finally:
            QDialog.exec = orig_exec
            win._skip_gen_confirm = False
        print("[OK] 10. 生成前参数摘要确认：GPT/本地行齐全 + 取消拦截 + 不再提示生效")

        # ---------- 11. 历史任务并排对比（A/B 复用） ----------
        from PySide6.QtWidgets import QDialog
        ra = tmp / "cmp_a.png"
        rb = tmp / "cmp_b.png"
        cv2.imwrite(str(ra), np.full((12, 16, 4), (30, 60, 200, 255), np.uint8))
        cv2.imwrite(str(rb), np.full((12, 16, 4), (200, 120, 30, 255), np.uint8))
        for rid, p in (("a1", ra), ("a2", rb)):
            cfgmod.upsert_queue_record({"id": rid, "results": [str(p)]})
        want = {"a1": None, "a2": None}
        for i in range(win.lst_queue.count()):
            it = win.lst_queue.item(i)
            tid = it.data(Qt.UserRole + 2)
            if tid in want:
                want[tid] = it
        assert all(want.values()), f"应能定位两条队列项: {want}"
        cap = {}
        orig_exec = QDialog.exec
        QDialog.exec = lambda self: (cap.setdefault("dlg", self), 0)[1]
        try:
            win._queue_compare([want["a1"], want["a2"]])
        finally:
            QDialog.exec = orig_exec
        dlg = cap.get("dlg")
        assert dlg is not None, "应弹出对比对话框"
        assert "历史任务对比" in dlg.windowTitle(), dlg.windowTitle()
        # 无产物 → 不弹窗
        cfgmod.upsert_queue_record({"id": "a1", "results": []})
        cap.clear()
        QDialog.exec = lambda self: (cap.setdefault("dlg", self), 0)[1]
        try:
            win._queue_compare([want["a1"], want["a2"]])
        finally:
            QDialog.exec = orig_exec
        assert cap.get("dlg") is None, "缺产物时应拦截不弹窗"
        print("[OK] 11. 历史任务对比：两条选中产物 A/B 并排 + 缺产物拦截")
        win.close()
    finally:
        cfgmod.CONFIG_FILE, cfgmod.GEN_QUEUE_FILE = old_cfg, old_q

    print("\n持久化生成队列全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
