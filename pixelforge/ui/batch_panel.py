"""批量动作队列面板：每行一个动作，批量生成+裁切+统一调色板，输出到 outputs/batch/。

用法：主窗口"批量动作"标签页内嵌。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMenu, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from pixelforge.comfyui_client import ComfyUIClient, build_workflow, random_seed
from pixelforge.config import PROJECT_ROOT, Settings
from pixelforge.motion_templates import get_template, template_names
from pixelforge.processing.image_processor import ImageProcessor, frame_hull


class _BatchSignals(QObject):
    action_start = Signal(int)
    action_done = Signal(int, bool, str)
    batch_done = Signal(int, int)  # 成功数, 总数


class _BatchWorker(QRunnable):
    """顺序执行每个动作：生成 -> 自动裁切对齐 -> 批内统一调色板 -> 导出。"""

    def __init__(self, settings: Settings, actions: list[tuple[str, str]],
                 out_root: Path, stop_flag: dict, resume: bool = False):
        super().__init__()
        self.s = settings
        self.actions = actions
        self.out_root = out_root
        self.stop_flag = stop_flag
        self.resume = resume  # 断点续跑：atlas.json 已存在的动作直接跳过
        self.shared_palette: list | None = None  # 批内统一调色板
        self.sig = _BatchSignals()

    def run(self) -> None:
        client = ComfyUIClient.from_settings(self.s)
        if not client.is_online():
            for i in range(len(self.actions)):
                self.sig.action_done.emit(i, False, "ComfyUI 未连接")
            self.sig.batch_done.emit(0, len(self.actions))
            return
        ok = 0
        for i, (name, prompt) in enumerate(self.actions):
            if self.stop_flag.get("stop"):
                self.sig.action_done.emit(i, False, "已跳过（用户停止）")
                continue
            if self.resume and (self.out_root / name / "atlas.json").is_file():
                self.sig.action_done.emit(i, True, "已完成（断点续跑跳过）")
                ok += 1
                continue
            self.sig.action_start.emit(i)
            try:
                frames = self._run_one(client, i, name, prompt)
                self.sig.action_done.emit(i, True, f"{len(frames)} 帧")
                ok += 1
            except Exception as e:  # noqa: BLE001
                self.sig.action_done.emit(i, False, f"{type(e).__name__}: {e}")
        self.sig.batch_done.emit(ok, len(self.actions))

    def _run_one(self, client: ComfyUIClient, i: int, name: str, prompt: str) -> int:
        s = self.s
        seed = s.seed if s.lock_seed else random_seed()
        full_prompt = f"{s.prompt}, {prompt}".strip(", ")
        workflow = build_workflow(
            self._template(),
            checkpoint=s.checkpoint, prompt=full_prompt,
            negative_prompt=s.negative_prompt, seed=seed,
            width=s.width, height=s.height, steps=s.steps, cfg=s.cfg,
            lora_name=s.lora_name, lora_strength=s.lora_strength)
        out_dir = self.out_root / name
        paths = client.generate(workflow, out_dir, timeout=900)
        return self._postprocess(paths[0], out_dir)

    @staticmethod
    def _template() -> dict:
        tpl = Path(__file__).resolve().parent.parent / "workflows" / "txt2img_basic.json"
        return json.loads(tpl.read_text("utf-8"))

    def _postprocess(self, sheet_path: Path, out_dir: Path) -> int:
        """生成结果 -> 去背景 -> 裁切对齐 -> 批内调色板 -> 硬化/外扩 -> 帧 + atlas。

        默认开启边缘硬化（去光晕）与 1px 边缘外扩（防引擎采样串色）。
        """
        proc = ImageProcessor(align_mode="feet_center",
                              harden_alpha=True, bleed_px=1)
        if self.shared_palette:
            proc.set_locked_palette(self.shared_palette)
        bgra = cv2.imread(str(sheet_path), cv2.IMREAD_UNCHANGED)
        if bgra.ndim == 2:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
        elif bgra.shape[2] == 3:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
        frames = proc.run_pipeline(bgra)
        if not self.shared_palette and proc.last_palette:
            self.shared_palette = proc.last_palette  # 首动作调色板锁定给整批
        meta = []
        for i, f in enumerate(frames):
            fn = f"{out_dir.name}_{i:03d}.png"
            cv2.imwrite(str(out_dir / fn), f.image)
            meta.append({"file": fn, "x": f.rect.x, "y": f.rect.y,
                         "width": f.rect.width, "height": f.rect.height,
                         "pivotX": f.rect.pivot_x, "pivotY": 1.0,
                         "hull": frame_hull(f.image)})
        atlas = {"name": out_dir.name, "frame_count": len(meta), "frames": meta}
        (out_dir / "atlas.json").write_text(
            json.dumps(atlas, ensure_ascii=False, indent=2), "utf-8")
        return len(frames)


class BatchPanel(QWidget):
    state_changed = Signal(str)

    def __init__(self, settings_provider, parent=None):
        super().__init__(parent)
        self._settings_provider = settings_provider  # callable -> Settings
        self._busy = False
        self._stop_flag = {"stop": False}
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "每行一个动作：动作名（输出子目录）+ 动作提示词（拼接在“生成配置”的角色描述后）。\n"
            "开始后按顺序生成，自动裁切对齐并统一调色板（批内颜色一致）。输出到 outputs/batch/。"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["动作名", "动作提示词", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for name, prompt in (("待机", "idle animation, standing, breathing"),
                             ("行走", "walk cycle, side view"),
                             ("射击", "shooting pose, muzzle flash")):
            self._add_row(name, prompt)
        lay.addWidget(self.table, 1)

        bar = QHBoxLayout()
        b_add = QPushButton("添加动作")
        b_add.clicked.connect(lambda: self._add_row("新动作", ""))
        bar.addWidget(b_add)
        b_tpl = QPushButton("动作模板...")
        b_tpl.setToolTip("一键填入方向/战斗动作模板（下/上/左/右四向等）")
        tpl_menu = QMenu(self)
        for name in template_names():
            tpl_menu.addAction(name, lambda n=name: self.apply_template(n))
        b_tpl.setMenu(tpl_menu)
        bar.addWidget(b_tpl)
        b_del = QPushButton("删除选中")
        b_del.clicked.connect(self._del_row)
        bar.addWidget(b_del)
        self.btn_run = QPushButton("▶ 开始批量生成")
        self.btn_run.setProperty("accent", True)
        self.btn_run.clicked.connect(self.start_batch)
        bar.addWidget(self.btn_run)
        b_stop = QPushButton("⏹ 停止")
        b_stop.setProperty("danger", True)
        b_stop.clicked.connect(self._request_stop)
        bar.addWidget(b_stop)
        self.chk_resume = QCheckBox("断点续跑（跳过已完成动作）")
        self.chk_resume.setChecked(True)
        self.chk_resume.setToolTip("输出目录已有 atlas.json 的动作不再重新生成；\n"
                                   "中途失败/停止后直接重新开始即可接续")
        bar.addWidget(self.chk_resume)
        bar.addStretch(1)
        lay.addLayout(bar)

    def apply_template(self, name: str) -> int:
        """把动作模板逐行填入表格，返回新增行数（纯逻辑，可测）。"""
        rows = get_template(name)
        for action, prompt in rows:
            self._add_row(action, prompt)
        self.status(f"已按模板「{name}」填入 {len(rows)} 个动作")
        return len(rows)

    def _add_row(self, name: str, prompt: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(prompt))
        self.table.setItem(r, 2, QTableWidgetItem("待执行"))

    def _del_row(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _actions(self) -> list[tuple[str, str]]:
        out = []
        for r in range(self.table.rowCount()):
            n_it, p_it = self.table.item(r, 0), self.table.item(r, 1)
            name = (n_it.text() if n_it else "").strip()
            prompt = (p_it.text() if p_it else "").strip()
            if name and prompt:
                out.append((name, prompt))
        return out

    def start_batch(self) -> None:
        if self._busy:
            return
        s: Settings = self._settings_provider()
        actions = self._actions()
        if not actions:
            QMessageBox.information(self, "批量动作", "没有可执行的动作行")
            return
        if not s.checkpoint:
            QMessageBox.warning(self, "批量动作", "请先在“生成配置”中选择 Checkpoint")
            return
        if not s.prompt:
            QMessageBox.warning(self, "批量动作",
                                "请先在“生成配置”填写角色描述（作为提示词前缀）")
            return
        self._busy = True
        self._stop_flag = {"stop": False}
        self.btn_run.setEnabled(False)
        worker = _BatchWorker(s, actions, PROJECT_ROOT / "outputs" / "batch",
                              self._stop_flag, resume=self.chk_resume.isChecked())
        worker.sig.action_start.connect(lambda i: self._set_status(i, "生成中..."))
        worker.sig.action_done.connect(self._on_action_done)
        worker.sig.batch_done.connect(self._on_batch_done)
        QThreadPool.globalInstance().start(worker)
        self.status(f"批量生成开始：{len(actions)} 个动作")

    def _request_stop(self) -> None:
        self._stop_flag["stop"] = True
        self.status("将在当前动作完成后停止")

    def _set_status(self, i: int, text: str) -> None:
        it = self.table.item(i, 2)
        if it:
            it.setText(text)

    def _on_action_done(self, i: int, ok: bool, msg: str) -> None:
        it = self.table.item(i, 2)
        if it:
            it.setText(("✔ " if ok else "✘ ") + msg)

    def _on_batch_done(self, ok: int, total: int) -> None:
        self._busy = False
        self.btn_run.setEnabled(True)
        self.status(f"批量生成完成：{ok}/{total} 个动作成功")

    def status(self, msg: str) -> None:
        self.state_changed.emit(msg)
