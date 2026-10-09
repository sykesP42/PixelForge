"""Batch Forge 多角色批量生成面板：角色表 × 动作 × 数量的任务矩阵。

用法：主窗口「批量工坊」标签页内嵌。
流程：参考图上传（每角色一次）→ /free 清显存 → 串行提交任务
（失败同 seed 重试一次，仍败记入失败列表）→ 归档
outputs/<project>/<代号>/<代号>_<动作>_<seed>.png
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from pixelforge.batch_forge import (
    ForgeRole,
    ForgeTask,
    build_tasks,
    sanitize_code,
)
from pixelforge.comfyui_client import ComfyUIClient, build_workflow
from pixelforge.config import PROJECT_ROOT, Settings

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def _to_pixmap(arr) -> QPixmap:
    if arr is None:
        return QPixmap()
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class _ForgeSignals(QObject):
    progress = Signal(int, int, str)        # 当前序号, 总数, 描述
    thumb = Signal(str)                     # 当前任务完成图路径
    log = Signal(str)
    task_failed = Signal(str, str)          # 代号_动作_seed, 错误
    done = Signal(int, int)                 # 成功数, 总数


class _ForgeWorker(QRunnable):
    """串行执行任务矩阵：/free → 提交 → WS 等待 → 归档；失败重试一次。"""

    def __init__(self, settings: Settings, tasks: list[ForgeTask],
                 roles: dict[str, ForgeRole], template: dict | None,
                 stop_flag: dict, retries: int = 1):
        super().__init__()
        self.s = settings
        self.tasks = tasks
        self.roles = roles
        self.template = template
        self.stop_flag = stop_flag
        self.retries = retries
        self.sig = _ForgeSignals()
        self._server_cache: dict[str, str] = {}   # ref_path -> server name

    def run(self) -> None:
        client = ComfyUIClient.from_settings(self.s)
        if not client.is_online():
            for t in self.tasks:
                self.sig.task_failed.emit(self._tag(t), "ComfyUI 未连接")
            self.sig.done.emit(0, len(self.tasks))
            return
        ok = 0
        total = len(self.tasks)
        for i, t in enumerate(self.tasks):
            if self.stop_flag.get("stop"):
                self.sig.log.emit(f"任务 {self._tag(t)} 跳过（用户停止）")
                continue
            self.sig.progress.emit(i, total,
                                   f"{t.role_code} ({t.index}) 动作:{t.action_name}")
            client.free_memory()  # 每任务前清显存（老版本无 /free 则静默）
            err = self._run_one(client, t)
            for attempt in range(1, self.retries + 1):
                if err is None:
                    break
                self.sig.log.emit(f"{self._tag(t)} 失败（第 {attempt} 次尝试）: "
                                  f"{err}，用相同 seed 重试")
                t.attempts += 1
                err = self._run_one(client, t)
            if err is None:
                ok += 1
                self.sig.thumb.emit(str(t.out_path))
            else:
                t.error = err
                self.sig.task_failed.emit(self._tag(t), err)
        self.sig.done.emit(ok, total)

    @staticmethod
    def _tag(t: ForgeTask) -> str:
        return f"{t.role_code}_{t.action_name}_{t.seed}"

    def _ref_server(self, client: ComfyUIClient, role: ForgeRole) -> str:
        """参考图每角色只上传一次（含重试场景复用）。中文路径安全读取。"""
        if role.server_image:
            return role.server_image
        arr = cv2.imread(role.ref_path, cv2.IMREAD_UNCHANGED)
        if arr is None:  # cv2.imread 中文路径在 Windows 会失败
            data = np.fromfile(role.ref_path, np.uint8)
            arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) \
                if data.size else None
        if arr is None:
            raise ComfyUIError_IO(f"参考图读取失败: {role.ref_path}")
        name = f"PF_Temp_{role.code}_{abs(hash(role.ref_path)) % 9999}.png"
        role.server_image = client.upload_image(arr, name)
        return role.server_image

    def _run_one(self, client: ComfyUIClient, t: ForgeTask) -> str | None:
        """执行单个任务；成功返回 None，失败返回错误信息。"""
        try:
            role = self.roles[t.role_code]
            server_ref = self._ref_server(client, role)
            workflow = build_workflow(
                self.template, checkpoint=self.s.checkpoint,
                prompt=t.prompt, negative_prompt=self.s.negative_prompt,
                seed=t.seed, width=self.s.width, height=self.s.height,
                steps=self.s.steps, cfg=self.s.cfg,
                lora_name=self.s.lora_name, lora_strength=self.s.lora_strength)
            for node in workflow.values():
                ins = node.get("inputs", {})
                for k, v in list(ins.items()):
                    if v == "__IMAGE__":
                        ins[k] = server_ref
                    elif v == "__DENOISE__":
                        ins[k] = self.s.i2i_denoise
            prompt_id = client.queue_prompt(workflow)
            produced = client._wait_for_result(prompt_id, timeout=900)
            t.out_path.parent.mkdir(parents=True, exist_ok=True)
            first = None
            for filename, subfolder, ftype in produced:
                qs = f"filename={filename}&subfolder={subfolder}&type={ftype}"
                data = client._get(f"/view?{qs}", timeout=120)
                if first is None:
                    first = data
            if first is None:
                return "任务完成但没有任何输出图片"
            t.out_path.write_bytes(first)
            return None
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"


class ComfyUIError_IO(RuntimeError):
    pass


class ForgePanel(QWidget):
    state_changed = Signal(str)

    def __init__(self, settings_provider, parent=None):
        super().__init__(parent)
        self._settings_provider = settings_provider
        self._busy = False
        self._stop_flag = {"stop": False}
        self._failed: list[tuple[str, str]] = []   # (tag, error)
        self._last_tasks: list[ForgeTask] = []
        self._build_ui()
        self._refresh_templates()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        lay = QHBoxLayout(self)

        # 左：角色表 + 主提示词 + 参数
        left = QVBoxLayout()
        left.addWidget(QLabel(
            "1) 添加参考图（可多选）→ 2) 填代号与角色描述 → 3) 主提示词用 "
            "[动作] 和 [角色描述] 占位 → 4) 开始"))
        bar_r = QHBoxLayout()
        b_add = QPushButton("添加参考图...")
        b_add.clicked.connect(self._add_refs)
        bar_r.addWidget(b_add)
        b_del = QPushButton("删除选中")
        b_del.clicked.connect(self._del_role)
        bar_r.addWidget(b_del)
        bar_r.addStretch(1)
        left.addLayout(bar_r)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["预览", "代号", "角色描述", "张数"])
        self.table.setColumnWidth(0, 72)
        self.table.setColumnWidth(1, 110)
        self.table.horizontalHeader().setStretchLastSection(False)
        try:
            from PySide6.QtWidgets import QHeaderView
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        except Exception:
            pass
        left.addWidget(self.table, 1)

        left.addWidget(QLabel("主提示词（[动作] / [角色描述] 为占位符）"))
        self.ed_master = QLineEdit()
        self.ed_master.setPlaceholderText(
            "masterpiece, best quality, pixel art, [动作], [角色描述], "
            "solid green background")
        left.addWidget(self.ed_master)

        left.addWidget(QLabel("动作列表（每行一个：动作名 / 动作名: 自定义描述）"))
        self.ed_actions = QLineEdit()
        self.ed_actions.setPlaceholderText("idle, walk, attack（逗号分隔，可用内置名或 name: 描述）")
        left.addWidget(self.ed_actions)

        grid = QHBoxLayout()
        grid.addWidget(QLabel("每动作张数"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 32)
        self.spin_count.setValue(2)
        grid.addWidget(self.spin_count)
        grid.addWidget(QLabel("Seed"))
        self.cmb_seed = QComboBox()
        self.cmb_seed.addItems(["fixed", "increment", "random"])
        self.cmb_seed.setToolTip("fixed=风格统一 / increment=递增防重复 / random=抽卡")
        grid.addWidget(self.cmb_seed)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2 ** 31 - 1)
        self.spin_seed.setValue(1024)
        grid.addWidget(self.spin_seed)
        left.addLayout(grid)

        grid2 = QHBoxLayout()
        grid2.addWidget(QLabel("项目名"))
        self.ed_project = QLineEdit("batch_forge")
        grid2.addWidget(self.ed_project, 1)
        grid2.addWidget(QLabel("模板"))
        self.cmb_tpl = QComboBox()
        self.cmb_tpl.setToolTip("选含 __IMAGE__ 的 img2img 类模板（如 qwen_i2i）")
        grid2.addWidget(self.cmb_tpl, 1)
        left.addLayout(grid2)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("🔥 开始批量生成")
        self.btn_run.setProperty("accent", True)
        self.btn_run.clicked.connect(self.start_forge)
        run_row.addWidget(self.btn_run)
        b_stop = QPushButton("⏹ 停止")
        b_stop.setProperty("danger", True)
        b_stop.clicked.connect(self._request_stop)
        run_row.addWidget(b_stop)
        b_zip = QPushButton("打包 ZIP")
        b_zip.clicked.connect(self._export_zip)
        run_row.addWidget(b_zip)
        run_row.addStretch(1)
        left.addLayout(run_row)

        self.progress = QLabel("就绪")
        left.addWidget(self.progress)
        left.addWidget(QLabel("当前任务"))
        self.lbl_thumb = QLabel("（无）")
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setMinimumSize(240, 240)
        self.lbl_thumb.setStyleSheet("background:#202225; color:#9aa0a6;")
        left.addWidget(self.lbl_thumb)
        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)

        # 右：日志 + 失败列表 + 画廊
        right = QVBoxLayout()
        right.addWidget(QLabel("生成画廊（点击放大）"))
        self.lst_gallery = QListWidget()
        self.lst_gallery.setViewMode(QListWidget.IconMode)
        self.lst_gallery.setIconSize(QPixmap(96, 96).size())
        self.lst_gallery.setResizeMode(QListWidget.Adjust)
        self.lst_gallery.itemDoubleClicked.connect(self._open_image)
        right.addWidget(self.lst_gallery, 2)
        right.addWidget(QLabel("失败任务（可一键重试）"))
        self.lst_failed = QListWidget()
        right.addWidget(self.lst_failed, 1)
        b_retry = QPushButton("重试失败任务")
        b_retry.clicked.connect(self._retry_failed)
        right.addWidget(b_retry)
        right_widget = QWidget()
        right_widget.setLayout(right)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_widget)
        split.addWidget(right_widget)
        split.setSizes([520, 420])
        lay.addWidget(split)

    def _refresh_templates(self) -> None:
        self.cmb_tpl.clear()
        for f in sorted(WORKFLOWS_DIR.glob("*.json")):
            self.cmb_tpl.addItem(f.stem, str(f))
        i = self.cmb_tpl.findText("qwen_i2i")
        if i >= 0:
            self.cmb_tpl.setCurrentIndex(i)

    # ---------------- 角色表 ----------------

    def _add_refs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择角色参考图（可多选）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        used = {self.table.item(r, 1).text() for r in range(self.table.rowCount())
                if self.table.item(r, 1)}
        for f in files:
            r = self.table.rowCount()
            self.table.insertRow(r)
            code = sanitize_code(Path(f).stem, used)
            it = QTableWidgetItem("")
            arr = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if arr is not None:
                it.setData(Qt.DecorationRole, _to_pixmap(arr).scaled(
                    64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            it.setToolTip(f)  # 参考图路径存 tooltip（_roles 读取）
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(code))
            self.table.setItem(r, 2, QTableWidgetItem(""))
            cnt = QTableWidgetItem(str(self.spin_count.value()))
            self.table.setItem(r, 3, cnt)
        if files:
            self.state_changed.emit(f"已添加 {len(files)} 个角色参考图")

    def _del_role(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _roles(self) -> list[ForgeRole]:
        out = []
        for r in range(self.table.rowCount()):
            code_it = self.table.item(r, 1)
            desc_it = self.table.item(r, 2)
            cnt_it = self.table.item(r, 3)
            if not code_it:
                continue
            dec = self.table.item(r, 0)
            ref = ""
            if dec is not None and dec.toolTip():
                ref = dec.toolTip()
            count = int(cnt_it.text()) if cnt_it and cnt_it.text().isdigit() \
                else self.spin_count.value()
            out.append(ForgeRole(code=code_it.text().strip(),
                                 ref_path=ref, count=max(1, count),
                                 desc=(desc_it.text().strip() if desc_it else "")))
        return [x for x in out if x.ref_path]

    # ---------------- 生成 ----------------

    def _load_template(self) -> dict | None:
        f = self.cmb_tpl.currentData()
        if not f:
            return None
        try:
            return json.loads(Path(f).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Batch Forge", f"模板解析失败: {e}")
            return None

    def start_forge(self) -> None:
        if self._busy:
            return
        s: Settings = self._settings_provider()
        roles = self._roles()
        if not roles:
            QMessageBox.information(self, "Batch Forge", "请先添加角色参考图")
            return
        if not s.checkpoint:
            QMessageBox.warning(self, "Batch Forge", "请先在「生成配置」选择 Checkpoint")
            return
        template = self._load_template()
        if template is None:
            return
        if not any(v == "__IMAGE__" for n in template.values()
                   for v in n.get("inputs", {}).values()):
            QMessageBox.warning(
                self, "Batch Forge",
                "所选模板不含 __IMAGE__ 占位符（参考图无法注入）。\n"
                "请选择 img2img 类模板（如 qwen_i2i）。")
            return
        actions = [l for l in self.ed_actions.text().split(",") if l.strip()]
        tasks = build_tasks(
            roles, actions, self.cmb_seed.currentText(), self.spin_seed.value(),
            self.ed_master.text().strip(), PROJECT_ROOT / "outputs" / "forge",
            self.ed_project.text().strip() or "batch_forge")
        if not tasks:
            QMessageBox.information(self, "Batch Forge", "任务矩阵为空")
            return
        # 参考图路径写进表格项 tooltip（_roles 从 tooltip 读）
        self._failed.clear()
        self.lst_failed.clear()
        self.lst_gallery.clear()
        self._busy = True
        self._stop_flag = {"stop": False}
        self.btn_run.setEnabled(False)
        worker = _ForgeWorker(s, tasks,
                              {r.code: r for r in roles}, template,
                              self._stop_flag)
        self._last_tasks = tasks
        worker.sig.progress.connect(self._on_progress)
        worker.sig.thumb.connect(self._on_thumb)
        worker.sig.log.connect(lambda m: self.state_changed.emit(f"[Forge] {m}"))
        worker.sig.task_failed.connect(self._on_task_failed)
        worker.sig.done.connect(self._on_done)
        QThreadPool.globalInstance().start(worker)
        self.state_changed.emit(f"[Forge] 批量生成开始：{len(tasks)} 个任务，"
                                "串行执行")

    def _request_stop(self) -> None:
        self._stop_flag["stop"] = True
        self.state_changed.emit("[Forge] 将在当前任务完成后停止")

    def _on_progress(self, i: int, total: int, desc: str) -> None:
        self.progress.setText(f"正在生成: {desc}  ({i + 1}/{total})")

    def _on_thumb(self, path: str) -> None:
        pm = QPixmap(path)
        if not pm.isNull():
            self.lbl_thumb.setPixmap(pm.scaled(
                self.lbl_thumb.width(), self.lbl_thumb.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            it = QListWidgetItem(Path(path).name)
            it.setIcon(QIcon(pm))
            it.setData(Qt.UserRole, path)
            self.lst_gallery.addItem(it)

    def _on_task_failed(self, tag: str, err: str) -> None:
        self._failed.append((tag, err))
        self.lst_failed.addItem(f"{tag}  ✘ {err}")

    def _on_done(self, ok: int, total: int) -> None:
        self._busy = False
        self.btn_run.setEnabled(True)
        self.progress.setText(f"完成：{ok}/{total} 成功"
                              + (f"，{len(self._failed)} 个失败（可重试）"
                                 if self._failed else ""))
        self.state_changed.emit(f"[Forge] 批量生成完成：{ok}/{total} 成功")
        if ok > 0:
            out_dir = PROJECT_ROOT / "outputs" / "forge" / \
                (self.ed_project.text().strip() or "batch_forge")
            try:
                os.startfile(str(out_dir))  # noqa: S606  自动打开输出目录
            except Exception:
                pass

    # ---------------- 重试 / 导出 ----------------

    def _retry_failed(self) -> None:
        if self._busy:
            return
        if not self._failed:
            QMessageBox.information(self, "Batch Forge", "没有失败任务")
            return
        tags = {t for t, _ in self._failed}
        retry_tasks = [t for t in self._last_tasks
                       if f"{t.role_code}_{t.action_name}_{t.seed}" in tags]
        self._failed.clear()
        self.lst_failed.clear()
        s: Settings = self._settings_provider()
        roles = {r.code: r for r in self._roles()}
        template = self._load_template()
        if template is None or not retry_tasks:
            return
        self._busy = True
        self.btn_run.setEnabled(False)
        worker = _ForgeWorker(s, retry_tasks, roles, template,
                              self._stop_flag)
        worker.sig.progress.connect(self._on_progress)
        worker.sig.thumb.connect(self._on_thumb)
        worker.sig.log.connect(lambda m: self.state_changed.emit(f"[Forge] {m}"))
        worker.sig.task_failed.connect(self._on_task_failed)
        worker.sig.done.connect(self._on_done)
        QThreadPool.globalInstance().start(worker)
        self.state_changed.emit(f"[Forge] 重试 {len(retry_tasks)} 个失败任务")

    def _export_zip(self) -> None:
        project = self.ed_project.text().strip() or "batch_forge"
        src = PROJECT_ROOT / "outputs" / "forge" / project
        if not src.is_dir():
            QMessageBox.information(self, "Batch Forge", "该项目目录为空")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "导出 ZIP", f"{project}.zip", "ZIP (*.zip)")
        if not out:
            return
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(src.parent))
        self.state_changed.emit(f"[Forge] 已打包 {project}.zip")

    def _open_image(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path and Path(path).is_file():
            import os
            os.startfile(path)  # noqa: S606

    def status(self, msg: str) -> None:
        self.state_changed.emit(msg)
