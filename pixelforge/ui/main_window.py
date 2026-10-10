"""PySide6 主窗口 — 阶段1 MVP。

布局（PRD §5）：
  左：环境与引擎配置 + 生成参数
  中：生成结果预览
  右：任务队列 + 日志
  底：连接状态灯 / 显存信息

所有阻塞操作（conda、HTTP、WebSocket）均在线程池执行，不卡 UI。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import (
    QCursor, QImage, QKeySequence, QPainter, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QApplication, QComboBox, QDialog,
    QDockWidget, QDoubleSpinBox,
    QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSlider, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QToolBar,
    QVBoxLayout, QWidget,
)

from pixelforge.comfyui_client import (
    ComfyUIError,
    ComfyUIClient,
    build_workflow,
    random_seed,
)
from pixelforge import __version__
from pixelforge.conda_manager import find_conda, launch_comfyui, list_envs
from pixelforge.config import (
    PROJECT_ROOT,
    Settings,
    append_gen_history,
    load_gen_history,
    load_profiles,
    load_prompt_templates,
    save_profiles,
    save_prompt_templates,
)
from pixelforge.ui.image_lab import ImageLab
from pixelforge.ui.anim_player import AnimPlayer
from pixelforge.ui.batch_panel import BatchPanel
from pixelforge.ui.review_panel import ReviewWidget
from pixelforge.ui.assets_panel import AssetLibraryPanel
from pixelforge.ui.forge_panel import ForgePanel

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


# ---------------------------------------------------------------- 线程工作单元

class _Signals(QObject):
    progress = Signal(str)
    conn = Signal(bool, str)          # 在线?, 信息（显卡/显存）
    envs = Signal(list, str)          # [(name, path)...], conda_exe
    checkpoints = Signal(list)
    cnets = Signal(list)              # ControlNet 模型列表
    done = Signal(dict)               # {"paths": [...], "seed": int}
    failed = Signal(str)
    processed = Signal(list, dict)    # [帧PNG路径], atlas dict


def _log_line(msg: str) -> str:
    return f"[{datetime.now():%H:%M:%S}] {msg}"


class _ConnCheckWorker(QRunnable):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.sig = _Signals()

    def run(self) -> None:
        client = ComfyUIClient.from_settings(self.s)
        if not client.is_online():
            self.sig.conn.emit(False, "")
            return
        try:
            stats = client.system_stats()
            dev = (stats.get("devices") or [{}])[0]
            free = dev.get("vram_free", 0) / 1024**3
            total = dev.get("vram_total", 0) / 1024**3
            info = f"{dev.get('name', '?')} | 显存 {free:.1f}/{total:.1f} GB"
        except Exception:  # noqa: BLE001
            info = "已连接"
        self.sig.conn.emit(True, info)


class _EnvRefreshWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.sig = _Signals()

    def run(self) -> None:
        try:
            conda = find_conda()
            if not conda:
                self.sig.envs.emit([], "")
                self.sig.failed.emit("未找到 Conda，请确认已安装并配置 PATH 或 CONDA_EXE 环境变量")
                return
            self.sig.envs.emit(list_envs(conda), conda)
        except Exception as e:  # noqa: BLE001
            self.sig.failed.emit(f"Conda 环境识别失败: {e}")


class _CheckpointsWorker(QRunnable):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.sig = _Signals()

    def run(self) -> None:
        client = ComfyUIClient.from_settings(self.s)
        self.sig.checkpoints.emit(client.get_checkpoints())


class _ControlNetsWorker(QRunnable):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.sig = _Signals()

    def run(self) -> None:
        client = ComfyUIClient.from_settings(self.s)
        self.sig.cnets.emit(client.get_controlnets())


class _PostProcessWorker(QRunnable):
    """后处理管线 Worker：精灵表 -> 帧序列 + atlas JSON。"""

    def __init__(self, image_path: str, palette_size: int = 24):
        super().__init__()
        self.image_path = image_path
        self.palette_size = palette_size
        self.sig = _Signals()

    def run(self) -> None:
        try:
            from pixelforge.processing.image_processor import ImageProcessor
            src = Path(self.image_path)
            out_dir = src.parent / "processed" / src.stem
            self.sig.progress.emit(f"后处理中: {src.name} ...")
            files = ImageProcessor(palette_size=self.palette_size).process(src, out_dir)
            atlas = json.loads(
                (out_dir / f"{src.stem}.atlas.json").read_text("utf-8"))
            self.sig.processed.emit([str(p) for p in files], atlas)
        except Exception as e:  # noqa: BLE001
            self.sig.failed.emit(f"后处理失败: {type(e).__name__}: {e}")


def _compose_prompt(prompt: str, negative: str) -> str:
    """统一提示词合成：负面以 Avoid: 拼入（GPT-Image 无独立负面参数）。"""
    p = (prompt or "").strip()
    neg = (negative or "").strip()
    if neg:
        p = f"{p}\nAvoid: {neg}"
    return p


class _GPTEngineWorker(QRunnable):
    """云端 GPT-Image 生成 worker：只生图（网格原图），切分交给人工。"""

    class Sig(QObject):
        progress = Signal(str)
        done = Signal(list)           # 网格图路径列表
        failed = Signal(str)

    def __init__(self, s: Settings, actions: list[str],
                 ref_paths: list[str] | None = None):
        super().__init__()
        self.s = s
        self.actions = actions
        self.ref_paths = ref_paths or []
        self.sig = self.Sig()
        self.cancelled = False      # 停止按钮置位，轮询间隙检查

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        from pixelforge.error_hints import is_retryable_error
        from pixelforge.gpt_engine import (
            CloudEngineError,
            generate_grid,
        )

        out_dir = self.s.output_path()
        refs = [p for p in self.ref_paths if Path(p).is_file()]
        grid_paths = []
        try:
            for i, action_desc in enumerate(self.actions):
                if self.cancelled:
                    raise CloudEngineError("⏹ 任务已停止")
                tag = action_desc.split(',')[0].strip()[:16] or f"a{i + 1}"
                self.sig.progress.emit(
                    f"[GPT] {i + 1}/{len(self.actions)}: {tag}")
                # 统一合成：负面词拼入（笨蛋模式已在 prompt 内含 Avoid 则不重复）
                base = self.s.prompt if "Avoid:" in self.s.prompt else \
                    _compose_prompt(self.s.prompt, self.s.negative_prompt)
                # 笨蛋模式动作即整段提示词，避免 "prompt — prompt" 重复
                full_prompt = (base if action_desc == self.s.prompt
                               else f"{base} — {action_desc}")
                grid = self._generate_with_retry(
                    generate_grid, full_prompt, refs, tag, out_dir)
                grid_paths.append(str(grid))
                self.sig.progress.emit(
                    f"[GPT] {tag} 完成: {Path(grid).name}")
            self.sig.done.emit(grid_paths)
        except CloudEngineError as e:
            self.sig.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.sig.failed.emit(f"{type(e).__name__}: {e}")

    def _generate_with_retry(self, generate_grid, full_prompt: str,
                             refs: list, tag: str, out_dir) -> str:
        """单个动作生成，网络类错误自动重试 2 次（指数退避 5s/10s，可中断）。"""
        from pixelforge.error_hints import is_retryable_error
        from pixelforge.gpt_engine import CloudEngineError
        max_retry = 2
        for attempt in range(max_retry + 1):
            try:
                return generate_grid(
                    self.s.gpt_api_key, self.s.gpt_base_url,
                    full_prompt,
                    refs, self.s.gpt_model, self.s.gpt_size, out_dir,
                    quality=getattr(self.s, "gpt_quality", "medium"),
                    oversea=bool(getattr(self.s, "gpt_oversea", False)),
                    background=getattr(self.s, "gpt_background", "auto"),
                    should_cancel=lambda: self.cancelled,
                    log=lambda m: self.sig.progress.emit(m))
            except CloudEngineError as e:
                if (self.cancelled or attempt >= max_retry
                        or not is_retryable_error(str(e))):
                    raise
                wait = 5 * (attempt + 1)
                self.sig.progress.emit(
                    f"[GPT] {tag} 网络类错误，{wait}s 后自动重试 "
                    f"{attempt + 1}/{max_retry}: {str(e)[:120]}")
                for _ in range(wait * 10):   # 可中断退避
                    if self.cancelled:
                        raise CloudEngineError("⏹ 任务已停止")
                    time.sleep(0.1)


class _GenerateWorker(QRunnable):
    """完整生成流程：连通检查 ->（必要时）Conda 拉起 ComfyUI -> 提交 -> WS 等待 -> 落盘。"""

    def __init__(self, settings: Settings, workflow_template: dict, conda_exe: str,
                 pose_paths: list | None = None, actions: list | None = None):
        super().__init__()
        self.s = settings
        self.template = workflow_template
        self.conda_exe = conda_exe
        self.pose_paths = pose_paths or []
        self.actions = actions or [""]
        self.sig = _Signals()

    def run(self) -> None:
        s = self.s
        client = ComfyUIClient.from_settings(s)
        try:
            if not client.is_online():
                if s.cloud_mode:
                    raise ComfyUIError(
                        "云端 ComfyUI 不可达：请检查网络、API 地址与密钥")
                self._ensure_engine(client)
            seed = s.seed if s.lock_seed else random_seed()
            workflow = build_workflow(
                self.template,
                checkpoint=s.checkpoint,
                prompt=s.prompt,
                negative_prompt=s.negative_prompt,
                seed=seed,
                width=s.width, height=s.height,
                steps=s.steps, cfg=s.cfg,
                lora_name=s.lora_name, lora_strength=s.lora_strength,
            )
            self.sig.progress.emit(f"已提交任务 seed={seed} {s.width}x{s.height} steps={s.steps}")
            if self.pose_paths and s.controlnet_name:
                self._run_pose_series(client, workflow, seed)
                return
            if s.ref_image:
                if self._grid_mode():
                    self._run_grid_series(client, seed)
                    return
                self._run_ref_series(client, seed)
                return
            if s.i2i_frames >= 2:
                self._run_series(client, workflow, seed)
                return
            t0 = time.monotonic()
            paths = self._generate_with_retry_local(client, workflow, s)
            self.sig.progress.emit(f"生成完成，耗时 {time.monotonic() - t0:.1f}s")
            self.sig.done.emit({"paths": [str(p) for p in paths], "seed": seed})
        except ComfyUIError as e:
            self.sig.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.sig.failed.emit(f"{type(e).__name__}: {e}")

    def _generate_with_retry_local(self, client, workflow, s) -> list:
        """本地单帧生成：网络类错误自动重试 2 次（5s/10s 退避）。"""
        from pixelforge.error_hints import is_retryable_error
        max_retry = 2
        for attempt in range(max_retry + 1):
            try:
                return client.generate(workflow, s.output_path(), timeout=900)
            except ComfyUIError as e:
                if attempt >= max_retry or not is_retryable_error(str(e)):
                    raise
                wait = 5 * (attempt + 1)
                self.sig.progress.emit(
                    f"网络类错误，{wait}s 后自动重试 {attempt + 1}/{max_retry}"
                    f": {str(e)[:120]}")
                time.sleep(wait)

    def _series_template(self) -> dict | None:
        """worker 用：传入模板含 __IMAGE__ 时用于系列生成（Qwen i2i 等）。"""
        try:
            if any(v == "__IMAGE__" for n in self.template.values()
                   for v in n.get("inputs", {}).values()):
                return self.template
        except Exception:
            pass
        return None

    def _resolve_anchor(self, client: ComfyUIClient, workflow: dict | None,
                        out_dir) -> tuple:
        """锚定帧解析：参考图优先；无参考图则 txt2img 生成首帧并去背景。

        返回 (anchor_bgra, first_path)；first_path 仅 txt2img 模式非 None。
        """
        import cv2
        s = self.s
        if s.ref_image:
            if not Path(s.ref_image).is_file():
                raise ComfyUIError(f"参考图不存在: {s.ref_image}")
            img = cv2.imread(s.ref_image, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ComfyUIError(f"参考图读取失败: {s.ref_image}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.sig.progress.emit("使用参考图作为外观锚定")
            return img, None
        first = client.generate(workflow, out_dir, timeout=900)[0]
        self.sig.progress.emit("首帧完成，去背景得到外观锚定帧 ...")
        img = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ComfyUIError(f"首帧读取失败: {first}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        from pixelforge.processing.image_processor import ImageProcessor
        anchor = ImageProcessor().remove_background(img)
        return anchor, first

    def _grid_mode(self) -> bool:
        """系列模式选择「网格模式」。"""
        return self.cmb_mode.currentIndex() == 1

    def _run_grid_series(self, client: ComfyUIClient, seed: int) -> None:
        """网格模式：单次生成 4x2 整表（帧间一致性最高）→ 拆帧归一化。"""
        import numpy as np
        from pixelforge.processing.image_processor import (
            normalize_grid_cells,
            split_grid_even,
            strip_ground_lines,
        )
        s = self.s
        t0 = time.monotonic()
        out_dir = s.output_path()
        anchor, _ = self._resolve_anchor(client, None, out_dir)
        comp = anchor[:, :, :3]
        if anchor.shape[2] == 4:
            af = anchor[:, :, 3:4].astype(np.float32) / 255.0
            comp = (anchor[:, :, :3] * af
                    + np.array([0, 255, 0]) * (1 - af)).astype(np.uint8)
        # pad 到 2:1 横版（匹配 4x2 网格比例）
        h, w = comp.shape[:2]
        W, H = 1024, 512
        sc = min(W / w, H / h)
        nw, nh = int(w * sc), int(h * sc)
        r = cv2.resize(comp, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W, 3), (0, 255, 0), np.uint8)
        canvas[(H - nh) // 2:(H - nh) // 2 + nh,
               (W - nw) // 2:(W - nw) // 2 + nw] = r
        ref_png = out_dir / f"grid_ref_{seed}.png"
        cv2.imwrite(str(ref_png), canvas)
        server = client.upload_image(canvas, f"pf_grid_{int(time.time())}")

        actions = self.actions if self.actions and self.actions[0] else [
            "standing straight, front view",
            "turned 45 degrees clockwise, three-quarter view",
            "turned 90 degrees, side view facing right",
            "turned 135 degrees, back three-quarter view",
            "turned 180 degrees, full back view",
            "turned 225 degrees, back three-quarter view",
            "turned 270 degrees, side view facing left",
            "turned 315 degrees, three-quarter view, almost front",
        ]
        n = min(8, len(actions)) if len(actions) > 1 else 8
        while len(actions) < n:
            actions.append(actions[-1])
        frame_desc = ", ".join(
            f"frame {i + 1} shows the character {a}"
            for i, a in enumerate(actions[:n]))
        prompt = (
            "Using <image1> as the exact character reference, generate a "
            "pixel art sprite sheet with exactly 4 columns and 2 rows "
            "(8 frames in a grid, equal spacing, each frame shows the FULL "
            "BODY): this same character, " + frame_desc + ". CRITICAL: the "
            "character must look IDENTICAL in every frame - same face, "
            "hairstyle, outfit, colors, proportions, same art style. Flat "
            "solid pure green background #00FF00, 16-bit retro pixel style")
        wf2 = build_workflow(self._series_template() or self.template,
                             checkpoint=s.checkpoint, prompt=prompt,
                             negative_prompt=s.negative_prompt, seed=seed,
                             width=0, height=0, steps=s.steps, cfg=s.cfg,
                             lora_name=s.lora_name,
                             lora_strength=s.lora_strength)
        for node in wf2.values():
            ins = node.get("inputs", {})
            for k, v in list(ins.items()):
                if v == "__IMAGE__":
                    ins[k] = server
        self.sig.progress.emit(
            f"网格模式：单次生成 {n} 帧整表（帧间一致性最高）")
        paths = client.generate(wf2, out_dir, timeout=1200)
        grid = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        if grid.shape[2] == 4:
            grid = cv2.cvtColor(grid, cv2.COLOR_BGRA2BGR)
        grid = grid[30:-30, 30:-30]
        bgra = proc.remove_background(grid)
        bgra = strip_ground_lines(bgra)
        cells = split_grid_even(bgra, 2, 4)
        frames = normalize_grid_cells(cells, align="free", pad=14)
        frame_paths = []
        for i, f in enumerate(frames):
            fp = out_dir / f"grid_frame_{i:03d}.png"
            cv2.imwrite(str(fp), f)
            frame_paths.append(str(fp))
        self.sig.progress.emit(
            f"网格模式完成：{len(frames)} 帧，耗时 {time.monotonic() - t0:.1f}s")
        self.sig.done.emit({"paths": frame_paths, "seed": seed})

    def _run_ref_series(self, client: ComfyUIClient, seed: int) -> None:
        """参考图模式：每帧 = 参考图 img2img，动作提示词逐帧变化。

        动作行数即帧数；无动作行时用「系列帧数」。模板自适应：
        选中模板含 __IMAGE__（如 Qwen i2i）则直接使用。
        """
        s = self.s
        t0 = time.monotonic()
        out_dir = s.output_path()
        anchor, _ = self._resolve_anchor(client, None, out_dir)
        actions = self.actions or [""]
        n = len(actions) if len(actions) > 1 else max(1, int(s.i2i_frames) or 1)
        template = self._series_template()
        self.sig.progress.emit(
            f"参考图模式：生成 {n} 帧（denoise={s.i2i_denoise}，"
            f"模板={'选中模板' if template else '默认 img2img'}）")
        all_paths = []
        for i in range(n):
            action = actions[i % len(actions)]
            prompt = f"{s.prompt}, {action}".strip(", ") if action else s.prompt
            paths = client.img2img(
                anchor, checkpoint=s.checkpoint, prompt=prompt,
                negative_prompt=s.negative_prompt, denoise=s.i2i_denoise,
                seed=None, steps=s.steps, cfg=s.cfg, out_dir=out_dir,
                lora_name=s.lora_name, lora_strength=s.lora_strength,
                timeout=900, template=template)
            all_paths.append(str(paths[0]))
            self.sig.progress.emit(f"帧 {i + 1}/{n} 完成"
                                   + (f"（{action}）" if action else ""))
        self.sig.progress.emit(
            f"参考图系列完成：{len(all_paths)} 帧，"
            f"耗时 {time.monotonic() - t0:.1f}s")
        self.sig.done.emit({"paths": all_paths, "seed": seed})

    def _run_series(self, client: ComfyUIClient, workflow: dict, seed: int) -> None:
        """跨帧一致性系列：首帧锚定 -> 逐帧 img2img 变体。"""
        s = self.s
        t0 = time.monotonic()
        out_dir = s.output_path()
        anchor, first = self._resolve_anchor(client, workflow, out_dir)
        n = int(s.i2i_frames)
        self.sig.progress.emit(
            f"img2img 系列生成 {n - 1} 帧（denoise={s.i2i_denoise}，锚定首帧）")
        frames = client.generate_frame_series(
            anchor, n, checkpoint=s.checkpoint, prompt=s.prompt,
            negative_prompt=s.negative_prompt, denoise=s.i2i_denoise,
            steps=s.steps, cfg=s.cfg, out_dir=out_dir,
            lora_name=s.lora_name, lora_strength=s.lora_strength,
            timeout=900, template=self._series_template(),
            progress=lambda i, total: self.sig.progress.emit(
                f"系列帧 {i}/{n - 1} 完成"))
        all_paths = ([str(first)] if first else []) + [str(p) for p in frames]
        self.sig.progress.emit(
            f"系列生成完成：{len(all_paths)} 帧，耗时 {time.monotonic() - t0:.1f}s")
        self.sig.done.emit({"paths": all_paths, "seed": seed})

    def _run_pose_series(self, client: ComfyUIClient, workflow: dict,
                         seed: int) -> None:
        """骨架驱动系列：首帧锚定 -> 每张骨架图一帧（CN 定姿势）。"""
        s = self.s
        t0 = time.monotonic()
        out_dir = s.output_path()
        n = len(self.pose_paths)
        self.sig.progress.emit(
            f"首帧锚定中 seed={seed}，随后按 {n} 张骨架图逐帧生成")
        anchor, first = self._resolve_anchor(client, workflow, out_dir)
        frames = client.generate_pose_series(
            anchor, self.pose_paths, checkpoint=s.checkpoint,
            controlnet_name=s.controlnet_name, prompt=s.prompt,
            negative_prompt=s.negative_prompt, pose_strength=s.pose_strength,
            denoise=s.i2i_denoise, steps=s.steps, cfg=s.cfg, out_dir=out_dir,
            lora_name=s.lora_name, lora_strength=s.lora_strength,
            timeout=900,
            progress=lambda i, total: self.sig.progress.emit(
                f"骨架帧 {i}/{total} 完成"))
        all_paths = ([str(first)] if first else []) + [str(p) for p in frames]
        self.sig.progress.emit(
            f"骨架驱动系列完成：{len(all_paths)} 帧，"
            f"耗时 {time.monotonic() - t0:.1f}s")
        self.sig.done.emit({"paths": all_paths, "seed": seed})

    def _ensure_engine(self, client: ComfyUIClient) -> None:
        s = self.s
        if not (s.auto_start_comfyui and s.conda_env and s.comfyui_path):
            raise ComfyUIError(
                "ComfyUI 未连接。请手动启动，或在左侧配置 Conda 环境与 ComfyUI 目录以启用自动启动。"
            )
        self.sig.progress.emit(f"ComfyUI 离线，正在通过 conda run -n {s.conda_env} 启动 ...")
        launch_comfyui(self.conda_exe, s.conda_env, s.comfyui_path, s.startup_extra_args)
        for i in range(120):  # 最长等 240s
            if client.is_online():
                self.sig.progress.emit("ComfyUI 启动成功")
                return
            time.sleep(2)
            if i % 5 == 4:
                self.sig.progress.emit(f"等待 ComfyUI 启动... {(i + 1) * 2}s")
        raise ComfyUIError("ComfyUI 启动超时（240s），请查看 logs/comfyui_launcher.log")


class _RerollWorker(QRunnable):
    """单帧 AI 重抽：以该帧为底图 img2img（保留 alpha），可指定提示词与强度。"""

    def __init__(self, settings: Settings, frame, prompt: str | None = None,
                 denoise: float | None = None, template: dict | None = None,
                 extra_images: dict | None = None, row: int | None = None,
                 orig=None):
        super().__init__()
        self.s = settings
        self.frame = frame
        self._prompt = prompt
        self._denoise = denoise
        self._template = template
        self._extra = extra_images
        self.row = row
        self.orig = orig
        self.sig = _Signals()

    def run(self) -> None:
        import cv2
        client = ComfyUIClient.from_settings(self.s)
        try:
            if not client.is_online():
                raise ComfyUIError("ComfyUI 未连接，无法重抽")
            paths = client.img2img(
                self.frame, checkpoint=self.s.checkpoint,
                prompt=self._prompt or self.s.prompt,
                negative_prompt=self.s.negative_prompt,
                denoise=self._denoise if self._denoise is not None else 0.5,
                steps=self.s.steps, cfg=self.s.cfg,
                out_dir=self.s.output_path(),
                lora_name=self.s.lora_name, lora_strength=self.s.lora_strength,
                timeout=600, template=self._template,
                extra_images=self._extra)
            img = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ComfyUIError("重抽结果读取失败")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            h, w = self.frame.shape[:2]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            self.sig.done.emit({"arr": img, "row": self.row, "orig": self.orig})
        except ComfyUIError as e:
            self.sig.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.sig.failed.emit(f"{type(e).__name__}: {e}")


class RerollDialog(QDialog):
    """单帧重抽参数：本帧提示词覆盖 + 重绘强度 + 前后帧上下文参考。"""

    def __init__(self, parent, prompt: str, denoise: float = 0.5,
                 score_info: str = ""):
        super().__init__(parent)
        self.setWindowTitle("AI 重抽本帧")
        lay = QVBoxLayout(self)
        if score_info:
            lb = QLabel(score_info)
            lb.setWordWrap(True)
            lay.addWidget(lb)
        self.chk_neighbors = QCheckBox("参考前后帧（过渡姿态优先，保动作连贯）")
        self.chk_neighbors.setChecked(True)
        self.chk_neighbors.setToolTip(
            "勾选：底图换为前后帧 50% 混合（Qwen 模板还会把前/后帧作为\n"
            "<image2>/<image3> 参考图），重抽结果落在前后姿态之间；\n"
            "不勾：仅以当前帧为底图，自由度更大")
        lay.addWidget(self.chk_neighbors)
        lay.addWidget(QLabel("本帧动作提示词（留空则用全局提示词）:"))
        self.ed_prompt = QPlainTextEdit()
        self.ed_prompt.setPlainText(prompt)
        self.ed_prompt.setFixedHeight(64)
        self.ed_prompt.setPlaceholderText(
            "如：Using <image1> refine this frame, <image2> is the previous "
            "frame, <image3> is the next frame")
        lay.addWidget(self.ed_prompt)
        row = QHBoxLayout()
        row.addWidget(QLabel("重绘强度"))
        self.spin_denoise = QDoubleSpinBox()
        self.spin_denoise.setRange(0.1, 0.9)
        self.spin_denoise.setSingleStep(0.05)
        self.spin_denoise.setValue(denoise)
        self.spin_denoise.setToolTip("低=只微调细节，高=大幅改变姿态")
        row.addWidget(self.spin_denoise)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QHBoxLayout()
        ok = QPushButton("开始重抽")
        ok.setProperty("accent", True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        bb.addWidget(ok)
        bb.addWidget(cancel)
        lay.addLayout(bb)


# ---------------------------------------------------------------- 主窗口

def _hbox(*widgets) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        h.addWidget(x)
    return w


def _is_dark_theme() -> bool:
    from PySide6.QtCore import QSettings
    return QSettings("PixelForge", "PixelForge").value(
        "ui_theme", "dark") != "light"


class RefThumb(QFrame):
    """单张参考图缩略图卡片：悬停浮窗大图预览，× 单张移除。"""

    def __init__(self, path: str, remove_cb, hover_show, hover_hide,
                 parent=None):
        super().__init__(parent)
        self.path = path
        self._hover_show = hover_show
        self._hover_hide = hover_hide
        self.setFixedSize(104, 118)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("refThumb")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(3)
        img = QLabel()
        img.setFixedSize(92, 76)
        img.setAlignment(Qt.AlignCenter)
        img.setObjectName("refThumbImg")
        pm = QPixmap(path)
        if not pm.isNull():
            img.setPixmap(pm.scaled(
                92, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(img)
        name = QLabel(Path(path).name)
        name.setFixedWidth(92)
        fm = name.fontMetrics()
        name.setText(fm.elidedText(name.text(), Qt.ElideMiddle, 90))
        name.setObjectName("refThumbName")
        lay.addWidget(name)
        btn = QPushButton("×", self)
        btn.setFixedSize(18, 18)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setObjectName("refThumbClose")
        btn.move(self.width() - 22, 2)
        btn.clicked.connect(lambda: remove_cb(self))
        name.setToolTip(path)

    def enterEvent(self, e) -> None:
        self._hover_show(self.path, QCursor.pos())
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self._hover_hide()
        super().leaveEvent(e)


class DropCard(QGroupBox):
    """支持拖入图片文件上传的分组卡片（拖到卡片任意位置即可）。"""

    files_dropped = Signal(list)
    IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls() and any(
                u.toLocalFile().lower().endswith(self.IMG_EXT)
                for u in e.mimeData().urls()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        files = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith(self.IMG_EXT)
                 and Path(u.toLocalFile()).is_file()]
        if files:
            e.acceptProposedAction()
            self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PixelForge Studio v{__version__}")
        self.resize(1360, 860)
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)        # 拖入图片/JSON/帧目录即可打开
        self.pool = QThreadPool.globalInstance()
        self.conda_exe = ""
        self.settings = Settings.load()
        self._busy = False
        self._task_no = 0
        self._gpt_active = 0          # 并发云端任务数（支持排队同时生成）
        self._gpt_workers: dict = {}  # task_no -> worker（停止用）
        self._task_snaps: dict = {}   # task_no -> 生成快照（登记提示词用）
        self._qid_of_tn: dict = {}    # task_no -> 持久化队列记录 id
        self._qo_syncing = False      # 画质/线路控件双向同步防循环
        self._connected_once = False  # 首次连接成功时自动拉取模型列表
        self._skip_gen_confirm = False  # 生成前摘要确认「本次会话不再提示」

        self._build_ui()
        self._wire_quality_oversea()
        self._load_settings_to_ui()
        self._refresh_workflows()
        self._refresh_profiles()
        self._refresh_gen_history()

        # 布局持久化：恢复上次关闭时的 Dock 位置/大小/浮动状态与窗口几何
        qs = QSettings("PixelForge", "PixelForge")
        geo = qs.value("window_geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = qs.value("window_state")
        if state is not None:
            self.restoreState(state)
        # 左侧配置面板给足默认宽度（避免内容挤压）；右侧日志 340
        self.resizeDocks([self.dock_config, self.dock_log], [460, 340],
                         Qt.Horizontal)

        self._conn_timer = QTimer(self)
        self._conn_timer.timeout.connect(self.check_connection)
        self._conn_timer.start(8000)
        QTimer.singleShot(200, self.refresh_envs)
        QTimer.singleShot(200, self.check_connection)

    # ---------- UI 构建 ----------

    def _build_ui(self) -> None:
        # 工作区布局：中央多标签 + 左右可停靠面板（可浮动/关闭，菜单可恢复）
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.tabs = tabs
        self._center_panel = self._build_center_panel()
        tabs.addTab(self._center_panel, "生成与审核")
        self.image_lab = ImageLab()
        self.image_lab.status = lambda m: self.log(f"[工具坊] {m}")
        self.image_lab.frame_back.connect(self._on_frame_back)
        tabs.addTab(self.image_lab, "图像工具坊")
        self.anim_player = AnimPlayer()
        self.anim_player.status = lambda m: self.log(f"[动画] {m}")
        tabs.addTab(self.anim_player, "动画播放")
        self.batch_panel = BatchPanel(settings_provider=self._collect_settings)
        self.batch_panel.status = lambda m: self.log(f"[批量] {m}")
        tabs.addTab(self.batch_panel, "批量动作")
        self.forge_panel = ForgePanel(settings_provider=self._collect_settings)
        self.forge_panel.state_changed.connect(
            lambda m: self.log(f"[Forge] {m}"))
        tabs.addTab(self.forge_panel, "批量工坊")
        self.assets_panel = AssetLibraryPanel()
        self.assets_panel.state_changed.connect(
            lambda m: self.log(f"[资产库] {m}"))
        self.assets_panel.frames_to_review.connect(self._on_assets_to_review)
        self.assets_panel.frames_to_player.connect(self._on_assets_to_player)
        self.assets_panel.image_to_lab.connect(self._on_asset_to_lab)
        self.assets_panel.prompt_loaded.connect(self._on_prompt_loaded)
        self.assets_panel.gen_material.connect(self._on_gen_material)
        self.assets_panel.gen_material_extra.connect(
            lambda paths: self._on_gen_material(paths, append=True))
        self.assets_panel.preview_requested.connect(self._on_asset_preview)
        tabs.addTab(self.assets_panel, "资产库")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.image_lab.frames_to_player.connect(self._on_frames_to_player)
        self.setCentralWidget(tabs)

        self.dock_config = QDockWidget("生成配置", self)
        self.dock_config.setWidget(self._build_left_panel())
        self.dock_config.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_config)

        self.dock_log = QDockWidget("队列与日志", self)
        self.dock_log.setWidget(self._build_right_panel())
        self.dock_log.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_log)

        menu_view = self.menuBar().addMenu("视图")
        menu_view.addAction(self.dock_config.toggleViewAction())
        menu_view.addAction(self.dock_log.toggleViewAction())

        self.lbl_conn = QLabel("● 检测中...")
        self.lbl_gpu = QLabel("")
        self.lbl_tab_tips = QLabel("")
        self.statusBar().addWidget(self.lbl_conn)
        self.statusBar().addPermanentWidget(self.lbl_gpu)  # 显卡信息固定右侧
        self.statusBar().addPermanentWidget(self.lbl_tab_tips)
        self._install_shortcuts()        # 全局快捷键（F1 一览表同源）
        self._on_tab_changed(self.tabs.currentIndex())   # 初始模块提示

    # ---- 快捷键（单一数据源：面板展示与实际绑定同源） ----

    # (分组, 按键, 说明)；keys 为空 = 仅供展示（由控件内部处理）
    SHORTCUT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
        ("生成", [
            ("Ctrl+G", "开始生成（当前模式与参数）"),
            ("Ctrl+Shift+S", "停止队列中选中的任务"),
            ("Ctrl+Shift+O", "打开产物输出目录"),
        ]),
        ("界面", [
            ("F1", "打开本快捷键一览"),
            ("Ctrl+,", "打开设置对话框"),
            ("Ctrl+1 / 2 / 3", "切换 笨蛋 / 专注 / 专业 模式"),
            ("Ctrl+R", "刷新资产库"),
        ]),
        ("生成与审核 · 看图器", [
            ("滚轮", "缩放画布（放大镜开启时改为调倍率 2x~20x）"),
            ("左键拖拽", "平移画面"),
            ("双击 / F", "适配窗口"),
            ("1", "1:1 原始比例"),
            ("M", "切换放大镜"),
            ("按住 Alt", "临时放大镜（松开还原）"),
        ]),
        ("动画播放", [
            ("Ctrl+P", "播放 / 暂停"),
        ]),
        ("动画播放 · 洋葱皮", [
            ("勾选「洋葱皮」", "叠加前(红)/后(蓝)若干帧半透明残影，右侧微调前后帧数"),
        ]),
        ("资产库", [
            ("双击", "在预览中打开该资产"),
            ("右键", "重载 / 作为生成素材 / 收藏 / 分类"),
            ("框选拖拽", "多选（收藏、上云、移动文件夹）"),
        ]),
        ("队列日志", [
            ("双击", "载入该任务设置用于复用修改"),
            ("右键", "载入 / 重跑 / 导入提示词 / 预览 / 删除"),
            ("悬停", "浮窗预览参考图 + 产物 + 提示词"),
        ]),
        ("拖拽打开（拖到窗口任意空白处）", [
            ("拖入 Aseprite JSON", "载入为多动画组并跳到动画播放"),
            ("拖入图片", "1 张 → 看图器预览；多张 → 作为动画帧播放"),
            ("拖入文件夹", "把其中的 PNG 序列载入动画播放"),
        ]),
    ]

    def _install_shortcuts(self) -> None:
        """注册全局快捷键。与 SHORTCUT_GROUPS 中标注的按键保持一致。"""
        def bind(seq: str, slot) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)   # 保持引用防回收

        self._shortcuts: list[QShortcut] = []
        bind("F1", self._show_shortcuts)
        bind("Ctrl+,", self._open_settings_dialog)
        bind("Ctrl+G", self.on_generate)
        bind("Ctrl+Shift+S", self._stop_selected_task)
        bind("Ctrl+Shift+O", self._open_output_dir)
        bind("Ctrl+R", lambda: self.assets_panel.refresh())
        bind("Ctrl+P", lambda: self.anim_player.toggle_play())
        for i in range(3):
            bind(f"Ctrl+{i + 1}",
                 lambda idx=i: self.cmb_ui_mode.setCurrentIndex(idx))

    def _show_shortcuts(self) -> None:
        """F1：快捷键一览（按分组渲染，深浅主题自适应）。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("快捷键一览")
        dlg.resize(560, 620)
        lay = QVBoxLayout(dlg)
        box = QLabel()
        box.setTextFormat(Qt.RichText)
        box.setWordWrap(True)
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rows = []
        for group, items in self.SHORTCUT_GROUPS:
            rows.append(
                f'<p style="margin:10px 0 4px 0;"><b>{group}</b></p>'
                '<table cellspacing="0" cellpadding="4" width="100%">')
            for keys, desc in items:
                rows.append(
                    f'<tr><td width="150"><b>{keys}</b></td>'
                    f'<td>{desc}</td></tr>')
            rows.append("</table>")
        box.setText("".join(rows))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        lay.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("关闭")
        b.setProperty("accent", True)
        b.clicked.connect(dlg.accept)
        row.addWidget(b)
        lay.addLayout(row)
        dlg.exec()

    # ---- 拖拽打开（把文件/文件夹拖到窗口任意空白处） ----

    DROP_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tga")

    def dragEnterEvent(self, e) -> None:
        md = e.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()
                 if u.isLocalFile()]
        paths = [p for p in paths if p.exists()]
        if not paths:
            e.ignore()
            return
        e.acceptProposedAction()
        self._handle_dropped(paths)

    def _handle_dropped(self, paths: list[Path]) -> None:
        """按类型分流：Aseprite JSON→播放器多动画；目录→帧目录；
        单图→看图器；多图→动画帧。"""
        from pixelforge.processing.aseprite_io import frames_by_tag
        jsons = [p for p in paths
                 if p.is_file() and p.suffix.lower() == ".json"]
        imgs = [p for p in paths if p.is_file()
                and p.suffix.lower() in self.DROP_IMG_EXTS]
        dirs = [p for p in paths if p.is_dir()]
        done: list[str] = []
        for j in jsons:
            try:
                clips = frames_by_tag(str(j))
            except Exception as ex:  # noqa: BLE001 非动画 JSON 直接忽略
                self.log(f"⚠ 拖入的 JSON 不是可解析的 Aseprite 动画（{j.name}）：{ex}")
                continue
            if not clips:
                continue
            self.anim_player._clips = {
                n: self.anim_player._normalize(list(a))
                for n, a in clips.items()}
            first = next(iter(self.anim_player._clips))
            self.anim_player.set_frames(
                self.anim_player._clips[first], name=first, keep_clips=True)
            self.tabs.setCurrentWidget(self.anim_player)
            done.append(f"{j.name}（{len(clips)} 个动画组）")
        for d in dirs:
            if self.anim_player.load_folder(d) == 0:
                self.log(f"⚠ 目录 {d.name} 中没有 PNG 帧，已忽略")
                continue
            self.tabs.setCurrentWidget(self.anim_player)
            done.append(f"{d.name}/（{len(self.anim_player._frames)} 帧）")
        if imgs:
            if len(imgs) == 1:
                self.viewer.set_image(str(imgs[0]))
                self.tabs.setCurrentIndex(0)
                done.append(imgs[0].name)
            else:
                files = sorted(imgs)
                self.anim_player.set_frames(list(files),
                                            name=files[0].stem or "dropped")
                self.tabs.setCurrentWidget(self.anim_player)
                done.append(f"{len(files)} 张图片 → 动画帧")
        if done:
            self.log("📥 已打开拖入内容：" + "；".join(done))
        else:
            self.log("⚠ 拖入内容中没有可打开的图片 / Aseprite JSON / 帧目录")

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        # ---- 低频控件（由设置对话框承载，此处只创建）----
        self._build_engine_widgets()

        # ---- 笨蛋模式卡片（一比一对齐在线生图交互）----
        g_basic = DropCard("① 上传角色图片")
        g_basic.files_dropped.connect(self._on_basic_refs_dropped)
        fb = QVBoxLayout(g_basic)
        self.btn_upload_basic = QPushButton("📁  点击上传 / 拖入图片（可多选，不上传 = 文生图）")
        self.btn_upload_basic.setMinimumHeight(56)
        self.btn_upload_basic.clicked.connect(self._upload_basic_refs)
        fb.addWidget(self.btn_upload_basic)
        self.w_basic_thumbs = QWidget()
        self.w_basic_thumbs.setObjectName("thumbStrip")
        self.h_thumbs = QHBoxLayout(self.w_basic_thumbs)
        self.h_thumbs.setContentsMargins(6, 6, 6, 6)
        self.h_thumbs.setSpacing(8)
        self.h_thumbs.addStretch(1)
        self.sa_thumbs = QScrollArea()
        self.sa_thumbs.setWidgetResizable(True)
        self.sa_thumbs.setWidget(self.w_basic_thumbs)
        self.sa_thumbs.setFixedHeight(136)
        self.sa_thumbs.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.sa_thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lbl_thumbs_empty = QLabel("（未上传 → 文生图模式：上传图片可锁定角色外观）")
        self.lbl_thumbs_empty.setObjectName("thumbEmpty")
        fb.addWidget(self.lbl_thumbs_empty)
        fb.addWidget(self.sa_thumbs)
        self._hover_preview: QLabel | None = None
        fmt_tip = QLabel("支持 JPG、PNG、WEBP（可多选）；留空则按提示词文生图")
        fmt_tip.setObjectName("dimTip")
        fb.addWidget(fmt_tip)
        lay.addWidget(g_basic)

        g_basic_res = QGroupBox("输出参数（当前生成设置一览）")
        fres = QHBoxLayout()   # 注意：g_basic_res 只能有 fres2 一个顶层布局
        self.btn_res_1k = QPushButton("1K（标准）")
        self.btn_res_2k = QPushButton("2K（高清）")
        self.btn_res_4k = QPushButton("4K（超清·更慢）")
        for b in (self.btn_res_1k, self.btn_res_2k, self.btn_res_4k):
            b.setCheckable(True)
            b.setMinimumHeight(40)
            fres.addWidget(b, 1)
        self.btn_res_1k.setChecked(True)
        self.btn_res_1k.clicked.connect(
            lambda: self._basic_res_pick("1k"))
        self.btn_res_2k.clicked.connect(
            lambda: self._basic_res_pick("2k"))
        self.btn_res_4k.clicked.connect(
            lambda: self._basic_res_pick("4k"))
        self.lbl_basic_info = QLabel("（参数信息）")
        self.lbl_basic_info.setObjectName("dimTip")
        self.lbl_basic_info.setWordWrap(True)
        # 画质 + 线路（与设置对话框的 cmb_gpt_quality/chk_gpt_oversea 双向同步）
        qo_row = QHBoxLayout()
        qo_row.addWidget(QLabel("画质"))
        self.cmb_basic_quality = QComboBox()
        self.cmb_basic_quality.addItems(
            ["low（快速）", "medium（标准）", "high（精细）"])
        self.cmb_basic_quality.setCurrentIndex(1)
        self.cmb_basic_quality.setToolTip(
            "思考深度：low 快而省、medium 标准、high 最精细最慢")
        qo_row.addWidget(self.cmb_basic_quality, 1)
        qo_row.addWidget(QLabel("线路"))
        self.cmb_basic_oversea = QComboBox()
        self.cmb_basic_oversea.addItems(["国内线路", "海外线路"])
        self.cmb_basic_oversea.setToolTip(
            "海外线路 = oversea=true；默认国内线路")
        qo_row.addWidget(self.cmb_basic_oversea, 1)
        fres2 = QVBoxLayout(g_basic_res)
        fres2.addLayout(fres)
        fres2.addLayout(qo_row)
        fres2.addWidget(self.lbl_basic_info)
        self._refresh_basic_info()
        lay.addWidget(g_basic_res)

        g_basic2 = QGroupBox("② 提示词（自动预填，可直接修改）")
        fb2 = QVBoxLayout(g_basic2)
        lbl_pos = QLabel("正面提示词")
        lbl_pos.setObjectName("sectionLabel")
        fb2.addWidget(lbl_pos)
        self.ed_prompt_basic = QPlainTextEdit()
        self.ed_prompt_basic.setPlaceholderText(
            "描述想要的动画，如：8 帧原地跳跃动画（蹲下→腾空→落地）")
        self.ed_prompt_basic.setFixedHeight(260)
        self.ed_prompt_basic.setPlainText(
            "transform this character into a pixel art sprite sheet, "
            "4 columns 2 rows, 8 frames of the same character performing "
            "ONE complete jump in place (crouch, leap, apex, fall, land), "
            "character IDENTICAL in every frame, flat solid pure green "
            "background #00FF00, 16-bit retro pixel style")
        fb2.addWidget(self.ed_prompt_basic)
        lbl_neg = QLabel("负面提示词（拼入提示词，可留空）")
        lbl_neg.setObjectName("sectionLabel")
        fb2.addWidget(lbl_neg)
        self.ed_negative_basic = QPlainTextEdit()
        self.ed_negative_basic.setPlaceholderText(
            "负面提示词（可选，可留空）：不希望出现的元素，如 blurry, text, "
            "extra limbs, background noise\n（GPT-Image 无独立负面参数，"
            "会以 Avoid: … 形式智能拼入提示词）")
        self.ed_negative_basic.setFixedHeight(92)
        fb2.addWidget(self.ed_negative_basic)
        tpl_row_b = QHBoxLayout()
        b_tpl_b = QPushButton("📚 模板库 / 保存提示词")
        b_tpl_b.clicked.connect(self._open_template_library)
        tpl_row_b.addWidget(b_tpl_b, 1)
        fb2.addLayout(tpl_row_b)
        lay.addWidget(g_basic2)

        self.btn_generate_basic = QPushButton("🎨  生成")
        self.btn_generate_basic.setMinimumHeight(52)
        self.btn_generate_basic.setProperty("accent", True)
        self.btn_generate_basic.clicked.connect(self.on_generate)
        lay.addWidget(self.btn_generate_basic)
        self._basic_ref_paths: list[str] = []
        self._basic_cards = [g_basic, g_basic_res, g_basic2,
                             self.btn_generate_basic]

        # ---- 核心区（专注/专业模式的生成设置）----
        g_core = QGroupBox("生成设置")
        f2 = QFormLayout(g_core)
        self._form_g2 = f2
        prof_row = QHBoxLayout()
        self.cmb_profile = QComboBox()
        self.cmb_profile.setToolTip("角色档案：checkpoint/LoRA/提示词等组合一键切换")
        self.cmb_profile.currentTextChanged.connect(self._on_profile_selected)
        prof_row.addWidget(self.cmb_profile, 1)
        b_psave = QPushButton("保存...")
        b_psave.clicked.connect(self._save_profile_dialog)
        prof_row.addWidget(b_psave)
        b_pdel = QPushButton("删除")
        b_pdel.clicked.connect(self._delete_profile)
        prof_row.addWidget(b_pdel)
        b_pexp = QPushButton("导出")
        b_pexp.setToolTip("把该档案的全部生成设置导出为 .json 文件，可发给其他人导入使用")
        b_pexp.clicked.connect(self._export_profile)
        prof_row.addWidget(b_pexp)
        b_pimp = QPushButton("导入")
        b_pimp.setToolTip("导入他人分享的档案 .json 文件")
        b_pimp.clicked.connect(self._import_profile)
        prof_row.addWidget(b_pimp)
        f2.addRow("角色档案", prof_row)

        hist_row = QHBoxLayout()
        self.cmb_gen_history = QComboBox()
        self.cmb_gen_history.setToolTip(
            "生成历史快照：每次生成自动记录当时的全部设置（含提示词/参数/引擎），\n"
            "选择即恢复到该次生成前的完整配置；悬停条目可预览参考图/生成图/提示词")
        hist_row.addWidget(self.cmb_gen_history, 1)
        b_hist = QPushButton("加载")
        b_hist.clicked.connect(self._load_gen_snapshot)
        hist_row.addWidget(b_hist)
        f2.addRow("历史快照", hist_row)
        # 悬停浮窗：预览该快照的参考图/生成图/提示词
        _hv = self.cmb_gen_history.view()
        _hv.setMouseTracking(True)
        _hv.entered.connect(lambda mi: self._on_history_hover(mi.row()))
        _hv.viewport()._pf_hide_hover = True
        _hv.viewport().installEventFilter(self)

        self.ed_prompt = QPlainTextEdit()
        self.ed_prompt.setPlaceholderText(
            "角色描述 + 动作，如：pixel art knight, idle animation")
        self.ed_prompt.setFixedHeight(220)
        f2.addRow("提示词", self.ed_prompt)
        self.ed_negative = QPlainTextEdit()
        self.ed_negative.setFixedHeight(110)
        f2.addRow("负面提示词", self.ed_negative)

        tpl_row = QHBoxLayout()
        b_tpl = QPushButton("📚 模板库（套用/保存/分享提示词）")
        b_tpl.clicked.connect(self._open_template_library)
        tpl_row.addWidget(b_tpl, 1)
        f2.addRow("模板库", tpl_row)

        self.spin_w = QSpinBox(); self.spin_w.setRange(64, 2048); self.spin_w.setSingleStep(64)
        self.spin_h = QSpinBox(); self.spin_h.setRange(64, 2048); self.spin_h.setSingleStep(64)
        f2.addRow("宽 / 高", _hbox(self.spin_w, self.spin_h))
        self.spin_steps = QSpinBox(); self.spin_steps.setRange(1, 100)
        self.spin_cfg = QDoubleSpinBox(); self.spin_cfg.setRange(1.0, 20.0); self.spin_cfg.setSingleStep(0.5)
        f2.addRow("步数 / CFG", _hbox(self.spin_steps, self.spin_cfg))
        self.chk_lock_seed = QCheckBox("锁定种子（多帧一致性核心）")
        f2.addRow(self.chk_lock_seed)
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 2**31 - 1)
        f2.addRow("种子", self.spin_seed)

        ref_row = QHBoxLayout()
        self.ed_ref = QLineEdit()
        self.ed_ref.setPlaceholderText("角色参考图路径（可选，作为外观锚定）")
        ref_row.addWidget(self.ed_ref, 1)
        b_ref = QPushButton("选择...")
        b_ref.clicked.connect(self._choose_ref)
        ref_row.addWidget(b_ref)
        b_refclr = QPushButton("清除")
        b_refclr.clicked.connect(self.ed_ref.clear)
        ref_row.addWidget(b_refclr)
        f2.addRow("参考图", ref_row)
        self.ed_actions = QPlainTextEdit()
        self.ed_actions.setPlaceholderText(
            "每行一个动作（参考图模式：行数=帧数；网格模式：每行=一个动作）\n"
            "如：arms raised\n    walking, left leg forward")
        self.ed_actions.setFixedHeight(88)
        f2.addRow("动作序列", self.ed_actions)

        self.chk_auto_post = QCheckBox("生成后自动后处理（去背景/裁切/对齐/调色板）")
        f2.addRow(self.chk_auto_post)
        lay.addWidget(g_core)

        # ---- 生成按钮（设置入口已移至顶部工具栏）----

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        self.pbar.setFormat("生成中 %p%")
        lay.addWidget(self.pbar)

        self.btn_generate = QPushButton("生成")
        self.btn_generate.setMinimumHeight(42)
        self.btn_generate.setProperty("accent", True)
        self.btn_generate.clicked.connect(self.on_generate)
        lay.addWidget(self.btn_generate)
        lay.addStretch(1)

        # ---- 模式显隐矩阵 ----
        self._g2_focus = [self.cmb_profile, b_psave, b_pdel,
                          b_pexp, b_pimp,
                          self.cmb_gen_history, b_hist,
                          self.ed_prompt, self.ed_negative,
                          self.spin_w, self.spin_h,
                          self.spin_steps, self.spin_cfg,
                          self.chk_lock_seed, self.spin_seed,
                          self.ed_ref, b_ref, b_refclr,
                          self.ed_actions, self.chk_auto_post]
        self._g2_basic: list = []   # 笨蛋模式用独立卡片（_basic_cards）
        # ---- 专注/专业的核心云端行：思考深度放主界面（模型/尺寸/线路/背景在设置对话框）----
        self._form_g2.addRow("画质（思考深度）", self.cmb_gpt_quality)
        self._g2_focus.extend([self.cmb_gpt_quality])
        # ---- 专业模式专属行（专注隐藏）----
        self._lbl_pro_sep = QLabel("— 专业选项 —")
        self._form_g2.addRow(self._lbl_pro_sep)
        lora_row = QHBoxLayout()
        lora_row.addWidget(self.ed_lora, 1)
        lora_row.addWidget(self.spin_lora)
        self._form_g2.addRow("LoRA（名称/强度）", lora_row)
        self._form_g2.addRow(self.chk_i2i)
        self._form_g2.addRow("系列模式", self.cmb_mode)
        self._form_g2.addRow("系列帧数", self.spin_i2i_n)
        self._form_g2.addRow("重绘强度", self.slide_i2i_denoise)
        self._form_g2.addRow(self.chk_pose)
        self._form_g2.addRow("ControlNet", self.cmb_cn)
        self._form_g2.addRow("骨架图序列", self.lst_pose)
        self._form_g2.addRow("姿势强度", self.spin_pose_str)
        self._g2_pro = [
            self._lbl_pro_sep,
            self.ed_lora, self.spin_lora, self.chk_i2i, self.cmb_mode,
            self.spin_i2i_n, self.slide_i2i_denoise, self.chk_pose,
            self.cmb_cn, self.lst_pose, self.spin_pose_str]

        # ---- 菜单栏顶栏：引擎切换居中 + 右侧模式切换/停止/设置 ----
        corner = QWidget()
        corner.setObjectName("menuCorner")
        corner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(8, 0, 6, 0)
        cl.setSpacing(8)
        # 居中组：引擎模式切换（API / 本地 ComfyUI）
        cl.addStretch(1)
        self.btn_engine_api = QPushButton("☁ API")
        self.btn_engine_api.setCheckable(True)
        self.btn_engine_api.setToolTip("云端 GPT-Image 引擎（输入 API Key 即用）")
        self.btn_engine_local = QPushButton("🖥 本地 Comfy")
        self.btn_engine_local.setCheckable(True)
        self.btn_engine_local.setToolTip("本地 ComfyUI 工作流引擎")
        for b in (self.btn_engine_api, self.btn_engine_local):
            b.setFixedHeight(26)
        self.btn_engine_api.clicked.connect(
            lambda: self._set_engine_mode(True))
        self.btn_engine_local.clicked.connect(
            lambda: self._set_engine_mode(False))
        cl.addWidget(self.btn_engine_api)
        cl.addWidget(self.btn_engine_local)
        cl.addSpacing(20)
        # 停止按钮：终止队列中选中的任务
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setFixedHeight(26)
        self.btn_stop.setToolTip("终止队列中当前选中的任务")
        self.btn_stop.clicked.connect(self._stop_selected_task)
        self.btn_stop.setEnabled(False)
        cl.addWidget(self.btn_stop)
        # 右侧组：界面模式 + 设置
        self.cmb_ui_mode = QComboBox()
        self.cmb_ui_mode.addItems(["笨蛋模式", "专注模式", "专业模式"])
        self.cmb_ui_mode.setToolTip(
            "笨蛋=选图选动作就生成 / 专注=核心参数 / 专业=全部设置")
        cl.addWidget(self.cmb_ui_mode)
        self.btn_open_out = QPushButton("📂 输出目录")
        self.btn_open_out.setToolTip("在文件管理器中打开产物输出目录（所有生成/导出默认落盘位置）")
        self.btn_open_out.setFixedHeight(26)
        self.btn_open_out.clicked.connect(self._open_output_dir)
        cl.addWidget(self.btn_open_out)
        self.btn_shortcuts = QPushButton("⌨ 快捷键")
        self.btn_shortcuts.setToolTip("快捷键一览（F1）")
        self.btn_shortcuts.setFixedHeight(26)
        self.btn_shortcuts.clicked.connect(self._show_shortcuts)
        cl.addWidget(self.btn_shortcuts)
        self.btn_settings_top = QPushButton("⚙ 设置")
        self.btn_settings_top.setToolTip("引擎 / API 密钥 / 高级生成 / 通用")
        self.btn_settings_top.setFixedHeight(26)
        self.btn_settings_top.clicked.connect(self._open_settings_dialog)
        cl.addWidget(self.btn_settings_top)
        cl.addStretch(1)
        self._menu_corner = corner   # 保持 Python 引用，防 C++ 对象被回收
        self._sync_engine_buttons()
        self.chk_gpt.toggled.connect(lambda _: self._sync_engine_buttons())
        self.cmb_ui_mode.currentIndexChanged.connect(self._apply_ui_mode)
        self.menuBar().setCornerWidget(corner, Qt.TopRightCorner)
        self._apply_ui_mode(self.settings.ui_mode)

        # 滚动容器：内容超长时可滚动（修复无法滚动问题）
        w.setMinimumWidth(430)   # 保证行内按钮不被压缩；窄窗口出横向滚动条
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    def _build_engine_widgets(self) -> None:
        """创建低频设置控件（不加布局；由设置对话框按标签承载）。"""
        # 本地引擎
        self.cmb_env = QComboBox()
        self.ed_comfy_path = QLineEdit()
        self.ed_comfy_path.setPlaceholderText(r"如 D:\ComfyUI（含 main.py）")
        self.chk_autostart = QCheckBox("离线时自动启动 ComfyUI（conda run）")
        self.chk_cloud = QCheckBox("启用远程 ComfyUI 服务器（协议同本地）")
        self.ed_cloud_base = QLineEdit()
        self.ed_cloud_base.setPlaceholderText(
            "如 https://api.comfyicu.com 或 liblibAI 端点")
        self.ed_cloud_key = QLineEdit()
        self.ed_cloud_key.setEchoMode(QLineEdit.Password)
        self.ed_cloud_key.setPlaceholderText("Bearer Token（本地模式留空）")
        self.chk_gpt = QCheckBox(
            "启用云端 GPT-Image（勾选后「生成」走云端引擎，无需本地 ComfyUI）")
        self.ed_gpt_base = QLineEdit()
        self.ed_gpt_base.setPlaceholderText(
            "中转站地址，如 https://duomiapi.com（OpenAI 兼容格式）")
        self.ed_gpt_key = QLineEdit()
        self.ed_gpt_key.setEchoMode(QLineEdit.Password)
        self.ed_gpt_key.setPlaceholderText("中转站控制台的 API Key")
        self.cmb_gpt_model = QComboBox()
        self.cmb_gpt_model.setEditable(True)
        self.cmb_gpt_model.addItems([
            "gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
            "nano-banana", "nano-banana-pro", "nano-banana-2",
            "gpt-image-1", "seedream-3.0", "flux-pro-1.1"])
        self.cmb_gpt_model.setToolTip(
            "常见模型下拉选择，也可手动输入中转站支持的任意模型名\n"
            "gpt-image-* 走 /v1/images/generations；\n"
            "nano-banana*（gemini 系列）走 /api/gemini/nano-banana，"
            "参考图自动上传图床")
        self.cmb_gpt_size = QComboBox()
        self.cmb_gpt_size.addItems(["1k", "2k", "4k"])
        self.cmb_gpt_quality = QComboBox()
        self.cmb_gpt_quality.addItems(
            ["low（快速）", "medium（标准）", "high（精细）"])
        self.cmb_gpt_quality.setCurrentIndex(1)
        self.cmb_gpt_quality.setToolTip(
            "思考深度：low 快而省、medium 标准、high 最精细最慢")
        self.chk_gpt_oversea = QCheckBox("海外线路（oversea）")
        self.chk_gpt_oversea.setToolTip(
            "勾选 = 走 oversea=true 海外线路；默认国内线路")
        self.cmb_gpt_bg = QComboBox()
        self.cmb_gpt_bg.addItems(
            ["背景 auto（跟随提示词）", "opaque（不透明）",
             "transparent（透明）"])
        self.cmb_gpt_bg.setToolTip(
            "transparent = 透明背景（gpt-image 系列）")
        self.chk_gpt.toggled.connect(
            lambda _: self._update_generate_button_text())
        self.ed_gpt_key.textChanged.connect(
            lambda _: self._update_generate_button_text())
        # 高级生成
        self.ed_lora = QLineEdit()
        self.ed_lora.setPlaceholderText("留空不使用，如：myCharacter.safetensors")
        self.spin_lora = QDoubleSpinBox()
        self.spin_lora.setRange(0.0, 1.5)
        self.spin_lora.setSingleStep(0.05)
        self.spin_lora.setValue(1.0)
        self.chk_i2i = QCheckBox("跨帧一致性系列生成（img2img 驱动）")
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["逐帧 img2img（动作提示词驱动）",
                                "网格模式（单次生成整表，一致性最高）"])
        self.spin_i2i_n = QSpinBox()
        self.spin_i2i_n.setRange(2, 12)
        self.spin_i2i_n.setValue(4)
        self.slide_i2i_denoise = QSlider(Qt.Horizontal)
        self.slide_i2i_denoise.setRange(20, 80)
        self.slide_i2i_denoise.setValue(45)
        # 骨架驱动
        self.chk_pose = QCheckBox("启用骨架驱动（优先于 img2img 系列）")
        self.cmb_cn = QComboBox()
        self.lst_pose = QListWidget()
        self.lst_pose.setMaximumHeight(88)
        self.spin_pose_str = QDoubleSpinBox()
        self.spin_pose_str.setRange(0.4, 1.3)
        self.spin_pose_str.setSingleStep(0.05)
        self.spin_pose_str.setValue(1.0)
        self.cmb_workflow = QComboBox()
        self.cmb_ckpt = QComboBox()
        self.cmb_ckpt.setEditable(True)

    # ---------- 提示词模板库 ----------

    def _open_template_library(self) -> None:
        """模板库对话框：列表/预览/套用/保存当前/删除/导出/导入。"""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QHBoxLayout,
            QListWidget,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
        if getattr(self, "_tpl_dlg", None) is not None:
            self._tpl_dlg.show()
            self._tpl_dlg.raise_()
            self._refresh_tpl_list()
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("提示词模板库")
        dlg.setModal(False)
        dlg.setMinimumSize(640, 480)
        v = QVBoxLayout(dlg)

        split = QHBoxLayout()
        self.lst_tpl = QListWidget()
        self.lst_tpl.currentTextChanged.connect(self._on_tpl_selected)
        split.addWidget(self.lst_tpl, 1)

        w_prev = QWidget()
        v_prev = QVBoxLayout(w_prev)
        v_prev.setContentsMargins(4, 4, 4, 4)
        self.ed_tpl_prompt = QPlainTextEdit()
        self.ed_tpl_prompt.setReadOnly(True)
        self.ed_tpl_prompt.setFixedHeight(200)
        self.ed_tpl_negative = QPlainTextEdit()
        self.ed_tpl_negative.setReadOnly(True)
        self.ed_tpl_negative.setFixedHeight(100)
        v_prev.addWidget(QLabel("正面提示词"))
        v_prev.addWidget(self.ed_tpl_prompt)
        v_prev.addWidget(QLabel("负面提示词"))
        v_prev.addWidget(self.ed_tpl_negative)
        split.addWidget(w_prev, 1)
        v.addLayout(split, 1)

        btns = QHBoxLayout()
        b_apply = QPushButton("✔ 套用到主界面")
        b_apply.clicked.connect(self._apply_template)
        btns.addWidget(b_apply)
        b_save = QPushButton("💾 保存当前提示词为新模板")
        b_save.clicked.connect(self._save_tpl_from_ui)
        btns.addWidget(b_save)
        b_del = QPushButton("🗑 删除选中")
        b_del.clicked.connect(self._delete_template)
        btns.addWidget(b_del)
        v.addLayout(btns)

        share = QHBoxLayout()
        b_exp = QPushButton("📤 导出选中模板（分享）")
        b_exp.clicked.connect(self._export_template)
        share.addWidget(b_exp)
        b_imp = QPushButton("📥 导入模板")
        b_imp.clicked.connect(self._import_template)
        share.addWidget(b_imp)
        share.addStretch(1)
        v.addLayout(share)

        bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        bb.rejected.connect(dlg.hide)
        v.addWidget(bb)
        self._tpl_dlg = dlg
        self._refresh_tpl_list()
        dlg.show()

    def _refresh_tpl_list(self, select: str = "") -> None:
        if getattr(self, "lst_tpl", None) is None:
            return   # 模板库对话框尚未创建（无列表可刷新）
        self.lst_tpl.blockSignals(True)
        self.lst_tpl.clear()
        self.lst_tpl.addItems(list(load_prompt_templates().keys()))
        if select:
            for i in range(self.lst_tpl.count()):
                if self.lst_tpl.item(i).text() == select:
                    self.lst_tpl.setCurrentRow(i)
                    break
        self.lst_tpl.blockSignals(False)
        if self.lst_tpl.currentRow() < 0 and self.lst_tpl.count():
            self.lst_tpl.setCurrentRow(0)

    def _on_tpl_selected(self, name: str) -> None:
        tpl = load_prompt_templates().get(name)
        if tpl:
            self.ed_tpl_prompt.setPlainText(tpl.get("prompt", ""))
            self.ed_tpl_negative.setPlainText(tpl.get("negative_prompt", ""))

    def _active_prompt_editors(self) -> tuple:
        """当前模式对应的（正面编辑框, 负面编辑框）。"""
        if self.cmb_ui_mode.currentIndex() == 0:
            return self.ed_prompt_basic, self.ed_negative_basic
        return self.ed_prompt, self.ed_negative

    def _apply_template(self) -> None:
        name = self.lst_tpl.currentItem().text() if \
            self.lst_tpl.currentItem() else ""
        tpl = load_prompt_templates().get(name)
        if not tpl:
            return
        ed_p, ed_n = self._active_prompt_editors()
        ed_p.setPlainText(tpl.get("prompt", ""))
        ed_n.setPlainText(tpl.get("negative_prompt", ""))
        self.log(f"已套用提示词模板: {name} → "
                 f"{'笨蛋模式' if self.cmb_ui_mode.currentIndex() == 0 else '主界面'}")

    def _save_tpl_from_ui(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        ed_p, ed_n = self._active_prompt_editors()
        prompt = ed_p.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "PixelForge",
                                    "提示词为空，请先在当前模式填写提示词")
            return
        name, ok = QInputDialog.getText(self, "保存模板",
                                        "模板名称（如：像素骑士-待机）：")
        if not ok or not name.strip():
            return
        templates = load_prompt_templates()
        templates[name.strip()] = {
            "prompt": prompt,
            "negative_prompt": ed_n.toPlainText().strip(),
        }
        save_prompt_templates(templates)
        self._refresh_tpl_list(select=name.strip())
        self.log(f"提示词模板已保存: {name.strip()}")

    def _delete_template(self) -> None:
        name = self.lst_tpl.currentItem().text() if \
            self.lst_tpl.currentItem() else ""
        templates = load_prompt_templates()
        if name not in templates:
            return
        del templates[name]
        save_prompt_templates(templates)
        self._refresh_tpl_list()
        self.log(f"提示词模板已删除: {name}")

    def _export_template(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        name = self.lst_tpl.currentItem().text() if \
            self.lst_tpl.currentItem() else ""
        templates = load_prompt_templates()
        if name not in templates:
            QMessageBox.information(self, "PixelForge", "请先选中要导出的模板")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出模板（可发给其他人）", f"{name}.pftemplate",
            "PixelForge 模板 (*.pftemplate *.json)")
        if not path:
            return
        payload = {"type": "pixelforge_template", "version": __version__,
                   "name": name, "template": templates[name]}
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        self.log(f"提示词模板已导出: {path}")

    def _import_template(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "导入模板", "", "PixelForge 模板 (*.pftemplate *.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "PixelForge", f"导入失败: {e}")
            return
        tpl = payload.get("template") if payload.get("type") == \
            "pixelforge_template" else payload
        if not isinstance(tpl, dict) or "prompt" not in tpl:
            QMessageBox.critical(self, "PixelForge",
                                 "文件格式不正确（缺少 template 字段）")
            return
        name = (payload.get("name") or Path(path).stem).strip()
        templates = load_prompt_templates()
        final = name
        i = 2
        while final in templates:
            final = f"{name}_{i}"
            i += 1
        templates[final] = tpl
        save_prompt_templates(templates)
        self._refresh_tpl_list(select=final)
        self.log(f"提示词模板已导入: {final}")

    def _open_output_dir(self) -> None:
        """在系统文件管理器中打开产物输出目录。"""
        try:
            out = Path(self.settings.output_path())
        except Exception:  # noqa: BLE001
            out = Path(PROJECT_ROOT) / "outputs"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.log(f"❌ 无法创建输出目录 {out}: {e}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out))
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(out)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(out)])
        except OSError as e:
            self.log(f"❌ 打开输出目录失败: {e}")
            return
        self.log(f"📂 已打开输出目录: {out}")

    def _open_settings_dialog(self) -> None:
        """设置对话框（三标签）。缓存实例：关闭=隐藏，控件与填写内容保留。"""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QTabWidget,
        )
        if getattr(self, "_settings_dlg", None) is not None:
            self._settings_dlg.show()
            self._settings_dlg.raise_()
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("PixelForge 设置")
        dlg.setModal(False)
        dlg.setMinimumSize(520, 560)
        v = QVBoxLayout(dlg)
        tabs = QTabWidget(dlg)

        # 标签 1：引擎与 API
        w1 = QWidget()
        f1 = QFormLayout(w1)
        f1.addRow(QLabel("— 本地 ComfyUI —"))
        f1.addRow("Conda 环境", self.cmb_env)
        f1.addRow("ComfyUI 目录", self.ed_comfy_path)
        f1.addRow(self.chk_autostart)
        f1.addRow(QLabel("— 远程 ComfyUI 服务器（协议同本地）—"))
        f1.addRow(self.chk_cloud)
        f1.addRow("API 地址", self.ed_cloud_base)
        f1.addRow("API Key", self.ed_cloud_key)
        f1.addRow(QLabel("— 云端 GPT-Image —"))
        f1.addRow(self.chk_gpt)
        f1.addRow("中转站地址", self.ed_gpt_base)
        f1.addRow("API Key", self.ed_gpt_key)
        f1.addRow("模型", self.cmb_gpt_model)
        f1.addRow("分辨率档", self.cmb_gpt_size)
        f1.addRow(self.chk_gpt_oversea)
        f1.addRow("背景", self.cmb_gpt_bg)
        f1.addRow(QLabel("思考深度（画质）在主界面「生成设置」/笨蛋模式中选择。"))
        tabs.addTab(w1, "引擎与 API")

        # 标签 2：高级生成（LoRA/系列/骨架已移至主界面专业模式）
        w2 = QWidget()
        f2b = QFormLayout(w2)
        f2b.addRow(QLabel("— 模型 —"))
        f2b.addRow("工作流模板", self.cmb_workflow)
        f2b.addRow("模型 Checkpoint", self.cmb_ckpt)
        tabs.addTab(w2, "高级生成")

        # 标签 3：通用
        w3 = QWidget()
        f3 = QFormLayout(w3)
        f3.addRow(self.chk_auto_post)
        self.cmb_theme = QComboBox()
        self.cmb_theme.addItem("🌙 深色（工业风）", "dark")
        self.cmb_theme.addItem("☀ 浅色（明亮）", "light")
        from PySide6.QtCore import QSettings as _QS
        cur = _QS("PixelForge", "PixelForge").value("ui_theme", "dark")
        self.cmb_theme.setCurrentIndex(0 if cur == "dark" else 1)
        self.cmb_theme.currentIndexChanged.connect(self._on_theme_changed)
        f3.addRow("界面主题", self.cmb_theme)
        f3.addRow(QLabel("主题切换立即生效并保存，重启后保持。"))
        # 云资产目录：资产库「上云」目标（网盘同步文件夹）
        from PySide6.QtWidgets import QHBoxLayout as _QHL
        self.ed_cloud_dir = QLineEdit()
        self.ed_cloud_dir.setPlaceholderText(
            "选择网盘同步文件夹（如 OneDrive/坚果云 目录）")
        self.ed_cloud_dir.setText(_QS("PixelForge", "PixelForge").value(
            "cloud_asset_dir", ""))
        btn_cloud = QPushButton("浏览…")
        btn_cloud.clicked.connect(self._browse_cloud_dir)
        row = QWidget()
        hl = _QHL(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.ed_cloud_dir, 1)
        hl.addWidget(btn_cloud)
        f3.addRow("云资产目录", row)
        f3.addRow(QLabel("资产库右键「上云」会把资产复制到该目录。"))
        f3.addRow(QLabel("输出目录与更多设置将在后续版本加入。"))
        tabs.addTab(w3, "通用")

        v.addWidget(tabs, 1)
        bb = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Ok
            | QDialogButtonBox.Close, dlg)
        bb.button(QDialogButtonBox.Apply).setText("应用")
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Close).setText("关闭")
        # 应用：保存但不关闭（可继续调其他标签）
        bb.button(QDialogButtonBox.Apply).clicked.connect(
            self._apply_settings_dialog)
        # 确定：保存并关闭
        bb.accepted.connect(self._close_settings_dialog)
        bb.rejected.connect(dlg.hide)   # 关闭：不保存直接关
        v.addWidget(bb)
        self._settings_dlg = dlg
        dlg.show()

    def _browse_cloud_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择云资产目录（网盘同步文件夹）")
        if d:
            self.ed_cloud_dir.setText(d)

    def _on_theme_changed(self, _idx: int) -> None:
        """主题切换：立即应用并持久化。"""
        from pixelforge.ui.theme import apply_theme
        dark = self.cmb_theme.currentData() != "light"
        apply_theme(QApplication.instance(), dark=dark)
        self.log(f"界面主题已切换: {'深色' if dark else '浅色'}")

    def _apply_settings_dialog(self) -> None:
        """应用：保存设置但保持窗口打开（可继续调其他标签）。"""
        s = self._collect_settings()
        s.save()
        # 云资产目录持久化
        from PySide6.QtCore import QSettings as _QSet
        _QSet("PixelForge", "PixelForge").setValue(
            "cloud_asset_dir", self.ed_cloud_dir.text().strip())
        self._update_generate_button_text()
        self._refresh_basic_info()
        self.log("设置已应用（窗口保持打开，可继续调整；"
                 "「确定」保存并关闭）")

    def _close_settings_dialog(self) -> None:
        """确定：保存设置并关闭窗口。"""
        self._apply_settings_dialog()
        if self._settings_dlg is not None:
            self._settings_dlg.hide()
        self.log("设置已保存并关闭")

    def _update_generate_button_text(self) -> None:
        """生成按钮统一文案「生成」（笨蛋/专注/专业通用）。"""
        if not hasattr(self, "btn_generate"):
            return
        self.btn_generate.setText("生成")
        self.btn_generate_basic.setText("🎨  生成")

    def _wire_quality_oversea(self) -> None:
        """笨蛋模式与设置对话框的 画质/线路 控件双向同步。"""
        self.cmb_basic_quality.currentIndexChanged.connect(
            self._on_basic_qo_changed)
        self.cmb_basic_oversea.currentIndexChanged.connect(
            self._on_basic_qo_changed)
        self.cmb_gpt_quality.currentIndexChanged.connect(
            self._on_dialog_qo_changed)
        self.chk_gpt_oversea.toggled.connect(self._on_dialog_qo_changed)

    def _on_basic_qo_changed(self, *_):
        if self._qo_syncing:
            return
        self._qo_syncing = True
        self.cmb_gpt_quality.setCurrentIndex(
            self.cmb_basic_quality.currentIndex())
        self.chk_gpt_oversea.setChecked(
            self.cmb_basic_oversea.currentIndex() == 1)
        self._qo_syncing = False
        self._refresh_basic_info()

    def _on_dialog_qo_changed(self, *_):
        if self._qo_syncing:
            return
        self._qo_syncing = True
        self.cmb_basic_quality.setCurrentIndex(
            self.cmb_gpt_quality.currentIndex())
        self.cmb_basic_oversea.setCurrentIndex(
            1 if self.chk_gpt_oversea.isChecked() else 0)
        self._qo_syncing = False

    def _refresh_basic_info(self) -> None:
        """笨蛋模式「输出参数」只读信息：实际像素/模型/参考图/输出目录。"""
        if not hasattr(self, "lbl_basic_info"):
            return
        tier = {"1k": "1024×1024", "2k": "2048×2048", "4k": "2880×2880"}
        size = self.cmb_gpt_size.currentText() if hasattr(
            self, "cmb_gpt_size") else "1k"
        model = self.cmb_gpt_model.currentText() if hasattr(
            self, "cmb_gpt_model") else "gpt-image-2"
        engine = "云端 GPT-Image" if self.chk_gpt.isChecked() else "本地 ComfyUI"
        n = len(getattr(self, "_basic_ref_paths", []) or [])
        quality = self.cmb_basic_quality.currentText().split("（")[0] \
            if hasattr(self, "cmb_basic_quality") else "medium"
        route = "海外" if (hasattr(self, "cmb_basic_oversea")
                           and self.cmb_basic_oversea.currentIndex() == 1) \
            else "国内"
        try:
            out_dir = str(self.settings.output_path())
        except Exception:  # noqa: BLE001
            out_dir = "-"
        self.lbl_basic_info.setText(
            f"引擎 {engine} ｜ 模型 {model}\n"
            f"输出尺寸 {size} → {tier.get(size, size)} 像素"
            f"（网格图，切分在工具坊完成）\n"
            f"画质 {quality} ｜ 线路 {route}\n"
            f"参考图 {n} 张（留空 = 文生图） ｜ 输出目录 {out_dir}")

    def _basic_res_pick(self, size: str) -> None:
        """笨蛋模式分辨率档选择 → 同步云端引擎参数。"""
        self.cmb_gpt_size.setCurrentText(size)
        for btn, s in ((self.btn_res_1k, "1k"), (self.btn_res_2k, "2k"),
                       (self.btn_res_4k, "4k")):
            btn.setChecked(s == size)
        self._refresh_basic_info()

    def _set_row_visible(self, form, widget: object, visible: bool) -> None:
        """隐藏控件及其 QFormLayout 标签。"""
        if hasattr(widget, "setVisible"):
            widget.setVisible(visible)
        try:
            lbl = form.labelForField(widget)
        except Exception:
            lbl = None
        if lbl is not None:
            lbl.setVisible(visible)

    def _apply_ui_mode(self, mode: int) -> None:
        """UI 三档模式：0=笨蛋（极简） 1=专注（核心） 2=专业（全量）。"""
        mode = max(0, min(2, int(mode)))
        basic = mode == 0
        # 核心区：笨蛋隐藏（用笨蛋卡片），专注/专业显示
        for wdg in self._g2_focus:
            self._set_row_visible(self._form_g2, wdg, not basic)
        # 专业专属行：仅专业模式显示
        for wdg in getattr(self, "_g2_pro", []):
            self._set_row_visible(self._form_g2, wdg, mode == 2)
        # 笨蛋卡片：仅笨蛋显示
        for wdg in self._basic_cards:
            wdg.setVisible(basic)
        # 生成按钮：笨蛋用专属按钮，专注/专业用标准按钮
        self.btn_generate.setVisible(not basic)
        self.btn_generate_basic.setVisible(basic)
        # 低频设置在「⚙ 设置」对话框中（与模式无关）
        self.settings.ui_mode = mode

    def _set_row_visible(self, form, widget: object, visible: bool) -> None:
        """隐藏控件及其 QFormLayout 标签。"""
        if hasattr(widget, "setVisible"):
            widget.setVisible(visible)
        try:
            lbl = form.labelForField(widget)
        except Exception:
            lbl = None
        if lbl is not None:
            lbl.setVisible(visible)
    def _build_center_panel(self) -> QWidget:
        """简化：专业看图器 + 两个按钮（审核区改为隐藏承载，功能不删）。"""
        from pixelforge.ui.image_viewer import ProImageViewer
        self.viewer = ProImageViewer()
        self.viewer.state_message.connect(self.log)   # waifu2x 进度/错误进日志

        # 审核区改为隐藏承载：资产重载/后处理照常喂数据，不再占中央面板
        self.review = ReviewWidget()
        self.review.setVisible(False)
        self.review.state_changed.connect(self.log)
        self.review.reroll_requested.connect(self._on_reroll_frame)
        self.review.edit_requested.connect(self._on_edit_frame)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        self.btn_to_lab = QPushButton("→ 图像工具坊")
        self.btn_to_lab.setToolTip("把当前预览图发送到图像工具坊处理")
        self.btn_to_lab.setProperty("accent", True)
        self.btn_to_lab.clicked.connect(self._send_preview_to_lab)
        self.btn_edit_preview = QPushButton("✏ 编辑")
        self.btn_edit_preview.setToolTip("用像素编辑器编辑当前预览图")
        self.btn_edit_preview.clicked.connect(self._edit_preview_image)
        bl.addWidget(self.btn_to_lab)
        bl.addWidget(self.btn_edit_preview)
        bl.addStretch(1)
        # 右侧常用按钮：当前预览图直接作为生成素材
        self.btn_prev_as_material = QPushButton("🎯 作为生成素材")
        self.btn_prev_as_material.setToolTip(
            "把当前预览图作为参考图（替换现有参考图），用于 AI 修图/编辑")
        self.btn_prev_as_material.clicked.connect(
            lambda: self._preview_as_material(append=False))
        self.btn_prev_as_material_extra = QPushButton("📎 作为附属生成素材")
        self.btn_prev_as_material_extra.setToolTip(
            "把当前预览图追加到参考图列表（不清空现有参考图），"
            "适合多图合成/换装/加道具")
        self.btn_prev_as_material_extra.clicked.connect(
            lambda: self._preview_as_material(append=True))
        bl.addWidget(self.btn_prev_as_material)
        bl.addWidget(self.btn_prev_as_material_extra)

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.viewer, 1)
        v.addWidget(btn_row)
        return w

    def _preview_as_material(self, append: bool = False) -> None:
        """把当前预览图作为生成素材（替换/追加参考图）。"""
        p = self.viewer.current_path()
        if not p:
            self.log("预览区没有图片——先生成或打开一张图")
            return
        self._on_gen_material([p], append=append)
        self.log(f"预览图已{'追加到' if append else '设为'}参考图: "
                 f"{Path(p).name}")

    def _send_preview_to_lab(self) -> None:
        """当前预览图 → 图像工具坊。"""
        p = self.viewer.current_path()
        if not p:
            self.log("没有可发送的预览图")
            return
        self.image_lab.load_path(Path(p))
        self.centralWidget().setCurrentWidget(self.image_lab)
        self.log(f"已发送预览图到图像工具坊: {Path(p).name}")

    def _edit_preview_image(self) -> None:
        """当前预览图 → 像素编辑器，保存后刷新预览。"""
        p = self.viewer.current_path()
        if not p:
            self.log("没有可编辑的预览图")
            return
        from pixelforge.ui.pixel_editor import PixelEditorDialog
        dlg = PixelEditorDialog(p, self)
        dlg.edit_done.connect(lambda _p: self.viewer.set_image(p))
        dlg.show()

    # ---------- 持久化任务队列 ----------

    def _load_persistent_queue(self) -> None:
        """启动时把历史队列记录回填到队列列表（最新在前）。"""
        from pixelforge.config import load_gen_queue, save_gen_queue
        recs = load_gen_queue()
        changed = False
        self._qf_suspend = True          # 批量恢复时暂停逐条过滤，最后统一刷一次
        for rec in reversed(recs):          # 最新在前插入
            if rec.get("status") == "running":   # 上次会话未结束（崩溃/强退）
                rec["status"] = "stopped"
                rec.setdefault("error", "上次会话未结束（程序退出）")
                changed = True
            self._queue_add_item(rec)
        self._qf_suspend = False
        if hasattr(self, "cmb_qfilter"):
            self._apply_queue_filter()
        if changed:
            save_gen_queue(recs)
        if recs:
            self.log(f"已恢复历史队列 {len(recs)} 条（右键可复用/删除）")

    def _queue_add_item(self, rec: dict, tn: int | None = None) -> QListWidgetItem:
        """队列列表加一项（最新在前）。
        tn 非空 = 本次会话的运行中任务：UserRole 存会话任务号（停止/完成标记用），
        UserRole+2 存持久化 id；恢复的历史条目两者都存 id。"""
        mark = {"done": "✔", "failed": "✘", "stopped": "⏹"}.get(
            rec.get("status", ""), "…")
        label = f"#{tn}" if tn is not None else f"#{rec.get('id')}"
        prompt = rec.get("prompt") or ""
        it = QListWidgetItem(
            f"{label} {mark} [{(rec.get('engine') or 'GPT')}] "
            f"{rec.get('time', '')} {prompt[:28]}"
            + ("…" if len(prompt) > 28 else ""))
        rid = rec.get("id")
        it.setData(Qt.UserRole, tn if tn is not None else rid)
        it.setData(Qt.UserRole + 1, rec.get("results") or [])
        it.setData(Qt.UserRole + 2, rid)
        # 不设原生 tooltip：悬停信息由 _on_queue_hover 的卡片统一展示
        # （原生 tooltip 会与悬停卡片重叠成两个浮窗）
        self.lst_queue.insertItem(0, it)
        if hasattr(self, "cmb_qfilter") and not getattr(
                self, "_qf_suspend", False):
            self._apply_queue_filter()
        return it

    def _queue_id_of(self, it) -> object:
        """列表项对应的持久化记录 id（运行中任务存在 UserRole+2）。"""
        if it is None:
            return None
        return it.data(Qt.UserRole + 2) or it.data(Qt.UserRole)

    def _queue_record_of(self, it) -> dict | None:
        from pixelforge.config import load_gen_queue
        tid = self._queue_id_of(it)
        if tid is None:
            return None
        for r in load_gen_queue():
            if r.get("id") == tid:
                return r
        return None

    def _queue_update(self, qid, **fields) -> None:
        """更新一条持久化队列记录（按 id 合并字段）。"""
        from pixelforge.config import load_gen_queue, save_gen_queue
        recs = load_gen_queue()
        for i, r in enumerate(recs):
            if r.get("id") == qid:
                recs[i].update(fields)
                recs[i]["id"] = qid
                save_gen_queue(recs)
                break
        if hasattr(self, "cmb_qfilter"):
            self._apply_queue_filter()

    @staticmethod
    def _fuzzy_hit(text: str, query: str) -> bool:
        """模糊匹配：query 空白分词，每词需为子串或顺序子序列命中。"""
        if not query:
            return True
        t = text.lower()
        for tok in query.lower().split():
            if tok in t:
                continue
            i = 0
            for ch in t:
                if i < len(tok) and ch == tok[i]:
                    i += 1
                    if i == len(tok):
                        break
            if i < len(tok):
                return False
        return True

    def _apply_queue_filter(self) -> None:
        """按状态下拉 + 关键词过滤队列列表（隐藏不匹配项，不删数据）。"""
        if not hasattr(self, "cmb_qfilter"):
            return
        from pixelforge.config import load_gen_queue
        status_map = {"进行中": "running", "已完成": "done",
                      "失败": "failed", "已停止": "stopped"}
        want = status_map.get(self.cmb_qfilter.currentText())
        query = self.ed_qsearch.text().strip()
        recs = {r.get("id"): r for r in load_gen_queue()}
        total = self.lst_queue.count()
        shown = 0
        for i in range(total):
            it = self.lst_queue.item(i)
            rec = recs.get(self._queue_id_of(it)) or {}
            ok = True
            if want and rec.get("status", "running") != want:
                ok = False
            if ok and query:
                hay = " ".join([
                    str(rec.get("prompt", "")), str(rec.get("negative_prompt", "")),
                    str(rec.get("model", "")), str(rec.get("size", "")),
                    str(rec.get("engine", "")), str(rec.get("status", "")),
                    str(rec.get("error", "")), it.text(),
                ])
                if not self._fuzzy_hit(hay, query):
                    ok = False
            it.setHidden(not ok)
            shown += 1 if ok else 0
        if hasattr(self, "lbl_qcount"):
            self.lbl_qcount.setText(f"显示 {shown} / 共 {total}")

    def _queue_load_settings(self) -> None:
        """双击队列项：把该历史任务的提示词/模型/尺寸等载入界面复用。"""
        rec = self._queue_record_of(self.lst_queue.currentItem())
        if rec is None:
            return
        ed_p, ed_n = self._active_prompt_editors()
        ed_p.setPlainText(rec.get("prompt", ""))
        ed_n.setPlainText(rec.get("negative_prompt", ""))
        if rec.get("model"):
            self.cmb_gpt_model.setCurrentText(rec["model"])
        if rec.get("size"):
            self.cmb_gpt_size.setCurrentText(rec["size"])
        if rec.get("quality"):
            self.cmb_gpt_quality.setCurrentText(
                {"low": "low（快速）", "high": "high（精细）"}.get(
                    rec["quality"], "medium（标准）"))
        if "oversea" in rec:
            self.chk_gpt_oversea.setChecked(bool(rec.get("oversea")))
        refs = [p for p in (rec.get("refs") or []) if Path(p).is_file()]
        if refs:
            self._basic_ref_paths = self._stash_local_assets(refs)[:9]
            self._refresh_basic_thumbs()
        self.chk_gpt.setChecked(rec.get("engine", "GPT") != "本地")
        if hasattr(self, "_sync_engine_buttons"):
            self._sync_engine_buttons()
        self.log(f"已载入任务 #{rec.get('id')} 的设置（可直接修改后重新生成）")

    def _queue_menu(self, pos) -> None:
        """队列右键菜单：复用/导入提示词/预览/工具坊/删除/永久删除。"""
        from PySide6.QtWidgets import QMenu
        it = self.lst_queue.itemAt(pos)
        if it is None:
            return
        if it not in self.lst_queue.selectedItems():
            self.lst_queue.setCurrentItem(it)   # 保留多选状态
        rec = self._queue_record_of(it) or {}
        n_res = len(rec.get("results") or [])
        menu = QMenu(self)
        sel = self.lst_queue.selectedItems()
        act_cmp = None
        if len(sel) == 2:
            act_cmp = menu.addAction("⚖ 并排对比两条任务（A/B）")
            menu.addSeparator()
        act_load = menu.addAction("📥 载入该任务设置（复用修改）")
        act_retry = menu.addAction("↻ 重跑该任务（复用相同设置立即生成）")
        act_prompt = menu.addAction("📋 导入该图提示词")
        menu.addSeparator()
        act_preview = menu.addAction("🔍 在预览中打开产物")
        act_lab = menu.addAction("→ 发送到图像工具坊")
        act_dir = menu.addAction("📂 打开产物目录")
        menu.addSeparator()
        act_del = menu.addAction("🗑 从队列移除（保留文件）")
        act_purge = menu.addAction(f"☠ 永久删除（{n_res} 个产物文件）")
        chosen = menu.exec(self.lst_queue.mapToGlobal(pos))
        if chosen is None:
            return
        if act_cmp is not None and chosen == act_cmp:
            self._queue_compare(sel)
        elif chosen == act_load:
            self._queue_load_settings()
        elif chosen == act_retry:
            self._queue_retry(rec)
        elif chosen == act_prompt:
            self._queue_import_prompt(rec)
        elif chosen == act_preview:
            self._queue_open_result(rec, preview=True)
        elif chosen == act_lab:
            self._queue_open_result(rec, lab=True)
        elif chosen == act_dir:
            self._queue_open_result(rec, open_dir=True)
        elif chosen == act_del:
            self._queue_delete(it, purge=False)
        elif chosen == act_purge:
            self._queue_delete(it, purge=True)

    def _queue_compare(self, items: list) -> None:
        """⚖ 历史任务对比：队列两条选中任务的产物并排 A/B。"""
        recs = [self._queue_record_of(it) or {} for it in items]
        paths = []
        for r in recs:
            found = next((p for p in (r.get("results") or [])
                          if Path(p).is_file()), None)
            paths.append(found)
        missing = [i + 1 for i, p in enumerate(paths) if p is None]
        if len(missing) == 2:
            self.log("两条任务都没有产物文件，无法对比")
            return
        if missing:
            self.log(f"第 {missing[0]} 条任务没有产物文件，无法对比")
            return
        import cv2
        import numpy as np
        arrs = []
        for p in paths:
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8),
                              cv2.IMREAD_UNCHANGED)
            if im is None:
                self.log(f"无法读取产物图: {p}")
                return
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            arrs.append(im)

        def _lbl(idx: int) -> str:
            r = recs[idx] or {}
            t = (r.get("time") or "").strip()
            pr = (r.get("prompt") or "").strip().replace("\n", " ")[:18]
            return f"{'A' if idx == 0 else 'B'}  {t} {pr}".strip()

        from pixelforge.ui.image_lab import open_ab_compare_images
        dlg = open_ab_compare_images(
            self, arrs[0], arrs[1],
            title="历史任务对比 —— 队列选中 2 条",
            label_a=_lbl(0), label_b=_lbl(1))
        dlg.exec()

    def _queue_retry(self, rec: dict) -> None:
        """↻ 重跑：载入该任务设置后，用完全相同参数立即重新提交生成。"""
        if not rec:
            self.log("重跑失败：该队列条目没有可复用的记录")
            return
        tid = rec.get("id")
        local = rec.get("engine", "GPT") == "本地"
        if not local and not self.ed_gpt_key.text().strip():
            self.log("已载入设置，但云端 API Key 为空——"
                     "请在设置里填写 API Key 后再点「生成」")
            return
        self._queue_load_settings()     # 回填提示词/模型/尺寸/参考图/引擎
        self.log(f"↻ 已载入任务 #{tid} 的设置，正在以相同参数重新提交…")
        self.on_generate()

    def _queue_import_prompt(self, rec: dict) -> None:
        if not rec.get("prompt"):
            self.log("该队列条目没有提示词记录")
            return
        self._on_prompt_loaded({
            "prompt": rec.get("prompt", ""),
            "negative_prompt": rec.get("negative_prompt", ""),
            "model": rec.get("model", ""),
            "size": rec.get("size", ""),
        })

    def _queue_open_result(self, rec: dict, preview: bool = False,
                           lab: bool = False, open_dir: bool = False) -> None:
        results = [p for p in (rec.get("results") or []) if Path(p).is_file()]
        if not results:
            self.log("该队列条目没有可打开的产物文件")
            return
        first = results[0]
        if preview:
            self.viewer.set_image(first)
            self.tabs.setCurrentIndex(0)
        elif lab:
            self.image_lab.load_path(Path(first))
            self.tabs.setCurrentWidget(self.image_lab)
        elif open_dir:
            try:
                os.startfile(str(Path(first).parent))
            except Exception:  # noqa: BLE001
                self.log(f"打开目录失败: {Path(first).parent}")

    def _queue_delete(self, it, purge: bool) -> None:
        """删除队列项；purge=True 时同时删除本地产物文件并从资产库移除。"""
        tid = self._queue_id_of(it)
        rec = self._queue_record_of(it) or {}
        results = [p for p in (rec.get("results") or []) if Path(p).is_file()]
        if not purge:
            r = QMessageBox.question(
                self, "PixelForge",
                f"仅从队列移除任务 #{tid}？\n（本地文件与资产库保留）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        else:
            names = "\n".join(Path(p).name for p in results[:6]) or "（无文件）"
            r = QMessageBox.warning(
                self, "PixelForge — 危险操作",
                f"⚠️ 此操作不可撤销！\n\n将永久删除任务 #{tid} 的 "
                f"{len(results)} 个产物文件，并从队列、生成资产库、资产库中移除，"
                f"只保留日志。\n\n文件：\n{names}"
                + (f"\n…等共 {len(results)} 个" if len(results) > 6 else "")
                + "\n\n确定继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
            # 再确认一次（危险操作二次确认）
            r2 = QMessageBox.warning(
                self, "PixelForge — 最后确认",
                f"真的要删除这 {len(results)} 个文件吗？此操作无法撤销。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r2 != QMessageBox.Yes:
                return
            if self._purge_results(results):
                self._drop_results_from_indexes(results)
        # 队列记录与列表项移除
        from pixelforge.config import load_gen_queue, save_gen_queue
        recs = [r for r in load_gen_queue() if r.get("id") != tid]
        save_gen_queue(recs)
        row = self.lst_queue.row(it)
        self.lst_queue.takeItem(row)
        self.log(f"已{'永久删除' if purge else '移除'}队列任务 #{tid}"
                 + (f"（删除 {len(results)} 个文件）" if purge else "（保留文件）"))

    def _purge_results(self, results: list) -> bool:
        """删除产物文件（永久删除时留一份日志，便于追溯）。"""
        from datetime import datetime as _dt
        n = 0
        lines = []
        for p in results:
            try:
                path = Path(p)
                if path.is_file():
                    path.unlink()
                    n += 1
                    lines.append(json.dumps(
                        {"time": f"{_dt.now():%Y-%m-%d %H:%M:%S}",
                         "file": str(path)}, ensure_ascii=False))
            except OSError as e:
                self.log(f"删除失败 {p}: {e}")
        if lines:
            try:
                self.PURGE_LOG.parent.mkdir(parents=True, exist_ok=True)
                with self.PURGE_LOG.open("a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except OSError as e:  # noqa: BLE001
                self.log(f"[删除日志] 写入失败: {e}")
        return n > 0

    def _drop_results_from_indexes(self, results: list) -> None:
        """从生成资产库（画廊）与资产库索引里剔除已删文件。"""
        names = {Path(p).name for p in results}
        # 生成资产库：gallery_index.json（纯路径列表）
        try:
            try:
                paths = json.loads(self.GALLERY_INDEX.read_text("utf-8"))
                paths = paths if isinstance(paths, list) else \
                    (paths.get("items") if isinstance(paths, dict) else [])
            except (OSError, json.JSONDecodeError):
                paths = []
            kept = [p for p in paths if Path(str(p)).name not in names]
            self.GALLERY_INDEX.parent.mkdir(parents=True, exist_ok=True)
            self.GALLERY_INDEX.write_text(
                json.dumps(kept, ensure_ascii=False, indent=2), "utf-8")
            self._refresh_gallery_wall()
        except Exception as e:  # noqa: BLE001
            self.log(f"[生成资产库] 清理失败: {e}")
        # 资产库索引（favorites/folders/categories/prompts）
        try:
            idx = self.assets_panel._load_index()
            for key in ("favorites", "folders", "categories", "prompts"):
                if isinstance(idx.get(key), list):
                    idx[key] = [x for x in idx[key]
                                if Path(str(x)).name not in names]
                elif isinstance(idx.get(key), dict):
                    idx[key] = {k: v for k, v in idx[key].items()
                                if k not in names}
            self.assets_panel._save_index(idx)
            self.assets_panel.refresh()
        except Exception as e:  # noqa: BLE001
            self.log(f"[资产库索引] 清理失败: {e}")

    def _refresh_gallery_wall(self) -> None:
        """按索引文件重建生成资产库（画廊）缩略图墙。"""
        if not hasattr(self, "lst_gallery_wall"):
            return
        self.lst_gallery_wall.clear()
        try:
            paths = json.loads(self.GALLERY_INDEX.read_text("utf-8"))
            paths = paths if isinstance(paths, list) else \
                (paths.get("items") if isinstance(paths, dict) else [])
        except (OSError, json.JSONDecodeError):
            paths = []
        seen = set()
        for p in reversed(paths[-200:]):   # 倒序插入 → 最上 = 最新
            key = Path(str(p)).resolve() if Path(str(p)).is_absolute() \
                else p
            if key in seen:
                continue
            seen.add(key)
            self._add_gallery_item(str(p))

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self.right_tabs = QTabWidget(w)

        # 标签 1：队列与日志
        w_log = QWidget()
        lay_log = QVBoxLayout(w_log)
        lay_log.setContentsMargins(4, 4, 4, 4)
        # 队列筛选栏：状态下拉 + 关键词模糊过滤
        row_qf = QHBoxLayout()
        row_qf.addWidget(QLabel("筛选"))
        self.cmb_qfilter = QComboBox()
        self.cmb_qfilter.addItems(["全部", "进行中", "已完成", "失败", "已停止"])
        self.cmb_qfilter.setToolTip("按任务状态过滤队列")
        self.cmb_qfilter.currentIndexChanged.connect(
            lambda _: self._apply_queue_filter())
        row_qf.addWidget(self.cmb_qfilter)
        self.ed_qsearch = QLineEdit()
        self.ed_qsearch.setPlaceholderText("🔍 过滤队列（提示词/模型/引擎，空格分词）")
        self.ed_qsearch.setClearButtonEnabled(True)
        self.ed_qsearch.setToolTip("模糊过滤：多个关键词用空格分隔，需全部命中")
        self.ed_qsearch.textChanged.connect(
            lambda _: self._apply_queue_filter())
        row_qf.addWidget(self.ed_qsearch, 1)
        self.lbl_qcount = QLabel("")
        self.lbl_qcount.setStyleSheet("color:#8a8f98; font-size:11px;")
        row_qf.addWidget(self.lbl_qcount)
        lay_log.addLayout(row_qf)
        self.lst_queue = QListWidget()
        self.lst_queue.setObjectName("queueList")
        self.lst_queue.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst_queue.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst_queue.customContextMenuRequested.connect(self._queue_menu)
        self.lst_queue.itemDoubleClicked.connect(
            lambda _: self._queue_load_settings())
        # 悬停浮窗：预览该任务的参考图/产物/提示词
        self.lst_queue.setMouseTracking(True)
        self.lst_queue.itemEntered.connect(self._on_queue_hover)
        self.lst_queue.viewport()._pf_hide_hover = True
        self.lst_queue.viewport().installEventFilter(self)
        lay_log.addWidget(self.lst_queue, 1)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setObjectName("logView")
        self.txt_log.setReadOnly(True)
        lay_log.addWidget(self.txt_log, 2)
        self._load_persistent_queue()   # 需在 txt_log 之后（日志输出）
        tips_log = QLabel("右键队列项=载入设置/导入提示词/预览/删除｜"
                          "双击=复用该任务设置修改后重新生成｜⏹=停止选中任务")
        tips_log.setStyleSheet("color:#8a8f98; font-size:11px;")
        tips_log.setWordWrap(True)
        lay_log.addWidget(tips_log)
        self.right_tabs.addTab(w_log, "队列日志")

        # 标签 2：生成资产（缩略图墙，持久化索引）
        w_gal = QWidget()
        lay_gal = QVBoxLayout(w_gal)
        lay_gal.setContentsMargins(4, 4, 4, 4)
        self.lst_gallery_wall = QListWidget()
        self.lst_gallery_wall.setViewMode(QListWidget.IconMode)
        self.lst_gallery_wall.setIconSize(QSize(112, 112))
        self.lst_gallery_wall.setResizeMode(QListWidget.Adjust)
        self.lst_gallery_wall.setWordWrap(True)
        self.lst_gallery_wall.itemDoubleClicked.connect(
            self._open_gallery_item)
        self.lst_gallery_wall.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst_gallery_wall.customContextMenuRequested.connect(
            self._gallery_menu)
        lay_gal.addWidget(self.lst_gallery_wall, 1)
        row_gal = QHBoxLayout()
        b_gal_open = QPushButton("打开原图")
        b_gal_open.clicked.connect(self._open_gallery_item)
        row_gal.addWidget(b_gal_open)
        b_gal_clean = QPushButton("清理失效条目")
        b_gal_clean.clicked.connect(self._clean_gallery)
        row_gal.addWidget(b_gal_clean)
        row_gal.addStretch(1)
        lay_gal.addLayout(row_gal)
        tips_gal = QLabel("最新生成在最上面｜双击=打开原图｜右键=打开/移除｜"
                          "新图自动入墙（最多保留 200 条）")
        tips_gal.setStyleSheet("color:#8a8f98; font-size:11px;")
        tips_gal.setWordWrap(True)
        lay_gal.addWidget(tips_gal)
        self.right_tabs.addTab(w_gal, "生成资产")

        lay.addWidget(self.right_tabs, 1)
        self._load_gallery_index()
        return w

    # ---------- 生成画廊 ----------

    GALLERY_INDEX = (Path(__file__).resolve().parent.parent.parent /
                     "config" / "gallery_index.json")
    PURGE_LOG = (Path(__file__).resolve().parent.parent.parent /
                 "config" / "purge_log.jsonl")

    def _load_gallery_index(self) -> None:
        """加载画廊索引（路径列表，约定最新在前），去重并过滤已删文件。
        倒序插入（_add_gallery_item 恒插顶部）→ 墙面最上 = 最新生成。"""
        try:
            paths = json.loads(self.GALLERY_INDEX.read_text("utf-8"))
            paths = paths if isinstance(paths, list) else \
                (paths.get("items") if isinstance(paths, dict) else [])
        except (OSError, json.JSONDecodeError):
            paths = []
        seen = set()
        for p in reversed(paths[-200:]):
            key = Path(str(p)).resolve() if Path(str(p)).is_absolute() \
                else p
            if key in seen:
                continue
            seen.add(key)
            if Path(p).is_file():
                self._add_gallery_item(str(p))

    def _add_gallery_item(self, path: str) -> None:
        from PySide6.QtGui import QIcon
        if not Path(path).is_file():
            return
        it = QListWidgetItem(Path(path).name)
        it.setIcon(QIcon(path))
        it.setData(Qt.UserRole, path)
        it.setToolTip(path)
        self.lst_gallery_wall.insertItem(0, it)   # 新图永远在最上面

    def _open_gallery_item(self, item=None) -> None:
        it = item or self.lst_gallery_wall.currentItem()
        if it is None:
            it = self.lst_gallery_wall.item(0)   # 无选中 → 打开最新一张
        if it is None:
            return
        os.startfile(it.data(Qt.UserRole))

    def _gallery_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu
        it = self.lst_gallery_wall.itemAt(pos)
        if it is None:
            return
        menu = QMenu(self)
        menu.addAction("打开原图", lambda: self._open_gallery_item(it))
        menu.addAction("打开所在文件夹", lambda: os.startfile(
            str(Path(it.data(Qt.UserRole)).parent)))
        menu.addAction("从画廊移除",
                       lambda: self.lst_gallery_wall.takeItem(
                           self.lst_gallery_wall.row(it)))
        menu.exec(self.lst_gallery_wall.mapToGlobal(pos))

    def _clean_gallery(self) -> None:
        removed = 0
        for i in range(self.lst_gallery_wall.count() - 1, -1, -1):
            p = self.lst_gallery_wall.item(i).data(Qt.UserRole)
            if not Path(p).is_file():
                self.lst_gallery_wall.takeItem(i)
                removed += 1
        self.log(f"画廊清理：移除 {removed} 个失效条目")

    def _save_gallery_index(self) -> None:
        """保存画廊索引：UI 顺序即最新在前，只保留前 200 条。"""
        self.GALLERY_INDEX.parent.mkdir(parents=True, exist_ok=True)
        paths = [self.lst_gallery_wall.item(i).data(Qt.UserRole)
                 for i in range(self.lst_gallery_wall.count())]
        self.GALLERY_INDEX.write_text(
            json.dumps(paths[:200], ensure_ascii=False, indent=2), "utf-8")

    # ---------- 设置 <-> UI ----------

    def _collect_settings(self) -> Settings:
        s = self.settings
        s.conda_env = self.cmb_env.currentText()
        s.comfyui_path = self.ed_comfy_path.text().strip()
        s.auto_start_comfyui = self.chk_autostart.isChecked()
        s.cloud_mode = self.chk_cloud.isChecked()
        s.cloud_api_base = self.ed_cloud_base.text().strip()
        s.cloud_api_key = self.ed_cloud_key.text().strip()
        s.gpt_engine = self.chk_gpt.isChecked()
        s.gpt_base_url = self.ed_gpt_base.text().strip() or "https://duomiapi.com"
        from pixelforge.gpt_engine import _sanitize_base
        s.gpt_base_url = _sanitize_base(s.gpt_base_url)
        s.gpt_api_key = self.ed_gpt_key.text().strip()
        s.gpt_model = self.cmb_gpt_model.currentText().strip()
        s.gpt_size = self.cmb_gpt_size.currentText()
        s.gpt_quality = self.cmb_gpt_quality.currentText().split("（")[0]
        s.gpt_oversea = self.chk_gpt_oversea.isChecked()
        s.gpt_background = (
            "transparent" if "transparent" in self.cmb_gpt_bg.currentText()
            else "opaque" if "opaque" in self.cmb_gpt_bg.currentText()
            else "auto")
        s.checkpoint = self.cmb_ckpt.currentText()
        s.prompt = self.ed_prompt.toPlainText().strip()
        s.negative_prompt = self.ed_negative.toPlainText().strip()
        s.width, s.height = self.spin_w.value(), self.spin_h.value()
        s.steps, s.cfg = self.spin_steps.value(), self.spin_cfg.value()
        s.lock_seed = self.chk_lock_seed.isChecked()
        s.seed = self.spin_seed.value()
        s.auto_postprocess = self.chk_auto_post.isChecked()
        s.lora_name = self.ed_lora.text().strip()
        s.lora_strength = self.spin_lora.value()
        s.i2i_frames = self.spin_i2i_n.value() if self.chk_i2i.isChecked() else 0
        s.i2i_denoise = self.slide_i2i_denoise.value() / 100.0
        s.controlnet_name = self.cmb_cn.currentText()
        s.pose_strength = self.spin_pose_str.value()
        s.pose_on = self.chk_pose.isChecked()
        s.pose_paths = [self.lst_pose.item(i).text()
                        for i in range(self.lst_pose.count())]
        s.ref_image = self.ed_ref.text().strip()
        s.action_list = self.ed_actions.toPlainText()
        return s

    def _load_settings_to_ui(self) -> None:
        s = self.settings
        if s.conda_env:
            self.cmb_env.addItem(s.conda_env)  # 刷新后会被真实列表替换/选中
        self.ed_comfy_path.setText(s.comfyui_path)
        self.chk_autostart.setChecked(s.auto_start_comfyui)
        self.chk_cloud.setChecked(s.cloud_mode)
        self.ed_cloud_base.setText(s.cloud_api_base)
        self.ed_cloud_key.setText(s.cloud_api_key)
        self.chk_gpt.setChecked(s.gpt_engine)
        self.ed_gpt_base.setText(s.gpt_base_url)
        self.ed_gpt_key.setText(s.gpt_api_key)
        self.cmb_gpt_model.setCurrentText(s.gpt_model)
        self.cmb_gpt_size.setCurrentText(s.gpt_size)
        self.cmb_gpt_quality.setCurrentText(
            {"low": "low（快速）", "high": "high（精细）"}.get(
                s.gpt_quality, "medium（标准）"))
        self.chk_gpt_oversea.setChecked(bool(s.gpt_oversea))
        self.cmb_gpt_bg.setCurrentIndex(
            {"transparent": 2, "opaque": 1}.get(s.gpt_background, 0))
        self._basic_res_pick(s.gpt_size)
        self.cmb_ui_mode.setCurrentIndex(max(0, min(2, int(s.ui_mode))))
        self.ed_prompt.setPlainText(s.prompt)
        self.ed_negative.setPlainText(s.negative_prompt)
        self.spin_w.setValue(s.width)
        self.spin_h.setValue(s.height)
        self.spin_steps.setValue(s.steps)
        self.spin_cfg.setValue(s.cfg)
        self.chk_lock_seed.setChecked(s.lock_seed)
        self.spin_seed.setValue(s.seed)
        self.chk_auto_post.setChecked(s.auto_postprocess)
        self.ed_lora.setText(s.lora_name)
        self.spin_lora.setValue(s.lora_strength)
        i2i_on = s.i2i_frames >= 2
        self.chk_i2i.setChecked(i2i_on)
        self.spin_i2i_n.setValue(max(2, s.i2i_frames) if i2i_on else 4)
        self.slide_i2i_denoise.setValue(int(round(s.i2i_denoise * 100)))
        self.chk_pose.setChecked(bool(getattr(s, "pose_on", False)))
        self.spin_pose_str.setValue(float(getattr(s, "pose_strength", 1.0)))
        if getattr(s, "controlnet_name", ""):
            if self.cmb_cn.findText(s.controlnet_name) < 0:
                self.cmb_cn.addItem(s.controlnet_name)
            self.cmb_cn.setCurrentText(s.controlnet_name)
        for p in getattr(s, "pose_paths", []) or []:
            if Path(p).is_file():
                self.lst_pose.addItem(p)
        if getattr(s, "ref_image", ""):
            self.ed_ref.setText(s.ref_image)
        if getattr(s, "action_list", ""):
            self.ed_actions.setPlainText(s.action_list)
        if s.checkpoint:
            self.cmb_ckpt.addItem(s.checkpoint)

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._busy:
            self._collect_settings().save()
        # 布局持久化：Dock 位置/大小/浮动状态 + 窗口几何（重启恢复）
        st = self.settings
        qs = QSettings("PixelForge", "PixelForge")
        qs.setValue("window_state", self.saveState())
        qs.setValue("window_geometry", self.saveGeometry())
        self._save_gallery_index()
        del st, qs
        event.accept()

    # ---------- 角色档案 ----------

    def _refresh_profiles(self, select: str = "") -> None:
        self.cmb_profile.blockSignals(True)
        self.cmb_profile.clear()
        self.cmb_profile.addItem("（选择档案）")
        self.cmb_profile.addItems(list(load_profiles().keys()))
        if select:
            self.cmb_profile.setCurrentText(select)
        self.cmb_profile.blockSignals(False)

    def _on_profile_selected(self, name: str) -> None:
        prof = load_profiles().get(name)
        if prof:
            self._apply_profile(prof)
            self.log(f"已应用角色档案: {name}")

    def _apply_profile(self, prof: dict) -> None:
        """把档案字段写入对应控件（纯逻辑，可测）。"""
        self.cmb_ckpt.setCurrentText(prof.get("checkpoint", ""))
        self.ed_lora.setText(prof.get("lora_name", ""))
        self.spin_lora.setValue(float(prof.get("lora_strength", 1.0)))
        self.ed_prompt.setPlainText(prof.get("prompt", ""))
        self.ed_negative.setPlainText(prof.get("negative_prompt", ""))
        self.spin_w.setValue(int(prof.get("width", 512)))
        self.spin_h.setValue(int(prof.get("height", 512)))
        self.spin_steps.setValue(int(prof.get("steps", 20)))
        self.spin_cfg.setValue(float(prof.get("cfg", 7.0)))

    def _profile_data(self) -> dict:
        """从控件抽取档案字段（纯逻辑，可测）。"""
        return {"checkpoint": self.cmb_ckpt.currentText(),
                "lora_name": self.ed_lora.text().strip(),
                "lora_strength": self.spin_lora.value(),
                "prompt": self.ed_prompt.toPlainText().strip(),
                "negative_prompt": self.ed_negative.toPlainText().strip(),
                "width": self.spin_w.value(), "height": self.spin_h.value(),
                "steps": self.spin_steps.value(), "cfg": self.spin_cfg.value()}

    def _save_profile_dialog(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存角色档案",
                                        "档案名称（如：骑士-银甲）：")
        if not ok or not name.strip():
            return
        name = name.strip()
        profs = load_profiles()
        profs[name] = self._profile_data()
        save_profiles(profs)
        self._refresh_profiles(select=name)
        self.log(f"角色档案已保存: {name}")

    def _delete_profile(self) -> None:
        name = self.cmb_profile.currentText()
        profs = load_profiles()
        if name not in profs:
            return
        del profs[name]
        save_profiles(profs)
        self._refresh_profiles()
        self.log(f"角色档案已删除: {name}")

    # ---------- 档案导出/导入（分享给其他人） ----------

    def _export_profile(self) -> None:
        """导出当前选中档案的全部生成设置为 .json（含 PF 版本号）。"""
        from PySide6.QtWidgets import QFileDialog
        name = self.cmb_profile.currentText()
        profs = load_profiles()
        if name not in profs:
            QMessageBox.information(self, "PixelForge",
                                    "请先在下拉中选择要导出的角色档案")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出角色档案", f"{name}.pfpreset",
            "PixelForge 预设 (*.pfpreset *.json)")
        if not path:
            return
        payload = {
            "type": "pixelforge_profile",
            "version": __version__,
            "name": name,
            "profile": profs[name],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        self.log(f"角色档案已导出: {path}")

    def _import_profile(self) -> None:
        """导入他人分享的档案 .json（重名自动加后缀）。"""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "导入角色档案", "",
            "PixelForge 预设 (*.pfpreset *.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "PixelForge", f"导入失败: {e}")
            return
        prof = payload.get("profile") if payload.get("type") == \
            "pixelforge_profile" else payload
        if not isinstance(prof, dict) or "prompt" not in prof:
            QMessageBox.critical(self, "PixelForge",
                                 "文件格式不正确（缺少 profile 字段）")
            return
        name = (payload.get("name") or Path(path).stem).strip()
        profs = load_profiles()
        final = name
        i = 2
        while final in profs:
            final = f"{name}_{i}"
            i += 1
        profs[final] = prof
        save_profiles(profs)
        self._refresh_profiles(select=final)
        self._on_profile_selected(final)
        self.log(f"角色档案已导入: {final}（来自 {Path(path).name}）")

    # ---------- 提示词预设 ----------


    # ---------- 生成历史快照 ----------

    def _save_gen_snapshot(self, s: Settings) -> None:
        """每次生成提交后自动记录当时的全部设置（最新在前，上限 100）。"""
        from dataclasses import asdict
        from datetime import datetime
        data = asdict(s)
        name = (s.prompt or "(空提示词)").strip()[:20] or "untitled"
        snapshot = {"name": name,
                    "time": datetime.now().strftime("%m-%d %H:%M"),
                    **data}
        append_gen_history(snapshot)
        self._refresh_gen_history(select=0)

    def _refresh_gen_history(self, select: int = -1) -> None:
        self.cmb_gen_history.blockSignals(True)
        self.cmb_gen_history.clear()
        for snap in load_gen_history():
            self.cmb_gen_history.addItem(
                f"{snap.get('name', '?')}  ({snap.get('time', '')})")
        if select >= 0 and load_gen_history():
            self.cmb_gen_history.setCurrentIndex(select)
        self.cmb_gen_history.blockSignals(False)

    def _load_gen_snapshot(self) -> None:
        idx = self.cmb_gen_history.currentIndex()
        history = load_gen_history()
        if idx < 0 or idx >= len(history):
            return
        snap = history[idx]
        self._apply_profile(snap)  # 字段兼容：prompt/负面/ckpt/尺寸/步数/CFG
        self.chk_lock_seed.setChecked(bool(snap.get("lock_seed", False)))
        self.spin_seed.setValue(int(snap.get("seed", 0)))
        self.ed_lora.setText(snap.get("lora_name", ""))
        self.spin_lora.setValue(float(snap.get("lora_strength", 1.0)))
        self.ed_ref.setText(snap.get("ref_image", ""))
        # 快照产物预览：有 result_paths 时显示首图
        rp = snap.get("result_paths") or []
        if rp and Path(rp[0]).is_file():
            self.viewer.set_image(rp[0])
        self.chk_gpt.setChecked(bool(snap.get("gpt_engine", False)))
        self.ed_gpt_base.setText(snap.get("gpt_base_url", "https://duomiapi.com"))
        self.ed_gpt_key.setText(snap.get("gpt_api_key", ""))
        self.cmb_gpt_model.setCurrentText(snap.get("gpt_model", "gpt-image-2"))
        self.cmb_gpt_size.setCurrentText(snap.get("gpt_size", "1k"))
        self.log(f"已加载生成快照: {snap.get('name', '?')} "
                 f"({snap.get('time', '')})")

    # ---------- 槽：环境/引擎 ----------

    def _browse_comfy_path(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择 ComfyUI 目录（含 main.py）")
        if d:
            self.ed_comfy_path.setText(d)

    def refresh_envs(self) -> None:
        worker = _EnvRefreshWorker()
        worker.sig.envs.connect(self._on_envs)
        worker.sig.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_envs(self, envs: list, conda_exe: str) -> None:
        self.conda_exe = conda_exe
        current = self.settings.conda_env
        self.cmb_env.blockSignals(True)
        self.cmb_env.clear()
        names = [n for n, _ in envs]
        self.cmb_env.addItems(names)
        if current in names:
            self.cmb_env.setCurrentText(current)
        self.cmb_env.blockSignals(False)
        self.log(f"识别到 {len(names)} 个 Conda 环境: {', '.join(names) or '（无）'}")

    def refresh_checkpoints(self) -> None:
        worker = _CheckpointsWorker(self._collect_settings())
        worker.sig.checkpoints.connect(self._on_checkpoints)
        self.pool.start(worker)

    def _on_checkpoints(self, models: list) -> None:
        if not models:
            self.log("未获取到模型列表（ComfyUI 未连接或无 checkpoint）")
            return
        current = self.cmb_ckpt.currentText()
        self.cmb_ckpt.clear()
        self.cmb_ckpt.addItems(models)
        if current in models:
            self.cmb_ckpt.setCurrentText(current)
        self.log(f"获取到 {len(models)} 个 Checkpoint")

    def refresh_controlnets(self) -> None:
        worker = _ControlNetsWorker(self._collect_settings())
        worker.sig.cnets.connect(self._on_controlnets)
        self.pool.start(worker)

    def _on_controlnets(self, models: list) -> None:
        if not models:
            self.log("未获取到 ControlNet 列表（未连接或 models/controlnet/ 为空）")
            return
        current = self.cmb_cn.currentText()
        self.cmb_cn.clear()
        self.cmb_cn.addItems(models)
        if current in models:
            self.cmb_cn.setCurrentText(current)
        self.log(f"获取到 {len(models)} 个 ControlNet 模型")

    def _add_pose_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "添加骨架图（可多选，按选择顺序生成）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        files = self._stash_local_assets(files)   # 骨架图也收编进本地资产
        for f in files:
            self.lst_pose.addItem(f)
        if files:
            self.chk_pose.setChecked(True)
            self.log(f"已添加 {len(files)} 张骨架图（共 {self.lst_pose.count()} 帧）")

    def _choose_ref(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择角色参考图（作为外观锚定）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if f:
            f = self._stash_local_assets([f])[0]
            self.ed_ref.setText(f)
            self.log(f"参考图已设置：{Path(f).name}（生成时每帧以其为底图）")

    def check_connection(self) -> None:
        worker = _ConnCheckWorker(self._collect_settings())
        worker.sig.conn.connect(self._on_conn)
        self.pool.start(worker)

    def _on_conn(self, online: bool, info: str) -> None:
        if online:
            cloud = self.chk_cloud.isChecked()
            self.lbl_conn.setText("● 已连接（云端）" if cloud else "● 已连接")
            self.lbl_conn.setStyleSheet("color: #35d073; font-weight: bold;")
            self.lbl_gpu.setText(info)
            if not self._connected_once:
                self._connected_once = True
                self.log("ComfyUI 已连接")
                self.refresh_checkpoints()
        else:
            self.lbl_conn.setText("● 未连接")
            self.lbl_conn.setStyleSheet("color: #ff6b5e; font-weight: bold;")
            self.lbl_gpu.setText("")
            self._connected_once = False
        self._update_generate_button_text()

    def _update_generate_button_text(self) -> None:
        """生成按钮统一文案「生成」（笨蛋/专注/专业通用）。"""
        if not hasattr(self, "btn_generate"):
            return
        self.btn_generate.setText("生成")
        self.btn_generate_basic.setText("🎨  生成")

    # ---------- 槽：生成 ----------

    def _refresh_workflows(self) -> None:
        self.cmb_workflow.clear()
        for f in sorted(WORKFLOWS_DIR.glob("*.json")):
            self.cmb_workflow.addItem(f.stem, str(f))
        if self.cmb_workflow.count() == 0:
            self.log(f"警告：{WORKFLOWS_DIR} 下没有工作流模板")

    def _basic_prompt_full(self) -> str:
        """笨蛋模式完整提示词：正面 + 负面框统一合成。"""
        return _compose_prompt(
            self.ed_prompt_basic.toPlainText(),
            self.ed_negative_basic.toPlainText())

    def _start_gpt_generate(self, s: Settings, basic: bool = False) -> None:
        """云端 GPT-Image 引擎生成（动作列表逐个，异步 worker）。"""
        if basic:
            actions = [self._basic_prompt_full() or
                       "8 frames of the same character standing idle with "
                       "subtle breathing"]
            refs = list(self._basic_ref_paths)
            if not refs:
                self.log("[文生图模式] 未上传参考图：纯提示词生成"
                         "（角色外观由提示词描述决定）")
        else:
            actions = [l.strip() for l in
                       (self.ed_actions.toPlainText().splitlines()
                        if hasattr(self.ed_actions, "toPlainText")
                        else self.ed_actions.text().splitlines()) if l.strip()]
            if not actions:
                actions = [
                    "standing idle, subtle breathing and hair sway, 8 nearly "
                    "identical frames",
                    "walking cycle in place, 8 phases of one full stride",
                    "jumping in place: crouch, launch, apex, fall, land",
                ]
            refs = [self.ed_ref.text().strip()] if \
                self.ed_ref.text().strip() else []
        self._save_gen_snapshot(s)
        snap = load_gen_history()[0]   # 本次任务的快照（并发时按 name+time 定位）
        self.pbar.setRange(0, 0)   # 忙碌摆动
        self.pbar.setVisible(True)
        self._gpt_active += 1        # 并发排队：不再全局锁死，可继续点「生成」
        self._task_no += 1
        tn = self._task_no
        self._task_snaps[tn] = {"name": snap.get("name", ""),
                                "time": snap.get("time", "")}
        from pixelforge.config import upsert_queue_record
        from datetime import datetime as _dt
        _now = _dt.now()
        qid = f"{_now:%Y%m%d%H%M%S}-{tn}"
        self._qid_of_tn[tn] = qid
        rec = {
            "id": qid,
            "time": f"{_now:%m-%d %H:%M}",
            "status": "running",
            "engine": "GPT",
            "prompt": s.prompt,
            "negative_prompt": s.negative_prompt,
            "model": s.gpt_model,
            "size": s.gpt_size,
            "quality": s.gpt_quality,
            "oversea": bool(s.gpt_oversea),
            "refs": [str(p) for p in refs],
            "results": [],
        }
        upsert_queue_record(rec)
        item = self._queue_add_item(rec, tn)
        self.lst_queue.setCurrentItem(item)
        self.btn_stop.setEnabled(True)
        self.log(f"云端 GPT 任务 #{tn}：{len(actions)} 个动作，"
                 f"参考图 {len(refs)} 张，模型 {s.gpt_model}（约 1~3 分钟/个）"
                 f"｜可继续点「生成」排队多任务")
        worker = _GPTEngineWorker(s, actions, refs)
        self._gpt_workers[tn] = worker
        worker.sig.progress.connect(self._on_gpt_progress)
        worker.sig.done.connect(lambda paths, t=tn: self._on_gpt_done(paths, t))
        worker.sig.failed.connect(lambda msg, t=tn: self._on_gpt_failed(msg, t))
        QThreadPool.globalInstance().start(worker)

    # ---- 队列任务收尾辅助 ----

    def _find_task_snapshot(self, tn: int) -> dict:
        key = self._task_snaps.get(tn) or {}
        for snap in load_gen_history():
            if (snap.get("name"), snap.get("time")) == \
                    (key.get("name"), key.get("time")):
                return snap
        return {}

    def _finish_gpt_task(self, tn: int) -> None:
        """任务结束（成功/失败/停止）统一收尾：计数、停止按钮。"""
        self._gpt_active = max(0, self._gpt_active - 1)
        self._gpt_workers.pop(tn, None)
        self._task_snaps.pop(tn, None)
        if self._gpt_active == 0:
            self.pbar.setVisible(False)
            self.pbar.setRange(0, 100)
            self.pbar.setValue(0)
            self.btn_stop.setEnabled(False)
        self._update_generate_button_text()

    def _set_queue_done(self, tn: int, mark: str) -> None:
        for i in range(self.lst_queue.count()):
            it = self.lst_queue.item(i)
            if it.data(Qt.UserRole) == tn and not it.text().endswith(mark):
                it.setText(it.text() + f"  {mark}")
                break

    def _stop_selected_task(self) -> None:
        """停止队列中当前选中的任务。"""
        it = self.lst_queue.currentItem()
        if it is None:
            self.log("⏹ 队列为空，没有可停止的任务")
            return
        tn = it.data(Qt.UserRole)
        worker = self._gpt_workers.get(tn)
        if worker is None:
            self.log(f"⏹ 任务 #{tn} 已结束，无需停止")
            return
        worker.cancel()
        it.setText(it.text() + "  ⏹停止中…")
        self.log(f"⏹ 已请求停止任务 #{tn}（云端轮询间隙生效，"
                 "已提交的请求无法撤回）")

    def _set_engine_mode(self, use_api: bool) -> None:
        """顶栏引擎切换：☁ API / 🖥 本地 Comfy。"""
        self.chk_gpt.setChecked(use_api)
        self._sync_engine_buttons()
        self.log("引擎已切换: " + ("☁ 云端 GPT-Image API" if use_api
                                   else "🖥 本地 ComfyUI"))

    def _sync_engine_buttons(self) -> None:
        """顶栏引擎按钮与 chk_gpt 状态互同步。"""
        if not hasattr(self, "btn_engine_api"):
            return
        api = self.chk_gpt.isChecked()
        self.btn_engine_api.setChecked(api)
        self.btn_engine_local.setChecked(not api)
        self._refresh_basic_info()

    def _on_gpt_done(self, grid_paths: list, tn: int = 0) -> None:
        """只生图：网格原图交付，切分/动画由人工在图像工具坊完成。"""
        self._finish_gpt_task(tn)
        self._set_queue_done(tn, "✔")
        qid = self._qid_of_tn.pop(tn, None)
        if qid:
            self._queue_update(qid, status="done",
                               results=[str(p) for p in grid_paths])
        # 回填产物路径到本次任务的快照（并发时按 name+time 定位）
        snap = self._find_task_snapshot(tn)
        history = load_gen_history()
        for k, h in enumerate(history):
            if (h.get("name"), h.get("time")) == \
                    (snap.get("name"), snap.get("time")):
                history[k]["result_paths"] = grid_paths
                from pixelforge.config import GEN_HISTORY_FILE
                GEN_HISTORY_FILE.write_text(
                    json.dumps(history[-100:], ensure_ascii=False, indent=2),
                    "utf-8")
                break
        for p in grid_paths:
            self.viewer.set_image(p)
            break
        self.log(f"✔ 云端生成完成（任务 #{tn}）：{len(grid_paths)} 张网格图")
        for p in grid_paths:
            self._add_gallery_item(p)
        self.log("→ 请到「图像工具坊」用 投影拆分/网格切分 人工切分帧，"
                 "再在审核区筛选与导出")
        # 资产库登记生成提示词（右键「导入提示词」可回填主界面）
        try:
            self.assets_panel.register_prompt(
                [str(p) for p in grid_paths], {
                    "prompt": snap.get("prompt", ""),
                    "negative_prompt": snap.get("negative_prompt", ""),
                    "model": snap.get("gpt_model", ""),
                    "size": snap.get("gpt_size", ""),
                })
        except Exception as e:
            self.log(f"[资产库] 提示词登记失败: {e}")

    def _on_gpt_failed(self, msg: str, tn: int = 0) -> None:
        """云端任务失败/停止：统一收尾 + 队列标记。"""
        self._finish_gpt_task(tn)
        self._set_queue_done(tn, "✘" if "已停止" not in msg else "⏹已停止")
        qid = self._qid_of_tn.pop(tn, None)
        if qid:
            self._queue_update(
                qid, status=("stopped" if "已停止" in msg else "failed"),
                error=msg)
        self._on_failed(msg)

    def _on_asset_preview(self, path: str) -> None:
        """资产库双击 → 在中央预览看图器打开并切到「生成与审核」。"""
        self.viewer.set_image(path)
        self.tabs.setCurrentIndex(0)
        self.log(f"[资产库] 已在预览打开 {Path(path).name}"
                 "（放大镜/滚轮缩放可细看，可发送工具坊或编辑）")

    TAB_TIPS = {
        "gen": "滚轮=缩放｜拖拽=平移｜M/Alt按住=放大镜｜F=适配｜1=1:1｜"
               "生成完成新图自动进「生成资产」",
        "lab": "把图片发进工具坊处理｜处理完「→ 动画播放」或回填｜"
               "误操作可撤销",
        "anim": "载入帧序列/atlas 播放｜FPS 与往返模式可调",
        "batch": "多动作批量排队生成｜可在队列日志右键复用设置",
        "assets": "双击=预览｜右键=重载/生成素材/附属素材/导入提示词｜"
                  "切到本页自动刷新",
        "gallery": "最新生成在最上面｜双击=打开原图",
        "queue": "右键队列项=复用/导入提示词/删除｜双击=载入设置",
    }

    def _on_tab_changed(self, idx: int) -> None:
        """点标签切模块：状态栏提示该模块操作 + 资产库自动刷新。"""
        w = self.tabs.widget(idx) if 0 <= idx < self.tabs.count() else None
        for key, wl in (("gen", self._center_panel), ("lab", self.image_lab),
                        ("anim", self.anim_player),
                        ("batch", self.batch_panel),
                        ("assets", self.assets_panel)):
            if w is wl:
                self.lbl_tab_tips.setText("提示：" + self.TAB_TIPS[key])
        if w is self.assets_panel:
            self.assets_panel.refresh()

    def _stash_local_assets(self, paths: list) -> list[str]:
        """把用户上传/引用的外部素材收编进「本地资产」，返回收编后的路径。

        上传的参考图/骨架图不再散落在临时目录：统一复制到资产库 uploads/
        并登记为本地资产，生成流程改用收编后的持久路径。
        """
        try:
            new = self.assets_panel.import_external(paths, category="local")
        except Exception as e:  # noqa: BLE001 收编失败不阻塞生成
            self.log(f"⚠ 本地资产收编失败（仍按原路径使用）: {e}")
            return [str(p) for p in paths]
        copied = sum(1 for a, b in zip(paths, new) if Path(a) != Path(b))
        if copied:
            self.log(f"[本地资产] 已收编 {copied} 张上传素材 → "
                     "资产库 / 本地资产（uploads/）")
            self.assets_panel.refresh()
        return new

    def _on_gen_material(self, paths: list, append: bool = False) -> None:
        """资产库/预览「作为生成素材」：图片进笨蛋模式参考图。
        append=False 替换现有参考图；True 追加（去重，上限 9 张）。"""
        valid = [str(p) for p in paths if Path(p).is_file()]
        if not valid:
            self.log("生成素材：没有有效图片")
            return
        valid = self._stash_local_assets(valid)
        if append:
            merged = list(self._basic_ref_paths)
            for f in valid:
                if f in merged:
                    continue
                if len(merged) >= 9:
                    self.log("[附属素材] 参考图最多 9 张，多余的已忽略")
                    break
                merged.append(f)
            self._basic_ref_paths = merged
            self.log(f"[附属素材] 已追加参考图，当前 "
                     f"{len(self._basic_ref_paths)} 张"
                     "（可写提示词描述要修改的效果后点「生成」）")
        else:
            dropped = len(valid) - len(valid[:9])
            self._basic_ref_paths = valid[:9]   # 替换式发送，避免旧图混入
            self.log(f"[生成素材] 已发送 {len(self._basic_ref_paths)} 张图作为"
                     "参考图"
                     + (f"（超出上限，忽略 {dropped} 张）" if dropped else "")
                     + "；可写提示词描述要修改的效果后点「生成」")
        self._refresh_basic_thumbs()
        self._refresh_basic_info()
        self.cmb_ui_mode.setCurrentIndex(0)   # 切到笨蛋模式（上传参考图处）
        if not append:
            self.log("[生成素材] 例：keep the character/palette, change only "
                     "the background to a forest scene")

    def _on_prompt_loaded(self, info: dict) -> None:
        """资产库「导入提示词」：回填正面/负面/模型/尺寸到主界面。"""
        prompt = info.get("prompt", "") or ""
        negative = info.get("negative_prompt", "") or ""
        ed_p, ed_n = self._active_prompt_editors()
        ed_p.setPlainText(prompt)
        ed_n.setPlainText(negative)
        model = info.get("model", "")
        if model and hasattr(self, "cmb_gpt_model"):
            self.cmb_gpt_model.setCurrentText(model)
        size = info.get("size", "")
        if size and hasattr(self, "cmb_gpt_size"):
            self.cmb_gpt_size.setCurrentText(size)
        self.chk_gpt.setChecked(True)
        self.log("已从资产库导入生成提示词（含模型/尺寸参数）")

    def _on_gpt_progress(self, msg: str) -> None:
        """云端进度：含 % 的行更新进度条，其余进日志。"""
        import re as _re
        m = _re.search(r"(\d{1,3})%", msg)
        if m:
            self.pbar.setRange(0, 100)
            self.pbar.setValue(int(m.group(1)))
            self.pbar.setVisible(True)
        self.log(msg)

    def _on_basic_refs_dropped(self, files: list) -> None:
        """拖入图片 → 收编进本地资产后追加到参考图（去重，上限 9 张）。"""
        files = self._stash_local_assets(
            [f for f in files if Path(f).is_file()])
        added = 0
        for f in files:
            if f in self._basic_ref_paths:
                continue
            if len(self._basic_ref_paths) >= 9:
                self.log("[笨蛋模式] 参考图最多 9 张，多余的已忽略")
                break
            self._basic_ref_paths.append(f)
            added += 1
        if added:
            self._refresh_basic_thumbs()
            self.log(f"[笨蛋模式] 拖入 {added} 张参考图，"
                     f"当前 {len(self._basic_ref_paths)} 张")

    def _upload_basic_refs(self) -> None:
        """笨蛋模式：多选参考图 → 收编进本地资产 + 独立缩略图卡片。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择角色图片（可多选）", "",
            "图片 (*.png *.jpg *.jpeg *.webp)")
        if not files:
            return
        self._basic_ref_paths = self._stash_local_assets(
            [f for f in files if Path(f).is_file()])
        self._refresh_basic_thumbs()
        self.log(f"[笨蛋模式] 已上传 {len(self._basic_ref_paths)} 张参考图")

    def _remove_basic_ref(self, thumb: RefThumb) -> None:
        """单张移除参考图（× 按钮）。"""
        if thumb.path in self._basic_ref_paths:
            self._basic_ref_paths.remove(thumb.path)
        self._hide_hover_preview()
        self._refresh_basic_thumbs()

    def _refresh_basic_thumbs(self) -> None:
        """重建缩略图卡片流：每图独立卡片，横向可滚动，悬停预览。"""
        while self.h_thumbs.count() > 1:  # 末尾 stretch 不动
            it = self.h_thumbs.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.lbl_thumbs_empty.setVisible(not self._basic_ref_paths)
        for p in self._basic_ref_paths[:9]:
            self.h_thumbs.insertWidget(
                self.h_thumbs.count() - 1,
                RefThumb(p, self._remove_basic_ref,
                         self._show_hover_preview, self._hide_hover_preview))
        n = len(self._basic_ref_paths)
        if n > 9:
            self.log(f"[笨蛋模式] 参考图超过 9 张，仅使用前 9 张")

    def eventFilter(self, obj, event) -> bool:
        """悬停预览：鼠标离开列表/下拉视口时收起浮窗。"""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Leave and \
                getattr(obj, "_pf_hide_hover", False):
            self._hide_hover_preview()
        return super().eventFilter(obj, event)

    def _show_hover_preview(self, path: str, gpos) -> None:
        """悬停浮窗：跟随光标的大图预览（最大 380px）。"""
        self._hide_hover_preview()
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._hover_preview = QLabel(self, Qt.ToolTip)
        self._hover_preview.setObjectName("hoverPreview")
        self._hover_preview.setPixmap(pm.scaled(
            380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._hover_preview.adjustSize()
        # 光标右下偏移，出屏时自动回折
        x = gpos.x() + 18
        y = gpos.y() + 18
        screen = self.screen().geometry()
        if x + self._hover_preview.width() > screen.right():
            x = gpos.x() - self._hover_preview.width() - 10
        if y + self._hover_preview.height() > screen.bottom():
            y = gpos.y() - self._hover_preview.height() - 10
        self._hover_preview.move(max(screen.left(), x), max(screen.top(), y))
        self._hover_preview.show()

    def _hide_hover_preview(self) -> None:
        if self._hover_preview is not None:
            self._hover_preview.deleteLater()
            self._hover_preview = None

    # ---- 悬停卡片：历史快照 / 队列项（参考图 + 生成图 + 提示词）----

    @staticmethod
    def _make_strip(paths: list, height: int = 110,
                    max_n: int = 6) -> QPixmap | None:
        """多张图等高拼成一条缩略图带（最多 max_n 张，超额加 +n 尾块）。"""
        valid = [str(p) for p in paths if p and Path(p).is_file()]
        if not valid:
            return None
        pixmaps = []
        for p in valid[:max_n]:
            pm = QPixmap(p)
            if pm.isNull():
                continue
            pixmaps.append(pm.scaledToHeight(
                height, Qt.SmoothTransformation))
        if not pixmaps:
            return None
        gap = 4
        tail = max(0, len(valid) - max_n)
        total_w = sum(pm.width() for pm in pixmaps) + gap * (len(pixmaps) - 1)
        if tail:
            tail_pm = QPixmap(height, height)
            tail_pm.fill(Qt.transparent)
            pt = QPainter(tail_pm)
            pt.setPen(Qt.white)
            f = pt.font()
            f.setPointSize(14)
            pt.setFont(f)
            pt.drawText(tail_pm.rect(), Qt.AlignCenter, f"+{tail}")
            pt.end()
            total_w += gap + height
        out = QPixmap(max(1, total_w), height)
        out.fill(Qt.transparent)
        pt = QPainter(out)
        x = 0
        for pm in pixmaps:
            pt.drawPixmap(x, 0, pm)
            x += pm.width() + gap
        if tail:
            pt.drawPixmap(x, 0, tail_pm)
        pt.end()
        return out

    def _show_hover_card(self, refs: list, results: list, text: str,
                         gpos, meta: str = "") -> None:
        """悬停卡片：参考图带 + 生成图带 + 提示词 + 参数说明。"""
        from PySide6.QtWidgets import QVBoxLayout as _QV
        strip_r = self._make_strip(refs)
        strip_o = self._make_strip(results)
        text = (text or "").strip()
        meta = (meta or "").strip()
        if strip_r is None and strip_o is None and not text and not meta:
            return
        self._hide_hover_preview()
        card = QWidget(self, Qt.ToolTip)
        card.setObjectName("hoverPreview")
        lay = _QV(card)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)
        if strip_r is not None:
            lbl = QLabel("参考图")
            lbl.setObjectName("sectionLabel")
            lay.addWidget(lbl)
            iv = QLabel()
            iv.setPixmap(strip_r)
            lay.addWidget(iv)
        if strip_o is not None:
            lbl = QLabel("生成图")
            lbl.setObjectName("sectionLabel")
            lay.addWidget(lbl)
            iv = QLabel()
            iv.setPixmap(strip_o)
            lay.addWidget(iv)
        if text:
            tl = QLabel(text if len(text) <= 400 else text[:400] + "…")
            tl.setWordWrap(True)
            tl.setMaximumWidth(380)
            lay.addWidget(tl)
        if meta:
            ml = QLabel(meta)
            ml.setObjectName("dimTip")
            ml.setWordWrap(True)
            ml.setMaximumWidth(380)
            lay.addWidget(ml)
        card.adjustSize()
        x = gpos.x() + 18
        y = gpos.y() + 18
        screen = self.screen().geometry()
        if x + card.width() > screen.right():
            x = gpos.x() - card.width() - 10
        if y + card.height() > screen.bottom():
            y = gpos.y() - card.height() - 10
        card.move(max(screen.left(), x), max(screen.top(), y))
        card.show()
        self._hover_preview = card

    def _on_history_hover(self, idx: int) -> None:
        """历史快照下拉悬停：浮窗预览参考图/生成图/提示词。"""
        if hasattr(idx, "row"):        # 直接接到 QModelIndex 信号也兼容
            idx = idx.row()
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            self._hide_hover_preview()
            return
        history = load_gen_history()
        if not (0 <= idx < len(history)):
            self._hide_hover_preview()
            return
        snap = history[idx]
        refs = [snap.get("ref_image", "")] if snap.get("ref_image") else []
        results = snap.get("result_paths") or []
        text = snap.get("prompt", "")
        meta = "\n".join(filter(None, [
            f"时间 {snap.get('time', '')}｜" +
            ("云端 GPT" if snap.get("gpt_engine") else "本地 ComfyUI"),
            f"模型 {snap.get('gpt_model', '') or snap.get('checkpoint', '')}"
            f"｜尺寸 {snap.get('gpt_size', '') or str(snap.get('width', ''))}"
            f"x{snap.get('height', '')}",
            f"步数 {snap.get('steps', '')}｜CFG {snap.get('cfg', '')}"
            f"｜种子 {snap.get('seed', '')}",
            f"负面: {snap.get('negative_prompt', '')[:120]}"
            if snap.get("negative_prompt") else "",
            "双击/加载 = 恢复该次生成的完整设置",
        ]))
        self._show_hover_card(refs, results, text, QCursor.pos(), meta=meta)

    def _on_queue_hover(self, it) -> None:
        """队列项悬停：浮窗预览参考图/产物/提示词。"""
        rec = self._queue_record_of(it) or {}
        refs = rec.get("refs") or []
        results = rec.get("results") or (it.data(Qt.UserRole + 1) or [])
        text = rec.get("prompt") or ""
        if not refs and not results and not text:
            self._hide_hover_preview()
            return
        marks = {"done": "已完成", "failed": "失败", "stopped": "已停止",
                 "running": "进行中"}
        if rec:
            meta = "\n".join(filter(None, [
                f"#{rec.get('id')} ｜ {marks.get(rec.get('status', ''), '')}"
                f" ｜ {rec.get('time', '')} ｜ "
                f"{'云端 GPT' if rec.get('engine', 'GPT') != '本地' else '本地 ComfyUI'}",
                f"模型 {rec.get('model', '')}｜尺寸 {rec.get('size', '')}"
                + (f"｜画质 {rec.get('quality', '')}"
                   if rec.get("quality") else ""),
                f"参考图 {len(refs)} 张｜产物 {len(results)} 个",
                f"负面: {rec.get('negative_prompt', '')[:120]}"
                if rec.get("negative_prompt") else "",
                f"错误: {str(rec.get('error', ''))[:120]}"
                if rec.get("error") else "",
                "双击=载入该任务设置｜右键=复用/删除",
            ]))
        else:   # 无持久化记录（本会话旧条目）：至少给双击提示
            meta = "双击=载入该任务设置｜右键=复用/删除"
        self._show_hover_card(refs, results, text, QCursor.pos(), meta=meta)

    # ---- 生成前参数摘要确认 ----

    def _gpt_action_count(self) -> int:
        """云端动作行数（空则按默认 3 个动作计）。"""
        txt = (self.ed_actions.toPlainText()
               if hasattr(self.ed_actions, "toPlainText")
               else self.ed_actions.text())
        n = sum(1 for l in txt.splitlines() if l.strip())
        return n if n else 3

    def _gen_summary_rows(self, s, engine: str, refs: int, actions: int):
        """生成参数摘要行（纯数据，便于测试与复用）。"""
        if engine == "gpt":
            quality = {"low": "low（快速）", "high": "high（精细）"}.get(
                s.gpt_quality, "medium（标准）")
            return [
                ("引擎", "云端 GPT-Image"),
                ("模式", "笨蛋模式" if self.cmb_ui_mode.currentIndex() == 0
                 else "专业模式"),
                ("模型", s.gpt_model or "-"),
                ("尺寸", s.gpt_size or "-"),
                ("画质", quality),
                ("线路", "海外" if s.gpt_oversea else "国内"),
                ("背景", {"transparent": "透明", "opaque": "不透明"}.get(
                    s.gpt_background, "自动")),
                ("动作数", str(actions)),
                ("参考图", f"{refs} 张"),
            ]
        rows = [
            ("引擎", "本地 ComfyUI"),
            ("Checkpoint", s.checkpoint or "-"),
            ("尺寸", f"{s.width}x{s.height}"),
            ("步数 / CFG", f"{s.steps} / {s.cfg}"),
            ("种子", f"{s.seed}" + ("（锁定）" if s.lock_seed else "（随机）")),
            ("动作数", str(actions)),
            ("参考图", f"{refs} 张"),
        ]
        if s.pose_on and s.pose_paths:
            rows.append(("骨架驱动", f"开（{len(s.pose_paths)} 图）"))
        if s.i2i_frames >= 2:
            rows.append(("跨帧一致性", f"开（{s.i2i_frames} 帧）"))
        return rows

    def _confirm_generate_summary(self, s, engine: str, refs: int = 0,
                                  actions: int = 1) -> bool:
        """生成前只读摘要确认；返回是否继续提交。勾选「不再提示」后本会话跳过。"""
        if getattr(self, "_skip_gen_confirm", False):
            return True
        from PySide6.QtWidgets import QDialogButtonBox, QGridLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("生成前确认 — PixelForge")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("<b>请确认本次生成参数：</b>"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        for r, (k, v) in enumerate(self._gen_summary_rows(
                s, engine, refs, actions)):
            kl = QLabel(k)
            kl.setStyleSheet("color:#8a8f98;")
            vl = QLabel(str(v))
            vl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(kl, r, 0)
            grid.addWidget(vl, r, 1)
        lay.addLayout(grid)
        chk = QCheckBox("本次会话不再提示")
        lay.addWidget(chk)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确认提交")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        accepted = dlg.exec() == QDialog.Accepted
        if accepted and chk.isChecked():
            self._skip_gen_confirm = True
        return accepted

    def on_generate(self) -> None:
        s = self._collect_settings()
        basic = self.cmb_ui_mode.currentIndex() == 0
        if basic:
            # 笨蛋模式：完整提示词 = 正面 + 负面（Avoid: 拼入）
            s.prompt = self._basic_prompt_full()
            if self.ed_gpt_key.text().strip():
                s.gpt_engine = True
                s.gpt_api_key = self.ed_gpt_key.text().strip()
            if not s.gpt_engine:
                QMessageBox.warning(
                    self, "PixelForge",
                    "笨蛋模式使用云端 GPT-Image 引擎（无需本地 ComfyUI）：\n\n"
                    "请点「展开高级设置」→ 云端 GPT-Image 引擎 →\n"
                    "粘贴 duomiapi.com 的 API Key 后再点生成")
                return
            if not s.prompt:
                QMessageBox.warning(self, "PixelForge",
                                    "请先填写提示词（描述想要的动画）")
                return
            if not self._confirm_generate_summary(
                    s, "gpt", refs=len(self._basic_ref_paths), actions=1):
                self.log("已取消本次生成（参数摘要未确认）")
                return
            self._save_gen_snapshot(s)
            self._start_gpt_generate(s, basic=True)
            return
        if not s.prompt:
            QMessageBox.warning(self, "PixelForge", "请先输入提示词")
            return
        if s.gpt_engine:
            if not s.gpt_api_key:
                QMessageBox.warning(self, "PixelForge",
                                    "请先在高级设置 → 云端引擎 GPT-Image 填写 API Key")
                return
            if not self._confirm_generate_summary(
                    s, "gpt", refs=1 if s.ref_image else 0,
                    actions=self._gpt_action_count()):
                self.log("已取消本次生成（参数摘要未确认）")
                return
            self._start_gpt_generate(s)
            return
        if not s.checkpoint:
            QMessageBox.warning(self, "PixelForge", "请先选择模型 Checkpoint（可点击“从 ComfyUI 获取模型列表”）")
            return
        pose_on = s.pose_on and len(s.pose_paths) > 0
        if s.pose_on and len(s.pose_paths) == 0:
            QMessageBox.warning(self, "PixelForge",
                                "已勾选骨架驱动，但骨架图序列为空——请点「添加骨架图...」")
            return
        if s.pose_on and not s.controlnet_name:
            QMessageBox.warning(self, "PixelForge",
                                "骨架驱动需要选择 ControlNet 模型——点「获取列表」刷新")
            return
        wf_file = self.cmb_workflow.currentData()
        if not wf_file:
            QMessageBox.warning(self, "PixelForge", "没有可用的工作流模板")
            return
        try:
            template = json.loads(Path(wf_file).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "PixelForge", f"工作流模板解析失败: {e}")
            return

        if not self._confirm_generate_summary(
                s, "local", refs=1 if s.ref_image else 0,
                actions=len(self._action_list())):
            self.log("已取消本次生成（参数摘要未确认）")
            return
        s.save()
        self._save_gen_snapshot(s)   # 自动记录本次生成的全部设置（可回载）
        self._busy = True
        self.btn_generate.setEnabled(False)
        self._task_no += 1
        tn = self._task_no
        from pixelforge.config import upsert_queue_record
        from datetime import datetime as _dt
        _now = _dt.now()
        qid = f"{_now:%Y%m%d%H%M%S}-{tn}"
        self._qid_of_tn[tn] = qid
        rec = {
            "id": qid,
            "time": f"{_now:%m-%d %H:%M}",
            "status": "running",
            "engine": "本地",
            "prompt": s.prompt,
            "negative_prompt": s.negative_prompt,
            "model": s.checkpoint,
            "size": f"{s.width}x{s.height}",
            "refs": [s.ref_image] if s.ref_image else [],
            "results": [],
        }
        upsert_queue_record(rec)
        item = self._queue_add_item(rec, tn)
        self.lst_queue.setCurrentItem(item)
        self.log(f"任务 #{tn} 已加入队列（本地 ComfyUI）")

        worker = _GenerateWorker(s, template, self.conda_exe,
                                 pose_paths=s.pose_paths if pose_on else None,
                                 actions=self._action_list())
        worker.sig.progress.connect(lambda m, it=item: self.log(m))
        worker.sig.done.connect(lambda r, it=item: self._on_done(r, it))
        worker.sig.failed.connect(lambda m, it=item: self._on_failed(m, it))
        self.pool.start(worker)

    def _on_done(self, result: dict, item: QListWidgetItem) -> None:
        paths = result.get("paths", [])
        for p in paths:
            self._add_gallery_item(p)
        tid = self._queue_id_of(item)
        if tid:
            self._queue_update(tid, status="done",
                               results=[str(p) for p in paths])
        self._set_queue_done(item.data(Qt.UserRole), "✔")
        self.log(f"已保存 {len(paths)} 张图片: {paths[0] if paths else ''}")
        if not self.settings.lock_seed:
            used = result.get("seed", 0)
            self.spin_seed.setValue(used)
            self.log(f"本次 seed={used}（已填入种子框，需要复现请勾选“锁定种子”）")
        if paths:
            pm = QPixmap(paths[0])
            if not pm.isNull():
                self.viewer.set_image(paths[0])
            if self.chk_auto_post.isChecked():
                worker = _PostProcessWorker(paths[0])
                worker.sig.processed.connect(self._on_processed)
                worker.sig.failed.connect(lambda m: self._on_failed(m))
                self.pool.start(worker)
                self.log("已加入后处理队列")
        self._busy = False
        self.btn_generate.setEnabled(True)

    def _on_processed(self, files: list, atlas: dict) -> None:
        frames = [f for f in files if f.endswith(".png")]
        self.log(f"后处理完成: 裁出 {len(frames)} 帧 -> {atlas.get('name')}.atlas.json")
        self.review.set_frames([Path(f) for f in frames], atlas)

    def _on_failed(self, msg: str, item: QListWidgetItem | None = None) -> None:
        from pixelforge.error_hints import translate_error
        if item is not None:
            tid = self._queue_id_of(item)
            if tid:
                self._queue_update(tid, status="failed", error=str(msg))
            self._set_queue_done(item.data(Qt.UserRole), "✘")
        self.log(f"错误: {translate_error(msg)}")
        self._busy = False
        self.btn_generate.setEnabled(True)
        if item is None:
            QMessageBox.warning(self, "PixelForge", translate_error(msg))

    def _on_frames_to_player(self, frames: list) -> None:
        self.anim_player.set_frames(list(frames))
        self.centralWidget().setCurrentWidget(self.anim_player)
        self.log(f"已发送 {len(frames)} 帧到动画播放器")

    def _on_assets_to_review(self, files: list, atlas: dict) -> None:
        self.review.set_frames([Path(f) for f in files], atlas)

    def _on_assets_to_player(self, frames: list) -> None:
        self.anim_player.set_frames(list(frames))  # 资产库重载不切换标签页

    def _on_asset_to_lab(self, path: Path) -> None:
        self.image_lab.load_path(path)

    def _on_reroll_frame(self, row: int) -> None:
        """审核区单帧重抽：可选前后帧上下文参考 + 强度/提示词对话框。"""
        import numpy as np
        from pixelforge.processing.motion_metrics import (
            coherence_score,
            neighbor_blend,
        )
        if getattr(self, "_reroll_busy", False):
            self.log("已有重抽任务进行中，请稍候")
            return
        s = self._collect_settings()
        if not s.checkpoint:
            QMessageBox.warning(self, "PixelForge", "重抽前请先选择 Checkpoint")
            return
        arr = self.review.frame_arr(row)
        if arr is None:
            return
        prev = self.review.frame_arr(row - 1)
        nxt = self.review.frame_arr(row + 1)
        score_info = ""
        if prev is not None or nxt is not None:
            parts = []
            if prev is not None:
                parts.append(f"与上一帧连贯性 {coherence_score(arr, prev)}")
            if nxt is not None:
                parts.append(f"与下一帧连贯性 {coherence_score(arr, nxt)}")
            score_info = "当前帧上下文：" + "，".join(parts) + \
                "（0~1，越高越连贯；<0.45 通常意味着跳帧或崩帧）"
        dlg = RerollDialog(self, s.prompt, 0.5, score_info=score_info)
        if not dlg.exec():
            return
        prompt = dlg.ed_prompt.toPlainText().strip()
        denoise = dlg.spin_denoise.value()
        use_ctx = dlg.chk_neighbors.isChecked()
        self._reroll_busy = True
        template = self._mw_series_template()
        extra = None
        if use_ctx and (prev is not None or nxt is not None):
            base = neighbor_blend(arr, prev, nxt)
            base = base.copy()
            # 混合产生的半透明 alpha 二值化，避免回贴后新帧边缘半透明
            base[:, :, 3] = np.where(base[:, :, 3] >= 128, 255, 0).astype(np.uint8)
            self.log(f"单帧重抽中：帧 #{row}（前后帧混合底图，"
                     f"denoise={denoise}）")
            if template is not None:
                # Qwen 上下文模板：前/后帧作为 <image2>/<image3> 额外参考
                ctx_tpl = ROOT / "pixelforge" / "workflows" / "qwen_reroll_ctx.json"
                if ctx_tpl.is_file():
                    template = json.loads(ctx_tpl.read_text("utf-8"))
                    extra = {}
                    if prev is not None:
                        extra["__PREV__"] = np.ascontiguousarray(prev)
                    if nxt is not None:
                        extra["__NEXT__"] = np.ascontiguousarray(nxt)
                    prompt = prompt or (
                        "Using <image1> as the frame to refine, <image2> as "
                        "the previous frame and <image3> as the next frame, "
                        "generate the frame between them, same character, "
                        "same style, smooth motion")
        else:
            base = np.ascontiguousarray(arr)
            self.log(f"单帧重抽中：帧 #{row}（仅当前帧底图，denoise={denoise}）")
        worker = _RerollWorker(s, base, prompt=prompt, denoise=denoise,
                               template=template, extra_images=extra,
                               row=row, orig=np.ascontiguousarray(arr))
        worker.sig.done.connect(self._on_reroll_done)
        worker.sig.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _mw_series_template(self) -> dict | None:
        """MainWindow 版模板判定：选中模板含 __IMAGE__ 时用于重抽/系列。"""
        wf_file = self.cmb_workflow.currentData()
        if not wf_file:
            return None
        try:
            tpl = json.loads(Path(wf_file).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if any(v == "__IMAGE__" for n in tpl.values()
               for v in n.get("inputs", {}).values()):
            return tpl
        return None

    def _action_list(self) -> list[str]:
        """动作序列行（至少一行空动作）。"""
        txt = (self.ed_actions.toPlainText() or "").strip()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        return lines or [""]

    def _on_reroll_done(self, result: dict) -> None:
        """重抽完成：替换帧 + 新旧连贯性对比。"""
        from pixelforge.processing.motion_metrics import coherence_score
        self._reroll_busy = False
        arr = result.get("arr")
        row = result.get("row")
        orig = result.get("orig")
        if arr is None or row is None:
            return
        if not self.review.replace_frame(row, arr):
            return
        info = [f"帧 #{row} 已被重抽结果替换"]
        prev = self.review.frame_arr(row - 1)
        nxt = self.review.frame_arr(row + 1)
        scores = []
        if prev is not None:
            scores.append(f"与上帧 {coherence_score(arr, prev)}")
        if nxt is not None:
            scores.append(f"与下帧 {coherence_score(arr, nxt)}")
        if scores:
            info.append("重抽后连贯性：" + "，".join(scores))
            if orig is not None:
                old = [coherence_score(orig, x) for x in (prev, nxt)
                       if x is not None]
                if old:
                    info.append(f"（原帧 {round(sum(old) / len(old), 3)}）")
        self.log("；".join(info))

    def _on_edit_frame(self, row: int) -> None:
        """帧送工具坊编辑：记住行号，编辑后回填原位置。"""
        arr = self.review.frame_arr(row)
        if arr is None:
            return
        self._lab_edit_row = row
        self.image_lab.load_arr(arr, f"frame_{row}")
        self.image_lab.set_edit_target(row)
        self.tabs.setCurrentIndex(1)
        self.log(f"帧 #{row} 已送工具坊（编辑后点「↩ 回填」按钮返回）")

    def _on_frame_back(self, arr) -> None:
        row = getattr(self, "_lab_edit_row", None)
        if row is None:
            return
        if self.review.replace_frame(row, arr):
            self.log(f"工具坊编辑结果已回填到帧 #{row}")
        self.image_lab.set_edit_target(None)
        self._lab_edit_row = None
        self.tabs.setCurrentIndex(0)

    def log(self, msg: str) -> None:
        self.txt_log.appendPlainText(_log_line(msg))
