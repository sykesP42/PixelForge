"""Conda 环境管理：动态识别环境、通过 conda run 拉起本地 ComfyUI。

铁律（PRD §8）：不硬编码 Python 路径；优先 `conda run -n <env> python main.py`。
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

_CONDA_CANDIDATES = (
    Path.home() / "anaconda3" / "Scripts" / "conda.exe",
    Path.home() / "miniconda3" / "Scripts" / "conda.exe",
    Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
    Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
)


def find_conda() -> str | None:
    """定位 conda 可执行文件：CONDA_EXE -> PATH -> 常见安装路径。"""
    exe = os.environ.get("CONDA_EXE", "")
    if exe and Path(exe).is_file():
        return exe
    which = shutil.which("conda")
    if which:
        return which
    for c in _CONDA_CANDIDATES:
        if c.is_file():
            return str(c)
    return None


def list_envs(conda_exe: str) -> list[tuple[str, str]]:
    """返回 [(环境名, 环境路径), ...]。环境名取路径末段（base 环境显示其安装目录名）。"""
    out = subprocess.run(
        [conda_exe, "env", "list", "--json"],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"conda env list 失败: {out.stderr.strip() or out.stdout.strip()}")
    envs = json.loads(out.stdout).get("envs", [])
    result: list[tuple[str, str]] = []
    for p in envs:
        path = Path(p)
        name = path.name if path.name.lower() != "python" else path.parent.name
        result.append((name, str(path)))
    result.sort(key=lambda x: (x[0] != "base", x[0].lower()))
    return result


def launch_comfyui(
    conda_exe: str,
    env_name: str,
    comfyui_path: str,
    extra_args: str = "--fast --fp16_accumulation",
) -> subprocess.Popen:
    """后台启动 ComfyUI，日志写入 logs/comfyui_launcher.log。"""
    root = Path(comfyui_path)
    if not (root / "main.py").is_file():
        raise FileNotFoundError(f"未找到 {root / 'main.py'}，请检查 ComfyUI 目录配置")

    cmd = [conda_exe, "run", "--no-capture-output", "-n", env_name,
           "python", "main.py"] + shlex.split(extra_args)

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logf = open(log_dir / "comfyui_launcher.log", "ab")

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        cmd, cwd=str(root), stdout=logf, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
