"""PyInstaller 打包脚本：产出 dist/PixelForgeStudio/ 单目录可执行程序。

用法：python scripts/build_exe.py
说明：
- 单目录模式（启动快）；--windowed 无控制台
- workflows 模板随包分发（运行时按 __file__ 相对定位）
- 图标可后续加 --icon assets/app.ico
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    workflows = ROOT / "pixelforge" / "workflows"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--windowed",
        "--name", "PixelForgeStudio",
        "--add-data", f"{workflows};pixelforge/workflows",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        str(ROOT / "main.py"),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
