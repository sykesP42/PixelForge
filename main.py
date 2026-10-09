"""PixelForge Studio 入口。

用法：python main.py
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from pixelforge import __version__
from pixelforge.ui.main_window import MainWindow
from pixelforge.ui.theme import apply_theme


def main() -> int:
    # 文件对话框也走 Qt 绘制，跟随主题（须在 QApplication 创建前设置）
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    app = QApplication(sys.argv)
    app.setApplicationName("PixelForge Studio")
    app.setApplicationVersion(__version__)
    dark = QSettings("PixelForge", "PixelForge").value(
        "ui_theme", "dark") != "light"
    apply_theme(app, dark=dark)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
