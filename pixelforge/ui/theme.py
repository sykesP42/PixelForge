"""工业风深色主题：Fusion 基座 + 深色调色板 + 全局 QSS。

对标专业 DCC/工业软件（Substance / Blender / Houdini）的视觉规范：
- 三层深灰分区：窗口 #1f2025 < 面板 #26272d < 输入 #1a1b20，1px 边框 + 4~6px 圆角
- 单一强调色（工业蓝）：选中 / 焦点 / 主操作按钮；成功 / 危险色仅作状态表达
- 紧凑控件密度：细滚动条、4px 滑块凹槽、15px 复选框、12px 基准字号
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BG_WINDOW = "#1f2025"
BG_PANEL = "#26272d"
BG_INPUT = "#1a1b20"
BG_HOVER = "#2e3038"
BORDER = "#3a3d46"
BORDER_STRONG = "#565a67"
TEXT = "#e3e5ea"
TEXT_DIM = "#9aa0ab"
TEXT_DISABLE = "#6b6e77"
ACCENT = "#4a8cff"
ACCENT_SOLID = "#2f6fdb"
GREEN = "#35d073"
RED = "#ff6b5e"

_QSS = """
* { outline: none; }

QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 12px;
    color: #e3e5ea;
}
QWidget:disabled { color: #6b6e77; }

QMainWindow, QDialog { background: #1f2025; }
QSplitter::handle { background: #3a3d46; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

QToolTip {
    background: #2a2b31; color: #e3e5ea;
    border: 1px solid #565a67; padding: 4px 8px;
}

/* ---------- 分组框 ---------- */
QGroupBox {
    background: #26272d;
    border: 1px solid #3a3d46;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 9px;
    padding: 0 5px;
    color: #9cc3ff;
    background: #1f2025;
}

/* ---------- 标签页 ---------- */
QTabWidget::pane {
    border: 1px solid #3a3d46;
    border-radius: 4px;
    top: -1px;
    background: #1f2025;
}
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent;
    color: #9aa0ab;
    padding: 7px 20px 6px;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #4a8cff; }
QTabBar::tab:hover:!selected { color: #cfd3da; }

/* ---------- 停靠面板 ---------- */
QDockWidget::title {
    background: #26272d;
    border: 1px solid #3a3d46;
    border-bottom: 2px solid #4a8cff;
    padding: 6px 12px;
    font-weight: 600;
}

/* ---------- 按钮 ---------- */
QPushButton {
    background: #2e3038;
    border: 1px solid #454853;
    border-radius: 4px;
    padding: 5px 14px;
    color: #e3e5ea;
    min-height: 17px;
}
QPushButton:hover { background: #383b45; border-color: #565a67; }
QPushButton:pressed { background: #2a2c33; }
QPushButton:disabled { background: #26272d; border-color: #3a3d46; color: #6b6e77; }
QPushButton[accent="true"] {
    background: #2f6fdb; border-color: #2f6fdb; color: #ffffff; font-weight: 600;
}
QPushButton[accent="true"]:hover { background: #3d7ee8; border-color: #3d7ee8; }
QPushButton[accent="true"]:pressed { background: #2861c0; }
QPushButton[accent="true"]:disabled {
    background: #26314a; border-color: #3a3d46; color: #8a93a3; font-weight: 400;
}
QPushButton[danger="true"] { border-color: #6e3a38; color: #ff8a80; background: #2b2426; }
QPushButton[danger="true"]:hover { background: #3a2a2c; border-color: #8a4a48; }

/* ---------- 输入控件 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #1a1b20;
    border: 1px solid #3a3d46;
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: #2f6fdb;
    selection-color: #ffffff;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #4a8cff; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled { color: #6b6e77; }

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #2e3038; border: none; width: 17px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #3d7ee8; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid #9aa0ab;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #9aa0ab;
}

QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    width: 0; height: 0; margin-right: 7px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #9aa0ab;
}
QComboBox QAbstractItemView {
    background: #22232a;
    border: 1px solid #565a67;
    selection-background-color: #2f6fdb;
    selection-color: #ffffff;
    outline: none;
}

/* ---------- 复选 / 单选 ---------- */
QCheckBox, QRadioButton { spacing: 7px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px; height: 15px;
    border: 1px solid #4a4d58; border-radius: 3px;
    background: #1a1b20;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #4a8cff; }
QCheckBox::indicator:checked { background: #2f6fdb; border-color: #2f6fdb; }
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #ffffff, stop:0.42 #ffffff, stop:0.48 #2f6fdb, stop:1 #2f6fdb);
    border-color: #2f6fdb;
}

/* ---------- 滑块 ---------- */
QSlider::groove:horizontal { height: 4px; background: #3a3d46; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2f6fdb; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: #e3e5ea;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider::groove:vertical { width: 4px; background: #3a3d46; border-radius: 2px; }
QSlider::handle:vertical {
    height: 14px; width: 14px; margin: 0 -5px; border-radius: 7px;
    background: #e3e5ea;
}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: #43464f; border-radius: 5px; min-height: 24px; margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #565a67; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal {
    background: #43464f; border-radius: 5px; min-width: 24px; margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: #565a67; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---------- 菜单 ---------- */
QMenuBar { background: #1f2025; border-bottom: 1px solid #3a3d46; }
QMenuBar::item { padding: 5px 11px; background: transparent; border-radius: 3px; }
QMenuBar::item:selected { background: #2e3038; }
QMenu { background: #22232a; border: 1px solid #565a67; padding: 4px; }
QMenu::item { padding: 5px 26px 5px 14px; border-radius: 3px; }
QMenu::item:selected { background: #2f6fdb; color: #ffffff; }
QMenu::item:disabled { color: #6b6e77; }
QMenu::separator { height: 1px; background: #3a3d46; margin: 4px 8px; }

/* ---------- 列表 / 表格 ---------- */
QListWidget, QListView, QTableWidget, QTableView, QTreeView {
    background: #1a1b20;
    border: 1px solid #3a3d46;
    border-radius: 4px;
    alternate-background-color: #22232a;
    outline: none;
}
QListWidget::item, QListView::item { padding: 3px; border-radius: 3px; }
QListWidget::item:hover, QListView::item:hover { background: #2e3038; }
QListWidget::item:selected, QListView::item:selected {
    background: #2f6fdb; color: #ffffff;
}
QTableWidget { gridline-color: #2c2e35; }
QTableWidget::item:selected { background: #2f6fdb; color: #ffffff; }
QHeaderView::section {
    background: #26272d;
    border: none;
    border-bottom: 1px solid #3a3d46;
    border-right: 1px solid #3a3d46;
    padding: 5px 9px;
    color: #9aa0ab;
    font-weight: 600;
}

/* ---------- 状态栏 / 滚动区 ---------- */
QStatusBar { background: #1b1c21; border-top: 1px solid #3a3d46; }
QStatusBar QLabel { padding: 0 6px; }
QScrollArea { border: none; }

/* ---------- 专用对象 ---------- */
QPlainTextEdit#logView {
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    background: #17181c;
}
QListWidget#frameStrip { background: #1d1e23; }

/* ---------- 笨蛋模式卡片 ---------- */
QLabel#dimTip { color: #9aa0a6; font-size: 11px; }
QLabel#sectionLabel { color: #9cc3ff; font-weight: 600; }
QLabel#thumbEmpty { color: #9aa0a6; background: transparent; }
QWidget#thumbStrip { background: #202225; border-radius: 6px; }
QFrame#refThumb {
    background: #2a2b31;
    border: 1px solid #3a3b42;
    border-radius: 6px;
}
QFrame#refThumb:hover { border: 1px solid #2f6fdb; }
QLabel#refThumbImg { background: #202225; border-radius: 4px; }
QLabel#refThumbName {
    color: #c8ccd4; font-size: 10px; background: transparent; border: none;
}
QPushButton#refThumbClose {
    background: rgba(0,0,0,140); color: #e8e8e8;
    border: none; border-radius: 9px; font-weight: bold;
}
QPushButton#refThumbClose:hover { background: #d9534f; }
QLabel#hoverPreview {
    background: #1f2025; border: 1px solid #2f6fdb;
    border-radius: 6px; padding: 2px;
}
QLabel#genPreview {
    background: #2b2d31; color: #9aa0a6; font-size: 14px;
}
"""


def apply_theme(app: QApplication, dark: bool = True) -> None:
    """应用工业主题（在 QApplication 创建后、任何窗口创建前调用）。

    dark=True 深色；False 浅色（同一套 QSS 做色值映射，保证观感一致）。
    """
    from PySide6.QtCore import QSettings
    app.setStyle("Fusion")
    if dark:
        qss = _QSS
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(BG_WINDOW))
        pal.setColor(QPalette.WindowText, QColor(TEXT))
        pal.setColor(QPalette.Base, QColor(BG_INPUT))
        pal.setColor(QPalette.AlternateBase, QColor("#22232a"))
        pal.setColor(QPalette.Text, QColor(TEXT))
        pal.setColor(QPalette.Button, QColor(BG_PANEL))
        pal.setColor(QPalette.ButtonText, QColor(TEXT))
        pal.setColor(QPalette.Highlight, QColor(ACCENT_SOLID))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ToolTipBase, QColor("#2a2b31"))
        pal.setColor(QPalette.ToolTipText, QColor(TEXT))
        pal.setColor(QPalette.Link, QColor(ACCENT))
        pal.setColor(QPalette.PlaceholderText, QColor("#6f7480"))
        pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DISABLE))
        pal.setColor(QPalette.Disabled, QPalette.ButtonText,
                     QColor(TEXT_DISABLE))
        pal.setColor(QPalette.Disabled, QPalette.WindowText,
                     QColor(TEXT_DISABLE))
    else:
        qss = _QSS
        for dk, lt in _LIGHT_MAP.items():
            qss = qss.replace(dk, lt)
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#f5f6f8"))
        pal.setColor(QPalette.WindowText, QColor("#24262b"))
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase, QColor("#f2f4f7"))
        pal.setColor(QPalette.Text, QColor("#24262b"))
        pal.setColor(QPalette.Button, QColor("#ffffff"))
        pal.setColor(QPalette.ButtonText, QColor("#24262b"))
        pal.setColor(QPalette.Highlight, QColor(ACCENT_SOLID))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ToolTipBase, QColor("#eef0f4"))
        pal.setColor(QPalette.ToolTipText, QColor("#24262b"))
        pal.setColor(QPalette.Link, QColor(ACCENT))
        pal.setColor(QPalette.PlaceholderText, QColor("#9aa0ab"))
        pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#a0a4ad"))
        pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#a0a4ad"))
        pal.setColor(QPalette.Disabled, QPalette.WindowText,
                     QColor("#a0a4ad"))
    app.setPalette(pal)
    app.setStyleSheet(qss)
    QSettings("PixelForge", "PixelForge").setValue("ui_theme",
                                                   "dark" if dark else "light")


# 深色值 → 浅色值映射（唯一 7 位 token，安全替换）
_LIGHT_MAP = {
    # 背景三层
    "#1f2025": "#f5f6f8",   # 窗口
    "#26272d": "#ffffff",   # 面板/按钮
    "#1a1b20": "#ffffff",   # 输入/列表
    "#1b1c21": "#eef0f3",   # 状态栏
    "#17181c": "#fafbfc",   # 日志
    "#1d1e23": "#f4f5f7",   # 帧条
    "#22232a": "#ffffff",   # 菜单/弹层/交替行
    "#2a2b31": "#eef0f4",   # tooltip/缩略图底
    "#202225": "#f0f1f4",   # 缩略图条/图片底
    "#3a3b42": "#d0d3da",   # 缩略图卡片边框
    "#c8ccd4": "#3a3d46",   # 缩略图文件名
    "#9aa0a6": "#6b7078",   # 次级提示文本
    "#2b2d31": "#eef0f3",   # 中央预览底
    # 交互层
    "#2e3038": "#e3e6ec",   # hover
    "#383b45": "#d5d9e0",
    "#2a2c33": "#cfd4dc",
    # 边框
    "#3a3d46": "#d0d3da",
    "#454853": "#c2c6cf",
    "#565a67": "#a8adb8",
    "#43464f": "#b8bdc7",
    "#4a4d58": "#b0b5bf",
    "#2c2e35": "#e2e4e9",
    # 文本
    "#e3e5ea": "#24262b",
    "#cfd3da": "#3a3d46",
    "#9aa0ab": "#6b7078",
    "#6b6e77": "#a0a4ad",
    "#6f7480": "#9aa0ab",
    "#8a93a3": "#7d8694",
    "#9cc3ff": "#2f6fdb",   # 分组标题
    # 危险色
    "#2b2426": "#fdeeec",
    "#3a2a2c": "#f8dbd9",
    "#6e3a38": "#d8a09c",
    "#8a4a48": "#d0807c",
    "#ff8a80": "#d84a3e",
    # 强调按钮禁用
    "#26314a": "#dbe6f7",
}
