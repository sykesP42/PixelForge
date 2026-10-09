# -*- coding: utf-8 -*-
"""启动 GUI 并预填待耳测试参数：Kanami idle 8帧，图生图模式。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from pixelforge.ui.main_window import MainWindow

REFS = [
    r"C:\Users\24311\.workbuddy\clipboard-images\clipboard-2026-10-09T19-06-09-473Z-fd891556.jpg",
    r"C:\Users\24311\.workbuddy\clipboard-images\clipboard-2026-10-09T19-06-09-475Z-38361577.jpg",
]
PROMPT = """Pixel art sprite sheet of the character in the reference image, idle looping animation, 2 rows x 4 columns, 8 frames, arranged left-to-right, top-to-bottom.
Each frame has the exact same canvas size. The character is horizontally centered in every cell, and the feet align to one shared baseline.
Frames: complete 8-frame idle loop, subtle motion, natural standing pose.
Frame 1: base idle stance, start of breathing loop.
Frame 2: slight inhale, chest and shoulders lift a little.
Frame 3: top of breath, tiny head movement, arms subtly sway.
Frame 4: begin exhale, shoulders lower slightly.
Frame 5: mid exhale, center of mass sinks slightly.
Frame 6: slight weight shift, one shoulder a bit lower.
Frame 7: return close to base pose, optional blink or subtle eye movement.
Frame 8: seamless transition back to frame 1.
Style: 16-bit pixel art, crisp hard-edged pixels, no anti-aliasing, no blur, no gradients. Limited palette under 24 colors. Dark outline around the character. Bright saturated retro game colors, flat cel shading.
View: same view and facing direction as the reference image.
Keep the character's proportions, hairstyle, outfit, color scheme and facial features identical to the reference image.
Background: solid flat background, recommended #00FF00 green screen or pure white. No ground, no shadow, no grid lines, no text, no numbers, no watermark, no UI, no extra props, no extra characters.
--ar 2:1 --style raw --niji 6"""

app = QApplication(sys.argv)
w = MainWindow()
w.show()
# 预填：笨蛋模式 + 云端引擎 + 提示词 + 参考图
# API Key 不硬编码：从环境变量 PF_GPT_KEY 读取，或启动后在设置里填写
w.cmb_ui_mode.setCurrentIndex(0)
w.ed_prompt_basic.setPlainText(PROMPT)
_key = os.environ.get("PF_GPT_KEY", "")
if _key:
    w.ed_gpt_key.setText(_key)
    w.chk_gpt.setChecked(True)
w._basic_ref_paths = [p for p in REFS if Path(p).is_file()]
w._refresh_basic_thumbs()
w.log("[GUI 测试] 已预填 Kanami idle 测试：2 张参考图 + 提示词 + API Key")
w.log("[GUI 测试] 请点击「生成」按钮开始（云端 GPT-Image）")
w.resize(1500, 950)
sys.exit(app.exec())
