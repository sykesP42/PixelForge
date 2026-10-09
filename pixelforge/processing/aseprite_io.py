"""Aseprite JSON 互导（横向精灵表 + 官方 JSON 格式）。

export_aseprite: 帧列表 -> 横向拼接 PNG + Aseprite JSON（hash 格式）
import_aseprite: Aseprite JSON（hash / array 均兼容）-> 帧列表 + fps

JSON 结构遵循 Aseprite 官方导出格式：
    {"frames": {"<名> <i>": {"frame": {x,y,w,h}, "duration": ms, ...}},
     "meta": {"app", "image", "size", "frameTags", ...}}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def export_aseprite(frames: list[np.ndarray], out_base, fps: int = 8,
                    name: str | None = None) -> list[Path]:
    """把 BGRA 帧列表导出为横向精灵表 + Aseprite JSON，返回生成的文件。

    out_base: 输出基路径（无后缀），生成 <base>.png 与 <base>.json。
    帧按底部对齐拼入横向表（角色脚底基线一致）；每帧 rect 精确覆盖
    自己的画布（若已做 bleed，外渗像素位于画布内部，不会串色）。
    """
    out_base = Path(out_base)
    if not frames:
        raise ValueError("没有可导出的帧")
    fps = max(1, int(fps))
    height = max(f.shape[0] for f in frames)
    total_w = sum(int(f.shape[1]) for f in frames)
    sheet = np.zeros((height, total_w, 4), np.uint8)
    entries = []
    x = 0
    for f in frames:
        h, w = f.shape[:2]
        y = height - h  # 底部对齐
        sheet[y:y + h, x:x + w] = f
        entries.append({
            "frame": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": int(w), "h": int(h)},
            "sourceSize": {"w": int(w), "h": int(h)},
            "duration": int(round(1000.0 / fps)),
        })
        x += int(w)
    label = name or out_base.name
    data = {
        "frames": {f"{label} {i}": e for i, e in enumerate(entries)},
        "meta": {
            "app": "PixelForge Studio",
            "version": "1.0",
            "image": out_base.name + ".png",
            "format": "RGBA8888",
            "size": {"w": int(total_w), "h": int(height)},
            "scale": "1",
            "frameTags": [{"name": "default", "from": 0,
                            "to": len(entries) - 1, "direction": "forward"}],
        },
    }
    png_path = out_base.with_suffix(".png")
    json_path = out_base.with_suffix(".json")
    cv2.imwrite(str(png_path), sheet)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return [png_path, json_path]


def import_aseprite(json_path) -> tuple[list[np.ndarray], int]:
    """读取 Aseprite JSON，切出帧列表（BGRA），返回 (frames, fps)。

    兼容 hash 格式（frames 为对象，按文件内顺序读取）与
    array 格式（frames 为数组）；duration 取平均换算 fps。
    """
    jp = Path(json_path)
    data = json.loads(jp.read_text("utf-8"))
    raw = data.get("frames", {})
    entries = list(raw.values()) if isinstance(raw, dict) else list(raw)
    if not entries:
        raise ValueError(f"Aseprite JSON 中没有帧: {jp}")
    sheet_rel = data.get("meta", {}).get("image",
                                         jp.with_suffix(".png").name)
    sheet_path = jp.parent / sheet_rel
    sheet = cv2.imread(str(sheet_path), cv2.IMREAD_UNCHANGED)
    if sheet is None:
        raise FileNotFoundError(f"无法读取精灵表: {sheet_path}")
    if sheet.ndim == 2:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGRA)
    elif sheet.shape[2] == 3:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)

    frames: list[np.ndarray] = []
    durations: list[float] = []
    for e in entries:
        r = e["frame"] if "frame" in e else e  # 兼容缺 frame 包装的简化格式
        x, y = int(r["x"]), int(r["y"])
        w, h = int(r["w"]), int(r["h"])
        frames.append(sheet[y:y + h, x:x + w].copy())
        if e.get("duration"):
            durations.append(float(e["duration"]))
    fps = int(round(1000.0 / (sum(durations) / len(durations)))) \
        if durations else 8
    return frames, max(1, fps)


def frames_by_tag(json_path) -> dict[str, list[np.ndarray]]:
    """按 Aseprite frameTags 把精灵表切分为多个动画组。

    返回 {"标签名": [BGRA 帧...]}；JSON 无 frameTags 时返回
    {"default": 全部帧}。hash / array 格式均兼容。
    """
    jp = Path(json_path)
    data = json.loads(jp.read_text("utf-8"))
    raw = data.get("frames", {})
    entries = list(raw.values()) if isinstance(raw, dict) else list(raw)
    if not entries:
        raise ValueError(f"Aseprite JSON 中没有帧: {jp}")
    sheet_rel = data.get("meta", {}).get("image",
                                         jp.with_suffix(".png").name)
    sheet_path = jp.parent / sheet_rel
    sheet = cv2.imread(str(sheet_path), cv2.IMREAD_UNCHANGED)
    if sheet is None:
        raise FileNotFoundError(f"无法读取精灵表: {sheet_path}")
    if sheet.ndim == 2:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGRA)
    elif sheet.shape[2] == 3:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)

    frames: list[np.ndarray] = []
    for e in entries:
        r = e["frame"] if "frame" in e else e
        x, y = int(r["x"]), int(r["y"])
        w, h = int(r["w"]), int(r["h"])
        frames.append(sheet[y:y + h, x:x + w].copy())

    out: dict[str, list[np.ndarray]] = {}
    for t in data.get("meta", {}).get("frameTags") or []:
        a = max(0, int(t.get("from", 0)))
        b = min(len(frames) - 1, int(t.get("to", len(frames) - 1)))
        if a <= b:
            out[str(t.get("name", "clip"))] = frames[a:b + 1]
    if not out:
        out["default"] = frames
    return out
