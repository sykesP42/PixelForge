"""方向/战斗动作提示词模板集：一键填入批量动作队列。

每个模板是 (动作名, 提示词片段) 列表；动作名即输出子目录名，
提示词会拼接在「生成配置」的角色描述之后。方向约定为游戏常用
下/上/左/右四向（left 帧水平翻转即可得到 right 的镜像）。
"""
from __future__ import annotations

MOTION_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "待机 · 4 方向": [
        ("idle_down", "idle animation, standing, breathing, facing front, front view"),
        ("idle_up", "idle animation, standing, breathing, back view"),
        ("idle_left", "idle animation, standing, breathing, side view facing left"),
        ("idle_right", "idle animation, standing, breathing, side view facing right"),
    ],
    "行走 · 4 方向": [
        ("walk_down", "walk cycle, front view, facing camera"),
        ("walk_up", "walk cycle, back view"),
        ("walk_left", "walk cycle, side view facing left"),
        ("walk_right", "walk cycle, side view facing right"),
    ],
    "跑步 · 4 方向": [
        ("run_down", "run cycle, front view, dynamic motion"),
        ("run_up", "run cycle, back view, dynamic motion"),
        ("run_left", "run cycle, side view facing left, dynamic motion"),
        ("run_right", "run cycle, side view facing right, dynamic motion"),
    ],
    "战斗三连": [
        ("attack", "attack pose, weapon slash, action frame"),
        ("hurt", "hurt pose, stagger backward, pain expression"),
        ("death", "defeated, falling to ground, final frame"),
    ],
    "跳跃 · 4 帧": [
        ("jump_crouch", "crouching, preparing to jump"),
        ("jump_up", "jumping, rising, legs tucked"),
        ("jump_fall", "falling, arms raised"),
        ("jump_land", "landing, knees bent, dust"),
    ],
}


def template_names() -> list[str]:
    """全部模板名（保持定义顺序）。"""
    return list(MOTION_TEMPLATES.keys())


def get_template(name: str) -> list[tuple[str, str]]:
    """按名取模板；未知名称抛 KeyError。"""
    return MOTION_TEMPLATES[name]
