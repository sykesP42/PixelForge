"""Batch Forge 多角色批量生成引擎：核心逻辑（纯函数，可独立测试）。

任务矩阵 = 角色 × 动作 × 数量；prompt 注入 = 主提示词中的
[动作] / [角色描述] 占位符替换（无占位符时尾部拼接）。
归档：outputs/<project>/<代号>/<代号>_<动作>_<seed>.png
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 内置动作字典：动作名 -> 提示词片段（用户可用 "name: 自由文本" 覆盖）
ACTION_DICT: dict[str, str] = {
    "idle": "standing idle, breathing animation",
    "walk": "walking cycle, side view",
    "run": "running cycle, dynamic motion",
    "attack": "attack pose, weapon slash, action frame",
    "hurt": "hurt pose, stagger backward, pain expression",
    "death": "defeated, falling to ground, final frame",
    "jump": "jumping pose, mid-air, legs tucked",
    "stomp": "violent stomping down, impact frame",
    "cast": "casting spell, glowing hands",
    "sit": "sitting pose, relaxed",
}


@dataclass
class ForgeRole:
    """一个角色：参考图 + 代号 + 描述 + 生成数量。"""
    code: str                      # 代号（输出前缀，如 Hero_A）
    ref_path: str                  # 参考图路径
    desc: str = ""                 # 角色描述（注入 [角色描述]）
    count: int = 1                 # 该角色生成张数（独立计数模式用）
    server_image: str = ""         # 上传后的 ComfyUI input 文件名（运行时填）


@dataclass
class ForgeTask:
    """一个最小生成单元。"""
    role_code: str
    action_name: str               # 动作名（归档用，可为 "custom"）
    prompt: str                    # 注入后的完整提示词
    seed: int
    index: int                     # 该角色的第几张（1 起）
    out_path: Path                 # 归档目标路径
    attempts: int = 0              # 已尝试次数（重试用）
    error: str = ""                # 最近一次失败原因


def resolve_action(line: str) -> tuple[str, str]:
    """解析一行动作：'name' 查字典 / 'name: 自由文本' / 自由文本。

    返回 (动作名, 提示词片段)。
    """
    line = (line or "").strip()
    if not line:
        return "default", ""
    m = re.match(r"^([A-Za-z0-9_\u4e00-\u9fff]+)\s*[:：]\s*(.+)$", line)
    if m:
        return m.group(1), m.group(2).strip()
    if line in ACTION_DICT:
        return line, ACTION_DICT[line]
    return "custom", line


def inject_prompt(master: str, action_text: str, role_desc: str) -> str:
    """把动作/角色描述注入主提示词。

    - 含 [动作] 或 [Action]：替换为动作片段（无则移除占位符）
    - 含 [角色描述] 或 [Character]：替换为角色描述（无则移除占位符）
    - 都不含：尾部拼接 ", action, desc"（空段跳过）
    """
    out = (master or "").strip()
    a_keys = ("[动作]", "[Action]", "[action]")
    c_keys = ("[角色描述]", "[Character]", "[character]")
    replaced = False
    for keys, value in ((a_keys, action_text), (c_keys, role_desc)):
        key = next((k for k in keys if k in out), None)
        if key is not None:
            out = out.replace(key, value or "")
            replaced = True
    if not replaced:
        tail = ", ".join(x for x in (action_text, role_desc) if x)
        if tail:
            out = f"{out}, {tail}" if out else tail
    out = re.sub(r",\s*,+", ",", out)   # 清理连续逗号
    out = re.sub(r"\s+,", ",", out)
    return out.strip().strip(",").strip()


def build_tasks(roles: list[ForgeRole], action_lines: list[str],
                seed_mode: str, seed_base: int,
                master: str, out_root: Path,
                project: str = "batch_forge") -> list[ForgeTask]:
    """生成任务矩阵：角色 × 动作 × 数量。

    seed_mode: fixed（全任务同 seed）/ increment（seed_base 递增）/
               random（逐任务随机）。
    数量：角色 count>1 时为「该角色 × 每动作 × count」；count=1 时
    「每动作一张」。动作行数 = 0 时视为一个空动作（仅角色数量）。
    """
    from pixelforge.comfyui_client import random_seed

    actions = [resolve_action(l) for l in action_lines if l.strip()] or \
        [("default", "")]
    tasks: list[ForgeTask] = []
    counter = 0
    for role in roles:
        n = max(1, int(role.count))
        for action_name, action_text in actions:
            for k in range(1, n + 1):
                if seed_mode == "fixed":
                    seed = int(seed_base)
                elif seed_mode == "increment":
                    seed = int(seed_base) + counter
                else:
                    seed = random_seed()
                prompt = inject_prompt(master, action_text, role.desc)
                out_path = archive_path(Path(out_root), project, role.code,
                                        action_name, seed, k)
                tasks.append(ForgeTask(
                    role_code=role.code, action_name=action_name,
                    prompt=prompt, seed=seed, index=k, out_path=out_path))
                counter += 1
    return tasks


def archive_path(root: Path, project: str, code: str, action_name: str,
                 seed: int, index: int = 1) -> Path:
    """归档路径：root/project/code/code_action_seed.png（重名加序号）。"""
    safe_code = re.sub(r"[^\w\-]", "_", code or "char")
    safe_action = re.sub(r"[^\w\-]", "_", action_name or "default")
    d = Path(root) / (project or "batch_forge") / safe_code
    return d / f"{safe_code}_{safe_action}_{seed}.png"


def sanitize_code(name: str, used: set[str]) -> str:
    """角色代号合法化：仅字母数字下划线、去首尾下划线、去重。"""
    code = re.sub(r"[^\w]", "_", (name or "").strip()).strip("_") or "char"
    base, i = code, 1
    while code in used:
        i += 1
        code = f"{base}_{i}"
    used.add(code)
    return code
