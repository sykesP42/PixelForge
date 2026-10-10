"""应用配置：JSON 持久化，字段与 GUI 表单一一对应。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.json"
CHARACTERS_FILE = PROJECT_ROOT / "config" / "characters.json"

# 旧版出厂默认负面词（2026-10 起弃用，load 时一次性清除，需要可自行在 UI 填回）
_LEGACY_NEGATIVE = "blurry, low quality, watermark, text"


@dataclass
class Settings:
    # 引擎管理
    conda_env: str = ""                 # 运行 ComfyUI 的 Conda 环境名
    comfyui_path: str = ""              # ComfyUI 源码目录（含 main.py）
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    auto_start_comfyui: bool = True     # 离线时是否通过 conda run 自动拉起
    startup_extra_args: str = "--fast --fp16_accumulation"

    # 云端托管 ComfyUI（同协议托管服务：ComfyICU / RunComfy / liblibAI 等）
    cloud_mode: bool = False            # 启用后直连云端 API，跳过本地 conda 拉起
    cloud_api_base: str = ""            # API 根地址，如 https://api.comfyicu.com
    cloud_api_key: str = ""             # Bearer Token，本地模式留空

    # 生成参数
    checkpoint: str = ""
    output_dir: str = "outputs"
    prompt: str = ""
    negative_prompt: str = ""           # 负面提示词（默认空，按需自填）
    steps: int = 20
    cfg: float = 7.0
    width: int = 512
    height: int = 512
    lock_seed: bool = False
    seed: int = 0
    lora_name: str = ""               # LoRA 文件名（空 = 不使用）
    lora_strength: float = 1.0        # LoRA 强度

    # 跨帧一致性（img2img 系列生成）
    i2i_frames: int = 0               # 系列总帧数（0/1 = 关闭，普通 txt2img）
    i2i_denoise: float = 0.45         # 重绘强度：低=更一致，高=姿态变化更大

    # 骨架驱动（ControlNet OpenPose）
    controlnet_name: str = ""         # ControlNet 模型文件名
    pose_strength: float = 1.0        # 姿势控制强度（0.4~1.3）

    # 参考图锚定模式
    ref_image: str = ""               # 角色参考图路径（空 = txt2img 首帧锚定）
    action_list: str = ""             # 动作序列（每行一个动作，参考图模式行数=帧数）

    # 云端 GPT-Image 引擎（任意 OpenAI 兼容中转站，与本地 ComfyUI 互斥切换）
    gpt_engine: bool = False          # 勾选 = 走云端引擎（忽略本地 ComfyUI）
    gpt_base_url: str = "https://duomiapi.com"   # 中转站地址（OpenAI 兼容）
    gpt_api_key: str = ""             # 中转站控制台的 API Key
    gpt_model: str = "gpt-image-2"    # gpt-image-2 / nano-banana / seedream / Flux / 自定义
    gpt_size: str = "1k"              # 1k / 2k / 4k
    gpt_quality: str = "medium"       # 思考深度：low / medium / high
    gpt_oversea: bool = False         # True = 海外线路（oversea=true）
    gpt_background: str = "auto"      # auto / opaque / transparent

    # UI 模式：0=笨蛋（极简） 1=专注（核心） 2=专业（全量）
    ui_mode: int = 1

    # 审核
    auto_postprocess: bool = True    # 生成完成后自动跑后处理管线

    def output_path(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(CONFIG_FILE.read_text("utf-8"))
            known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
            s = cls(**known)
            if s.negative_prompt == _LEGACY_NEGATIVE:
                s.negative_prompt = ""   # 旧版默认负面词一次性清除
            return s
        except (OSError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), "utf-8"
        )


def load_profiles() -> dict:
    """角色档案：{档案名: {checkpoint/lora/prompt/尺寸/步数/CFG...}}。"""
    try:
        data = json.loads(CHARACTERS_FILE.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_profiles(profiles: dict) -> None:
    CHARACTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHARACTERS_FILE.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), "utf-8")


PROMPT_PRESETS_FILE = CONFIG_FILE.parent / "prompt_presets.json"
GEN_HISTORY_FILE = CONFIG_FILE.parent / "gen_history.json"
GEN_QUEUE_FILE = CONFIG_FILE.parent / "gen_queue.json"


def load_gen_queue() -> list:
    """持久化任务队列（最新在前）：[{id, time, status, engine, prompt,
    negative_prompt, model, size, refs, results, error}]。"""
    try:
        data = json.loads(GEN_QUEUE_FILE.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_gen_queue(records: list, limit: int = 300) -> None:
    GEN_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GEN_QUEUE_FILE.write_text(
        json.dumps(records[:limit], ensure_ascii=False, indent=2), "utf-8")


def upsert_queue_record(rec: dict, limit: int = 300) -> None:
    """按 id 更新/追加队列记录（最新在前）；已存在则字段合并而非整体替换。"""
    recs = load_gen_queue()
    rid = rec.get("id")
    for i, r in enumerate(recs):
        if r.get("id") == rid:
            recs[i].update(rec)
            recs[i]["id"] = rid
            save_gen_queue(recs, limit)
            return
    recs.insert(0, rec)
    save_gen_queue(recs, limit)


def load_prompt_presets() -> dict:
    """提示词预设：{预设名: {prompt, negative_prompt}}。"""
    try:
        data = json.loads(PROMPT_PRESETS_FILE.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_prompt_presets(presets: dict) -> None:
    PROMPT_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PRESETS_FILE.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), "utf-8")


PROMPT_TEMPLATES_FILE = CONFIG_FILE.parent / "prompt_templates.json"


def load_prompt_templates() -> dict:
    """提示词模板库：{模板名: {prompt, negative_prompt, note}}（可分享）。"""
    try:
        data = json.loads(PROMPT_TEMPLATES_FILE.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_prompt_templates(templates: dict) -> None:
    PROMPT_TEMPLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_TEMPLATES_FILE.write_text(
        json.dumps(templates, ensure_ascii=False, indent=2), "utf-8")


def load_gen_history() -> list:
    """生成设置快照（最新在前）：[{name, time, settings...}]。"""
    try:
        data = json.loads(GEN_HISTORY_FILE.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def append_gen_history(snapshot: dict, limit: int = 100) -> None:
    """追加生成快照（最新在前，上限 limit 条）。"""
    GEN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_gen_history()
    history.insert(0, snapshot)
    GEN_HISTORY_FILE.write_text(
        json.dumps(history[:limit], ensure_ascii=False, indent=2), "utf-8")
