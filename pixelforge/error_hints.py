"""ComfyUI 报错翻译：原始错误信息 -> 人话 + 建议。

原则：不给用户看天书。匹配到已知模式时在原始信息后附加「建议」；
未命中时原样返回（不编造原因）。
"""
from __future__ import annotations

_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("out of memory", "cuda error", "memory"),
     "显存/内存不足：降低生成分辨率（宽/高）或步数，关闭其他占用显存的程序，"
     "或在 ComfyUI 启动参数中加 --lowvram"),
    (("ckpt_name", "checkpoint"),
     "Checkpoint 模型未找到：确认模型文件在 ComfyUI/models/checkpoints/ 下，"
     "然后点「从 ComfyUI 获取模型列表」刷新"),
    (("lora",),
     "LoRA 文件未找到：确认文件在 ComfyUI/models/loras/ 下且名称拼写一致"),
    (("control_net", "controlnet"),
     "ControlNet 模型未找到：确认模型在 ComfyUI/models/controlnet/ 下，"
     "并刷新 ControlNet 列表"),
    (("value not in list",),
     "参数不在可选列表中：通常是模型/节点文件名变更，请刷新对应列表后重试"),
    (("failed to validate", "return type", "required input is missing"),
     "工作流校验失败：模板节点与当前 ComfyUI 版本不匹配，"
     "或占位符未被正确替换——检查工作流模板"),
    (("繁忙", "请稍后重试"),
     "中转上游容量不足：换个模型别名、画质降为 medium、或换个时段重试"
     "（应用对繁忙类错误会自动退避重试 4 次）"),
    (("connection", "urlopen error", "timed out", "10061", "connection refused"),
     "无法连接 ComfyUI：确认服务已启动、地址端口正确；"
     "云端模式请检查网络与 API 密钥"),
    (("upload", "image"),
     "图片上传失败：确认参考图/骨架图可读且 ComfyUI 服务正常"),
]


def translate_error(msg: str) -> str:
    """把 ComfyUI 原始报错翻译为「原始信息 + 建议」；未命中则原样返回。"""
    low = (msg or "").lower()
    for keys, hint in _HINTS:
        if any(k in low for k in keys):
            return f"{msg}\n\n建议：{hint}"
    return msg


# ---- 网络类错误自动重试分类 ----

# ---- 网络类错误自动重试分类 ----

_RETRYABLE_KEYS = (
    "网络错误", "timed out", "timeout", "同步请求重试均失败",
    "图片下载全部通道失败", "图床上传失败",
    "connection", "urlopen error", "10061", "临时维护",
    "繁忙", "请稍后重试", "try again later", "server is busy",
    "upstream", "http 429", "http 500", "http 502", "http 503",
    "http 504", "http 529", "远程主机", "connectionreset",
)
_NON_RETRYABLE_KEYS = (
    "已停止", "提示词为空", "模型名不能为空", "api key", "鉴权",
    "审核", "敏感", "崩坏", "拆帧不足", "云端任务超时",
    "http 400", "http 401", "http 403", "http 404",
    "value not in list", "ckpt_name", "checkpoint", "out of memory",
    "cuda", "lora", "control_net", "failed to validate",
)


def is_retryable_error(msg: str) -> bool:
    """判断是否为可自动重试的网络类错误（重试有意义且不会白花钱）。

    规则：先排除「确定性失败」（鉴权/参数/审核/已停止/OOM 等），
    再命中「瞬时性失败」（连接/超时/5xx/上游繁忙）。
    """
    low = (msg or "").lower()
    if any(k in low for k in _NON_RETRYABLE_KEYS):
        return False
    return any(k in low for k in _RETRYABLE_KEYS)


_BUSY_KEYS = (
    "繁忙", "请稍后重试", "server is busy", "try again later",
    "http 429", "http 529", "系统繁忙",
)


def is_busy_error(msg: str) -> bool:
    """上游繁忙类错误：窗口通常持续数分钟，退避应比普通网络抖动更久。"""
    low = (msg or "").lower()
    return any(k in low for k in _BUSY_KEYS)
