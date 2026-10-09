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
