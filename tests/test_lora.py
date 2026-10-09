"""LoRA 注入链路单元测试：build_workflow 注入 LoraLoader 并正确改线。

用法：python tests/test_lora.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelforge.comfyui_client import build_workflow

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "pixelforge" / "workflows" / "txt2img_basic.json"


def main() -> int:
    template = json.loads(TPL.read_text("utf-8"))

    # 1. 不用 LoRA：无注入，连线保持原样
    wf = build_workflow(
        template, checkpoint="a.safetensors", prompt="p", negative_prompt="n",
        seed=1, width=64, height=64, steps=10, cfg=7.0,
        lora_name="", lora_strength=1.0)
    assert "90" not in wf, "空 lora_name 不应注入 LoraLoader"
    assert wf["5"]["inputs"]["model"] == ["1", 0], "KSampler 应直连 checkpoint"
    assert wf["2"]["inputs"]["clip"] == ["1", 1], "正向编码应直连 checkpoint CLIP"
    print("[OK] 无 LoRA：连线保持 checkpoint 直连")

    # 2. 用 LoRA：注入 LoraLoader(node 90)，KSampler/CLIP 改线到 LoRA 输出
    wf = build_workflow(
        template, checkpoint="a.safetensors", prompt="p", negative_prompt="n",
        seed=1, width=64, height=64, steps=10, cfg=7.0,
        lora_name="myChar.safetensors", lora_strength=0.8)
    lora = wf.get("90")
    assert lora and lora["class_type"] == "LoraLoader", "应注入 LoraLoader"
    ins = lora["inputs"]
    assert ins["lora_name"] == "myChar.safetensors"
    assert ins["strength_model"] == 0.8 and ins["strength_clip"] == 0.8
    assert ins["model"] == ["1", 0] and ins["clip"] == ["1", 1], \
        "LoraLoader 应从 checkpoint 取 model+clip"
    assert wf["5"]["inputs"]["model"] == ["90", 0], "KSampler 应改接 LoRA 模型输出"
    assert wf["2"]["inputs"]["clip"] == ["90", 1], "正向编码应改接 LoRA CLIP 输出"
    assert wf["3"]["inputs"]["clip"] == ["90", 1], "负向编码应改接 LoRA CLIP 输出"
    # VAE 不受 LoRA 影响
    assert wf["6"]["inputs"]["vae"] == ["1", 2], "VAE 应保持 checkpoint 直连"
    print("[OK] LoRA 注入: LoraLoader(90) + KSampler/CLIP 改线 + VAE 不变")

    # 3. 多次调用幂等（deep copy，不污染模板）
    wf_b = build_workflow(
        template, checkpoint="b.safetensors", prompt="p2", negative_prompt="n2",
        seed=2, width=64, height=64, steps=10, cfg=7.0,
        lora_name="other.safetensors", lora_strength=1.2)
    assert wf_b["90"]["inputs"]["lora_name"] == "other.safetensors"
    assert wf_b["5"]["inputs"]["model"] == ["90", 0]
    # 模板本身不应被污染
    tpl2 = json.loads(TPL.read_text("utf-8"))
    assert "90" not in tpl2, "模板文件不应被注入污染"
    assert tpl2["5"]["inputs"]["model"] == ["1", 0]
    print("[OK] 幂等: 模板文件未被污染，多次构建互不影响")

    print("\nLoRA 注入全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
