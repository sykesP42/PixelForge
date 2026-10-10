"""负面提示词默认值测试：默认为空 + 旧版默认值一次性迁移清除。

用法：QT_QPA_PLATFORM=offscreen python tests/test_negative_default.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.config import Settings
from pixelforge.ui.main_window import _compose_prompt


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pf_neg_"))
    old = cfgmod.CONFIG_FILE
    cfgmod.CONFIG_FILE = tmp / "settings.json"
    try:
        # 1. 新默认：负面提示词为空
        s = Settings()
        assert s.negative_prompt == "", f"默认应为空: {s.negative_prompt!r}"
        print("[OK] 1. Settings 默认 negative_prompt = ''（不再出厂自带）")

        # 2. 迁移：settings.json 里存着旧版默认值 → load 后清空
        cfgmod.CONFIG_FILE.write_text(
            json.dumps({"negative_prompt":
                        "blurry, low quality, watermark, text"}), "utf-8")
        s2 = Settings.load()
        assert s2.negative_prompt == "", \
            f"旧版默认值应被迁移清除: {s2.negative_prompt!r}"
        print("[OK] 2. 旧版默认负面词 load 时一次性清除")

        # 3. 用户自填的负面词保留不动
        cfgmod.CONFIG_FILE.write_text(
            json.dumps({"negative_prompt": "extra fingers"}), "utf-8")
        s3 = Settings.load()
        assert s3.negative_prompt == "extra fingers", s3.negative_prompt
        print("[OK] 3. 用户自填负面词不受迁移影响")

        # 4. 空负面 → 提示词不拼 Avoid: 前缀
        p = _compose_prompt("hero idle", "")
        assert "Avoid:" not in p, p
        assert p == "hero idle"
        # 有负面 → 仍正常拼接
        p2 = _compose_prompt("hero idle", "extra fingers")
        assert p2 == "hero idle\nAvoid: extra fingers", p2
        print("[OK] 4. 空负面词不拼 Avoid:，自填时正常拼接")
    finally:
        cfgmod.CONFIG_FILE = old

    print("\n负面提示词默认值测试全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
