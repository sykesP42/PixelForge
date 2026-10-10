"""失败自动重试测试（离屏）：
1. is_retryable_error 分类（网络类可重试 / 确定性失败不重试）
2. GPT worker 逐动作自动重试：失败 2 次第 3 次成功
3. 不可重试错误立即失败、不烧重试次数

用法：QT_QPA_PLATFORM=offscreen python tests/test_auto_retry.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.gpt_engine as gpt_mod
from pixelforge.error_hints import is_retryable_error
from pixelforge.gpt_engine import CloudEngineError


def main() -> int:
    app = QApplication(sys.argv)

    # ---------- 1. 错误分类 ----------
    retryable = [
        "网络错误: connection refused",
        "同步请求重试均失败: URLError",
        "图片下载全部通道失败:\n  timeout",
        "HTTP 503: service unavailable",
        "HTTP 429: too many requests",
        "图床上传失败（3 次重试）: eof",
        "提交失败: 上游繁忙，系统繁忙请稍后再试",
    ]
    non_retryable = [
        "⏹ 任务已停止",
        "提示词为空",
        "模型名不能为空",
        "HTTP 401: invalid api key",
        "HTTP 400: bad request",
        "云端任务超时（900s）",
        "内容审核未通过：敏感内容",
        "色键后内容占比 0.02，疑似生成崩坏",
        "CUDA out of memory",
        "ckpt_name not in list",
    ]
    for m in retryable:
        assert is_retryable_error(m), f"应可重试: {m}"
    for m in non_retryable:
        assert not is_retryable_error(m), f"不应重试: {m}"
    assert not is_retryable_error(""), "空消息不可重试"
    print("[OK] 1. is_retryable_error 分类：网络类放行 / 确定性失败拦截")

    # ---------- 2. GPT worker 自动重试（失败 2 次 → 第 3 次成功） ----------
    from pixelforge.ui.main_window import _GPTEngineWorker
    tmp = Path(tempfile.mkdtemp(prefix="pf_retry_"))
    grid_path = tmp / "grid.png"
    grid_path.write_bytes(b"png")

    calls = {"n": 0}

    def fake_generate_grid(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise CloudEngineError("网络错误: connection refused")
        return grid_path

    orig = gpt_mod.generate_grid
    orig_sleep = __import__("time").sleep
    __import__("time").sleep = lambda s: None   # 退避不等
    gpt_mod.generate_grid = fake_generate_grid
    try:
        s = types.SimpleNamespace(
            gpt_api_key="k", gpt_base_url="https://x", prompt="hero idle",
            negative_prompt="", gpt_model="gpt-image-2", gpt_size="1k",
            gpt_quality="medium", gpt_oversea=False, gpt_background="auto",
            output_path=lambda: tmp)
        w = _GPTEngineWorker(s, ["walking cycle"], [])
        log, done, failed = [], [], []
        w.sig.progress.connect(log.append)
        w.sig.done.connect(done.append)
        w.sig.failed.connect(failed.append)
        w.run()   # 同步执行
        assert not failed, f"重试后应成功: {failed}"
        assert done and Path(done[0][0]) == grid_path
        assert calls["n"] == 3, f"应调用 3 次（2 失败+1 成功）: {calls['n']}"
        assert any("自动重试" in m for m in log), \
            f"日志应含重试提示: {[m for m in log if '重试' in m]}"
        print("[OK] 2. GPT worker 网络类失败自动重试 2 次后成功 + 日志可见")
    finally:
        gpt_mod.generate_grid = orig
        __import__("time").sleep = orig_sleep

    # ---------- 3. 不可重试错误立即失败 ----------
    calls["n"] = 0

    def fake_auth_fail(*a, **k):
        calls["n"] += 1
        raise CloudEngineError("HTTP 401: invalid api key")

    gpt_mod.generate_grid = fake_auth_fail
    try:
        s2 = types.SimpleNamespace(
            gpt_api_key="k", gpt_base_url="https://x", prompt="hero idle",
            negative_prompt="", gpt_model="gpt-image-2", gpt_size="1k",
            gpt_quality="medium", gpt_oversea=False, gpt_background="auto",
            output_path=lambda: tmp)
        w2 = _GPTEngineWorker(s2, ["idle"], [])
        failed2 = []
        w2.sig.failed.connect(failed2.append)
        w2.run()
        assert failed2 and "401" in failed2[0], f"应立即失败: {failed2}"
        assert calls["n"] == 1, f"鉴权错误不应重试: {calls['n']}"
        print("[OK] 3. 不可重试错误（401）立即失败，不烧重试次数")
    finally:
        gpt_mod.generate_grid = orig

    print("\n失败自动重试全部测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
