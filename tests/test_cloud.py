"""云端托管 ComfyUI 支持：from_settings 路由 + 全请求鉴权头 + 配置兼容。

用法：python tests/test_cloud.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pixelforge.config as cfgmod
from pixelforge.comfyui_client import ComfyUIClient
from pixelforge.config import Settings

NONCE = "CLD7"


class _Handler(BaseHTTPRequestHandler):
    seen: dict = {}

    def _capture(self) -> None:
        _Handler.seen["auth"] = self.headers.get("Authorization")

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._capture()
        self._reply({"devices": []})

    def do_POST(self) -> None:  # noqa: N802
        self._capture()
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._reply({"prompt_id": f"pid_{NONCE}"})

    def log_message(self, *args) -> None:  # 静默
        pass


def main() -> int:
    # 1. from_settings 路由：云端模式 -> api_base + https->wss + 尾斜杠去除
    s = Settings(cloud_mode=True, cloud_api_base=f"https://api.example.com/{NONCE}/",
                 cloud_api_key=f"k_{NONCE}")
    c = ComfyUIClient.from_settings(s)
    assert c.http_base == f"https://api.example.com/{NONCE}", "api_base 应去尾斜杠"
    assert c.ws_base == f"wss://api.example.com/{NONCE}/ws", "https 应映射 wss 且保留路径"
    assert c.api_key == f"k_{NONCE}"
    print("[OK] 云端路由: from_settings -> api_base 去尾斜杠 + https->wss + 路径前缀")

    # 2. 回落与 http 前缀：云端未填地址回落本地；http -> ws
    c2 = ComfyUIClient.from_settings(Settings(cloud_mode=True))
    assert c2.http_base == "http://127.0.0.1:8188", "无 api_base 应回落本地 host:port"
    assert c2.ws_base == "ws://127.0.0.1:8188/ws"
    assert c2.api_key == "", "回落本地不应携带云端密钥"
    c3 = ComfyUIClient(api_base=f"http://relay.local:9000/comfy", api_key="x")
    assert c3.http_base == "http://relay.local:9000/comfy"
    assert c3.ws_base == "ws://relay.local:9000/comfy/ws"
    print("[OK] 回落与前缀: 云端无地址回落本地, http->ws 保留路径前缀")

    # 3+4. 鉴权头覆盖 GET/POST（真实本地服务器抓包）
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ck = ComfyUIClient(port=port, api_key=f"tok_{NONCE}")
        assert ck.is_online(), "带密钥 GET 应成功"
        assert _Handler.seen.get("auth") == f"Bearer tok_{NONCE}", "GET 应带 Bearer 头"
        pid = ck.queue_prompt({"1": {"class_type": "X", "inputs": {}}})
        assert pid == f"pid_{NONCE}", "POST 应返回 prompt_id"
        assert _Handler.seen.get("auth") == f"Bearer tok_{NONCE}", "POST 应带 Bearer 头"

        ck0 = ComfyUIClient(port=port)
        assert ck0.is_online()
        assert _Handler.seen.get("auth") is None, "无密钥不应带 Authorization 头"
        print("[OK] 鉴权头: GET/POST 携带 Bearer, 无密钥时不携带")
    finally:
        srv.shutdown()

    # 5. ws 握手头列表（纯内存检查，不建立连接）
    assert ck._auth_header_list() == [f"Authorization: Bearer tok_{NONCE}"]
    assert ck0._auth_header_list() == []
    assert ComfyUIClient(api_key="  spaced  ").api_key == "spaced", "密钥应去空白"
    print("[OK] WebSocket 握手头: header 列表按密钥有无生成, 密钥去空白")

    # 6. 配置兼容（临时 CONFIG_FILE，不触碰用户真实配置）
    tmp_dir = Path(tempfile.mkdtemp(prefix="pf_cloud_test_"))
    cfgmod.CONFIG_FILE = tmp_dir / "settings.json"
    sA = Settings(cloud_mode=True, cloud_api_base="https://a.b", cloud_api_key="k2")
    sA.save()
    sB = Settings.load()
    assert sB.cloud_mode is True and sB.cloud_api_base == "https://a.b"
    assert sB.cloud_api_key == "k2", "云端三字段应完整往返"
    (tmp_dir / "settings.json").write_text(
        json.dumps({"comfyui_host": "192.168.1.5", "legacy_unknown": 1}), "utf-8")
    sC = Settings.load()
    assert sC.comfyui_host == "192.168.1.5", "旧 JSON 普通字段应正常加载"
    assert sC.cloud_mode is False and sC.cloud_api_key == "", "旧 JSON 缺失字段取默认值"
    print("[OK] 配置兼容: 新字段 save/load 往返, 旧 JSON+未知字段安全加载")

    print(f"\n云端托管支持全部测试通过 (nonce {NONCE}) ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
