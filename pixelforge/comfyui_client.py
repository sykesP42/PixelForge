"""ComfyUI API 客户端（本地 HTTP + WebSocket，兼容云端托管同协议服务）。

通信规范（PRD §8）：
- 提交任务: POST /prompt
- 进度监听: WebSocket /ws?clientId=xxx，等待 execution 完成（非轮询）
- 取图:     GET /view?filename=&subfolder=&type=output

云端托管（ComfyICU / RunComfy / liblibAI 等）：传入 api_base + api_key，
所有 HTTP 请求自动附带 Authorization: Bearer 头，ws 握手同样携带。
"""
from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import websocket  # websocket-client


class ComfyUIError(RuntimeError):
    """ComfyUI 执行/协议错误。"""


class ComfyUIOffline(ComfyUIError):
    """ComfyUI 服务不可达。"""


class ComfyUIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188,
                 api_base: str = "", api_key: str = ""):
        self.host = host
        self.port = port
        self.api_key = (api_key or "").strip()
        base = (api_base or "").strip().rstrip("/")
        if base:  # 云端托管：整段替换根地址（可含路径前缀）
            self.http_base = base
            scheme, _, rest = base.partition("://")
            self.ws_base = f"{'wss' if scheme == 'https' else 'ws'}://{rest}/ws"
        else:
            self.http_base = f"http://{host}:{port}"
            self.ws_base = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    @classmethod
    def from_settings(cls, s) -> "ComfyUIClient":
        """按设置路由：启用云端托管且填了 API 地址 -> 云端；否则本地 host:port。"""
        if getattr(s, "cloud_mode", False) and getattr(s, "cloud_api_base", "").strip():
            return cls(api_base=s.cloud_api_base, api_key=getattr(s, "cloud_api_key", ""))
        return cls(host=s.comfyui_host, port=s.comfyui_port)

    # ---------- 基础请求 ----------

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _auth_header_list(self) -> list:
        return [f"Authorization: Bearer {self.api_key}"] if self.api_key else []

    def _get(self, path: str, timeout: float = 10) -> bytes:
        req = urllib.request.Request(self.http_base + path, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def _get_json(self, path: str, timeout: float = 10) -> dict:
        return json.loads(self._get(path, timeout).decode("utf-8"))

    def _post_json(self, path: str, payload: dict, timeout: float = 15) -> dict:
        req = urllib.request.Request(
            self.http_base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    # ---------- 状态与元信息 ----------

    def is_online(self) -> bool:
        try:
            self._get("/system_stats", timeout=2.5)
            return True
        except Exception:
            return False

    def system_stats(self) -> dict:
        return self._get_json("/system_stats", timeout=3)

    def get_checkpoints(self) -> list[str]:
        """从 /object_info 拉取可用模型列表，供 GUI 下拉框使用。"""
        try:
            data = self._get_json("/object_info/CheckpointLoaderSimple", timeout=10)
            return list(
                data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            )
        except Exception:
            return []

    def get_controlnets(self) -> list[str]:
        """从 /object_info 拉取 ControlNet 模型列表，供骨架驱动生成使用。"""
        try:
            data = self._get_json("/object_info/ControlNetLoader", timeout=10)
            return list(
                data["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
            )
        except Exception:
            return []

    def free_memory(self) -> bool:
        """调 /free 清理 ComfyUI 显存/内存缓存（批量任务间隙防 OOM）。

        老版本 ComfyUI 无此接口：失败静默返回 False，不阻塞任务。
        """
        try:
            self._post_json("/free", {"unload_models": True,
                                      "free_memory": True}, timeout=30)
            return True
        except Exception:
            return False

    # ---------- 生成流程 ----------

    def queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            resp = self._post_json("/prompt", payload)
        except urllib.error.HTTPError as e:  # 校验失败（节点错误等）
            detail = e.read().decode("utf-8", "replace")
            raise ComfyUIError(f"工作流被 ComfyUI 拒绝 (HTTP {e.code}): {detail[:800]}") from e
        except urllib.error.URLError as e:
            raise ComfyUIOffline(f"无法连接 ComfyUI: {e.reason}") from e
        pid = resp.get("prompt_id")
        if not pid:
            raise ComfyUIError(f"/prompt 响应异常: {resp}")
        return pid

    def generate(
        self, workflow: dict, out_dir, timeout: float = 600.0
    ) -> list:
        """提交工作流并阻塞等待完成，返回保存到本地的图片路径列表。"""
        from pathlib import Path

        out_dir = Path(out_dir)
        prompt_id = self.queue_prompt(workflow)
        produced = self._wait_for_result(prompt_id, timeout)

        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for filename, subfolder, folder_type in produced:
            qs = urllib.parse.urlencode(
                {"filename": filename, "subfolder": subfolder, "type": folder_type}
            )
            data = self._get(f"/view?{qs}", timeout=60)
            path = out_dir / f"{prompt_id[:8]}_{filename}"
            path.write_bytes(data)
            saved.append(path)
        if not saved:
            raise ComfyUIError("任务完成但未收集到任何输出图片")
        return saved

    def upload_image(self, arr_bgra: np.ndarray, name: str) -> str:
        """上传 BGRA 图像到 ComfyUI input 目录，返回服务器端文件名。"""
        import io
        ok, buf = cv2.imencode(".png", arr_bgra)
        if not ok:
            raise ComfyUIError("图像编码失败")
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{name}.png"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode("utf-8")
        body += buf.tobytes()
        body += (f"\r\n--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
                 f"--{boundary}--\r\n").encode("utf-8")
        req = urllib.request.Request(
            self.http_base + "/upload/image", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     **self._auth_headers()})
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
        sub = resp.get("subfolder", "")
        return f"{sub}/{resp['name']}" if sub else resp["name"]

    def img2img(self, image_bgra: np.ndarray, *, checkpoint: str, prompt: str,
                negative_prompt: str, denoise: float = 0.6,
                seed: int | None = None, steps: int = 20, cfg: float = 7.0,
                out_dir, lora_name: str = "", lora_strength: float = 1.0,
                timeout: float = 300, template: dict | None = None,
                extra_images: dict | None = None) -> list:
        """以输入帧为底图做 img2img 重绘，保留原 alpha，返回保存路径列表。

        template: 自定义 img2img 工作流（须含 __IMAGE__ 占位符）；
        None = 默认 img2img_frame.json。Qwen 等专用环境应传对应模板。
        extra_images: 额外参考图 {占位符: BGRA}，如 {"__PREV__": 帧, "__NEXT__": 帧}
        （配合含 __PREV__/__NEXT__ 的上下文模板，动态多图槽）。
        """
        server_image = self.upload_image(image_bgra, f"pf_redraw_{int(time.time())}")
        if template is None:
            tpl_path = Path(__file__).parent / "workflows" / "img2img_frame.json"
            template = json.loads(tpl_path.read_text("utf-8"))
        extra_server = {k: self.upload_image(v, f"pf_ctx_{int(time.time())}_{k.strip('_')}")
                        for k, v in (extra_images or {}).items()}
        if seed is None:
            seed = random.randint(0, 2**48 - 1)
        workflow = build_workflow(
            template, checkpoint=checkpoint, prompt=prompt,
            negative_prompt=negative_prompt, seed=seed,
            width=image_bgra.shape[1], height=image_bgra.shape[0],
            steps=steps, cfg=cfg, lora_name=lora_name,
            lora_strength=lora_strength)
        for node in workflow.values():
            ins = node.get("inputs", {})
            for k2, v2 in list(ins.items()):
                if v2 == "__IMAGE__":
                    ins[k2] = server_image
                elif v2 == "__DENOISE__":
                    ins[k2] = float(denoise)
                elif isinstance(v2, str) and v2 in extra_server:
                    ins[k2] = extra_server[v2]
        prompt_id = self.queue_prompt(workflow)
        produced = self._wait_for_result(prompt_id, timeout)
        from pathlib import Path as _P
        out_dir = _P(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for filename, subfolder, folder_type in produced:
            qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder,
                                         "type": folder_type})
            data = self._get(f"/view?{qs}", timeout=60)
            p = out_dir / f"{prompt_id[:8]}_{filename}"
            p.write_bytes(data)
            saved.append(p)
        # 保留原 alpha（img2img 输出无透明通道，透明区按原 alpha 恢复）
        if saved:
            res = cv2.imdecode(np.frombuffer(saved[0].read_bytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            if res.shape[:2] != image_bgra.shape[:2]:  # 防御：引擎输出尺寸漂移
                res = cv2.resize(res, (image_bgra.shape[1], image_bgra.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            out = np.dstack([res, image_bgra[:, :, 3]])
            cv2.imwrite(str(saved[0]), out)
        return saved

    def generate_frame_series(self, anchor_bgra: np.ndarray, n_frames: int, *,
                              checkpoint: str, prompt: str,
                              negative_prompt: str, denoise: float = 0.45,
                              steps: int = 20, cfg: float = 7.0,
                              out_dir, lora_name: str = "",
                              lora_strength: float = 1.0,
                              timeout: float = 300, template: dict | None = None,
                              progress=None) -> list:
        """跨帧一致性系列生成：以 anchor 首帧为底图逐帧 img2img。

        每帧独立 seed、底图恒为 anchor（角色形状/构图保持一致，
        姿态变化幅度由 denoise 控制），返回 n_frames-1 个变体帧路径
        （按生成顺序）。progress(i, total) 可选回调用于 GUI 进度。
        template 透传给 img2img（Qwen 等专用模板）。
        """
        frames: list = []
        total = max(0, int(n_frames) - 1)
        for i in range(1, total + 1):
            paths = self.img2img(
                anchor_bgra, checkpoint=checkpoint, prompt=prompt,
                negative_prompt=negative_prompt, denoise=denoise,
                seed=None, steps=steps, cfg=cfg, out_dir=out_dir,
                lora_name=lora_name, lora_strength=lora_strength,
                timeout=timeout, template=template)
            frames.append(paths[0])
            if progress:
                progress(i, total)
        return frames

    def generate_pose_series(self, anchor_bgra: np.ndarray,
                             pose_paths: list, *, checkpoint: str,
                             controlnet_name: str, prompt: str,
                             negative_prompt: str, pose_strength: float = 1.0,
                             denoise: float = 0.45, steps: int = 20,
                             cfg: float = 7.0, out_dir,
                             lora_name: str = "", lora_strength: float = 1.0,
                             timeout: float = 300,
                             progress=None) -> list:
        """骨架驱动系列生成：每张骨架图（火柴人）产出一帧。

        外观锚定：每帧以 anchor 首帧做 img2img（保形/保色）；
        姿势控制：ControlNet OpenPose 吃骨架图，彻底解耦动作与画风。
        骨架图自动缩放到 anchor 尺寸（最近邻保持线条）。返回各帧路径。
        """
        h, w = anchor_bgra.shape[:2]
        tpl_path = Path(__file__).parent / "workflows" / "pose2img_frame.json"
        template = json.loads(tpl_path.read_text("utf-8"))
        server_anchor = self.upload_image(anchor_bgra,
                                          f"pf_anchor_{int(time.time())}")
        frames: list = []
        total = len(pose_paths)
        for i, pp in enumerate(pose_paths):
            pose = prepare_pose(pp, w, h)
            server_pose = self.upload_image(pose, f"pf_pose_{i}_{int(time.time())}")
            workflow = build_workflow(
                template, checkpoint=checkpoint, prompt=prompt,
                negative_prompt=negative_prompt,
                seed=random.randint(0, 2**48 - 1),
                width=w, height=h, steps=steps, cfg=cfg,
                lora_name=lora_name, lora_strength=lora_strength,
                controlnet_name=controlnet_name, pose_strength=pose_strength)
            for node in workflow.values():
                ins = node.get("inputs", {})
                for k2, v2 in list(ins.items()):
                    if v2 == "__IMAGE__":
                        ins[k2] = server_anchor
                    elif v2 == "__POSE__":
                        ins[k2] = server_pose
                    elif v2 == "__DENOISE__":
                        ins[k2] = float(denoise)
            prompt_id = self.queue_prompt(workflow)
            produced = self._wait_for_result(prompt_id, timeout)
            out_dir_p = Path(out_dir)
            out_dir_p.mkdir(parents=True, exist_ok=True)
            for filename, subfolder, folder_type in produced:
                qs = urllib.parse.urlencode({"filename": filename,
                                             "subfolder": subfolder,
                                             "type": folder_type})
                data = self._get(f"/view?{qs}", timeout=60)
                p = out_dir_p / f"{prompt_id[:8]}_{filename}"
                p.write_bytes(data)
                frames.append(p)
                break  # 每帧取第一张输出
            if progress:
                progress(i + 1, total)
        return frames

    def _wait_for_result(self, prompt_id: str, timeout: float) -> list[tuple[str, str, str]]:
        """WebSocket 监听执行事件，直到当前任务完成/失败/超时。"""
        ws = websocket.create_connection(
            f"{self.ws_base}?clientId={self.client_id}", timeout=15,
            header=self._auth_header_list() or None,
        )
        images: list[tuple[str, str, str]] = []
        current_nodes: dict[str, str] = {}  # prompt_id -> 正在执行的节点（进度用）
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    continue
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue  # 忽略其他任务的事件
                if mtype == "executed":
                    for node_out in (data.get("output") or {}).get("images", []):
                        images.append(
                            (
                                node_out.get("filename", ""),
                                node_out.get("subfolder", ""),
                                node_out.get("type", "output"),
                            )
                        )
                elif mtype == "executing":
                    if data.get("node") is None:  # node=None 表示整个任务执行结束
                        if images:
                            return images
                        raise ComfyUIError("任务结束但没有任何节点输出图片")
                elif mtype == "execution_error":
                    raise ComfyUIError(
                        f"节点 {data.get('node_type')} 执行失败: "
                        f"{data.get('exception_message', '未知错误')}"
                    )
                elif mtype == "execution_interrupted":
                    raise ComfyUIError("任务被中断")
            raise ComfyUIError(f"等待执行结果超时（{timeout:.0f}s）")
        finally:
            ws.close()


# ---------- 工作流模板填充 ----------

def build_workflow(
    template: dict,
    *,
    checkpoint: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    lora_name: str = "",
    lora_strength: float = 1.0,
    controlnet_name: str = "",
    pose_strength: float = 1.0,
) -> dict:
    """将模板中的占位符替换为实际参数，返回 API 格式工作流 dict。

    占位符: __CHECKPOINT__ __PROMPT__ __NEGATIVE__ __SEED__
            __WIDTH__ __HEIGHT__ __STEPS__ __CFG__
            __CONTROLNET__ __POSE_STRENGTH__（仅 ControlNet 模板）
            __IMAGE__ / __POSE__ 为运行时值，由调用方单独替换
    """
    mapping = {
        "__CHECKPOINT__": checkpoint,
        "__PROMPT__": prompt,
        "__NEGATIVE__": negative_prompt,
        "__SEED__": int(seed),
        "__WIDTH__": int(width),
        "__HEIGHT__": int(height),
        "__STEPS__": int(steps),
        "__CFG__": float(cfg),
        "__CONTROLNET__": controlnet_name,
        "__POSE_STRENGTH__": float(pose_strength),
    }
    wf = json.loads(json.dumps(template))  # deep copy
    for node in wf.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, str) and val in mapping:
                node["inputs"][key] = mapping[val]
    if lora_name:  # 注入 LoraLoader：模型与文本编码都过 LoRA
        wf["90"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora_name, "strength_model": lora_strength,
            "strength_clip": lora_strength, "model": ["1", 0], "clip": ["1", 1]}}
        for node in wf.values():
            ins = node.get("inputs", {})
            for k2, v2 in ins.items():
                if v2 == ["1", 0] and node["class_type"] in ("KSampler", "KSamplerAdvanced"):
                    ins[k2] = ["90", 0]
                elif v2 == ["1", 1] and node["class_type"] == "CLIPTextEncode":
                    ins[k2] = ["90", 1]
    return wf


def random_seed() -> int:
    return random.randint(0, 2**48 - 1)


def prepare_pose(path, width: int, height: int) -> np.ndarray:
    """骨架图预处理：读图 -> 统一为 BGR -> 缩放到生成尺寸（最近邻保线条）。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ComfyUIError(f"骨架图读取失败: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.resize(img, (int(width), int(height)),
                      interpolation=cv2.INTER_NEAREST)
