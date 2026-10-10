"""云端图像引擎（通用 OpenAI 兼容中转站，多模型多中转）。

支持的中转格式：
1. **OpenAI 同步**（/v1/images/generations，大多数 one-api/new-api 中转）：
   响应 {data: [{url | b64_json}]}
2. **多米异步**（?async=true + /v1/tasks/{id} 轮询）：state succeeded + data.images[].url

参考图：base64 data URI（本地文件自动转换）；多图数组。
模型：gpt-image-2 / gpt-image-1 / nano-banana / seedream / Flux 等任意
中转站支持的 model 字符串（UI 下拉 + 可自定义输入）。
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from pixelforge.processing.image_processor import (
    ImageProcessor,
    split_grid_even,
    strip_ground_lines,
)

# 直连 opener：绕过系统代理（本机代理常把中转站/CDN 的 CONNECT 隧道掐成 502）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
from pixelforge.processing.motion_metrics import identity_report


class CloudEngineError(RuntimeError):
    pass


def _auth_header(key: str) -> dict:
    """鉴权头：兼容 Bearer 前缀与裸 key 两种中转习惯。"""
    k = key.strip()
    if k.lower().startswith(("bearer ", "sk-")) or " " in k:
        return {"Authorization": f"Bearer {k}" if not k.lower().startswith(
            "bearer") else k}
    return {"Authorization": k}


def _post_json(url: str, body: dict, key: str, timeout: int = 180) -> dict:
    r = urllib.request.Request(
        url, method="POST", data=json.dumps(body).encode(),
        headers={**_auth_header(key), "Content-Type": "application/json"})
    try:
        return json.loads(_OPENER.open(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise CloudEngineError(f"HTTP {e.code}: {detail}") from e


def _get_json(url: str, key: str, timeout: int = 60) -> dict:
    r = urllib.request.Request(
        url, headers={**_auth_header(key), "Content-Type": "application/json"})
    try:
        return json.loads(_OPENER.open(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise CloudEngineError(f"HTTP {e.code}: {detail}") from e


def img_to_data_uri_legacy(path: str, max_kb: int = 30) -> str:
    """本地图 → base64 data URI，自动压缩到 max_kb 以内。

    （中转站普遍有 ~64KB 请求体限制：大图 base64 会撑爆 body）
    迭代降质量/缩尺寸直到达标；无法读取时按原文件字节编码。
    """
    import cv2
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"

    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None:   # 非 OpenCV 可读格式（GIF/WebP/EMF 等）→ PIL 解码
        try:
            import io
            from PIL import Image
            pim = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.array(pim)                      # RGB
            im = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001  PIL 也读不了：原样编码
            b64 = base64.b64encode(raw).decode()
            return f"data:image/{ext};base64,{b64}"

    h, w = im.shape[:2]
    best = None
    for quality in (92, 80, 70, 60, 50):
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
            rw, rh = max(64, int(w * scale)), max(64, int(h * scale))
            ok, enc = cv2.imencode(
                ".jpg", cv2.resize(im, (rw, rh),
                                   interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            b64 = base64.b64encode(enc).decode()
            if len(b64) // 1024 <= max_kb:
                return f"data:image/jpeg;base64,{b64}"
            best = b64 if best is None or len(b64) < len(best) else best
    # 全档仍超标：最后一搏——强制缩到 320px 低质量（尽量保证能提交）
    scale = min(1.0, 320 / max(h, w))
    rw, rh = max(64, int(w * scale)), max(64, int(h * scale))
    ok, enc = cv2.imencode(
        ".jpg", cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA),
        [cv2.IMWRITE_JPEG_QUALITY, 30])
    if ok:   # 320px 档是所有档里最小的：即便略超 max_kb 也优先返回
        return f"data:image/jpeg;base64,{base64.b64encode(enc).decode()}"
    return best or f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _fetch_bytes(url: str, timeout: int) -> bytes:
    """单次取字节（带浏览器 UA，防止 CDN 拦截 python-urllib）。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Accept": "*/*", "Referer": "https://duomiapi.com/"})
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, out: Path, timeout: int = 180) -> Path:
    """多路降级下载：直链(默认网络) → 直链(绕过系统代理) → HTTP 明文
    → weserv 图片代理 → duomiapi /cdn 转发。任一路成功即返回。"""
    attempts: list[tuple[str, str]] = [("直链", url)]
    import os
    no_proxy = url.replace("https://", "http://")
    attempts += [
        ("直链-绕过代理", "noproxy:" + url),
        ("HTTP明文", no_proxy),
        ("weserv代理",
         "https://images.weserv.nl/?url=" +
         urllib.parse.quote(url.replace("https://", ""), safe="")),
    ]
    if "dmiapi" in url:
        attempts.append(("中转转发", url.replace(
            "https://cdn3.dmiapi.com", "https://duomiapi.com/cdn")))
    errors = []
    for name, u in attempts:
        for retry in range(2):
            try:
                if u.startswith("noproxy:"):
                    real = u[8:]
                    opener = urllib.request.build_opener(
                        urllib.request.ProxyHandler({}))
                    req = urllib.request.Request(
                        real, headers={"User-Agent": _UA})
                    data = opener.open(req, timeout=timeout).read()
                else:
                    data = _fetch_bytes(u, timeout)
                if len(data) < 512:
                    raise CloudEngineError(f"响应过小({len(data)}B)疑似错误页")
                out.write_bytes(data)
                return out
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}#{retry + 1}: {str(e)[:80]}")
                time.sleep(1.5)
    raise CloudEngineError("图片下载全部通道失败:\n  " + "\n  ".join(errors[-8:]))


def _extract_images(resp: dict) -> list[str]:
    """从各类响应结构提取图片（url 或 b64_json）。"""
    # OpenAI 标准: {"data": [{"url"|"b64_json"}]}
    if isinstance(resp.get("data"), list) and resp["data"]:
        out = []
        for d in resp["data"]:
            if d.get("url"):
                out.append(d["url"])
            elif d.get("b64_json"):
                out.append("data:image/png;base64," + d["b64_json"])
        if out:
            return out
    # 多米异步: {"data": {"images": [{"url"}]}}
    data = resp.get("data")
    if isinstance(data, dict):
        imgs = data.get("images") or []
        out = [im["url"] for im in imgs if im.get("url")]
        if out:
            return out
    # 通用兜底: 顶层 images
    imgs = resp.get("images") or []
    return [im["url"] for im in imgs if isinstance(im, dict) and im.get("url")]


def _save_image(src: str, out_dir: Path, tag: str) -> Path:
    """url 下载 或 data URI 解码 → 落盘。"""
    out = out_dir / f"cloud_{tag}_{int(time.time())}.png"
    if src.startswith("data:"):
        b64 = src.split("base64,", 1)[-1]
        out.write_bytes(base64.b64decode(b64))
        return out
    _download(src, out)
    return out


# NANO-BANANA 系列（gemini 接口）：GUI 别名 → 中转站真实模型名
NANO_ALIASES = {
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
}


def _generate_nano(api_key: str, base_url: str, prompt: str,
                   ref_paths: list[str] | None, model: str, size: str,
                   out_dir: Path, timeout_s: int = 900,
                   oversea: bool = False, should_cancel=None,
                   log=print) -> Path:
    """NANO-BANANA（gemini）系列：文生图 / 图生图（编辑）+ 任务轮询。

    - 文生图  POST {base}/api/gemini/nano-banana
    - 图生图  POST {base}/api/gemini/nano-banana-edit（image_urls 仅公网 URL）
    - 轮询    GET  {base}/api/gemini/nano-banana/{task_id}
    """
    import math
    base = _sanitize_base(base_url)
    model = NANO_ALIASES.get(model.strip().lower(), model.strip())
    body: dict = {"model": model, "prompt": prompt, "oversea": bool(oversea)}
    k = (size or "1k").strip().lower()
    if k in ("1k", "2k", "4k"):
        body["image_size"] = k.upper()   # K 大写；nano-pro / nano-2 支持
    if ref_paths:
        endpoint = f"{base}/api/gemini/nano-banana-edit"
        urls = []
        for p in ref_paths:
            p = str(p)
            if p.startswith("http://") or p.startswith("https://"):
                urls.append(p)
                continue
            log(f"  参考图上传图床（nano 系列要求公网 URL）: {Path(p).name}")
            url = _upload_tmpfiles(p)
            if not url:
                raise CloudEngineError(
                    f"参考图上传图床失败: {p}\n"
                    "nano-banana 系列不支持 base64，请检查网络或改用 gpt-image 系列")
            urls.append(url)
        if urls:
            body["image_urls"] = urls
    else:
        endpoint = f"{base}/api/gemini/nano-banana"

    log(f"  提交云端任务: {base}  模型 {model}")
    resp = _post_json(endpoint, body, api_key)
    if resp.get("code") not in (None, 200, "200"):
        raise CloudEngineError(f"提交失败: {resp.get('msg', resp)[:200]}")
    tid = (resp.get("data") or {}).get("task_id") or resp.get("id")
    if not tid:
        raise CloudEngineError(f"中转未返回 task_id: {str(resp)[:200]}")
    log(f"  中转返回异步任务: {tid[:12]}...")

    # 轮询（响应结构 {code,msg,data:{state,data:{images[]}}}）
    start = time.monotonic()
    deadline = start + timeout_s
    last_pct = -1
    while time.monotonic() < deadline:
        time.sleep(5)
        if should_cancel and should_cancel():
            raise CloudEngineError("⏹ 任务已停止")
        r = _get_json(f"{base}/api/gemini/nano-banana/{tid}", api_key)
        if r.get("code") not in (200, "200"):
            continue
        d = r.get("data") or {}
        state = str(d.get("state") or "").lower()
        if state in ("succeeded", "success", "completed", "done"):
            try:
                _persist_task(d, base, tid, out_dir)
            except Exception:  # noqa: BLE001
                pass
            inner = d.get("data") or {}
            imgs = [i.get("url") for i in (inner.get("images") or [])
                    if i.get("url")]
            if not imgs:
                raise CloudEngineError(
                    f"任务成功但无图片: {json.dumps(d, ensure_ascii=False)[:300]}")
            log("  99% 生成完成，正在下载图片...")
            try:
                return _save_image(imgs[0], out_dir, "nano")
            except CloudEngineError as e:
                raise CloudEngineError(
                    f"{e}\n任务 id 已保存: task_{tid}.json（可手动重试下载）") from e
        if state in ("failed", "error", "cancelled"):
            err = d.get("msg") or ""
            raise CloudEngineError(f"云端任务失败: {state} {str(err)[:200]}")
        elapsed = time.monotonic() - start
        est_pct = min(95, int(100 * (1 - math.exp(-elapsed / 45))))
        if est_pct != last_pct:
            last_pct = est_pct
            log(f"  云端生成中... {est_pct}%（已等待 {int(elapsed)}s，"
                f"通常 1~3 分钟）")
    raise CloudEngineError(f"云端任务超时（{timeout_s}s）")


def _upload_tmpfiles(path: str, timeout: int = 60) -> str:
    """本地图上传到 tmpfiles.org 免费图床（无需注册），返回公网 URL。

    中转站的 image 参数要求公网 URL（不接受 base64 data URI）。
    """
    boundary = "----PixelForgeBoundary"
    fname = Path(path).name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{fname}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    import ssl
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://tmpfiles.org/api/v1/upload", data=body,
                method="POST",
                headers={"Content-Type":
                         f"multipart/form-data; boundary={boundary}"})
            ctx = ssl.create_default_context()
            resp = json.loads(_OPENER.open(
                req, timeout=timeout, context=ctx).read())
            url = resp.get("data", {}).get("url", "")
            if not url:
                raise CloudEngineError(
                    f"图床上传失败: {json.dumps(resp)[:200]}")
            return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  图床上传第 {attempt + 1} 次失败: {e}，重试...")
            time.sleep(2)
    raise CloudEngineError(f"图床上传失败（3 次重试）: {last_err}")


# ---- 官方参数硬约束（来自 duomiapi gpt-image-2 接口文档）----
PROMPT_MAX_LEN = 5000            # 提示词最大字符数
SIZE_EDGE_MIN, SIZE_EDGE_MAX = 16, 3840        # 每条边范围
SIZE_AREA_MIN, SIZE_AREA_MAX = 655360, 8294400  # 像素预算 width×height
SIZE_RATIOS = {"1:1", "3:2", "2:3", "16:9", "9:16", "1:2", "2:1",
               "4:3", "3:4", "5:4", "4:5"}
QUALITIES = ("low", "medium", "high")
BACKGROUNDS = ("auto", "opaque", "transparent")

DEFAULT_BASE = "https://duomiapi.com"


def _sanitize_base(url: str) -> str:
    """中转站地址净化：
    1) 用户把完整端点（…/v1/images/generations）粘进来 → 截回域名；
    2) 占位示例地址（example-relay 等）→ 回退默认多米站。"""
    import re as _re
    u = (url or "").strip().rstrip("/")
    if not u:
        return DEFAULT_BASE
    m = _re.match(r"^(https?://[^/]+)", u)
    if not m:
        return DEFAULT_BASE
    host = m.group(1)
    if "example" in host:
        return DEFAULT_BASE
    return host


def validate_prompt(prompt: str) -> str:
    """提示词校验：非空、≤5000 字符（超限截断）。"""
    p = (prompt or "").strip()
    if not p:
        raise CloudEngineError("提示词为空")
    if len(p) > PROMPT_MAX_LEN:
        p = p[:PROMPT_MAX_LEN]
    return p


def validate_quality(quality: str) -> str:
    """quality 校验：仅允许 low/medium/high。"""
    q = (quality or "medium").strip().lower()
    if q not in QUALITIES:
        raise CloudEngineError(
            f"quality 只允许 {'/'.join(QUALITIES)}，收到: {quality!r}")
    return q


def validate_background(bg: str) -> str:
    """background 校验：仅允许 auto/opaque/transparent。"""
    b = (bg or "auto").strip().lower()
    if b not in BACKGROUNDS:
        raise CloudEngineError(
            f"background 只允许 {'/'.join(BACKGROUNDS)}，收到: {bg!r}")
    return b


def _normalize_size(size: str, grid_wh=(1536, 768)) -> str:
    """size 参数规范化 + 硬校验（非法输入直接报错，防无效请求）。

    官方规则：像素格式 widthxheight（宽高均被 16 整除，每边 [16,3840]，
    像素预算 655360 ≤ w×h ≤ 8294400）；或比例串（1:1/2:1/16:9 等）；或 auto。
    """
    s = (size or "").strip().lower()
    if not s or s == "auto":
        return "auto"
    # GUI 分辨率档位映射（像素预算内最大正方形：4k→2880 恰好=预算上限）
    tier = {"1k": "1024x1024", "2k": "2048x2048", "4k": "2880x2880"}
    if s in tier:
        return tier[s]
    if s in SIZE_RATIOS:
        return s
    if "x" in s:
        parts = s.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
            if w % 16 or h % 16:
                raise CloudEngineError(
                    f"尺寸 {w}x{h} 非法：宽和高都必须被 16 整除")
            if not (SIZE_EDGE_MIN <= w <= SIZE_EDGE_MAX
                    and SIZE_EDGE_MIN <= h <= SIZE_EDGE_MAX):
                raise CloudEngineError(
                    f"尺寸 {w}x{h} 非法：每条边须在 "
                    f"[{SIZE_EDGE_MIN}, {SIZE_EDGE_MAX}] 内")
            area = w * h
            if not (SIZE_AREA_MIN <= area <= SIZE_AREA_MAX):
                raise CloudEngineError(
                    f"尺寸 {w}x{h} 非法：像素预算须在 "
                    f"{SIZE_AREA_MIN:,} ~ {SIZE_AREA_MAX:,} 之间（当前 {area:,}）")
            return f"{w}x{h}"
        raise CloudEngineError(
            f"size 格式非法: {size!r}（应为 1024x1024 像素格式或 2:1 比例格式）")
    raise CloudEngineError(
        f"size 非法: {size!r}（支持比例 {'/'.join(sorted(SIZE_RATIOS))} "
        f"或 auto 或自定义像素 WxH）")


def generate_grid(api_key: str, base_url: str, prompt: str,
                  ref_paths: list[str] | None, model: str, size: str,
                  out_dir: Path, timeout_s: int = 900,
                  background: str = "auto",
                  quality: str = "medium",
                  oversea: bool = False,
                  should_cancel=None,
                  log=print) -> Path:
    """通用云端生成：按模型自动路由 gpt-image / nano-banana 系列中转。

    返回网格图路径。入口处硬校验 prompt/size/quality/background，
    非法参数在本地即报错，不会发出无效请求。
    """
    if (model or "").strip().lower() in NANO_ALIASES or \
            (model or "").strip().lower().startswith("gemini-"):
        return _generate_nano(api_key, base_url, prompt, ref_paths,
                              model, size, out_dir, timeout_s,
                              oversea=oversea, should_cancel=should_cancel,
                              log=log)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _sanitize_base(base_url)
    prompt = validate_prompt(prompt)
    size = _normalize_size(size, (1536, 768))
    background = validate_background(background)
    quality = validate_quality(quality)
    body: dict = {"model": model, "prompt": prompt, "size": size,
                  "quality": quality, "oversea": bool(oversea)}
    if background != "auto":
        body["background"] = background
    if not (model or "").strip():
        raise CloudEngineError("模型名不能为空")
    if ref_paths:
        # 压缩到 data URI <= 20KB/张（中转站创建时请求体 ~64KB 限制）
        imgs = [img_to_data_uri_legacy(p, max_kb=20)
                for p in ref_paths if Path(p).is_file()]
        if imgs:
            body["image"] = imgs
            sizes = ", ".join(str(len(u) // 1024) + "KB" for u in imgs)
            log(f"  参考图已压缩编码: {len(imgs)} 张（{sizes}）")   # 单数键；多余字段会被部分中转拒绝

    log(f"  提交云端任务: {base}  模型 {model}")
    # ---- 尝试同步（OpenAI 标准，网络抖动自动重试）----
    resp = None
    last_err = None
    for attempt in range(4):
        if should_cancel and should_cancel():
            raise CloudEngineError("⏹ 任务已停止")
        try:
            resp = _post_json(f"{base}/v1/images/generations", body, api_key)
            break
        except CloudEngineError as e:
            if "HTTP 4" in str(e)[:8]:
                hint = ""
                if "HTTP 400" in str(e)[:9]:
                    hint = ("\n建议：① 参考图过大或不被支持（可换 PNG/JPG 或去掉参考图）；"
                            "② 检查模型名是否为该中转站支持的模型")
                raise CloudEngineError(f"{e}{hint}") from e   # 请求错误不重试
            last_err = e
            log(f"  网络抖动重试 {attempt + 1}/4: {str(e)[:60]}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            log(f"  网络抖动重试 {attempt + 1}/4: {str(e)[:60]}")
        time.sleep(3)
    if resp is None:
        raise CloudEngineError(f"同步请求重试均失败: {last_err}")
    try:
        imgs = _extract_images(resp)
        if imgs:
            log("  同步接口成功")
            _persist_task(resp, base,
                          "sync_" + time.strftime("%Y%m%d_%H%M%S"), out_dir)
            log("  99% 正在下载图片...")
            return _save_image(imgs[0], out_dir, "sync")
        # 同步返回了任务 id（部分异步中转无 ?async=true 也返回 id）
        tid = resp.get("id") or resp.get("task_id")
        if tid:
            log(f"  中转返回异步任务: {tid[:12]}...")
            return _poll_task(base, tid, api_key, out_dir, timeout_s, log,
                              should_cancel)
        raise CloudEngineError(
            f"响应无图片且无任务 id: {json.dumps(resp, ensure_ascii=False)[:300]}")
    except CloudEngineError:
        raise
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            log("  同步端点不存在，尝试异步端点...")
        else:
            detail = e.read().decode(errors="replace")[:300]
            raise CloudEngineError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CloudEngineError(f"网络错误: {e}") from e

    # ---- 异步端点（多米风格）----
    try:
        resp = _post_json(f"{base}/v1/images/generations?async=true", body,
                          api_key)
        tid = resp.get("id") or resp.get("task_id")
        if not tid:
            raise CloudEngineError(
                f"异步创建失败: {json.dumps(resp, ensure_ascii=False)[:300]}")
        return _poll_task(base, tid, api_key, out_dir, timeout_s, log,
                          should_cancel)
    except CloudEngineError:
        raise
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise CloudEngineError(f"异步端点也失败 HTTP {e.code}: {detail}") from e


def _persist_task(st: dict, base: str, tid: str, out_dir: Path) -> None:
    """把任务完整响应（含图片 URL 与任务 id）落盘——下载失败可凭
    task_{id}.json 免费重试下载，不必重新花钱生成。"""
    try:
        rec = {"base": base, "task_id": tid, "saved_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"), "response": st}
        (out_dir / f"task_{tid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _poll_task(base: str, tid: str, api_key: str, out_dir: Path,
               timeout_s: int, log=print, should_cancel=None) -> Path:
    """轮询异步任务（兼容 state/status 字段与多种结果结构）。

    多米等中转的 progress 字段恒为 0，不提供真实百分比——
    改用耗时指数估算模拟进度（前快后慢，封顶 95%，成功即 100%）；
    API 若某天返回真实 progress>0 则优先采用。
    """
    import math
    start = time.monotonic()
    deadline = start + timeout_s
    last_pct = -1
    while time.monotonic() < deadline:
        time.sleep(5)
        if should_cancel and should_cancel():
            raise CloudEngineError("⏹ 任务已停止")
        st = _get_json(f"{base}/v1/tasks/{tid}", api_key)
        state = str(st.get("state") or st.get("status") or "").lower()
        if state in ("succeeded", "success", "completed", "done"):
            _persist_task(st, base, tid, out_dir)
            imgs = _extract_images(st)
            if not imgs:
                raise CloudEngineError(
                    f"任务成功但无图片: {json.dumps(st, ensure_ascii=False)[:300]}")
            log("  99% 生成完成，正在下载图片...")
            try:
                return _save_image(imgs[0], out_dir, "async")
            except CloudEngineError as e:
                raise CloudEngineError(
                    f"{e}\n任务 id 已保存: task_{tid}.json "
                    f"（可在浏览器打开图片 URL 手动保存，或稍后重试下载，无需再次生成）") from e
        if state in ("failed", "error", "cancelled"):
            err = st.get("error") or st.get("message") or ""
            raise CloudEngineError(
                f"云端任务失败: {state} {str(err)[:200]}")
        # 进度显示：API 真实进度 > 0 用真实值，否则按耗时估算
        api_pct = int(st.get("progress") or 0)
        elapsed = time.monotonic() - start
        est_pct = min(95, int(100 * (1 - math.exp(-elapsed / 45))))
        pct = api_pct if api_pct > 0 else est_pct
        if pct != last_pct:               # 进度条只在变化时刷行
            last_pct = pct
            log(f"  云端生成中... {pct}%（已等待 {int(elapsed)}s，"
                f"通常 1~3 分钟）")
    raise CloudEngineError(f"云端任务超时（{timeout_s}s）")


def split_grid_frames(grid_path: Path, proc: ImageProcessor,
                      gif_h: int = 512) -> tuple:
    """网格图 → 色键 → 地面线清除（带保护）→ 等分切割 → 帧级紧致
    → 底对齐统一缩放（同因子）→ (frames, identity_min)。"""
    grid = cv2.imread(str(grid_path), cv2.IMREAD_UNCHANGED)
    if grid is None:
        raise CloudEngineError(f"网格图读取失败: {grid_path}")
    if grid.shape[2] == 4:
        grid = cv2.cvtColor(grid, cv2.COLOR_BGRA2BGR)
    grid = grid[30:-30, 30:-30]
    bgra = proc.remove_background(grid)
    content = (bgra[:, :, 3] > 0).mean()
    if content < 0.05:
        raise CloudEngineError(f"色键后内容占比 {content:.2f}，疑似生成崩坏")
    before = int((bgra[:, :, 3] > 0).sum())
    cleaned = strip_ground_lines(bgra)
    if before > 0 and int((cleaned[:, :, 3] > 0).sum()) < before * 0.6:
        cleaned = bgra  # 误删保护回滚
    cells = split_grid_even(cleaned, 2, 4)
    tights = []
    for c in cells:
        ys, xs = np.where(c[:, :, 3] > 8)
        if ys.size:
            t = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if t.shape[0] > 20 and t.shape[1] > 20:
                tights.append(t)
    if len(tights) < 8:
        raise CloudEngineError(f"拆帧不足 {len(tights)}/8")
    max_h = max(t.shape[0] for t in tights)
    scale = min(1.0, gif_h / max_h)
    norm = []
    for t in tights:
        nw = max(8, int(t.shape[1] * scale))
        nh = max(8, int(t.shape[0] * scale))
        norm.append(cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA))
    H = max(r.shape[0] for r in norm)
    W = max(r.shape[1] for r in norm)
    frames = []
    for r in norm:
        cv = np.zeros((H, W, 4), np.uint8)
        dx, dy = (W - r.shape[1]) // 2, H - r.shape[0]
        cv[dy:dy + r.shape[0], dx:dx + r.shape[1]] = r
        frames.append(cv)
    rep = identity_report(frames)
    return frames, rep["min"]


def to_gif(frames: list, path: Path, durations: list[int]) -> None:
    from PIL import Image
    gifs = []
    for f in frames:
        af = f[:, :, 3:4].astype(np.float32) / 255.0
        comp = (f[:, :, :3] * af + 255 * (1 - af)).astype(np.uint8)
        gifs.append(Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)))
    gifs[0].save(path, save_all=True, append_images=gifs[1:],
                 duration=durations, loop=0)
