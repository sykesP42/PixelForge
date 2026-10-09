# PixelForge Studio

> **一句话**：PixelForge Studio 是一条像素游戏动画生产线——丢进提示词或角色参考图，AI 生成动作序列，自动去背、裁帧、统一调色板，人工审完一键导出 Unity / Godot / Aseprite / GIF。
>
> **In one line:** PixelForge Studio is a pixel-art game animation pipeline — feed it a prompt or a character reference, generate motion frames with AI, auto-remove background / crop / unify palette, then export to Unity, Godot, Aseprite or GIF in one click.

* Windows 桌面应用（PySide6）｜GUI, no coding needed
* 双引擎：本地 ComfyUI 或云端 GPT-Image API ｜Dual engine: local ComfyUI or cloud GPT-Image API
* 全流程本地处理，API Key 只存在你自己的 `config/` ｜Everything runs locally; your API key stays on your machine

---

## 你能用它做什么 / What you can do

| 你想做的事 / Goal | 用哪个功能 / Where |
|---|---|
| 一句提示词生成角色动画 | 笨蛋模式：上传参考图（可省略）→ 写提示词 → 生成 |
| 用已有角色图做 AI 修图 / 换装 / 换背景 | 资产右键「作为生成素材」或预览下方「🎯 作为生成素材」 |
| 把 AI 出的网格图切成逐帧 | 图像工具坊：投影拆分 / 网格切分 |
| 去绿幕、去背、去光晕 | 图像工具坊：色键（HSV / ExG）+ despill |
| 让整套动画用同一套 24 色 | 调色板锁定（KMeans 统一调色板） |
| 批量做待机/行走/跑步 4 方向 | 批量动作 + 方向动作模板集 |
| 导出到引擎 | Unity .unitypackage / Godot .tres / Aseprite JSON / GIF |
| 管理历史产物 | 资产库：分类、收藏、文件夹、云资产、搜索复用提示词 |

---

## 快速开始 / Quick Start

```bash
pip install -r requirements.txt
python main.py
```

想要免 Python 的 exe？直接下载 Release 里的 `PixelForge_x.y.z` 压缩包，双击 `PixelForgeStudio.exe` 即可（打包步骤见 `docs/release_guide.md`）。

首次启动只需两步 / First-run setup:

1. **选引擎**：顶栏 ☁ API（云端）或 🖥 本地 Comfy
   Pick an engine: ☁ API (cloud) or 🖥 Local Comfy
2. **填配置**：云端填中转站地址 + API Key；本地填 Conda 环境 + ComfyUI 目录
   Cloud: relay URL + API key. Local: conda env + ComfyUI folder.

然后在「笨蛋模式」上传一张角色图（或留空纯文生图）、写提示词，点「生成」。
Then upload a character image (or leave it empty for text-to-image), write a prompt, hit **Generate**.

---

## 界面导览 / UI Overview

* **生成与审核**：中央是专业看图器——滚轮缩放、拖拽平移、**按住 Alt 出放大镜**（滚轮调倍率）、右下角鸟瞰图；下方四个快捷按钮：→ 图像工具坊 / ✏ 编辑 / 🎯 作为生成素材 / 📎 作为附属生成素材
  Center: pro image viewer (scroll to zoom, drag to pan, **hold Alt for loupe**, minimap bottom-right)
* **图像工具坊**：去背、裁切、像素化、调色板、换色、旋转、动作宏
  Image Lab: keying, cropping, pixelization, palette, palette swap, macros
* **动画播放**：FPS / 正放倒放往返 / 透明背景预览 / 帧间补间
  Player: FPS, ping-pong, transparent preview, tweening
* **批量动作 / 批量工坊**：多动作排队生成，断点续跑，统一调色板
  Batch: queued generation, resume, shared palette
* **资产库**：切到该页自动刷新；双击预览，右键重载/编辑/素材/导入提示词；📥 导入外部图片、☁ 上云
  Asset Library: auto-refresh on tab switch, double-click preview, right-click actions
* **队列日志（右侧）**：任务并发排队、⏹ 停止选中；右键任务可「载入该任务设置」复用修改、导入提示词、永久删除
  Queue & Log: concurrent tasks, stop button, right-click to reuse/import/delete

---

## 常用操作速查 / Cheatsheet

| 操作 / Action | 快捷键或入口 / How |
|---|---|
| 放大看细节 | 滚轮缩放；按住 **Alt** = 临时放大镜（滚轮调 2x~20x） |
| 适配窗口 / 1:1 | 双击 或 **F** / **1** |
| 放大镜开关 | **M** 或左下「🔍 放大镜」按钮 |
| 参考图发送至生成 | 资产右键「🖼 作为生成素材」（替换）/「📎 作为附属生成素材」（追加） |
| 复用上次的生成设置 | 队列右键「📥 载入该任务设置」 |
| 找回某张图的提示词 | 资产右键「📋 导入提示词」 |
| 切换深浅主题 | 设置 → 通用 → 主题 |

每个模块底部和状态栏都有操作提示，跟着提示走即可。
Every panel has a hint line at the bottom — just follow it.

---

## 导出格式 / Export

* Unity：帧序列 + `atlas.json`（含 hull 碰撞轮廓）→ **.unitypackage**，hull 自动生成 PolygonCollider2D
* Godot：SpriteFrames `.tres`（支持多动画组）
* Aseprite：JSON（hash / array 双格式）
* GIF：普通与透明背景
* 质量门禁：勾选「审核通过」才能导出

---

## 配置文件在哪里 / Where things are stored

| 文件 | 作用 |
|---|---|
| `config/settings.json` | 生成参数与 API Key（**不会上传 GitHub**） |
| `config/gen_queue.json` | 历史生成任务（可复用/删除） |
| `config/characters.json` | 角色档案 |
| `outputs/` | 所有生成与导出产物 |
| `outputs/library.json` | 资产库收藏/分类/文件夹索引 |

---

## 开发者 / Developers

* 自定义 ComfyUI 工作流模板规范（占位符、LoRA 注入约定）：`docs/workflow_spec.md`
* 打包与发布流程：`docs/release_guide.md`
* 测试（离屏）：

```bash
export QT_QPA_PLATFORM=offscreen   # Windows: set QT_QPA_PLATFORM=offscreen
python tests/test_processing.py && python tests/test_gen_queue.py && python tests/smoke_test.py
```

## 已知限制 / Known limitations

* Unity 侧的 C# 导入器需在编辑器内跑一次「PixelForge/运行链路自测」完成验收
* 帧对齐在强不对称道具场景有局限（推荐用「脚底中心」对齐）
* 云端生成通常 1~3 分钟/任务，取决于中转站负载；下载失败时任务响应已落盘为 `task_*.json`，可免费重试下载
* `pixelforge/workflows/qwen_*.json` 为私人环境模板（模型名写死），其他环境请按需修改
