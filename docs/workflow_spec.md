# ComfyUI Workflow Template Spec / 工作流模板规范

## ⭐ 给 ComfyUI 工作流 Agent 的对接规范（请先读这节）

你的任务：产出可被 PixelForge 直接使用的 **ComfyUI 工作流 JSON 模板**。
把 JSON 交给 PixelForge 侧助手后，它会放入 `pixelforge/workflows/`，
GUI 的「工作流模板」下拉框会自动发现，无需改任何代码。

### 硬性格式要求

1. **必须是 API 格式**，不是 ComfyUI 界面里 Ctrl+S 导出的 UI 格式。
   - ✅ API 格式长这样：顶层是 `{ "1": {"class_type": "...", "inputs": {...}}, "2": {...} }`，
     连线用 `["<节点号>", <输出槽位号>]` 表示
   - ❌ UI 格式有 `"nodes": [...]`、`"links": [...]` 数组 —— 这种**不能**用
   - 在 ComfyUI 里：设置开启 `Enable Dev mode`（设置 → 高质/开发者），然后用
     **Save (API Format)** 按钮导出的才是 API 格式
2. 文件放在 `pixelforge/workflows/*.json`，文件名即 GUI 显示名（如 `sdxl_t2i.json` → 显示 "sdxl_t2i"）
3. 必须恰好含 **一个 `SaveImage` 节点**（程序收集它的输出图片）
4. `CheckpointLoaderSimple` 的 `ckpt_name` 必须是占位符 `__CHECKPOINT__`（模型由 GUI 选择）
5. **不要**在模板里写死 checkpoint 文件名、种子、尺寸、步数 —— 全部用占位符

### 占位符表（程序提交前自动替换）

| 占位符 | 替换为 | 出现位置 |
|---|---|---|
| `__CHECKPOINT__` | 模型文件名 | CheckpointLoaderSimple.ckpt_name |
| `__PROMPT__` | 角色描述 + 动作提示词 | 正向 CLIPTextEncode.text |
| `__NEGATIVE__` | 负面提示词 | 负向 CLIPTextEncode.text |
| `__SEED__` | 随机/锁定整数 | KSampler.seed |
| `__STEPS__` | 采样步数 | KSampler.steps |
| `__CFG__` | CFG 浮点 | KSampler.cfg |
| `__WIDTH__` / `__HEIGHT__` | 生成尺寸 | EmptyLatentImage |
| `__IMAGE__` | 服务端已上传的参考图文件名 | LoadImage.image（仅 img2img 类模板） |
| `__DENOISE__` | 重绘强度浮点 | KSampler.denoise（仅 img2img 类模板） |
| `__POSE__` | 服务端已上传的骨架图文件名 | LoadImage.image（仅 ControlNet 模板） |
| `__CONTROLNET__` | ControlNet 模型文件名 | ControlNetLoader.control_net_name |
| `__POSE_STRENGTH__` | 姿势控制强度浮点 | ControlNetApply.strength |

### LoRA 支持的连线约定（重要）

程序会在勾选 LoRA 时**自动注入**一个 `LoraLoader` 节点（节点号 `"90"`），改线规则：
- `KSampler.model` 为 `["1", 0]` → 替换为 `["90", 0]`
- 每个正向/负向 `CLIPTextEncode.clip` 为 `["1", 1]` → 替换为 `["90", 1]`

因此模板里请保持：**模型链从 `["1", 0]` 取、文本编码从 `["1", 1]` 取**，即可免费获得 LoRA 支持。不需要自己写 LoraLoader 节点。

### 现有模板样例（照这个结构写）

`txt2img_basic.json`（文生图精灵表）：

```json
{
  "1": {"class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "__CHECKPOINT__"}},
  "2": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "__PROMPT__", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "__NEGATIVE__", "clip": ["1", 1]}},
  "4": {"class_type": "EmptyLatentImage",
        "inputs": {"width": "__WIDTH__", "height": "__HEIGHT__", "batch_size": 1}},
  "5": {"class_type": "KSampler",
        "inputs": {"seed": "__SEED__", "steps": "__STEPS__", "cfg": "__CFG__",
                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                    "latent_image": ["4", 0]}},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "PixelForge", "images": ["6", 0]}}
}
```

`img2img_frame.json`（跨帧一致性：首帧锚定 + alpha 保形，供 img2img 系列生成逐帧重绘）：

```json
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "__CHECKPOINT__"}},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "__PROMPT__", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "__NEGATIVE__", "clip": ["1", 1]}},
  "4": {"class_type": "LoadImage", "inputs": {"image": "__IMAGE__"}},
  "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
  "6": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["5", 0], "mask": ["4", 1]}},
  "7": {"class_type": "KSampler", "inputs": {"seed": "__SEED__", "steps": "__STEPS__",
        "cfg": "__CFG__", "sampler_name": "euler", "scheduler": "normal",
        "denoise": "__DENOISE__", "model": ["1", 0], "positive": ["2", 0],
        "negative": ["3", 0], "latent_image": ["6", 0]}},
  "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
  "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "PixelForgeRedraw", "images": ["8", 0]}}
}
```

### 欢迎的扩展方向（PixelForge 侧已具备的消费能力）

- **精灵表布局控制**：PixelForge 自带投影拆分/网格切分，你的提示词/工作流只要能产出
  「横向排列、纯色背景、帧间距均匀」的角色动作序列即可，布局不需要完美
- **跨帧一致性**：如用 AnimateDiff / IPAdapter / ControlNet 等节点，只需保证
  上述占位符存在且有一个 SaveImage 输出；专有节点的输入参数可写死在模板里
  - 已内置 `pose2img_frame.json`：锚定 img2img + ControlNet OpenPose 双控
    （骨架定姿势、首帧定外观），客户端按骨架图序列逐帧驱动
- **Qwen Image 2.1 专用模板（已实测可用）**：本仓库自带两枚适配用户本地环境的模板，
  模型文件名写死（私人环境模板，其他环境按需改）：
  - `qwen_t2i.json`：UNETLoader + CLIPLoader(type=qwen_image) + **TextEncodeQwenImage21**
    （prompt/negative/latent 三合一节点，槽 0/1/2）+ QwenImage21Cache + EmptyLatentImage。
    注意：Qwen 2.1 **不能用普通 CLIPTextEncode**（会输出噪声图）
  - `qwen_i2i.json`：参考图编辑链（LoadImage -> TextEncodeQwenImage21 的
    `images.image_1` + `Using <image1>` 提示词语法，latent 跟随参考图尺寸），
    客户端上传前把参考图 pad 到 16 倍数白底
- **云端托管**：ComfyICU / liblibAI 等同协议服务也走同一套 JSON，
  节点包在目标平台是否可用请一并注明

### 验收流程（PixelForge 侧执行）

1. `json.loads` 通过且为 API 格式（无 `nodes`/`links` 顶层键）
2. 占位符齐全（`__CHECKPOINT__` 必须存在；img2img 类需 `__IMAGE__`/`__DENOISE__`）
3. 恰好一个 `SaveImage`；`KSampler` 存在
4. 放入 workflows/ 后 GUI 下拉可见，替换后提交本地 ComfyUI 实测出图

有任何一个检查不过，PixelForge 侧会把具体错误发回给你修改。

