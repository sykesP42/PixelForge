# PixelForge GitHub + EXE Release Guide

## 1. GitHub 上传（当前项目已清理完毕）

已确认安全：
- API Key 已从所有源码移除（grep 全仓零匹配）
- config/（settings/gen_history/gen_queue 等，含 Key）整体移出并忽略
- outputs/、logs/、build/、dist/、task_*.json 均已忽略

步骤（在项目根目录）：

    git init
    git add .
    git status        # 确认没有 config/、outputs/、build/、dist/
    git commit -m "PixelForge Studio v1.0"
    git branch -M main
    git remote add origin https://github.com/<你的用户名>/PixelForge.git
    git push -u origin main

注意：.workbuddy/（AI 工作记忆）已被忽略，不会上传。

## 2. 打包 EXE（PyInstaller 单文件夹版，推荐）

    pip install pyinstaller
    pyinstaller --noconfirm --windowed --name PixelForge ^
      --add-data "pixelforge/workflows;pixelforge/workflows" ^
      --collect-all PySide6 ^
      main.py

- 产物在 dist/PixelForge/，把整个文件夹压缩发给别人
- 对方双击 PixelForge.exe 即可（首次启动会自动生成 config/）

可选单文件版（启动稍慢）：把 --windowed 换成 --onefile --windowed

## 3. 打包前跑一遍测试

    python tests/smoke_test.py
    QT_QPA_PLATFORM=offscreen python tests/test_gen_queue.py

## 4. 收件人需要的运行环境

- Windows 10/11 x64（无需装 Python）
- 想用本地 ComfyUI 引擎：需自行安装 ComfyUI
- 云端引擎：在设置里填自己的 API Key

## 5. 首次启动脚本（可选）

start.bat 已提供；EXE 版可直接创建快捷方式。
