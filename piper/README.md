📘 Piper TTS Mac 用户打包与分发指南
1️⃣ 目录结构要求（开发者侧）

确保你的项目目录干净、只包含必要文件：

piper/
├── tts_service_piper.py       # 核心服务代码，已包含 __main__ 入口
├── models/                    # Piper TTS 模型文件
│   ├── zh_CN-xiao_ya-medium.onnx
│   ├── zh_CN-xiao_ya-medium.onnx.json
│   ├── zh_CN-chaowen-medium.onnx
│   ├── zh_CN-chaowen-medium.onnx.json
│   ├── zh_CN-huayan-medium.onnx
│   └── zh_CN-huayan-medium.onnx.json
├── g2pW/                      # G2P 数据
│   ├── config.py
│   ├── g2pw.onnx
│   ├── MONOPHONIC_CHARS.txt
│   ├── POLYPHONIC_CHARS.txt
│   └── version
├── outputs/                   # 输出目录，可为空
├── go_piper.command           # 启动脚本
└── README.md                  # 开源协议及使用说明
2️⃣ 打包 Piper TTS 为单文件可执行文件（开发者操作）

激活你的 Python 环境（如 Conda 环境 piper310）：

conda activate piper310

使用 PyInstaller 打包：

pyinstaller tts_service_piper.py \
  --onefile \
  --name piper_exec \
  --collect-all onnxruntime \
  --collect-all fastapi \
  --collect-all uvicorn \
  --collect-all soundfile \
  --collect-all librosa \
  --collect-all pydub \
  --add-data "models:models" \
  --add-data "g2pW:g2pW"

⚠️ 注意：

--add-data "models:models" 和 --add-data "g2pW:g2pW" 会把目录打包进单文件。

打包后的 piper_exec 会包含模型和 G2P 数据，用户无需额外安装 Python。

3️⃣ 编写启动脚本 go_piper.command
#!/bin/bash

# 切换到命令文件所在目录
cd "$(dirname "$0")"

echo "📍 Working dir: $(pwd)"
echo "🚀 Starting Piper TTS service..."

# 使用 caffeinate 防止 Mac 休眠
caffeinate -i ./piper_exec

echo ""
echo "🛑 TTS service stopped. Press Enter to close."
read

⚠️ 注意：

caffeinate -i 可以防止 Mac 进入睡眠。

双击 go_piper.command 就会启动服务。

用户无需安装 Python 或依赖库。

4️⃣ 用户操作指南

用户下载你的压缩包并解压，例如：

piper_mac/
├── piper_exec
├── go_piper.command
└── outputs/

打开终端（Terminal）并进入解压目录：

cd /path/to/piper_mac

给命令文件授权（第一次运行必须）：

chmod +x go_piper.command
chmod +x piper_exec

双击 go_piper.command 或在终端运行：

./go_piper.command

服务启动后，HTTP API 默认监听 127.0.0.1:8011：

健康检查：GET http://127.0.0.1:8011/health

TTS 生成：POST http://127.0.0.1:8011/tts/generate

输出音频文件会生成在 outputs/ 目录下。

5️⃣ 开源协议提示

Piper 最新版本为 GPLv3。

分发时必须：

提供原始源代码或你的修改版源代码（例如 GitHub 仓库）。

保留 GPLv3 授权说明。

在 README.md 中说明使用了 Piper TTS 以及其他开源依赖。

明确用户可自由使用、修改和再发布。

示例 README 内容：

# Piper TTS Mac 版本

本软件基于开源 Piper TTS (GPLv3) 以及以下依赖：
- fastapi
- uvicorn
- onnxruntime
- librosa
- pydub
- soundfile

遵循 GPLv3 协议，源代码可在 https://github.com/你的账号/仓库 查看。
6️⃣ 注意事项

piper_exec 是独立可执行文件，包含模型和依赖，无需 Python。

outputs/ 可用来存储用户生成的音频。

未来增加模型或音色，只需重新打包 piper_exec。

macOS 可能会提示“来自未知开发者”，用户只需右键 → 打开即可。

模型下载地址：
https://huggingface.co/rhasspy/piper-voices/tree/main