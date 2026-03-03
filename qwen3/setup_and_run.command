#!/bin/bash

# ============================================
# Qwen3-TTS HTTP Service 一键启动脚本 (Mac)
# ============================================

# 设置脚本退出行为：遇到错误立即退出
set -e

# 获取脚本所在绝对目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🤖 Qwen3-TTS 服务启动向导"
echo "当前目录: $SCRIPT_DIR"
echo "----------------------------------------"

# 定义变量
ENV_DIR="$SCRIPT_DIR/qwen3-tts_env"
ENV_TAR="qwen3-tts_env.tar.gz"
SERVICE_SUBDIR="qwen3-tts"  # 代码所在的子目录名

# 1. 检查并解压 Conda 环境
if [ ! -d "$ENV_DIR" ]; then
    if [ ! -f "$ENV_TAR" ]; then
        echo "❌ 错误：找不到环境包 $ENV_TAR"
        echo "请确保该文件与脚本在同一目录下。"
        exit 1
    fi

    echo "⏳ 首次运行，正在解压运行环境 (约 3-5 分钟，请耐心等待)..."
    mkdir -p "$ENV_DIR"
    
    # 解压
    tar -xzf "$ENV_TAR" -C "$ENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ 解压失败，请检查磁盘空间或压缩包完整性。"
        exit 1
    fi

    echo "🔧 正在修复环境路径链接 (conda-unpack)..."
    # 运行 conda-unpack 修复硬编码路径
    "$ENV_DIR/bin/conda-unpack"
    
    if [ $? -ne 0 ]; then
        echo "❌ 环境修复失败。"
        exit 1
    fi
    echo "✅ 环境准备完成！"
else
    echo "✅ 检测到已有环境，跳过解压步骤。"
fi

# 2. 激活环境
echo "🚀 正在激活 Python 环境..."
source "$ENV_DIR/bin/activate"

# 3. 进入服务代码目录 (关键步骤！)
# 你的代码中 model_path 使用了相对路径 "./Qwen3-TTS-12Hz-0.6B-Base"
# 所以必须 cd 到 qwen3-tts 目录下运行
if [ ! -d "$SERVICE_SUBDIR" ]; then
    echo "❌ 错误：找不到服务目录 $SERVICE_SUBDIR"
    exit 1
fi

cd "$SERVICE_SUBDIR"
echo "📂 已切换到服务目录: $(pwd)"

# 4. 检查模型目录是否存在 (预防性检查)
MODEL_DIR="./Qwen3-TTS-12Hz-0.6B-Base"
if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  警告：未找到模型目录 $MODEL_DIR"
    echo "   请确保 models 文件夹中的内容已正确解压或移动到此处。"
    echo "   程序即将启动，可能会因找不到模型而报错..."
fi

# 5. 启动 HTTP 服务
echo "----------------------------------------"
echo "🌐 正在启动 Qwen3-TTS HTTP 服务..."
echo "   监听地址: http://127.0.0.1:8011"
echo "   健康检查: http://127.0.0.1:8011/health"
echo "----------------------------------------"
echo "日志输出如下 (按 Ctrl+C 停止服务):"
echo ""

# 启动命令
# 直接运行 tts_service_qwen3tts.py，它内部会启动 uvicorn
python tts_service_qwen3tts.py

# 6. 服务停止后的提示
echo ""
echo "🛑 服务已停止。"
echo "按任意键关闭此窗口..."
read -n 1