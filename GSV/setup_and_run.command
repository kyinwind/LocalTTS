#!/bin/bash

# ============================================
# gsv TTS 一键启动脚本 (Mac - 优化版)
# ============================================

# 获取脚本所在绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🔍 [1/4] 检查环境..."

# 定义环境变量
ENV_DIR="$SCRIPT_DIR/gsv_env"
TARBALL="gsv_env.tar.gz"
APP_DIR="$SCRIPT_DIR/gsv"
PYTHON_SCRIPT="tts_service_gsv.py"

# 1. 检查压缩包是否存在
if [ ! -f "$TARBALL" ]; then
    echo "❌ 错误：找不到 $TARBALL"
    echo "💡 请确保将打包好的 .tar.gz 文件与此脚本放在同一目录。"
    exit 1
fi

# 2. 如果环境未解压，则执行解压
if [ ! -d "$ENV_DIR" ]; then
    echo "⏳ [2/4] 首次运行，正在解压环境（约需1-2分钟）..."
    
    # 创建目录
    mkdir -p "$ENV_DIR"
    
    # 解压
    if ! tar -xzf "$TARBALL" -C "$ENV_DIR"; then
        echo "❌ 错误：解压失败，可能是压缩包损坏。"
        rm -rf "$ENV_DIR" # 清理半成品的目录
        exit 1
    fi
    
    # 运行 conda-unpack 修复硬编码路径 (关键步骤！)
    echo "🔧 [3/4] 修复环境路径 (conda-unpack)..."
    if [ -f "$ENV_DIR/bin/conda-unpack" ]; then
        "$ENV_DIR/bin/conda-unpack"
        if [ $? -ne 0 ]; then
            echo "❌ 错误：conda-unpack 执行失败。"
            exit 1
        fi
    else
        echo "⚠️ 警告：未找到 conda-unpack，可能不是标准的 conda-pack 包。"
    fi
    
    echo "✅ 环境准备完成！"
else
    echo "✅ 环境已存在，跳过解压。"
fi

# 3. 激活环境
echo "🚀 [4/4] 激活环境并启动服务..."
source "$ENV_DIR/bin/activate"

# 【关键新增】显式验证 ffmpeg 是否在当前 PATH 中
# 这能提前拦截 90% 的 pydub 报错
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ 致命错误：环境激活后仍无法找到 'ffmpeg'！"
    echo "   当前 PATH: $PATH"
    echo "   请检查打包时是否真正安装了 ffmpeg (conda install -c conda-forge ffmpeg)。"
    exit 1
else
    FFMPEG_PATH=$(command -v ffmpeg)
    echo "✅ 检测到 ffmpeg: $FFMPEG_PATH"
    # 可选：打印版本确认
    # ffmpeg -version | head -n 1
fi

# 4. 进入应用目录并运行
if [ ! -d "$APP_DIR" ]; then
    echo "❌ 错误：找不到应用目录 $APP_DIR"
    exit 1
fi

cd "$APP_DIR"

# 运行服务
# 使用 exec 替换当前 shell 进程，这样 Ctrl+C 能正确终止 python 进程
exec python "$PYTHON_SCRIPT"

# 下面的代码只有在 python 异常退出后才会执行
echo ""
echo "⚠️ 服务已停止。按任意键退出..."
read -n 1