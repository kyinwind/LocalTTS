#!/bin/bash

# ============================================
# gsv TTS 一键启动脚本 (Mac)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔍 检查环境..."

# 检查是否已解压环境
ENV_DIR="$SCRIPT_DIR/gsv_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "⏳ 首次运行，正在解压环境（约需1-2分钟）..."
    
    # 检查压缩包是否存在
    if [ ! -f "gsv_env.tar.gz" ]; then
        echo "❌ 错误：找不到 gsv_env.tar.gz"
        exit 1
    fi
    
    # 创建环境目录并解压
    mkdir -p "$ENV_DIR"
    tar -xzf gsv_env.tar.gz -C "$ENV_DIR"
    
    # 运行 conda-unpack 修复路径
    echo "🔧 修复环境路径..."
    "$ENV_DIR/bin/conda-unpack"
    
    echo "✅ 环境解压完成！"
fi

# 激活环境
echo "🚀 启动 GSV TTS 引擎..."
source "$ENV_DIR/bin/activate"

# 进入 piper 目录
cd "$SCRIPT_DIR/gsv"

# 运行你的服务
python tts_service_gsv.py

# 保持窗口（可选）
echo ""
echo "按任意键退出..."
read -n 1