#!/bin/bash

cd "$(dirname "$0")"

echo "📍 Working dir: $(pwd)"

PYTHON="/opt/miniconda3/envs/qwen3-tts/bin/python"
echo "🐍 Python: $PYTHON"

echo "🚀 Starting TTS service (preventing sleep)..."

# 使用 caffeinate 防止系统睡眠
caffeinate -i $PYTHON -m uvicorn tts_service:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 1800 \
  --timeout-graceful-shutdown 1800

echo ""
echo "🛑 TTS service stopped. Press Enter to close."
read
