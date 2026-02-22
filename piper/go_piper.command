#!/bin/bash

cd "$(dirname "$0")"

echo "📍 Working dir: $(pwd)"
echo "🚀 Starting Piper TTS service..."

PYTHON="/opt/miniconda3/envs/piper310/bin/python"

# 防止睡眠 + 启动可执行文件
caffeinate -i $PYTHON -m uvicorn tts_service_piper:app \
  --host 127.0.0.1 \
  --port 8011 \
  --workers 1 \
  --timeout-keep-alive 1800 \
  --timeout-graceful-shutdown 1800

echo ""
echo "🛑 TTS service stopped. Press Enter to close."
read