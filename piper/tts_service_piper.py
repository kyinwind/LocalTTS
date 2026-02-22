
import os
import re
import uuid
import threading
import json
from typing import List, Optional
import time
import datetime
from dataclasses import dataclass
import asyncio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import sys

# ==================== PyInstaller 资源路径 ====================
def get_resource_path(relative_path: str) -> str:
    """获取 PyInstaller 打包后的资源路径"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)
from piper.voice import PiperVoice
import soundfile as sf
import librosa
from pydub import AudioSegment
os.environ["OMP_NUM_THREADS"] = "1"

# =========================================================
# 数据结构
# =========================================================
@dataclass
class VoiceProfile:
    name: str
    ref_audio: str
    ref_text: str
    rate: float = 1.0

@dataclass
class TTSInput:
    seq: int
    text: str
    voice: VoiceProfile

# =========================================================
# HTTP 模型
# =========================================================
class VoiceProfileReq(BaseModel):
    name: str
    ref_audio: str
    ref_text: str
    rate: float = 1.0

class TTSInputReq(BaseModel):
    seq: int
    text: str
    voice: VoiceProfileReq

class TTSRequest(BaseModel):
    inputs: List[TTSInputReq]
    output_dir: str = "outputs"

class TTSResponse(BaseModel):
    success: bool
    output_path: str

# =========================================================
# Piper 配置
# =========================================================
#MODELS_DIR = "./models"
# 新写法
MODELS_DIR = get_resource_path("models")
G2PW_DIR = get_resource_path("g2pW")

VOICE_TO_MODEL = {
    "female": {
        "model": "zh_CN-xiao_ya-medium.onnx",
        "config": "zh_CN-xiao_ya-medium.onnx.json"
    },
    "male": {
        "model": "zh_CN-chaowen-medium.onnx",
        "config": "zh_CN-chaowen-medium.onnx.json"
    },
    "female2": {
        "model": "zh_CN-huayan-medium.onnx",
        "config": "zh_CN-huayan-medium.onnx.json"
    }
}

# 全局缓存 & 锁
VOICE_CACHE = {}
_TTS_LOCK = threading.Lock()

# =========================================================
# 工具函数
# =========================================================
def print_with_timestamp(message, show_full_datetime=True):
    fmt = "%Y-%m-%d %H:%M:%S" if show_full_datetime else "%H:%M:%S"
    print(f"[{datetime.datetime.now().strftime(fmt)}] {message}")

def split_text_for_tts(text: str, min_len: int = 80, max_len: int = 260) -> List[str]:
    """文本切分"""
    text = re.sub(r'\s+', ' ', text).strip()
    rough_segments = re.split(r'(?<=[。！？!?])', text)
    
    segments, buffer = [], ""
    for seg in rough_segments:
        if not seg.strip():
            continue
        if len(buffer) + len(seg) <= max_len:
            buffer += seg
        else:
            if buffer:
                segments.append(buffer.strip())
            buffer = seg
        if len(buffer) >= min_len:
            segments.append(buffer.strip())
            buffer = ""
    if buffer:
        segments.append(buffer.strip())
    return segments

# =========================================================
# Piper 核心调用（完全适配 AudioChunk 官方规范）
# =========================================================
def load_piper_voice(voice_name: str) -> PiperVoice:
    """加载 Piper 语音模型"""
    if voice_name in VOICE_CACHE:
        return VOICE_CACHE[voice_name]
    
    model_info = VOICE_TO_MODEL.get(voice_name)
    if not model_info:
        raise ValueError(f"不支持的音色: {voice_name}，可选：{list(VOICE_TO_MODEL.keys())}")
    
    # 新写法
    model_path = get_resource_path(os.path.join("models", model_info["model"]))
    config_path = get_resource_path(os.path.join("models", model_info["config"]))
    print_with_timestamp(f"加载模型：{model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 确保 Piper 内部能访问 g2pw 目录
    os.environ["PIPER_G2PW_PATH"] = G2PW_DIR
    try:
        voice = PiperVoice.load(model_path, config_path)
        VOICE_CACHE[voice_name] = voice
        print_with_timestamp(f"✅ 模型加载成功：{voice_name} (采样率: {voice.config.sample_rate})")
        return voice
    except Exception as e:
        print_with_timestamp(f"❌ 模型加载失败：{str(e)}")
        raise

def synthesize_with_piper(text: str, voice_name: str, output_wav: str) -> bool:
    try:
        voice = load_piper_voice(voice_name)

        text = text.strip()
        if not text:
            raise ValueError("空文本无法合成")

        sample_rate = voice.config.sample_rate
        audio_stream = voice.synthesize(text)

        chunks = []

        for chunk in audio_stream:
            # ✅ 正确字段
            if hasattr(chunk, "audio_int16_array") and chunk.audio_int16_array is not None:
                chunks.append(chunk.audio_int16_array)

        if not chunks:
            raise ValueError("未生成任何有效音频数据")

        full_audio = np.concatenate(chunks)

        sf.write(
            output_wav,
            full_audio,
            sample_rate,
            format="WAV",
            subtype="PCM_16"
        )

        print("✅ 合成成功")
        return True

    except Exception as e:
        print("❌ 合成失败:", e)
        return False

# =========================================================
# TTS 核心逻辑
# =========================================================
def tts_generate(inputs: List[TTSInput], output_dir: str) -> str:
    print_with_timestamp("进入 TTS 生成阶段，处理文本切分和合成...")
    print("inputs =", inputs)          # 直接打印
    if not inputs:
        raise ValueError("输入数据不能为空")
    
    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(inputs, key=lambda x: x.seq)
    
    all_audio_paths = []
    sample_rate = None

    for item in inputs:
        voice_name = item.voice.name.lower()
        segments = split_text_for_tts(item.text)
        
        for seg in segments:
            if not seg.strip():
                continue
            
            tmp_wav = os.path.join(output_dir, f"tmp_{uuid.uuid4().hex}.wav")
            print_with_timestamp(f"调用 Piper 生成语音: {seg[:30]}...")
            
            if not synthesize_with_piper(seg, voice_name, tmp_wav):
                continue
            
            # 变速处理
            if item.voice.rate != 1.0:
                voice = load_piper_voice(voice_name)
                y, sr = librosa.load(tmp_wav, sr=voice.config.sample_rate)
                y_stretched = librosa.effects.time_stretch(y, rate=item.voice.rate)
                sf.write(tmp_wav, y_stretched, sr, format="WAV", subtype="PCM_16")
                sample_rate = sr
            elif sample_rate is None:
                voice = load_piper_voice(voice_name)
                sample_rate = voice.config.sample_rate
            
            all_audio_paths.append(tmp_wav)
    
    if not all_audio_paths:
        raise RuntimeError("未生成任何有效音频")
    
    # 拼接音频
    combined = AudioSegment.empty()
    for path in all_audio_paths:
        try:
            audio = AudioSegment.from_wav(path)
            combined += audio
        except Exception as e:
            print_with_timestamp(f"⚠️ 音频拼接失败 {path}：{str(e)}")
            continue
    
    # 导出最终文件
    output_filename = f"tts_mix_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, output_filename)
    combined.export(output_path, format="wav", codec="pcm_s16le")
    
    # 清理临时文件
    for path in all_audio_paths:
        try:
            os.remove(path)
        except:
            pass
    
    output_path = os.path.abspath(output_path)
    print_with_timestamp(f"最终音频路径: {output_path}")
    return output_path

def run_tts_locked(inputs, output_dir):
    with _TTS_LOCK:
        return tts_generate(inputs, output_dir)

# =========================================================
# FastAPI 接口
# =========================================================
app = FastAPI(title="Piper TTS Service (v1.4.1 完美适配版)")

@app.on_event("startup")
def startup_event():
    print_with_timestamp("🚀 启动 Piper TTS 服务 (完美适配版)")
    if not os.path.exists(MODELS_DIR):
        raise RuntimeError(f"模型目录不存在: {MODELS_DIR}")
    if VOICE_TO_MODEL:
        first_voice = list(VOICE_TO_MODEL.keys())[0]
        try:
            load_piper_voice(first_voice)
            print_with_timestamp(f"✅ 预加载模型成功：{first_voice}")
        except Exception as e:
            print_with_timestamp(f"⚠️ 预加载模型失败：{first_voice} - {str(e)}")

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "piper_version": getattr(__import__("piper"), "__version__", "1.4.1"),
        "models_available": len(VOICE_TO_MODEL),
        "cached_voices": len(VOICE_CACHE)
    }

@app.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(req: TTSRequest):
    """TTS 生成接口"""
    try:
        start_time = time.time()
        print_with_timestamp("📥 收到 TTS 请求")
        
        inputs = [
            TTSInput(
                seq=item.seq,
                text=item.text,
                voice=VoiceProfile(
                    name=item.voice.name,
                    ref_audio=item.voice.ref_audio,
                    ref_text=item.voice.ref_text,
                    rate=item.voice.rate
                )
            ) for item in req.inputs
        ]
        
        loop = asyncio.get_running_loop()
        output_path = await loop.run_in_executor(None, run_tts_locked, inputs, req.output_dir)
        
        cost = round(time.time() - start_time, 1)
        print_with_timestamp(f"✅ TTS 生成完成，耗时 {cost}s")
        return TTSResponse(success=True, output_path=output_path)
    
    except Exception as e:
        error_msg = str(e)
        print_with_timestamp(f"❌ TTS 错误：{error_msg}")
        return TTSResponse(success=False, output_path=error_msg)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Piper TTS HTTP service...")
    uvicorn.run(
        app,                   # 直接传 app 对象
        host="127.0.0.1",
        port=8011,
        reload=False,
        workers=1
    )