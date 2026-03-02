
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
import torch
import torch.multiprocessing as mp
import gc
mp.set_start_method("fork", force=True)
# ==================== PyInstaller 资源路径 ====================
def get_resource_path(relative_path: str) -> str:
    """获取 PyInstaller 打包后的资源路径"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)
from piper.voice import PiperVoice
import soundfile as sf
from scipy.signal import resample

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import re
import threading
from typing import List


class NovelTTSPreprocessor:
    """
    工业级中文小说 TTS 文本预处理器
    线程安全单例模式
    """

    _instance = None
    _lock = threading.Lock()

    # ===============================
    # 单例入口
    # ===============================
    @classmethod
    def getProcessor(
        cls,
        max_segment_length: int = 250,
        remove_english_parentheses: bool = True,
        strict_mode: bool = True,
    ):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        max_segment_length,
                        remove_english_parentheses,
                        strict_mode,
                    )
        return cls._instance

    # ===============================
    # 初始化
    # ===============================
    def __init__(
        self,
        max_segment_length: int,
        remove_english_parentheses: bool,
        strict_mode: bool,
    ):
        self.max_segment_length = max_segment_length
        self.remove_english_parentheses = remove_english_parentheses
        self.strict_mode = strict_mode

    # ===============================
    # 可选：运行时更新配置
    # ===============================
    def update_config(
        self,
        max_segment_length: int = None,
        remove_english_parentheses: bool = None,
        strict_mode: bool = None,
    ):
        if max_segment_length is not None:
            self.max_segment_length = max_segment_length
        if remove_english_parentheses is not None:
            self.remove_english_parentheses = remove_english_parentheses
        if strict_mode is not None:
            self.strict_mode = strict_mode

    # ===============================
    # 对外入口
    # ===============================
    def process(self, text: str) -> List[str]:
        text = self._basic_cleanup(text)
        text = self._normalize_symbols(text)
        text = self._fix_number_spacing(text)
        text = self._remove_large_numbers(text)

        if self.remove_english_parentheses:
            text = self._remove_english_parentheses(text)

        if self.strict_mode:
            text = self._remove_long_english_blocks(text)

        segments = self._smart_split(text)
        return [seg.strip() for seg in segments if seg.strip()]

    # ===============================
    # 基础清理
    # ===============================
    def _basic_cleanup(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # ===============================
    # 标点归一化
    # ===============================
    def _normalize_symbols(self, text: str) -> str:
        replacements = {
            "…": "。",
            "...": "。",
            "——": "，",
            "—": "，",
            "“": "",
            "”": "",
            "‘": "",
            "’": "",
            "（": "(",
            "）": ")",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    # ===============================
    # 修复数字空格
    # ===============================
    def _fix_number_spacing(self, text: str) -> str:
        return re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # ===============================
    # 删除包含英文的括号
    # ===============================
    def _remove_english_parentheses(self, text: str) -> str:
        return re.sub(r'\([^)]*[A-Za-z][^)]*\)', '', text)

    # ===============================
    # 删除超长英文块
    # ===============================
    def _remove_long_english_blocks(self, text: str) -> str:
        return re.sub(r'[A-Za-z]{15,}', '', text)
    
    # ===============================
    # 删除超长数字（防止 g2p 崩溃）
    # ===============================
    def _remove_large_numbers(self, text: str) -> str:
        # 删除 15 位以上的连续数字
        text = re.sub(r'\d{15,}', '', text)

        # 删除括号里超长数字（包括有空格的）
        text = re.sub(r'\([\d\s]{10,}\)', '', text)

        return text
    # ===============================
    # 智能分段
    # ===============================
    def _smart_split(self, text: str) -> List[str]:
        sentences = re.split(r'([。！？!?])', text)
        segments = []
        current = ""

        for part in sentences:
            if len(current) + len(part) > self.max_segment_length:
                if current:
                    segments.append(current)
                current = part
            else:
                current += part

        if current:
            segments.append(current)

        # 二次安全切分
        final_segments = []
        for seg in segments:
            if len(seg) <= self.max_segment_length:
                final_segments.append(seg)
            else:
                for i in range(0, len(seg), self.max_segment_length):
                    final_segments.append(seg[i:i+self.max_segment_length])

        return final_segments

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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
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

def synthesize_to_numpy(text: str, voice_name: str) -> tuple[np.ndarray, int]:
    voice = load_piper_voice(voice_name)

    text = text.strip()
    if not text:
        raise ValueError("空文本无法合成")

    sample_rate = voice.config.sample_rate
    audio_stream = voice.synthesize(text)

    chunks = []
    for chunk in audio_stream:
        if hasattr(chunk, "audio_int16_array") and chunk.audio_int16_array is not None:
            chunks.append(chunk.audio_int16_array)

    if not chunks:
        raise ValueError("未生成任何音频数据")

    full_audio = np.concatenate(chunks)
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return full_audio, sample_rate

def change_speed(audio: np.ndarray, rate: float) -> np.ndarray:
    if rate == 1.0:
        return audio

    new_length = int(len(audio) / rate)
    indices = np.linspace(0, len(audio) - 1, new_length).astype(np.int32)
    return audio[indices]

# =========================================================
# TTS 核心逻辑
# =========================================================
def tts_generate(inputs: List[TTSInput], output_dir: str) -> str:
    print_with_timestamp("进入 TTS 生成阶段...")

    if not inputs:
        raise ValueError("输入数据不能为空")
    
    # 预处理器（适配小说文本，优化 Piper 表现）
    processor = NovelTTSPreprocessor.getProcessor()

    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(inputs, key=lambda x: x.seq)

    all_audio_arrays = []
    sample_rate = None

    for item in inputs:
        voice_name = item.voice.name.lower()
        #segments = split_text_for_tts(item.text)
        segments = processor.process(item.text)

        for seg in segments:
            if not seg.strip():
                continue

            print_with_timestamp(f"合成片段: {seg[:30]}...")

            audio_np, sr = synthesize_to_numpy(seg, voice_name)

            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise RuntimeError("不同音色采样率不一致")

            # 变速
            audio_np = change_speed(audio_np, item.voice.rate)

            all_audio_arrays.append(audio_np)

    if not all_audio_arrays:
        raise RuntimeError("未生成任何有效音频")

    # 🔥 直接内存拼接
    final_audio = np.concatenate(all_audio_arrays)

    output_filename = f"tts_mix_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, output_filename)

    sf.write(
        output_path,
        final_audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16"
    )

    output_path = os.path.abspath(output_path)
    print_with_timestamp(f"最终音频路径: {output_path}")

    return output_path

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


def run_tts_locked(inputs, output_dir):
    with _TTS_LOCK:
        return tts_generate(inputs, output_dir)
    
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