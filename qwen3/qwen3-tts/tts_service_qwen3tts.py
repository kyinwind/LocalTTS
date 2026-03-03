import os
import re
import uuid
import threading
from typing import List
import time
import gc

import torch
import soundfile as sf
import numpy as np
import librosa
import datetime
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel

# =========================================================
# 工具函数
# =========================================================
def print_with_timestamp(message, show_full_datetime=True):
    fmt = "%Y-%m-%d %H:%M:%S" if show_full_datetime else "%H:%M:%S"
    print(f"[{datetime.datetime.now().strftime(fmt)}] {message}")

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
# 全局缓存 prompt
# =========================================================
_VOICE_PROMPTS = {}  # key: voice.name, value: voice_clone_prompt

# =========================================================
# TTS 核心
# =========================================================

def get_or_create_prompt(model: Qwen3TTSModel, voice: VoiceProfile, xvec_only=False):
    """
    获取或生成 voice_clone_prompt
    """
    key = f"{voice.name}_{xvec_only}"
    if key not in _VOICE_PROMPTS:
        print(f"✨ 生成 prompt: {voice.name}")
        prompt = model.create_voice_clone_prompt(
            ref_audio=voice.ref_audio,
            ref_text=voice.ref_text,
            x_vector_only_mode=xvec_only
        )
        _VOICE_PROMPTS[key] = prompt
    else:
        print("直接使用缓存的 prompt")
    return _VOICE_PROMPTS[key]
# =========================================================
# 文本切分
# =========================================================

def split_text_for_tts(text: str, min_len: int = 80, max_len: int = 260) -> List[str]:
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
# 模型单例
# =========================================================

_MODEL = None
_TTS_LOCK = threading.Lock()

def get_model():
    global _MODEL
    if _MODEL is None:
        model_path = os.path.abspath("./Qwen3-TTS-12Hz-0.6B-Base")
        print(f"📂 Loading model: {model_path}")

        _MODEL = Qwen3TTSModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float32,
            #dtype=torch.bfloat16,
            attn_implementation="eager"
        )

        print("🔥 Warming up model...")
        try:
            _MODEL.generate_voice_clone(
                text="你好",
                language="chinese",
                ref_audio="ref.wav",
                ref_text="你好",
                x_vector_only_mode=True,
            )
        except Exception:
            pass

        print("✅ 模型准备好了！")

    return _MODEL


def print_with_timestamp(message, show_full_datetime=True):
    """
    打印带时间戳的日志信息
    
    Args:
        message (str): 要打印的提示文本
        show_full_datetime (bool): 是否显示完整日期时间，True显示"年-月-日 时:分:秒"，False仅显示"时:分:秒"
    """
    # 根据参数选择时间格式
    if show_full_datetime:
        time_format = "%Y-%m-%d %H:%M:%S"
    else:
        time_format = "%H:%M:%S"
    
    # 获取并格式化当前时间
    current_time = datetime.datetime.now().strftime(time_format)
    # 拼接时间戳和消息并打印
    print(f"[{current_time}] {message}")

# =========================================================
# TTS 核心
# =========================================================

def tts_generate(inputs: List[TTSInput], output_dir: str) -> str:
    # 获取当前时间并格式化为 年-月-日 时:分:秒
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 进入 TTS 生成阶段，正在处理文本切分和语音合成...")
    if not inputs:
        raise ValueError("inputs 不能为空")

    os.makedirs(output_dir, exist_ok=True)
    
    
    inputs = sorted(inputs, key=lambda x: x.seq)

    all_audio = []
    sample_rate = None
    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )
    for item in inputs:
        model = get_model()
        print("拆分文本")
        #segments = split_text_for_tts(item.text)
        # 获取 prompt（提前抽特征）
        voice_prompt = get_or_create_prompt(model, item.voice, xvec_only=False)

        #for idx, seg in enumerate(segments):
        print_with_timestamp(f"开始调用模型生成语音generate_voice_clone，\n文本: {item.text}")

        wavs, sr = model.generate_voice_clone(
            text=item.text,
            language="chinese",
            # ref_audio=item.voice.ref_audio,
            # ref_text=item.voice.ref_text,
            voice_clone_prompt=voice_prompt,
            x_vector_only_mode=True,
            **common_gen_kwargs,
        )
        # wav = wavs[0].detach().cpu().numpy()
        if not wavs:
            continue

        wav = wavs[0]
        sample_rate = sr
        
        # 保存每段 seg 音频到临时文件，方便排查
        seq_filename = f"tmp_{item.voice.name}_seq{item.seq}.wav"
        seq_path = os.path.join(output_dir, seq_filename)
        sf.write(seq_path, wav, sample_rate)

        if item.voice.rate != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=item.voice.rate)

        all_audio.append(wav)

    
    valid_audio = [a for a in all_audio if a is not None and len(a) > 0]
    if not valid_audio:
        raise RuntimeError("模型未生成任何音频")

    final_audio = np.concatenate(valid_audio, axis=0)

    filename = f"tts_mix_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, filename)
    print_with_timestamp(f"开始写入音频文件: {output_path}")
    sf.write(output_path, final_audio, sample_rate)
    # ⭐⭐ 返回绝对路径
    output_path = os.path.abspath(output_path)

    print_with_timestamp(f"最终音频绝对路径: {output_path}")
    return output_path


def run_tts_locked(inputs, output_dir):
    with _TTS_LOCK:
        return tts_generate(inputs, output_dir)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(title="Qwen TTS Service")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting TTS service...")
    get_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None
    }


import asyncio

@app.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(req: TTSRequest):
    try:
        start = time.time()
        print("📥 TTS request received")

        inputs = [
            TTSInput(
                seq=item.seq,
                text=item.text,
                voice=VoiceProfile(
                    name=item.voice.name,
                    ref_audio=item.voice.ref_audio,
                    ref_text=item.voice.ref_text,
                    rate=item.voice.rate,
                )
            )
            for item in req.inputs
        ]

        loop = asyncio.get_running_loop()

        output_path = await loop.run_in_executor(
            None,
            run_tts_locked,
            inputs,
            req.output_dir
        )


        print(f"✅ TTS finished, cost {time.time() - start:.1f}s")

        return TTSResponse(success=True, output_path=output_path)

    except Exception as e:
        print("❌ TTS error:", e)
        return TTSResponse(success=False, output_path=str(e))


def run_tts_locked(inputs, output_dir):
    with _TTS_LOCK:
        return tts_generate(inputs, output_dir)
    
if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Qwen3 TTS HTTP service...")
    uvicorn.run(
        app,                   # 直接传 app 对象
        host="127.0.0.1",
        port=8011,
        reload=False,
        workers=1
    )
