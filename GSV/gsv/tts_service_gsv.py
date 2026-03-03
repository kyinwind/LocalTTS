import os
import re
import uuid
import threading
import time
import datetime
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import logging
import shutil
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
# 仅保留桥接需要的轻量级库
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 【新增】导入 gradio_client
from gradio_client import Client, handle_file

# ================= 新增：初始化 Logger =================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
# =====================================================

# =================配置区域=================
# 【重要修改】默认地址改为 WebUI 的 Gradio API 端口 (通常是 9872)
# 如果你还在用 9880，请改回 9880，但 9880 不支持动态换模型
GS_BASE_URL = os.getenv("GS_BASE_URL", "http://127.0.0.1:9872")
GS_API_ENDPOINT = GS_BASE_URL 

# 使用绝对路径作为临时目录
BASE_DIR = Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp_gs_outputs"
TEMP_DIR.mkdir(exist_ok=True)

# 全局锁
_TTS_LOCK = threading.Lock()

app = FastAPI(title="GPT-SoVITS Bridge Server (WebUI Mode)")

# =================文本预处理逻辑 (保持不变)=================
class NovelTTSPreprocessor:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_processor(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, max_segment_length: int = 250):
        self.max_segment_length = max_segment_length

    def process(self, text: str) -> List[str]:
        text = self._basic_cleanup(text)
        text = self._normalize_symbols(text)
        text = self._remove_large_numbers(text)
        segments = self._smart_split(text)
        return [seg.strip() for seg in segments if seg.strip()]

    def _basic_cleanup(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _normalize_symbols(self, text: str) -> str:
        replacements = {
            "…": "。", "...": "。", "——": "，", "—": "，",
            "“": "", "”": "", "‘": "", "’": "",
            "（": "(", "）": ")",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _remove_large_numbers(self, text: str) -> str:
        text = re.sub(r'\d{15,}', '', text)
        text = re.sub(r'\([\d\s]{10,}\)', '', text)
        return text

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
        final_segments = []
        for seg in segments:
            if len(seg) <= self.max_segment_length:
                final_segments.append(seg)
            else:
                for i in range(0, len(seg), self.max_segment_length):
                    final_segments.append(seg[i:i+self.max_segment_length])
        return final_segments

# =================数据模型 (新增模型路径字段)=================
class VoiceProfileReq(BaseModel):
    name: str
    ref_audio: str
    ref_text: str
    rate: float = 1.0
    # 【新增】允许用户指定模型路径，如果不填则使用服务端默认/当前加载的
    gpt_model_path: Optional[str] = None
    sovits_model_path: Optional[str] = None

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

# =================核心工具函数=================

def call_gs_single_segment_webui(
    text: str, 
    ref_audio: str, 
    ref_text: str, 
    rate: float, 
    save_path: str,
    gpt_model: Optional[str] = None,
    sovits_model: Optional[str] = None,
    lang_code: str = "中英混合"
) -> str:
    """
    [同步版本] 调用 GPT-SoVITS WebUI (Gradio) API 生成音频
    支持动态切换模型
    """
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"参考音频文件不存在: {ref_audio}")

    try:
        # 初始化 Gradio 客户端 (连接一次即可，但为了线程安全，每次调用新建或需加锁，这里简单起见每次新建短连接)
        # 注意：gradio_client 内部有缓存，多次调用同一 URL 开销不大
        client = Client(GS_API_ENDPOINT)
        
        logger.info(f"📡 正在请求 WebUI API: {GS_API_ENDPOINT}")
        if gpt_model:
            logger.info(f"🔄 切换 GPT 模型: {gpt_model}")
            client.predict(gpt_path=gpt_model, api_name="/change_gpt_weights")
        
        if sovits_model:
            logger.info(f"🔄 切换 SoVITS 模型: {sovits_model}")
            client.predict(sovits_path=sovits_model, prompt_language=lang_code, text_language=lang_code, api_name="/change_sovits_weights")

        # 调用合成接口 /get_tts_wav
        # 参数映射参考你提供的 API 文档
        result = client.predict(
            ref_wav_path=handle_file(ref_audio),
            prompt_text=ref_text,
            prompt_language=lang_code,
            text=text,
            text_language=lang_code,
            how_to_cut="凑四句一切", # 或者 "不切"，因为我们在外部已经切好了
            top_k=15,
            top_p=1.0,
            temperature=1.0,
            ref_free=False,
            speed=rate,
            if_freeze=False,
            inp_refs=None, # 多参考音频暂不支持，留空
            sample_steps="32", # 高质量设为 32 或 16
            if_sr=False,
            pause_second=0.0, # 外部已分段，这里不需要额外停顿
            api_name="/get_tts_wav"
        )
        
        # result 通常是一个文件路径字符串 (Gradio 生成的临时文件路径)
        # 例如：'/tmp/gradio/...'
        if isinstance(result, str) and os.path.exists(result):
            # 将 Gradio 生成的临时文件复制到我们指定的 save_path
            shutil.copy2(result, save_path)
            logger.info(f"💾 成功保存片段: {save_path}")
            return save_path
        else:
            raise Exception(f"Gradio API 返回无效路径: {result}")

    except Exception as e:
        logger.error(f"❌ WebUI 调用异常: {e}")
        raise e

def merge_audio_files(file_paths: List[str], output_path: str):
    """合并音频文件"""
    if not file_paths:
        raise ValueError("No audio files to merge")
    
    if len(file_paths) == 1:
        shutil.copy(file_paths[0], output_path)
        return

    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing segment: {fp}")
            sound = AudioSegment.from_wav(fp)
            combined += sound
        
        combined = combined.set_frame_rate(44100).set_sample_width(2)
        combined.export(output_path, format="wav")
    except ImportError:
        logger.warning("⚠️ Warning: pydub not found. Falling back to copying first segment only.")
        shutil.copy(file_paths[0], output_path)
    except Exception as e:
        logger.error(f"❌ Merge failed: {e}")
        shutil.copy(file_paths[0], output_path)

# =================API 路由=================
@app.get("/health")
def health_check():
    return {"status": "ok", "gs_url": GS_BASE_URL, "mode": "WebUI_Gradio"}

@app.post("/tts/generate", response_model=TTSResponse)
def generate_tts(request: TTSRequest):
    generated_files = []
    task_id = str(uuid.uuid4())[:8]
    final_output_filename = f"tts_{task_id}.wav"
    
    if request.output_dir:
        out_dir = Path(request.output_dir).resolve()
    else:
        out_dir = (BASE_DIR / "outputs").resolve()
        
    out_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = (out_dir / final_output_filename).resolve()

    with _TTS_LOCK:
        try:
            preprocessor = NovelTTSPreprocessor.get_processor()
            all_segments = []

            logger.info(f"📥 开始处理任务 (Seq count: {len(request.inputs)})")
            logger.info(f"📂 输出目标绝对路径: {final_output_path}")

            # 1. 预处理与分段
            for item in request.inputs:
                segments = preprocessor.process(item.text)
                for idx, seg_text in enumerate(segments):
                    all_segments.append({
                        "seq": item.seq,
                        "sub_idx": idx,
                        "text": seg_text,
                        "voice": item.voice
                    })
            
            if not all_segments:
                raise ValueError("No valid text segments after preprocessing")

            # 2. 逐个生成音频
            for i, seg in enumerate(all_segments):
                temp_filename = f"{task_id}_s{seg['seq']}_sub{i}.wav"
                temp_file_path = (TEMP_DIR / temp_filename).resolve()
                
                logger.info(f"🎤 生成片段 {i+1}/{len(all_segments)}: '{seg['text'][:20]}...'")
                
                # 确定语言 (简单判断)
                lang = "中文"
                if not any('\u4e00' <= c <= '\u9fff' for c in seg['text']):
                    lang = "英文" # 简单 fallback，实际可更复杂

                # 【关键】调用新的 WebUI 函数，传入模型路径
                path = call_gs_single_segment_webui(
                    text=seg['text'],
                    ref_audio=seg['voice'].ref_audio,
                    ref_text=seg['voice'].ref_text,
                    rate=seg['voice'].rate,
                    save_path=str(temp_file_path),
                    gpt_model=seg['voice'].gpt_model_path,
                    sovits_model=seg['voice'].sovits_model_path,
                    lang_code=lang
                )
                generated_files.append(path)

            # 3. 合并
            logger.info("🔗 正在合并音频...")
            merge_audio_files(generated_files, str(final_output_path))

            logger.info(f"✅ 任务完成: {final_output_path}")
            
            return TTSResponse(success=True, output_path=str(final_output_path))

        except Exception as e:
            logger.error(f"💥 任务失败: {str(e)}")
            for fp in generated_files:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except Exception as clean_err:
                        logger.warning(f"清理临时文件失败 {fp}: {clean_err}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GPT-SoVITS Bridge (WebUI Mode) on port 8011...")
    uvicorn.run(app, host="127.0.0.1", port=8011, workers=1)