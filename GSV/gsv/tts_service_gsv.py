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
from pydub import AudioSegment

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
import re
import threading
from typing import List

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

    def __init__(self, max_segment_length: int = 35):
        self.max_segment_length = max_segment_length

    def process(self, text: str) -> List[str]:
        text = self._basic_cleanup(text)
        text = self._normalize_symbols(text)
        text = self._remove_large_numbers(text)
        segments = self._smart_split(text)
        # 最后一步清洗，但要注意不要洗掉标点
        return [seg.strip() for seg in segments if seg.strip()]

    def _basic_cleanup(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text) # 保留 \t \n \r 如果需要，或者全部去掉
        text = re.sub(r'\s+', ' ', text)
        return text

    def _normalize_symbols(self, text: str) -> str:
        replacements = {
            "…": "……", # 省略号通常保留或转为标准
            "...": "……", 
            "——": "，", 
            "—": "，",
            "“": "\"", "”": "\"", # 或者保留中文引号，看模型支持
            "‘": "'", "’": "'",
            "（": "(", "）": ")", 
            "○": "零", "〇": "零",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _remove_large_numbers(self, text: str) -> str:
        text = re.sub(r'\d{15,}', '', text)
        text = re.sub(r'\([\d\s]{10,}\)', '', text)
        return text
    
    def _sanitize_for_gpt_sovits(self, text: str) -> str:
        text = text.replace("○", "零")
        text = text.replace("〇", "零")
        
        # 【关键修复】不再删除所有标点，只删除真正的乱码或非打印字符
        # 保留：中文、英文、数字、常见中英文标点
        # \u3000-\u303F: CJK 标点符号
        # \uFF00-\uFFEF: 半角及全角形式
        # 也可以简单写为保留特定标点集合
        keep_pattern = r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：”“‘’（）《》…—,.!?;:'\"()\[\]\s]"
        text = re.sub(keep_pattern, "", text)
        
        return text
    
    def _smart_split(self, text: str) -> List[str]:
        # 使用正则分割，保留分隔符
        # 注意：这里分割后，标点符号会作为单独的元素出现在列表中
        sentences = re.split(r'([.。！？!?,，])', text)
        
        segments = []
        current = ""
        
        for part in sentences:
            if not part: continue
            
            # 如果加上当前部分超过长度限制
            if len(current) + len(part) > self.max_segment_length:
                if current:
                    segments.append(current)
                # 关键点：如果 part 是标点，它应该依附于前一句还是作为下一句开头？
                # TTS 通常希望标点在句尾。
                # 如果 current 刚被清空，而 part 是标点，我们应该尝试把它加到下一句的开头，
                # 或者如果标点单独存在且很短，直接忽略（如果不影响语义），
                # 但最好的策略是：如果 current 满了，把标点留给下一句的开头，或者强制截断。
                
                # 简化策略：直接开始新的一句，包含当前 part
                current = part
            else:
                current += part
        
        if current:
            segments.append(current)
            
        final_segments = []
        for seg in segments:
            seg = seg.strip()
            if not seg: continue
            
            # 如果分段后仍然超长（比如一长串没有标点的文字），强制按长度切分
            if len(seg) > self.max_segment_length:
                # 强制切分，尽量不打断单词（中文按字切分即可）
                for i in range(0, len(seg), self.max_segment_length):
                    final_segments.append(seg[i:i+self.max_segment_length])
            else:
                final_segments.append(seg)
        
        # ✅ 统一清洗
        clean_segments = []
        for seg in final_segments:
            seg = self._sanitize_for_gpt_sovits(seg)
            seg = seg.strip()
            # 只有当清洗后还有内容且长度合理时才保留
            if len(seg) >= 1: 
                clean_segments.append(seg)
                
        return clean_segments

# =================数据模型 (新增模型路径字段)=================
class VoiceProfileReq(BaseModel):
    name: str
    ref_audio: str
    ref_text: str
    rate: float = 1.0
    # 【新增】允许用户指定模型路径，如果不填则使用服务端默认/当前加载的
    gpt_model_path: Optional[str] = "GPT_weights_v2/yunyang-e15.ckpt"
    sovits_model_path: Optional[str] = "SoVITS_weights_v2/yunyang_e8_s392.pth"

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

# =================核心工具函数 (修复版)=================

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
    【修复】增加严格的文件校验，确保生成的文件真实存在且大小>0
    """
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"参考音频文件不存在: {ref_audio}")

    client = None
    try:
        # 初始化客户端
        client = Client(GS_API_ENDPOINT, verbose=False) # 关闭 gradio_client 自身的冗长日志
        
        logger.info(f"📡 请求 WebUI: '{text[:20]}...' (Lang: {lang_code})")
        
        # 动态切换模型 (如果有指定)
        if gpt_model:
            logger.debug(f"🔄 切换 GPT: {gpt_model}")
            client.predict(gpt_path=gpt_model, api_name="/change_gpt_weights")
        
        if sovits_model:
            logger.debug(f"🔄 切换 SoVITS: {sovits_model}")
            client.predict(sovits_path=sovits_model, prompt_language=lang_code, text_language=lang_code, api_name="/change_sovits_weights")

        # 调用合成
        # 注意：how_to_cut 设为 "不切"，因为我们已经在 Python 层切好了
        result = client.predict(
            ref_wav_path=handle_file(ref_audio),
            prompt_text=ref_text,
            prompt_language=lang_code,
            text=text,
            text_language=lang_code,
            how_to_cut="不切", 
            top_k=15,
            top_p=1.0,
            temperature=1.0,
            ref_free=False,
            speed=rate,
            if_freeze=False,
            inp_refs=None,
            sample_steps="32",
            if_sr=False,
            pause_second=0.0,
            api_name="/get_tts_wav"
        )
        
        # 【关键修复】验证结果
        if not isinstance(result, str):
            raise Exception(f"API 返回非字符串类型: {type(result)}")
            
        if not os.path.exists(result):
            raise FileNotFoundError(f"Gradio 返回的文件路径不存在: {result}")
            
        file_size = os.path.getsize(result)
        if file_size == 0:
            raise Exception(f"Gradio 生成的文件为空 (0 bytes): {result}")

        # 复制到目标路径
        shutil.copy2(result, save_path)
        
        # 再次确认目标文件
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            raise Exception(f"文件复制后验证失败: {save_path}")
            
        logger.debug(f"💾 片段成功保存: {os.path.basename(save_path)} ({file_size} bytes)")
        return save_path

    except Exception as e:
        logger.error(f"❌ 片段生成失败: {e}")
        raise e
    finally:
        if client:
            client.close() # 显式关闭连接释放资源

def merge_audio_files(file_paths: List[str], output_path: str):
    """
    合并音频文件
    【修复】严格检查每个文件，如果有任何一个缺失或为空，直接报错，绝不降级复制第一段
    """
    if not file_paths:
        raise ValueError("没有可合并的音频文件列表")
    
    # 1. 预检查：确保所有文件都存在且有效
    valid_paths = []
    for fp in file_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"合并失败：片段文件丢失 -> {fp}")
        if os.path.getsize(fp) == 0:
            raise ValueError(f"合并失败：片段文件为空 -> {fp}")
        valid_paths.append(fp)

    if len(valid_paths) == 1:
        logger.info("🔗 只有一个片段，直接复制")
        shutil.copy2(valid_paths[0], output_path)
        return

    logger.info(f"🔗 开始合并 {len(valid_paths)} 个片段...")
    
    # try:
    #     from pydub import AudioSegment
    # except ImportError:
    #     # 如果没有 pydub，这是一个严重错误，不能简单复制第一段，否则用户会以为合并成功了
    #     raise ImportError("❌ 合并失败: 缺少 'pydub' 库且片段数 > 1。请运行: pip install pydub ffmpeg-python")

    try:
        combined = AudioSegment.empty()
        for i, fp in enumerate(valid_paths):
            logger.debug(f"   - 加载片段 {i+1}/{len(valid_paths)}: {os.path.basename(fp)}")
            sound = AudioSegment.from_wav(fp)
            combined += sound
        
        # 统一导出格式
        logger.debug("   - 正在导出最终文件...")
        combined = combined.set_frame_rate(44100).set_sample_width(2)
        combined.export(output_path, format="wav")
        
        # 最终验证
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("合并后的文件生成失败或为空")
            
        logger.info(f"✅ 合并完成: {os.path.basename(output_path)} ({os.path.getsize(output_path)} bytes)")
        
    except Exception as e:
        logger.error(f"❌ 合并过程发生严重错误: {e}")
        # 这里不再静默降级，而是直接抛出，让用户知道合并失败了
        raise Exception(f"音频合并失败: {str(e)}")


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
                
                logger.info(f"🎤 生成片段 {i+1}/{len(all_segments)}: '{seg['text']}'")
                
                # 确定语言 (简单判断)
                lang = "中英混合"
                # if not any('\u4e00' <= c <= '\u9fff' for c in seg['text']):
                #     lang = "英文" # 简单 fallback，实际可更复杂

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