# test_tts.py
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForTextToSpeech

text = "你好，我是 TTSMate 的新语音引擎。这段文字将被转换成自然流畅的语音。"
output_path = "output.wav"

model_id = "Qwen/Qwen3-TTS-0.6B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToSpeech.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

if torch.backends.mps.is_available():
    model.to("mps")

# 生成语音
inputs = processor(text=text, return_tensors="pt")
if torch.backends.mps.is_available():
    inputs = {k: v.to("mps") for k, v in inputs.items()}

with torch.no_grad():
    output = model(**inputs).waveform

# 保存为 WAV
waveform = output.squeeze().cpu().numpy()
sf.write(output_path, waveform, samplerate=24000)

print(f"✅ 音频已保存至: {output_path}")