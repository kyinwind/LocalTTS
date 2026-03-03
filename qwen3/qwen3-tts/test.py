import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    model_path = os.path.abspath("./Qwen3-TTS-12Hz-0.6B-Base")
    print(f"📂 加载本地模型: {model_path}")

    # ⚠️ Mac 必须 float32
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32
    )

    # ⚠️ 防止MPS采样NaN
    model.model.generation_config.use_cache = False

    ref_audio = "male.wav"
    ref_text  = "生和死，是每个人最大的两桩事情。"

    wavs, sr = model.generate_voice_clone(
        text="用一个简单的词就可以描写屏幕上的景色：“废墟”。我们看到的街道条条都是乱七八糟。街上散布着一些“土墩”样的东西，一般是一个接着一个。有些离街道远一些，而另一些停在大楼的路中央。在几乎没有察觉到的情况下，摄像机的焦距越来越大。我很快就明白了，这些“土墩”可能原本是一些运载工具—一些在形状上多少像平地船的运输工具。",
        language="chinese",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    sf.write("output_voice_clone.wav", wavs[0], sr)

if __name__ == "__main__":
    main()
