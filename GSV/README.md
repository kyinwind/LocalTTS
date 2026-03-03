# gsv TTS Mac 版 - 使用指南

## 系统要求
- macOS 10.15 或更高版本
- 至少 5GB 可用磁盘空间
- Intel 或 Apple Silicon (M1/M2/M3) Mac

## 快速开始

### 方法一：一键启动（推荐）
1. 解压 `gsv-tts-mac-release.zip`
2. 双击 `setup_and_run.command`
3. 首次运行会自动解压环境（约1-2分钟）
4. 之后每次双击即可启动

### 方法二：终端启动
```bash
cd /path/to/gsv-tts-release
./setup_and_run.command
```

### 调用服务器方法
（使用 TTSMate可不用看，因为 TTSMate 调用已写好调用逻辑。如果是使用自己写的 clent 调用 TTS 服务，则需要参考）
调用地址:http://127.0.0.1:8011/tts/generate
入参示例：
{
  "output_dir": "outputs",
  "inputs": [
    {
      "seq": 1,
      "text": "如何看待般舟念佛法门？",
      "voice": {
        "name": "female",
        "ref_audio": "/Users/kylin/Documents/dev/qwen-tts/female.wav",
        "ref_text": "生和死，是每个人最大的两桩事情。",
        "rate": 1
      }
    },
    {
      "seq": 2,
      "text": "般舟念佛法门，也可以，什么方法都可以。",
      "voice": {
        "name": "male",
        "ref_audio": "/Users/kylin/Documents/dev/qwen-tts/male.wav",
        "ref_text": "生和死，是每个人最大的两桩事情。",
        "rate": 1
      }
    }
  ]
}


## 常见问题
Q: 提示"无法打开"或"开发者无法验证"
A: 前往 系统设置 → 隐私与安全性 → 允许打开

Q: 提示权限不足
A: 在终端执行：chmod +x setup_and_run.command

Q: 环境解压失败
A: 确保有足够磁盘空间，删除 gsv310_env 目录后重试

## 技术支持
https://github.com/kyinwind/LocalTTS

