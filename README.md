# LocalTTS
方便快速的在本地部署好一套开源的TTS引擎，并启动http服务


本项目缘起：
我是一个音频博主，为了方便把一些电子书制作为音频文件，为此我开发了一款app，叫TTSMate。
本来仅支持Azure TTS服务的批量调用，后来由于上架审核的原因，就增加了Apple TTS引擎的支持。
这时候就想能不能支持开源TTS的调用呢，于是就选择了Qwen3-TTS和Piper两款开源TTS。
本项目的目的是，能够让用户尽快的利用开源引擎在自己的机器上部署一套可用的TTS引擎。


输出物：
每种TTS引擎输出一个压缩包，用户解压后运行command文件即可启动TTS引擎。
默认地址和端口是：http://127.0.0.1:8011

目前先支持在macos上的部署



声明：
本项目是基于以下两个项目：
https://github.com/OHF-Voice/piper1-gpl
https://github.com/QwenLM/Qwen3-TTS
