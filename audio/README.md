# 音频功能使用指南 (ASR & TTS)

本系统现已集成语音识别（ASR）和文本转语音（TTS）功能，让视觉障碍用户可以通过语音命令指定目标，并通过语音反馈获得引导指令。

## 功能特点

### 🎤 ASR (自动语音识别)
- 使用 OpenAI Whisper 模型进行语音识别
- 支持中文和英文
- 自动解析用户指令，提取目标物体
- 支持多种命令格式（如"找到杯子"、"where is the cup"）

### 🔊 TTS (文本转语音)
- 支持两种后端：pyttsx3（离线）和 MiMo 云端（高质量）
- 默认使用 MiMo 云端 TTS（需 API Key），可切换为 pyttsx3 离线模式
- 自动播放引导指令（向左、向右、向上等）
- 可调节语速和音量

### 🎙️ 录音功能
- 支持固定时长录音
- 支持自动检测静音停止录音
- 高质量音频录制 (16kHz, 单声道)

## 安装依赖

```bash
# 安装所有音频依赖
pip install openai-whisper pyttsx3 sounddevice scipy

# 或使用 requirements.txt
pip install -r requirements.txt
```

### Windows 额外要求
- Windows 自带 SAPI5 语音引擎，无需额外安装
- 建议安装中文语音包以获得更好的中文语音效果

### macOS 额外要求
- macOS 自带语音合成引擎
- 可在"系统偏好设置 -> 辅助功能 -> 语音内容"中选择中文语音

### Linux 额外要求
```bash
# 安装 espeak（TTS 引擎）
sudo apt-get install espeak
```

## 使用方法

### 1. 测试音频功能

首先测试音频功能是否正常工作：

```bash
python tests/test_audio.py
```

这将测试：
- TTS 文本转语音
- 录音器功能
- ASR 语音识别

### 2. 启用语音功能运行主程序

#### 方式一：使用预设配置

```bash
# 使用语音配置（同时启用 ASR 和 TTS）
python main.py --config voice

# 仅启用 TTS（ASR 关闭）
python main.py --config tts
```

#### 方式二：修改配置文件

编辑 `config.py`，修改 `AudioConfig`:

```python
@dataclass
class AudioConfig:
    enable_asr: bool = True        # 启用语音识别
    enable_tts: bool = True        # 启用文本转语音
    whisper_model: str = "base"    # Whisper 模型大小
    asr_language: str = "zh,en"    # 识别语言
    tts_provider: str = "mimo"     # TTS 提供商
    tts_rate: int = 180            # 语速
```

然后正常运行：

```bash
python main.py
```

### 3. 使用语音控制

运行程序后，使用以下键盘控制：

- `v` - 开始语音输入（录制语音命令）
- `q` - 退出程序
- `d` - 切换深度显示

#### 语音命令示例

**中文：**
- "找到杯子"
- "帮我找一下手机"
- "寻找瓶子"
- "定位鼠标"

**英文：**
- "find the cup"
- "where is the bottle"
- "locate the phone"

系统会自动：
1. 录制你的语音（自动检测静音停止，或固定 5 秒）
2. 使用 Whisper 识别语音内容
3. 解析指令，提取目标物体
4. 更新检测目标
5. 开始引导你找到目标

### 4. 语音反馈

当启用 TTS 后，系统会自动：
- 播放引导指令（"向左移动"、"向右移动"等）
- 播放状态提示（"正在录音"、"正在识别"等）
- 播放确认信息（"正在寻找杯子"）

## 配置选项

### ASR 配置

- `enable_asr`: 是否启用语音识别（默认 **True**）
- `whisper_model`: Whisper 模型大小
  - `tiny`: 最快，准确率较低 (~39M)
  - `base`: 速度快，准确率一般 (~74M) - **推荐**
  - `small`: 平衡选择 (~244M)
  - `medium`: 准确率高 (~769M)
  - `large`: 最准确 (~1550M)
- `asr_language`: 识别语言（`zh` 中文, `en` 英文）

### TTS 配置

- `enable_tts`: 是否启用语音输出（默认 **True**）
- `tts_provider`: TTS 提供商（默认 `"mimo"`）
  - `"pyttsx3"`: 离线，使用系统语音引擎
  - `"mimo"`: 云端，需 API Key 但音质更好
- `tts_rate`: 语速（默认 **180**）
  - 范围：100-300
  - 推荐：150-200
- `tts_volume`: 音量（默认 1.0）
  - 范围：0.0-1.0
- `tts_async`: 异步播放（默认 True）
  - True: 不阻塞主程序
  - False: 等待播放完成

### 录音配置

- `record_sample_rate`: 采样率（默认 16000 Hz）
- `record_duration`: 最大录制时长（默认 5.0 秒）
- `auto_detect_silence`: 自动检测静音停止（默认 True）
- `silence_threshold`: 静音阈值（默认 0.01）
- `silence_duration`: 静音持续时长（默认 1.5 秒）

## 性能优化

### 选择合适的 Whisper 模型

| 模型 | 大小 | 速度 | 准确率 | 推荐场景 |
|------|------|------|--------|----------|
| tiny | 39M | 很快 | 低 | 测试或低配置设备 |
| base | 74M | 快 | 中 | **日常使用（推荐）** |
| small | 244M | 中 | 高 | 需要高准确率 |
| medium | 769M | 慢 | 很高 | 专业应用 |
| large | 1550M | 很慢 | 最高 | 研究或离线场景 |

### 设备选择

```python
# 使用 GPU 加速（如果可用）
config.optimization.device = "cuda"

# 使用 CPU（兼容性更好）
config.optimization.device = "cpu"
```

## 故障排除

### TTS 无声音

1. 检查系统音量
2. 检查是否安装了 pyttsx3: `pip install pyttsx3`
3. Windows: 检查语音引擎是否正常
4. Linux: 确保安装了 espeak: `sudo apt-get install espeak`

### ASR 识别失败

1. 确保麦克风工作正常
2. 检查录音权限（Windows 需要在设置中允许应用访问麦克风）
3. 尝试较小的 Whisper 模型（如 `base` 或 `tiny`）
4. 检查网络连接（首次运行需要下载模型）

### 录音无法启动

1. 检查麦克风是否连接
2. 确保安装了 sounddevice: `pip install sounddevice`
3. 运行 `test_audio.py` 查看可用音频设备
4. 如果有多个麦克风，先运行 `python tests/test_audio.py` 查看设备

### 模型下载慢

Whisper 模型首次运行时会自动下载。如果下载慢：

1. 等待下载完成（仅首次需要）
2. 使用较小的模型（如 `tiny` 或 `base`）
3. 配置代理或手动下载模型到缓存目录

模型缓存位置：
- Windows: `C:\Users\<用户名>\.cache\whisper`
- macOS/Linux: `~/.cache/whisper`

## 示例代码

### 单独使用 TTS

```python
from audio.tts.base import TTSEngine

# 初始化 TTS
tts = TTSEngine(rate=180, volume=1.0)

# 播放文本
tts.speak("你好，欢迎使用视觉辅助系统")

# 关闭
tts.close()
```

### 单独使用 ASR

```python
from audio.asr import ASREngine
from audio.audio_utils import AudioRecorder

# 初始化 ASR
asr = ASREngine(model_name="base", language="zh,en")

# 录制音频
recorder = AudioRecorder(sample_rate=16000)
audio = recorder.record(duration=5.0)

# 识别
result = asr.transcribe_audio(audio)
print(f"识别结果: {result['text']}")

# 解析指令
target = asr.parse_command(result['text'])
print(f"目标: {target}")
```

## 参考资料

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - 文本转语音库
- [sounddevice](https://python-sounddevice.readthedocs.io/) - 录音库

## 未来改进

- [ ] 支持连续语音识别（无需按键）
- [ ] 支持多轮对话
- [ ] 添加唤醒词功能
- [ ] 支持更多语言
- [ ] 优化语音识别延迟
- [ ] 添加语音情感检测
