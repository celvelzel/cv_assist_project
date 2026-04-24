# 音频功能使用指南（ASR & TTS）

本文档说明当前代码中的音频子系统实现与使用方式（已与 `config.py`、`core/system.py`、`requirements.txt` 对齐）。

---

## 1. 功能概览

### ASR（语音识别）

- 引擎：`faster-whisper`（`audio/asr.py`）
- 支持语言：`zh` / `en` / `zh,en` / `auto`
- 支持指令解析：
  - 中文：如“找杯子”“帮我找手机”
  - 英文：如“find the cup”
- 支持 LLM 视觉辅助纠错（可选）：`audio/llm_vision.py`（Poe API）

### TTS（语音播报）

- 后端：
  - `pyttsx3`（离线）
  - `mimo`（云端，需 API Key）
- 支持普通引导播报、生命周期播报、抢占播报
- 支持队列节流与过期消息丢弃（降低播报堆积）

### 语音交互行为

- 安装 `pynput` 时：**长按 `v`** 进入录音，松开后识别
- 未安装 `pynput` 时：自动退化为窗口内短按 `v` 兼容路径

---

## 2. 安装依赖

推荐直接安装项目依赖：

```bash
pip install -r requirements.txt
```

音频相关核心依赖（节选）：

- `faster-whisper`
- `sounddevice`
- `scipy`
- `pyttsx3`
- `pygame`
- `pynput`
- `openai`（用于 MiMo / Poe API 客户端）

> 注意：当前实现是 `faster-whisper`，不是 `openai-whisper`。

---

## 3. API Key 配置

复制模板：

```bash
cp .env.example .env
```

按需设置：

```env
MIMO_API_KEY=your_mimo_api_key_here
POE_API_KEY=your_poe_api_key_here
```

- `MIMO_API_KEY`：启用 MiMo 云端 TTS
- `POE_API_KEY`：启用 LLM 视觉纠错

---

## 4. 运行方式

```bash
# 默认平衡模式
python main.py --config balanced

# 仅视觉（关闭 ASR/TTS）
python main.py --config no-voice

# 性能优先
python main.py --config fast
```

> `config.yaml` 当前定义的 profile 为：`fast`、`no-voice`、`light`。  
> `balanced` 为默认基础配置（来自 `config.py`）。

---

## 5. 常用音频配置项（`config.py`）

### ASR

- `audio.enable_asr`
- `audio.whisper_model`（默认 `medium`）
- `audio.whisper_compute_type`（可选）
- `audio.asr_language`（默认 `zh,en`）

### TTS

- `audio.enable_tts`
- `audio.tts_provider`（`pyttsx3` / `mimo`）
- `audio.tts_rate`
- `audio.tts_volume`
- `audio.tts_async`
- `audio.tts_instruction_interval_sec`
- `audio.tts_grab_repeat_sec`
- `audio.tts_max_queue_size`
- `audio.tts_drop_stale`
- `audio.tts_state_change_bypass`

### 长按 v 相关

- `audio.voice_v_long_press_sec`
- `audio.voice_record_after_enter_tts`
- `audio.voice_after_enter_tts_delay_sec`
- `audio.voice_min_capture_sec`
- `audio.voice_min_capture_rms`

### LLM 视觉纠错

- `llm_vision.enable_llm_parsing`
- `llm_vision.poe_model`
- `llm_vision.poe_timeout_sec`
- `llm_vision.max_frames_for_vision`
- `llm_vision.api_retry_count`

---

## 6. 测试

```bash
python tests/test_audio.py
python tests/test_asr_language_mode.py
python tests/test_llm_vision.py
python tests/test_system_tts_policy.py
```

说明：

- `test_audio.py` 为脚本型测试，可能依赖真实麦克风/扬声器环境
- `test_llm_vision.py` 含 API/可选依赖路径，部分场景可能 skip

---

## 7. 排障建议

### 没有声音

1. 检查系统输出设备与音量
2. 检查 `audio.enable_tts` 与 `audio.tts_provider`
3. 使用 MiMo 时确认 `MIMO_API_KEY` 有效

### 语音识别无结果

1. 检查麦克风权限与设备
2. 检查 `audio.enable_asr`
3. 调整 `audio.whisper_model` 与 `audio.asr_language`

### 长按 v 不生效

1. 确认已安装 `pynput`
2. 查看日志是否提示已启用长按监听
3. 未安装 `pynput` 时可使用短按 `v` 兼容路径
