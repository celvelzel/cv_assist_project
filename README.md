[English](#english) | [中文](#中文)

---

<a name="english"></a>

# CV Assist — Real-Time Visual Assistance System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

CV Assist helps visually impaired users locate and grasp target objects in real time. It combines:

- Open-vocabulary object detection (OWL-ViT)
- Monocular depth estimation (MiDaS)
- Hand tracking (MediaPipe)
- Voice interaction (faster-whisper + TTS)

---

## Table of Contents

- [Pipeline & Architecture](#pipeline--architecture)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Keys](#api-keys)
- [Audio Interaction](#audio-interaction)
- [Keyboard Controls](#keyboard-controls)
- [Recent Feature Additions](#recent-feature-additions)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Pipeline & Architecture

```
Camera Frame
   ├── OWL-ViT (open-vocabulary detection)
   ├── MiDaS (depth estimation)
   └── MediaPipe (hand landmarks/gesture)
            ↓
     GuidanceController (hysteresis + task state)
            ↓
  Voice layer (ASR + TTS + optional LLM vision correction)
```

---

## Key Features

- Open-vocabulary target detection (e.g., "cup", "phone", "book")
- Real-time depth-aware hand-to-target guidance
- Hysteresis-based stable guidance (reduces oscillation)
- Long-press `v` voice capture (with `pynput`), fallback to short press when unavailable
- faster-whisper ASR (Chinese/English/mixed)
- Optional LLM vision-assisted command correction via Poe API
- TTS guidance (`pyttsx3` offline or MiMo cloud)
- Spatial briefing (clock direction + estimated distance)
- Proximity beep cues for approach feedback
- Task lifecycle metrics and structured task reports (`logs/task_metrics/`)

---

## Requirements

- Python 3.8+
- Windows / Linux / macOS
- Webcam
- Optional CUDA GPU (recommended for better real-time performance)
- Optional API keys:
  - `MIMO_API_KEY` (MiMo cloud TTS)
  - `POE_API_KEY` (LLM vision correction)

---

## Quick Start

### Option 1 — Scripts (recommended)

**Linux / macOS**

```bash
chmod +x run.sh
./run.sh
```

`run.sh` will:

1. Create `.venv` if needed
2. Activate the environment
3. Auto-install missing dependencies from `requirements.txt`
4. Launch `python -X utf8 main.py`

**Windows**

```cmd
run.bat
```

`run.bat` will:

1. Set UTF-8 environment
2. Auto-install missing dependencies from `requirements.txt`
3. Launch `python main.py --config balanced %*`

### Option 2 — Manual

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .\.venv\Scripts\activate     # Windows

pip install -r requirements.txt
python main.py --config balanced
```

---

## Configuration

Configuration priority:

```
.env environment variables > config.yaml profile overrides > config.py defaults
```

### Profile behavior (current implementation)

`config.yaml` currently defines these profile overrides:

| Profile | Description |
| --- | --- |
| `fast` | lower compute load (smaller input + more frame skipping) |
| `no-voice` | disable ASR and TTS |
| `light` | very aggressive low-resource mode |

`balanced` is the default baseline from `config.py` (no YAML override needed).

CLI profile options are now aligned with current config presets: `balanced`, `fast`, `no-voice`, `light`.

### Example: `config.yaml`

```yaml
profiles:
  fast:
    model:
      owlvit_input_size: [320, 320]
    optimization:
      skip_frames_detection: 3
      skip_frames_depth: 3

  no-voice:
    audio:
      enable_asr: false
      enable_tts: false

  light:
    model:
      owlvit_input_size: [256, 256]
      midas_model: MiDaS_small
    optimization:
      skip_frames_detection: 5
      skip_frames_depth: 5
```

---

## API Keys

Create `.env` (copy from `.env.example`) and set only what you use.

```env
MIMO_API_KEY=your_mimo_api_key_here
POE_API_KEY=your_poe_api_key_here
```

- `MIMO_API_KEY`: enables MiMo cloud TTS
- `POE_API_KEY`: enables LLM vision parser for ASR correction

---

## Audio Interaction

- ASR engine: **faster-whisper**
- TTS providers: **pyttsx3** (offline) or **mimo** (cloud)
- If `pynput` is available: **long-press `v`** to record, release to transcribe
- If `pynput` is unavailable: short-press `v` fallback path is used

For subsystem details, see [`audio/README.md`](audio/README.md).

---

## Keyboard Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `d` | Toggle depth visualization |
| `v` | Voice input (long press when `pynput` is available) |

---

## Recent Feature Additions

Based on recent git history, user-facing additions include:

- ASR upgrade to faster-whisper with long-press voice flow and echo stripping
- TTS improvements: async queue behavior, lifecycle/priority speech handling, search-timeout feedback
- Spatial briefing enhancements (clock direction + estimated distance)
- Proximity beep stabilization and playback conflict handling
- Task metrics and test coverage expansion (`test_task_metrics.py`, `test_system_tts_policy.py`)

---

## Testing

```bash
# Smoke import
python tests/test_all.py

# Guidance logic
python tests/test_guidance.py

# Config loading
python tests/test_config_loading.py

# Task metrics
python tests/test_task_metrics.py

# System TTS policy
python tests/test_system_tts_policy.py

# LLM vision
python tests/test_llm_vision.py

# Audio / logging script tests
python tests/test_audio.py
python tests/test_logging.py
```

---

## Project Structure

```text
cv_assist_project/
├── main.py
├── config.py
├── config.yaml
├── run.sh / run.bat
├── requirements.txt
├── .env.example
├── core/
├── detectors/
├── audio/
├── utils/
├── tests/
└── docs/
```

---

<a name="中文"></a>

# CV Assist — 实时视觉辅助系统

CV Assist 面向视觉障碍用户，提供“目标识别 + 手部引导 + 语音反馈”的实时辅助能力。

## 核心能力

- OWL-ViT 开放词汇目标检测
- MiDaS 单目深度估计
- MediaPipe 手部关键点与手势识别
- faster-whisper 语音识别
- `pyttsx3` / MiMo 双 TTS 后端
- LLM 视觉纠错（Poe API，可选）
- 空间简报（钟点方向 + 约距）与接近提示音
- 任务级指标与日志报告

## 快速启动

- Linux/macOS：`./run.sh`
- Windows：`run.bat`
- 手动：`python main.py --config balanced`

## 配置模式（当前实现）

`config.yaml` 已定义：`fast`、`no-voice`、`light`。

`balanced` 为默认基础配置（来自 `config.py`）。

CLI 与当前配置保持一致，仅支持：`balanced`、`fast`、`no-voice`、`light`。

## API Key

在 `.env` 中按需设置：

```env
MIMO_API_KEY=your_mimo_api_key_here
POE_API_KEY=your_poe_api_key_here
```

## 按键

- `q`：退出
- `d`：切换深度图
- `v`：语音输入（安装 `pynput` 时为长按）

## 测试

```bash
python tests/test_all.py
python tests/test_guidance.py
python tests/test_config_loading.py
python tests/test_task_metrics.py
python tests/test_system_tts_policy.py
python tests/test_llm_vision.py
python tests/test_audio.py
python tests/test_logging.py
```

更多细节请查看：

- `audio/README.md`
- `docs/userguide/config-setup.md`
- `docs/system_testing_and_metrics.md`
