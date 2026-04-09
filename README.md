[English](#english) | [中文](#中文)

---

<a name="english"></a>

# CV Assist — Visual Assistance System for the Visually Impaired

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

A real-time visual assistance system designed to help visually impaired users locate and grasp objects. The system combines **open-vocabulary object detection**, **hand tracking**, and **monocular depth estimation** with **voice interaction** to provide real-time spatial guidance through speech.

---

## Table of Contents

- [Pipeline & Architecture](#pipeline--architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Key Setup](#api-key-setup)
- [Audio Guide (ASR & TTS)](#audio-guide-asr--tts)
- [Keyboard Controls](#keyboard-controls)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Pipeline & Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Webcam     │────▶│  OWL-ViT         │────▶│  Detection   │
│   Input      │     │  (Zero-shot OD)  │     │  Results     │
└──────────────┘     └──────────────────┘     └──────┬───────┘
       │                                              │
       │            ┌──────────────────┐              │
       ├───────────▶│  MiDaS           │──────────────┤
       │            │  (Depth Est.)    │              │
       │            └──────────────────┘              │
       │                                              │
       │            ┌──────────────────┐              ▼
       └───────────▶│  MediaPipe       │     ┌────────────────┐
                    │  (Hand Tracking) │────▶│  Guidance      │
                    └──────────────────┘     │  Controller    │
                                             └───────┬────────┘
                                                     │
                    ┌──────────────────┐              ▼
                    │  Whisper (ASR)   │     ┌────────────────┐
                    │  ← Voice Cmd     │     │  TTS Output    │
                    └──────────────────┘     │  (pyttsx3 /    │
                                            │   MiMo Cloud)  │
                                            └────────────────┘
```

### Technical Pipeline

1. **Object Detection** — OWL-ViT v2 (`google/owlv2-base-patch16-ensemble`) performs open-vocabulary, zero-shot detection from natural-language queries (e.g. "a cup").
2. **Depth Estimation** — MiDaS (`MiDaS_small`) generates a monocular depth map from a single RGB frame. Depth values are sampled at detection centers to estimate object distance.
3. **Hand Tracking** — MediaPipe provides 21-keypoint hand landmarks and gesture recognition (open / closed / pointing).
4. **Spatial Guidance** — The `GuidanceController` computes directional instructions ("move left", "move closer") using hysteresis thresholds to prevent jitter.
5. **Voice Interaction** — Speech input via Whisper ASR sets the search target; guidance instructions are spoken aloud via TTS.

---

## Features

- **Open-vocabulary detection** — search for any object by name, no retraining needed
- **Real-time depth estimation** — monocular depth via MiDaS for distance-aware guidance
- **Hand tracking with gesture recognition** — detects open, closed, and pointing gestures
- **Voice control** — speak to set the target object (Chinese & English supported)
- **LLM-assisted voice recognition** — Uses Poe API + deepseek LLM with visual context to correct speech errors and improve accuracy
- **Speech guidance** — audio instructions guide the user's hand toward the target
- **Hysteresis-based guidance** — prevents oscillation at alignment boundaries
- **Configurable profiles** — fast / balanced / voice / tts / mimo-tts presets
- **Modular design** — each detector is independent and easily extensible

---

## Requirements

- **Python** 3.8 or higher
- **OS**: Windows 10+, Linux, macOS
- **Hardware**: Webcam; GPU (CUDA) recommended but not required
- **Disk**: ~2 GB for models (downloaded on first run)
- _Optional_: Poe API key for LLM-assisted voice recognition enhancement

---

## Quick Start

### Option 1 — Using deployment scripts (recommended)

The deployment scripts handle environment setup, dependency installation, and launch automatically.

**Linux / macOS:**

```bash
chmod +x run.sh
./run.sh
```

The script will:

1. Create a `.venv` virtual environment (if not exists)
2. Activate it
3. Install dependencies from `requirements.txt` (if missing)
4. Launch with TTS enabled (`--config tts`)

**Windows:**

```cmd
run.bat
```

The script will:

1. Set UTF-8 environment variables
2. Check for core dependencies and install if missing
3. Launch with TTS enabled (`--config tts`)

### Option 2 — Manual setup

```bash
# Clone
git clone <repo-url>
cd cv_assist_project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .\.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py --config tts
```

### Verify Installation

```bash
python tests/test_all.py
```

---

## Configuration

Configuration is loaded with the following priority (highest first):

```
Environment variables (.env)  >  config.yaml profile overrides  >  config.yaml base  >  Code defaults
```

### Profiles

Pass `--config <profile>` to select a preset. Profiles are defined in the `profiles:` section of `config.yaml` — each profile is a partial override merged on top of the base configuration:

| Profile    | ASR | TTS | TTS Provider | Use Case                       |
| ---------- | --- | --- | ------------ | ------------------------------ |
| `balanced` | On  | On  | mimo         | Default, full voice (default)  |
| `fast`     | Off | Off | —            | Low-resource / testing         |
| `no-voice` | Off | Off | —            | Visual processing only         |

> `balanced` has no overrides — it uses the base configuration directly.

### config.yaml

Edit `config.yaml` to customize parameters. The file is organized into two parts: **profile presets** (top) and **base configuration** (bottom).

```yaml
# Profile presets — only list fields that differ from base
profiles:
  fast:
    model:
      owlvit_input_size: [320, 320]
    optimization:
      skip_frames_detection: 3
      skip_frames_depth: 3

  tts:
    audio:
      enable_tts: true
      tts_rate: 180

  mimo-tts:
    audio:
      enable_tts: true
      tts_provider: mimo

# Base configuration (shared by all profiles)
target_queries: ["a cup", "a bottle"]

camera:
  width: 640
  height: 480

model:
  owlvit_model: google/owlv2-base-patch16-ensemble
  owlvit_input_size: [384, 384]
  midas_model: MiDaS_small
  hand_max_num: 1

optimization:
  use_fp16: true
  skip_frames_detection: 2
  skip_frames_depth: 2
  device: auto

audio:
  enable_tts: true
  tts_provider: mimo
  tts_rate: 180
  tts_instruction_interval_sec: 2.0
  # ...
```

To add a custom profile, add an entry under `profiles:` and select it with `--config <name>`.

---

## API Key Setup

### MiMo TTS (Xiaomi Cloud TTS)

The `mimo-tts` profile uses Xiaomi's MiMo cloud-based TTS for higher-quality speech synthesis.

**Step 1** — Obtain an API key from [api.xiaomimimo.com](https://api.xiaomimimo.com).

**Step 2** — Create a `.env` file in the project root (or copy from `.env.example`):

```bash
cp .env.example .env
```

**Step 3** — Edit `.env` and set your key:

```env
MIMO_API_KEY=your_api_key_here
```

**Step 4** — Run with the `mimo-tts` profile:

```bash
python main.py --config mimo-tts
```

> **Tip:** The `.env` file is gitignored and will never be committed.

### OpenAI Whisper (for ASR)

Whisper runs locally and **does not require an API key**. Models are downloaded automatically on first use.

If you also want to use OpenAI's cloud-based models, set:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## Audio Guide (ASR & TTS)

### TTS (Text-to-Speech)

Two backends are available:

| Provider  | Offline | Quality | Setup                             |
| --------- | ------- | ------- | --------------------------------- |
| `pyttsx3` | Yes     | Basic   | No extra config needed            |
| `mimo`    | No      | High    | Requires `MIMO_API_KEY` in `.env` |

### ASR (Automatic Speech Recognition)

- Engine: OpenAI Whisper (runs locally)
- Supported languages: Chinese (`zh`), English (`en`)
- Voice commands are parsed to extract target objects, e.g.:
  - "找到杯子" → target: "杯子"
  - "where is the cup" → target: "cup"

### Usage

1. Launch with `--config voice` or `--config tts`
2. Press `v` to start voice input
3. Speak the target object name
4. The system will start guiding you with audio instructions

For detailed audio documentation, see [audio/README.md](audio/README.md).

---

## Keyboard Controls

| Key | Action                     |
| --- | -------------------------- |
| `q` | Quit                       |
| `d` | Toggle depth visualization |
| `v` | Start voice input          |

---

## LLM-Assisted Voice Recognition with Vision

The system integrates with Poe API (using deepseek-v3.2 LLM) to enhance voice command recognition using visual context from camera frames. This significantly improves accuracy for users with unclear speech, accents, or background noise.

### How It Works

When you speak a command (e.g., "find a phone" but ASR hears "ana more i´m up phone"):

1. **Whisper ASR** transcribes the audio (may have errors)
2. **LLM + Vision** analyzes the transcribed text together with 4 recent camera frames
3. **Corrected Output**: LLM returns the most likely target object (e.g., "phone")
4. **OWL-ViT Detection** searches for the corrected target

### Setup

**Step 1** — Get a Poe API key:

- Visit [poe.com/api/key](https://poe.com/api/key)
- Create an account and generate your API key

**Step 2** — Add to `.env`:

```env
POE_API_KEY=your_poe_api_key_here
```

**Step 3** — Enable in config (automatic if ASR is enabled):

- No additional configuration needed!
- The feature is automatically enabled when:
  - ASR is enabled (`enable_asr: true`)
  - `POE_API_KEY` is set in `.env`

**Step 4** — Use voice commands:

```bash
python main.py --config voice  # Full ASR + TTS + LLM Vision
# or
python main.py --config tts    # TTS guidance only
```

### Configuration

Edit `config.yaml` to customize LLM behavior:

```yaml
llm_vision:
  enable_llm_parsing: true # Enable/disable LLM vision parsing
  poe_model: "deepseek-v3.2" # LLM model (don't change)
  poe_timeout_sec: 5.0 # API call timeout
  max_frames_for_vision: 4 # Number of frames to send
  api_retry_count: 1 # Retry attempts on timeout
```

### Performance Notes

- **Accuracy**: Significantly improves robustness to speech errors and accents
- **Latency**: API call adds ~1-2 seconds to target update (acceptable for accessibility use)
- **Async Processing**: LLM analysis happens in the background; video streaming is **never blocked**
- **Fallback**: If API fails or times out, system falls back to regex-based parsing automatically
- **Frames**: Uses 4 recent frames to provide rich visual context for the LLM

### Example Scenarios

| What User Says                | Whisper Output | LLM + Vision Output | Result      |
| ----------------------------- | -------------- | ------------------- | ----------- |
| "找手机" (find phone)         | "找爱手及"     | "phone"             | ✓ Correct   |
| "find the cup" (unclear)      | "find the cut" | "cup"               | ✓ Corrected |
| "哪里是书" (where's the book) | "哪里是鼠"     | "book"              | ✓ Fixed     |

### Troubleshooting

**LLM Vision not working?**

- Check `.env` file has `POE_API_KEY=<your_key>`
- Verify key is valid: `python -c "from audio.llm_vision import LLMVisionParser; LLMVisionParser('your_key')"
- Check logs for "LLM Vision Parser initialized" message
- If disabled, you'll see logs like "LLM vision parsing disabled" or "POE_API_KEY not configured"

**Performance too slow?**

- Reduce `max_frames_for_vision` in config.yaml (default 4) to use fewer frames
- Increase `poe_timeout_sec` if getting frequent timeouts

**API errors?**

- Ensure you have internet connectivity
- Verify Poe API key is valid and has remaining quota
- Check Poe status page for API outages

---

## Testing

```bash
# Smoke test — verify all modules import correctly
python tests/test_all.py

# Audio test — TTS, recording, and ASR
python tests/test_audio.py

# LLM Vision test — API integration and fallback
python tests/test_llm_vision.py

# Guidance logic test — hysteresis and stability
python tests/test_guidance.py

# Logging & FPS test
python tests/test_logging.py
```

---

## Project Structure

```
cv_assist_project/
├── main.py                      # Entry point
├── config.py                    # Configuration management (dataclasses + YAML + .env)
├── config.yaml                  # User-editable configuration file
├── requirements.txt             # Python dependencies
├── run.sh                       # Linux/macOS deployment script
├── run.bat                      # Windows deployment script
├── .env.example                 # API key template
│
├── core/
│   ├── system.py                # CVAssistSystem — main orchestrator & run loop
│   └── guidance.py              # GuidanceController — spatial instruction logic
│
├── detectors/
│   ├── owl_vit_detector.py      # OWL-ViT zero-shot object detector
│   ├── hand_tracker.py          # MediaPipe hand landmark & gesture detector
│   ├── depth_estimator.py       # MiDaS monocular depth estimator
│   └── hand_landmarker.task     # MediaPipe binary model
│
├── audio/
│   ├── asr.py                   # Whisper ASR engine
│   ├── audio_utils.py           # Audio recording utilities
│   ├── llm_vision.py            # LLM Vision Parser — Poe API integration for voice correction
│   └── tts/
│       ├── base.py              # Abstract TTS interface
│       ├── pyttsx3_backend.py   # Offline TTS (pyttsx3)
│       └── mimo_backend.py      # Cloud TTS (Xiaomi MiMo)
│
├── utils/
│   └── logger.py                # Logging & FPS counter
│
└── tests/
    ├── test_all.py              # Import smoke test
    ├── test_audio.py            # Audio subsystem tests
    ├── test_guidance.py         # Guidance hysteresis tests
    ├── test_logging.py          # Logging & FPS tests
    └── test_llm_vision.py       # LLM Vision Parser tests
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow [PEP 8](https://peps.python.org/pep-0008/) code style
4. Add tests for new functionality
5. Run `python tests/test_all.py` before submitting
6. Open a Pull Request

---

---

<a name="中文"></a>

# CV Assist — 视觉辅助系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

一个面向视觉障碍用户的实时视觉辅助系统。系统整合了**开放词汇目标检测**、**手部追踪**、**单目深度估计**和**语音交互**，通过语音播报实时空间引导信息，帮助用户定位并抓取目标物体。

---

## 目录

- [技术架构与管线](#技术架构与管线)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API Key 设置](#api-key-设置)
- [音频功能（ASR & TTS）](#音频功能asr--tts)
- [键盘控制](#键盘控制)
- [测试](#测试)
- [项目结构](#项目结构)
- [协作指南](#协作指南)

---

## 技术架构与管线

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   摄像头     │────▶│  OWL-ViT         │────▶│  检测结果    │
│   视频流     │     │  (零样本检测)    │     │              │
└──────────────┘     └──────────────────┘     └──────┬───────┘
       │                                              │
       │            ┌──────────────────┐              │
       ├───────────▶│  MiDaS           │──────────────┤
       │            │  (深度估计)      │              │
       │            └──────────────────┘              │
       │                                              │
       │            ┌──────────────────┐              ▼
       └───────────▶│  MediaPipe       │     ┌────────────────┐
                    │  (手部追踪)      │────▶│  空间引导      │
                    └──────────────────┘     │  控制器        │
                                             └───────┬────────┘
                                                     │
                    ┌──────────────────┐              ▼
                    │  Whisper (ASR)   │     ┌────────────────┐
                    │  ← 语音指令      │     │  TTS 语音播报  │
                    └──────────────────┘     │  (pyttsx3 /    │
                                            │   MiMo 云端)   │
                                            └────────────────┘
```

### 技术管线

1. **目标检测** — OWL-ViT v2（`google/owlv2-base-patch16-ensemble`）基于自然语言查询进行开放词汇零样本检测（如 "a cup"）。
2. **深度估计** — MiDaS（`MiDaS_small`）从单张 RGB 图像生成深度图，在检测框中心采样距离值。
3. **手部追踪** — MediaPipe 提供 21 个手部关键点及手势识别（张开 / 握拳 / 指向）。
4. **空间引导** — `GuidanceController` 计算方向指令（"向左移"、"靠近"），使用滞回阈值防止边界抖动。
5. **语音交互** — Whisper ASR 接收语音设置搜索目标；TTS 将引导指令通过语音播放。

---

## 功能特性

- **开放词汇检测** — 通过自然语言指定检测目标，无需重新训练
- **实时深度估计** — 基于 MiDaS 的单目深度感知，提供距离引导
- **手部追踪与手势识别** — 检测张开、握拳、指向三种手势
- **语音控制** — 支持中英文语音输入设定搜索目标
- **LLM 增强语音识别** — 用 Poe API + deepseek LLM + 视觉上下文，纠正语音错误并提高准确度
- **语音播报引导** — 通过语音指令引导用户将手移向目标
- **滞回引导算法** — 防止对齐边界的指令来回跳动
- **可配置预设** — fast / balanced / voice / tts / mimo-tts 五种配置模式
- **模块化设计** — 各检测器独立，便于扩展

---

## 环境要求

- **Python** 3.8 或更高版本
- **操作系统**：Windows 10+、Linux、macOS
- **硬件**：摄像头；推荐 GPU（CUDA），非必须
- **磁盘**：约 2 GB（模型首次运行时自动下载）
- _可选_：Poe API Key，用于增强 LLM 语音识别

---

## 快速开始

### 方式一：使用部署脚本（推荐）

部署脚本自动完成环境创建、依赖安装和启动。

**Linux / macOS：**

```bash
chmod +x run.sh
./run.sh
```

脚本将自动：

1. 创建 `.venv` 虚拟环境（如不存在）
2. 激活虚拟环境
3. 安装 `requirements.txt` 中的依赖（如缺失）
4. 以 TTS 模式启动（`--config tts`）

**Windows：**

```cmd
run.bat
```

脚本将自动：

1. 设置 UTF-8 环境变量
2. 检查并安装缺失的依赖
3. 以 TTS 模式启动（`--config tts`）

### 方式二：手动安装

```bash
# 克隆仓库
git clone <repo-url>
cd cv_assist_project

# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py --config tts
```

### 验证安装

```bash
python tests/test_all.py
```

---

## 配置说明

配置加载优先级（从高到低）：

```
环境变量 (.env)  >  config.yaml profile overrides  >  config.yaml 基础配置  >  代码默认值
```

### 配置预设

通过 `--config <profile>` 选择预设。预设定义在 `config.yaml` 的 `profiles:` 段中，每个预设只列出与基础配置不同的字段，系统会自动合并：

| 预设       | ASR | TTS | TTS 提供商 | 适用场景                   |
| ---------- | --- | --- | ---------- | -------------------------- |
| `balanced` | 开  | 开  | mimo       | 默认，完整语音（默认）     |
| `fast`     | 关  | 关  | —          | 低资源环境 / 测试          |
| `no-voice` | 关  | 关  | —          | 仅视觉处理                 |

> `balanced` 无额外 override，直接使用基础配置。

### config.yaml

编辑 `config.yaml` 自定义参数。文件分为两部分：**profile 预设**（顶部）和**基础配置**（底部）。

```yaml
# Profile 预设 —— 仅列出与 base 不同的字段
profiles:
  fast:
    model:
      owlvit_input_size: [320, 320]
    optimization:
      skip_frames_detection: 3
      skip_frames_depth: 3

  tts:
    audio:
      enable_tts: true
      tts_rate: 180

  mimo-tts:
    audio:
      enable_tts: true
      tts_provider: mimo

# 基础配置（所有 profile 共享）
target_queries: ["a cup", "a bottle"]

camera:
  width: 640
  height: 480

model:
  owlvit_model: google/owlv2-base-patch16-ensemble
  owlvit_input_size: [384, 384]
  midas_model: MiDaS_small
  hand_max_num: 1

optimization:
  use_fp16: true
  skip_frames_detection: 2
  skip_frames_depth: 2
  device: auto

audio:
  enable_tts: true
  tts_provider: mimo
  tts_rate: 180
  tts_instruction_interval_sec: 2.0
  # ...
```

如需自定义预设，在 `profiles:` 下添加新条目，然后用 `--config <name>` 选择即可。

---

## API Key 设置

### MiMo TTS（小米云端 TTS）

`mimo-tts` 预设使用小米 MiMo 云端 TTS 以获得更高质量的语音合成。

**第一步** — 从 [api.xiaomimimo.com](https://api.xiaomimimo.com) 获取 API Key。

**第二步** — 在项目根目录创建 `.env` 文件（或从 `.env.example` 复制）：

```bash
cp .env.example .env
```

**第三步** — 编辑 `.env`，填入你的 API Key：

```env
MIMO_API_KEY=your_api_key_here
```

**第四步** — 使用 `mimo-tts` 配置启动：

```bash
python main.py --config mimo-tts
```

> **提示：** `.env` 文件已在 `.gitignore` 中排除，不会被提交到仓库。

### OpenAI Whisper（语音识别）

Whisper 在本地运行，**不需要 API Key**。模型首次使用时自动下载。

如需使用 OpenAI 的云端模型，可设置：

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## 音频功能（ASR & TTS）

### TTS（文本转语音）

支持两种后端：

| 提供商    | 离线 | 质量 | 配置要求                   |
| --------- | ---- | ---- | -------------------------- |
| `pyttsx3` | 是   | 基础 | 无需额外配置               |
| `mimo`    | 否   | 高   | 需在 `.env` 中设置 API Key |

### ASR（语音识别）

- 引擎：OpenAI Whisper（本地运行）
- 支持语言：中文（`zh`）、英文（`en`）
- 语音命令自动解析目标物体，例如：
  - "找到杯子" → 目标："杯子"
  - "where is the cup" → 目标："cup"

### 使用方法

1. 使用 `--config voice` 或 `--config tts` 启动
2. 按 `v` 开始语音输入
3. 说出目标物体名称
4. 系统将通过语音引导你移动手部

详细的音频功能文档请参考 [audio/README.md](audio/README.md)。

---

## 键盘控制

| 按键 | 操作           |
| ---- | -------------- |
| `q`  | 退出           |
| `d`  | 切换深度可视化 |
| `v`  | 开始语音输入   |

---

## LLM 增强语音识别与视觉理解

系统现已集成 Poe API（使用 deepseek-v3.2 LLM）以增强语音命令识别准确度。LLM 可同时分析识别到的文本和摄像头中的视觉信息，大幅提高了对口音、背景噪声和不清晰语音的鲁棒性。

### 工作流程

当用户说一个命令时（例如用户说"找手机"，但 Whisper ASR 识别为"找爱手及"）：

1. **Whisper ASR** 转录音频（可能有错误）
2. **LLM + 视觉** 将转录文本与最近 4 帧摄像头图像一起分析
3. **纠正输出**: LLM 返回最可能的目标物体（例如"手机"）
4. **OWL-ViT 检测** 搜索纠正后的目标

### 配置步骤

**第一步** — 获取 Poe API Key：

- 访问 [poe.com/api/key](https://poe.com/api/key)
- 创建账户并生成你的 API Key

**第二步** — 添加到 `.env` 文件：

```env
POE_API_KEY=your_poe_api_key_here
```

**第三步** — 在配置中启用（如果 ASR 启用则自动启用）：

- 无需额外配置！
- 满足以下条件时自动启用：
  - ASR 已启用（`enable_asr: true`）
  - `.env` 中已设置 `POE_API_KEY`

**第四步** — 使用语音命令：

```bash
python main.py --config voice   # 完整的 ASR + TTS + LLM 视觉分析
# 或
python main.py --config tts     # 仅 TTS 播报
```

### 配置参数

编辑 `config.yaml` 自定义 LLM 行为：

```yaml
llm_vision:
  enable_llm_parsing: true # 启用/禁用 LLM 视觉解析
  poe_model: "deepseek-v3.2" # LLM 模型（不建议修改）
  poe_timeout_sec: 5.0 # API 调用超时时间（秒）
  max_frames_for_vision: 4 # 发送的帧数
  api_retry_count: 1 # 超时时的重试次数
```

### 性能说明

- **准确性**：大幅提高了对语音错误和口音的鲁棒性
- **延迟**：API 调用增加约 1-2 秒的目标更新延迟（可接受）
- **异步处理**: LLM 分析在后台进行；视频流处理**绝不会**被阻塞
- **回退机制**: 如果 API 失败或超时，系统自动回退到正则解析
- **视觉信息**: 使用 4 帧最近的摄像头画面为 LLM 提供丰富的视觉上下文

### 效果示例

| 用户说的话 | Whisper 输出 | LLM + 视觉输出 | 结果   |
| ---------- | ------------ | -------------- | ------ |
| "找手机"   | "找爱手及"   | "手机"         | ✓ 正确 |
| "杯子在哪" | "怖子在哪"   | "cup" / "杯子" | ✓ 纠正 |
| "找书"     | "找鼠"       | "book" / "书"  | ✓ 修正 |

### 故障排除

**LLM 视觉功能无法工作？**

- 检查 `.env` 文件中是否有 `POE_API_KEY=<你的密钥>`
- 验证 API Key 有效性
- 查看日志中是否有 "LLM Vision Parser initialized" 消息
- 如果禁用，你会看到 "LLM vision parsing disabled" 或 "POE_API_KEY not configured" 的日志

**性能过慢？**

- 在 config.yaml 中减少 `max_frames_for_vision`（默认为 4）以使用更少的帧
- 如果经常超时，增加 `poe_timeout_sec`

**API 错误？**

- 确保网络连接正常
- 验证 Poe API Key 有效且配额充足
- 检查 Poe 状态页面是否有 API 故障

---

## 测试

```bash
# 冒烟测试 — 验证所有模块可正常导入
python tests/test_all.py

# 音频测试 — TTS、录音和 ASR
python tests/test_audio.py

# LLM 视觉测试 — API 集成与回退机制
python tests/test_llm_vision.py

# 引导逻辑测试 — 滞回阈值和稳定性
python tests/test_guidance.py

# 日志与 FPS 测试
python tests/test_logging.py
```

---

## 项目结构

```
cv_assist_project/
├── main.py                      # 程序入口
├── config.py                    # 配置管理（dataclass + YAML + .env）
├── config.yaml                  # 用户可编辑的配置文件
├── requirements.txt             # Python 依赖
├── run.sh                       # Linux/macOS 部署脚本
├── run.bat                      # Windows 部署脚本
├── .env.example                 # API Key 模板
│
├── core/
│   ├── system.py                # CVAssistSystem — 主控制器与运行循环
│   └── guidance.py              # GuidanceController — 空间引导逻辑
│
├── detectors/
│   ├── owl_vit_detector.py      # OWL-ViT 零样本目标检测器
│   ├── hand_tracker.py          # MediaPipe 手部关键点与手势检测
│   ├── depth_estimator.py       # MiDaS 单目深度估计器
│   └── hand_landmarker.task     # MediaPipe 二进制模型
│
├── audio/
│   ├── asr.py                   # Whisper 语音识别引擎
│   ├── audio_utils.py           # 录音工具
│   ├── llm_vision.py            # LLM 视觉解析器 — Poe API 集成用于语音纠正
│   └── tts/
│       ├── base.py              # TTS 抽象接口
│       ├── pyttsx3_backend.py   # 离线 TTS（pyttsx3）
│       └── mimo_backend.py      # 云端 TTS（小米 MiMo）
│
├── utils/
│   └── logger.py                # 日志与 FPS 计数器
│
└── tests/
    ├── test_all.py              # 导入冒烟测试
    ├── test_audio.py            # 音频子系统测试
    ├── test_guidance.py         # 引导滞回逻辑测试
    ├── test_logging.py          # 日志与 FPS 测试
    └── test_llm_vision.py       # LLM 视觉解析器测试
```

---

## 协作指南

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 遵循 [PEP 8](https://peps.python.org/pep-0008/) 代码风格
4. 为新功能添加测试
5. 提交前运行 `python tests/test_all.py`
6. 发起 Pull Request
