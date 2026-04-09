# 配置设置指南

> 本指南详细介绍 CV 视觉辅助系统的所有配置项。所有默认值均在 `config.py` 中定义，每个配置项都附有详细的中文注释说明其含义、可选值和修改效果。

## 目录

- [配置架构概览](#配置架构概览)
- [快速开始](#快速开始)
- [Profile 预设详解](#profile-预设详解)
- [模型配置 (ModelConfig)](#模型配置-modelconfig)
- [性能优化配置 (OptimizationConfig)](#性能优化配置-optimizationconfig)
- [空间引导配置 (GuidanceConfig)](#空间引导配置-guidanceconfig)
- [音频配置 (AudioConfig)](#音频配置-audioconfig)
- [LLM 视觉增强配置 (LLMVisionConfig)](#llm-视觉增强配置-llmvisionconfig)
- [日志配置 (LoggingConfig)](#日志配置-loggingconfig)
- [系统级配置 (SystemConfig)](#系统级配置-systemconfig)
- [环境变量与 API Key](#环境变量与-api-key)
- [常见问题](#常见问题)

---

## 配置架构概览

系统采用三层配置管理架构，职责清晰：

| 文件 | 职责 | 修改频率 |
|------|------|----------|
| `config.py` | **所有默认值的唯一定义处**，包含详尽中文文档 | 开发者修改 |
| `config.yaml` | **仅包含 Profile 预设覆盖**，不定义基础配置 | 运维/用户修改 |
| `.env` | **统一管理所有 API Key**，不参与业务配置 | 部署时设置一次 |

配置加载优先级（从高到低）：

```
环境变量 (.env)  >  config.yaml profile 预设覆盖  >  config.py 默认值
```

---

## 快速开始

### 选择运行模式

通过 `--config` 参数选择预设：

```bash
# 默认模式（完整语音，ASR + TTS）
python main.py

# 仅视觉处理，无语音交互
python main.py --config no-voice

# 性能优先（低配设备）
python main.py --config fast
```

### 设置 API Key

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 Key
# MIMO_API_KEY=your_key_here
# POE_API_KEY=your_key_here
```

---

## Profile 预设详解

### `balanced`（默认）

- ASR：开启 (Whisper base 模型)
- TTS：开启 (mimo 提供商，语速 180)
- 适用场景：默认完整体验

### `no-voice`（仅视觉处理）

- ASR：关闭
- TTS：关闭
- 适用场景：开发调试、无语音需求的测试

### `fast`（性能优先）

| 参数 | 默认值 | fast 值 | 效果 |
|------|--------|---------|------|
| `owlvit_input_size` | (384, 384) | (320, 320) | 检测速度提升约 30%，小目标检测率下降 |
| `skip_frames_detection` | 2 | 3 | 检测频率降低，CPU/GPU 负载减少 |
| `skip_frames_depth` | 2 | 3 | 深度估计频率降低 |

---

## 模型配置 (ModelConfig)

> 在 `config.py` 中修改 `ModelConfig` 类的字段默认值。

### OWL-ViT 目标检测

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `owlvit_version` | `"v2"` | `"v1"`, `"v2"` | v2 精度更高但慢 20-30%；v1 速度更快适合实时场景 |
| `owlvit_model` | `"google/owlv2-base-patch16-ensemble"` | 任意 HuggingFace 模型路径 | 使用自定义微调模型时修改此项 |
| `owlvit_input_size` | `(384, 384)` | `(320,320)`, `(384,384)`, `(512,512)`, `(640,640)` | 尺寸越大检测越准但越慢；低配设备用 320，小目标检测用 512 |
| `owlvit_confidence_threshold` | `0.1` | `0.0 ~ 1.0` | 值越低检测越多（含误检）；值越高越严格（可能漏检）；误检多时提高到 0.2~0.3 |

### MiDaS 深度估计

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `midas_model` | `"MiDaS_small"` | `"MiDaS_small"`, `"DPT_Hybrid"`, `"DPT_Large"` | Small 约 30ms，Large 约 200ms；实时用 Small，精确测量用 Large |
| `midas_scale` | `0.5` | `0.1 ~ 1.0` | 值越小深度图越粗糙但越快；值越大越精细但越慢 |

### 手部追踪

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `hand_max_num` | `1` | `1`, `2` | 1=单手，2=双手（计算开销增加约 50%） |
| `hand_min_confidence` | `0.5` | `0.0 ~ 1.0` | MediaPipe 官方推荐值；光照差时可适当降低到 0.3 |

---

## 性能优化配置 (OptimizationConfig)

> 在 `config.py` 中修改 `OptimizationConfig` 类的字段默认值。

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `use_fp16` | `True` | `True`, `False` | 启用半精度可减少 50% 显存，提速 20-40%；CPU 设备自动禁用 |
| `skip_frames_detection` | `2` | `1`~`5` | 每隔 N 帧做一次检测；1=逐帧检测；实时场景 2-3；高精度跟踪 1 |
| `skip_frames_depth` | `2` | `1`~`5` | 每隔 N 帧做一次深度估计；建议与 skip_frames_detection 保持一致 |
| `device` | `"auto"` | `"auto"`, `"cuda"`, `"cpu"` | auto=自动检测 GPU；cuda=强制 GPU；cpu=强制 CPU；调试时指定具体设备 |

**注意事项：**
- 当 `device` 为 `"auto"` 时，系统启动时自动检测 CUDA 可用性
- 当 `device` 为 `"cpu"` 时，`use_fp16` 会被自动设为 `False`（CPU 不支持半精度加速）
- 旧 GPU（如 GTX 10 系列）FP16 加速效果不明显

---

## 空间引导配置 (GuidanceConfig)

> 在 `config.py` 中修改 `GuidanceConfig` 类的字段默认值。

### 基础阈值

| 配置项 | 默认值 | 单位 | 说明 |
|--------|--------|------|------|
| `horizontal_threshold` | `30` | 像素 | 目标中心与画面中心的水平距离超过此值时触发方向引导 |
| `vertical_threshold` | `30` | 像素 | 目标中心与画面中心的垂直距离超过此值时触发方向引导 |
| `depth_threshold` | `0.15` | 归一化值 (0~1) | 目标深度变化超过此值时触发前后移动引导 |

### 迟滞阈值 (Hysteresis)

迟滞机制防止状态频繁切换。每个方向有 enter（进入）和 exit（退出）两个阈值：

```
目标偏移 < enter → "已对准"
目标偏移 > exit  → "已偏离"
enter ≤ 偏移 ≤ exit → 保持上一状态
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `horizontal_threshold_enter` | `24` | 水平进入对准阈值（约为基础值的 80%） |
| `horizontal_threshold_exit` | `36` | 水平退出对准阈值（约为基础值的 120%） |
| `vertical_threshold_enter` | `24` | 垂直进入对准阈值 |
| `vertical_threshold_exit` | `36` | 垂直退出对准阈值 |
| `depth_threshold_enter` | `0.12` | 深度进入对准阈值 |
| `depth_threshold_exit` | `0.18` | 深度退出对准阈值 |

### 抓取状态判定

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `grasp_stable_frames` | `8` | 连续 N 帧"对准"才判定抓取稳定（约 0.3 秒 @ 30fps） |
| `grasp_release_frames` | `3` | 连续 N 帧"不对准"才判定释放（约 0.1 秒 @ 30fps） |

**调整建议：**
- 手抖严重：增大 enter/exit 阈值（如 30/42）
- 需要精细操作：减小 enter/exit 阈值（如 18/28）
- 抓取判定不稳定：增大 `grasp_stable_frames`

---

## 音频配置 (AudioConfig)

> 在 `config.py` 中修改 `AudioConfig` 类的字段默认值。

### ASR 语音识别

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `enable_asr` | `True` | `True`, `False` | 启用后可通过麦克风接收语音指令 |
| `whisper_model` | `"base"` | `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"` | 模型越大越准但越慢；tiny ~1GB, base ~1.5GB, medium ~5GB, large ~10GB |
| `asr_language` | `"zh,en"` | `"zh,en"`, `"zh"`, `"en"`, `"auto"` | 指定语言可提高准确率；auto 增加约 100ms 延迟 |

### TTS 语音合成

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `enable_tts` | `True` | `True`, `False` | 启用后系统播报检测结果和引导信息 |
| `tts_provider` | `"mimo"` | `"pyttsx3"`, `"mimo"` | pyttsx3=离线；mimo=云端（需 API Key） |
| `tts_rate` | `180` | `100 ~ 300` | 语速（每分钟字数）；150 正常偏快，180 较快 |
| `tts_volume` | `1.0` | `0.0 ~ 1.0` | 音量；1.0 最大，0.5 一半 |
| `tts_async` | `True` | `True`, `False` | 异步不阻塞主循环；同步需等待播报完成 |
| `tts_instruction_interval_sec` | `2.0` | `1.0 ~ 10.0` | 两次引导播报的最小间隔，防止语音重叠 |
| `tts_max_queue_size` | `1` | `1 ~ 5` | 播报队列长度；1=只保留最新消息 |

### 录音参数

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `record_sample_rate` | `16000` | 采样率（Hz）；16000 已满足 Whisper 需求 |
| `record_duration` | `5.0` | 单次录音最大时长（秒）；超时后自动识别 |

### 静音检测

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `auto_detect_silence` | `True` | 启用后检测到静音自动结束录音 |
| `silence_threshold` | `0.01` | 音量低于此值判定为静音；嘈杂环境可提高到 0.02~0.03 |
| `silence_duration` | `1.5` | 持续静音 N 秒后结束录音；语速慢时可增加到 2.0 |

### 语音反馈

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `voice_feedback_after_recording` | `True` | 录音结束后播放"正在识别..." |
| `voice_feedback_on_target_confirm` | `True` | 目标确认时播放"已找到目标" |
| `target_found_feedback_enabled` | `True` | 启用"目标已找到"反馈 |
| `target_found_frame_threshold` | `8` | 连续 N 帧检测到目标后触发反馈 |
| `target_missing_feedback_enabled` | `True` | 启用"目标丢失"反馈 |
| `target_missing_frame_threshold` | `45` | 连续 N 帧未检测到目标后触发反馈（约 1.5 秒） |
| `guidance_suppress_after_voice_sec` | `1.5` | 语音交互后抑制引导播报的时间，避免重叠 |

---

## LLM 视觉增强配置 (LLMVisionConfig)

> 在 `config.py` 中修改 `LLMVisionConfig` 类的字段默认值。

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `enable_llm_parsing` | `True` | `True`, `False` | 启用后将画面发送给 LLM 获取场景理解 |
| `poe_api_key` | `""` | — | **通过 .env 的 POE_API_KEY 设置，不要硬编码** |
| `poe_model` | `"deepseek-v3.2"` | `"deepseek-v3.2"`, `"claude-3.5-sonnet"`, `"gpt-4o"` | 不同模型有不同视觉理解能力；deepseek 性价比高，claude 视觉最强 |
| `poe_timeout_sec` | `5.0` | `1.0 ~ 30.0` | API 超时时间；网络差时可增加到 10.0 |
| `max_frames_for_vision` | `4` | `1 ~ 10` | 发送给 LLM 的帧数；越多上下文越丰富但费用越大 |
| `api_retry_count` | `1` | `0 ~ 5` | API 失败重试次数；1 次可应对偶发网络抖动 |

---

## 日志配置 (LoggingConfig)

> 在 `config.py` 中修改 `LoggingConfig` 类的字段默认值。

| 配置项 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `log_dir` | `"logs"` | 任意路径 | 日志文件存储目录；不存在时自动创建 |
| `log_level` | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` | DEBUG 最详细适合开发调试；INFO 推荐用于生产环境 |
| `log_to_file` | `True` | `True`, `False` | 是否将日志写入文件 |
| `log_to_console` | `True` | `True`, `False` | 是否输出到终端；生产环境可设为 False 减少输出 |
| `enable_fps_stats` | `True` | `True`, `False` | 是否统计各模块帧率数据 |
| `fps_window_size` | `30` | `10 ~ 100` | FPS 统计滑动窗口大小；30 帧约 1 秒 @ 30fps |

---

## 系统级配置 (SystemConfig)

> 在 `config.py` 中修改 `SystemConfig` 类的字段默认值。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `target_queries` | `["a cup", "a bottle"]` | 默认检测目标列表；使用英文名词短语；运行时可被 `core/system.py` 动态修改 |
| `camera_width` | `640` | 摄像头画面宽度（像素）；可选 320/640/1280/1920 |
| `camera_height` | `480` | 摄像头画面高度（像素）；推荐 640x480 (4:3) 或 1280x720 (16:9) |

---

## 环境变量与 API Key

### `.env` 文件

所有 API Key 统一在 `.env` 文件中管理：

```env
# 小米 MiMo TTS 服务 API Key
MIMO_API_KEY=your_mimo_api_key_here

# Poe LLM 服务 API Key
POE_API_KEY=your_poe_api_key_here

# OpenAI API Key（可选，用于云端 Whisper 或 GPT）
# OPENAI_API_KEY=your_openai_api_key_here
```

### 支持的变量

| 环境变量 | 影响配置 | 说明 |
|----------|----------|------|
| `MIMO_API_KEY` | `config.audio.mimo_api_key` | MiMo 云端 TTS 密钥 |
| `XIAOMI_MIMO_API_KEY` | `config.audio.mimo_api_key` | MiMo 密钥的别名 |
| `POE_API_KEY` | `config.llm_vision.poe_api_key` | Poe LLM 服务密钥 |

### 获取 API Key

- **MiMo TTS**: 访问 [小米开放平台](https://open.mi.com/) 申请
- **Poe API**: 访问 [poe.com/api](https://poe.com/api) 注册获取
- **OpenAI**: 访问 [platform.openai.com/api-keys](https://platform.openai.com/api-keys) 创建

---

## 常见问题

### Q: 如何修改所有 profile 共享的默认值？

A: 直接编辑 `config.py` 中对应 dataclass 字段的默认值。例如修改默认语速：

```python
# config.py 中
@dataclass
class AudioConfig:
    tts_rate: int = 180  # 将默认值从 150 改为 180
```

### Q: 如何添加自定义 profile？

A: 在 `config.yaml` 的 `profiles:` 段下添加新条目：

```yaml
profiles:
  my-custom:
    model:
      owlvit_input_size: [512, 512]
    audio:
      enable_tts: true
      tts_rate: 200
```

然后用 `python main.py --config my-custom` 启动。

### Q: 修改配置后需要重启吗？

A: 是的，配置在系统启动时加载。修改 `config.py` 或 `config.yaml` 后需要重启程序。

### Q: 为什么我的 API Key 不生效？

A: 检查以下几点：
1. `.env` 文件是否在**项目根目录**
2. Key 值前后是否有空格
3. 重启程序使新配置生效
4. 查看日志中是否有 "LLM Vision Parser initialized" 或 "MiMo TTS initialized" 等成功消息

### Q: 如何查看当前加载的完整配置？

A: 运行以下命令：

```bash
python config.py
# 或指定 profile
python config.py fast
```

这将以 JSON 格式输出所有配置项的当前值。

### Q: 低配设备如何优化？

A: 推荐组合：
1. 使用 `--config fast` 预设
2. 在 `config.py` 中将 `owlvit_input_size` 改为 `(320, 320)`
3. 将 `midas_model` 改为 `"MiDaS_small"`（如果还不是）
4. 将 `skip_frames_detection` 和 `skip_frames_depth` 增加到 3~4
5. 关闭不需要的功能（ASR、TTS、LLM Vision）
