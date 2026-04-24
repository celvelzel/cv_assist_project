# 配置设置指南

> 本指南以 **当前代码实现** 为准（`config.py` + `config.yaml` + `.env`）。

---

## 1. 配置架构

| 文件 | 作用 |
|---|---|
| `config.py` | 所有 dataclass 默认值（系统事实基线） |
| `config.yaml` | profile 覆盖（只写与默认值不同的项） |
| `.env` | 环境变量覆盖（最高优先级） |

加载优先级：

```text
.env > config.yaml profile override > config.py 默认值
```

---

## 2. 快速使用

```bash
# 默认平衡配置
python main.py --config balanced

# 低资源配置
python main.py --config fast

# 纯视觉（关闭 ASR/TTS）
python main.py --config no-voice

# 极低资源模式
python main.py --config light
```

---

## 3. Profile 说明（当前实现）

### 3.1 `balanced`（默认）

- 来自 `config.py` 的默认值
- `config.yaml` 不需要定义 `balanced`

### 3.2 `fast`

- `owlvit_input_size` 降到 `320x320`
- 检测/深度跳帧都提升到 `3`

### 3.3 `no-voice`

- `audio.enable_asr = false`
- `audio.enable_tts = false`

### 3.4 `light`

- `owlvit_input_size = 256x256`
- `midas_model = MiDaS_small`
- `skip_frames_detection = 5`
- `skip_frames_depth = 5`

> 说明：CLI 目前兼容 `voice/tts/mimo-tts` 参数值，但若 `config.yaml` 未定义对应 profile override，会回落到默认配置行为。

---

## 4. 关键默认值速览（来自 `config.py`）

## 4.1 SystemConfig

| 字段 | 默认值 |
|---|---|
| `camera_id` | `1` |
| `camera_width` | `848` |
| `camera_height` | `480` |
| `target_queries` | `[]` |

## 4.2 ModelConfig

| 字段 | 默认值 |
|---|---|
| `owlvit_version` | `v2` |
| `owlvit_model` | `google/owlv2-base-patch16-ensemble` |
| `owlvit_input_size` | `(384, 384)` |
| `owlvit_confidence_threshold` | `0.3` |
| `midas_model` | `DPT_Large` |
| `midas_scale` | `0.5` |
| `hand_max_num` | `1` |
| `hand_min_confidence` | `0.2` |

## 4.3 OptimizationConfig

| 字段 | 默认值 |
|---|---|
| `use_fp16` | `True`（CPU 下会自动关闭） |
| `skip_frames_detection` | `2` |
| `skip_frames_depth` | `2` |
| `device` | `auto` |

## 4.4 AudioConfig（节选）

| 字段 | 默认值 |
|---|---|
| `enable_asr` | `True` |
| `whisper_model` | `medium` |
| `asr_language` | `zh,en` |
| `enable_tts` | `True` |
| `tts_provider` | `mimo` |
| `tts_rate` | `200` |
| `tts_instruction_interval_sec` | `3.0` |
| `voice_v_long_press_sec` | `0.45` |
| `record_sample_rate` | `16000` |

## 4.5 GuidanceConfig（节选）

| 字段 | 默认值 |
|---|---|
| `horizontal_threshold` / `vertical_threshold` | `30 / 30` |
| `depth_threshold` | `0.15` |
| `horizontal enter/exit` | `24 / 36` |
| `vertical enter/exit` | `24 / 36` |
| `depth enter/exit` | `0.2 / 0.3` |
| `grasp_stable_frames` / `grasp_release_frames` | `5 / 3` |
| `clock_mode` | `horizontal_plane` |

## 4.6 LLMVisionConfig

| 字段 | 默认值 |
|---|---|
| `enable_llm_parsing` | `True` |
| `poe_model` | `deepseek-v3.2` |
| `poe_timeout_sec` | `5.0` |
| `max_frames_for_vision` | `1` |
| `api_retry_count` | `1` |

## 4.7 LoggingConfig（节选）

| 字段 | 默认值 |
|---|---|
| `log_level` | `INFO` |
| `enable_fps_stats` | `True` |
| `enable_task_metrics` | `True` |
| `task_metrics_interval_sec` | `1.0` |
| `task_target_search_timeout_sec` | `30.0` |
| `task_lost_target_window_sec` | `4.0` |

---

## 5. 环境变量与 API Key

`.env` 示例：

```env
MIMO_API_KEY=your_mimo_api_key_here
POE_API_KEY=your_poe_api_key_here
TARGET_QUERY=a cup
# 或
TARGET_QUERIES=a cup,a bottle
```

当前代码支持变量：

| 环境变量 | 对应字段 |
|---|---|
| `MIMO_API_KEY` / `XIAOMI_MIMO_API_KEY` | `config.audio.mimo_api_key` |
| `POE_API_KEY` | `config.llm_vision.poe_api_key` |
| `TARGET_QUERY` | `config.target_queries`（单目标） |
| `TARGET_QUERIES` | `config.target_queries`（逗号分隔） |

---

## 6. 常见问题

### Q1: 我改了配置，为什么没生效？

- 检查 `.env` 是否覆盖了同名项
- 修改后需要重启程序

### Q2: 如何查看最终合并后的配置？

```bash
python config.py
python config.py fast
```

### Q3: 低配机器建议怎么调？

1. 先用 `--config light`
2. 如仍卡顿，切 `--config no-voice`
3. 再按需降低模型与播报频率
