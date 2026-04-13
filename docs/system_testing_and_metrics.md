# 系统测试与指标打点说明

本文档整理 **CV Assist** 项目中与「自动化测试」「运行时指标采集（打点）」以及 **系统运行过程中终端日志输出** 的说明，包含代码路径、数据流、配置项与典型日志含义，便于维护与排障。

---

## 一、系统测试体系

### 1.1 测试代码位置

所有测试位于仓库根目录下的 `tests/`：

| 文件 | 框架 | 主要内容 |
|------|------|----------|
| `test_all.py` | `unittest` | 冒烟：验证 `detectors` / `core` / `utils` 可导入 |
| `test_config_loading.py` | `unittest` | 配置加载行为 |
| `test_guidance.py` | `unittest` | 引导控制器滞回与稳定性 |
| `test_logging.py` | 脚本式 `main()` | 日志配置与 `FPSCounter` 行为 |
| `test_audio.py` | 脚本式 `main()` | TTS、录音、ASR 子系统（依赖硬件/环境时可能需人工环境） |
| `test_llm_vision.py` | `pytest` | LLM 视觉解析与 OpenCV 相关用例（无 OpenCV 时部分用例跳过） |
| `test_asr_language_mode.py` | `unittest` | ASR 语言模式相关逻辑 |
| `test_task_metrics.py` | `unittest` | **任务指标采集器**、语音事件解析、异步报告写入 |
| `test_system_tts_policy.py` | `unittest` | **系统级 TTS 策略**与任务指标在桩对象上的联动（不跑真实摄像头） |

### 1.2 如何运行

- **unittest 单文件**（与 README 中部分说明一致）：

  ```bash
  python tests/test_all.py
  python tests/test_guidance.py
  python tests/test_task_metrics.py
  python tests/test_system_tts_policy.py
  ```

- **pytest**（用于 `test_llm_vision.py`）：

  ```bash
  pytest tests/test_llm_vision.py -v
  ```

- **脚本型测试**（直接执行，内部打印通过/失败）：

  ```bash
  python tests/test_logging.py
  python tests/test_audio.py
  ```

### 1.3 测试分层（概念）

1. **纯单元 / 逻辑测试**：不依赖摄像头，如 `test_guidance.py`、`test_task_metrics.py` 中对 `TaskMetricsCollector` 的「就绪 + X 轴位移触发成功」序列。
2. **子系统集成**：`test_audio.py`、`test_llm_vision.py` 可能依赖外部 API、模型或本机设备。
3. **系统行为桩测**：`test_system_tts_policy.py` 通过 `CVAssistSystem.__new__` 构造桩实例，配合 `TaskMetricsCollector` 与假 TTS，验证播报策略与指标侧不破坏主流程契约。

### 1.4 与「指标打点」强相关的测试

- `tests/test_task_metrics.py`：`FrameMetrics` 构造、`record_frame` / `finish_task` / `build_terminal_summary`、抓取成功判定窗口、`AsyncReportWriter` 队列与写盘。
- `tests/test_system_tts_policy.py`：在接近真实配置的 `logging` 桩上验证 TTS 与任务状态；会调用 `task_metrics_collector.start_task` 等。

---

## 二、指标打点（监测）体系

本项目的「打点」主要指三类输出：

1. **常规日志**：`utils/logger.py` 的 `setup_logging`（文件 + 控制台）。
2. **FPS 统计**：`FPSCounter`（处理管线帧时；主循环中还可维护端到端 FPS），由 `LoggingConfig.enable_fps_stats` 等控制。
3. **任务级指标**：`utils/task_metrics.py` 中的 `TaskMetricsCollector`、`AsyncReportWriter`，与 `core/system.py` 主循环深度集成。

以下重点说明第3 类（任务指标），因其字段多、与业务状态机耦合紧。

### 2.1 核心模块与数据类型

| 组件 | 文件 | 职责 |
|------|------|------|
| `FrameMetrics` | `utils/task_metrics.py` | 单帧冻结事实：各阶段耗时、检测/深度是否执行、目标与手几何、引导状态、就绪与 FPS 等 |
| `TaskMetricsCollector` |同上 | 任务生命周期内聚合指标、**结束原因判定**（丢失目标 /抓取成功等）、生成终端摘要行与 JSON 报告字典 |
| `TaskReportEnvelope` | 同上 | 主线程投递给写盘线程的快照 |
| `AsyncReportWriter` | 同上 | 后台线程异步 `json.dump`，避免阻塞主循环 |

### 2.2 配置项（`config.py` → `LoggingConfig`）

与任务打点直接相关的字段包括：

- `enable_task_metrics`：总开关；为 `False` 时不写 JSON、不输出 `[task]` 周期摘要（主循环中相关调用会短路）。
- `task_metrics_dir`：JSON 报告目录（默认 `logs/task_metrics`）。
- `task_metrics_interval_sec`：任务进行中 **`[task]` 单行摘要** 的输出周期。
- `task_ready_confirm_window_sec`：进入「就绪」后的确认窗口（与采集器内 `ready_confirm_window_sec` 对应）。
- `task_start_confirm_window_sec`：**语音触发后**，目标需连续被检测到若干秒才真正 `start_task`（激活前为「确认」阶段，此间不计入任务指标起点）。
- `task_lost_target_window_sec`：目标连续不可见超过该时长则 `pending_end_reason = lost_target`。
- `catch_x_min_displacement_px` / `catch_x_stable_frames` / `catch_x_stable_max_std_px` / `catch_hand_near_target_px`：**抓取成功**的横向位移与稳定性判定参数。

FPS 相关：`enable_fps_stats`、`fps_window_size`（`FPSCounter` 滑动窗口）。

### 2.3 在 `CVAssistSystem` 中的生命周期

1. **初始化**（`_init_task_metrics`）  
   - 生成 `session_id`。  
   - 构造 `TaskMetricsCollector`（传入 `guidance.grasp_stable_frames` 与上述 logging 阈值）。  
   - 若 `enable_task_metrics`：创建并 `start()` `AsyncReportWriter` 守护线程。

2. **任务开始**  
   - 语音路径：先 `_pending_task` + 确认窗口，满足后 `_activate_pending_task()` → `task_metrics_collector.start_task(...)`，并可选 `record_voice_metrics(...)`（含 ASR、录音、TTS 提示、Poe 耗时等）。  
   - 预设目标立即启动：`_start_task_now()` → `start_task(..., target_search_to_activate_sec=None)`。

3. **运行中（主循环每帧）**  
   - 若存在 `current_task`：  
     - `_build_frame_metrics(...)` → `record_frame(frame_metrics)`。  
     - 语音抑制窗结束后的首帧处理耗时：`record_post_suppress_first_frame_process_ms`。  
     - 读帧失败：`record_camera_read_failure()`。  
     - `should_finish_task()` 非空则播报并 `_finish_current_task(end_reason)`。  
     - 否则 `_maybe_log_task_summary()`：按间隔调用 `_sample_task_resources()` + `build_terminal_summary()` → `logger.info`输出 **`[task]` 行**。

4. **引导 TTS 打点**  
   - 每次 `_speak_guidance` 成功调用 TTS 后，若 `enable_task_metrics` 为真，会调用 `note_first_guidance_instruction_tts(now)`（采集器内仅记录**首次**），用于摘要中的 `tts1=+Xs`。

5. **资源采样**（`_sample_task_resources`）  
   - 在输出周期摘要前触发：CPU、RSS、可选 `torch.cuda.memory_allocated`、以及基于 `psutil` 的全机网络累计差分换算的 KB/s。未安装 `psutil` 时仍可能写入 GPU 采样。

6. **任务结束**（`_finish_current_task`）  
   - `should_emit_report()`：仅当任务期间**至少有一次目标可见**（`first_target_detected_ts` 与 `target_visible_frames`）才入队 JSON。  
   - `finish_task(end_reason, ...)` 得到 `report_dict`，`AsyncReportWriter.enqueue(TaskReportEnvelope(...))`。  
   - `end_reason == success` 时额外 `logger.info` 打印 `build_success_console_report` 多行成功简报。

### 2.4 任务结束判定（采集器内部）

`TaskMetricsCollector._update_completion_state` 在每帧 `record_frame` 末尾执行，主要包括：

- **丢失目标**：目标不可见持续 ≥ `lost_target_window_sec` → `pending_end_reason = lost_target`（可被更高优先级原因覆盖策略需结合代码当前顺序理解；成功路径会设 `finishing`）。  
- **抓取成功**：在「手曾靠近目标」或「曾进入就绪」等前置下，对目标 X 坐标滑动窗口求位移与标准差，满足 `catch_*` 阈值则 `pending_end_reason = success`，并记录 `catch_trigger`（如 `target_x_displacement_ready` 或基于手靠近的基准）。

主循环读到 `should_finish_task()` 后即结束任务并清空 `current_task`。

### 2.5 输出物对照

| 输出 | 触发条件 | 说明 |
|------|----------|------|
| `[task] ...` 单行 | `enable_task_metrics` 且任务进行中，且到达 `task_metrics_interval_sec` | 见下文引用文档 |
| `logs/task_metrics/task_metrics_<HKT时间>_<task_id>.json` | 任务结束且 `should_emit_report()` 为真 | 结构化多章节报告（会话、任务、时延分位、语音、质量、完成判定等） |
| 成功多行块 | `end_reason == success` | `build_success_console_report` |
| 常规 FPS 日志 | `enable_fps_stats`，每 100 帧等 | `run()` 主循环内 `logger.info` |

### 2.6 `[task]` 行字段详解

终端单行各段含义已单独维护，避免本文重复冗长字段表：

- 请参阅 **[`docs/task_log_line_fields.md`](task_log_line_fields.md)**（与 `TaskMetricsCollector.build_terminal_summary()` 一一对应）。

### 2.7 JSON 报告与时间字段

- `finish_task` 生成的字典中，带 `_ts` / `_time` / `_at` 后缀的数值字段在写入 JSON 时会被格式化为 **香港时间（HKT）可读字符串**（见 `_field_entry`），便于人工阅读；耗时类多处以 `*_sec` 键名、固定小数位汇总。

---

## 三、系统运行过程中终端输出日志说明

本节描述 **`python main.py` / `core.system:main()` → `CVAssistSystem.run()`** 在典型流程下，终端（及同步写入的日志文件）中会出现的日志类型。实现入口为 `CVAssistSystem.__init__` 中的 `setup_logging`（`utils/logger.py`）以及 `core/system.py` 内大量 `logger.info` / `warning` / `error` / `debug`。

### 3.1 日志格式与开关

- **行格式**（根日志器：`setup_logging` 内 `logging.Formatter`）：  
  `[asctime] [logger名称] [LEVEL] 消息`  
  例如：`[2026-04-12 10:00:00] [core.system] [INFO] 开始主循环`  
  其中 `logger名称` 多为发出记录的模块名（如 `core.system`、`utils.logger`）。
- **配置**（`config.py` → `LoggingConfig`）：  
  - `log_level`：`DEBUG` / `INFO` / `WARNING` / `ERROR` — 低于设定级别的行**不会**出现在终端或文件。  
  - `log_to_console`：是否打印到终端。  
  - `log_to_file`：是否写入 `log_dir` 下带时间戳的 `cv_assist_*.log`（与终端内容同源、同级别）。  
- **第三方库降噪**：`setup_logging` 会将 `httpx`、`huggingface_hub`、`PIL`、`matplotlib` 等设为 `WARNING`，减少刷屏。

### 3.2 启动与组件初始化阶段

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `日志系统初始化完成` / `日志文件: ...` | INFO | `utils.logger` 在 `setup_logging` 成功后输出；随后才有 `core.system` 的横幅 |
| `============================================================` 与 `CV 视觉辅助系统`、OWL-ViT / MiDaS / 设备 / FP16 | INFO | 配置与模型概况 |
| `初始化检测器组件...` / `所有组件初始化成功` | INFO | 检测器流水线就绪 |
| 各模型 `加载失败` | ERROR | OWL-ViT、手部、深度等子模块异常，常伴随 `exc_info` 栈 |
| `正在初始化 ASR/TTS/LLM Vision...` 与成功或未启用说明 | INFO | 音频与可选视觉解析 |
| `音频模块不可用，请安装依赖...` | WARNING | 配置了 ASR/TTS 但缺少可选依赖 |
| `音频模块未安装...`（模块级） | WARNING | 在部分导入失败场景下出现 |
| `系统初始化完成` | INFO | 构造 `CVAssistSystem` 结束，即将进入 `run()` |

### 3.3 进入 `run()`：摄像头与交互提示

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `启动 CV 视觉辅助系统` | INFO | 主循环即将开始 |
| `控制: q - 退出, d - 切换深度显示`；可选 `v - 语音输入` | INFO | 键盘操作说明 |
| `TTS 已启用...` /首次 `语音播报已开启`（TTS 播报，非纯文本日志） | INFO + 音频 | 提示用户播报链路已开 |
| `检测目标: [...]`、`摄像头选择: N` | INFO | 当前目标列表与设备 ID |
| `摄像头 N 初始化成功，分辨率: WxH` | INFO | 采流正常 |
| `摄像头初始化失败`及 `请检查：1. ... 4. ...` | ERROR | 无法打开或无法读首帧，程序通常在此 `return`，不进入主循环 |
| `开始主循环` | INFO | 进入 `while True` 帧循环 |

### 3.4 主循环内周期性输出

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `FPS 统计 [帧 N] \|处理: 当前=... 平均=... \|端到端: ...`（或仅处理一行） | INFO | 约每 **100 帧** 输出一次；需 `enable_fps_stats` 且存在 `fps_counter` |
| `深度显示: 开启/关闭` | INFO | 用户按 `d` |
| `帧处理错误: ...` | ERROR | `process_frame` 异常，本帧跳过，循环继续 |
| `无法读取摄像头帧...` / `摄像头读取错误` | WARNING / INFO | 读帧失败或 OpenCV 异常，通常随后 `break` 并记录退出原因 |
| `主循环结束: ...` | INFO | 见下节「退出原因」 |

### 3.5 任务与指标相关日志（与第二节衔接）

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `任务进入确认等待: task_id=... 需持续检测 X秒` | INFO | 语音设置目标后，等待目标稳定可见达到 `task_start_confirm_window_sec` |
| `任务正式激活: task_id=...` | INFO | 确认通过，`start_task` 开始计时与采集 |
| `预设目标任务已自动激活: ...` | INFO | 配置中 `target_queries` 非空时的自动任务路径 |
| **`[task] \| task_0001 \| ...`**（整行） | INFO | 任务进行中按 `task_metrics_interval_sec` 输出的**实时指标摘要**；字段释义见 [`task_log_line_fields.md`](task_log_line_fields.md) |
| `任务未检测到目标，跳过报告写入: task_id=...` | INFO | 结束任务时从未稳定看到目标，不写 JSON |
| 多行 `========================================================` 包裹的 `[SUCCESS] 抓取任务完成` 块 | INFO | **仅成功**结束时由 `build_success_console_report` 打印 |
| `任务终止...` / `任务完成...`（TTS 生命周期播报） | — | 用户听到的是语音；终端仍可能有对应流程中的 INFO（如结束任务前的逻辑） |

### 3.6 语音交互相关

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `语音输入进行中，请稍候...` | INFO | 重复按 `v` 时后台仍在录音识别 |
| `=== 开始语音录制 ===`、`录音中...`、`录音完成，开始识别...` | INFO | 录音与 ASR 流程阶段 |
| `未录制到音频` / `识别结果为空` | WARNING | 无有效音频或 ASR 无文本 |
| `语音事件解析成功: action=... target=...` | INFO | `parse_voice_event` 成功 |
| `无法解析指令: '...'` | WARNING | 无法理解用户语句 |
| `语音输入处理失败` | ERROR | `_handle_voice_input` 异常 |
| `检测目标已更新为: ...` | INFO | 新目标已写入配置并进入搜索/确认流程 |
| `收到语音退出指令` | INFO | 用户说出退出类指令，随后可能 `主循环结束: 语音指令请求退出程序` |

### 3.7 进程结束与汇总

| 典型内容 | 级别 | 含义概要 |
|----------|------|----------|
| `主循环结束: 用户按下 q 键退出` | INFO | `exit_reason = user_quit` |
| `主循环结束: 视频窗口已关闭或变为不可见...` | INFO | `exit_reason = window_closed` |
| `主循环结束: 摄像头读帧失败` / `摄像头读取异常` | INFO | `camera_lost` / `camera_error` |
| `主循环结束: 语音指令请求退出程序` | INFO | `voice_exit` |
| `收到键盘中断信号` | INFO | Ctrl+C，`interrupt` |
| `本次运行退出原因: <code>` | INFO | 统一打出本次 `exit_reason`，便于对照日志 |
| 若仍有未结束任务 | — | `finally` 中可能以 `user_exit` / `error` 等原因调用 `_finish_current_task` |
| `============================================================` + `最终统计` + 总帧数与各 FPS | INFO | 退出前汇总；端到端 FPS 依赖 `e2e_fps_counter` |
| `系统已关闭` | INFO | 释放摄像头、销毁窗口、停止 `AsyncReportWriter` 之后 |

### 3.8 `DEBUG` 级别下额外可见内容

将 `logging.log_level` 设为 `DEBUG` 后，除上述 INFO 外，还可能看到例如：

- 任务确认阶段：`任务确认期间目标消失，重置计时`  
- 主循环：`检测到 N 个目标`、LLM 视觉帧缓冲 `Captured ... frames` / `Frame buffer is empty`  
- 近距离提示音：`近距离滴滴声模块未加载`（模块导入失败时也可能仅在 DEBUG出现）

生产环境通常用 `INFO`；需要逐帧或语音排障时再临时改为 `DEBUG`。

---

## 四、相关源码索引（便于跳转）

| 主题 | 路径 |
|------|------|
| 主循环与打点调用 | `core/system.py`（`run`、`_build_frame_metrics`、`_maybe_log_task_summary`、`_finish_current_task`） |
| 采集与报告 | `utils/task_metrics.py` |
| FPS 与日志初始化 | `utils/logger.py` |
| 配置定义 | `config.py` → `LoggingConfig` |
| 任务日志行字段 | `docs/task_log_line_fields.md` |
| 终端日志文案（汇总） | 本节「三、系统运行过程中终端输出日志说明」+ `core/system.py` 内 `logger.*` |

---

## 五、维护建议

- 修改 **`[task]`** 行格式时：同步更新 `docs/task_log_line_fields.md`，并补充或调整 `tests/test_task_metrics.py` 中断言。  
- 修改 **结束判定**（丢失目标 / 抓取成功）时：优先在 `TaskMetricsCollector` 内保持纯逻辑，再在 `system.py` 中接播报与状态；对应单元测试应覆盖边界帧序列。  
- 新增 **语音链路耗时** 字段时：同时更新 `record_voice_metrics`、`finish_task` 的 `voice_summary` 段，以及 ASR 侧 `voice_event` 字典契约。
