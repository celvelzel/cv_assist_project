"""
配置模块
===============
集中管理系统所有组件的配置参数。

本模块是系统唯一的配置定义来源（Single Source of Truth）。
所有默认值均在此处的 dataclass 字段中定义。

配置加载优先级（从高到低）:
  1. 环境变量（.env 文件）  — 如 MIMO_API_KEY、POE_API_KEY
  2. config.yaml profile overrides — 预设覆盖（如 fast、tts 等）
  3. dataclass 默认值 — 本文件中定义的默认值

用法:
    from config import load_config
    config = load_config(profile="tts")
"""

import os
import sys
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Tuple, Dict, Any, Type, get_type_hints
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构 — 所有配置类
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """模型相关配置。

    包含 OWL-ViT 开放词汇目标检测模型、MiDaS 深度估计模型
    以及 MediaPipe 手部追踪的参数配置。

    调整建议:
    - 低配设备: 降低 owlvit_input_size 至 (320, 320)，使用 v1 版本
    - 高精度需求: 使用 v2 版本，增大 input_size 至 (512, 512)
    - 手部追踪: hand_max_num 设为 2 可检测双手，但会增加计算开销
    """
    # ---- OWL-ViT 目标检测设置 ----

    owlvit_version: str = "v2"
    # OWL-ViT 模型版本。
    # 可选值: "v1" (google/owlvit-base-patch32), "v2" (google/owlv2-base-patch16-ensemble)
    # 效果: v2 检测精度更高，但推理速度比 v1 慢约 20-30%。
    #       v1 适合实时性要求高的场景，v2 适合精度优先的场景。

    owlvit_model: str = "google/owlv2-base-patch16-ensemble"
    # OWL-ViT 模型名称或本地模型路径。
    # 默认使用 HuggingFace 官方模型。如需使用微调后的自定义模型，
    # 可在此填入本地路径（如 "models/my-owlvit-finetuned"）。
    # 注意: 当 owlvit_version 为 "v1" 或 "v2" 时，此字段会被 get_owlvit_model_name() 覆盖。

    owlvit_input_size: Tuple[int, int] = (384, 384)
    # OWL-ViT 模型输入图像的宽高（像素）。
    # 可选值: (320, 320) / (384, 384) / (512, 512) / (640, 640)
    # 效果: 尺寸越大检测精度越高，但推理时间呈平方增长。
    #       (320, 320) 适合低配设备或高帧率场景；(512, 512) 适合小目标检测。
    # 推荐: 默认 (384, 384) 在精度和速度之间取得良好平衡。

    owlvit_confidence_threshold: float = 0.3
    # OWL-ViT 检测置信度阈值。低于此值的检测结果将被过滤。
    # 范围: 0.0 ~ 1.0
    # 效果: 值越低检测到的目标越多（包含更多误检）；值越高检测越严格（可能漏检）。
    # 推荐: 0.1 适合开放词汇场景（因为 OWL-ViT 对未见过的类别置信度普遍偏低）。
    #       如果误检太多，可提高到 0.2~0.3。

    # ---- MiDaS 深度估计设置 ----

    midas_model: str = "DPT_Large"
    # MiDaS 深度估计模型名称。
    # 可选值: "MiDaS_small" (最快，精度较低), "DPT_Large" (最慢，精度最高),
    #        "DPT_Hybrid" (中等速度，中等精度)
    # 效果: MiDaS_small 推理速度约 30ms，DPT_Large 约 200ms。
    # 推荐: 实时场景使用 MiDaS_small；需要精确深度测量时使用 DPT_Large。

    midas_scale: float = 0.5
    # MiDaS 深度图缩放因子。
    # 范围: 0.1 ~ 1.0
    # 效果: 值越小深度图分辨率越低、速度越快；值越大深度图越精细但越慢。
    # 推荐: 0.5 在大多数场景下表现良好。

    # ---- MediaPipe 手部追踪设置 ----

    hand_max_num: int = 1
    # 最大检测手部数量。
    # 可选值: 1 (单手), 2 (双手)
    # 效果: 设为 2 可同时检测双手，但 MediaPipe 计算开销增加约 50%。
    # 推荐: 单手操作场景设为 1；需要双手交互场景设为 2。

    hand_min_confidence: float = 0.2
    # 手部检测最小置信度阈值。
    # 范围: 0.0 ~ 1.0
    # 效果: 值越高手部检测越严格（可能漏检部分手部）；值越低越容易检测到手部（可能误检）。
    # 推荐: 0.5 是 MediaPipe 官方推荐值，在大多数光照条件下表现稳定。

    # ---- 工具方法 ----

    def get_owlvit_model_name(self) -> str:
        """根据 owlvit_version 返回对应的默认 HuggingFace 模型名称。

        返回:
            v1 -> "google/owlvit-base-patch32"
            v2 -> "google/owlv2-base-patch16-ensemble"
        """
        if self.owlvit_version == "v2":
            return "google/owlv2-base-patch16-ensemble"
        else:
            return "google/owlvit-base-patch32"


@dataclass
class OptimizationConfig:
    """性能优化相关配置。

    控制模型推理的计算设备、精度模式以及帧跳过策略。
    这些参数直接影响系统的运行帧率和资源占用。

    调整建议:
    - CPU 设备: use_fp16 会被自动禁用（CPU 不支持半精度加速）
    - 低帧率设备: 增大 skip_frames_detection 和 skip_frames_depth
    - 高帧率设备: 设为 1 实现逐帧检测
    """

    use_fp16: bool = True
    # 是否启用 FP16（半精度浮点数）推理加速。
    # 可选值: True (启用), False (禁用)
    # 效果: 启用后可减少约 50% 显存占用，推理速度提升 20-40%（取决于 GPU）。
    # 注意: 仅在 CUDA 设备上有效。CPU 设备会自动禁用此选项。
    #       某些旧 GPU（如 GTX 10 系列）FP16 加速效果不明显。

    skip_frames_detection: int = 2
    # OWL-ViT 目标检测的帧跳过间隔。
    # 含义: 每隔 N 帧执行一次目标检测。设为 2 表示每 2 帧检测一次。
    # 效果: 值越大检测频率越低、CPU/GPU 负载越小，但目标位置更新延迟增加。
    # 推荐: 实时场景设为 2-3；高精度跟踪设为 1（逐帧检测）。

    skip_frames_depth: int = 2
    # MiDaS 深度估计的帧跳过间隔。
    # 含义: 每隔 N 帧执行一次深度估计。
    # 效果: 深度估计计算量大，适当跳过可显著提升帧率。
    # 推荐: 与 skip_frames_detection 保持一致，设为 2-3。

    device: str = "auto"
    # 模型推理的计算设备。
    # 可选值: "auto" (自动检测), "cuda" (强制 GPU), "cpu" (强制 CPU)
    # 效果: "auto" 模式下，系统启动时自动检测 CUDA 可用性。
    #       有 GPU 则使用 "cuda"，否则回退到 "cpu"。
    # 注意: 当 device 为 "cpu" 时，use_fp16 会被自动设为 False。
    # 推荐: 保持 "auto"，除非需要强制指定设备进行调试。

    def __post_init__(self):
        """初始化后自动处理设备检测逻辑。

        当 device 为 "auto" 时:
        1. 尝试导入 torch 并检测 CUDA 可用性
        2. 有 CUDA -> device = "cuda"
        3. 无 CUDA 或 torch 未安装 -> device = "cpu"
        4. CPU 模式下自动禁用 FP16
        """
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        if self.device == "cpu":
            self.use_fp16 = False


@dataclass
class GuidanceConfig:
    """空间引导（Guidance）阈值配置。

    控制系统如何根据目标在画面中的位置（水平、垂直、深度）
    来判断用户是否"对准"了目标，以及何时判定"抓取稳定"。

    迟滞机制（Hysteresis）:
    每个方向都有 enter（进入）和 exit（退出）两个阈值，形成迟滞区间。
    例如水平方向: enter=24, exit=36
    - 目标偏移 < 24 像素 -> 判定为"对准"
    - 目标偏移 > 36 像素 -> 判定为"偏离"
    - 24 ~ 36 像素之间 -> 保持上一状态（防止频繁切换）

    调整建议:
    - 手抖严重: 增大 enter/exit 阈值（如 30/42）
    - 需要精细操作: 减小 enter/exit 阈值（如 18/28）
    - 抓取判定不稳定: 增大 grasp_stable_frames
    """

    horizontal_threshold: int = 30
    # 水平方向判定阈值（像素）。
    # 含义: 目标中心与画面中心的水平距离超过此值时，触发方向引导。
    # 推荐: 30 像素适合 640x480 分辨率。分辨率变化时需相应调整。

    vertical_threshold: int = 30
    # 垂直方向判定阈值（像素）。
    # 含义: 目标中心与画面中心的垂直距离超过此值时，触发方向引导。

    depth_threshold: float = 0.15
    # 深度变化判定阈值（归一化值）。
    # 含义: 目标深度变化超过此值时，触发前后移动引导。
    # 范围: 0.0 ~ 1.0，值越小对深度变化越敏感。

    # ---- 迟滞阈值（Hysteresis） ----

    horizontal_threshold_enter: int = 24
    # 水平方向"进入对准状态"的阈值（像素）。
    # 当目标偏移 < 此值时，系统判定为"已对准"。
    # 推荐: 约为基础 threshold 的 80%。

    horizontal_threshold_exit: int = 36
    # 水平方向"退出对准状态"的阈值（像素）。
    # 当目标偏移 > 此值时，系统判定为"已偏离"。
    # 推荐: 约为基础 threshold 的 120%。

    vertical_threshold_enter: int = 24
    # 垂直方向"进入对准状态"的阈值（像素）。

    vertical_threshold_exit: int = 36
    # 垂直方向"退出对准状态"的阈值（像素）。

    depth_threshold_enter: float = 0.2
    # 深度方向"进入对准状态"的阈值（归一化值）。

    depth_threshold_exit: float = 0.3
    # 深度方向"退出对准状态"的阈值（归一化值）。

    # ---- 抓取状态判定 ----

    grasp_stable_frames: int = 5
    # 判定"抓取稳定"所需的连续帧数。
    # 含义: 目标必须连续 N 帧保持在"对准"区域内，才判定为抓取稳定。
    # 效果: 值越大判定越严格（减少误触发），但响应延迟增加。
    # 推荐: 8 帧（约 0.3 秒 @ 30fps）在响应速度和稳定性之间取得平衡。

    grasp_release_frames: int = 3
    # 判定"释放"所需的连续帧数。
    # 含义: 目标连续 N 帧不在"对准"区域内，判定为已释放。
    # 效果: 值越小释放响应越快，但可能因短暂遮挡误判为释放。
    # 推荐: 3 帧（约 0.1 秒 @ 30fps），快速响应释放动作。

    hand_stable_frames: int = 5
    # 手部稳定出现所需的连续帧数，达到后才开始输出移动引导指令。
    # 含义: 手部进入画面后必须连续出现 N 帧，才视为"手部稳定"，引导播报才开始触发。
    # 效果: 避免手部短暂入画时立即播报，减少误触发。
    # 推荐: 5 帧（约 0.17 秒 @ 30fps）。


@dataclass
class AudioConfig:
    """音频子系统配置。

    包含语音识别（ASR）、语音合成（TTS）、录音参数、
    静音检测以及各类语音反馈的开关和阈值。

    主要功能模块:
    1. ASR (语音识别): 将用户语音转为文本，使用 Whisper 模型
    2. TTS (语音合成): 将系统文本转为语音播报
    3. 录音控制: 采样率、时长、静音检测
    4. 语音反馈: 目标确认、目标丢失、录制完成等场景的语音提示
    """

    # ---- ASR 语音识别设置 ----

    enable_asr: bool = True
    # 是否启用自动语音识别（ASR）功能。
    # 可选值: True (启用), False (禁用)
    # 效果: 启用后系统可通过麦克风接收用户语音指令。
    # 注意: 启用后需安装 whisper 相关依赖。

    whisper_model: str = "medium"
    # Whisper 语音识别模型大小。
    # 可选值: "tiny" (最快), "base" (推荐), "small", "medium", "large" (最准)
    # 效果: 模型越大识别准确率越高，但内存占用和推理时间也越大。
    #       tiny: ~1GB RAM, base: ~1.5GB, medium: ~5GB, large: ~10GB
    # 推荐: "base" 在中文场景下表现良好且资源占用适中。
    #       中文识别质量要求高时使用 "medium"。

    asr_language: str = "zh,en"
    # Whisper 识别的目标语言。
    # 可选值: "zh,en" (中英混合), "zh" (纯中文), "en" (纯英文), "auto" (自动检测)
    # 效果: 指定语言可提高识别准确率。"auto" 模式会增加约 100ms 的语言检测延迟。

    # ---- TTS 语音合成设置 ----

    enable_tts: bool = True
    # 是否启用语音合成（TTS）播报功能。
    # 可选值: True (启用), False (禁用)
    # 效果: 启用后系统会通过 TTS 引擎播报检测结果和引导信息。

    tts_provider: str = "mimo"
    # TTS 服务提供商。
    # 可选值: "pyttsx3" (离线，使用系统语音引擎), "mimo" (小米 MiMo 云端 TTS)
    # 效果: pyttsx3 无需联网但音质一般；mimo 音质更好但需要 API Key 和网络。
    # 注意: 使用 "mimo" 时需在 .env 中配置 MIMO_API_KEY。

    tts_rate: int = 200
    # TTS 语速（每分钟字数）。
    # 范围: 100 ~ 300
    # 效果: 值越大语速越快。150 为正常偏快语速，180 为较快语速。
    # 推荐: 150 适合大多数场景；信息密集时可提高到 180。

    tts_volume: float = 1.0
    # TTS 音量（0.0 ~ 1.0）。
    # 效果: 1.0 为系统最大音量，0.5 为一半音量。

    tts_async: bool = True
    # 是否启用异步 TTS 播报。
    # 可选值: True (异步，不阻塞主线程), False (同步，阻塞等待播报完成)
    # 效果: 异步模式下 TTS 在后台线程播放，不影响视觉检测主循环。
    # 推荐: 保持 True，除非需要确保播报完成后才执行下一步操作。

    tts_instruction_interval_sec: float = 3    # 两次 TTS 引导播报之间的最小间隔（秒）。
    # 效果: 防止系统过于频繁地播报引导信息，造成语音重叠或干扰。
    # 推荐: 3.0 秒适合实时引导场景。

    tts_grab_repeat_sec: float = 1.5
    # 抓取引导信息的重复播报间隔（秒）。
    # 效果: 当用户需要抓取引导时，每隔 N 秒重复一次提示。

    tts_max_queue_size: int = 1
    # TTS 播报队列最大长度。
    # 效果: 设为 1 表示只保留最新一条待播报消息，旧消息被丢弃。
    #       增大此值可缓存更多消息，但可能导致播报延迟累积。

    tts_drop_stale: bool = True
    # 是否丢弃过时的 TTS 消息。
    # 效果: 当队列满时，如果新消息与旧消息描述的是同一状态，
    #       则丢弃旧消息。防止重复播报相同内容。

    tts_state_change_bypass: bool = True
    # 状态变化时是否跳过队列限制直接播报。
    # 效果: 当检测到目标状态发生变化（如从"未找到"变为"已找到"），
    #       即使队列已满也立即播报，确保关键信息不丢失。

    # ---- MiMo 云端 TTS 设置 ----

    mimo_api_key: str = ""
    # 小米 MiMo TTS 服务的 API 密钥。
    # 注意: 此字段不应在代码中硬编码，请通过 .env 文件的 MIMO_API_KEY 环境变量加载。
    # 获取方式: 访问小米开放平台申请 MiMo TTS 服务。

    mimo_voice: str = "mimo_default"
    # MiMo TTS 使用的音色名称。
    # 可选值: 参考 MiMo API 文档中提供的音色列表。
    # 效果: 不同音色有不同的语音风格（男声/女声/童声等）。

    # ---- 录音参数设置 ----

    record_sample_rate: int = 16000
    # 录音采样率（Hz）。
    # 可选值: 8000 (电话质量), 16000 (推荐), 44100 (CD 质量)
    # 效果: 16000 Hz 已足够满足 Whisper 语音识别需求，更高采样率会增加数据量。

    record_duration: float = 5.0
    # 单次录音最大时长（秒）。
    # 效果: 超过此时长后自动停止录音并开始识别。
    # 推荐: 5.0 秒适合短指令场景；长句交互可增加到 10.0 秒。

    # ---- 静音检测设置 ----

    auto_detect_silence: bool = True
    # 是否启用自动静音检测。
    # 效果: 启用后，当检测到用户停止说话（静音）时自动结束录音。
    #       可避免用户等待固定录音时长。

    silence_threshold: float = 0.01
    # 静音判定的音量阈值（归一化值）。
    # 范围: 0.0 ~ 1.0
    # 效果: 音量低于此值时判定为静音。值越大越容易判定为静音。
    # 推荐: 0.01 适合安静环境；嘈杂环境可适当提高到 0.02~0.03。

    silence_duration: float = 1.5
    # 持续静音多长时间后结束录音（秒）。
    # 效果: 用户说完话后，等待 N 秒确认无后续语音，然后自动停止录音。
    # 推荐: 1.5 秒适合正常语速；语速较慢时可增加到 2.0 秒。

    # ---- 语音反馈设置 ----

    voice_feedback_after_recording: bool = True
    # 录音结束后是否播放语音反馈（如"正在识别..."）。
    # 效果: 提供操作确认反馈，让用户知道系统正在处理。

    voice_feedback_on_target_confirm: bool = True
    # 目标确认时是否播放语音反馈（如"已找到目标"）。
    # 效果: 当系统确认检测到目标时，通过语音告知用户。

    target_found_feedback_enabled: bool = True
    # 是否启用"目标已找到"的语音反馈。
    # 效果: 当目标连续 target_found_frame_threshold 帧被检测到时触发。

    target_found_frame_threshold: int = 8
    # 触发"目标已找到"反馈所需的连续检测帧数。
    # 效果: 值越大越不容易误触发，但反馈延迟增加。
    # 推荐: 8 帧（约 0.3 秒 @ 30fps）。

    target_missing_feedback_enabled: bool = True
    # 是否启用"目标丢失"的语音反馈。
    # 效果: 当目标连续 target_missing_frame_threshold 帧未被检测到时触发。

    target_missing_frame_threshold: int = 45
    # 触发"目标丢失"反馈所需的连续未检测帧数。
    # 效果: 值越大容忍短暂丢失的时间越长。
    # 推荐: 45 帧（约 1.5 秒 @ 30fps），避免因短暂遮挡误报"丢失"。

    target_missing_repeat_interval_sec: float = 30.0
    # 未进入任务时"暂未找到目标"提示的重复播报间隔（秒）。
    # 效果: 在非任务状态下，每隔此秒数重复提醒用户调整位置。任务进行中不重复。
    # 推荐: 30.0 秒。

    guidance_suppress_after_voice_sec: float = 1.5
    # 语音交互后抑制引导播报的时间（秒）。
    # 效果: 用户进行语音操作后，在此时间内系统暂停空间引导播报，
    #       避免语音重叠。


@dataclass
class LLMVisionConfig:
    """LLM 视觉增强配置。

    控制是否使用大语言模型（通过 Poe API）来增强视觉理解能力。
    系统会将摄像头画面发送给 LLM，获取更丰富的场景描述。

    工作流程:
    1. 截取最近 max_frames_for_vision 帧画面
    2. 编码为 Base64 发送给 Poe API
    3. LLM 返回结构化的场景描述
    4. 系统根据描述调整引导策略

    注意: 使用此功能需要 Poe API Key，请在 .env 中配置 POE_API_KEY。
    """

    enable_llm_parsing: bool = True
    # 是否启用 LLM 视觉解析功能。
    # 可选值: True (启用), False (禁用)
    # 效果: 启用后系统会将画面发送给 LLM 获取更丰富的场景理解。
    # 注意: 需要有效的 POE_API_KEY 才能使用此功能。

    poe_api_key: str = ""
    # Poe API 密钥。
    # 注意: 此字段不应在代码中硬编码，请通过 .env 文件的 POE_API_KEY 环境变量加载。
    # 获取方式: 访问 https://poe.com/api 申请 API Key。

    poe_model: str = "deepseek-v3.2"
    # 使用的 Poe LLM 模型名称。
    # 可选值: "deepseek-v3.2", "claude-3.5-sonnet", "gpt-4o" 等（参考 Poe API 文档）
    # 效果: 不同模型有不同的视觉理解能力和响应速度。
    # 推荐: "deepseek-v3.2" 性价比高；需要最强视觉理解时使用 "claude-3.5-sonnet"。

    poe_timeout_sec: float = 5.0
    # Poe API 请求超时时间（秒）。
    # 效果: 超过此时长未收到响应则判定为超时。
    # 推荐: 5.0 秒适合大多数模型；网络较差时可增加到 10.0 秒。

    max_frames_for_vision: int = 4
    # 发送给 LLM 的最大画面帧数。
    # 效果: 帧数越多 LLM 获得的上下文信息越丰富，但请求体积和费用也越大。
    # 推荐: 4 帧已能提供足够的运动上下文信息。

    api_retry_count: int = 1
    # API 请求失败时的重试次数。
    # 效果: 设为 1 表示失败后重试 1 次（总共尝试 2 次）。
    # 推荐: 1 次重试可应对偶发的网络抖动。


@dataclass
class LoggingConfig:
    """日志与性能监控配置。

    控制系统日志的输出方式（文件/控制台）、日志级别
    以及 FPS（帧率）统计功能。

    日志级别说明:
    - DEBUG:   最详细，包含每帧检测数据、模型推理时间等
    - INFO:    推荐级别，记录系统启动、配置变更、关键事件
    - WARNING: 仅记录警告信息
    - ERROR:   仅记录错误信息
    """

    log_dir: str = "logs"
    # 日志文件存储目录。
    # 效果: 系统会在此目录下创建按日期命名的日志文件。
    # 注意: 目录不存在时会自动创建。

    log_level: str = "INFO"
    # 日志输出级别。
    # 可选值: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    # 效果: 级别越低输出的日志越详细。DEBUG 模式会产生大量日志，适合开发调试。
    # 推荐: 生产环境使用 "INFO"；开发调试时使用 "DEBUG"。

    log_to_file: bool = True
    # 是否将日志写入文件。
    # 效果: 启用后日志会保存到 log_dir 指定的目录中，便于事后分析。

    log_to_console: bool = True
    # 是否将日志输出到控制台（终端）。
    # 效果: 启用后可在运行终端中实时查看日志。
    # 推荐: 开发时启用；生产环境如不需要可设为 False 减少终端输出。

    enable_fps_stats: bool = True
    # 是否启用 FPS（每秒帧数）统计功能。
    # 效果: 启用后系统会定期计算并记录各模块的帧率数据，
    #       便于性能分析和瓶颈定位。

    fps_window_size: int = 30
    # FPS 统计的滑动窗口大小（帧数）。
    # 效果: 值越大 FPS 数据越平滑但响应越慢；值越小 FPS 波动越大但响应越快。
    # 推荐: 30 帧（约 1 秒 @ 30fps）在平滑度和实时性之间取得平衡。

    enable_task_metrics: bool = True
    # 是否启用任务指标采集与报告写入功能。
    # 效果: 启用后每次任务结束时会将详细指标异步写入 task_metrics_dir 指定目录。

    task_metrics_dir: str = "logs/task_metrics"
    # 任务指标报告输出目录。
    # 效果: 系统会在此目录下生成按任务命名的 JSON 报告文件。
    # 注意: 目录不存在时会自动创建。

    task_metrics_interval_sec: float = 1.0
    # 任务实时摘要输出周期（秒）。
    # 效果: 任务进行中每隔此秒数向日志输出一次实时指标摘要。
    # 推荐: 1.0 秒。

    task_ready_confirm_window_sec: float = 2.0
    # 任务就绪确认时间窗口（秒）。
    # 效果: 目标进入就绪状态后，需在此时间窗口内完成抓取确认。
    # 推荐: 2.0 秒。

    task_start_confirm_window_sec: float = 4.0
    # 任务开始确认时间窗口（秒）。
    # 效果: 语音触发后，目标需持续被稳定检测 task_start_confirm_window_sec 秒，
    #       才正式激活任务（开始计时与指标采集）。
    # 推荐: 4.0 秒，过滤误检和短暂扫过。

    task_lost_target_window_sec: float = 4.0
    # 目标丢失判定时间窗口（秒）。
    # 效果: 目标连续消失超过此秒数后，判定为目标丢失并终止任务。
    # 推荐: 4.0 秒，避免短暂遮挡导致误判终止。


@dataclass
class SystemConfig:
    """系统根配置。

    聚合所有子模块的配置对象，是 load_config() 函数的返回值类型。
    包含模型、优化、引导、日志、音频、LLM 视觉等全部配置子模块，
    以及摄像头分辨率和默认检测目标等系统级参数。

    配置加载优先级（从高到低）:
    1. 环境变量（.env 文件）
    2. config.yaml profile overrides
    3. dataclass 默认值（本文件中定义）
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    # 模型配置（OWL-ViT、MiDaS、手部追踪）

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    # 性能优化配置（设备、精度、帧跳过）

    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    # 空间引导阈值配置（方向判定、抓取状态）

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # 日志与性能监控配置

    audio: AudioConfig = field(default_factory=AudioConfig)
    # 音频子系统配置（ASR、TTS、录音、反馈）

    llm_vision: LLMVisionConfig = field(default_factory=LLMVisionConfig)
    # LLM 视觉增强配置（Poe API、模型选择）

    target_queries: List[str] = field(default_factory=list)
    # 默认检测目标列表（OWL-ViT 的文本查询条件）。
    # 效果: 系统会检测画面中符合这些文本描述的物体。启动时为空，等待语音指令指定目标。
    # 格式: 使用英文名词短语，如 "a cup", "a bottle", "a person"。
    # 注意: 此值可在运行时被 core/system.py 动态修改。

    camera_id: int = 1
    # 摄像头设备 ID。
    # 效果: 对应系统摄像头编号，0 为默认摄像头，多摄像头时可设为 1、2 等。
    # 推荐: 单摄像头环境保持默认 0。

    camera_width: int = 848
    # 摄像头采集画面宽度（像素）。
    # 可选值: 320 / 640 / 848 / 1280 / 1920
    # 效果: 分辨率越高画面细节越丰富，但处理速度越慢。
    # 推荐: 848x480 在画质与性能之间取得平衡。

    camera_height: int = 480
    # 摄像头采集画面高度（像素）。
    # 推荐: 与 camera_width 配合使用，848x480 (16:9) 或 640x480 (4:3)。


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并两个字典，override 中的值覆盖 base。

    用于 config.yaml 中 profile 预设覆盖基础配置的场景。
    对于嵌套字典会递归合并，对于其他类型直接覆盖。

    参数:
        base: 基础字典
        override: 覆盖字典

    返回:
        合并后的新字典（不修改原始 base）
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_config(data: Dict[str, Any], cls: Type) -> Any:
    """将字典递归转换为 dataclass 实例。

    处理:
    - 嵌套 dataclass（如 SystemConfig.model -> ModelConfig）
    - Tuple 字段（YAML list -> Python tuple）
    - 未知 key 直接跳过（不会报错）

    参数:
        data: 配置字典（通常来自 YAML 解析）
        cls:  目标 dataclass 类型

    返回:
        cls 类型的实例
    """
    if not is_dataclass(cls):
        return data

    field_map = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}

    for name, fld in field_map.items():
        if name not in data:
            continue

        value = data[name]
        field_type = hints.get(name, fld.type)

        # 嵌套 dataclass
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _dict_to_config(value, field_type)

        # Tuple 字段：YAML list -> Python tuple
        elif _is_tuple_type(field_type) and isinstance(value, list):
            kwargs[name] = tuple(value)

        else:
            kwargs[name] = value

    return cls(**kwargs)


def _is_tuple_type(tp) -> bool:
    """判断类型注解是否为 Tuple 类型。"""
    origin = getattr(tp, '__origin__', None)
    return origin is tuple


def _flatten_camera(data: Dict[str, Any]) -> Dict[str, Any]:
    """将 YAML 中嵌套的 camera.width / camera.height 映射到
    顶层的 camera_width / camera_height 字段。

    这是为了兼容 YAML 中的嵌套写法:
        camera:
          width: 640
          height: 480
    与 SystemConfig 中的扁平字段:
        camera_width: 640
        camera_height: 480
    """
    if 'camera' in data and isinstance(data['camera'], dict):
        cam = data.pop('camera')
        if 'id' in cam:
            data['camera_id'] = cam['id']
        if 'width' in cam:
            data['camera_width'] = cam['width']
        if 'height' in cam:
            data['camera_height'] = cam['height']
    return data


def _load_env(dotenv_path: str = ".env") -> None:
    """加载 .env 文件中的环境变量。

    使用 python-dotenv 库加载指定路径的 .env 文件。
    如果 dotenv 未安装或文件不存在，静默跳过。

    参数:
        dotenv_path: .env 文件路径（默认 ".env"）
    """
    if load_dotenv is None:
        return
    p = Path(dotenv_path)
    if p.exists():
        load_dotenv(p, override=False)


def _load_yaml(path: str) -> Dict[str, Any]:
    """加载并解析 YAML 配置文件。

    如果 yaml 库未安装或文件不存在，返回空字典。

    参数:
        path: YAML 文件路径

    返回:
        解析后的字典，失败时返回空字典
    """
    if yaml is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _apply_env(config: SystemConfig) -> None:
    """将环境变量应用到配置对象（最高优先级覆盖）。

    当前支持的环境变量:
    - MIMO_API_KEY / XIAOMI_MIMO_API_KEY -> config.audio.mimo_api_key
    - POE_API_KEY -> config.llm_vision.poe_api_key

    参数:
        config: 待覆盖的 SystemConfig 实例
    """
    mimo_key = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY")
    if mimo_key:
        config.audio.mimo_api_key = mimo_key

    poe_key = os.environ.get("POE_API_KEY")
    if poe_key:
        config.llm_vision.poe_api_key = poe_key


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_config(profile: str = "balanced",
                yaml_path: str = "config.yaml",
                dotenv_path: str = ".env") -> SystemConfig:
    """加载系统配置。

    按照以下优先级合并配置（从高到低）:
    1. 环境变量（.env 文件）
    2. config.yaml 中指定 profile 的覆盖值
    3. dataclass 默认值（本文件中定义）

    参数:
        profile:     预设名称，可选值: "balanced"（默认）/ "fast" / "voice" / "tts" / "mimo-tts"
        yaml_path:   YAML 配置文件路径（默认 "config.yaml"）
        dotenv_path: .env 文件路径（默认 ".env"）

    返回:
        SystemConfig 实例，包含所有合并后的配置值

    示例:
        >>> config = load_config(profile="tts")
        >>> print(config.audio.enable_tts)  # True
    """
    # 1. 加载环境变量（.env 文件）
    _load_env(dotenv_path)

    # 2. 读取 YAML 配置 + 应用 profile 覆盖
    yaml_data = _load_yaml(yaml_path)
    profiles = yaml_data.pop('profiles', {})

    if profile and profile != 'balanced' and profile in profiles:
        yaml_data = _deep_merge(yaml_data, profiles[profile])
        logger.debug("Applied profile overrides: %s", profile)

    # 3. 扁平化 camera 嵌套字段: {width, height} -> camera_width, camera_height
    _flatten_camera(yaml_data)

    # 4. 将字典转换为 SystemConfig dataclass 实例
    config = _dict_to_config(yaml_data, SystemConfig)

    # 5. 应用环境变量覆盖（最高优先级）
    _apply_env(config)

    return config


# ---------------------------------------------------------------------------
# CLI 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def _to_dict(obj):
        """将 dataclass 递归转换为可 JSON 序列化的字典。"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(i) for i in obj]
        return obj

    name = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    cfg = load_config(profile=name)
    print(json.dumps(_to_dict(cfg), indent=2, ensure_ascii=False))
