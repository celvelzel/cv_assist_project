"""
主系统集成
==========
集成所有模块，实现完整的视觉辅助系统。
"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig, load_config
from detectors.owl_vit_detector import OWLViTDetector
from detectors.hand_tracker import HandTracker
from detectors.depth_estimator import DepthEstimator
from core.guidance import GuidanceController, GuidanceResult
from utils.logger import setup_logging, FPSCounter
from utils.task_metrics import AsyncReportWriter, FrameMetrics, TaskMetricsCollector, TaskReportEnvelope

logger = logging.getLogger(__name__)

# 音频模块（可选）
try:
    from audio.asr import ASREngine
    from audio.tts import TTSEngine, create_tts
    from audio.audio_utils import AudioRecorder
    from audio.llm_vision import LLMVisionParser, create_llm_vision_parser
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("音频模块未安装，ASR/TTS功能将不可用")


@dataclass
class FrameResult:
    """帧处理结果数据类
    
    封装一帧图像的处理结果，包括检测、追踪、深度估计和引导信息。
    """
    detections: List[Dict]           # OWL-ViT 检测结果列表
    hands: List[Dict]                # 手部检测结果列表
    depth_map: Optional[np.ndarray]  # 深度图
    guidance: Optional[GuidanceResult]  # 引导信息
    frame_start_ts: float            # 帧处理开始时间戳
    frame_end_ts: float              # 帧处理结束时间戳
    total_time_ms: float             # 总处理时间(毫秒)
    detection_time_ms: float         # 目标检测时间
    hand_time_ms: float              # 手部检测时间
    depth_time_ms: float             # 深度估计时间
    guidance_time_ms: float          # 引导计算时间
    detection_executed: bool         # 当前帧是否执行了检测
    depth_executed: bool             # 当前帧是否执行了深度估计
    has_target: bool                 # 当前帧是否检测到目标
    has_hand: bool                   # 当前帧是否检测到手
    has_guidance: bool               # 当前帧是否生成引导
    guidance_state: str              # 引导状态
    ready_to_grab: bool              # 是否进入可抓取状态
    stable_ready_frames: int         # ready 持续帧数
    gesture: str                     # 当前手势
    target_visible: bool             # 当前任务目标是否可见
    detections_count: int            # 检测目标数量
    hands_count: int                 # 手部数量


class CVAssistSystem:
    """计算机视觉辅助系统
    
    集成多个深度学习模型，实现实时的视觉引导系统。
    
    主要组件:
    - OWL-ViT: 开放词汇目标检测器，检测指定物体
    - MediaPipe: 手部追踪器，检测手部位置和手势
    - MiDaS: 深度估计器，估计场景深度
    - GuidanceController: 引导控制器，生成导航指令
    
    工作流程:
    1. 读取摄像头图像
    2. 检测目标物体（可跳帧）
    3. 检测手部位置和手势
    4. 估计深度信息（可跳帧）
    5. 计算引导指令
    6. 绘制结果并显示
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化视觉辅助系统
        
        参数:
            config: 系统配置对象，如果为 None 则使用默认配置
        """
        self.config = config or SystemConfig()
        
        # 配置日志系统
        setup_logging(
            log_dir=self.config.logging.log_dir,
            log_level=self.config.logging.log_level,
            log_to_file=self.config.logging.log_to_file,
            log_to_console=self.config.logging.log_to_console
        )
        
        # 输出系统初始化信息
        logger.info("="*60)
        logger.info(" CV 视觉辅助系统")
        logger.info("="*60)
        # 根据版本选择合适的模型名称
        model_name = self.config.model.get_owlvit_model_name() if self.config.model.owlvit_model == self.config.model.get_owlvit_model_name() else self.config.model.owlvit_model
        logger.info(f" OWL-ViT: {self.config.model.owlvit_version} - {model_name}")
        logger.info(f" MiDaS: {self.config.model.midas_model}")
        logger.info(f" 设备: {self.config.optimization.device}")
        logger.info(f" FP16: {self.config.optimization.use_fp16}")
        logger.info("="*60)
        
        # 初始化组件（带异常处理）
        try:
            self._init_components()
        except Exception as e:
            logger.error(f"组件初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"系统初始化失败: {e}")
        
        # 初始化音频模块（如果启用）
        self.asr_engine = None
        self.tts_engine = None
        self.audio_recorder = None
        
        if AUDIO_AVAILABLE:
            self._init_audio_components()
        elif self.config.audio.enable_asr or self.config.audio.enable_tts:
            logger.warning("音频模块不可用，请安装依赖: pip install openai-whisper pyttsx3 sounddevice")
        
        # 初始化计数器和缓存
        self.frame_count = 0        # 已处理的帧数
        self.cached_detections = []  # 缓存的检测结果（用于跳帧）
        self.cached_depth = None     # 缓存的深度图（用于跳帧）

        # 运行时状态（线程与播报节流）
        self._init_runtime_states()
        self._init_task_metrics()
        
        # 初始化 FPS 计数器
        if self.config.logging.enable_fps_stats:
            self.fps_counter = FPSCounter(window_size=self.config.logging.fps_window_size)
            self.e2e_fps_counter = FPSCounter(window_size=self.config.logging.fps_window_size)
        else:
            self.fps_counter = None
            self.e2e_fps_counter = None
        
        logger.info("系统初始化完成")
    
    def _init_components(self):
        """初始化所有模型组件"""
        cfg = self.config
        opt = cfg.optimization
        
        logger.info("初始化检测器组件...")
        
        # 根据版本选择模型名称（如果用户没有自定义设置）
        if cfg.model.owlvit_model == "google/owlvit-base-patch32":
            # 使用默认模型，根据版本选择
            owlvit_model = cfg.model.get_owlvit_model_name()
        else:
            # 用户自定义模型名称
            owlvit_model = cfg.model.owlvit_model
        
        # 初始化 OWL-ViT 目标检测器
        try:
            self.detector = OWLViTDetector(
                model_name=owlvit_model,
                input_size=cfg.model.owlvit_input_size,
                confidence_threshold=cfg.model.owlvit_confidence_threshold,
                use_fp16=opt.use_fp16,
                device=opt.device
            )
        except Exception as e:
            logger.error(f"OWL-ViT 模型加载失败: {e}", exc_info=True)
            raise
        
        # 初始化 MediaPipe 手部追踪器
        try:
            self.hand_tracker = HandTracker(
                max_num_hands=cfg.model.hand_max_num,
                min_confidence=cfg.model.hand_min_confidence
            )
        except Exception as e:
            logger.error(f"手部追踪器加载失败: {e}", exc_info=True)
            raise
        
        # 初始化 MiDaS 深度估计器
        try:
            self.depth_estimator = DepthEstimator(
                model_name=cfg.model.midas_model,
                scale=cfg.model.midas_scale,
                use_fp16=opt.use_fp16,
                device=opt.device
            )
        except Exception as e:
            logger.error(f"深度估计器加载失败: {e}", exc_info=True)
            raise
        
        # 初始化引导控制器
        self.guidance = GuidanceController(
            horizontal_threshold=cfg.guidance.horizontal_threshold,
            vertical_threshold=cfg.guidance.vertical_threshold,
            depth_threshold=cfg.guidance.depth_threshold,
            horizontal_threshold_enter=cfg.guidance.horizontal_threshold_enter,
            horizontal_threshold_exit=cfg.guidance.horizontal_threshold_exit,
            vertical_threshold_enter=cfg.guidance.vertical_threshold_enter,
            vertical_threshold_exit=cfg.guidance.vertical_threshold_exit,
            depth_threshold_enter=cfg.guidance.depth_threshold_enter,
            depth_threshold_exit=cfg.guidance.depth_threshold_exit,
            grasp_stable_frames=cfg.guidance.grasp_stable_frames,
            grasp_release_frames=cfg.guidance.grasp_release_frames,
        )
        
        logger.info("所有组件初始化成功")
    
    def _init_audio_components(self):
        """初始化音频组件 (ASR, LLM Vision, 和 TTS)"""
        cfg = self.config.audio
        llm_cfg = self.config.llm_vision
        
        # 初始化 ASR
        if cfg.enable_asr:
            try:
                logger.info("正在初始化 ASR 引擎...")
                self.asr_engine = ASREngine(
                    model_name=cfg.whisper_model,
                    device=self.config.optimization.device,
                    language=cfg.asr_language
                )
                
                # 初始化录音器
                self.audio_recorder = AudioRecorder(
                    sample_rate=cfg.record_sample_rate,
                    channels=1,
                    dtype='float32'
                )
                
                logger.info("ASR 引擎初始化成功")
            except Exception as e:
                logger.error(f"ASR 引擎初始化失败: {e}")
                self.asr_engine = None
                self.audio_recorder = None
        
        # 初始化 LLM Vision Parser（用于增强语音识别）
        self.llm_vision_parser = None
        if AUDIO_AVAILABLE and cfg.enable_asr and llm_cfg.enable_llm_parsing:
            try:
                logger.info("正在初始化 LLM Vision 解析器...")
                # 通过 config 创建 LLM parser
                config_dict = {
                    "poe_api_key": llm_cfg.poe_api_key,
                    "poe_model": llm_cfg.poe_model,
                    "poe_timeout_sec": llm_cfg.poe_timeout_sec,
                    "max_frames_for_vision": llm_cfg.max_frames_for_vision,
                    "api_retry_count": llm_cfg.api_retry_count
                }
                self.llm_vision_parser = create_llm_vision_parser(config_dict)
                
                if self.llm_vision_parser:
                    logger.info("LLM Vision 解析器初始化成功")
                else:
                    logger.info("LLM Vision 解析器未启用（API key 未配置）")
            except Exception as e:
                logger.error(f"LLM Vision 初始化失败: {e}")
                self.llm_vision_parser = None
        
        # 初始化 TTS
        if cfg.enable_tts:
            try:
                logger.info("正在初始化 TTS 引擎...")
                self.tts_engine = create_tts(self.config)
                self.tts_engine.get_debug_info()
                logger.info("TTS 引擎初始化成功")
            except Exception as e:
                logger.error(f"TTS 引擎初始化失败: {e}")
                self.tts_engine = None

    def _init_runtime_states(self):
        """初始化运行时状态（线程、播报节流、帧缓冲等）。"""
        self._voice_thread = None
        self._voice_in_progress = False
        self._voice_prompt_playing = False
        self._voice_result_queue = queue.Queue()
        self._suppress_guidance_until_ts = 0.0
        # 生命周期 TTS 专用锁：保证同时只有一条生命周期播报在执行
        self._lifecycle_tts_lock = threading.Lock()
        # 生命周期播报进行中标志：为 True 时屏蔽所有普通指令播报
        self._lifecycle_speaking = False
        self._search_feedback_target = None
        self._search_feedback_state = "idle"
        self._target_found_streak = 0
        self._target_missing_streak = 0
        self._target_missing_last_spoken_ts = 0.0  # 上次播报"未找到"的时间戳
        self._hand_stable_streak = 0  # 手部连续出现帧数，达到阈值后才播报引导

        self._last_instruction = None
        self._last_instruction_state = None
        self._last_instruction_ts = 0.0
        self._last_grab_ts = 0.0
        
        # 帧缓冲，用于 LLM Vision 解析（存储最后 4 帧）
        # 在 _handle_voice_input 中使用
        self._frame_buffer = deque(maxlen=self.config.llm_vision.max_frames_for_vision)
        self._frame_buffer_lock = threading.Lock()  # 保护帧缓冲的线程安全

    def _init_task_metrics(self):
        """初始化任务监测与异步报告写入。"""
        self.session_id = uuid.uuid4().hex
        self.current_task = None
        self.task_state = "idle"
        self._task_index = 0
        self._last_task_summary_ts = 0.0
        self._requested_shutdown = False

        # 任务开始确认挂起态：语音触发后等待目标稳定检测 task_start_confirm_window_sec 秒再激活
        self._pending_task: Optional[Dict] = None  # 保存待激活的任务参数
        self._pending_task_target_since_ts: Optional[float] = None  # 目标首次稳定出现的时间戳

        self.task_metrics_collector = TaskMetricsCollector(
            grasp_stable_frames=self.config.guidance.grasp_stable_frames,
            ready_confirm_window_sec=self.config.logging.task_ready_confirm_window_sec,
            lost_target_window_sec=self.config.logging.task_lost_target_window_sec,
        )

        if self.config.logging.enable_task_metrics:
            self.report_writer = AsyncReportWriter(
                output_dir=self.config.logging.task_metrics_dir
            )
            self.report_writer.start()
        else:
            self.report_writer = None

    def _reset_tts_context(self):
        """目标切换后清理播报上下文，避免旧指令残留。"""
        self._last_instruction = None
        self._last_instruction_state = None
        self._last_instruction_ts = 0.0
        self._last_grab_ts = 0.0
        if self.tts_engine:
            self.tts_engine.clear_queue()

    def _suppress_guidance_temporarily(self):
        """在语音确认后短时抑制普通引导播报。"""
        duration = max(0.0, float(getattr(self.config.audio, 'guidance_suppress_after_voice_sec', 0.0)))
        if duration > 0:
            self._suppress_guidance_until_ts = max(self._suppress_guidance_until_ts, time.time() + duration)

    def _speak_priority_message(self, text: str, block: bool = False):
        """优先播报重要语音反馈，打断旧队列并短时抑制普通引导。"""
        if not self.tts_engine or not text or not text.strip():
            return

        self.tts_engine.stop()
        self.tts_engine.clear_queue()
        self._suppress_guidance_temporarily()
        self.tts_engine.speak(text.strip(), block=block)

    def _speak_lifecycle_message(self, text: str):
        """播报任务生命周期提示（开始/结束/完成），在独立线程中阻塞播完，不阻塞主循环。

        行为约束：
        - 不清空队列、不打断当前正在播放的内容
        - 播报期间延长引导抑制窗口，避免普通指令立即插入
        - 通过 _lifecycle_tts_lock 保证同时只有一条生命周期播报在执行
        """
        if not self.tts_engine or not text or not text.strip():
            return
        cfg = self.config.audio
        suppress_sec = max(
            1.5,
            float(getattr(cfg, 'guidance_suppress_after_voice_sec', 1.5))
        )
        # 主线程提前设置抑制窗口（预留播报时间），子线程播完后精确刷新
        self._suppress_guidance_until_ts = max(
            self._suppress_guidance_until_ts,
            time.time() + suppress_sec + 10.0
        )

        text_stripped = text.strip()
        # 主线程立即置位，让 _should_speak_guidance 在子线程合成期间就停止播报
        self._lifecycle_speaking = True

        def _worker():
            if not self._lifecycle_tts_lock.acquire(blocking=False):
                self._lifecycle_tts_lock.acquire()
            try:
                self._lifecycle_speaking = True
                self.tts_engine.speak_lifecycle(text_stripped)
            finally:
                self._lifecycle_speaking = False
                self._suppress_guidance_until_ts = 0.0
                self._lifecycle_tts_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _speak_serial_feedback(self, text: str, block: bool = False):
        """将搜索结果反馈串行加入 TTS 队列，不打断当前确认播报。"""
        if not self.tts_engine or not text or not text.strip():
            return
        if self._lifecycle_speaking:
            return

        self._suppress_guidance_temporarily()
        self.tts_engine.speak(text.strip(), block=block)

    def _play_voice_prompt_and_wait(self, text: str):
        """在开启录音前阻塞播报语音提示，并禁止背景 TTS 打断。"""
        if not self.tts_engine or not text or not text.strip():
            return

        self._voice_prompt_playing = True
        try:
            self.tts_engine.stop()
            self.tts_engine.clear_queue()
            self.tts_engine.speak(text.strip(), block=True)
        finally:
            self._voice_prompt_playing = False

    def _begin_target_search_feedback(self, target: str):
        """为新的语音目标初始化搜索反馈状态。"""
        target = (target or "").strip()
        self._search_feedback_target = target or None
        self._search_feedback_state = "searching" if target else "idle"
        self._target_found_streak = 0
        self._target_missing_streak = 0
        self._target_missing_last_spoken_ts = 0.0

    def _next_task_id(self) -> str:
        """生成会话内递增任务 ID。"""
        self._task_index += 1
        return f"task_{self._task_index:04d}"

    def _start_task(self, target: str, voice_event: Optional[Dict] = None):
        """将任务挂入等待确认态；目标稳定检测 task_start_confirm_window_sec 秒后由主循环激活。"""
        target = (target or "").strip()
        task_id = self._next_task_id()
        self._pending_task = {
            "task_id": task_id,
            "target_query": target,
            "voice_event": voice_event,
        }
        self._pending_task_target_since_ts = None
        self.task_state = "confirming"
        logger.info("任务进入确认等待: task_id=%s target=%s 需持续检测 %.1f 秒",
                    task_id, target, self.config.logging.task_start_confirm_window_sec)

    def _activate_pending_task(self):
        """确认时间窗口满足后真正激活任务。"""
        if not self._pending_task:
            return
        pending = self._pending_task
        self._pending_task = None
        self._pending_task_target_since_ts = None

        now = time.time()
        target = pending["target_query"]
        task_id = pending["task_id"]
        voice_event = pending.get("voice_event")

        self.current_task = {
            "task_id": task_id,
            "target_query": target,
            "start_time": now,
        }
        self.task_state = "running"
        self._last_task_summary_ts = 0.0
        self.task_metrics_collector.start_task(
            task_id=task_id,
            target_query=target,
            start_time=now,
            session_id=self.session_id,
        )
        if voice_event:
            self.task_metrics_collector.record_voice_metrics(
                voice_total_time_ms=voice_event.get("voice_total_time_ms", 0.0),
                voice_asr_time_ms=voice_event.get("voice_asr_time_ms", 0.0),
                raw_text=voice_event.get("raw_text", ""),
            )
        logger.info("任务正式激活: task_id=%s target=%s", task_id, target)
        self._speak_lifecycle_message(f"已找到{target}，开始执行任务，请伸出抓握手")

    def _update_pending_task_confirmation(self, detections: List[Dict], now: float):
        """主循环每帧调用：根据当前检测结果推进任务开始确认进度。"""
        if not self._pending_task:
            return
        confirm_window = self.config.logging.task_start_confirm_window_sec
        if detections:
            if self._pending_task_target_since_ts is None:
                self._pending_task_target_since_ts = now
            elif now - self._pending_task_target_since_ts >= confirm_window:
                self._activate_pending_task()
        else:
            # 目标消失，重置确认计时
            if self._pending_task_target_since_ts is not None:
                logger.debug("任务确认期间目标消失，重置计时: task_id=%s",
                             self._pending_task["task_id"])
            self._pending_task_target_since_ts = None

    def _enqueue_task_report(self, report_dict: Dict, task_id: str, created_at: float):
        """将冻结后的任务报告入队后台写盘。"""
        if not self.report_writer:
            return
        output_path = self.report_writer.build_output_path(task_id, created_at=created_at)
        envelope = TaskReportEnvelope(
            task_id=task_id,
            output_path=output_path,
            report_dict=report_dict,
            created_at=created_at,
        )
        self.report_writer.enqueue(envelope)

    def _finish_current_task(self, end_reason: str, error_message: str = "") -> Optional[Dict]:
        """结束当前任务并提交冻结快照。同时丢弃尚未激活的挂起任务。"""
        self._pending_task = None
        self._pending_task_target_since_ts = None
        if not self.current_task:
            self.task_state = "idle"
            return None

        now = time.time()
        task_id = self.current_task["task_id"]
        should_emit_report = self.task_metrics_collector.should_emit_report()
        report_dict = self.task_metrics_collector.finish_task(
            end_reason=end_reason,
            end_time=now,
            error_message=error_message,
        )
        if should_emit_report:
            self._enqueue_task_report(report_dict, task_id=task_id, created_at=now)
        else:
            logger.info("任务未检测到目标，跳过报告写入: task_id=%s", task_id)
        self.current_task = None
        self.task_state = "idle"
        return report_dict

    def _update_target_search_feedback(self, detections: List[Dict]):
        """基于当前目标的检测结果播报找到/未找到反馈。"""
        target = self._search_feedback_target
        if not target or self._voice_in_progress or self._voice_prompt_playing:
            return

        cfg = self.config.audio
        if detections:
            self._target_found_streak += 1
            self._target_missing_streak = 0
            found_threshold = max(1, int(getattr(cfg, 'target_found_frame_threshold', 8)))
            if self._target_found_streak < found_threshold:
                return
            if self._search_feedback_state != "found":
                self._search_feedback_state = "found"
                # 任务处于确认等待阶段时，"已找到"与"开始执行"合并
                # 由 _activate_pending_task 统一播报，此处不单独播
                in_confirming = self._pending_task is not None
                if not in_confirming and self.tts_engine and getattr(cfg, 'target_found_feedback_enabled', True):
                    self._speak_serial_feedback(f"已找到目标主体{target}")
            return

        self._target_found_streak = 0
        self._target_missing_streak += 1
        threshold = max(1, int(getattr(cfg, 'target_missing_frame_threshold', 45)))
        if self._target_missing_streak < threshold:
            return

        now = time.time()
        repeat_interval = max(
            5.0,
            float(getattr(cfg, 'target_missing_repeat_interval_sec', 30.0))
        )
        # 任务进行中只播报一次；未进入任务时按间隔重复播报
        in_task = self.current_task is not None
        first_time = self._search_feedback_state != "missing"
        due_for_repeat = (
            not in_task
            and now - self._target_missing_last_spoken_ts >= repeat_interval
        )

        if first_time or due_for_repeat:
            self._search_feedback_state = "missing"
            if self.tts_engine and getattr(cfg, 'target_missing_feedback_enabled', True):
                self._speak_serial_feedback(f"暂未找到目标主体{target}，请调整位置后重试")
                self._target_missing_last_spoken_ts = now

    def _guidance_state(self, guidance: GuidanceResult) -> str:
        """提取引导语义状态。"""
        return getattr(guidance, 'state', 'ready' if guidance.ready_to_grab else 'moving')

    def _should_speak_guidance(self, guidance: GuidanceResult) -> bool:
        """根据节流规则判断是否应播报当前引导。"""
        cfg = self.config.audio
        now = time.time()
        if self._voice_in_progress or self._voice_prompt_playing:
            return False
        if self._lifecycle_speaking:
            return False
        if now < self._suppress_guidance_until_ts:
            return False
        state = self._guidance_state(guidance)

        if state in ('ready', 'grabbed'):
            if self._last_instruction_state != state and cfg.tts_state_change_bypass:
                return True
            return (now - self._last_grab_ts) >= cfg.tts_grab_repeat_sec

        if self._last_instruction_state != state and cfg.tts_state_change_bypass:
            return True

        if self._last_instruction != guidance.instruction:
            return (now - self._last_instruction_ts) >= cfg.tts_instruction_interval_sec

        return (now - self._last_instruction_ts) >= cfg.tts_instruction_interval_sec

    def _speak_guidance(self, guidance: GuidanceResult):
        """播放当前引导并更新节流状态。"""
        if not self.tts_engine:
            return

        state = self._guidance_state(guidance)
        now = time.time()
        self.tts_engine.speak_instruction(guidance.instruction)
        self._last_instruction = guidance.instruction
        self._last_instruction_state = state
        self._last_instruction_ts = now
        if state in ('ready', 'grabbed'):
            self._last_grab_ts = now

    def _start_voice_input_async(self):
        """异步启动语音输入，避免主循环阻塞。"""
        if self._voice_in_progress:
            logger.info("语音输入进行中，请稍候...")
            return

        self._voice_in_progress = True
        self._voice_thread = threading.Thread(target=self._voice_input_worker, daemon=True)
        self._voice_thread.start()

    def _voice_input_worker(self):
        """后台执行录音和识别，结果回传给主循环。"""
        try:
            result = self._handle_voice_input()
            if result is not None:
                self._voice_result_queue.put(result)
        except Exception as e:
            logger.error(f"后台语音线程失败: {e}", exc_info=True)
            self._voice_result_queue.put({'status': 'error', 'message': '语音识别出错'})
        finally:
            self._voice_in_progress = False

    def _drain_voice_results(self):
        """消费后台语音结果并在主线程提交状态更新。"""
        while not self._voice_result_queue.empty():
            try:
                result = self._voice_result_queue.get_nowait()
            except queue.Empty:
                break

            status = result.get('status')
            action = result.get('action')
            target = (result.get('target') or '').strip()
            message = result.get('message')

            if status != 'ok':
                if message:
                    logger.warning(message)
                    if self.tts_engine:
                        self._speak_priority_message(message)
                continue

            if action == 'user_voice_exit':
                logger.info("收到语音退出指令")
                if self.current_task:
                    self._speak_lifecycle_message(
                        f"任务终止，语音退出{self.current_task['target_query']}"
                    )
                    self._finish_current_task('user_voice_exit')
                elif message and self.tts_engine:
                    self._speak_lifecycle_message(message)
                self._requested_shutdown = True
                self._begin_target_search_feedback("")
                continue

            if action not in ('set_target', 'switch_target') or not target:
                if message:
                    logger.warning(message)
                continue

            if action == 'switch_target' and self.current_task:
                old_target = self.current_task['target_query']
                self._speak_lifecycle_message(f"切换目标，停止{old_target}")
                self._finish_current_task('switch_target')

            self.config.target_queries = [target]
            self.cached_detections = []
            self._reset_tts_context()
            self._begin_target_search_feedback(target)
            self._start_task(target, voice_event=result)
            logger.info(f"检测目标已更新为: {target}")

            if self.tts_engine and getattr(self.config.audio, 'voice_feedback_on_target_confirm', True):
                confirm_message = message or f"开始寻找{target}"
                self._speak_lifecycle_message(confirm_message)
    
    def process_frame(self, frame: np.ndarray,
                      queries: Optional[List[str]] = None) -> FrameResult:
        """
        处理单帧图像
        
        执行完整的处理流程：目标检测 -> 手部检测 -> 深度估计 -> 引导计算
        使用跳帧机制优化性能：目标检测和深度估计可以在多帧之间共享结果。
        
        参数:
            frame: 输入图像
            queries: 目标查询列表，如果为 None 则使用配置中的默认值
            
        返回:
            FrameResult 对象，包含所有处理结果和耗时信息
        """
        perf_start = time.perf_counter()
        frame_start_ts = time.time()
        
        if queries is None:
            queries = self.config.target_queries
        
        self.frame_count += 1
        
        det_time = 0.0
        hand_time = 0.0
        depth_time = 0.0
        guidance_time = 0.0
        detection_executed = False
        depth_executed = False
        
        # 执行目标检测（按配置的跳帧率）
        skip_det = self.config.optimization.skip_frames_detection
        # 只有当 skip_det=0 或当前帧数满足跳帧条件时才执行检测，否则使用缓存结果
        if skip_det == 0 or self.frame_count % (skip_det + 1) == 0:
            t0 = time.perf_counter()
            self.cached_detections = self.detector.detect(frame, queries)
            det_time = (time.perf_counter() - t0) * 1000  # 转换为毫秒
            detection_executed = True
        detections = self.cached_detections  # 使用缓存的检测结果
        
        # 执行手部检测（每帧都执行，因为很快）
        t0 = time.perf_counter()
        hand_result = self.hand_tracker.detect(frame)
        hands = hand_result['hands']
        hand_time = (time.perf_counter() - t0) * 1000
        
        # 执行深度估计（按配置的跳帧率）
        skip_depth = self.config.optimization.skip_frames_depth
        if skip_depth == 0 or self.frame_count % (skip_depth + 1) == 0:
            t0 = time.perf_counter()
            self.cached_depth = self.depth_estimator.estimate(frame)
            depth_time = (time.perf_counter() - t0) * 1000
            depth_executed = True
        depth_map = self.cached_depth  # 使用缓存的深度图
        
        # 计算引导指令（只有当同时检测到手和目标时）
        guidance_result = None
        guidance_state = "idle"
        ready_to_grab = False
        stable_ready_frames = 0
        gesture = hands[0].get('gesture', 'unknown') if hands else 'unknown'
        if hands and detections:
            hand = hands[0]      # 取第一只手
            target = detections[0]  # 取第一个目标
            
            # 获取手部和目标的深度值
            hand_depth = 0.5    # 默认中等深度
            target_depth = 0.5
            
            if depth_map is not None:
                hand_depth = self.depth_estimator.get_depth_at_point(depth_map, hand['center'])
                target_depth = self.depth_estimator.get_depth_at_point(depth_map, target['center'])
            
            # 计算引导指令
            guidance_t0 = time.perf_counter()
            guidance_result = self.guidance.calculate(
                hand['center'], target['center'],
                hand_depth, target_depth,
                hand.get('gesture', 'unknown')
            )
            guidance_time = (time.perf_counter() - guidance_t0) * 1000
            guidance_state = self._guidance_state(guidance_result)
            ready_to_grab = bool(getattr(guidance_result, 'ready_to_grab', False))
            stable_ready_frames = int(getattr(guidance_result, 'stable_ready_frames', 0))
        
        # 计算总耗时
        total_time = (time.perf_counter() - perf_start) * 1000
        frame_end_ts = time.time()
        has_target = bool(detections)
        has_hand = bool(hands)
        has_guidance = guidance_result is not None
        
        # 返回处理结果
        return FrameResult(
            detections=detections,
            hands=hands,
            depth_map=depth_map,
            guidance=guidance_result,
            frame_start_ts=frame_start_ts,
            frame_end_ts=frame_end_ts,
            total_time_ms=total_time,
            detection_time_ms=det_time,
            hand_time_ms=hand_time,
            depth_time_ms=depth_time,
            guidance_time_ms=guidance_time,
            detection_executed=detection_executed,
            depth_executed=depth_executed,
            has_target=has_target,
            has_hand=has_hand,
            has_guidance=has_guidance,
            guidance_state=guidance_state,
            ready_to_grab=ready_to_grab,
            stable_ready_frames=stable_ready_frames,
            gesture=gesture,
            target_visible=has_target,
            detections_count=len(detections),
            hands_count=len(hands),
        )
    
    def draw_results(
        self,
        frame: np.ndarray,
        result: FrameResult,
        proc_stats: Optional[Dict] = None,
        e2e_stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        在图像上绘制所有检测结果
        
        绘制内容:
        - 目标检测框和标签
        - 手部关键点和手势
        - 引导指令和连接线
        - FPS 信息
        
        参数:
            frame: 输入图像
            result: 帧处理结果
            
        返回:
            绘制后的图像
        """
        output = frame.copy()
        
        # 绘制目标检测结果
        output = self.detector.draw(output, result.detections)
        # 绘制手部检测结果
        output = self.hand_tracker.draw(output, {'hands': result.hands})
        
        # 绘制引导信息（只有当同时检测到手和目标时）
        if result.guidance and result.hands and result.detections:
            output = self.guidance.draw(
                output,
                result.hands[0]['center'],
                result.detections[0]['center'],
                result.guidance
            )
        
        # 显示 FPS 信息
        fps = 1000 / result.total_time_ms if result.total_time_ms > 0 else 0
        if proc_stats:
            cv2.putText(output, f"Proc FPS: {proc_stats['current']:.1f}",
                       (output.shape[1] - 220, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Proc Avg: {proc_stats['average']:.1f}",
                       (output.shape[1] - 220, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if e2e_stats:
                cv2.putText(output, f"E2E FPS: {e2e_stats['current']:.1f}",
                           (output.shape[1] - 220, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
                cv2.putText(output, f"E2E Avg: {e2e_stats['average']:.1f}",
                           (output.shape[1] - 220, 98),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 1)
        else:
            # 不使用计数器，只显示当前 FPS
            cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def _handle_voice_input(self):
        """
        处理语音输入
        
        录制用户语音，使用 ASR 转录并解析目标。
        返回结构化结果，由主线程统一提交状态更新。
        """
        total_start = time.perf_counter()
        try:
            if self.tts_engine:
                self._play_voice_prompt_and_wait("正在录音，请给出描述")
            logger.info("=== 开始语音录制 ===")
            
            # 录制音频
            cfg = self.config.audio
            if cfg.auto_detect_silence:
                logger.info(f"录音中... (自动检测静音，最长 {cfg.record_duration}s)")
                audio = self.audio_recorder.record_until_silence(
                    max_duration=cfg.record_duration,
                    silence_threshold=cfg.silence_threshold,
                    silence_duration=cfg.silence_duration
                )
            else:
                logger.info(f"录音中... ({cfg.record_duration}s)")
                audio = self.audio_recorder.record(cfg.record_duration)
            
            if len(audio) == 0:
                logger.warning("未录制到音频")
                return {
                    'status': 'error',
                    'message': '未录制到音频，请重试',
                    'voice_total_time_ms': (time.perf_counter() - total_start) * 1000,
                    'voice_asr_time_ms': 0.0,
                    'raw_text': '',
                }
            
            logger.info(f"录音完成，开始识别...")
            if self.tts_engine:
                if getattr(cfg, 'voice_feedback_after_recording', True):
                    self._speak_priority_message("语音录入结束，正在识别")
                else:
                    self.tts_engine.speak("正在识别")
            
            # 语音识别
            asr_start = time.perf_counter()
            result = self.asr_engine.transcribe_audio(audio, cfg.record_sample_rate)
            voice_asr_time_ms = (time.perf_counter() - asr_start) * 1000
            text = result.get('text', '').strip()
            
            if not text:
                logger.warning("识别结果为空")
                return {
                    'status': 'error',
                    'message': '没有识别到内容，请重试',
                    'voice_total_time_ms': (time.perf_counter() - total_start) * 1000,
                    'voice_asr_time_ms': voice_asr_time_ms,
                    'raw_text': '',
                }
            
            logger.info(f"识别结果: '{text}'")
            
            # 准备用于 LLM Vision 的帧
            frames_snapshot = None
            if self.llm_vision_parser and self.llm_vision_parser.enabled:
                # 获取当前帧缓冲的快照
                with self._frame_buffer_lock:
                    if len(self._frame_buffer) > 0:
                        frames_snapshot = list(self._frame_buffer)
                        logger.debug(f"Captured {len(frames_snapshot)} frames for LLM vision")
                    else:
                        logger.debug("Frame buffer is empty, LLM vision will not be used")
            
            voice_event = self.asr_engine.parse_voice_event(
                text=text,
                has_active_task=self.current_task is not None,
                frames=frames_snapshot,
                llm_parser=self.llm_vision_parser,
            )
            voice_event['voice_total_time_ms'] = (time.perf_counter() - total_start) * 1000
            voice_event['voice_asr_time_ms'] = voice_asr_time_ms
            voice_event['raw_text'] = text

            if voice_event.get('status') == 'ok':
                logger.info(
                    "语音事件解析成功: action=%s target=%s",
                    voice_event.get('action'),
                    voice_event.get('target'),
                )
            else:
                logger.warning(f"无法解析指令: '{text}'")
            return voice_event
            
        except Exception as e:
            logger.error(f"语音输入处理失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': '语音识别出错',
                'voice_total_time_ms': (time.perf_counter() - total_start) * 1000,
                'voice_asr_time_ms': 0.0,
                'raw_text': '',
            }

    def _build_frame_metrics(
        self,
        result: FrameResult,
        capture_time_ms: float,
        draw_time_ms: float,
        display_time_ms: float,
        loop_time_ms: float,
        proc_stats: Dict,
        e2e_stats: Dict,
    ) -> FrameMetrics:
        """汇总主循环与算法阶段指标为统一单帧事实。"""
        return FrameMetrics(
            frame_index=self.frame_count,
            frame_start_ts=result.frame_start_ts,
            frame_end_ts=result.frame_end_ts,
            capture_time_ms=capture_time_ms,
            process_time_ms=result.total_time_ms,
            draw_time_ms=draw_time_ms,
            display_time_ms=display_time_ms,
            e2e_loop_time_ms=loop_time_ms,
            detection_time_ms=result.detection_time_ms,
            hand_time_ms=result.hand_time_ms,
            depth_time_ms=result.depth_time_ms,
            guidance_time_ms=result.guidance_time_ms,
            detection_executed=result.detection_executed,
            depth_executed=result.depth_executed,
            detections_count=result.detections_count,
            hands_count=result.hands_count,
            has_target=result.has_target,
            has_hand=result.has_hand,
            has_guidance=result.has_guidance,
            guidance_state=result.guidance_state,
            ready_to_grab=result.ready_to_grab,
            stable_ready_frames=result.stable_ready_frames,
            gesture=result.gesture,
            target_visible=result.target_visible,
            proc_fps_current=proc_stats.get('current', 0.0),
            proc_fps_avg=proc_stats.get('average', 0.0),
            e2e_fps_current=e2e_stats.get('current', 0.0),
            e2e_fps_avg=e2e_stats.get('average', 0.0),
        )

    def _maybe_log_task_summary(self):
        """按配置周期输出任务实时摘要。"""
        if not self.current_task or not self.config.logging.enable_task_metrics:
            return
        now = time.time()
        interval = max(0.2, float(self.config.logging.task_metrics_interval_sec))
        if now - self._last_task_summary_ts < interval:
            return
        summary = self.task_metrics_collector.build_terminal_summary(now)
        if summary:
            logger.info(summary)
            self._last_task_summary_ts = now
    
    def run(self, camera_id: Optional[int] = None):
        """
        运行视觉辅助系统主循环
        
        打开摄像头，循环处理每一帧，显示结果，直到用户退出。
        
        控制键:
        - 'q': 退出程序
        - 'd': 切换深度图显示
        - 'v': 开始语音输入 (如果启用了ASR)
        
        参数:
            camera_id: 摄像头 ID，未传入时使用配置文件中的 camera.id
        """
        if camera_id is None:
            camera_id = self.config.camera_id

        logger.info("启动 CV 视觉辅助系统")
        logger.info(f"控制: q - 退出, d - 切换深度显示")
        
        if self.asr_engine:
            logger.info(f"       v - 语音输入 (ASR 已启用)")
        if self.tts_engine:
            logger.info(f"TTS 已启用，将自动播放引导指令")
            self.tts_engine.speak("语音播报已开启")
        
        logger.info(f"检测目标: {self.config.target_queries}")
        logger.info(f"摄像头选择: {camera_id}")
        
        # 尝试打开摄像头，带异常处理
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")
            
            # 测试读取一帧
            ret, test_frame = cap.read()
            if not ret:
                raise RuntimeError(f"摄像头 {camera_id} 无法读取图像")
                
            logger.info(f"摄像头 {camera_id} 初始化成功，分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}", exc_info=True)
            logger.error("请检查：")
            logger.error("  1. 摄像头是否已连接")
            logger.error("  2. 摄像头是否被其他程序占用")
            logger.error("  3. 摄像头驱动是否正常")
            logger.error("  4. 尝试使用不同的 camera_id (0, 1, 2...)")
            return

        window_name = "CV Assist System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        show_depth = False   # 是否显示深度图
        frame_counter = 0    # 帧计数器，用于定期输出统计
        exit_reason = None
        
        try:
            logger.info("开始主循环")
            while True:
                loop_start = time.perf_counter()

                # 主线程提交后台语音识别结果
                self._drain_voice_results()

                # 读取摄像头帧
                capture_start = time.perf_counter()
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    logger.warning(f"摄像头读取错误: {e}")
                    exit_reason = 'error'
                    break
                capture_time_ms = (time.perf_counter() - capture_start) * 1000
                
                if not ret:
                    logger.warning("无法读取摄像头帧，可能摄像头已断开")
                    exit_reason = 'error'
                    break
                
                # 水平翻转图像（镜像效果）
                frame = cv2.flip(frame, 1)
                
                # 更新帧缓冲（用于 LLM Vision 解析）
                # 保存最后 N 帧（默认 4 帧），用于语音输入时的视觉上下文
                with self._frame_buffer_lock:
                    self._frame_buffer.append(frame.copy())
                
                # 处理帧
                try:
                    result = self.process_frame(frame)
                    frame_counter += 1
                    self._update_target_search_feedback(result.detections)
                    self._update_pending_task_confirmation(result.detections, time.time())

                    # 更新手部稳定帧计数
                    if result.has_hand:
                        self._hand_stable_streak += 1
                    else:
                        self._hand_stable_streak = 0

                    # 移动引导指令仅在任务正式激活且手部稳定出现后播报
                    hand_stable_threshold = max(1, int(getattr(
                        self.config.guidance, 'hand_stable_frames', 20
                    )))
                    hand_is_stable = self._hand_stable_streak >= hand_stable_threshold
                    if self.current_task and hand_is_stable and self.tts_engine and result.guidance:
                        if self._should_speak_guidance(result.guidance):
                            self._speak_guidance(result.guidance)
                    
                    proc_stats = {}
                    if self.fps_counter:
                        self.fps_counter.update(frame_time_ms=result.total_time_ms)
                        proc_stats = self.fps_counter.get_stats()

                    # 每 100 帧记录一次统计
                    if self.fps_counter and frame_counter % 100 == 0:
                        if self.e2e_fps_counter:
                            e2e_stats = self.e2e_fps_counter.get_stats()
                            logger.info(f"FPS 统计 [帧 {frame_counter}] | "
                                      f"处理: 当前={proc_stats['current']:.1f}, 平均={proc_stats['average']:.1f}, 最小={proc_stats['min']:.1f}, 最大={proc_stats['max']:.1f} | "
                                      f"端到端: 当前={e2e_stats['current']:.1f}, 平均={e2e_stats['average']:.1f}, 最小={e2e_stats['min']:.1f}, 最大={e2e_stats['max']:.1f}")
                        else:
                            logger.info(f"FPS 统计 [帧 {frame_counter}]: "
                                      f"当前={proc_stats['current']:.1f}, "
                                      f"平均={proc_stats['average']:.1f}, "
                                      f"最小={proc_stats['min']:.1f}, "
                                      f"最大={proc_stats['max']:.1f}")
                        if result.detections:
                            logger.debug(f"检测到 {len(result.detections)} 个目标")
                    
                except Exception as e:
                    logger.error(f"帧处理错误: {e}", exc_info=True)
                    continue
                
                # 绘制结果
                draw_start = time.perf_counter()
                prev_e2e_stats = self.e2e_fps_counter.get_stats() if self.e2e_fps_counter else {}
                output = self.draw_results(frame, result, proc_stats=proc_stats, e2e_stats=prev_e2e_stats)
                draw_time_ms = (time.perf_counter() - draw_start) * 1000
                
                # 如果启用了深度显示，在右上角显示深度图
                if show_depth and result.depth_map is not None:
                    depth_vis = self.depth_estimator.visualize(result.depth_map)
                    depth_vis = cv2.resize(depth_vis, (160, 120))  # 缩小尺寸
                    output[0:120, -160:] = depth_vis  # 放在右上角
                
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

                display_start = time.perf_counter()
                cv2.imshow(window_name, output)
                
                # 如果窗口被关闭，cv2.getWindowProperty返回值会小于1
                key = cv2.waitKey(1) & 0xFF
                display_time_ms = (time.perf_counter() - display_start) * 1000
                if key == ord('q'):
                    exit_reason = 'error'
                    break
                elif key == ord('d'):
                    show_depth = not show_depth
                    logger.info(f"深度显示: {'开启' if show_depth else '关闭'}")
                elif key == ord('v'):
                    # 语音输入
                    if self.asr_engine and self.audio_recorder:
                        self._start_voice_input_async()
                    else:
                        logger.warning("ASR 功能未启用")

                # 端到端 FPS：包含采集、处理、绘制和显示等待
                loop_time_ms = (time.perf_counter() - loop_start) * 1000
                if self.e2e_fps_counter:
                    self.e2e_fps_counter.update(frame_time_ms=loop_time_ms)
                    e2e_stats = self.e2e_fps_counter.get_stats()
                else:
                    e2e_stats = {}

                if self.current_task:
                    frame_metrics = self._build_frame_metrics(
                        result=result,
                        capture_time_ms=capture_time_ms,
                        draw_time_ms=draw_time_ms,
                        display_time_ms=display_time_ms,
                        loop_time_ms=loop_time_ms,
                        proc_stats=proc_stats,
                        e2e_stats=e2e_stats,
                    )
                    self.task_metrics_collector.record_frame(frame_metrics)
                    finish_reason = self.task_metrics_collector.should_finish_task()
                    if finish_reason:
                        target_query = self.current_task['target_query']
                        if finish_reason == 'success':
                            self._speak_lifecycle_message(f"任务完成，已抓取{target_query}")
                        elif finish_reason == 'lost_target':
                            self._speak_lifecycle_message(f"任务终止，无法定位{target_query}")
                        self._finish_current_task(finish_reason)
                        self._begin_target_search_feedback("")
                    else:
                        self._maybe_log_task_summary()

                if self._requested_shutdown:
                    break
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
            exit_reason = 'error'
        except Exception as e:
            logger.exception(f"运行过程中发生异常: {e}")
            exit_reason = 'error'
        finally:
            if self.current_task:
                final_reason = exit_reason or 'error'
                self._finish_current_task(final_reason)
            # 输出最终统计
            if self.fps_counter:
                stats = self.fps_counter.get_stats()
                e2e_stats = self.e2e_fps_counter.get_stats() if self.e2e_fps_counter else None
                logger.info("="*60)
                logger.info("最终统计")
                logger.info(f"总帧数: {stats['total_frames']}")
                logger.info(f"处理平均 FPS: {stats['average']:.2f}")
                logger.info(f"处理最小 FPS: {stats['min']:.2f}")
                logger.info(f"处理最大 FPS: {stats['max']:.2f}")
                if e2e_stats:
                    logger.info(f"端到端平均 FPS: {e2e_stats['average']:.2f}")
                    logger.info(f"端到端最小 FPS: {e2e_stats['min']:.2f}")
                    logger.info(f"端到端最大 FPS: {e2e_stats['max']:.2f}")
                logger.info("="*60)
            
            # 释放资源
            if self.report_writer:
                self.report_writer.stop(timeout_sec=2.0)
            cap.release()
            cv2.destroyAllWindows()
            logger.info("系统已关闭")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CV 视觉辅助系统')
    parser.add_argument('--config', choices=['fast', 'balanced', 'voice', 'tts', 'mimo-tts'], 
                       default='balanced',
                       help='配置模式: fast=快速, balanced=平衡, voice=启用ASR+TTS, tts=仅启用TTS, mimo-tts=MiMo云端TTS')
    parser.add_argument('--camera', type=int, default=None,
                       help='摄像头 ID，未传时使用 config.yaml 中的 camera.id')
    args = parser.parse_args()

    config = load_config(profile=args.config)
    
    system = CVAssistSystem(config)
    system.run(args.camera)
    # 程序正常结束时确保返回码为0
    sys.exit(0)


if __name__ == "__main__":
    main()
