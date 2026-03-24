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
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig, load_config
from detectors.owl_vit_detector import OWLViTDetector
from detectors.hand_tracker import HandTracker
from detectors.depth_estimator import DepthEstimator
from detectors.obstacle_detector import ObstacleDetector, ObstacleDetectionResult
from core.guidance import GuidanceController, GuidanceResult
from core.interfaces import TrackedDetection
from core.spatial_resolver import create_spatial_resolver, ISpatialResolver
from core.tracker import create_object_tracker
from utils.logger import setup_logging, FPSCounter

logger = logging.getLogger(__name__)

# 音频模块（可选）
try:
    from audio.asr import ASREngine
    from audio.tts import TTSEngine, create_tts
    from audio.audio_utils import AudioRecorder
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
    obstacle: Optional[ObstacleDetectionResult] = None  # 障碍物检测结果
    total_time_ms: float = 0.0       # 总处理时间(毫秒)
    detection_time_ms: float = 0.0   # 目标检测时间
    hand_time_ms: float = 0.0        # 手部检测时间
    depth_time_ms: float = 0.0       # 深度估计时间


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
        logger.info(f" OWL-ViT: {self.config.model.owlvit_model}")
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
        
        # 初始化 OWL-ViT 目标检测器
        try:
            self.detector = OWLViTDetector(
                model_name=cfg.model.owlvit_model,
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
        
        # 初始化障碍物检测器
        self.obstacle_detector = ObstacleDetector(
            trajectory_frames=cfg.obstacle.obstacle_trajectory_frames,
            prediction_distance=cfg.obstacle.obstacle_prediction_distance,
            depth_threshold=cfg.obstacle.obstacle_depth_threshold,
            warning_cooldown=cfg.obstacle.obstacle_warning_cooldown,
            sample_count=cfg.obstacle.obstacle_sample_count,
            enabled=cfg.obstacle.enable_obstacle_detection,
        )
        
        # 初始化目标追踪器（用于多目标选择）
        tracker_type = getattr(cfg, 'tracker_type', 'simple')
        self.object_tracker = create_object_tracker(tracker_type)
        logger.info(f"目标追踪器初始化成功: {tracker_type}")
        
        # 初始化空间解析器（处理视角反转）
        camera_facing = getattr(cfg, 'camera_facing', 'outward')
        self.spatial_resolver = create_spatial_resolver(camera_facing)
        logger.info(f"空间解析器初始化成功: camera_facing={camera_facing}")
        
        logger.info("所有组件初始化成功")
    
    def _init_audio_components(self):
        """初始化音频组件 (ASR 和 TTS)"""
        cfg = self.config.audio
        
        # 初始化意图解析器（用于解析语音指令）
        from audio.intent_parser import create_intent_parser
        intent_parser_type = getattr(cfg, 'intent_parser_type', 'regex')
        self.intent_parser = create_intent_parser(intent_parser_type)
        logger.info(f"意图解析器初始化成功: {intent_parser_type}")
        
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
        """初始化运行时状态（线程、播报节流等）。"""
        self._voice_thread = None
        self._voice_in_progress = False
        self._voice_result_queue = queue.Queue()

        self._last_instruction = None
        self._last_instruction_state = None
        self._last_instruction_ts = 0.0
        self._last_grab_ts = 0.0
        
        # 空间提示（来自语音指令，如 "left cup" -> "left"）
        self._current_spatial_hint = None

    def _reset_tts_context(self):
        """目标切换后清理播报上下文，避免旧指令残留。"""
        self._last_instruction = None
        self._last_instruction_state = None
        self._last_instruction_ts = 0.0
        self._last_grab_ts = 0.0
        self._current_spatial_hint = None  # 重置空间提示
        if self.tts_engine:
            self.tts_engine.clear_queue()

    def _guidance_state(self, guidance: GuidanceResult) -> str:
        """提取引导语义状态。"""
        return getattr(guidance, 'state', 'ready' if guidance.ready_to_grab else 'moving')

    def _should_speak_guidance(self, guidance: GuidanceResult) -> bool:
        """根据节流规则判断是否应播报当前引导。"""
        cfg = self.config.audio
        now = time.time()
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
            if status == 'ok' and result.get('target'):
                target = result['target']
                spatial_hint = result.get('spatial_hint')  # 新增：空间提示
                
                self.config.target_queries = [target]
                self.cached_detections = []
                self._current_spatial_hint = spatial_hint  # 存储空间提示
                self._reset_tts_context()
                
                if spatial_hint:
                    logger.info(f"检测目标已更新为: {target} (空间: {spatial_hint})")
                    if self.tts_engine:
                        self.tts_engine.speak(f"正在寻找{spatial_hint}边的{target}")
                else:
                    logger.info(f"检测目标已更新为: {target}")
                    if self.tts_engine:
                        self.tts_engine.speak(f"正在寻找{target}")
            else:
                message = result.get('message')
                if message:
                    logger.warning(message)
                    if self.tts_engine:
                        self.tts_engine.speak(message)
    
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
        start = time.perf_counter()
        
        if queries is None:
            queries = self.config.target_queries
        
        self.frame_count += 1
        
        det_time = 0
        hand_time = 0
        depth_time = 0
        
        # 执行目标检测（按配置的跳帧率）
        skip_det = self.config.optimization.skip_frames_detection
        # 只有当 skip_det=0 或当前帧数满足跳帧条件时才执行检测，否则使用缓存结果
        if skip_det == 0 or self.frame_count % (skip_det + 1) == 0:
            t0 = time.perf_counter()
            self.cached_detections = self.detector.detect(frame, queries)
            det_time = (time.perf_counter() - t0) * 1000  # 转换为毫秒
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
        depth_map = self.cached_depth  # 使用缓存的深度图
        
        # 计算引导指令（只有当同时检测到手和目标时）
        guidance_result = None
        if hands and detections:
            # 使用追踪器为检测结果分配稳定的 tracking_id
            tracked_detections = self.object_tracker.update(detections)
            
            # 如果有空间提示（来自语音指令），使用空间解析器选择目标
            if hasattr(self, '_current_spatial_hint') and self._current_spatial_hint:
                # 过滤出匹配目标类别的追踪结果
                target_label = self.config.target_queries[0] if self.config.target_queries else None
                if target_label:
                    matching_tracks = [t for t in tracked_detections if t.label == target_label]
                    # 使用空间解析器选择目标
                    selected_target = self.spatial_resolver.resolve(self._current_spatial_hint, matching_tracks)
                else:
                    selected_target = tracked_detections[0] if tracked_detections else None
            else:
                # 无空间提示，选择第一个追踪结果
                selected_target = tracked_detections[0] if tracked_detections else None
            
            if selected_target:
                hand = hands[0]      # 取第一只手
                
                # 获取手部和目标的深度值
                hand_depth = 0.5    # 默认中等深度
                target_depth = 0.5
                
                if depth_map is not None:
                    hand_depth = self.depth_estimator.get_depth_at_point(depth_map, hand['center'])
                    target_depth = self.depth_estimator.get_depth_at_point(depth_map, selected_target.center)
                
                # 计算引导指令（传入 tracking_id）
                guidance_result = self.guidance.calculate(
                    hand['center'], selected_target.center,
                    hand_depth, target_depth,
                    hand.get('gesture', 'unknown'),
                    tracking_id=selected_target.tracking_id
                )
                
                # 更新 FrameResult 中的 detections 为选中的目标（保持向后兼容）
                detections = [vars(selected_target)]
            else:
                guidance_result = None
        
        # 障碍物检测（只要检测到手部且深度图可用）
        obstacle_result = None
        if hands and depth_map is not None:
            hand_center = hands[0]['center']
            obstacle_result = self.obstacle_detector.detect(
                depth_map, hand_center, current_time=time.time()
            )
        
        # 计算总耗时
        total_time = (time.perf_counter() - start) * 1000
        
        # 返回处理结果
        return FrameResult(
            detections=detections,
            hands=hands,
            depth_map=depth_map,
            guidance=guidance_result,
            obstacle=obstacle_result,
            total_time_ms=total_time,
            detection_time_ms=det_time,
            hand_time_ms=hand_time,
            depth_time_ms=depth_time
        )
    
    def draw_results(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
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

        # 绘制障碍物检测结果
        if result.obstacle and result.hands:
            output = self.obstacle_detector.draw(output, result.obstacle)
        
        # 显示 FPS 信息
        fps = 1000 / result.total_time_ms if result.total_time_ms > 0 else 0
        if self.fps_counter:
            # 处理 FPS（仅模型与算法处理时间）
            self.fps_counter.update(frame_time_ms=result.total_time_ms)
            proc_stats = self.fps_counter.get_stats()
            cv2.putText(output, f"Proc FPS: {proc_stats['current']:.1f}",
                       (output.shape[1] - 220, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Proc Avg: {proc_stats['average']:.1f}",
                       (output.shape[1] - 220, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 端到端 FPS（采集 + 处理 + 绘制 + 显示等待）
            if self.e2e_fps_counter:
                e2e_stats = self.e2e_fps_counter.get_stats()
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
        try:
            if self.tts_engine:
                self.tts_engine.speak("正在录音，请说出您要找的物品")
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
                return {'status': 'error', 'message': '未录制到音频，请重试'}
            
            logger.info(f"录音完成，开始识别...")
            if self.tts_engine:
                self.tts_engine.speak("正在识别")
            
            # 语音识别
            result = self.asr_engine.transcribe_audio(audio, cfg.record_sample_rate)
            text = result.get('text', '').strip()
            
            if not text:
                logger.warning("识别结果为空")
                return {'status': 'error', 'message': '没有识别到内容，请重试'}
            
            logger.info(f"识别结果: '{text}'")
            
            # 使用意图解析器解析指令（支持空间修饰符）
            if hasattr(self, 'intent_parser') and self.intent_parser:
                parsed = self.intent_parser.parse(text)
                if parsed:
                    logger.info(f"意图解析成功: 目标='{parsed.target_class}', 空间='{parsed.spatial_modifier}'")
                    return {
                        'status': 'ok',
                        'target': parsed.target_class,
                        'spatial_hint': parsed.spatial_modifier
                    }
                else:
                    logger.warning(f"意图解析失败: '{text}'")
                    return {'status': 'error', 'message': '抱歉，无法理解您的指令'}
            else:
                # 向后兼容：如果没有意图解析器，使用旧的解析方法
                target = self.asr_engine.parse_command(text)
                if target:
                    logger.info(f"语音解析目标成功: {target}")
                    return {'status': 'ok', 'target': target}
                else:
                    logger.warning(f"无法解析指令: '{text}'")
                    return {'status': 'error', 'message': '抱歉，无法理解您的指令'}
            
        except Exception as e:
            logger.error(f"语音输入处理失败: {e}", exc_info=True)
            return {'status': 'error', 'message': '语音识别出错'}
    
    def run(self, camera_id: int = 0):
        """
        运行视觉辅助系统主循环
        
        打开摄像头，循环处理每一帧，显示结果，直到用户退出。
        
        控制键:
        - 'q': 退出程序
        - 'd': 切换深度图显示
        - 'v': 开始语音输入 (如果启用了ASR)
        
        参数:
            camera_id: 摄像头 ID，默认 0
        """
        logger.info("启动 CV 视觉辅助系统")
        logger.info(f"控制: q - 退出, d - 切换深度显示")
        
        if self.asr_engine:
            logger.info(f"       v - 语音输入 (ASR 已启用)")
        if self.tts_engine:
            logger.info(f"TTS 已启用，将自动播放引导指令")
            self.tts_engine.speak("语音播报已开启")
        
        logger.info(f"检测目标: {self.config.target_queries}")
        
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
        
        try:
            logger.info("开始主循环")
            while True:
                loop_start = time.perf_counter()

                # 主线程提交后台语音识别结果
                self._drain_voice_results()

                # 读取摄像头帧
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    logger.warning(f"摄像头读取错误: {e}")
                    break
                
                if not ret:
                    logger.warning("无法读取摄像头帧，可能摄像头已断开")
                    break
                
                # 水平翻转图像（镜像效果）
                frame = cv2.flip(frame, 1)
                
                # 处理帧
                try:
                    result = self.process_frame(frame)
                    frame_counter += 1
                    
                    # 如果启用了TTS且有引导指令，播放语音
                    if self.tts_engine and result.guidance:
                        if self._should_speak_guidance(result.guidance):
                            self._speak_guidance(result.guidance)

                    # 障碍物检测：高优先级 TTS 警告（打断当前播报）
                    if result.obstacle and result.obstacle.detected:
                        if self.tts_engine and result.obstacle.warning_text:
                            self.tts_engine.clear_queue()  # 清空队列并停止当前播放
                            self.tts_engine.speak(result.obstacle.warning_text)
                            logger.warning(f"避障警告: {result.obstacle.warning_text}")
                    
                    # 每 100 帧记录一次统计
                    if self.fps_counter and frame_counter % 100 == 0:
                        proc_stats = self.fps_counter.get_stats()
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
                output = self.draw_results(frame, result)
                
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

                cv2.imshow(window_name, output)
                
                # 如果窗口被关闭，cv2.getWindowProperty返回值会小于1
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
                if self.e2e_fps_counter:
                    loop_time_ms = (time.perf_counter() - loop_start) * 1000
                    self.e2e_fps_counter.update(frame_time_ms=loop_time_ms)
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        except Exception as e:
            logger.exception(f"运行过程中发生异常: {e}")
        finally:
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
            cap.release()
            cv2.destroyAllWindows()
            logger.info("系统已关闭")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CV 视觉辅助系统')
    parser.add_argument('--config', choices=['fast', 'balanced', 'voice', 'tts', 'mimo-tts'], 
                       default='balanced',
                       help='配置模式: fast=快速, balanced=平衡, voice=启用ASR+TTS, tts=仅启用TTS, mimo-tts=MiMo云端TTS')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头 ID')
    args = parser.parse_args()

    config = load_config(profile=args.config)
    
    system = CVAssistSystem(config)
    system.run(args.camera)
    # 程序正常结束时确保返回码为0
    sys.exit(0)


if __name__ == "__main__":
    main()
