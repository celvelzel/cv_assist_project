"""
主系统集成
==========
集成所有模块，实现完整的视觉辅助系统。
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig, get_fast_config, get_balanced_config
from detectors.owl_vit_detector import OWLViTDetector
from detectors.hand_tracker import HandTracker
from detectors.depth_estimator import DepthEstimator
from core.guidance import GuidanceController, GuidanceResult
from utils.logger import setup_logging, FPSCounter

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """帧处理结果"""
    detections: List[Dict]
    hands: List[Dict]
    depth_map: Optional[np.ndarray]
    guidance: Optional[GuidanceResult]
    total_time_ms: float
    detection_time_ms: float
    hand_time_ms: float
    depth_time_ms: float


class CVAssistSystem:
    """视觉辅助系统"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # 配置日志系统
        setup_logging(
            log_dir=self.config.logging.log_dir,
            log_level=self.config.logging.log_level,
            log_to_file=self.config.logging.log_to_file,
            log_to_console=self.config.logging.log_to_console
        )
        
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
        
        self.frame_count = 0
        self.cached_detections = []
        self.cached_depth = None
        
        # 初始化 FPS 计数器
        if self.config.logging.enable_fps_stats:
            self.fps_counter = FPSCounter(window_size=self.config.logging.fps_window_size)
        else:
            self.fps_counter = None
        
        logger.info("系统初始化完成")
    
    def _init_components(self):
        cfg = self.config
        opt = cfg.optimization
        
        logger.info("初始化检测器组件...")
        
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
        
        try:
            self.hand_tracker = HandTracker(
                max_num_hands=cfg.model.hand_max_num,
                min_confidence=cfg.model.hand_min_confidence
            )
        except Exception as e:
            logger.error(f"手部追踪器加载失败: {e}", exc_info=True)
            raise
        
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
        
        self.guidance = GuidanceController(
            horizontal_threshold=cfg.guidance.horizontal_threshold,
            vertical_threshold=cfg.guidance.vertical_threshold,
            depth_threshold=cfg.guidance.depth_threshold
        )
        
        logger.info("所有组件初始化成功")
    
    def process_frame(self, frame: np.ndarray,
                      queries: Optional[List[str]] = None) -> FrameResult:
        start = time.time()
        
        if queries is None:
            queries = self.config.target_queries
        
        self.frame_count += 1
        
        det_time = 0
        hand_time = 0
        depth_time = 0
        
        skip_det = self.config.optimization.skip_frames_detection
        if skip_det == 0 or self.frame_count % (skip_det + 1) == 0:
            t0 = time.time()
            self.cached_detections = self.detector.detect(frame, queries)
            det_time = (time.time() - t0) * 1000
        detections = self.cached_detections
        
        t0 = time.time()
        hand_result = self.hand_tracker.detect(frame)
        hands = hand_result['hands']
        hand_time = (time.time() - t0) * 1000
        
        skip_depth = self.config.optimization.skip_frames_depth
        if skip_depth == 0 or self.frame_count % (skip_depth + 1) == 0:
            t0 = time.time()
            self.cached_depth = self.depth_estimator.estimate(frame)
            depth_time = (time.time() - t0) * 1000
        depth_map = self.cached_depth
        
        guidance_result = None
        if hands and detections:
            hand = hands[0]
            target = detections[0]
            
            hand_depth = 0.5
            target_depth = 0.5
            
            if depth_map is not None:
                hand_depth = self.depth_estimator.get_depth_at_point(depth_map, hand['center'])
                target_depth = self.depth_estimator.get_depth_at_point(depth_map, target['center'])
            
            guidance_result = self.guidance.calculate(
                hand['center'], target['center'],
                hand_depth, target_depth,
                hand.get('gesture', 'unknown')
            )
        
        total_time = (time.time() - start) * 1000
        
        return FrameResult(
            detections=detections,
            hands=hands,
            depth_map=depth_map,
            guidance=guidance_result,
            total_time_ms=total_time,
            detection_time_ms=det_time,
            hand_time_ms=hand_time,
            depth_time_ms=depth_time
        )
    
    def draw_results(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        output = frame.copy()
        
        output = self.detector.draw(output, result.detections)
        output = self.hand_tracker.draw(output, {'hands': result.hands})
        
        if result.guidance and result.hands and result.detections:
            output = self.guidance.draw(
                output,
                result.hands[0]['center'],
                result.detections[0]['center'],
                result.guidance
            )
        
        # 显示 FPS 信息
        fps = 1000 / result.total_time_ms if result.total_time_ms > 0 else 0
        if self.fps_counter:
            self.fps_counter.update(fps)
            stats = self.fps_counter.get_stats()
            cv2.putText(output, f"FPS: {stats['current']:.1f}", 
                       (output.shape[1] - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Avg: {stats['average']:.1f}", 
                       (output.shape[1] - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def run(self, camera_id: int = 0):
        logger.info("启动 CV 视觉辅助系统")
        logger.info(f"控制: q - 退出, d - 切换深度显示")
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
        
        show_depth = False
        frame_counter = 0
        
        try:
            logger.info("开始主循环")
            while True:
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    logger.warning(f"摄像头读取错误: {e}")
                    break
                
                if not ret:
                    logger.warning("无法读取摄像头帧，可能摄像头已断开")
                    break
                
                frame = cv2.flip(frame, 1)
                
                try:
                    result = self.process_frame(frame)
                    frame_counter += 1
                    
                    # 每 100 帧记录一次统计
                    if self.fps_counter and frame_counter % 100 == 0:
                        stats = self.fps_counter.get_stats()
                        logger.info(f"FPS 统计 [帧 {frame_counter}]: "
                                  f"当前={stats['current']:.1f}, "
                                  f"平均={stats['average']:.1f}, "
                                  f"最小={stats['min']:.1f}, "
                                  f"最大={stats['max']:.1f}")
                        if result.detections:
                            logger.debug(f"检测到 {len(result.detections)} 个目标")
                    
                except Exception as e:
                    logger.error(f"帧处理错误: {e}", exc_info=True)
                    continue
                
                output = self.draw_results(frame, result)
                
                if show_depth and result.depth_map is not None:
                    depth_vis = self.depth_estimator.visualize(result.depth_map)
                    depth_vis = cv2.resize(depth_vis, (160, 120))
                    output[0:120, -160:] = depth_vis
                
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
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        except Exception as e:
            logger.exception(f"运行过程中发生异常: {e}")
        finally:
            # 输出最终统计
            if self.fps_counter:
                stats = self.fps_counter.get_stats()
                logger.info("="*60)
                logger.info("最终统计")
                logger.info(f"总帧数: {stats['total_frames']}")
                logger.info(f"平均 FPS: {stats['average']:.2f}")
                logger.info(f"最小 FPS: {stats['min']:.2f}")
                logger.info(f"最大 FPS: {stats['max']:.2f}")
                logger.info("="*60)
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("系统已关闭")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['fast', 'balanced'], default='balanced')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    config = get_fast_config() if args.config == 'fast' else get_balanced_config()
    
    system = CVAssistSystem(config)
    system.run(args.camera)
    # 程序正常结束时确保返回码为0
    sys.exit(0)


if __name__ == "__main__":
    main()
