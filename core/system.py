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

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
# 降低第三方库的日志级别，避免过多无关信息干扰输出
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
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
        
        print("\n" + "="*60)
        print(" CV 视觉辅助系统")
        print("="*60)
        print(f" OWL-ViT: {self.config.model.owlvit_model}")
        print(f" MiDaS: {self.config.model.midas_model}")
        print(f" 设备: {self.config.optimization.device}")
        print(f" FP16: {self.config.optimization.use_fp16}")
        print("="*60 + "\n")
        
        self._init_components()
        
        self.frame_count = 0
        self.cached_detections = []
        self.cached_depth = None
        
        logger.info("系统初始化完成")
    
    def _init_components(self):
        cfg = self.config
        opt = cfg.optimization
        
        self.detector = OWLViTDetector(
            model_name=cfg.model.owlvit_model,
            input_size=cfg.model.owlvit_input_size,
            confidence_threshold=cfg.model.owlvit_confidence_threshold,
            use_fp16=opt.use_fp16,
            device=opt.device
        )
        
        self.hand_tracker = HandTracker(
            max_num_hands=cfg.model.hand_max_num,
            min_confidence=cfg.model.hand_min_confidence
        )
        
        self.depth_estimator = DepthEstimator(
            model_name=cfg.model.midas_model,
            scale=cfg.model.midas_scale,
            use_fp16=opt.use_fp16,
            device=opt.device
        )
        
        self.guidance = GuidanceController(
            horizontal_threshold=cfg.guidance.horizontal_threshold,
            vertical_threshold=cfg.guidance.vertical_threshold,
            depth_threshold=cfg.guidance.depth_threshold
        )
    
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
        
        fps = 1000 / result.total_time_ms if result.total_time_ms > 0 else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def run(self, camera_id: int = 0):
        print("\n控制:")
        print("  q - 退出")
        print("  d - 切换深度显示")
        print(f"\n检测目标: {self.config.target_queries}\n")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        window_name = "CV Assist System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return
        
        show_depth = False
        fps_list = []
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    # 有时在窗口关闭时cv2会抛出错误，安全退出
                    logger.info("摄像头读取时发生错误，可能是窗口已关闭。退出循环。")
                    break
                
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                result = self.process_frame(frame)
                
                fps_list.append(1000 / max(result.total_time_ms, 1))
                if len(fps_list) > 30:
                    fps_list.pop(0)
                
                output = self.draw_results(frame, result)
                
                avg_fps = sum(fps_list) / len(fps_list)
                cv2.putText(output, f"Avg: {avg_fps:.1f}", 
                           (output.shape[1] - 100, output.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
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
        except Exception as e:
            # 捕获其它异常，避免程序崩溃
            logger.exception("运行过程中发生异常，程序即将退出。")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n演示结束")


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
