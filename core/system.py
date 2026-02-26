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
    """帧处理结果数据类
    
    封装一帧图像的处理结果，包括检测、追踪、深度估计和引导信息。
    """
    detections: List[Dict]           # OWL-ViT 检测结果列表
    hands: List[Dict]                # 手部检测结果列表
    depth_map: Optional[np.ndarray]  # 深度图
    guidance: Optional[GuidanceResult]  # 引导信息
    total_time_ms: float             # 总处理时间(毫秒)
    detection_time_ms: float         # 目标检测时间
    hand_time_ms: float              # 手部检测时间
    depth_time_ms: float             # 深度估计时间


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
        
        # 初始化计数器和缓存
        self.frame_count = 0        # 已处理的帧数
        self.cached_detections = []  # 缓存的检测结果（用于跳帧）
        self.cached_depth = None     # 缓存的深度图（用于跳帧）
        
        # 初始化 FPS 计数器
        if self.config.logging.enable_fps_stats:
            self.fps_counter = FPSCounter(window_size=self.config.logging.fps_window_size)
        else:
            self.fps_counter = None
        
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
            depth_threshold=cfg.guidance.depth_threshold
        )
        
        logger.info("所有组件初始化成功")
    
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
        start = time.time()
        
        if queries is None:
            queries = self.config.target_queries
        
        self.frame_count += 1
        
        det_time = 0
        hand_time = 0
        depth_time = 0
        
        # 执行目标检测（按配置的跳帧率）
        skip_det = self.config.optimization.skip_frames_detection
        if skip_det == 0 or self.frame_count % (skip_det + 1) == 0:
            t0 = time.time()
            self.cached_detections = self.detector.detect(frame, queries)
            det_time = (time.time() - t0) * 1000  # 转换为毫秒
        detections = self.cached_detections  # 使用缓存的检测结果
        
        # 执行手部检测（每帧都执行，因为很快）
        t0 = time.time()
        hand_result = self.hand_tracker.detect(frame)
        hands = hand_result['hands']
        hand_time = (time.time() - t0) * 1000
        
        # 执行深度估计（按配置的跳帧率）
        skip_depth = self.config.optimization.skip_frames_depth
        if skip_depth == 0 or self.frame_count % (skip_depth + 1) == 0:
            t0 = time.time()
            self.cached_depth = self.depth_estimator.estimate(frame)
            depth_time = (time.time() - t0) * 1000
        depth_map = self.cached_depth  # 使用缓存的深度图
        
        # 计算引导指令（只有当同时检测到手和目标时）
        guidance_result = None
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
            guidance_result = self.guidance.calculate(
                hand['center'], target['center'],
                hand_depth, target_depth,
                hand.get('gesture', 'unknown')
            )
        
        # 计算总耗时
        total_time = (time.time() - start) * 1000
        
        # 返回处理结果
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
        if self.fps_counter:
            # 使用 FPS 计数器，显示当前和平均 FPS
            self.fps_counter.update(fps)
            stats = self.fps_counter.get_stats()
            cv2.putText(output, f"FPS: {stats['current']:.1f}", 
                       (output.shape[1] - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Avg: {stats['average']:.1f}", 
                       (output.shape[1] - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            # 不使用计数器，只显示当前 FPS
            cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def run(self, camera_id: int = 0):
        """
        运行视觉辅助系统主循环
        
        打开摄像头，循环处理每一帧，显示结果，直到用户退出。
        
        控制键:
        - 'q': 退出程序
        - 'd': 切换深度图显示
        
        参数:
            camera_id: 摄像头 ID，默认 0
        """
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
        
        show_depth = False   # 是否显示深度图
        frame_counter = 0    # 帧计数器，用于定期输出统计
        
        try:
            logger.info("开始主循环")
            while True:
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
            
            # 释放资源
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
