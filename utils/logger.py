"""
日志和统计工具
==============
提供日志配置和 FPS 统计功能。
"""

import logging
import os
from datetime import datetime
from collections import deque
from typing import Optional


class FPSCounter:
    """FPS 计数器"""
    
    def __init__(self, window_size: int = 30):
        """
        初始化 FPS 计数器
        
        Args:
            window_size: 滑动窗口大小，用于计算平均 FPS
        """
        self.window_size = window_size
        self.fps_queue = deque(maxlen=window_size)
        self.min_fps = float('inf')
        self.max_fps = 0.0
        self.total_frames = 0
        
    def update(self, fps: float):
        """更新 FPS 数据"""
        self.fps_queue.append(fps)
        self.total_frames += 1
        
        if fps < self.min_fps:
            self.min_fps = fps
        if fps > self.max_fps:
            self.max_fps = fps
    
    def get_avg_fps(self) -> float:
        """获取平均 FPS"""
        if not self.fps_queue:
            return 0.0
        return sum(self.fps_queue) / len(self.fps_queue)
    
    def get_current_fps(self) -> float:
        """获取当前 FPS"""
        if not self.fps_queue:
            return 0.0
        return self.fps_queue[-1]
    
    def get_stats(self) -> dict:
        """获取 FPS 统计信息"""
        return {
            'current': self.get_current_fps(),
            'average': self.get_avg_fps(),
            'min': self.min_fps if self.min_fps != float('inf') else 0.0,
            'max': self.max_fps,
            'total_frames': self.total_frames
        }
    
    def reset(self):
        """重置统计"""
        self.fps_queue.clear()
        self.min_fps = float('inf')
        self.max_fps = 0.0
        self.total_frames = 0


def setup_logging(log_dir: str = "logs", 
                  log_level: str = "INFO",
                  log_to_file: bool = True,
                  log_to_console: bool = True) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        
    Returns:
        配置好的 logger
    """
    # 创建日志目录
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cv_assist_{timestamp}.log")
    
    # 配置日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的 handlers
    root_logger.handlers.clear()
    
    # 添加文件处理器
    if log_to_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 降低第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成")
    if log_to_file:
        logger.info(f"日志文件: {log_file}")
    
    return root_logger
