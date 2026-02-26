"""
配置模块
================
集中管理系统所有组件的配置参数。
包含模型选择、优化设置、引导参数等。
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    # OWL-ViT 设置
    owlvit_model: str = "google/owlvit-base-patch32"
    owlvit_input_size: Tuple[int, int] = (384, 384)
    owlvit_confidence_threshold: float = 0.1
    
    # MiDaS 设置
    midas_model: str = "MiDaS_small"
    midas_scale: float = 0.5
    
    # MediaPipe 设置
    hand_max_num: int = 1
    hand_min_confidence: float = 0.5


@dataclass
class OptimizationConfig:
    """优化配置"""
    use_fp16: bool = True
    # 跳帧设置，减少检测的频率以提升性能
    skip_frames_detection: int = 2
    # 跳帧设置，减少深度估计的频率以提升性能
    skip_frames_depth: int = 2
    device: str = "auto"
    
    # 自动选择设备，优先使用 CUDA，如果不可用则回退到 CPU
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.use_fp16 = False


@dataclass
class GuidanceConfig:
    """引导配置"""
    horizontal_threshold: int = 30
    vertical_threshold: int = 30
    depth_threshold: float = 0.15


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    # FPS 统计配置
    enable_fps_stats: bool = True
    fps_window_size: int = 30  # FPS 平滑窗口


@dataclass
class SystemConfig:
    """系统配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    target_queries: List[str] = field(default_factory=lambda: ["a cup", "a bottle"])
    camera_width: int = 640
    camera_height: int = 480


def get_fast_config() -> SystemConfig:
    """快速配置"""
    config = SystemConfig()
    config.model.owlvit_input_size = (320, 320)
    config.model.midas_scale = 0.5
    config.optimization.skip_frames_detection = 3
    config.optimization.skip_frames_depth = 3
    return config


def get_balanced_config() -> SystemConfig:
    """平衡配置"""
    return SystemConfig()
