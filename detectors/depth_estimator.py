"""
MiDaS 深度估计模块
==================
单目深度估计，提供相对深度信息。
"""

import cv2
import numpy as np
import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """MiDaS 深度估计器
    
    MiDaS (Mixed Data Sampling) 是 Intel 开发的单目深度估计模型。
    从单个 RGB 图像推断场景的相对深度信息（非绝对距离）。
    
    特点:
    - 不需要立体相机或深度传感器
    - 输出相对深度值 (0-1)，0 表示远，1 表示近
    - 支持多种模型规模 (small/large/hybrid)
    """
    
    def __init__(self,
                 model_name: str = "MiDaS_small",
                 scale: float = 0.5,
                 use_fp16: bool = True,
                 device: str = "auto"):
        """
        初始化深度估计器
        
        参数:
            model_name: 模型名称，支持 "MiDaS_small", "DPT_Large", "DPT_Hybrid" 等
            scale: 输入图像缩放比例 (0-1)，较小的值可提升速度但降低精度
            use_fp16: 是否使用半精度浮点数（仅GPU有效）
            device: 运行设备，"auto" 自动选择，"cuda" GPU，"cpu" CPU
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.scale = scale
        
        logger.info(f"加载 MiDaS 模型: {model_name}")
        logger.info(f"设备: {self.device}, FP16: {self.use_fp16}, 缩放: {scale}")
        
        # 加载模型，带网络错误处理
        try:
            logger.info("下载/加载 MiDaS 模型...")
            # 从 PyTorch Hub 加载预训练模型
            self.model = torch.hub.load("intel-isl/MiDaS", model_name)
            if self.use_fp16:
                self.model = self.model.half()  # 转换为 FP16 提升性能
            self.model = self.model.to(self.device).eval()  # 移动到设备并设置为评估模式
            logger.info("MiDaS 模型加载成功")
        except Exception as e:
            logger.error(f"无法加载 MiDaS 模型: {e}", exc_info=True)
            logger.error("请检查:")
            logger.error("  1. 网络连接是否正常")
            logger.error("  2. GitHub 是否可访问")
            logger.error("  3. PyTorch Hub 缓存是否有效")
            self.model = None
        
        try:
            logger.info("下载/加载 MiDaS transforms...")
            # 加载图像变换器，用于预处理输入图像
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            # 根据模型类型选择合适的变换
            if "small" in model_name.lower():
                self.transform = self.transforms.small_transform
            else:
                self.transform = self.transforms.dpt_transform
            logger.info("MiDaS transforms 加载成功")
        except Exception as e:
            logger.error(f"无法加载 MiDaS transforms: {e}", exc_info=True)
            self.transform = None
        
        if self.model is not None and self.transform is not None:
            logger.info("MiDaS 初始化完成")
        else:
            logger.warning("MiDaS 未能完全初始化，深度估计将被禁用")
            logger.warning("系统将继续运行，但不会进行深度估计")
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        估计图像的深度图
        
        参数:
            image: 输入图像 (BGR 格式的 numpy 数组)
            
        返回:
            深度图 (float32 数组)，值域 [0, 1]
            0 表示远处，1 表示近处
        """
        # 如果模型或 transform 加载失败，返回空白深度图
        if self.model is None or self.transform is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        try:
            orig_h, orig_w = image.shape[:2]
            
            # 按缩放比例调整图像大小，以提升处理速度
            if self.scale < 1.0:
                new_w = int(orig_w * self.scale)
                new_h = int(orig_h * self.scale)
                image_scaled = cv2.resize(image, (new_w, new_h))
            else:
                image_scaled = image
            
            # 将 BGR 转换为 RGB（MiDaS 需要 RGB 格式）
            image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
            
            # 使用 MiDaS 变换器预处理图像
            input_batch = self.transform(image_rgb).to(self.device)
            if self.use_fp16:
                input_batch = input_batch.half()
            
            # 进行深度推理
            with torch.no_grad():  # 禁用梯度计算
                prediction = self.model(input_batch)
                # 将输出调整为输入图像的大小
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_scaled.shape[:2],  # 目标大小
                    mode="bicubic",                # 双三次插值
                    align_corners=False,
                ).squeeze()
            
            # 将深度图从 GPU 移动到 CPU 并转换为 numpy 数组
            depth = prediction.cpu().numpy()
            # 标准化深度值到 [0, 1] 区间
            # MiDaS 输出的是相对深度，需要归一化
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 0:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)
            
            # 如果之前进行了缩放，将深度图放大回原始尺寸
            if self.scale < 1.0:
                depth = cv2.resize(depth, (orig_w, orig_h))
            
            return depth
            
        except Exception as e:
            logger.error(f"深度估计失败: {e}", exc_info=True)
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def get_depth_at_point(self, depth_map: np.ndarray, 
                           point: Tuple[int, int]) -> float:
        """
        获取深度图中指定点的深度值
        
        参数:
            depth_map: 深度图
            point: 像素坐标 (x, y)
            
        返回:
            深度值 (0-1)，如果点在图像外则返回 0.5
        """
        x, y = point
        h, w = depth_map.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            return float(depth_map[y, x])
        return 0.5  # 默认中等深度
    
    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        将深度图转换为伪彩色可视化图像
        
        参数:
            depth_map: 深度图 (0-1)
            
        返回:
            伪彩色图像（使用 MAGMA 颜色映射）
            蓝色-紫色表示远，黄色-白色表示近
        """
        # 将 [0, 1] 范围的深度值转换为 [0, 255] 的无符号8位整数
        depth_8bit = (depth_map * 255).astype(np.uint8)
        # 应用 MAGMA 颜色映射，生成伪彩色图像
        return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)
