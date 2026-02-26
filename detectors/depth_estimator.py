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
    """MiDaS 深度估计器"""
    
    def __init__(self,
                 model_name: str = "MiDaS_small",
                 scale: float = 0.5,
                 use_fp16: bool = True,
                 device: str = "auto"):
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
            self.model = torch.hub.load("intel-isl/MiDaS", model_name)
            if self.use_fp16:
                self.model = self.model.half()
            self.model = self.model.to(self.device).eval()
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
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
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
        # 如果模型或 transform 加载失败，返回空白深度图
        if self.model is None or self.transform is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        try:
            orig_h, orig_w = image.shape[:2]
            
            if self.scale < 1.0:
                new_w = int(orig_w * self.scale)
                new_h = int(orig_h * self.scale)
                image_scaled = cv2.resize(image, (new_w, new_h))
            else:
                image_scaled = image
            
            image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
            
            input_batch = self.transform(image_rgb).to(self.device)
            if self.use_fp16:
                input_batch = input_batch.half()
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_scaled.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth = prediction.cpu().numpy()
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 0:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)
            
            if self.scale < 1.0:
                depth = cv2.resize(depth, (orig_w, orig_h))
            
            return depth
            
        except Exception as e:
            logger.error(f"深度估计失败: {e}", exc_info=True)
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def get_depth_at_point(self, depth_map: np.ndarray, 
                           point: Tuple[int, int]) -> float:
        x, y = point
        h, w = depth_map.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            return float(depth_map[y, x])
        return 0.5
    
    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        depth_8bit = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)
