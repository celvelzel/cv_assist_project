"""
OWL-ViT 目标检测模块
====================
开放词汇目标检测，支持任意文本描述的物体检测。
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OWLViTDetector:
    """OWL-ViT 开放词汇目标检测器
    
    OWL-ViT (Open-World Localization Vision Transformer) 是一个零样本目标检测模型，
    可以通过自然语言描述来检测任意物体，无需事先训练特定类别。
    
    主要特点：
    - 支持任意文本查询（如"a cup", "a bottle"）
    - 无需预定义类别，灵活性强
    - 基于 Vision Transformer 架构
    """
    
    def __init__(self,
                 model_name: str = "google/owlvit-base-patch32",
                 input_size: Tuple[int, int] = (384, 384),
                 confidence_threshold: float = 0.1,
                 use_fp16: bool = True,
                 device: str = "auto"):
        """
        初始化 OWL-ViT 检测器
        
        参数:
            model_name: HuggingFace 模型名称，支持 base 和 large 版本
            input_size: 输入图像尺寸 (宽, 高)，影响检测精度和速度
            confidence_threshold: 置信度阈值，低于此值的检测结果将被过滤
            use_fp16: 是否使用半精度浮点数（仅GPU有效），可提升性能
            device: 运行设备，"auto" 自动选择，"cuda" GPU，"cpu" CPU
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"加载 OWL-ViT 模型: {model_name}")
        logger.info(f"设备: {self.device}, FP16: {self.use_fp16}")
        
        try:
            dtype = torch.float16 if self.use_fp16 else torch.float32
            logger.info("下载/加载 OWL-ViT 处理器...")
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            
            logger.info("下载/加载 OWL-ViT 模型...")
            self.model = OwlViTForObjectDetection.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(self.device)
            self.model.eval()
            
            logger.info("预热模型...")
            self._warmup()
            logger.info("OWL-ViT 初始化完成")
            
        except Exception as e:
            logger.error(f"OWL-ViT 模型加载失败: {e}", exc_info=True)
            logger.error("请检查:")
            logger.error("  1. 网络连接是否正常")
            logger.error("  2. HuggingFace 是否可访问")
            logger.error("  3. 磁盘空间是否充足")
            logger.error("  4. PyTorch 和 transformers 是否正确安装")
            raise RuntimeError(f"无法加载 OWL-ViT 模型: {e}")
    
    def _warmup(self):
        """预热模型以避免首次推理延迟
        
        深度学习模型在首次推理时通常会有额外的初始化开销（如CUDA核心编译）。
        通过预热运行一次虚拟推理，可以避免在实际使用时出现延迟尖峰。
        """
        try:
            # 创建一个黑色的虚拟图像
            dummy = Image.fromarray(np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8))
            # 处理输入并传递给模型
            inputs = self.processor(text=["test"], images=dummy, return_tensors="pt").to(self.device)
            with torch.no_grad():  # 禁用梯度计算以节省内存
                _ = self.model(**inputs)
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def detect(self, image: np.ndarray, queries: List[str], 
               threshold: Optional[float] = None) -> List[Dict]:
        """
        在图像中检测指定的目标物体
        
        参数:
            image: 输入图像 (BGR 格式的 numpy 数组)
            queries: 文本查询列表，如 ["a cup", "a bottle"]，描述要检测的物体
            threshold: 可选的置信度阈值，覆盖初始化时的默认值
            
        返回:
            检测结果列表，每个结果包含:
            - box: 边界框坐标 [x1, y1, x2, y2]
            - score: 置信度分数 (0-1)
            - label: 匹配的查询文本
            - center: 边界框中心点坐标 (x, y)
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        if not queries:
            return []
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            orig_h, orig_w = image.shape[:2]
            
            inputs = self.processor(text=queries, images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
            # 后处理检测结果
            # transformers 新版本使用 grounded_object_detection 方法
            # 该方法会将检测框映射到原始图像尺寸，并过滤低置信度结果
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,  # 置信度阈值
                target_sizes=target_sizes,  # 目标图像尺寸
                text_labels=[queries],  # 文本标签，用于返回匹配的查询
            )
            
            detections = []
            for result in results:
                # 从GPU转移到CPU并转换为numpy数组
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                # processor 返回每个检测的文本标签
                text_labels = result.get("text_labels", [])
                
                for box, score, text_label in zip(boxes, scores, text_labels):
                    # 提取边界框坐标
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # 计算缩放比例，将坐标映射回原始图像尺寸
                    scale_x = orig_w / pil_image.size[0]
                    scale_y = orig_h / pil_image.size[1]
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    # 计算边界框中心点，用于后续的引导计算
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'score': float(score),
                        'label': text_label,
                        'center': center
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []
    
    def draw(self, image: np.ndarray, detections: List[Dict],
             color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.circle(output, det['center'], 5, color, -1)
            label = f"{det['label']}: {det['score']:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output
