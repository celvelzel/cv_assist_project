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
    """OWL-ViT 开放词汇目标检测器"""
    
    def __init__(self,
                 model_name: str = "google/owlvit-base-patch32",
                 input_size: Tuple[int, int] = (384, 384),
                 confidence_threshold: float = 0.1,
                 use_fp16: bool = True,
                 device: str = "auto"):
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
        """预热模型以避免首次推理延迟"""
        try:
            dummy = Image.fromarray(np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8))
            inputs = self.processor(text=["test"], images=dummy, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def detect(self, image: np.ndarray, queries: List[str], 
               threshold: Optional[float] = None) -> List[Dict]:
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
            # transformers recent versions renamed the object detection post
            # processing helper.  Use grounded_object_detection and provide the
            # text labels so the processor can return them directly.
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                text_labels=[queries],
            )
            
            detections = []
            for result in results:
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                # the processor now returns string labels per detection
                text_labels = result.get("text_labels", [])
                
                for box, score, text_label in zip(boxes, scores, text_labels):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    scale_x = orig_w / pil_image.size[0]
                    scale_y = orig_h / pil_image.size[1]
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
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
