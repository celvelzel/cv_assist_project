"""
MediaPipe 手部追踪模块
======================
实时手部关键点检测和手势识别。
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)


class HandTracker:
    """MediaPipe 手部追踪器"""
    
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    MIDDLE_MCP = 9
    
    def __init__(self, max_num_hands: int = 1, min_confidence: float = 0.5):
        logger.info("初始化 MediaPipe 手部追踪器")
        
        self.max_num_hands = max_num_hands
        self.min_confidence = min_confidence
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            if not os.path.exists(model_path):
                logger.info("下载手部追踪模型...")
                url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
                try:
                    urllib.request.urlretrieve(url, model_path)
                    logger.info(f"模型已保存: {model_path}")
                except Exception as e:
                    logger.error(f"下载模型失败: {e}", exc_info=True)
                    raise RuntimeError(f"无法下载 MediaPipe 手部追踪模型: {e}")
            else:
                logger.info(f"使用现有模型: {model_path}")
            
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            
            logger.info("手部追踪器初始化完成")
            
        except Exception as e:
            logger.error(f"手部追踪器初始化失败: {e}", exc_info=True)
            logger.error("请检查:")
            logger.error("  1. MediaPipe 是否正确安装")
            logger.error("  2. 模型文件是否存在或网络是否可用")
            logger.error("  3. 磁盘空间是否充足")
            raise RuntimeError(f"无法初始化手部追踪器: {e}")
    
    def detect(self, image: np.ndarray) -> Dict:
        from mediapipe import Image as MPImage
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = self.detector.detect(mp_image)
        
        output = {'hands': []}
        
        if not results.hand_landmarks:
            return output
        
        h, w = image.shape[:2]
        
        for idx, landmarks in enumerate(results.hand_landmarks):
            keypoints_2d = []
            keypoints_3d = []
            
            for lm in landmarks:
                keypoints_2d.append((int(lm.x * w), int(lm.y * h)))
                keypoints_3d.append((lm.x, lm.y, lm.z))
            
            center_x = int(sum(kp[0] for kp in keypoints_2d) / 21)
            center_y = int(sum(kp[1] for kp in keypoints_2d) / 21)
            
            gesture = self._detect_gesture(keypoints_2d)
            
            hand_label = 'Unknown'
            if results.handedness and idx < len(results.handedness):
                hand_label = results.handedness[idx][0].category_name
            
            output['hands'].append({
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'center': (center_x, center_y),
                'gesture': gesture,
                'handedness': hand_label
            })
        
        return output
    
    def _detect_gesture(self, kps: List[Tuple[int, int]]) -> str:
        if len(kps) < 21:
            return 'unknown'
        
        def dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        
        wrist = kps[self.WRIST]
        thumb = kps[self.THUMB_TIP]
        index = kps[self.INDEX_TIP]
        middle = kps[self.MIDDLE_TIP]
        ring = kps[self.RING_TIP]
        pinky = kps[self.PINKY_TIP]
        mcp = kps[self.MIDDLE_MCP]
        
        palm = dist(wrist, mcp)
        if palm < 10:
            return 'unknown'
        
        idx_d = dist(wrist, index) / palm
        mid_d = dist(wrist, middle) / palm
        ring_d = dist(wrist, ring) / palm
        pinky_d = dist(wrist, pinky) / palm
        thumb_pinky = dist(thumb, pinky) / palm
        
        if idx_d > 1.8 and mid_d < 1.5 and ring_d < 1.5 and pinky_d < 1.5:
            return 'pointing'
        
        if thumb_pinky > 2.0:
            return 'open'
        
        avg = (idx_d + mid_d + ring_d + pinky_d) / 4
        if avg < 1.3:
            return 'closed'
        
        return 'unknown'
    
    def draw(self, image: np.ndarray, results: Dict) -> np.ndarray:
        output = image.copy()
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for hand in results['hands']:
            kps = hand['keypoints_2d']
            
            for i, j in connections:
                cv2.line(output, kps[i], kps[j], (0, 255, 0), 2)
            
            for kp in kps:
                cv2.circle(output, kp, 3, (0, 255, 0), -1)
            
            center = hand['center']
            cv2.circle(output, center, 8, (255, 0, 0), -1)
            
            text = f"{hand['handedness']}: {hand['gesture']}"
            cv2.putText(output, text, (center[0] + 15, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return output
