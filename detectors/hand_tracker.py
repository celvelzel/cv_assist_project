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
    """MediaPipe 手部追踪器
    
    使用 Google MediaPipe 库进行实时手部检测和关键点识别。
    支持21个手部关键点的检测，包括手腕、手指关节和指尖等。
    
    关键点索引常量:
        WRIST: 0 - 手腕
        THUMB_TIP: 4 - 姆指指尖
        INDEX_TIP: 8 - 食指指尖
        MIDDLE_TIP: 12 - 中指指尖
        RING_TIP: 16 - 无名指指尖
        PINKY_TIP: 20 - 小指指尖
        MIDDLE_MCP: 9 - 中指掌指关节
    """
    
    # MediaPipe 手部关键点索引常量
    # 这些索引对应 MediaPipe 手部模型输出的 21 个关键点
    WRIST = 0          # 手腕，作为手部的基准点
    THUMB_TIP = 4      # 姆指指尖
    INDEX_TIP = 8      # 食指指尖
    MIDDLE_TIP = 12    # 中指指尖
    RING_TIP = 16      # 无名指指尖
    PINKY_TIP = 20     # 小指指尖
    MIDDLE_MCP = 9     # 中指掌指关节，用于计算手掌大小
    
    def __init__(self, max_num_hands: int = 1, min_confidence: float = 0.5):
        """
        初始化手部追踪器
        
        参数:
            max_num_hands: 最多检测的手部数量，默认1只手
            min_confidence: 最小置信度阈值 (0-1)，低于此值的检测将被忽略
        """
        logger.info("初始化 MediaPipe 手部追踪器")
        
        self.max_num_hands = max_num_hands
        self.min_confidence = min_confidence
        
        try:
            # 检查模型文件是否存在
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            if not os.path.exists(model_path):
                logger.info("下载手部追踪模型...")
                # MediaPipe 官方模型下载链接
                url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
                try:
                    urllib.request.urlretrieve(url, model_path)
                    logger.info(f"模型已保存: {model_path}")
                except Exception as e:
                    logger.error(f"下载模型失败: {e}", exc_info=True)
                    raise RuntimeError(f"无法下载 MediaPipe 手部追踪模型: {e}")
            else:
                logger.info(f"使用现有模型: {model_path}")
            
            # 创建 MediaPipe 手部检测器
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,  # 最多检测手部数量
                min_hand_detection_confidence=min_confidence,  # 手部检测的最小置信度
                min_hand_presence_confidence=min_confidence,   # 手部存在的最小置信度
                min_tracking_confidence=min_confidence         # 手部追踪的最小置信度
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
        """
        检测图像中的手部并识别手势
        
        参数:
            image: 输入图像 (BGR 格式的 numpy 数组)
            
        返回:
            字典包含 'hands' 列表，每个手部包含:
            - keypoints_2d: 21个关键点的 2D 坐标 (x, y)
            - keypoints_3d: 21个关键点的 3D 坐标 (x, y, z)
            - center: 手部中心点坐标
            - gesture: 识别的手势 ('open', 'closed', 'pointing', 'unknown')
            - handedness: 左手/右手 ('Left'/'Right')
        """
        from mediapipe import Image as MPImage
        
        # 将 BGR 图像转换为 RGB（MediaPipe 需要 RGB 格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 进行手部检测
        results = self.detector.detect(mp_image)
        
        output = {'hands': []}
        
        # 如果没有检测到手部，返回空列表
        if not results.hand_landmarks:
            return output
        
        h, w = image.shape[:2]
        
        # 遍历每只检测到的手
        for idx, landmarks in enumerate(results.hand_landmarks):
            keypoints_2d = []  # 2D 坐标列表
            keypoints_3d = []  # 3D 坐标列表
            
            # 将归一化坐标(0-1)转换为像素坐标
            for lm in landmarks:
                keypoints_2d.append((int(lm.x * w), int(lm.y * h)))
                keypoints_3d.append((lm.x, lm.y, lm.z))  # z 是相对于手腕的深度
            
            # 计算手部中心点（所有21个关键点的平均值）
            center_x = int(sum(kp[0] for kp in keypoints_2d) / 21)
            center_y = int(sum(kp[1] for kp in keypoints_2d) / 21)
            
            # 识别手势
            gesture = self._detect_gesture(keypoints_2d)
            
            # 获取手部方向（左手或右手）
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
        """
        根据手部关键点位置识别手势
        
        支持的手势:
        - 'open': 张开的手（所有手指展开）
        - 'closed': 闭合的手/拳头（所有手指卷起）
        - 'pointing': 指向手势（食指伸出，其他手指卷起）
        - 'unknown': 未识别的手势
        
        参数:
            kps: 21个关键点的 2D 坐标列表
            
        返回:
            手势名称字符串
        """
        if len(kps) < 21:
            return 'unknown'
        
        # 计算两点之间的欧几里得距离
        def dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        
        # 提取关键点
        wrist = kps[self.WRIST]        # 手腕
        thumb = kps[self.THUMB_TIP]    # 姆指指尖
        index = kps[self.INDEX_TIP]    # 食指指尖
        middle = kps[self.MIDDLE_TIP]  # 中指指尖
        ring = kps[self.RING_TIP]      # 无名指指尖
        pinky = kps[self.PINKY_TIP]    # 小指指尖
        mcp = kps[self.MIDDLE_MCP]     # 中指掌指关节
        
        # 计算手掌大小（手腕到掌指关节的距离），用于归一化
        palm = dist(wrist, mcp)
        if palm < 10:  # 手掌太小，可能是检测错误
            return 'unknown'
        
        # 计算每个指尖到手腕的距离，并用手掌大小归一化
        idx_d = dist(wrist, index) / palm    # 食指伸展程度
        mid_d = dist(wrist, middle) / palm   # 中指伸展程度
        ring_d = dist(wrist, ring) / palm    # 无名指伸展程度
        pinky_d = dist(wrist, pinky) / palm  # 小指伸展程度
        thumb_pinky = dist(thumb, pinky) / palm  # 姆指到小指的距离，用于判断手是否张开
        
        # 判断指向手势：食指伸直，其他手指卷起
        if idx_d > 1.8 and mid_d < 1.5 and ring_d < 1.5 and pinky_d < 1.5:
            return 'pointing'
        
        # 判断张开的手：姆指和小指间距离较大
        if thumb_pinky > 2.0:
            return 'open'
        
        # 判断闭合的手：所有手指到手腕的平均距离较小
        avg = (idx_d + mid_d + ring_d + pinky_d) / 4
        if avg < 1.3:
            return 'closed'
        
        # 默认返回未知手势
        return 'unknown'
    
    def draw(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        在图像上绘制手部关键点和连接线
        
        参数:
            image: 输入图像
            results: detect() 方法返回的结果
            
        返回:
            绘制后的图像
        """
        output = image.copy()
        
        # 手部关键点之间的连接关系（模拟手指骨架）
        connections = [
            # 姆指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (0, 17), (17, 18), (18, 19), (19, 20),
            # 手掌连接
            (5, 9), (9, 13), (13, 17)
        ]
        
        for hand in results['hands']:
            kps = hand['keypoints_2d']
            
            # 绘制关键点之间的连接线
            for i, j in connections:
                cv2.line(output, kps[i], kps[j], (0, 255, 0), 2)
            
            # 绘制关键点
            for kp in kps:
                cv2.circle(output, kp, 3, (0, 255, 0), -1)
            
            # 绘制手部中心点
            center = hand['center']
            cv2.circle(output, center, 8, (255, 0, 0), -1)
            
            # 显示手部信息（左/右手 + 手势）
            text = f"{hand['handedness']}: {hand['gesture']}"
            cv2.putText(output, text, (center[0] + 15, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return output
