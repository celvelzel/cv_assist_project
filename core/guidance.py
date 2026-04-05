"""
引导控制器
==========
根据手部和目标位置生成导航指令。
"""

import cv2
import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import os
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def clock_hour_from_hand_to_target(
    dx: int, dy: int, *, flip_horizontal: bool = False
) -> int:
    """以手为表心，目标相对手的钟点方向（1–12）。

    数据：仅使用手部与目标在**同一图像坐标系**下的中心点差值 ``dx = target_x - hand_x``、
    ``dy = target_y - hand_y``（与 ``GuidanceController.calculate`` 中定义一致）。
    **不使用深度图**；钟点表示的是像平面上的方位（把屏幕当成表盘：12 点 = 画面正上方）。

    算法：用 ``atan2(dx, -dy)`` 得到相对「向上」的方位角，再按每 30° 一扇区映射到 1–12 点
    （扇区边界用 floor，比 round 更稳定，减少边界抖动）。

    flip_horizontal=True 时对 dx 取反（例如原始帧未镜像、而钟点要以镜像自拍视角为准时）。
    """
    if flip_horizontal:
        dx = -int(dx)
    angle_deg = math.degrees(math.atan2(float(dx), float(-dy)))
    angle_deg = (angle_deg + 360.0) % 360.0
    sector = int(math.floor((angle_deg + 15.0) / 30.0)) % 12
    return 12 if sector == 0 else sector


def clock_hour_from_horizontal_plane(
    dx: int,
    depth_diff: float,
    *,
    flip_horizontal: bool = False,
    depth_scale: float = 450.0,
    depth_axis_sign: float = 1.0,
) -> int:
    """以水平面为表盘估计钟点（1–12），用左右 + 深度差，不用画面上下 dy。

    将目标相对手的位置近似投影到「水平面」：横轴为画面左右偏移 ``dx``（像素），
    纵轴为 MiDaS 归一化深度差 ``depth_diff = target_depth - hand_depth``（与引导中一致），
    乘以 ``depth_scale`` 后与 ``dx`` 量级相当，便于 ``atan2`` 合成方向。

    - **12 点**：目标在手的「正前方」一侧（沿深度轴分量为主，dx≈0）。
    - **3 点**：目标在手的右侧（dx>0，镜像后坐标系下多为用户右手侧）。
    - **6 点**：主要在深度轴反向。
    - **9 点**：左侧。

    ``depth_axis_sign`` 若与实际前后感相反，可在配置中改为 -1.0。
    ``depth_scale`` 越大，同样深度差对钟点角度影响越大。
    """
    if flip_horizontal:
        dx = -int(dx)
    scale = max(1e-6, float(depth_scale))
    v = float(depth_diff) * float(depth_axis_sign) * scale
    angle_deg = math.degrees(math.atan2(float(dx), float(v)))
    angle_deg = (angle_deg + 360.0) % 360.0
    sector = int(math.floor((angle_deg + 15.0) / 30.0)) % 12
    return 12 if sector == 0 else sector


def approximate_separation_meters(
    dx: int,
    dy: int,
    depth_diff: float,
    min_m: float,
    max_m: float,
    ref_diagonal_px: float,
    *,
    px_weight: float = 0.68,
    depth_weight: float = 0.32,
    depth_span: float = 0.5,
    distance_gamma: float = 1.12,
) -> float:
    """根据像素间距与相对深度差给出「约 X 米」的提示值（启发式，非真实测距）。

    - ``t_px``：手到目标的像素距离 / ``ref_diagonal_px``，截断到 [0,1]。
    - ``t_d``：``|depth_diff| / depth_span``，截断到 [0,1]（不再用 ``*2`` 以免深度项过快顶满）。
    - ``t = px_weight * t_px + depth_weight * t_d``，再 ``t ** distance_gamma``（默认略>1，整体略压低报数，减轻「偏大」体感）。
    单目 MiDaS 仅为相对深度，米数仅供口头参考，请用配置微调。
    """
    px = math.hypot(float(dx), float(dy))
    denom = max(1.0, float(ref_diagonal_px))
    t_px = min(1.0, px / denom)
    span = max(1e-6, float(depth_span))
    t_d = min(1.0, abs(float(depth_diff)) / span)
    wp = max(0.0, float(px_weight))
    wd = max(0.0, float(depth_weight))
    s = wp + wd
    if s <= 0:
        t = (t_px + t_d) / 2.0
    else:
        t = (wp * t_px + wd * t_d) / s
    t = max(0.0, min(1.0, t))
    g = max(0.05, float(distance_gamma))
    t = t**g
    return float(min_m + t * (max_m - min_m))


@dataclass
class GuidanceResult:
    """引导结果数据类
    
    包含引导系统输出的所有信息：
    - instruction: 给用户的中文指令
    - direction_h/v/d: 水平/垂直/深度方向指示
    - dx/dy/depth_diff: 具体的偏移值
    - ready_to_grab: 是否已就位，可以执行抓取
    """
    instruction: str       # 显示给用户的指令文本
    direction_h: str       # 水平方向: 'left', 'right', 'center'
    direction_v: str       # 垂直方向: 'up', 'down', 'center'
    direction_d: str       # 深度方向: 'forward', 'backward', 'hold'
    dx: int                # 水平偏移（像素）
    dy: int                # 垂直偏移（像素）
    depth_diff: float      # 深度差异 (0-1)
    ready_to_grab: bool    # 是否就位可抓取
    stable_ready_frames: int  # 当前已连续稳定对齐的帧数
    state: str             # 语义状态: 'moving' | 'ready' | 'grabbed'


class GuidanceController:
    """引导控制器
    
    根据手部和目标物体的相对位置（水平、垂直、深度）生成实时导航指令。
    
    工作原理:
    1. 计算手部中心与目标中心的位置差
    2. 根据阈值判断是否对齐（水平/垂直/深度）
    3. 生成相应的中文指令
    4. 当三个维度都对齐时，提示用户可以抓取
    """
    
    def __init__(self,
                 horizontal_threshold: int = 30,
                 vertical_threshold: int = 30,
                 depth_threshold: float = 0.15,
                 horizontal_threshold_enter: Optional[int] = None,
                 horizontal_threshold_exit: Optional[int] = None,
                 vertical_threshold_enter: Optional[int] = None,
                 vertical_threshold_exit: Optional[int] = None,
                 depth_threshold_enter: Optional[float] = None,
                 depth_threshold_exit: Optional[float] = None,
                 grasp_stable_frames: int = 8,
                 grasp_release_frames: int = 3,
                 depth_instruction_first: bool = False,
                 invert_depth_guidance: bool = False):
        """
        初始化引导控制器
        
        参数:
            horizontal_threshold: 水平方向的容差阈值（像素）
                                 在此范围内认为已对齐
            vertical_threshold: 垂直方向的容差阈值（像素）
            depth_threshold: 深度方向的容差阈值 (0-1)
            depth_instruction_first: 为 True 时，移动指令优先播报前后，再上下/左右
            invert_depth_guidance: True 时对深度差取反再判定前后
        """
        self.h_thresh = horizontal_threshold
        self.v_thresh = vertical_threshold
        self.d_thresh = depth_threshold
        self.h_enter = horizontal_threshold_enter if horizontal_threshold_enter is not None else horizontal_threshold
        self.h_exit = horizontal_threshold_exit if horizontal_threshold_exit is not None else horizontal_threshold
        self.v_enter = vertical_threshold_enter if vertical_threshold_enter is not None else vertical_threshold
        self.v_exit = vertical_threshold_exit if vertical_threshold_exit is not None else vertical_threshold
        self.d_enter = depth_threshold_enter if depth_threshold_enter is not None else depth_threshold
        self.d_exit = depth_threshold_exit if depth_threshold_exit is not None else depth_threshold
        self.grasp_stable_frames = max(1, int(grasp_stable_frames))
        self.grasp_release_frames = max(1, int(grasp_release_frames))
        self.depth_instruction_first = bool(depth_instruction_first)
        self.invert_depth_guidance = bool(invert_depth_guidance)

        # 状态缓存用于滞回与稳定帧判定
        self._prev_dir_h = 'center'
        self._prev_dir_v = 'center'
        self._prev_dir_d = 'hold'
        self._ready_streak = 0
        self._not_ready_streak = 0
        self._ready_latched = False
        
        logger.info(f"引导控制器初始化: h={horizontal_threshold}, v={vertical_threshold}, d={depth_threshold}")

    def _direction_with_hysteresis(self,
                                   value: float,
                                   prev_state: str,
                                   enter_th: float,
                                   exit_th: float,
                                   positive_label: str,
                                   negative_label: str,
                                   center_label: str) -> str:
        """使用内外阈值进行方向判定，避免临界抖动。"""
        abs_value = abs(value)
        if abs_value <= enter_th:
            return center_label
        if abs_value > exit_th:
            return positive_label if value > 0 else negative_label
        return prev_state
    
    def calculate(self,
                  hand_center: Tuple[int, int],
                  target_center: Tuple[int, int],
                  hand_depth: float,
                  target_depth: float,
                  gesture: str = 'unknown') -> GuidanceResult:
        """
        计算引导指令
        
        参数:
            hand_center: 手部中心点 (x, y)
            target_center: 目标物体中心点 (x, y)
            hand_depth: 手部深度值 (0-1)
            target_depth: 目标深度值 (0-1)
            gesture: 当前手势
            
        返回:
            GuidanceResult 对象，包含引导指令和方向信息
        """
        # 计算位置差异
        dx = target_center[0] - hand_center[0]  # 水平偏移（正值=目标在右）
        dy = target_center[1] - hand_center[1]  # 垂直偏移（正值=目标在下）
        # MiDaS 归一化深度：同一帧内数值越大通常越靠近相机（见 depth_estimator 伪彩说明）
        dd = target_depth - hand_depth  # 正值≈目标比手更靠近镜头
        if self.invert_depth_guidance:
            dd = -dd

        # 使用滞回阈值判定方向，减少边界来回跳变
        dir_h = self._direction_with_hysteresis(
            dx, self._prev_dir_h, self.h_enter, self.h_exit, 'right', 'left', 'center'
        )
        dir_v = self._direction_with_hysteresis(
            dy, self._prev_dir_v, self.v_enter, self.v_exit, 'down', 'up', 'center'
        )
        dir_d = self._direction_with_hysteresis(
            dd, self._prev_dir_d, self.d_enter, self.d_exit, 'forward', 'backward', 'hold'
        )

        self._prev_dir_h = dir_h
        self._prev_dir_v = dir_v
        self._prev_dir_d = dir_d

        raw_ready = (dir_h == 'center' and dir_v == 'center' and dir_d == 'hold')

        # 连续稳定帧：进入抓握态需满足 N 帧，退出也要求连续不满足若干帧
        if raw_ready:
            self._ready_streak += 1
            self._not_ready_streak = 0
        else:
            self._ready_streak = 0
            self._not_ready_streak += 1

        if not self._ready_latched and self._ready_streak >= self.grasp_stable_frames:
            self._ready_latched = True
        elif self._ready_latched and self._not_ready_streak >= self.grasp_release_frames:
            self._ready_latched = False

        ready = self._ready_latched
        
        # 生成指令文本
        if ready:
            # 已就位，根据手势给出不同指令
            if gesture == 'open':
                instruction = "抓取! 闭合手掌"
                state = 'ready'
            elif gesture == 'closed':
                instruction = "已抓住!"
                state = 'grabbed'
            else:
                instruction = "准备抓取!"
                state = 'ready'
        else:
            # 未就位，创建移动指令（可按配置优先播报前后移动）
            parts_h, parts_v, parts_d = [], [], []
            if dir_h != 'center':
                parts_h.append(f"向{self._translate(dir_h)}移动")
            if dir_v != 'center':
                parts_v.append(f"向{self._translate(dir_v)}移动")
            if dir_d != 'hold':
                parts_d.append(f"向{self._translate(dir_d)}移动")
            if self.depth_instruction_first:
                parts = parts_d + parts_v + parts_h
            else:
                parts = parts_h + parts_v + parts_d
            instruction = " | ".join(parts) if parts else "保持位置"
            state = 'moving'
        
        return GuidanceResult(
            instruction=instruction,
            direction_h=dir_h,
            direction_v=dir_v,
            direction_d=dir_d,
            dx=dx,
            dy=dy,
            depth_diff=dd,
            ready_to_grab=ready,
            stable_ready_frames=self._ready_streak,
            state=state
        )
    
    def _translate(self, direction: str) -> str:
        """将英文方向转换为中文"""
        mapping = {
            'left': '左', 'right': '右',
            'up': '上', 'down': '下',
            'forward': '前', 'backward': '后'
        }
        return mapping.get(direction, direction)

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        获取中文字体
        
        尝试加载系统中的中文字体，按优先级顺序尝试多个字体文件。
        如果都加载失败，使用默认字体（可能不支持中文）。
        
        参数:
            size: 字体大小
            
        返回:
            PIL 字体对象
        """
        # 字体文件候选列表（按优先级排序）
        candidates = [
            os.environ.get("CV_ASSIST_FONT"),           # 环境变量指定
            "C:\\Windows\\Fonts\\msyh.ttc",            # Windows 微软雅黑
            "C:\\Windows\\Fonts\\simhei.ttf",          # Windows 黑体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿正黑
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux 默认字体
        ]
        # 尝试加载每个候选字体
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        # 所有字体都加载失败，使用默认字体
        return ImageFont.load_default()

    def _draw_text(self, image: np.ndarray, text: str, origin: Tuple[int, int],
                   color: Tuple[int, int, int], size: int = 22) -> np.ndarray:
        """
        在图像上绘制文本（支持中文）
        
        OpenCV 的 putText 不支持中文，所以使用 PIL 绘制中文。
        如果是纯英文，直接使用 OpenCV 以提升性能。
        
        参数:
            image: 图像
            text: 要绘制的文本
            origin: 文本起始位置 (x, y)
            color: 文本颜色 (B, G, R)
            size: 字体大小
            
        返回:
            绘制后的图像
        """
        # 判断是否为纯 ASCII 字符
        if all(ord(ch) < 128 for ch in text):
            # 纯英文，直接使用 OpenCV
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return image

        # 包含中文，使用 PIL 绘制
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        font = self._get_font(size)
        # PIL 使用 RGB 顺序，需要从 BGR 转换
        draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def draw(self, image: np.ndarray, hand_center: Tuple[int, int],
             target_center: Tuple[int, int], result: GuidanceResult) -> np.ndarray:
        """
        在图像上绘制引导信息
        
        绘制内容:
        - 手部到目标的连接线
        - 引导指令文本（带背景）
        - 偏移信息
        
        参数:
            image: 输入图像
            hand_center: 手部中心点
            target_center: 目标中心点
            result: 引导结果
            
        返回:
            绘制后的图像
        """
        output = image.copy()
        
        # 根据是否就位选择颜色：绿色=已就位，黄色=未就位
        color = (0, 255, 0) if result.ready_to_grab else (0, 255, 255)
        # 绘制手部到目标的连接线
        cv2.line(output, hand_center, target_center, color, 2)
        
        # 绘制指令文本（带背景框）
        if all(ord(ch) < 128 for ch in result.instruction):
            # 英文文本，使用 OpenCV
            (tw, th), _ = cv2.getTextSize(result.instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(output, (10, 10), (20 + tw, 20 + th + 8), (0, 0, 0), -1)  # 黑色背景
            cv2.putText(output, result.instruction, (15, 15 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # 中文文本，使用 PIL
            font = self._get_font(24)
            bbox = font.getbbox(result.instruction)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cv2.rectangle(output, (10, 10), (20 + tw, 20 + th + 8), (0, 0, 0), -1)
            output = self._draw_text(output, result.instruction, (15, 15), color, size=24)
        
        # 绘制详细偏移信息（在屏幕底部）
        info = f"dx:{result.dx} dy:{result.dy} d:{result.depth_diff:.2f}"
        cv2.putText(output, info, (15, output.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output
