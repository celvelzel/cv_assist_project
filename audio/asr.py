"""
语音识别模块 (ASR - Automatic Speech Recognition)
使用 OpenAI Whisper 进行语音转文字
"""

import torch
import logging
import numpy as np
from typing import Optional
import warnings

from core.interfaces import ParsedIntent
from audio.intent_parser import create_intent_parser

logger = logging.getLogger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper 未安装，ASR 功能将不可用")


class ASREngine:
    """
    语音识别引擎
    
    使用 OpenAI Whisper 模型将语音转换为文本。
    支持多种语言，中文识别效果优秀。
    
    模型大小:
    - tiny: 最快，准确率较低 (~39M)
    - base: 速度快，准确率一般 (~74M)
    - small: 平衡选择 (~244M) - 默认推荐
    - medium: 准确率高，速度较慢 (~769M)
    - large: 最准确，速度慢 (~1550M)
    """
    
    def __init__(self, 
                 model_name: str = "base",
                 device: str = "cuda",
                 language: str = "zh",
                 intent_parser_type: str = "regex"):
        """
        初始化 ASR 引擎
        
        参数:
            model_name: Whisper 模型名称 (tiny/base/small/medium/large)
            device: 运行设备 ('cuda' 或 'cpu')
            language: 语言代码 ('zh' 中文, 'en' 英文等)
            intent_parser_type: 意图解析器类型 ("regex" 或未来 "llm")
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper 未安装。请运行: pip install openai-whisper")
        
        self.model_name = model_name
        self.language = language
        
        # 自动检测设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，切换到 CPU")
            device = "cpu"
        self.device = device
        
        # 初始化意图解析器
        self.intent_parser = create_intent_parser(intent_parser_type)
        logger.info(f"意图解析器初始化成功: {intent_parser_type}")
        
        logger.info(f"正在加载 Whisper 模型: {model_name} (设备: {device})")
        
        try:
            # 加载 Whisper 模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = whisper.load_model(model_name, device=device)
            
            logger.info(f"ASR 引擎初始化成功: {model_name}")
            
        except Exception as e:
            logger.error(f"Whisper 模型加载失败: {e}")
            raise
    
    def transcribe_file(self, audio_path: str, return_timestamps: bool = False) -> dict:
        """
        转录音频文件
        
        参数:
            audio_path: 音频文件路径 (支持 wav, mp3, m4a 等格式)
            return_timestamps: 是否返回时间戳信息
            
        返回:
            包含识别结果的字典:
            - text: 识别的文本
            - language: 检测到的语言
            - segments: 分段信息 (如果 return_timestamps=True)
        """
        try:
            logger.debug(f"开始转录: {audio_path}")
            
            # 使用 Whisper 转录
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                verbose=False,
                fp16=(self.device == "cuda")  # GPU 使用 FP16 加速
            )
            
            text = result['text'].strip()
            logger.info(f"ASR 识别结果: '{text}'")
            
            output = {
                'text': text,
                'language': result.get('language', self.language)
            }
            
            if return_timestamps:
                output['segments'] = result.get('segments', [])
            
            return output
            
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            return {'text': '', 'language': self.language, 'error': str(e)}
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        转录音频数据 (NumPy 数组)
        
        参数:
            audio_data: 音频数据 (NumPy 数组, float32, 范围 -1.0 到 1.0)
            sample_rate: 采样率 (Whisper 内部会重采样到 16kHz)
            
        返回:
            包含识别结果的字典
        """
        try:
            # 确保音频数据格式正确
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 规范化到 -1.0 到 1.0 范围
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            logger.debug(f"转录音频数据: {len(audio_data)} samples @ {sample_rate}Hz")
            
            # 使用 Whisper 转录
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                verbose=False,
                fp16=(self.device == "cuda")
            )
            
            text = result['text'].strip()
            logger.info(f"ASR 识别结果: '{text}'")
            
            return {
                'text': text,
                'language': result.get('language', self.language)
            }
            
        except Exception as e:
            logger.error(f"音频数据转录失败: {e}")
            return {'text': '', 'language': self.language, 'error': str(e)}
    
    def parse_command(self, text: str) -> Optional[str]:
        """
        解析用户指令，提取目标物体名称
        
        参数:
            text: 识别到的文本
            
        返回:
            提取的目标物体名称，如果无法解析则返回 None
            
        示例:
            "找到杯子" -> "杯子"
            "帮我找一下手机" -> "手机"
            "where is the cup" -> "cup"
        """
        text = text.strip().lower()
        
        # 中文命令模式
        keywords_zh = ['找', '找到', '寻找', '搜索', '定位', '在哪', '哪里']
        
        for keyword in keywords_zh:
            if keyword in text:
                # 提取关键词之后的内容
                idx = text.index(keyword)
                target = text[idx + len(keyword):].strip()
                
                # 移除常见的语气词和连接词
                for filler in ['一下', '我的', '这个', '那个']:
                    target = target.replace(filler, '')
                
                target = target.strip()
                if target:
                    logger.info(f"解析指令: '{text}' -> 目标: '{target}'")
                    return target
        
        # 英文命令模式
        keywords_en = ['find', 'locate', 'where', 'search', 'look for']
        
        for keyword in keywords_en:
            if keyword in text:
                words = text.split()
                if keyword in words:
                    idx = words.index(keyword)
                    # 提取关键词之后的内容
                    if idx + 1 < len(words):
                        target = ' '.join(words[idx+1:])
                        # 移除常见的冠词
                        target = target.replace('the ', '').replace('a ', '').replace('an ', '')
                        target = target.strip()
                        if target:
                            logger.info(f"解析指令: '{text}' -> 目标: '{target}'")
                            return target
        
        # 如果没有明确的命令关键词，返回整个文本作为目标
        logger.warning(f"无法解析指令: '{text}'，使用全文作为目标")
        return text if text else None
