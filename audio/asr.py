"""
语音识别模块 (ASR - Automatic Speech Recognition)
使用 OpenAI Whisper 进行语音转文字
"""

import torch
import logging
import numpy as np
import re
from typing import Optional
import warnings

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
                 language: str = "zh,en"):
        """
        初始化 ASR 引擎
        
        参数:
            model_name: Whisper 模型名称 (tiny/base/small/medium/large)
            device: 运行设备 ('cuda' 或 'cpu')
            language: 语言代码 ('zh,en' 中英双语自动判断, 'auto' 全语言自动检测, 'zh' 中文, 'en' 英文等)
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper 未安装。请运行: pip install openai-whisper")
        
        self.model_name = model_name
        language = (language or "auto").strip().lower()
        self.language = language
        
        # 自动检测设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，切换到 CPU")
            device = "cpu"
        self.device = device
        
        if self.language == "zh,en":
            lang_desc = "bilingual zh/en"
        elif self.language == "auto":
            lang_desc = "auto-detect"
        else:
            lang_desc = self.language
        logger.info(f"正在加载 Whisper 模型: {model_name} (设备: {device}, 语言: {lang_desc})")
        
        try:
            # 加载 Whisper 模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = whisper.load_model(model_name, device=device)
            
            logger.info(f"ASR 引擎初始化成功: {model_name}")
            
        except Exception as e:
            logger.error(f"Whisper 模型加载失败: {e}")
            raise
    
    def _build_mel(self, audio_data: np.ndarray) -> torch.Tensor:
        """生成与当前模型匹配的 mel 频谱（自动读取模型所需的 n_mels）。"""
        n_mels = self.model.dims.n_mels
        audio_data = whisper.pad_or_trim(audio_data.astype(np.float32))
        return whisper.log_mel_spectrogram(audio_data, n_mels=n_mels).to(self.model.device)

    def _detect_language_from_audio_array(self, audio_data: np.ndarray) -> str:
        """从音频波形中检测语言。"""
        mel = self._build_mel(audio_data)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def _resolve_language_for_audio(self, audio_data: np.ndarray) -> str:
        """将配置语言解析为实际用于本次识别的语言。"""
        if self.language == "auto":
            return self._detect_language_from_audio_array(audio_data)

        if self.language == "zh,en":
            mel = self._build_mel(audio_data)
            _, probs = self.model.detect_language(mel)
            zh_prob = probs.get("zh", 0.0)
            en_prob = probs.get("en", 0.0)
            return "zh" if zh_prob >= en_prob else "en"

        return self.language

    def _build_transcribe_kwargs(self, audio_data: Optional[np.ndarray] = None) -> tuple[dict, str]:
        kwargs = {
            'verbose': False,
            'fp16': (self.device == "cuda")
        }
        resolved_language = self.language
        if audio_data is not None:
            resolved_language = self._resolve_language_for_audio(audio_data)
        elif self.language != "auto":
            resolved_language = self.language

        if resolved_language != "auto":
            kwargs['language'] = resolved_language
        return kwargs, resolved_language

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

            audio_data = whisper.load_audio(audio_path)
            kwargs, resolved_language = self._build_transcribe_kwargs(audio_data)

            # 使用 Whisper 转录
            result = self.model.transcribe(
                audio_path,
                **kwargs
            )
            
            text = result['text'].strip()
            logger.info(f"ASR 识别结果: '{text}'")
            
            output = {
                'text': text,
                'language': result.get('language', resolved_language)
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

            kwargs, resolved_language = self._build_transcribe_kwargs(audio_data)

            # 使用 Whisper 转录
            result = self.model.transcribe(
                audio_data,
                **kwargs
            )
            
            text = result['text'].strip()
            logger.info(f"ASR 识别结果: '{text}'")
            
            return {
                'text': text,
                'language': result.get('language', resolved_language)
            }
            
        except Exception as e:
            logger.error(f"音频数据转录失败: {e}")
            return {'text': '', 'language': self.language, 'error': str(e)}

    def _normalize_command_text(self, text: str) -> str:
        """标准化语音文本，便于统一做控制词与目标提取。"""
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        for filler in ["请帮我", "帮我", "请", "一下", "一下子", "麻烦", "我想", "我想要"]:
            text = text.replace(filler, "")
        return text.strip()

    def _match_control_action(self, normalized_text: str) -> Optional[dict]:
        """优先匹配退出或停止类控制动作。"""
        control_keywords = [
            "退出程序",
            "停止任务",
            "退出任务",
            "结束任务",
            "停止",
            "退出",
            "stop task",
            "exit program",
            "quit program",
            "stop",
            "quit",
            "exit",
        ]
        for keyword in control_keywords:
            if keyword in normalized_text:
                return {
                    "status": "ok",
                    "action": "user_voice_exit",
                    "target": None,
                    "message": "正在退出程序",
                }
        return None
    
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

    def parse_voice_event(
        self,
        text: str,
        has_active_task: bool,
        frames: Optional[list] = None,
        llm_parser=None,
    ) -> dict:
        """
        解析语音结果，返回统一的动作事件对象。
        """
        normalized_text = self._normalize_command_text(text)
        if not normalized_text:
            return {
                "status": "error",
                "action": None,
                "target": None,
                "message": "没有识别到内容，请重试",
            }

        control_event = self._match_control_action(normalized_text)
        if control_event:
            return control_event

        target = self.parse_command_with_vision(
            text=normalized_text,
            frames=frames,
            llm_parser=llm_parser,
        )
        if not target:
            return {
                "status": "error",
                "action": None,
                "target": None,
                "message": "抱歉，无法理解您的指令",
            }

        action = "switch_target" if has_active_task else "set_target"
        target = target.strip()
        message = (
            f"原任务结束，切换至目标主体{target}"
            if action == "switch_target"
            else f"开始寻找目标主体{target}"
        )
        return {
            "status": "ok",
            "action": action,
            "target": target,
            "message": message,
        }
    
    def parse_command_with_vision(self,
                                   text: str,
                                   frames: Optional[list] = None,
                                   llm_parser=None) -> Optional[str]:
        """
        解析用户指令，优先使用 LLM + 视觉上下文，回退到正则解析
        
        参数:
            text: ASR 识别到的文本
            frames: 摄像头帧列表 (numpy arrays, BGR format) 用于视觉上下文
            llm_parser: LLMVisionParser 实例（可选）
            
        返回:
            提取的目标物体名称
            
        工作流程:
            1. 如果提供了 LLM parser 和 frames，尝试使用 LLM 解析
            2. 如果 LLM 解析成功，返回 LLM 结果
            3. 如果 LLM 失败或未提供，回退到正则 parse_command()
            4. 记录使用的方法 (LLM vs Regex)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to parse_command_with_vision")
            return None
        
        # 尝试 LLM 视觉解析
        if llm_parser is not None and frames:
            try:
                logger.debug(f"Attempting LLM vision parsing: text='{text}', frames={len(frames)}")
                result = llm_parser.parse_with_vision(text, frames)
                
                if result and 'target' in result:
                    target = result['target'].strip()
                    if target:
                        logger.info(
                            f"LLM vision parsing succeeded: "
                            f"'{text}' -> '{target}'"
                        )
                        return target
                else:
                    logger.debug("LLM vision parsing returned empty result")
                    
            except Exception as e:
                logger.error(
                    f"LLM vision parsing exception: {e}. "
                    f"Falling back to regex parsing.",
                    exc_info=True
                )
        else:
            if llm_parser is None:
                logger.debug("No LLM parser provided, using regex parsing")
            if not frames:
                logger.debug("No frames provided, using regex parsing")
        
        # 回退到正则解析
        logger.debug(f"Falling back to regex parse_command: '{text}'")
        target = self.parse_command(text)
        
        if target:
            logger.info(f"Regex parsing result: '{text}' -> '{target}'")
            return target
        
        logger.warning(f"Both LLM and regex parsing failed for: '{text}'")
        return None

