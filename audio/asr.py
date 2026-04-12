"""
语音识别模块 (ASR - Automatic Speech Recognition)
使用 faster-whisper（CTranslate2）进行语音转文字
"""

import logging
import time
import numpy as np
import re
from typing import Optional, List, Tuple
import warnings

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None  # type: ignore
    logger.warning("faster-whisper 未安装，ASR 功能将不可用")

try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ASREngine:
    """
    语音识别引擎（faster-whisper）

    模型大小与 OpenAI Whisper 命名一致:
    tiny / base / small / medium / large-v1 / large-v2 / large-v3 等
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda",
        language: str = "zh,en",
        compute_type: Optional[str] = None,
    ):
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper 未安装。请运行: pip install faster-whisper")

        self.model_name = model_name
        language = (language or "auto").strip().lower()
        self.language = language

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA 不可用，切换到 CPU")
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.compute_type = compute_type

        if self.language == "zh,en":
            lang_desc = "bilingual zh/en (auto-detect per utterance)"
        elif self.language == "auto":
            lang_desc = "auto-detect"
        else:
            lang_desc = self.language
        logger.info(
            "正在加载 faster-whisper 模型: %s (设备: %s, compute_type: %s, 语言: %s)",
            model_name,
            device,
            compute_type,
            lang_desc,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                )
            logger.info("ASR 引擎初始化成功: %s", model_name)
        except Exception as e:
            logger.error("faster-whisper 模型加载失败: %s", e)
            raise

    def _ensure_mono_float32(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        peak = float(np.abs(audio_data).max()) if audio_data.size else 0.0
        if peak > 1.0:
            audio_data = audio_data / peak
        return audio_data

    def _resample_to_16k(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_data = self._ensure_mono_float32(audio_data)
        if sample_rate == 16000:
            return audio_data
        if not SCIPY_AVAILABLE:
            raise RuntimeError("需要 scipy 以将非 16k 音频重采样: pip install scipy")
        num = int(len(audio_data) * 16000 / float(sample_rate))
        if num <= 0:
            return np.array([], dtype=np.float32)
        return scipy_signal.resample(audio_data, num).astype(np.float32)

    def _language_kw_for_transcribe(self, _audio_16k: np.ndarray) -> Tuple[Optional[str], str]:
        """返回 (language_kw_or_None, resolved_tag)。zh,en 与 auto 均交给模型在 transcribe 中检测。"""
        if self.language in ("auto", "zh,en"):
            return None, self.language
        return self.language, self.language

    def transcribe_file(self, audio_path: str, return_timestamps: bool = False) -> dict:
        try:
            logger.debug("开始转录文件: %s", audio_path)
            kwargs, resolved = self._build_transcribe_kwargs(None)
            t0 = time.perf_counter()
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                **kwargs,
            )
            seg_list = list(segments)
            text = "".join(seg.text for seg in seg_list).strip()
            asr_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("ASR 识别结果: '%s' | asr_ms=%.0fms", text, asr_ms)
            out_lang = info.language or resolved
            output = {"text": text, "language": out_lang, "asr_time_ms": asr_ms}
            if return_timestamps:
                output["segments"] = [
                    {"start": s.start, "end": s.end, "text": s.text} for s in seg_list
                ]
            return output
        except Exception as e:
            logger.error("音频转录失败: %s", e)
            return {"text": "", "language": self.language, "error": str(e)}

    def _build_transcribe_kwargs(self, audio_16k: Optional[np.ndarray]) -> Tuple[dict, str]:
        kwargs: dict = {}
        if audio_16k is None:
            if self.language in ("auto", "zh,en"):
                lang_kw, resolved = None, self.language
            else:
                lang_kw, resolved = self.language, self.language
        else:
            lang_kw, resolved = self._language_kw_for_transcribe(audio_16k)
        if lang_kw is not None:
            kwargs["language"] = lang_kw
        return kwargs, resolved

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        try:
            audio_16k = self._resample_to_16k(audio_data, sample_rate)
            if audio_16k.size == 0:
                return {"text": "", "language": self.language, "error": "empty audio"}
            logger.debug(
                "转录音频数据: %s samples @ 16kHz (src=%sHz)",
                len(audio_16k),
                sample_rate,
            )
            kwargs, resolved = self._build_transcribe_kwargs(audio_16k)
            t0 = time.perf_counter()
            segments, info = self.model.transcribe(
                audio_16k,
                beam_size=5,
                vad_filter=True,
                **kwargs,
            )
            text = "".join(seg.text for seg in segments).strip()
            asr_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("ASR 识别结果: '%s' | asr_ms=%.0fms", text, asr_ms)
            return {
                "text": text,
                "language": info.language or resolved,
                "asr_time_ms": asr_ms,
            }
        except Exception as e:
            logger.error("音频数据转录失败: %s", e)
            return {"text": "", "language": self.language, "error": str(e)}

    @staticmethod
    def strip_tts_echo_phrases(text: str, phrases: Optional[List[str]]) -> str:
        """从识别文本中移除已知 TTS 提示语片段（避免扬声器回灌）。"""
        if not text or not phrases:
            return (text or "").strip()
        stripped = text
        for p in phrases:
            if not p:
                continue
            stripped = stripped.replace(p, "")
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    def _normalize_command_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        for filler in ["请帮我", "帮我", "请", "一下", "一下子", "麻烦", "我想", "我想要"]:
            text = text.replace(filler, "")
        return text.strip()

    def _match_control_action(self, normalized_text: str) -> Optional[dict]:
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
        text = text.strip().lower()
        keywords_zh = ["找", "找到", "寻找", "搜索", "定位", "在哪", "哪里"]
        for keyword in keywords_zh:
            if keyword in text:
                idx = text.index(keyword)
                target = text[idx + len(keyword) :].strip()
                for filler in ["一下", "我的", "这个", "那个"]:
                    target = target.replace(filler, "")
                target = target.strip()
                if target:
                    logger.info("解析指令: '%s' -> 目标: '%s'", text, target)
                    return target
        keywords_en = ["find", "locate", "where", "search", "look for"]
        for keyword in keywords_en:
            if keyword in text:
                words = text.split()
                if keyword in words:
                    idx = words.index(keyword)
                    if idx + 1 < len(words):
                        target = " ".join(words[idx + 1 :])
                        target = target.replace("the ", "").replace("a ", "").replace("an ", "")
                        target = target.strip()
                        if target:
                            logger.info("解析指令: '%s' -> 目标: '%s'", text, target)
                            return target
        logger.warning("无法解析指令: '%s'，使用全文作为目标", text)
        return text if text else None

    def parse_voice_event(
        self,
        text: str,
        has_active_task: bool,
        frames: Optional[list] = None,
        llm_parser=None,
    ) -> dict:
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

        target, llm_poe_ms, llm_poe_invoked = self.parse_command_with_vision(
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
                "llm_poe_ms": llm_poe_ms,
                "llm_poe_invoked": llm_poe_invoked,
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
            "llm_poe_ms": llm_poe_ms,
            "llm_poe_invoked": llm_poe_invoked,
        }

    def parse_command_with_vision(
        self,
        text: str,
        frames: Optional[list] = None,
        llm_parser=None,
    ) -> Tuple[Optional[str], Optional[float], bool]:
        """返回 (目标或 None, Poe 耗时毫秒或 None, 是否实际调用过 Poe)。"""
        if not text or not text.strip():
            logger.warning("Empty text provided to parse_command_with_vision")
            return None, None, False

        llm_poe_ms: Optional[float] = None
        llm_poe_invoked = False
        if llm_parser is not None and frames:
            try:
                logger.debug(
                    "Attempting LLM vision parsing: text='%s', frames=%s", text, len(frames)
                )
                result, llm_poe_ms, llm_poe_invoked = llm_parser.parse_with_vision(text, frames)

                if result and "target" in result:
                    target = result["target"].strip()
                    if target:
                        logger.info(
                            "LLM vision parsing succeeded: '%s' -> '%s'", text, target
                        )
                        return target, llm_poe_ms, llm_poe_invoked
                else:
                    logger.debug("LLM vision parsing returned empty result")

            except Exception as e:
                logger.error(
                    "LLM vision parsing exception: %s. Falling back to regex parsing.",
                    e,
                    exc_info=True,
                )
                llm_poe_ms = None
                llm_poe_invoked = False
        else:
            if llm_parser is None:
                logger.debug("No LLM parser provided, using regex parsing")
            if not frames:
                logger.debug("No frames provided, using regex parsing")

        logger.debug("Falling back to regex parse_command: '%s'", text)
        target = self.parse_command(text)
        if target:
            logger.info("Regex parsing result: '%s' -> '%s'", text, target)
            return target, llm_poe_ms if llm_poe_invoked else None, llm_poe_invoked

        logger.warning("Both LLM and regex parsing failed for: %s", text)
        return None, llm_poe_ms if llm_poe_invoked else None, llm_poe_invoked
