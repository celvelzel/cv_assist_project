"""
TTS 模块 — 文本转语音服务

支持多种 TTS 后端：
- pyttsx3: 离线方案，基于 pyttsx3 / Windows SAPI
- mimo: 云端方案，基于小米 MiMo-V2-TTS 模型

通过 create_tts() 工厂函数自动选择后端。
"""

import logging

from .base import BaseTTS
from .pyttsx3_backend import Pyttsx3TTS, TTSEngine
from .mimo_backend import MiMoTTS

logger = logging.getLogger(__name__)


def create_tts(config) -> BaseTTS:
    """
    TTS 工厂函数 — 根据配置创建对应的 TTS 后端实例

    参数:
        config: SystemConfig 对象，读取 config.audio 中的配置项

    返回:
        BaseTTS 实例

    支持的 provider:
        - "pyttsx3": 离线 pyttsx3 引擎
        - "mimo": 小米 MiMo-V2-TTS 云端引擎
    """
    cfg = config.audio
    provider = getattr(cfg, 'tts_provider', 'pyttsx3').lower()

    logger.info(f"创建 TTS 引擎: provider={provider}")

    if provider == 'mimo':
        return MiMoTTS(
            api_key=getattr(cfg, 'mimo_api_key', ''),
            voice=getattr(cfg, 'mimo_voice', 'mimo_default'),
            volume=cfg.tts_volume,
            async_mode=cfg.tts_async,
            max_queue_size=cfg.tts_max_queue_size,
            drop_stale=cfg.tts_drop_stale,
            lifecycle_wait_playback_idle_sec=getattr(
                cfg, 'mimo_lifecycle_wait_playback_idle_sec', 4.0
            ),
        )
    elif provider == 'pyttsx3':
        return Pyttsx3TTS(
            rate=cfg.tts_rate,
            volume=cfg.tts_volume,
            async_mode=cfg.tts_async,
            max_queue_size=cfg.tts_max_queue_size,
            drop_stale=cfg.tts_drop_stale,
        )
    else:
        raise ValueError(
            f"不支持的 TTS provider: '{provider}'，"
            f"请使用 'pyttsx3' 或 'mimo'"
        )


def quick_speak(text: str, rate: int = 150):
    """
    快速播放文本（离线，创建临时 pyttsx3 引擎）

    参数:
        text: 要播放的文本
        rate: 语速
    """
    from .pyttsx3_backend import PYTTSX3_AVAILABLE
    if not PYTTSX3_AVAILABLE:
        logger.warning("pyttsx3 不可用，无法播放语音")
        return

    try:
        engine = Pyttsx3TTS(rate=rate, async_mode=False)
        engine.speak(text, block=True)
        engine.close()
    except Exception as e:
        logger.error(f"快速播放失败: {e}")


__all__ = [
    'BaseTTS',
    'Pyttsx3TTS',
    'TTSEngine',
    'MiMoTTS',
    'create_tts',
    'quick_speak',
]
