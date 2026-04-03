"""
MiMo TTS 后端（云端方案）
基于小米 MiMo-V2-TTS 模型的语音合成服务。
通过 OpenAI 兼容 API 调用云端模型，返回音频后本地播放。
"""

import base64
import logging
import os
import queue
import tempfile
import threading
import time

from .base import BaseTTS

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# MiMo TTS 预置音色
MIMO_VOICES = {
    'mimo_default': 'MiMo-默认',
    'default_zh': 'MiMo-中文女声',
    'default_en': 'MiMo-英文女声',
}

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MiMoTTS(BaseTTS):
    """
    基于小米 MiMo-V2-TTS 的云端语音合成引擎

    通过 OpenAI 兼容格式调用 MiMo TTS API，将返回的 WAV 音频本地播放。
    支持异步播放，不会阻塞主程序。

    特点:
    - 高质量语音合成，支持情感表达
    - 支持风格控制标签（<style>开心</style>）
    - 支持细粒度控制（语速、情绪、角色扮演、方言等）
    - 需要网络连接和 MiMo API Key
    """

    def __init__(self,
                 api_key: str = "",
                 voice: str = "mimo_default",
                 volume: float = 1.0,
                 async_mode: bool = True,
                 max_queue_size: int = 1,
                 drop_stale: bool = True,
                 base_url: str = MIMO_BASE_URL):
        """
        初始化 MiMo TTS 引擎

        参数:
            api_key: MiMo API Key（也可通过环境变量 MIMO_API_KEY 或 XIAOMI_MIMO_API_KEY 设置）
            voice: 音色名称 (mimo_default / default_zh / default_en)
            volume: 音量 (0.0-1.0, 通过 pygame 音量控制)
            async_mode: 是否异步播放 (True=不阻塞主线程)
            max_queue_size: 异步队列大小
            drop_stale: 队列满时是否丢弃旧消息保留新消息
            base_url: MiMo API 基础 URL
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai 库未安装。MiMo TTS 需要 openai 库，请运行: pip install openai"
            )

        if not PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame 库未安装。MiMo TTS 音频播放需要 pygame 库，请运行: pip install pygame"
            )

        # 解析 API Key：参数 > 环境变量
        self.api_key = api_key or os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError(
                "MiMo TTS 需要 API Key。请通过以下方式之一提供：\n"
                "  1. 构造时传入 api_key 参数\n"
                "  2. 设置环境变量 MIMO_API_KEY 或 XIAOMI_MIMO_API_KEY\n"
                "  3. 在 config.py 的 AudioConfig 中设置 mimo_api_key"
            )

        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))
        self.base_url = base_url
        self.rate = 150  # MiMo TTS 通过 <style> 标签控制语速，此处仅记录值
        self.async_mode = async_mode
        self.max_queue_size = max(1, int(max_queue_size))
        self.drop_stale = drop_stale
        self._stop_token = object()

        # 初始化 pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 异步队列 + 工作线程
        if self.async_mode:
            self.speech_queue = queue.Queue(maxsize=self.max_queue_size)
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

        logger.info(
            f"MiMo TTS 引擎初始化成功 "
            f"(voice={self.voice}, async={self.async_mode}, queue_size={self.max_queue_size})"
        )

    def _synthesize(self, text: str) -> bytes:
        """
        调用 MiMo TTS API 合成语音

        参数:
            text: 要合成的文本

        返回:
            WAV 音频字节数据
        """
        logger.debug(f"MiMo TTS 合成: '{text}'")

        completion = self.client.chat.completions.create(
            model="mimo-v2-tts",
            messages=[
                {
                    "role": "user",
                    "content": ""
                },
                {
                    "role": "assistant",
                    "content": text
                }
            ],
            audio={
                "format": "wav",
                "voice": self.voice
            }
        )

        message = completion.choices[0].message
        audio_data = base64.b64decode(message.audio.data)
        return audio_data

    def _play_audio(self, audio_bytes: bytes):
        """
        播放 WAV 音频字节数据

        使用 pygame.mixer 播放，通过临时文件实现。
        """
        tmp_path = None
        try:
            # 写入临时文件
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)

            # pygame 播放
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

        except Exception as e:
            logger.error(f"音频播放失败: {e}", exc_info=True)
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _worker(self):
        """异步播放工作线程"""
        while True:
            try:
                payload = self.speech_queue.get()
                if payload is self._stop_token:
                    self.speech_queue.task_done()
                    break

                text, done_event = payload
                try:
                    logger.debug(f"MiMo TTS 异步播放: '{text}'")
                    audio_bytes = self._synthesize(text)
                    self._play_audio(audio_bytes)
                except Exception as e:
                    logger.error(f"MiMo TTS 合成/播放失败: {e}", exc_info=True)
                finally:
                    if done_event is not None:
                        done_event.set()
                    self.speech_queue.task_done()

            except Exception as e:
                logger.error(f"MiMo TTS 工作线程错误: {e}", exc_info=True)

    def speak(self, text: str, block: bool = False):
        """播放文本"""
        if not text or not text.strip():
            logger.warning("空文本，跳过 MiMo TTS 播放")
            return

        text = text.strip()
        logger.info(f"MiMo TTS 请求: '{text}'")

        try:
            if self.async_mode and not block:
                self._enqueue_async(text)
            elif self.async_mode and block:
                self._enqueue_async(text, wait=True)
            else:
                # 同步模式
                audio_bytes = self._synthesize(text)
                self._play_audio(audio_bytes)

        except Exception as e:
            logger.error(f"MiMo TTS 播放失败: {e}", exc_info=True)

    def _enqueue_async(self, text: str, wait: bool = False):
        """将文本加入异步队列，支持队列满时去旧保新。"""
        done_event = threading.Event() if wait else None
        payload = (text, done_event)
        queued = False

        try:
            self.speech_queue.put_nowait(payload)
            queued = True
        except queue.Full:
            if not self.drop_stale:
                logger.debug("MiMo TTS 队列已满，保留旧消息，丢弃新消息")
                return

        if not queued:
            self.clear_queue()
            try:
                self.speech_queue.put_nowait(payload)
                queued = True
            except queue.Full:
                logger.debug("MiMo TTS 队列仍然繁忙，丢弃当前消息")
                return

        if wait and done_event is not None:
            if not done_event.wait(timeout=15.0):
                logger.warning("MiMo TTS 同步等待超时，文本可能未及时播报")

    def speak_instruction(self, instruction: str):
        """播放引导指令"""
        self.speak(instruction)

    def speak_lifecycle(self, text: str):
        """播报任务生命周期提示，直接合成并独占播放通道，不走队列，不可被打断。"""
        if not text or not text.strip():
            return
        text = text.strip()
        logger.info(f"MiMo TTS 生命周期播报: '{text}'")
        try:
            # 停止当前播放并清空队列，确保本条优先
            self.stop()
            self.clear_queue()
            # 直接在调用线程内合成并播放，绕过 speech_queue
            audio_bytes = self._synthesize(text)
            self._play_audio(audio_bytes)
        except Exception as e:
            logger.error(f"MiMo TTS 生命周期播报失败: {e}", exc_info=True)

    def stop(self):
        """停止当前播放"""
        try:
            if PYGAME_AVAILABLE and pygame.mixer.get_init():
                pygame.mixer.music.stop()
            logger.debug("MiMo TTS 停止播放")
        except Exception as e:
            logger.error(f"MiMo TTS 停止失败: {e}")

    def clear_queue(self):
        """清空播放队列"""
        if getattr(self, 'async_mode', False) and hasattr(self, 'speech_queue'):
            while not self.speech_queue.empty():
                try:
                    payload = self.speech_queue.get_nowait()
                    if isinstance(payload, tuple) and len(payload) == 2:
                        _, done_event = payload
                        if done_event is not None:
                            done_event.set()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
            logger.debug("MiMo TTS 队列已清空")

    def set_rate(self, rate: int):
        """设置语速（MiMo TTS 通过 style 标签控制语速，此处仅记录值）"""
        self.rate = rate
        logger.info(f"MiMo TTS 语速记录为: {rate}（实际语速通过 <style> 标签控制）")

    def set_volume(self, volume: float):
        """设置音量"""
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"MiMo TTS 音量设置为: {self.volume}")

    def list_voices(self):
        """列出可用音色"""
        voice_list = []
        for idx, (voice_id, voice_name) in enumerate(MIMO_VOICES.items()):
            voice_list.append({
                'index': idx,
                'id': voice_id,
                'name': voice_name,
                'languages': ['zh', 'en']
            })
            logger.info(f"  [{idx}] {voice_name} - {voice_id}")
        return voice_list

    def close(self):
        """关闭 MiMo TTS 引擎"""
        try:
            if getattr(self, 'async_mode', False) and hasattr(self, 'speech_queue'):
                self.clear_queue()
                self.speech_queue.put(self._stop_token)
                if hasattr(self, 'worker_thread'):
                    self.worker_thread.join(timeout=3.0)

            if PYGAME_AVAILABLE and pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()

            logger.info("MiMo TTS 引擎已关闭")
        except Exception as e:
            logger.error(f"关闭 MiMo TTS 引擎失败: {e}")

    def get_debug_info(self) -> dict:
        """返回当前 TTS 关键状态"""
        info = {
            'provider': 'mimo',
            'async_mode': self.async_mode,
            'voice': self.voice,
            'volume': self.volume,
            'base_url': self.base_url,
            'api_key_set': bool(self.api_key),
            'pygame_available': PYGAME_AVAILABLE,
            'openai_available': OPENAI_AVAILABLE,
        }

        if hasattr(self, 'speech_queue'):
            info['queue_size'] = self.speech_queue.qsize()
            info['queue_max_size'] = self.max_queue_size
        if hasattr(self, 'worker_thread'):
            info['worker_alive'] = self.worker_thread.is_alive()

        logger.info(f"MiMo TTS 调试信息: {info}")
        return info

    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass
