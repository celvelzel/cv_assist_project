"""
音频工具模块
提供录音、音频处理等功能
"""

import logging
import numpy as np
import wave
import threading
import time
from typing import Optional
import tempfile
import os

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice 未安装，录音功能将不可用")


class AudioRecorder:
    """
    音频录制器
    
    用于录制用户的语音输入。
    支持按键录制和语音活动检测(VAD)。
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 dtype: str = 'float32'):
        """
        初始化音频录制器
        
        参数:
            sample_rate: 采样率 (Hz，Whisper 推荐 16000)
            channels: 声道数 (1=单声道, 2=立体声)
            dtype: 数据类型 ('int16' 或 'float32')
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice 未安装。请运行: pip install sounddevice")
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        self.is_recording = False
        self.recorded_frames = []
        
        logger.info(f"音频录制器初始化: {sample_rate}Hz, {channels}ch, {dtype}")
    
    def record(self, duration: float = 5.0) -> np.ndarray:
        """
        录制指定时长的音频
        
        参数:
            duration: 录制时长 (秒)
            
        返回:
            音频数据 (NumPy 数组)
        """
        logger.info(f"开始录音: {duration} 秒")
        
        try:
            # 录制音频
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )
            sd.wait()  # 等待录制完成
            
            # 如果是单声道，压缩维度
            if self.channels == 1 and audio.ndim > 1:
                audio = audio.squeeze()
            
            logger.info(f"录音完成: {len(audio)} samples")
            return audio
            
        except Exception as e:
            logger.error(f"录音失败: {e}")
            return np.array([], dtype=self.dtype)
    
    def start_recording(self):
        """
        开始录音 (非阻塞，后台持续录制)
        """
        if self.is_recording:
            logger.warning("已在录音中")
            return
        
        self.is_recording = True
        self.recorded_frames = []
        
        logger.info("开始持续录音...")
        
        # 定义音频回调函数
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"录音状态: {status}")
            
            if self.is_recording:
                self.recorded_frames.append(indata.copy())
        
        # 启动音频流
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=callback
            )
            self.stream.start()
            logger.info("持续录音已启动")
            
        except Exception as e:
            logger.error(f"启动录音失败: {e}")
            self.is_recording = False
    
    def stop_recording(self) -> np.ndarray:
        """
        停止录音并返回录制的音频
        
        返回:
            音频数据 (NumPy 数组)
        """
        if not self.is_recording:
            logger.warning("当前未在录音")
            return np.array([], dtype=self.dtype)
        
        logger.info("停止录音...")
        
        self.is_recording = False
        
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            # 合并所有录制的帧
            if self.recorded_frames:
                audio = np.concatenate(self.recorded_frames, axis=0)
                
                # 如果是单声道，压缩维度
                if self.channels == 1 and audio.ndim > 1:
                    audio = audio.squeeze()
                
                logger.info(f"录音已停止: {len(audio)} samples")
                return audio
            else:
                logger.warning("未录制到音频")
                return np.array([], dtype=self.dtype)
            
        except Exception as e:
            logger.error(f"停止录音失败: {e}")
            return np.array([], dtype=self.dtype)
    
    def record_until_silence(self,
                            max_duration: float = 10.0,
                            silence_threshold: float = 0.01,
                            silence_duration: float = 1.5) -> np.ndarray:
        """
        录音直到检测到静音
        
        参数:
            max_duration: 最大录制时长 (秒)
            silence_threshold: 静音阈值 (音量低于此值视为静音)
            silence_duration: 静音持续时长 (秒，超过此时长则停止录制)
            
        返回:
            音频数据 (NumPy 数组)
        """
        logger.info(f"开始录音 (自动检测静音停止，最长 {max_duration}s)")
        
        frames = []
        silent_frames = 0
        max_silent_frames = int(silence_duration * self.sample_rate / 1024)  # 1024 是块大小
        # 避免「一开头全是静音」就立刻结束：须先检测到说话，之后的连续静音才视为说完
        has_speech = False
        
        start_time = time.time()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=1024
            ) as stream:
                
                while True:
                    # 读取音频块
                    audio_chunk, overflowed = stream.read(1024)
                    
                    if overflowed:
                        logger.warning("音频缓冲区溢出")
                    
                    frames.append(audio_chunk.copy())
                    
                    # 计算音量 (RMS)
                    volume = np.sqrt(np.mean(audio_chunk**2))
                    
                    # 检测静音
                    if volume < silence_threshold:
                        silent_frames += 1
                    else:
                        silent_frames = 0
                        has_speech = True
                    
                    # 检查停止条件
                    elapsed = time.time() - start_time
                    
                    if silent_frames >= max_silent_frames and has_speech:
                        logger.info(f"检测到静音，停止录音 ({elapsed:.1f}s)")
                        break
                    
                    if elapsed >= max_duration:
                        logger.info(f"达到最大时长，停止录音 ({max_duration}s)")
                        break
            
            # 合并所有帧
            if frames:
                audio = np.concatenate(frames, axis=0)
                
                if self.channels == 1 and audio.ndim > 1:
                    audio = audio.squeeze()
                
                logger.info(f"录音完成: {len(audio)} samples, {len(audio)/self.sample_rate:.2f}s")
                return audio
            else:
                return np.array([], dtype=self.dtype)
            
        except Exception as e:
            logger.error(f"录音失败: {e}")
            return np.array([], dtype=self.dtype)
    
    def save_audio(self, audio: np.ndarray, filepath: str):
        """
        保存音频到 WAV 文件
        
        参数:
            audio: 音频数据 (NumPy 数组)
            filepath: 保存路径
        """
        try:
            # 转换为 int16 格式
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)
            
            # 保存为 WAV 文件
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"音频已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """
        从 WAV 文件加载音频
        
        参数:
            filepath: 文件路径
            
        返回:
            音频数据 (NumPy 数组, float32 格式)
        """
        try:
            with wave.open(filepath, 'rb') as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                
                # 转换为 NumPy 数组
                audio = np.frombuffer(frames, dtype=np.int16)
                
                # 转换为 float32 格式 (-1.0 到 1.0)
                audio = audio.astype(np.float32) / 32767.0
                
                # 处理多声道
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                    if channels > self.channels:
                        # 转换为单声道 (取平均)
                        audio = audio.mean(axis=1)
                
                logger.info(f"音频已加载: {filepath} ({sample_rate}Hz, {len(audio)} samples)")
                return audio
                
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            return np.array([], dtype=self.dtype)
    
    @staticmethod
    def list_devices():
        """
        列出所有可用的音频设备
        
        返回:
            设备列表
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice 不可用")
            return []
        
        devices = sd.query_devices()
        logger.info("可用音频设备:")
        
        for idx, device in enumerate(devices):
            logger.info(f"  [{idx}] {device['name']} "
                       f"(输入:{device['max_input_channels']}, "
                       f"输出:{device['max_output_channels']})")
        
        return devices
    
    @staticmethod
    def set_default_device(device_id: int):
        """
        设置默认音频设备
        
        参数:
            device_id: 设备 ID
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice 不可用")
            return
        
        sd.default.device = device_id
        logger.info(f"默认音频设备已设置为: {device_id}")


# 便捷函数
def quick_record(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """
    快速录制音频
    
    参数:
        duration: 录制时长 (秒)
        sample_rate: 采样率
        
    返回:
        音频数据 (NumPy 数组)
    """
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice 不可用，无法录音")
        return np.array([])
    
    try:
        recorder = AudioRecorder(sample_rate=sample_rate)
        return recorder.record(duration)
    except Exception as e:
        logger.error(f"快速录音失败: {e}")
        return np.array([])
