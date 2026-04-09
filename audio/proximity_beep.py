"""
近距离滴滴声（倒车雷达式）
默认使用 sounddevice；连续模式下由后台线程循环播放密集脉冲串，避免「响一声停很久」。
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore
    SOUNDDEVICE_AVAILABLE = False

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore


class ProximityBeepPlayer:
    """非阻塞滴滴声；连续模式依赖 sounddevice 的阻塞播放仅在后台线程中执行。"""

    def __init__(self):
        self._last_beep_ts: float = 0.0
        self._pygame_mixer_ok: bool = False
        self._warned_play_fail: bool = False
        self._warned_continuous: bool = False

        self._lock = threading.Lock()
        self._cont_eligible: bool = False
        self._cont_snap: dict = {}
        self._stream_stop = False
        self._stream_thread: Optional[threading.Thread] = None

    def set_continuous_eligible(self, v: bool) -> None:
        with self._lock:
            self._cont_eligible = bool(v)
            if not v:
                self._cont_snap = {}
        if not v and SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass

    def reset_cooldown(self) -> None:
        self._last_beep_ts = 0.0
        self.set_continuous_eligible(False)

    def _ensure_stream_thread(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stream_stop = False

        def _loop() -> None:
            while not self._stream_stop:
                with self._lock:
                    elig = self._cont_eligible
                    snap = dict(self._cont_snap) if elig else {}
                if not elig or not snap:
                    time.sleep(0.03)
                    continue
                if not SOUNDDEVICE_AVAILABLE:
                    time.sleep(0.1)
                    continue
                try:
                    t = float(snap.get("t", 0.5))
                    g0 = float(snap.get("gap_min_ms", 0.0))
                    g1 = float(snap.get("gap_max_ms", 20.0))
                    gap_ms = g0 + t * max(0.0, g1 - g0)
                    wave = self._build_pulse_train(
                        pulse_ms=int(snap.get("duration_ms", 50)),
                        gap_ms=gap_ms,
                        n_pulses=int(snap.get("chunk_pulses", 32)),
                        frequency_hz=int(snap.get("frequency_hz", 900)),
                        sample_rate=int(snap.get("sample_rate", 44100)),
                        volume=float(snap.get("volume", 0.55)),
                    )
                    sr = int(snap["sample_rate"])
                    sd.play(wave, sr, blocking=True)
                except Exception as e:
                    logger.debug("proximity_beep 连续播放段失败: %s", e)
                    time.sleep(0.05)

        self._stream_thread = threading.Thread(target=_loop, daemon=True, name="proximity-beep")
        self._stream_thread.start()

    @staticmethod
    def _build_pulse_train(
        pulse_ms: int,
        gap_ms: float,
        n_pulses: int,
        frequency_hz: int,
        sample_rate: int,
        volume: float,
    ) -> np.ndarray:
        pulse_ms = max(1, int(pulse_ms))
        n_p = max(1, int(sample_rate * pulse_ms / 1000))
        n_g = max(0, int(sample_rate * max(0.0, gap_ms) / 1000))
        tp = np.linspace(0.0, pulse_ms / 1000.0, n_p, endpoint=False, dtype=np.float32)
        pulse = (np.sin(2.0 * np.pi * float(frequency_hz) * tp) * float(volume)).astype(
            np.float32
        )
        # 短淡入淡出，减轻爆音、在部分声卡上更易听清
        edge = min(n_p // 8, max(1, int(sample_rate * 0.002)))
        if edge * 2 < n_p:
            ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            pulse[:edge] *= ramp
            pulse[-edge:] *= ramp[::-1]
        gap = np.zeros(n_g, dtype=np.float32)
        n_pulses = max(1, int(n_pulses))
        parts = []
        for _ in range(n_pulses):
            parts.append(pulse)
            parts.append(gap)
        if not parts:
            return pulse
        return np.concatenate(parts)

    def update_continuous(
        self,
        *,
        now: float,
        density_t: float,
        start_delay_sec: float,
        task_start_ts: float,
        frequency_hz: int,
        duration_ms: int,
        gap_min_ms: float,
        gap_max_ms: float,
        chunk_pulses: int,
        sample_rate: int,
        backend: str,
        volume: float = 0.55,
    ) -> None:
        b = (backend or "sounddevice").strip().lower()
        if b == "pygame" and not self._warned_continuous:
            logger.warning(
                "proximity_beep_continuous 当前仅支持 sounddevice；"
                "backend=pygame 时已关闭连续滴滴，请改用 sounddevice 或关闭 proximity_beep_continuous。"
            )
            self._warned_continuous = True
            self.set_continuous_eligible(False)
            return
        if not SOUNDDEVICE_AVAILABLE:
            if not self._warned_continuous:
                logger.warning("proximity_beep 连续模式需要 sounddevice，已跳过。")
                self._warned_continuous = True
            self.set_continuous_eligible(False)
            return

        if now - float(task_start_ts) < float(start_delay_sec):
            self.set_continuous_eligible(False)
            return

        t = max(0.0, min(1.0, float(density_t)))

        self._ensure_stream_thread()
        with self._lock:
            self._cont_eligible = True
            self._cont_snap = {
                "t": t,
                "gap_min_ms": float(gap_min_ms),
                "gap_max_ms": float(gap_max_ms),
                "chunk_pulses": int(chunk_pulses),
                "frequency_hz": int(frequency_hz),
                "duration_ms": int(duration_ms),
                "sample_rate": int(sample_rate),
                "volume": max(0.05, min(1.0, float(volume))),
            }

    def _ensure_pygame_mixer(self, sample_rate: int = 22050) -> bool:
        if pygame is None:
            return False
        if self._pygame_mixer_ok:
            return True
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
            self._pygame_mixer_ok = True
            return True
        except Exception as e:
            logger.warning("proximity_beep: pygame mixer 初始化失败: %s", e)
            return False

    def _play_sounddevice(
        self,
        frequency_hz: int,
        duration_ms: int,
        sample_rate: int,
        volume: float = 0.35,
    ) -> bool:
        if not SOUNDDEVICE_AVAILABLE:
            return False
        try:
            n = max(1, int(sample_rate * duration_ms / 1000))
            t = np.linspace(0.0, duration_ms / 1000.0, n, dtype=np.float32, endpoint=False)
            wave = (np.sin(2.0 * np.pi * frequency_hz * t) * float(volume)).astype(np.float32)
            sd.play(wave, int(sample_rate), blocking=False)
            return True
        except Exception as e:
            if not self._warned_play_fail:
                logger.warning("proximity_beep: sounddevice 播放失败（将尝试 pygame）: %s", e)
                self._warned_play_fail = True
            return False

    def _play_pygame(
        self,
        frequency_hz: int,
        duration_ms: int,
        sample_rate: int = 22050,
        volume: float = 0.35,
    ) -> bool:
        if not self._ensure_pygame_mixer(sample_rate):
            return False
        try:
            n = max(1, int(sample_rate * duration_ms / 1000))
            t = np.arange(n, dtype=np.float32) / float(sample_rate)
            wave = (np.sin(2.0 * np.pi * frequency_hz * t) * volume * 32767.0).astype(np.int16)
            arr = np.ascontiguousarray(wave.reshape(-1, 1))
            snd = pygame.sndarray.make_sound(arr)
            ch = pygame.mixer.find_channel(True)
            if ch is None:
                return False
            ch.play(snd)
            return True
        except Exception as e:
            logger.debug("proximity_beep: pygame 播放失败: %s", e)
            return False

    def play_beep(
        self,
        frequency_hz: int,
        duration_ms: int,
        *,
        backend: str = "sounddevice",
        sample_rate: int = 44100,
        volume: float = 0.35,
    ) -> bool:
        b = (backend or "sounddevice").strip().lower()
        if b == "auto":
            if self._play_sounddevice(frequency_hz, duration_ms, sample_rate, volume):
                return True
            return self._play_pygame(frequency_hz, duration_ms, 22050, volume)
        if b == "pygame":
            return self._play_pygame(frequency_hz, duration_ms, 22050, volume)
        if self._play_sounddevice(frequency_hz, duration_ms, sample_rate, volume):
            return True
        if b != "sounddevice":
            return self._play_pygame(frequency_hz, duration_ms, 22050, volume)
        return self._play_pygame(frequency_hz, duration_ms, 22050, volume)

    def maybe_beep(
        self,
        now: float,
        distance_px: float,
        *,
        enabled: bool,
        start_delay_sec: float,
        task_start_ts: float,
        min_interval: float,
        max_interval: float,
        far_px: float,
        near_px: float,
        frequency_hz: int,
        duration_ms: int,
        backend: str = "sounddevice",
        sample_rate: int = 44100,
    ) -> None:
        if not enabled:
            return
        if now - float(task_start_ts) < start_delay_sec:
            return
        far_px = max(1.0, float(far_px))
        near_px = max(0.0, min(float(near_px), far_px - 1.0))
        d = float(distance_px)
        span = far_px - near_px
        if span <= 0:
            t = 0.0
        else:
            t = (d - near_px) / span
        t = max(0.0, min(1.0, t))
        interval = float(min_interval) + t * (float(max_interval) - float(min_interval))
        interval = max(float(min_interval), interval)
        if now - self._last_beep_ts < interval:
            return
        if self.play_beep(
            frequency_hz,
            duration_ms,
            backend=backend,
            sample_rate=int(sample_rate),
        ):
            self._last_beep_ts = now
