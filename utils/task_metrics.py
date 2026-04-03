"""
任务级监测与报告输出工具。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

_HKT = timezone(timedelta(hours=8))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameMetrics:
    """单帧任务事实。"""

    frame_index: int
    frame_start_ts: float
    frame_end_ts: float
    capture_time_ms: float
    process_time_ms: float
    draw_time_ms: float
    display_time_ms: float
    e2e_loop_time_ms: float
    detection_time_ms: float
    hand_time_ms: float
    depth_time_ms: float
    guidance_time_ms: float
    detection_executed: bool
    depth_executed: bool
    detections_count: int
    hands_count: int
    has_target: bool
    has_hand: bool
    has_guidance: bool
    guidance_state: str
    ready_to_grab: bool
    stable_ready_frames: int
    gesture: str
    target_visible: bool
    proc_fps_current: float
    proc_fps_avg: float
    e2e_fps_current: float
    e2e_fps_avg: float


@dataclass(frozen=True)
class TaskReportEnvelope:
    """主线程传递给后台写入线程的冻结快照。"""

    task_id: str
    output_path: str
    report_dict: Dict
    created_at: float
    retry_count: int = 0


class TaskMetricsCollector:
    """聚合单任务执行指标并负责结束判定。"""

    def __init__(
        self,
        grasp_stable_frames: int,
        ready_confirm_window_sec: float,
        lost_target_window_sec: float,
    ):
        self.grasp_stable_frames = max(1, int(grasp_stable_frames))
        self.ready_confirm_window_sec = max(0.5, float(ready_confirm_window_sec))
        self.lost_target_window_sec = max(0.5, float(lost_target_window_sec))
        self.reset()

    def reset(self):
        self.active = False
        self.task_id = None
        self.target_query = None
        self.session_id = None
        self.start_time = None
        self.end_time = None
        self.task_state = "idle"
        self.pending_end_reason = None
        self.error_message = ""
        self.voice_total_time_ms = 0.0
        self.voice_asr_time_ms = 0.0
        self.voice_raw_text = ""
        self.total_frames = 0
        self.capture_time_sum_ms = 0.0
        self.process_time_sum_ms = 0.0
        self.draw_time_sum_ms = 0.0
        self.display_time_sum_ms = 0.0
        self.e2e_loop_time_sum_ms = 0.0
        self.detection_time_sum_ms = 0.0
        self.hand_time_sum_ms = 0.0
        self.depth_time_sum_ms = 0.0
        self.guidance_time_sum_ms = 0.0
        self.target_visible_frames = 0
        self.guidance_frames = 0
        self.ready_frames = 0
        self.detection_executed_frames = 0
        self.depth_executed_frames = 0
        self.latest_frame_metrics = None
        self.ready_streak = 0
        self.ready_enter_ts = None
        self.ready_window_deadline_ts = None
        self.closed_after_ready_flag = False
        self.lost_target_since_ts = None  # 目标首次消失的时间戳，None 表示目标可见
        self.first_target_detected_ts = None
        self.first_guidance_ts = None
        self.first_ready_ts = None
        self.first_grabbed_ts = None

    def start_task(self, task_id: str, target_query: str, start_time: float, session_id: str):
        self.reset()
        self.active = True
        self.task_id = task_id
        self.target_query = target_query
        self.session_id = session_id
        self.start_time = start_time
        self.task_state = "running"

    def record_voice_metrics(
        self,
        voice_total_time_ms: float,
        voice_asr_time_ms: float,
        raw_text: str,
    ):
        if not self.active:
            return
        self.voice_total_time_ms = max(0.0, float(voice_total_time_ms or 0.0))
        self.voice_asr_time_ms = max(0.0, float(voice_asr_time_ms or 0.0))
        self.voice_raw_text = (raw_text or "").strip()

    def record_frame(self, frame_metrics: FrameMetrics):
        if not self.active:
            return

        self.latest_frame_metrics = frame_metrics
        self.total_frames += 1
        self.capture_time_sum_ms += frame_metrics.capture_time_ms
        self.process_time_sum_ms += frame_metrics.process_time_ms
        self.draw_time_sum_ms += frame_metrics.draw_time_ms
        self.display_time_sum_ms += frame_metrics.display_time_ms
        self.e2e_loop_time_sum_ms += frame_metrics.e2e_loop_time_ms
        self.detection_time_sum_ms += frame_metrics.detection_time_ms
        self.hand_time_sum_ms += frame_metrics.hand_time_ms
        self.depth_time_sum_ms += frame_metrics.depth_time_ms
        self.guidance_time_sum_ms += frame_metrics.guidance_time_ms

        if frame_metrics.detection_executed:
            self.detection_executed_frames += 1
        if frame_metrics.depth_executed:
            self.depth_executed_frames += 1
        if frame_metrics.target_visible:
            self.target_visible_frames += 1
        if frame_metrics.has_guidance:
            self.guidance_frames += 1
        if frame_metrics.ready_to_grab:
            self.ready_frames += 1

        if frame_metrics.target_visible and self.first_target_detected_ts is None:
            self.first_target_detected_ts = frame_metrics.frame_end_ts
        if frame_metrics.has_guidance and self.first_guidance_ts is None:
            self.first_guidance_ts = frame_metrics.frame_end_ts
        if frame_metrics.ready_to_grab and self.first_ready_ts is None:
            self.first_ready_ts = frame_metrics.frame_end_ts
        if (
            frame_metrics.guidance_state == "grabbed"
            or frame_metrics.gesture == "closed"
        ) and self.first_grabbed_ts is None:
            self.first_grabbed_ts = frame_metrics.frame_end_ts

        self._update_completion_state(frame_metrics)

    def _update_completion_state(self, frame_metrics: FrameMetrics):
        if frame_metrics.target_visible:
            self.lost_target_since_ts = None
        else:
            if self.lost_target_since_ts is None:
                self.lost_target_since_ts = frame_metrics.frame_end_ts
            elif (
                frame_metrics.frame_end_ts - self.lost_target_since_ts
                >= self.lost_target_window_sec
            ):
                self.pending_end_reason = self.pending_end_reason or "lost_target"

        if frame_metrics.ready_to_grab:
            self.ready_streak += 1
            if (
                self.ready_enter_ts is None
                and frame_metrics.stable_ready_frames >= self.grasp_stable_frames
            ):
                self.ready_enter_ts = frame_metrics.frame_end_ts
                self.ready_window_deadline_ts = (
                    self.ready_enter_ts + self.ready_confirm_window_sec
                )
                self.task_state = "ready"
        else:
            self.ready_streak = 0

        if (
            self.ready_enter_ts is not None
            and self.ready_window_deadline_ts is not None
            and frame_metrics.frame_end_ts > self.ready_window_deadline_ts
            and not self.closed_after_ready_flag
        ):
            self.ready_enter_ts = None
            self.ready_window_deadline_ts = None
            self.ready_streak = 0
            self.task_state = "running"

        if self.ready_window_deadline_ts is None:
            return

        in_window = frame_metrics.frame_end_ts <= self.ready_window_deadline_ts
        if not in_window:
            return

        # 仅在当前帧手部仍处于对准就绪状态时，才允许 closed/grabbed 触发成功
        hand_still_ready = frame_metrics.ready_to_grab
        if hand_still_ready and (
            frame_metrics.gesture == "closed" or frame_metrics.guidance_state == "grabbed"
        ):
            self.closed_after_ready_flag = True
            self.pending_end_reason = self.pending_end_reason or "success"
            self.task_state = "finishing"

    def should_finish_task(self) -> Optional[str]:
        return self.pending_end_reason

    def should_emit_report(self) -> bool:
        """只有任务期间至少检测到过一次目标时才生成报告。"""
        return self.first_target_detected_ts is not None and self.target_visible_frames > 0

    def finish_task(
        self,
        end_reason: str,
        end_time: float,
        error_message: str = "",
    ) -> Dict:
        if not self.active:
            raise RuntimeError("no active task to finish")

        self.end_time = end_time
        self.task_state = "finished"
        self.error_message = (error_message or "").strip()
        frames = max(1, self.total_frames)
        task_elapsed_sec = max(0.0, self.end_time - self.start_time)
        latest = self.latest_frame_metrics
        finished_state = "finished"

        report = {
            "session_info": {
                "section": self._section_meta("session_info", "会话信息"),
                "session_id": self._field_entry("session_id", "会话ID", self.session_id),
                "generated_at": self._field_entry("generated_at", "生成时间戳", self.end_time),
            },
            "task_info": {
                "section": self._section_meta("task_info", "任务信息"),
                "task_id": self._field_entry("task_id", "任务ID", self.task_id),
                "target_query": self._field_entry("target_query", "目标主体", self.target_query),
                "task_state": self._field_entry(
                    "task_state",
                    "任务状态",
                    finished_state,
                    value_zh=self._translate_task_state(finished_state),
                ),
                "start_time": self._field_entry("start_time", "任务开始时间戳", self.start_time),
                "end_time": self._field_entry("end_time", "任务结束时间戳", self.end_time),
                "task_elapsed_sec": self._field_entry("task_elapsed_sec", "任务耗时秒", task_elapsed_sec),
                "end_reason": self._field_entry(
                    "end_reason",
                    "结束原因",
                    end_reason,
                    value_zh=self._translate_end_reason(end_reason),
                ),
            },
            "runtime_summary": {
                "section": self._section_meta("runtime_summary", "运行摘要"),
                "total_frames": self._field_entry("total_frames", "总帧数", self.total_frames),
                "first_target_detected_ts": self._field_entry("first_target_detected_ts", "首次检测到目标时间戳", self.first_target_detected_ts),
                "first_guidance_ts": self._field_entry("first_guidance_ts", "首次生成引导时间戳", self.first_guidance_ts),
                "first_ready_ts": self._field_entry("first_ready_ts", "首次进入就绪时间戳", self.first_ready_ts),
                "first_grabbed_ts": self._field_entry("first_grabbed_ts", "首次抓取确认时间戳", self.first_grabbed_ts),
            },
            "voice_summary": {
                "section": self._section_meta("voice_summary", "语音摘要"),
                "voice_total_time_ms": self._field_entry("voice_total_time_ms", "语音总耗时毫秒", self.voice_total_time_ms),
                "voice_asr_time_ms": self._field_entry("voice_asr_time_ms", "语音识别耗时毫秒", self.voice_asr_time_ms),
                "raw_text": self._field_entry("raw_text", "语音原文", self.voice_raw_text),
            },
            "latency_summary": {
                "section": self._section_meta("latency_summary", "时延摘要"),
                "capture_time_avg_ms": self._field_entry("capture_time_avg_ms", "平均采集耗时毫秒", self._avg(self.capture_time_sum_ms, frames)),
                "process_time_avg_ms": self._field_entry("process_time_avg_ms", "平均处理耗时毫秒", self._avg(self.process_time_sum_ms, frames)),
                "detection_time_avg_ms": self._field_entry("detection_time_avg_ms", "平均检测耗时毫秒", self._avg(self.detection_time_sum_ms, frames)),
                "hand_time_avg_ms": self._field_entry("hand_time_avg_ms", "平均手部检测耗时毫秒", self._avg(self.hand_time_sum_ms, frames)),
                "depth_time_avg_ms": self._field_entry("depth_time_avg_ms", "平均深度估计耗时毫秒", self._avg(self.depth_time_sum_ms, frames)),
                "guidance_time_avg_ms": self._field_entry("guidance_time_avg_ms", "平均引导计算耗时毫秒", self._avg(self.guidance_time_sum_ms, frames)),
                "draw_time_avg_ms": self._field_entry("draw_time_avg_ms", "平均绘制耗时毫秒", self._avg(self.draw_time_sum_ms, frames)),
                "display_time_avg_ms": self._field_entry("display_time_avg_ms", "平均显示耗时毫秒", self._avg(self.display_time_sum_ms, frames)),
                "e2e_loop_time_avg_ms": self._field_entry("e2e_loop_time_avg_ms", "平均端到端耗时毫秒", self._avg(self.e2e_loop_time_sum_ms, frames)),
            },
            "fps_summary": {
                "section": self._section_meta("fps_summary", "FPS摘要"),
                "proc_fps_current": self._field_entry("proc_fps_current", "当前处理FPS", latest.proc_fps_current if latest else 0.0),
                "proc_fps_avg": self._field_entry("proc_fps_avg", "平均处理FPS", latest.proc_fps_avg if latest else 0.0),
                "e2e_fps_current": self._field_entry("e2e_fps_current", "当前端到端FPS", latest.e2e_fps_current if latest else 0.0),
                "e2e_fps_avg": self._field_entry("e2e_fps_avg", "平均端到端FPS", latest.e2e_fps_avg if latest else 0.0),
            },
            "quality_summary": {
                "section": self._section_meta("quality_summary", "质量摘要"),
                "target_detect_hit_rate": self._field_entry("target_detect_hit_rate", "目标命中率", self._ratio(self.target_visible_frames, frames)),
                "guidance_generate_rate": self._field_entry("guidance_generate_rate", "引导生成率", self._ratio(self.guidance_frames, frames)),
                "ready_rate": self._field_entry("ready_rate", "就绪帧占比", self._ratio(self.ready_frames, frames)),
                "detection_executed_frames": self._field_entry("detection_executed_frames", "执行检测的帧数", self.detection_executed_frames),
                "depth_executed_frames": self._field_entry("depth_executed_frames", "执行深度估计的帧数", self.depth_executed_frames),
                "lost_target_since_ts": self._field_entry("lost_target_since_ts", "目标首次消失时间戳", self.lost_target_since_ts),
            },
            "completion_summary": {
                "section": self._section_meta("completion_summary", "完成判定摘要"),
                "ready_streak": self._field_entry("ready_streak", "连续就绪帧数", self.ready_streak),
                "ready_enter_ts": self._field_entry("ready_enter_ts", "进入就绪窗口时间戳", self.ready_enter_ts),
                "ready_window_deadline_ts": self._field_entry("ready_window_deadline_ts", "就绪确认截止时间戳", self.ready_window_deadline_ts),
                "closed_after_ready_flag": self._field_entry("closed_after_ready_flag", "窗口内是否闭合确认", self.closed_after_ready_flag),
            },
            "error_summary": {
                "section": self._section_meta("error_summary", "错误摘要"),
                "message": self._field_entry("message", "错误信息", self.error_message),
            },
        }

        self.active = False
        return report

    def build_terminal_summary(self, now: float) -> str:
        if not self.active or self.latest_frame_metrics is None:
            return ""

        elapsed = max(0.0, now - self.start_time)
        latest = self.latest_frame_metrics
        frames = max(1, self.total_frames)
        task_state_label = self._translate_task_state(self.task_state)
        lines = [
            "[task_metrics]",
            f"task_id={self.task_id} 任务ID",
            f"target={self.target_query} 当前目标主体",
            f"task_state={self.task_state}({task_state_label}) 任务状态",
            f"task_elapsed_sec={elapsed:.1f} 任务耗时秒",
            "",
            f"voice_total_time_ms={self.voice_total_time_ms:.1f} 语音总耗时毫秒",
            f"voice_asr_time_ms={self.voice_asr_time_ms:.1f} 语音识别耗时毫秒",
            "",
            f"proc_fps_current={latest.proc_fps_current:.1f} 当前处理FPS",
            f"proc_fps_avg={latest.proc_fps_avg:.1f} 平均处理FPS",
            f"e2e_fps_current={latest.e2e_fps_current:.1f} 当前端到端FPS",
            f"e2e_fps_avg={latest.e2e_fps_avg:.1f} 平均端到端FPS",
            "",
            f"capture_time_avg_ms={self._avg(self.capture_time_sum_ms, frames):.1f} 平均采集耗时毫秒",
            f"process_time_avg_ms={self._avg(self.process_time_sum_ms, frames):.1f} 平均处理耗时毫秒",
            f"detection_time_avg_ms={self._avg(self.detection_time_sum_ms, frames):.1f} 平均检测耗时毫秒",
            f"depth_time_avg_ms={self._avg(self.depth_time_sum_ms, frames):.1f} 平均深度估计耗时毫秒",
            f"guidance_time_avg_ms={self._avg(self.guidance_time_sum_ms, frames):.1f} 平均引导计算耗时毫秒",
            f"draw_time_avg_ms={self._avg(self.draw_time_sum_ms, frames):.1f} 平均绘制耗时毫秒",
            "",
            f"target_detect_hit_rate={self._ratio(self.target_visible_frames, frames):.3f} 目标命中率",
            f"guidance_generate_rate={self._ratio(self.guidance_frames, frames):.3f} 引导生成率",
            f"lost_target_since_ts={self.lost_target_since_ts} 目标首次消失时间戳",
        ]
        return "\n".join(lines)

    @staticmethod
    def _translate_task_state(task_state: str) -> str:
        mapping = {
            "idle": "空闲",
            "running": "进行中",
            "ready": "就绪",
            "finishing": "结束中",
            "finished": "已结束",
        }
        return mapping.get(task_state, task_state or "未知")

    @staticmethod
    def _translate_end_reason(end_reason: str) -> str:
        mapping = {
            "success": "成功完成",
            "switch_target": "切换目标",
            "lost_target": "目标丢失",
            "user_voice_exit": "语音退出",
            "error": "异常结束",
        }
        return mapping.get(end_reason, end_reason or "未知")

    @staticmethod
    def _section_meta(en_name: str, zh_name: str) -> Dict:
        return {
            "en": en_name,
            "zh": zh_name,
        }

    @staticmethod
    def _field_entry(en_name: str, zh_name: str, value, value_zh: Optional[str] = None) -> Dict:
        entry = {
            "en": en_name,
            "zh": zh_name,
            "value": value,
        }
        if value_zh is not None:
            entry["value_zh"] = value_zh
        _ts_suffixes = ("_ts", "_time", "_at")
        if (
            isinstance(value, (int, float))
            and value
            and any(en_name.endswith(s) for s in _ts_suffixes)
        ):
            entry["value_hkt"] = datetime.fromtimestamp(value, tz=_HKT).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3] + " HKT"
        return entry

    @staticmethod
    def _avg(total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return total / count

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator


class AsyncReportWriter:
    """后台异步写入任务报告。"""

    def __init__(self, output_dir: str, queue_size: int = 32):
        self.output_dir = output_dir
        self.queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def enqueue(self, envelope: TaskReportEnvelope):
        if envelope is None:
            return
        try:
            self.queue.put_nowait(envelope)
        except queue.Full:
            logger.warning(
                "任务报告队列已满，丢弃报告: task_id=%s",
                envelope.task_id,
            )

    def stop(self, timeout_sec: float = 2.0):
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(0.0, timeout_sec))

    def _writer_loop(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                envelope = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._write_report(envelope)
            except Exception as exc:
                logger.error(
                    "任务报告写入失败: task_id=%s path=%s error=%s",
                    envelope.task_id,
                    envelope.output_path,
                    exc,
                )
                if envelope.retry_count < 1:
                    retry_envelope = TaskReportEnvelope(
                        task_id=envelope.task_id,
                        output_path=envelope.output_path,
                        report_dict=envelope.report_dict,
                        created_at=envelope.created_at,
                        retry_count=envelope.retry_count + 1,
                    )
                    try:
                        self.queue.put_nowait(retry_envelope)
                    except queue.Full:
                        logger.error(
                            "任务报告重试入队失败: task_id=%s",
                            envelope.task_id,
                        )
            finally:
                self.queue.task_done()

    def _write_report(self, envelope: TaskReportEnvelope):
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = f"{envelope.output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(envelope.report_dict, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, envelope.output_path)

    def build_output_path(self, task_id: str, created_at: Optional[float] = None) -> str:
        ts = datetime.fromtimestamp(created_at or time.time(), tz=_HKT).strftime("%Y%m%d_%H%M%S")
        filename = f"task_metrics_{ts}_{task_id}.json"
        return os.path.join(self.output_dir, filename)
