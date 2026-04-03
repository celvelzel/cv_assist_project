import os
import sys
import tempfile
import time
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio.asr import ASREngine
from utils.task_metrics import AsyncReportWriter, FrameMetrics, TaskMetricsCollector, TaskReportEnvelope


class VoiceEventParsingTests(unittest.TestCase):
    def _build_engine(self):
        engine = ASREngine.__new__(ASREngine)
        return engine

    def test_parse_voice_event_returns_exit_action_first(self):
        engine = self._build_engine()

        event = engine.parse_voice_event("退出程序", has_active_task=True)

        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["action"], "user_voice_exit")
        self.assertIsNone(event["target"])

    def test_parse_voice_event_returns_switch_target_for_active_task(self):
        engine = self._build_engine()
        engine.parse_command_with_vision = lambda text, frames=None, llm_parser=None: "a bottle"

        event = engine.parse_voice_event("帮我找一下瓶子", has_active_task=True)

        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["action"], "switch_target")
        self.assertEqual(event["target"], "a bottle")
        self.assertEqual(event["message"], "原任务结束，切换至目标主体a bottle")


class TaskMetricsCollectorTests(unittest.TestCase):
    def _frame(self, **overrides):
        data = dict(
            frame_index=1,
            frame_start_ts=100.0,
            frame_end_ts=100.1,
            capture_time_ms=5.0,
            process_time_ms=20.0,
            draw_time_ms=4.0,
            display_time_ms=2.0,
            e2e_loop_time_ms=31.0,
            detection_time_ms=8.0,
            hand_time_ms=3.0,
            depth_time_ms=5.0,
            guidance_time_ms=1.0,
            detection_executed=True,
            depth_executed=True,
            detections_count=1,
            hands_count=1,
            has_target=True,
            has_hand=True,
            has_guidance=True,
            guidance_state="moving",
            ready_to_grab=False,
            stable_ready_frames=0,
            gesture="open",
            target_visible=True,
            proc_fps_current=30.0,
            proc_fps_avg=28.0,
            e2e_fps_current=24.0,
            e2e_fps_avg=23.0,
        )
        data.update(overrides)
        return FrameMetrics(**data)

    def test_collector_marks_success_after_ready_then_closed(self):
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=5.0,
        )
        collector.start_task("task_0001", "a bottle", 100.0, "session-1")
        collector.record_voice_metrics(500.0, 120.0, "find bottle")

        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=True, stable_ready_frames=1, guidance_state="ready"))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=True, stable_ready_frames=2, guidance_state="ready"))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=True, stable_ready_frames=3, guidance_state="ready"))
        collector.record_frame(self._frame(frame_index=4, frame_end_ts=100.6, ready_to_grab=True, stable_ready_frames=4, guidance_state="grabbed", gesture="closed"))

        self.assertEqual(collector.should_finish_task(), "success")
        report = collector.finish_task("success", 101.0)

        self.assertEqual(report["task_info"]["section"]["zh"], "任务信息")
        self.assertEqual(report["task_info"]["end_reason"]["value"], "success")
        self.assertEqual(report["task_info"]["end_reason"]["value_zh"], "成功完成")
        self.assertEqual(report["task_info"]["task_state"]["value_zh"], "已结束")
        self.assertEqual(report["voice_summary"]["raw_text"]["value"], "find bottle")
        self.assertEqual(report["voice_summary"]["raw_text"]["zh"], "语音原文")
        self.assertTrue(report["completion_summary"]["closed_after_ready_flag"]["value"])

    def test_collector_marks_lost_target_after_threshold(self):
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=0.5,
        )
        collector.start_task("task_0002", "a cup", 200.0, "session-1")

        collector.record_frame(self._frame(frame_index=1, frame_end_ts=200.1, target_visible=False, has_target=False, detections_count=0, has_guidance=False, guidance_state="idle"))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=200.4, target_visible=False, has_target=False, detections_count=0, has_guidance=False, guidance_state="idle"))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=200.7, target_visible=False, has_target=False, detections_count=0, has_guidance=False, guidance_state="idle"))

        self.assertEqual(collector.should_finish_task(), "lost_target")
        self.assertFalse(collector.should_emit_report())

    def test_collector_emits_report_only_after_target_detected(self):
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=3.0,
        )
        collector.start_task("task_0002b", "a bottle", 210.0, "session-1")
        collector.record_frame(self._frame(frame_index=1, frame_end_ts=210.1, target_visible=True, has_target=True))

        self.assertTrue(collector.should_emit_report())

    def test_finish_task_can_emit_error_reason(self):
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=3.0,
        )
        collector.start_task("task_0003", "a phone", 300.0, "session-1")
        collector.record_frame(self._frame(frame_index=1, frame_end_ts=300.1))

        report = collector.finish_task("error", 301.0, error_message="camera disconnected")

        self.assertEqual(report["task_info"]["end_reason"]["value"], "error")
        self.assertEqual(report["error_summary"]["message"]["value"], "camera disconnected")

    def test_terminal_summary_is_multiline(self):
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=3.0,
        )
        collector.start_task("task_0004", "a bottle", 400.0, "session-1")
        collector.record_voice_metrics(10611.5, 864.2, "find bottle")
        collector.record_frame(
            self._frame(
                frame_index=1,
                frame_end_ts=400.1,
                guidance_state="ready",
                ready_to_grab=True,
                stable_ready_frames=3,
                proc_fps_current=14.0,
                proc_fps_avg=27.1,
                e2e_fps_current=11.7,
                e2e_fps_avg=20.1,
            )
        )

        summary = collector.build_terminal_summary(475.6)

        self.assertIn("[task_metrics]\n", summary)
        self.assertIn("task_id=task_0004 任务ID", summary)
        self.assertIn("target=a bottle 当前目标主体", summary)
        self.assertIn("task_state=ready(就绪) 任务状态", summary)
        self.assertIn("voice_total_time_ms=10611.5 语音总耗时毫秒", summary)
        self.assertIn("target_detect_hit_rate=", summary)
        self.assertIn("目标命中率", summary)
        self.assertGreaterEqual(summary.count("\n"), 15)


class AsyncReportWriterTests(unittest.TestCase):
    def test_writer_persists_report_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = AsyncReportWriter(output_dir=temp_dir)
            writer.start()
            output_path = writer.build_output_path("task_0001", created_at=1700000000)
            envelope = TaskReportEnvelope(
                task_id="task_0001",
                output_path=output_path,
                report_dict={"task_info": {"task_id": "task_0001"}},
                created_at=time.time(),
            )
            writer.enqueue(envelope)
            time.sleep(0.4)
            writer.stop(timeout_sec=1.0)

            self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
