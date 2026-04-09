import os
import sys
import tempfile
import time
import unittest
from typing import Any

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
        data: dict[str, Any] = dict(
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
            target_x=300.0,
            hand_near_target=True,
            hand_target_distance_px=20.0,
            proc_fps_current=30.0,
            proc_fps_avg=28.0,
            e2e_fps_current=24.0,
            e2e_fps_avg=23.0,
        )
        data.update(overrides)
        return FrameMetrics(**data)

    def test_collector_marks_success_after_ready_then_x_displacement(self):
        """就绪后目标 X 轴稳定位移触发抓取成功。"""
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=5.0,
            catch_x_min_displacement_px=20,
            catch_x_stable_frames=3,
            catch_x_stable_max_std_px=10.0,
        )
        collector.start_task("task_0001", "a bottle", 100.0, "session-1")
        collector.record_voice_metrics(500.0, 120.0, "find bottle")

        # 手部就绪（target_x=300 作为基准）
        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=True, stable_ready_frames=1, guidance_state="ready", target_x=300.0))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=True, stable_ready_frames=2, guidance_state="ready", target_x=300.0))
        # stable_ready_frames=3 时记录 _target_x_at_ready_enter=300
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=True, stable_ready_frames=3, guidance_state="ready", target_x=300.0))
        # 目标 X 发生横向移动，模拟被抓住后带动位置变化
        collector.record_frame(self._frame(frame_index=4, frame_end_ts=100.4, ready_to_grab=True, stable_ready_frames=4, guidance_state="ready", target_x=268.0))
        collector.record_frame(self._frame(frame_index=5, frame_end_ts=100.5, ready_to_grab=True, stable_ready_frames=5, guidance_state="ready", target_x=270.0))
        # 第三帧稳定：位移=30px > 20px，std 约 1px < 10px → 触发成功
        collector.record_frame(self._frame(frame_index=6, frame_end_ts=100.6, ready_to_grab=True, stable_ready_frames=6, guidance_state="ready", target_x=269.0))

        self.assertEqual(collector.should_finish_task(), "success")
        report = collector.finish_task("success", 101.0)

        self.assertEqual(report["task_info"]["end_reason"]["value"], "success")
        self.assertEqual(report["task_info"]["end_reason"]["value_zh"], "成功完成")
        self.assertEqual(report["task_info"]["task_state"]["value_zh"], "已结束")
        self.assertEqual(report["voice_summary"]["raw_text"]["value"], "find bottle")
        self.assertTrue(report["completion_summary"]["closed_after_ready_flag"]["value"])
        self.assertEqual(report["completion_summary"]["catch_trigger"]["value"], "target_x_displacement_ready")
        self.assertIsNotNone(report["completion_summary"]["target_x_displacement_px"]["value"])
        self.assertGreaterEqual(report["completion_summary"]["target_x_displacement_px"]["value"], 20.0)
        self.assertGreater(report["completion_summary"]["ready_to_catch_elapsed_sec"]["value"], 0.0)

    def test_collector_does_not_trigger_success_without_sufficient_displacement(self):
        """位移未达阈值时不应触发成功。"""
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_window_sec=5.0,
            catch_x_min_displacement_px=40,
            catch_x_stable_frames=3,
            catch_x_stable_max_std_px=10.0,
        )
        collector.start_task("task_0001b", "a cup", 100.0, "session-1")

        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=True, stable_ready_frames=1, guidance_state="ready", target_x=300.0))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=True, stable_ready_frames=2, guidance_state="ready", target_x=300.0))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=True, stable_ready_frames=3, guidance_state="ready", target_x=300.0))
        # 目标只移动了 10px，低于 40px 阈值
        collector.record_frame(self._frame(frame_index=4, frame_end_ts=100.4, ready_to_grab=True, stable_ready_frames=4, guidance_state="ready", target_x=291.0))
        collector.record_frame(self._frame(frame_index=5, frame_end_ts=100.5, ready_to_grab=True, stable_ready_frames=5, guidance_state="ready", target_x=290.0))
        collector.record_frame(self._frame(frame_index=6, frame_end_ts=100.6, ready_to_grab=True, stable_ready_frames=6, guidance_state="ready", target_x=291.0))

        self.assertIsNone(collector.should_finish_task())

    def test_collector_can_finish_by_x_move_after_leaving_ready(self):
        """进入 ready 后即使退出 ready，只要目标 X 位移稳定达标仍应判定成功。"""
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.0,
            lost_target_window_sec=5.0,
            catch_x_min_displacement_px=20,
            catch_x_stable_frames=3,
            catch_x_stable_max_std_px=10.0,
        )
        collector.start_task("task_0001c", "a can", 100.0, "session-1")

        # 先进入 ready，建立位移基准 X=300
        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=True, stable_ready_frames=1, guidance_state="ready", target_x=300.0))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=True, stable_ready_frames=2, guidance_state="ready", target_x=300.0))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=True, stable_ready_frames=3, guidance_state="ready", target_x=300.0))

        # 抓取动作后短暂离开 ready（例如遮挡/抖动），但目标已发生横向稳定移动
        collector.record_frame(self._frame(frame_index=4, frame_end_ts=101.8, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=269.0))
        collector.record_frame(self._frame(frame_index=5, frame_end_ts=101.9, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=270.0))
        collector.record_frame(self._frame(frame_index=6, frame_end_ts=102.0, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=268.0))

        self.assertEqual(collector.should_finish_task(), "success")

    def test_collector_can_finish_by_x_move_without_ready(self):
        """未进入 ready 时，只要手已靠近目标，目标 X 稳定位移也可触发成功。"""
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.0,
            lost_target_window_sec=5.0,
            catch_x_min_displacement_px=20,
            catch_x_stable_frames=3,
            catch_x_stable_max_std_px=10.0,
        )
        collector.start_task("task_0001d", "a bottle", 100.0, "session-1")

        # 全程不进入 ready，但手已经靠近目标，使用 hand-near 时的目标位置作为位移基准
        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=True))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=True))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=True))

        collector.record_frame(self._frame(frame_index=4, frame_end_ts=100.4, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=268.0, hand_near_target=True))
        collector.record_frame(self._frame(frame_index=5, frame_end_ts=100.5, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=269.0, hand_near_target=True))
        collector.record_frame(self._frame(frame_index=6, frame_end_ts=100.6, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=270.0, hand_near_target=True))

        self.assertEqual(collector.should_finish_task(), "success")

    def test_collector_does_not_finish_by_x_move_without_hand_near(self):
        """若手未靠近目标，即使目标 X 移动也不应触发成功。"""
        collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.0,
            lost_target_window_sec=5.0,
            catch_x_min_displacement_px=20,
            catch_x_stable_frames=3,
            catch_x_stable_max_std_px=10.0,
        )
        collector.start_task("task_0001e", "a bottle", 100.0, "session-1")

        collector.record_frame(self._frame(frame_index=1, frame_end_ts=100.1, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=False, hand_target_distance_px=200.0))
        collector.record_frame(self._frame(frame_index=2, frame_end_ts=100.2, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=False, hand_target_distance_px=200.0))
        collector.record_frame(self._frame(frame_index=3, frame_end_ts=100.3, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=300.0, hand_near_target=False, hand_target_distance_px=200.0))
        collector.record_frame(self._frame(frame_index=4, frame_end_ts=100.4, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=268.0, hand_near_target=False, hand_target_distance_px=200.0))
        collector.record_frame(self._frame(frame_index=5, frame_end_ts=100.5, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=269.0, hand_near_target=False, hand_target_distance_px=200.0))
        collector.record_frame(self._frame(frame_index=6, frame_end_ts=100.6, ready_to_grab=False, stable_ready_frames=0, guidance_state="moving", target_x=270.0, hand_near_target=False, hand_target_distance_px=200.0))

        self.assertIsNone(collector.should_finish_task())

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

    def test_terminal_summary_is_compact_line(self):
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

        self.assertIn("[task]", summary)
        self.assertIn("task_0004", summary)
        self.assertIn("a bottle", summary)
        self.assertIn("ready(就绪)", summary)
        self.assertIn("fps=", summary)
        self.assertIn("det=", summary)
        self.assertIn("ready=", summary)
        self.assertEqual(summary.count("\n"), 0)


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
