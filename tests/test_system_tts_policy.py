import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.system import CVAssistSystem
from utils.task_metrics import TaskMetricsCollector


class _DummyTTS:
    def __init__(self):
        self.messages = []
        self.stop_calls = 0
        self.clear_calls = 0

    def speak(self, text, block=False):
        self.messages.append((text, block))

    def speak_instruction(self, text):
        self.messages.append((text, False))

    def clear_queue(self):
        self.clear_calls += 1

    def stop(self):
        self.stop_calls += 1


class SystemTTSPolicyTests(unittest.TestCase):
    def _build_system_stub(self):
        system = CVAssistSystem.__new__(CVAssistSystem)
        system.config = SimpleNamespace(
            logging=SimpleNamespace(
                enable_task_metrics=True,
                task_metrics_interval_sec=1.0,
            ),
            audio=SimpleNamespace(
                tts_instruction_interval_sec=3.0,
                tts_grab_repeat_sec=3.0,
                tts_state_change_bypass=True,
                voice_feedback_on_target_confirm=True,
                voice_feedback_after_recording=True,
                target_found_feedback_enabled=True,
                target_found_frame_threshold=3,
                target_missing_feedback_enabled=True,
                target_missing_frame_threshold=3,
                guidance_suppress_after_voice_sec=1.5,
            )
        )
        system.config.target_queries = ["a cup"]
        system.tts_engine = _DummyTTS()
        system.cached_detections = []
        system._last_instruction = None
        system._last_instruction_state = None
        system._last_instruction_ts = 0.0
        system._last_grab_ts = 0.0
        system._voice_result_queue = __import__("queue").Queue()
        system._voice_in_progress = False
        system._voice_prompt_playing = False
        system._suppress_guidance_until_ts = 0.0
        system._search_feedback_target = None
        system._search_feedback_state = "idle"
        system._target_missing_streak = 0
        system._target_found_streak = 0
        system.session_id = "session-test"
        system.current_task = None
        system.task_state = "idle"
        system._task_index = 0
        system._last_task_summary_ts = 0.0
        system._requested_shutdown = False
        system.report_writer = None
        system.task_metrics_collector = TaskMetricsCollector(
            grasp_stable_frames=3,
            ready_confirm_window_sec=1.5,
            lost_target_frame_threshold=5,
        )
        return system

    def test_state_change_can_bypass_interval(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")
        ready = SimpleNamespace(instruction="准备抓取!", ready_to_grab=True, state="ready")

        with patch("core.system.time.time", return_value=100.0):
            self.assertTrue(system._should_speak_guidance(moving))
            system._speak_guidance(moving)

        # 仅过 1 秒，不满足常规 3 秒间隔；但状态变化 moving->ready，允许立即播报
        with patch("core.system.time.time", return_value=101.0):
            self.assertTrue(system._should_speak_guidance(ready))

    def test_same_moving_instruction_respects_interval(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")

        with patch("core.system.time.time", return_value=200.0):
            system._speak_guidance(moving)

        with patch("core.system.time.time", return_value=201.0):
            self.assertFalse(system._should_speak_guidance(moving))

        with patch("core.system.time.time", return_value=203.2):
            self.assertTrue(system._should_speak_guidance(moving))

    def test_priority_feedback_suppresses_guidance_temporarily(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")

        with patch("core.system.time.time", return_value=300.0):
            system._speak_priority_message("开始寻找目标主体杯子")
            self.assertEqual(system.tts_engine.messages[-1], ("开始寻找目标主体杯子", False))

        with patch("core.system.time.time", return_value=300.5):
            self.assertFalse(system._should_speak_guidance(moving))

        with patch("core.system.time.time", return_value=302.0):
            self.assertTrue(system._should_speak_guidance(moving))

    def test_voice_prompt_waits_for_tts_and_uses_blocking_speak(self):
        system = self._build_system_stub()

        system._play_voice_prompt_and_wait("请给出描述")

        self.assertEqual(system.tts_engine.stop_calls, 1)
        self.assertEqual(system.tts_engine.clear_calls, 1)
        self.assertEqual(system.tts_engine.messages[-1], ("请给出描述", True))
        self.assertFalse(system._voice_prompt_playing)

    def test_guidance_is_blocked_during_voice_input(self):
        system = self._build_system_stub()
        moving = SimpleNamespace(instruction="向右移动", ready_to_grab=False, state="moving")

        system._voice_in_progress = True
        with patch("core.system.time.time", return_value=350.0):
            self.assertFalse(system._should_speak_guidance(moving))

    def test_drain_voice_results_confirms_target_and_primes_search(self):
        system = self._build_system_stub()
        system._voice_result_queue.put(
            {
                "status": "ok",
                "action": "set_target",
                "target": "杯子",
                "message": "开始寻找目标主体杯子",
                "voice_total_time_ms": 123.0,
                "voice_asr_time_ms": 45.0,
                "raw_text": "找杯子",
            }
        )

        with patch("core.system.time.time", return_value=400.0):
            system._drain_voice_results()

        self.assertEqual(system.config.target_queries, ["杯子"])
        self.assertIsNotNone(system.current_task)
        self.assertEqual(system._search_feedback_target, "杯子")
        self.assertEqual(system._search_feedback_state, "searching")
        self.assertEqual(system.tts_engine.messages[-1], ("开始寻找目标主体杯子", False))

    def test_drain_voice_results_marks_shutdown_on_exit_action(self):
        system = self._build_system_stub()
        system.current_task = {"task_id": "task_0001", "target_query": "杯子", "start_time": 100.0}
        system.task_metrics_collector.start_task("task_0001", "杯子", 100.0, "session-test")
        system._voice_result_queue.put(
            {
                "status": "ok",
                "action": "user_voice_exit",
                "target": None,
                "message": "正在退出程序",
            }
        )

        with patch("core.system.time.time", return_value=410.0):
            system._drain_voice_results()

        self.assertTrue(system._requested_shutdown)
        self.assertIsNone(system.current_task)
        self.assertEqual(system.tts_engine.messages[-1], ("正在退出程序", False))

    def test_drain_voice_results_switches_target_and_restarts_task(self):
        system = self._build_system_stub()
        system.current_task = {"task_id": "task_0001", "target_query": "杯子", "start_time": 100.0}
        system.task_metrics_collector.start_task("task_0001", "杯子", 100.0, "session-test")
        system._voice_result_queue.put(
            {
                "status": "ok",
                "action": "switch_target",
                "target": "瓶子",
                "message": "原任务结束，切换至目标主体瓶子",
                "voice_total_time_ms": 210.0,
                "voice_asr_time_ms": 88.0,
                "raw_text": "找瓶子",
            }
        )

        with patch("core.system.time.time", return_value=420.0):
            system._drain_voice_results()

        self.assertIsNotNone(system.current_task)
        self.assertEqual(system.current_task["target_query"], "瓶子")
        self.assertEqual(system.config.target_queries, ["瓶子"])
        self.assertEqual(system.tts_engine.messages[-1], ("原任务结束，切换至目标主体瓶子", False))

    def test_search_feedback_reports_found_then_missing(self):
        system = self._build_system_stub()
        system._begin_target_search_feedback("杯子")

        with patch("core.system.time.time", return_value=500.0):
            system._update_target_search_feedback([{"label": "杯子"}])
        self.assertEqual(system._search_feedback_state, "searching")

        with patch("core.system.time.time", return_value=500.1):
            system._update_target_search_feedback([{"label": "杯子"}])
        self.assertEqual(system._search_feedback_state, "searching")

        with patch("core.system.time.time", return_value=500.2):
            system._update_target_search_feedback([{"label": "杯子"}])
        self.assertEqual(system._search_feedback_state, "found")
        self.assertEqual(system.tts_engine.messages[-1], ("已找到目标主体杯子", False))

        for idx in range(2):
            with patch("core.system.time.time", return_value=501.0 + idx):
                system._update_target_search_feedback([])
        self.assertEqual(system._search_feedback_state, "found")

        with patch("core.system.time.time", return_value=503.0):
            system._update_target_search_feedback([])
        self.assertEqual(system._search_feedback_state, "missing")
        self.assertEqual(system.tts_engine.messages[-1], ("暂未找到目标主体杯子，请调整位置后重试", False))

    def test_found_feedback_is_queued_after_confirm_without_interrupt(self):
        system = self._build_system_stub()

        with patch("core.system.time.time", return_value=600.0):
            system._speak_priority_message("开始寻找目标主体杯子")
        stop_calls = system.tts_engine.stop_calls
        clear_calls = system.tts_engine.clear_calls

        system._begin_target_search_feedback("杯子")
        with patch("core.system.time.time", return_value=600.2):
            system._update_target_search_feedback([{"label": "杯子"}])
        with patch("core.system.time.time", return_value=600.3):
            system._update_target_search_feedback([{"label": "杯子"}])
        with patch("core.system.time.time", return_value=600.4):
            system._update_target_search_feedback([{"label": "杯子"}])

        self.assertEqual(system.tts_engine.stop_calls, stop_calls)
        self.assertEqual(system.tts_engine.clear_calls, clear_calls)
        self.assertEqual(
            system.tts_engine.messages[-2:],
            [("开始寻找目标主体杯子", False), ("已找到目标主体杯子", False)]
        )


if __name__ == "__main__":
    unittest.main()
