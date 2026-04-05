import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.guidance import (
    GuidanceController,
    clock_hour_from_hand_to_target,
    clock_hour_from_horizontal_plane,
)


class GuidanceStabilityTests(unittest.TestCase):
    def setUp(self):
        self.guidance = GuidanceController(
            horizontal_threshold=30,
            vertical_threshold=30,
            depth_threshold=0.15,
            horizontal_threshold_enter=24,
            horizontal_threshold_exit=36,
            vertical_threshold_enter=24,
            vertical_threshold_exit=36,
            depth_threshold_enter=0.12,
            depth_threshold_exit=0.18,
            grasp_stable_frames=3,
            grasp_release_frames=2,
        )

    def test_ready_requires_consecutive_stable_frames(self):
        hand = (100, 100)
        target = (110, 108)  # 在 enter 阈值内

        r1 = self.guidance.calculate(hand, target, 0.5, 0.55, "unknown")
        r2 = self.guidance.calculate(hand, target, 0.5, 0.55, "unknown")
        r3 = self.guidance.calculate(hand, target, 0.5, 0.55, "unknown")

        self.assertFalse(r1.ready_to_grab)
        self.assertFalse(r2.ready_to_grab)
        self.assertTrue(r3.ready_to_grab)
        self.assertEqual(r3.state, "ready")

    def test_ready_state_has_release_debounce(self):
        hand = (100, 100)
        aligned = (110, 108)
        far_target = (180, 108)  # 远离阈值，触发 not-ready

        for _ in range(3):
            self.guidance.calculate(hand, aligned, 0.5, 0.55, "unknown")

        keep_ready_once = self.guidance.calculate(hand, far_target, 0.5, 0.55, "unknown")
        release_on_second = self.guidance.calculate(hand, far_target, 0.5, 0.55, "unknown")

        self.assertTrue(keep_ready_once.ready_to_grab)
        self.assertFalse(release_on_second.ready_to_grab)
        self.assertEqual(release_on_second.state, "moving")

    def test_hysteresis_keeps_previous_direction_in_middle_zone(self):
        hand = (100, 100)
        far_right = (150, 100)  # dx=50 > exit
        middle_right = (130, 100)  # dx=30, enter<dx<=exit

        first = self.guidance.calculate(hand, far_right, 0.5, 0.5, "unknown")
        second = self.guidance.calculate(hand, middle_right, 0.5, 0.5, "unknown")

        self.assertEqual(first.direction_h, "right")
        self.assertEqual(second.direction_h, "right")

    def test_clock_hour_twelve_and_three(self):
        self.assertEqual(clock_hour_from_hand_to_target(0, -50), 12)
        self.assertEqual(clock_hour_from_hand_to_target(50, 0), 3)

    def test_clock_flip_horizontal(self):
        self.assertEqual(clock_hour_from_hand_to_target(50, 0, flip_horizontal=True), 9)

    def test_clock_horizontal_twelve_and_three(self):
        # 仅深度轴（v>0）、无左右 → 12 点
        self.assertEqual(
            clock_hour_from_horizontal_plane(0, 0.01, depth_scale=450.0, depth_axis_sign=1.0),
            12,
        )
        # 仅右侧、无深度差 → 3 点
        self.assertEqual(
            clock_hour_from_horizontal_plane(50, 0.0, depth_scale=450.0, depth_axis_sign=1.0),
            3,
        )


if __name__ == "__main__":
    unittest.main()
