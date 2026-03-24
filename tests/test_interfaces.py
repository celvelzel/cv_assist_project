import os
import sys
import unittest
from typing import Optional, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.interfaces import ParsedIntent, TrackedDetection, IIntentParser, ISpatialResolver, CameraFacing


class MockIntentParser:
    """Mock implementation of IIntentParser for testing."""
    
    def parse(self, text: str) -> Optional[ParsedIntent]:
        """Simple mock that extracts 'left/right + object' patterns."""
        if not text:
            return None
        
        text = text.lower()
        
        # Simple pattern matching for testing
        spatial_modifiers = ["left", "right", "左边", "右边"]
        modifier = None
        target = text
        
        for mod in spatial_modifiers:
            if mod in text:
                modifier = mod
                target = text.replace(mod, "").strip()
                break
        
        return ParsedIntent(
            target_class=target,
            spatial_modifier=modifier,
            raw_text=text
        )


class MockSpatialResolver:
    """Mock implementation of ISpatialResolver for testing."""
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """Select based on spatial modifier and track positions."""
        if not tracks:
            return None
        
        # Sort by x-coordinate (center[0])
        sorted_tracks = sorted(tracks, key=lambda t: t.center[0])
        
        if spatial_modifier in ["left", "左边"]:
            return sorted_tracks[0] if sorted_tracks else None
        elif spatial_modifier in ["right", "右边"]:
            return sorted_tracks[-1] if sorted_tracks else None
        else:
            return sorted_tracks[0]  # Default to first


class TestInterfaces(unittest.TestCase):
    """Tests for interface contracts and mock implementations."""
    
    def test_parsed_intent_structure(self):
        """Verify ParsedIntent has correct structure."""
        intent = ParsedIntent(
            target_class="cup",
            spatial_modifier="left",
            raw_text="left cup"
        )
        
        self.assertEqual(intent.target_class, "cup")
        self.assertEqual(intent.spatial_modifier, "left")
        self.assertEqual(intent.raw_text, "left cup")
        
        # Test tuple unpacking
        target, spatial, raw = intent
        self.assertEqual(target, "cup")
        self.assertEqual(spatial, "left")
        self.assertEqual(raw, "left cup")
    
    def test_tracked_detection_structure(self):
        """Verify TrackedDetection has correct structure."""
        track = TrackedDetection(
            box=[100, 100, 200, 200],
            center=(150, 150),
            score=0.95,
            label="cup",
            tracking_id=42,
            age=5
        )
        
        self.assertEqual(track.box, [100, 100, 200, 200])
        self.assertEqual(track.center, (150, 150))
        self.assertEqual(track.score, 0.95)
        self.assertEqual(track.label, "cup")
        self.assertEqual(track.tracking_id, 42)
        self.assertEqual(track.age, 5)
    
    def test_mock_intent_parser(self):
        """Test mock implementation of IIntentParser."""
        parser = MockIntentParser()
        
        # Test spatial modifier extraction
        intent1 = parser.parse("left cup")
        self.assertIsNotNone(intent1)
        self.assertEqual(intent1.spatial_modifier, "left")
        self.assertEqual(intent1.target_class, "cup")
        
        # Test Chinese modifier
        intent2 = parser.parse("找到左边的杯子")
        self.assertIsNotNone(intent2)
        self.assertEqual(intent2.spatial_modifier, "左边")
        
        # Test no modifier
        intent3 = parser.parse("cup")
        self.assertIsNotNone(intent3)
        self.assertIsNone(intent3.spatial_modifier)
        
        # Test empty input
        intent4 = parser.parse("")
        self.assertIsNone(intent4)
    
    def test_mock_spatial_resolver(self):
        """Test mock implementation of ISpatialResolver."""
        resolver = MockSpatialResolver()
        
        tracks = [
            TrackedDetection(
                box=[300, 100, 400, 200],
                center=(350, 150),
                score=0.9,
                label="cup",
                tracking_id=1
            ),
            TrackedDetection(
                box=[50, 100, 150, 200],
                center=(100, 150),
                score=0.9,
                label="cup",
                tracking_id=2
            ),
            TrackedDetection(
                box=[200, 100, 300, 200],
                center=(250, 150),
                score=0.9,
                label="cup",
                tracking_id=3
            ),
        ]
        
        # Test left selection (should pick track with smallest x)
        left_track = resolver.resolve("left", tracks)
        self.assertEqual(left_track.tracking_id, 2)  # x=100 is smallest
        
        # Test right selection (should pick track with largest x)
        right_track = resolver.resolve("right", tracks)
        self.assertEqual(right_track.tracking_id, 1)  # x=350 is largest
        
        # Test empty tracks
        empty_result = resolver.resolve("left", [])
        self.assertIsNone(empty_result)
    
    def test_camera_facing_type(self):
        """Verify CameraFacing type accepts valid values."""
        from typing import get_type_hints
        import typing
        
        # CameraFacing should be Literal["outward", "user"]
        # This is a compile-time check, but we verify the definition exists
        from core.interfaces import CameraFacing
        
        # These should not raise type errors at runtime
        outward: CameraFacing = "outward"
        user: CameraFacing = "user"
        
        self.assertEqual(outward, "outward")
        self.assertEqual(user, "user")


class TestInterfaceCompatibility(unittest.TestCase):
    """Tests to ensure interfaces can be used as intended."""
    
    def test_parser_return_type_compatibility(self):
        """Verify parser output can be used to filter tracks."""
        parser = MockIntentParser()
        resolver = MockSpatialResolver()
        
        # Simulate full flow
        intent = parser.parse("left cup")
        self.assertIsNotNone(intent)
        
        # Create mock tracks for "cup"
        tracks = [
            TrackedDetection([100, 100, 200, 200], (150, 150), 0.9, "cup", 1),
            TrackedDetection([300, 100, 400, 200], (350, 150), 0.9, "cup", 2),
        ]
        
        # Filter by class
        class_tracks = [t for t in tracks if t.label == intent.target_class]
        
        # Resolve spatial selection
        selected = resolver.resolve(intent.spatial_modifier, class_tracks)
        
        self.assertIsNotNone(selected)
        self.assertEqual(selected.tracking_id, 1)  # leftmost
        self.assertEqual(selected.label, "cup")


if __name__ == "__main__":
    unittest.main()
