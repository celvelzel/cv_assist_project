import os
import sys
import unittest
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.interfaces import TrackedDetection
from core.tracker import (
    NorfairObjectTracker,
    SimpleCentroidTracker,
    create_object_tracker
)


class TestSimpleCentroidTracker(unittest.TestCase):
    """Tests for SimpleCentroidTracker."""
    
    def setUp(self):
        self.tracker = SimpleCentroidTracker(
            max_disappeared=30,
            max_distance=100.0
        )
    
    def test_single_detection_registration(self):
        """Test registration of single detection."""
        detections = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            }
        ]
        
        tracked = self.tracker.update(detections)
        
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].tracking_id, 0)
        self.assertEqual(tracked[0].box, [100, 100, 200, 200])
        self.assertEqual(tracked[0].center, (150, 150))
        self.assertEqual(tracked[0].label, 'cup')
    
    def test_multiple_detection_registration(self):
        """Test registration of multiple detections."""
        detections = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            },
            {
                'box': [300, 100, 400, 200],
                'center': (350, 150),
                'score': 0.8,
                'label': 'bottle'
            }
        ]
        
        tracked = self.tracker.update(detections)
        
        self.assertEqual(len(tracked), 2)
        # IDs should be 0 and 1
        ids = {t.tracking_id for t in tracked}
        self.assertEqual(ids, {0, 1})
    
    def test_tracking_continuity(self):
        """Test that objects maintain consistent IDs across frames."""
        # First frame
        frame1 = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            },
            {
                'box': [300, 100, 400, 200],
                'center': (350, 150),
                'score': 0.8,
                'label': 'bottle'
            }
        ]
        
        tracked1 = self.tracker.update(frame1)
        
        # Second frame with slight movement
        frame2 = [
            {
                'box': [110, 105, 210, 205],  # Slight movement
                'center': (160, 155),
                'score': 0.9,
                'label': 'cup'
            },
            {
                'box': [305, 105, 405, 205],  # Slight movement
                'center': (355, 155),
                'score': 0.8,
                'label': 'bottle'
            }
        ]
        
        tracked2 = self.tracker.update(frame2)
        
        # IDs should be consistent
        id1_cup = [t.tracking_id for t in tracked1 if t.label == 'cup'][0]
        id2_cup = [t.tracking_id for t in tracked2 if t.label == 'cup'][0]
        self.assertEqual(id1_cup, id2_cup)
        
        id1_bottle = [t.tracking_id for t in tracked1 if t.label == 'bottle'][0]
        id2_bottle = [t.tracking_id for t in tracked2 if t.label == 'bottle'][0]
        self.assertEqual(id1_bottle, id2_bottle)
    
    def test_empty_detections(self):
        """Test with empty detections."""
        tracked = self.tracker.update([])
        self.assertEqual(len(tracked), 0)
    
    def test_reset_tracker(self):
        """Test tracker reset."""
        # First frame
        frame1 = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            }
        ]
        
        tracked1 = self.tracker.update(frame1)
        self.assertEqual(len(tracked1), 1)
        self.assertEqual(tracked1[0].tracking_id, 0)
        
        # Reset
        self.tracker.reset()
        
        # New frame after reset
        frame2 = [
            {
                'box': [200, 200, 300, 300],
                'center': (250, 250),
                'score': 0.8,
                'label': 'bottle'
            }
        ]
        
        tracked2 = self.tracker.update(frame2)
        self.assertEqual(len(tracked2), 1)
        # After reset, ID should start from 0 again
        self.assertEqual(tracked2[0].tracking_id, 0)


class TestNorfairObjectTracker(unittest.TestCase):
    """Tests for NorfairObjectTracker."""
    
    def setUp(self):
        # Use mock tracker if norfair not available
        self.tracker = NorfairObjectTracker(
            distance_function="euclidean",
            distance_threshold=100
        )
    
    def test_update_returns_tracked_detections(self):
        """Test that update returns TrackedDetection objects."""
        detections = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            }
        ]
        
        tracked = self.tracker.update(detections, frame_id=0)
        
        # Should return list of TrackedDetection
        self.assertIsInstance(tracked, list)
        if len(tracked) > 0:
            self.assertIsInstance(tracked[0], TrackedDetection)
            self.assertGreaterEqual(tracked[0].tracking_id, 0)
    
    def test_empty_detections(self):
        """Test with empty detections."""
        tracked = self.tracker.update([], frame_id=0)
        self.assertEqual(len(tracked), 0)
    
    def test_reset(self):
        """Test tracker reset."""
        # First update
        detections = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            }
        ]
        
        tracked1 = self.tracker.update(detections, frame_id=0)
        self.assertEqual(len(tracked1), 1)
        
        # Reset
        self.tracker.reset()
        
        # New update after reset
        tracked2 = self.tracker.update(detections, frame_id=1)
        self.assertEqual(len(tracked2), 1)


class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_norfair_tracker(self):
        """Test creating Norfair tracker."""
        tracker = create_object_tracker("norfair")
        self.assertIsInstance(tracker, NorfairObjectTracker)
    
    def test_create_simple_tracker(self):
        """Test creating Simple tracker."""
        tracker = create_object_tracker("simple")
        self.assertIsInstance(tracker, SimpleCentroidTracker)
    
    def test_invalid_tracker_type(self):
        """Test creating invalid tracker type."""
        with self.assertRaises(ValueError):
            create_object_tracker("invalid_type")


class TestIntegration(unittest.TestCase):
    """Integration tests for tracking with spatial selection."""
    
    def test_tracking_with_spatial_selection(self):
        """Test integration of tracking with spatial selection."""
        tracker = SimpleCentroidTracker()
        
        # Simulate two cups in frame
        frame1 = [
            {
                'box': [100, 100, 200, 200],
                'center': (150, 150),
                'score': 0.9,
                'label': 'cup'
            },
            {
                'box': [300, 100, 400, 200],
                'center': (350, 150),
                'score': 0.8,
                'label': 'cup'
            }
        ]
        
        tracked = tracker.update(frame1)
        
        # Verify we have two tracked cups
        self.assertEqual(len(tracked), 2)
        
        # Filter cups
        cups = [t for t in tracked if t.label == 'cup']
        self.assertEqual(len(cups), 2)
        
        # Verify IDs are assigned
        ids = {t.tracking_id for t in cups}
        self.assertEqual(len(ids), 2)  # Two unique IDs
        
        # Verify positions
        left_cup = min(cups, key=lambda t: t.center[0])
        right_cup = max(cups, key=lambda t: t.center[0])
        
        self.assertEqual(left_cup.center[0], 150)
        self.assertEqual(right_cup.center[0], 350)


if __name__ == "__main__":
    unittest.main()
