import os
import sys
import unittest
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.interfaces import TrackedDetection
from core.spatial_resolver import (
    StandardSpatialResolver, 
    InvertedSpatialResolver, 
    AdaptiveSpatialResolver,
    create_spatial_resolver
)


class TestStandardSpatialResolver(unittest.TestCase):
    """Tests for StandardSpatialResolver."""
    
    def setUp(self):
        self.resolver = StandardSpatialResolver()
        
        # Create sample tracks
        self.tracks = [
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
    
    def test_left_selection(self):
        """Test leftmost track selection."""
        left_track = self.resolver.resolve("left", self.tracks)
        self.assertEqual(left_track.tracking_id, 2)  # x=100 is smallest
    
    def test_right_selection(self):
        """Test rightmost track selection."""
        right_track = self.resolver.resolve("right", self.tracks)
        self.assertEqual(right_track.tracking_id, 1)  # x=350 is largest
    
    def test_center_selection(self):
        """Test center track selection."""
        center_track = self.resolver.resolve("center", self.tracks)
        self.assertEqual(center_track.tracking_id, 3)  # x=250 is closest to center
    
    def test_nearest_selection(self):
        """Test nearest track selection with reference point."""
        reference = (120, 150)
        nearest_track = self.resolver.resolve("nearest", self.tracks, reference)
        self.assertEqual(nearest_track.tracking_id, 2)  # (100, 150) is closest to (120, 150)
    
    def test_empty_tracks(self):
        """Test with empty tracks."""
        result = self.resolver.resolve("left", [])
        self.assertIsNone(result)
    
    def test_invalid_modifier(self):
        """Test with invalid modifier."""
        result = self.resolver.resolve("invalid", self.tracks)
        self.assertIsNotNone(result)  # Should default to first track


class TestInvertedSpatialResolver(unittest.TestCase):
    """Tests for InvertedSpatialResolver."""
    
    def setUp(self):
        self.resolver = InvertedSpatialResolver()
        
        # Create sample tracks
        self.tracks = [
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
    
    def test_left_selection_inverted(self):
        """Test inverted left selection (user's left = image's right)."""
        left_track = self.resolver.resolve("left", self.tracks)
        self.assertEqual(left_track.tracking_id, 1)  # x=350 is largest (rightmost in image)
    
    def test_right_selection_inverted(self):
        """Test inverted right selection (user's right = image's left)."""
        right_track = self.resolver.resolve("right", self.tracks)
        self.assertEqual(right_track.tracking_id, 2)  # x=100 is smallest (leftmost in image)
    
    def test_center_selection_inverted(self):
        """Test center selection (should be same as standard)."""
        center_track = self.resolver.resolve("center", self.tracks)
        self.assertEqual(center_track.tracking_id, 3)  # x=250 is closest to center
    
    def test_nearest_selection_inverted(self):
        """Test nearest selection (should be same as standard)."""
        reference = (120, 150)
        nearest_track = self.resolver.resolve("nearest", self.tracks, reference)
        self.assertEqual(nearest_track.tracking_id, 2)  # (100, 150) is closest to (120, 150)


class TestAdaptiveSpatialResolver(unittest.TestCase):
    """Tests for AdaptiveSpatialResolver."""
    
    def test_camera_facing_outward(self):
        """Test standard camera mode."""
        resolver = AdaptiveSpatialResolver("outward")
        
        tracks = [
            TrackedDetection([100, 100, 200, 200], (150, 150), 0.9, "cup", 1),
            TrackedDetection([300, 100, 400, 200], (350, 150), 0.9, "cup", 2),
        ]
        
        left_track = resolver.resolve("left", tracks)
        self.assertEqual(left_track.tracking_id, 1)  # x=150 is smallest
    
    def test_camera_facing_user(self):
        """Test user-facing camera mode."""
        resolver = AdaptiveSpatialResolver("user")
        
        tracks = [
            TrackedDetection([100, 100, 200, 200], (150, 150), 0.9, "cup", 1),
            TrackedDetection([300, 100, 400, 200], (350, 150), 0.9, "cup", 2),
        ]
        
        left_track = resolver.resolve("left", tracks)
        self.assertEqual(left_track.tracking_id, 2)  # x=350 is largest (inverted)
    
    def test_set_camera_facing(self):
        """Test dynamic camera mode switching."""
        resolver = AdaptiveSpatialResolver("outward")
        
        tracks = [
            TrackedDetection([100, 100, 200, 200], (150, 150), 0.9, "cup", 1),
            TrackedDetection([300, 100, 400, 200], (350, 150), 0.9, "cup", 2),
        ]
        
        # Initially outward
        left_track = resolver.resolve("left", tracks)
        self.assertEqual(left_track.tracking_id, 1)
        
        # Switch to user-facing
        resolver.set_camera_facing("user")
        left_track = resolver.resolve("left", tracks)
        self.assertEqual(left_track.tracking_id, 2)


class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_standard_resolver(self):
        """Test creating standard resolver."""
        resolver = create_spatial_resolver("outward")
        self.assertIsInstance(resolver, AdaptiveSpatialResolver)
        self.assertEqual(resolver.camera_facing, "outward")
    
    def test_create_inverted_resolver(self):
        """Test creating inverted resolver."""
        resolver = create_spatial_resolver("user")
        self.assertIsInstance(resolver, AdaptiveSpatialResolver)
        self.assertEqual(resolver.camera_facing, "user")


if __name__ == "__main__":
    unittest.main()
