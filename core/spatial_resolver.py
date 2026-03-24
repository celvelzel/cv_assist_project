# core/spatial_resolver.py
"""
Spatial resolution module for mapping spatial modifiers to tracked objects.
Supports camera view inversion for front-facing camera scenarios.
"""

import logging
from typing import Optional, List

from core.interfaces import ISpatialResolver, TrackedDetection, CameraFacing

logger = logging.getLogger(__name__)


class StandardSpatialResolver:
    """
    Standard spatial resolver where image coordinates match user perspective.
    Left in image = user's left.
    """
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """
        Select a tracked object based on spatial modifier.
        
        Args:
            spatial_modifier: Spatial term ("left", "right", "center", "nearest")
            tracks: List of currently tracked detections
            reference_point: Optional (x,y) reference for "nearest" calculations
            
        Returns:
            Selected TrackedDetection, or None if no match found
        """
        if not tracks:
            return None
        
        # Filter out invalid tracks
        valid_tracks = [t for t in tracks if t.tracking_id >= 0]
        if not valid_tracks:
            return None
        
        if spatial_modifier in ["left", "左边"]:
            # Return track with smallest x-coordinate (leftmost in image)
            return min(valid_tracks, key=lambda t: t.center[0])
        
        elif spatial_modifier in ["right", "右边"]:
            # Return track with largest x-coordinate (rightmost in image)
            return max(valid_tracks, key=lambda t: t.center[0])
        
        elif spatial_modifier in ["center", "中间"]:
            # Return track closest to horizontal center
            if not valid_tracks:
                return None
            # Find horizontal center of all tracks
            centers = [t.center[0] for t in valid_tracks]
            center_x = sum(centers) / len(centers)
            # Return track closest to center
            return min(valid_tracks, key=lambda t: abs(t.center[0] - center_x))
        
        elif spatial_modifier in ["nearest", "最近的"]:
            # Return track closest to reference point
            if reference_point:
                return min(valid_tracks, key=lambda t: 
                    ((t.center[0] - reference_point[0])**2 + 
                     (t.center[1] - reference_point[1])**2)**0.5)
            else:
                # Default to center track if no reference
                return self.resolve("center", tracks)
        
        else:
            # Default to first track if modifier not recognized
            logger.warning(f"Unrecognized spatial modifier: {spatial_modifier}")
            return valid_tracks[0] if valid_tracks else None


class InvertedSpatialResolver:
    """
    Inverted spatial resolver for front-facing cameras.
    Left in image = user's right (mirrored perspective).
    """
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """
        Select a tracked object with inverted left/right logic.
        
        Args:
            spatial_modifier: Spatial term ("left", "right", "center", "nearest")
            tracks: List of currently tracked detections
            reference_point: Optional (x,y) reference for "nearest" calculations
            
        Returns:
            Selected TrackedDetection, or None if no match found
        """
        if not tracks:
            return None
        
        # Filter out invalid tracks
        valid_tracks = [t for t in tracks if t.tracking_id >= 0]
        if not valid_tracks:
            return None
        
        if spatial_modifier in ["left", "左边"]:
            # Inverted: user's left = image's right
            return max(valid_tracks, key=lambda t: t.center[0])
        
        elif spatial_modifier in ["right", "右边"]:
            # Inverted: user's right = image's left
            return min(valid_tracks, key=lambda t: t.center[0])
        
        elif spatial_modifier in ["center", "中间"]:
            # Center is the same regardless of inversion
            if not valid_tracks:
                return None
            centers = [t.center[0] for t in valid_tracks]
            center_x = sum(centers) / len(centers)
            return min(valid_tracks, key=lambda t: abs(t.center[0] - center_x))
        
        elif spatial_modifier in ["nearest", "最近的"]:
            # Nearest is the same regardless of inversion
            if reference_point:
                return min(valid_tracks, key=lambda t: 
                    ((t.center[0] - reference_point[0])**2 + 
                     (t.center[1] - reference_point[1])**2)**0.5)
            else:
                return self.resolve("center", tracks)
        
        else:
            # Default to first track if modifier not recognized
            logger.warning(f"Unrecognized spatial modifier: {spatial_modifier}")
            return valid_tracks[0] if valid_tracks else None


class AdaptiveSpatialResolver:
    """
    Adaptive spatial resolver that can switch between standard and inverted modes.
    """
    
    def __init__(self, camera_facing: CameraFacing = "outward"):
        """
        Initialize with camera facing mode.
        
        Args:
            camera_facing: "outward" (standard) or "user" (inverted)
        """
        self.camera_facing = camera_facing
        self._standard_resolver = StandardSpatialResolver()
        self._inverted_resolver = InvertedSpatialResolver()
        
        logger.info(f"AdaptiveSpatialResolver initialized with camera_facing={camera_facing}")
    
    def set_camera_facing(self, camera_facing: CameraFacing):
        """Update camera facing mode."""
        self.camera_facing = camera_facing
        logger.info(f"Camera facing updated to: {camera_facing}")
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """
        Resolve spatial selection using appropriate resolver based on camera mode.
        """
        if self.camera_facing == "user":
            return self._inverted_resolver.resolve(spatial_modifier, tracks, reference_point)
        else:
            return self._standard_resolver.resolve(spatial_modifier, tracks, reference_point)


# Factory function for creating spatial resolvers
def create_spatial_resolver(camera_facing: CameraFacing = "outward") -> ISpatialResolver:
    """
    Factory function to create spatial resolver instances.
    
    Args:
        camera_facing: Camera orientation ("outward" or "user")
        
    Returns:
        An implementation of ISpatialResolver
    """
    return AdaptiveSpatialResolver(camera_facing)
