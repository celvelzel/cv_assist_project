# core/tracker.py
"""
Object tracking module using Norfair for stable object identification.
Wraps Norfair's SORT implementation with a clean interface.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from core.interfaces import TrackedDetection

logger = logging.getLogger(__name__)

# Try to import norfair, fallback to mock implementation
try:
    from norfair import Detection as NorfairDetection, Tracker
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False
    logger.warning("Norfair not installed. Tracking will use mock implementation.")


class NorfairObjectTracker:
    """
    Object tracker using Norfair for stable object identification.
    Maintains consistent tracking IDs across frames.
    """
    
    def __init__(self, 
                 distance_function: str = "euclidean",
                 distance_threshold: int = 100,
                 hit_counter_max: int = 30,
                 initialization_delay: int = 5):
        """
        Initialize the object tracker.
        
        Args:
            distance_function: Distance metric for association ("euclidean", "iou")
            distance_threshold: Maximum distance for association
            hit_counter_max: Frames before track is considered lost
            initialization_delay: Frames before new track is confirmed
        """
        self.distance_function = distance_function
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.initialization_delay = initialization_delay
        
        if NORFAIR_AVAILABLE:
            self._tracker = Tracker(
                distance_function=distance_function,
                distance_threshold=distance_threshold,
                hit_counter_max=hit_counter_max,
                initialization_delay=initialization_delay
            )
            logger.info("NorfairObjectTracker initialized with Norfair backend")
        else:
            self._tracker = None
            # Simple mock tracker state
            self._next_id = 1
            self._tracks: Dict[int, Dict] = {}
            logger.info("NorfairObjectTracker initialized with mock backend")
    
    def update(self, detections: List[Dict[str, Any]], frame_id: int = 0) -> List[TrackedDetection]:
        """
        Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of detection dictionaries with 'box', 'score', 'label', 'center'
            frame_id: Optional frame identifier
            
        Returns:
            List of TrackedDetection with stable tracking IDs
        """
        if not detections:
            return []
        
        if NORFAIR_AVAILABLE and self._tracker:
            return self._update_norfair(detections, frame_id)
        else:
            return self._update_mock(detections, frame_id)
    
    def _update_norfair(self, detections: List[Dict[str, Any]], frame_id: int) -> List[TrackedDetection]:
        """Update using Norfair backend."""
        # Convert detections to Norfair format
        norfair_detections = []
        for det in detections:
            box = det.get('box', [0, 0, 0, 0])
            if len(box) != 4:
                continue
                
            # Norfair expects [x1, y1, x2, y2] format
            norfair_det = NorfairDetection(
                points=np.array([[box[0], box[1]], [box[2], box[3]]]),
                scores=np.array([det.get('score', 1.0), det.get('score', 1.0)]),
                label=det.get('label', 'unknown')
            )
            norfair_detections.append(norfair_det)
        
        # Update tracker
        tracked_objects = self._tracker.update(norfair_detections, period=1)
        
        # Convert back to TrackedDetection format
        tracked_detections = []
        for obj in tracked_objects:
            # Get bounding box from tracked object
            if hasattr(obj, 'estimate') and obj.estimate is not None:
                points = obj.estimate
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    tracked_det = TrackedDetection(
                        box=[float(x1), float(y1), float(x2), float(y2)],
                        center=center,
                        score=1.0,  # Norfair doesn't preserve scores
                        label=obj.label if hasattr(obj, 'label') else 'unknown',
                        tracking_id=int(obj.id),
                        age=getattr(obj, 'age', 0)
                    )
                    tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def _update_mock(self, detections: List[Dict[str, Any]], frame_id: int) -> List[TrackedDetection]:
        """Simple mock tracker implementation."""
        tracked_detections = []
        
        for det in detections:
            box = det.get('box', [0, 0, 0, 0])
            if len(box) != 4:
                continue
            
            center = det.get('center', ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            
            # Simple: assign new ID to each detection
            # In a real implementation, this would do association
            track_id = self._next_id
            self._next_id += 1
            
            tracked_det = TrackedDetection(
                box=box,
                center=center,
                score=det.get('score', 1.0),
                label=det.get('label', 'unknown'),
                tracking_id=track_id,
                age=0
            )
            tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def reset(self):
        """Reset tracker state."""
        if NORFAIR_AVAILABLE and self._tracker:
            self._tracker = Tracker(
                distance_function=self.distance_function,
                distance_threshold=self.distance_threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay
            )
        else:
            self._next_id = 1
            self._tracks = {}
        
        logger.info("Tracker reset")


class SimpleCentroidTracker:
    """
    Simple centroid-based tracker that doesn't require Norfair.
    Uses IoU and centroid distance for association.
    """
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 max_distance: float = 100.0):
        """
        Initialize the simple tracker.
        
        Args:
            max_disappeared: Frames before track is considered lost
            max_distance: Maximum centroid distance for association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        
        logger.info(f"SimpleCentroidTracker initialized (max_disappeared={max_disappeared}, max_distance={max_distance})")
    
    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedDetection]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of TrackedDetection with stable tracking IDs
        """
        if not detections:
            # Mark all existing tracks as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return []
        
        # Calculate centroids for new detections
        input_centroids = []
        for det in detections:
            box = det.get('box', [0, 0, 0, 0])
            if len(box) == 4:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                input_centroids.append((cx, cy))
        
        # If no existing tracks, register all detections
        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                box = det.get('box', [0, 0, 0, 0])
                if len(box) == 4:
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    self._register(det, center)
        else:
            # Match detections to existing tracks
            object_ids = list(self.objects.keys())
            object_centroids = [obj['center'] for obj in self.objects.values()]
            
            # Compute distance matrix
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            
            # Find minimum distance for each existing track
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id]['center'] = input_centroids[col]
                self.objects[object_id]['box'] = detections[col].get('box', [0, 0, 0, 0])
                self.objects[object_id]['score'] = detections[col].get('score', 1.0)
                self.objects[object_id]['label'] = detections[col].get('label', 'unknown')
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing tracks
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            
            # Handle unmatched new detections
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self._register(detections[col], input_centroids[col])
        
        # Return tracked detections
        tracked_detections = []
        for object_id, obj in self.objects.items():
            tracked_det = TrackedDetection(
                box=obj['box'],
                center=obj['center'],
                score=obj.get('score', 1.0),
                label=obj.get('label', 'unknown'),
                tracking_id=object_id,
                age=self.disappeared.get(object_id, 0)
            )
            tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def _register(self, detection: Dict[str, Any], centroid: tuple):
        """Register a new tracked object."""
        self.objects[self.next_id] = {
            'center': centroid,
            'box': detection.get('box', [0, 0, 0, 0]),
            'score': detection.get('score', 1.0),
            'label': detection.get('label', 'unknown')
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _deregister(self, object_id: int):
        """Deregister a tracked object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _compute_distance_matrix(self, centroids_a: List[tuple], centroids_b: List[tuple]) -> np.ndarray:
        """Compute Euclidean distance matrix between two sets of centroids."""
        D = np.zeros((len(centroids_a), len(centroids_b)), dtype="float")
        
        for i, ca in enumerate(centroids_a):
            for j, cb in enumerate(centroids_b):
                D[i, j] = np.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2)
        
        return D
    
    def reset(self):
        """Reset tracker state."""
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        logger.info("SimpleCentroidTracker reset")


# Factory function for creating trackers
def create_object_tracker(tracker_type: str = "norfair", **kwargs) -> Any:
    """
    Factory function to create object tracker instances.
    
    Args:
        tracker_type: Type of tracker ("norfair", "simple")
        **kwargs: Additional parameters for tracker initialization
        
    Returns:
        An object tracker instance
    """
    if tracker_type == "norfair":
        return NorfairObjectTracker(**kwargs)
    elif tracker_type == "simple":
        return SimpleCentroidTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
