# core/interfaces.py
"""
Abstract interfaces for extensible components.
These Protocol classes define contracts that allow future implementations
to be swapped without modifying the core system logic.
"""

from typing import Protocol, NamedTuple, List, Optional, Any, Literal


class ParsedIntent(NamedTuple):
    """Structured representation of user's spatial intent from ASR."""
    target_class: str          # e.g., "杯子" or "cup"
    spatial_modifier: Optional[str]  # e.g., "left", "right", "中间", or None
    raw_text: str              # original ASR text


class Detection(Protocol):
    """Protocol for a detection object with spatial coordinates."""
    box: List[float]  # [x1, y1, x2, y2]
    center: tuple     # (x, y)
    score: float
    label: str
    tracking_id: Optional[int]


class TrackedDetection(NamedTuple):
    """Standardized tracked detection with stable ID."""
    box: List[float]           # [x1, y1, x2, y2]
    center: tuple              # (x, y)
    score: float
    label: str
    tracking_id: int           # stable tracking ID from SORT
    age: int = 0               # frames since last seen


class IIntentParser(Protocol):
    """
    Interface for extracting structured intent from raw ASR text.
    
    Implementations:
    - RegexIntentParser: rule-based pattern matching (current)
    - LLMIntentParser: future LLM-based semantic extraction
    """
    
    def parse(self, text: str) -> Optional[ParsedIntent]:
        """
        Parse raw ASR text into structured intent.
        
        Args:
            text: Raw speech transcription (e.g., "找到左边的杯子")
            
        Returns:
            ParsedIntent with target class and spatial modifier, or None if unparseable.
        """
        ...


class ISpatialResolver(Protocol):
    """
    Interface for mapping spatial modifiers to specific tracked objects.
    
    Implementations:
    - StandardSpatialResolver: image-left = user-left (default)
    - InvertedSpatialResolver: image-left = user-right (front-facing camera)
    """
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """
        Select a specific tracked object based on spatial modifier.
        
        Args:
            spatial_modifier: Spatial term (e.g., "left", "right", "nearest")
            tracks: List of currently tracked detections for the target class
            reference_point: Optional (x,y) for relative calculations
            
        Returns:
            Selected TrackedDetection, or None if no match found.
        """
        ...


# Configuration type for camera orientation
CameraFacing = Literal["outward", "user"]
