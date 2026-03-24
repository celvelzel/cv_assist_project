# Spatial Target Selection & SORT Tracking Plan

## Goal
Implement multi-object tracking (SORT) and spatial selection ("left cup" vs "right cup") with a future-proof, decoupled architecture. 
This relies on three key patterns:
1. `IIntentParser`: Extracts object and spatial modifiers from ASR text (allowing Regex now, LLM later).
2. `ISpatialResolver`: Handles coordinate mapping and camera view inversion (Outward vs User-facing).
3. `ObjectTracker` (Norfair): Assigns stable tracking IDs to avoid target jitter across frames.

## Core Directives
- Apply TDD for all mathematical/parsing logic.
- NO hardcoded dependency instantiations inside `system.py` — they must be injected.
- Install `norfair` as the lightweight tracking dependency.

## Tasks

### Wave 1: Foundation & Dependencies
- [x] `requirements.txt`: Add `norfair` to dependencies.
- [x] `core/interfaces.py`: Define `ParsedIntent(NamedTuple)`, `IIntentParser(Protocol)`, and `ISpatialResolver(Protocol)`.
- [x] `core/interfaces.py`: Add `camera_facing: Literal["outward", "user"]` to configuration.
- [x] `tests/test_interfaces.py`: Write structural tests for mock implementations of these interfaces.

### Wave 2: Implementations (TDD)
- [x] `audio/intent_parser.py`: Implement `RegexIntentParser` adhering to `IIntentParser`.
- [x] `tests/test_intent_parser.py`: Verify 10+ phrasings (e.g. "找到左边的杯子", "find the cup on the right").
- [x] `core/spatial_resolver.py`: Implement `StandardSpatialResolver` and `InvertedSpatialResolver` (User-centric mirror).
- [x] `tests/test_spatial_resolver.py`: Verify box selection logic in both outward and user modes.
- [x] `core/tracker.py`: Create `ObjectTracker` wrapper around `norfair.Tracker`.
- [x] `tests/test_tracker.py`: Verify SORT ID assignment using mock `[x1, y1, x2, y2]` inputs.

### Wave 3: Integration
- [x] `audio/asr.py`: Refactor `parse_command()` to use injected `IIntentParser`.
- [x] `core/guidance.py`: Add `tracking_id: Optional[int]` to `GuidanceResult`.
- [x] `core/system.py`: Remove `detections[0]` hardcoding.
- [x] `core/system.py`: Inject `ObjectTracker` and `ISpatialResolver` into `process_frame()` to select the specific target.
- [x] `core/system.py`: Pass `tracking_id` to drawing logic to display bounding box IDs.

### Wave 4: Verification
- [x] Run full `pytest` suite ensuring all mocked E2E flows pass.
- [x] Commit: `feat: implement multi-object tracking and spatial target selection`
