# Plan: Physical Obstacle Avoidance via Depth Trajectory

## Context
Implement a safety feature that predicts hand trajectory, samples relative MiDaS depth ahead of the hand, and issues a preemptive, high-priority TTS warning by flushing the standard audio queue.

## Goal
Implement physical obstacle avoidance for visually impaired users by detecting protruding high points along the user's hand trajectory using MiDaS depth maps and providing high-priority TTS warnings.

## Scope Boundaries
- **IN**: Tracking hand movement history (trajectory), predicting future hand position based on velocity vector, sampling depth ahead of the hand to detect obstacles, issuing high-priority TTS interruptions via a Queue Flush mechanism.
- **OUT**: Converting relative depth to absolute metric depth (using relative thresholding instead), modifying the core guidance logic for target finding. Hard-stopping the audio thread itself.

## Technical Approach
1. **Hand Trajectory Tracking**: 
   - Add a historical buffer (e.g., `collections.deque`) in `system.py` or a dedicated class to store the last N hand centers.
   - Calculate a velocity vector representing movement direction and speed.
2. **Obstacle Detection (Relative Depth)**:
   - Create an `ObstacleDetector` class or add logic to `depth_estimator.py`.
   - Sample depth values along the predicted trajectory vector (ahead of the current hand position).
   - Compare sampled depth values against the hand's current depth using a configurable relative threshold (e.g., detecting sudden decreases in depth values which indicate objects closer to the camera).
3. **High Priority Audio Warning (Queue Flush)**:
   - Modify the TTS backend (`audio/tts/base.py`, `pyttsx3_backend.py`, `mimo_backend.py`) to support a `clear_queue()` method.
   - In `system.py`, check for obstacles. If detected, call `clear_queue()` to drop pending guidance audio and immediately queue the warning (e.g., "前方有障碍物" / "Obstacle ahead").
   - Add a throttling mechanism specific to obstacle warnings to prevent spamming.

## Test Strategy
- Unit tests for the trajectory calculation (vector math) and boundary clamping.
- Unit tests for depth thresholding logic using mock depth maps.
- Unit tests for the high-priority TTS interruption mechanism and queue flushing.
- Integration test for the main loop with debounce logic.

## Constraints & Config
- Need new config parameters in `config.yaml`:
  - `obstacle_trajectory_frames`: Number of frames to keep for history.
  - `obstacle_prediction_distance`: How far ahead (pixels or factor of velocity) to check.
  - `obstacle_depth_threshold`: The relative depth difference considered an obstacle.
  - `obstacle_warning_cooldown`: Throttling interval for warnings.

## Tasks

### Wave 1 (Start Immediately - No Dependencies)

- [ ] **1. Audio Queue Flush Mechanism**
  - **What**: Implement `clear_queue()` in the audio thread/engine (`audio/tts/base.py`, `pyttsx3_backend.py`, `mimo_backend.py`) to safely empty pending TTS tasks.
  - **Category**: `unspecified-high`
  - **Skills**: `test-driven-development`
  - **QA**: Run pytest on the audio engine queue logic; verify queue length goes to 0 safely.

- [ ] **2. Hand History Buffer**
  - **What**: Implement a `collections.deque` wrapper to track the last N `(x, y)` hand coordinates (e.g. in `system.py` or a helper class).
  - **Category**: `quick`
  - **Skills**: `test-driven-development`
  - **QA**: Run pytest to verify buffer eviction and coordinate retrieval.

### Wave 2 (After Wave 1 Completes)

- [ ] **3. Trajectory & Depth Sampling**
  - **What**: Build logic (in `depth_estimator.py` or new `ObstacleDetector`) to average historical points, create a movement vector, project it forward, clamp to `[0, width]` and `[0, height]`, and extract relative depth from the MiDaS array. Compare with current hand depth.
  - **Category**: `unspecified-high`
  - **Skills**: `test-driven-development`
  - **QA**: Run pytest on math functions using dummy 2D NumPy arrays to ensure no `IndexError` occurs and thresholds trigger correctly.

### Wave 3 (After Wave 2 Completes)

- [ ] **4. Integration & Debounce**
  - **What**: Wire trajectory sampling into the main frame loop (`system.py`). Add a timestamp-based debounce to prevent spam. Trigger the queue flush and TTS warning when an obstacle is detected.
  - **Category**: `unspecified-high`
  - **Skills**: `test-driven-development`
  - **QA**: Run integration tests simulating rapid consecutive frame detections to ensure the debounce prevents multiple TTS calls and properly clears the queue.

## Final Verification Wave

- [ ] Wait for user to explicitly say "okay" or "done" before concluding the task. DO NOT run any final verification commands or mark the task as complete until the user confirms the results are satisfactory.