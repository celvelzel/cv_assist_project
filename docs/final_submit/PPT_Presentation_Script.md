# PPT Presentation Script: CV Assist

[Slide 1: Title Slide]
Visual Suggestions: High-resolution image of the CV Assist system in action, showing a hand being guided to a cup with bounding boxes and depth maps overlaid.
Core Points:
- **CV Assist**: A Real-Time Visual Assistance System for the Visually Impaired.
- Developed by: [Your Name/Team].
- Key Technologies: Open-Vocabulary Detection, Monocular Depth, Hand Tracking, and LLM-Vision.

[Slide 2: Problem Statement & Motivation]
Visual Suggestions: A blurred image representing a visually impaired person's perspective, contrasted with a clear image of a target object.
Core Points:
- **Challenge**: Independent object localization and grasping for the visually impaired.
- **Gap**: Existing tools lack real-time, intuitive spatial guidance.
- **Goal**: Provide a "virtual eye" that guides the user's hand to any object using natural language.

[Slide 3: System Architecture & Pipeline]
Visual Suggestions: The architecture diagram from the README (Webcam -> OWL-ViT/MiDaS/MediaPipe -> Guidance Controller -> TTS).
Core Points:
- **Modular Pipeline**: Decoupled detectors for flexibility and performance.
- **Multimodal Input**: RGB Video + Voice Commands.
- **Real-Time Output**: Spatial audio instructions + Visual feedback.

[Slide 4: Zero-Shot Open-Vocabulary Detection (OWL-ViT)]
Visual Suggestions: Screenshots of the system detecting various objects (cup, bottle, phone) without retraining.
Core Points:
- **Innovation**: Using **OWL-ViT v2** for zero-shot detection.
- **Benefit**: Search for *any* object by name (Open-Vocabulary).
- **Soundness**: Robust detection across diverse environments.

[Slide 5: Monocular Depth Estimation (MiDaS)]
Visual Suggestions: A side-by-side comparison of an RGB frame and its corresponding MiDaS depth map (heatmap).
Core Points:
- **Technology**: **MiDaS_small** for monocular depth estimation.
- **Function**: Estimates distance to target and hand from a single camera.
- **Appropriateness**: Eliminates the need for expensive stereo cameras or LiDAR.

[Slide 6: Hand Tracking & Gesture Recognition (MediaPipe)]
Visual Suggestions: Image showing MediaPipe hand landmarks (21 points) and gesture labels (Open/Closed).
Core Points:
- **Tracking**: **MediaPipe** for high-fidelity 3D hand landmarks.
- **Gestures**: Detects "Open" (ready) and "Closed" (grabbed) states.
- **Precision**: Enables fine-grained spatial alignment between hand and target.

[Slide 7: Hysteresis-Based Spatial Guidance]
Visual Suggestions: A diagram showing the "Enter" and "Exit" thresholds of the hysteresis algorithm to explain stability.
Core Points:
- **Technical Highlight**: **Hysteresis-Based Guidance** algorithm.
- **Problem**: Instruction jitter at alignment boundaries.
- **Solution**: Dual-threshold logic prevents confusing oscillations.
- **Soundness**: Ensures stable, professional-grade audio instructions.

[Slide 8: LLM-Vision Assisted Voice Interaction]
Visual Suggestions: A flowchart showing Whisper ASR output -> LLM + 4 Frames -> Corrected Target.
Core Points:
- **Excitement Point**: **LLM-Vision Multimodal Error Correction**.
- **Model**: **DeepSeek-v3.2** via Poe API.
- **Context**: Uses 4 recent camera frames to correct ASR errors (e.g., "cut" -> "cup").
- **Robustness**: Handles accents, noise, and unclear speech.

[Slide 9: Performance & Optimization]
Visual Suggestions: A graph showing stable FPS and low latency for the LLM-Vision background processing.
Core Points:
- **Optimization**: FP16 inference and intelligent frame-skipping.
- **Concurrency**: Asynchronous audio pipeline prevents UI blocking.
- **Efficiency**: Runs on consumer-grade hardware without specialized sensors.

[Slide 10: Conclusion & Future Work]
Visual Suggestions: A photo of a user successfully grasping an object using the system.
Core Points:
- **Impact**: Significant improvement in independence for visually impaired users.
- **Summary**: A sound, appropriate, and exciting synthesis of CV and LLMs.
- **Future**: Expanding gesture controls and edge-device deployment.
