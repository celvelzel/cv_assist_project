# PPT Presentation Script: CV Assist

[Slide 1: Title Slide]
Visual Suggestions: High-resolution image of the CV Assist system in action, showing a hand being guided to a cup with bounding boxes and depth maps overlaid.
Core Points:
- **CV Assist**: A Real-Time Visual Assistance System for the Visually Impaired.
- Developed by: [Your Name/Team].
- Key Technologies: Open-Vocabulary Detection, Monocular Depth, Hand Tracking, and LLM-Vision.

[Slide 2: System Architecture & Soundness]
Visual Suggestions: The architecture diagram showing the modular pipeline (Webcam -> OWL-ViT/MiDaS/MediaPipe -> Guidance Controller -> TTS).
Core Points:
- **Soundness (Rubric Focus)**: A comprehensive, well-organized development process integrating state-of-the-art vision models.
- **Cost-Effective Design**: Uses a single RGB webcam (via MiDaS monocular depth) instead of expensive LiDAR or stereo cameras.
- **Robust Pipeline**: Decoupled detectors ensure logical data flow from environment capture to spatial audio instructions.

[Slide 4: Zero-Shot Open-Vocabulary Detection (Excitement/Innovation)]
Visual Suggestions: Screenshots of the system detecting various objects (cup, bottle, phone) without retraining.
Core Points:
- **Innovation (Rubric Focus)**: Using **OWL-ViT v2** for zero-shot detection.
- **Impact**: Users can search for *any* object by natural language name (e.g., "find the blue mug") without being limited to a predefined list.
- **Result**: Consistently captures attention by demonstrating high flexibility and intelligence.

[Slide 5: Soundness: Hysteresis-Based Spatial Guidance]
Visual Suggestions: A diagram showing the "Enter" and "Exit" thresholds of the hysteresis algorithm.
Core Points:
- **Technical Highlight**: **Hysteresis-Based Guidance** algorithm within the Guidance Controller.
- **The Problem**: Instruction jitter ("move left", "move right" oscillating at boundaries).
- **The Solution**: Dual-threshold logic creates a stable buffer zone, providing clear, professional-grade audio instructions without confusing oscillations.

[Slide 6: Excitement/Innovation: LLM-Vision Error Correction]
Visual Suggestions: A flowchart showing Whisper ASR output -> DeepSeek-v3.2 + 4 Recent Frames -> Corrected Target.
Core Points:
- **Major Innovation**: **LLM-Vision Multimodal Error Correction**.
- **How it Works**: If ASR mishears "find the cut", the system sends the text plus 4 recent camera frames to an LLM to "see" the environment and correct the intent to "find the cup".
- **Impact**: Highly robust to noisy environments and accents, providing an exciting and novel application of multimodal AI.

[Slide 7: Soundness: Real-Time Performance Optimizations]
Visual Suggestions: A graph showing stable FPS and low latency, or a bulleted list of optimization techniques.
Core Points:
- **Real-Time Engineering**: Heavy optimization ensures the system is actually usable in real life.
- **Techniques**: FP16 Half-Precision inference (~50% memory reduction), strategic frame skipping (detection and depth run every 3 frames while hand tracking runs every frame).
- **Asynchronous Design**: Voice input, Text-to-Speech (TTS), and background LLM correction run asynchronously to prevent UI blocking.

[Slide 8: Conclusion & Live Demonstration]
Visual Suggestions: A photo or video clip of a user successfully grasping an object using the system.
Core Points:
- **Summary**: CV Assist represents a highly appropriate, technically sound, and exciting solution for the visually impaired.
- **Demonstration**: [Transition smoothly into the live demo, showing polished and professional execution].
- **Future Impact**: Expanding to edge devices (e.g., smart glasses) for total independence.
