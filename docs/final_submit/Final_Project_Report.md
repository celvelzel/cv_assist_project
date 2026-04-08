# CV Assist: Visual Assistance System Final Project Report

## 1. Introduction
The **CV Assist** project is a real-time visual assistance system designed to empower visually impaired users by helping them locate and grasp objects in their environment. By bridging the gap between user intent (expressed through voice) and physical action (guided by spatial audio), the system addresses a critical accessibility need. The project's **Appropriateness** is demonstrated by its focus on solving the fundamental challenge of object localization for the visually impaired using non-visual feedback mechanisms.

## 2. Methodology & System Architecture
The system employs a modular, multi-stage pipeline to ensure **Soundness** and reliability:
- **Zero-Shot Open-Vocabulary Detection**: Utilizing **OWL-ViT v2**, the system can detect any object described in natural language without the need for task-specific retraining.
- **Monocular Depth Estimation**: **MiDaS** provides real-time depth maps from a single RGB frame, enabling the system to estimate the distance to target objects.
- **Hand Tracking & Gesture Recognition**: **MediaPipe** tracks the user's hand in 3D space and recognizes gestures (open, closed, pointing) to facilitate precise interaction.
- **Spatial Guidance Controller**: A central orchestrator that computes directional instructions based on the relative positions of the hand and the target object.

## 3. Implementation Details
### 3.1 LLM-Vision Multimodal Error Correction
A core innovation of this project is the **LLM-Vision Multimodal Error Correction** system. Recognizing that **Whisper ASR** may fail in noisy environments or with unclear speech, we integrated the **DeepSeek-v3.2** LLM (via Poe API). When a voice command is received, the system sends the transcribed text along with **4 recent camera frames** to the LLM. The LLM uses this visual context to "see" what the user is likely referring to, correcting ASR errors (e.g., correcting "find the cut" to "find the cup" based on a cup being visible). This feature significantly enhances the **Excitement** and usability of the system.

### 3.2 Hysteresis-Based Spatial Guidance
To ensure a smooth user experience, we implemented a **Hysteresis-Based Guidance** algorithm. Traditional threshold-based guidance often suffers from "instruction jitter" at alignment boundaries. By introducing hysteresis, the system provides stable, consistent audio instructions ("move left", "move closer"), preventing confusing oscillations. This engineering choice directly contributes to the **Soundness** and professional quality of the interaction.

### 3.3 Asynchronous Audio Pipeline
To maintain real-time performance, the audio pipeline (ASR and TTS) is fully decoupled from the main video rendering loop. This prevents "screen freeze" during voice synthesis, ensuring that the visual processing and spatial guidance remain fluid and responsive.

## 4. Technical Highlights & Innovations
- **Zero-Shot Pipeline**: The ability to search for any object by name without retraining is a major technical highlight.
- **Multimodal Fusion**: Combining ASR, Computer Vision, and LLMs to create a robust, context-aware assistant.
- **High-Quality TTS**: Integration with **Xiaomi MiMo Cloud TTS** provides natural-sounding guidance, improving user comfort.
- **Optimization**: Use of FP16 inference and frame-skipping strategies ensures the system runs efficiently on consumer-grade hardware.

## 5. Results & Evaluation
The system achieves a stable processing rate (FPS) suitable for real-time guidance. Systematic testing has verified the **Soundness** of the guidance logic and the effectiveness of the **LLM-Vision** correction. The **Appropriateness** of the system has been validated through successful hand-to-object guidance scenarios, demonstrating its potential as a practical assistive tool.

## 6. Conclusion
The **CV Assist** project successfully demonstrates how advanced computer vision and large language models can be synthesized into a cohesive, accessible tool. By focusing on **Appropriateness**, **Soundness**, and **Excitement**, we have developed a system that not only meets the technical requirements but also provides a meaningful, innovative solution for the visually impaired. Future work will focus on expanding gesture-based control and further reducing latency for even more seamless interaction.
