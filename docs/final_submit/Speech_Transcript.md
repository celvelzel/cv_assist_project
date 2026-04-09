# Speech Transcript: CV Assist

[Slide 1: Title Slide]
**Icebreaker & Hook**: Imagine navigating a world where you can't see the objects around you. You know a cup is on the table, but where exactly? How do you reach for it without knocking it over? 
**Speech**: Good morning/afternoon everyone. Today, I am excited to present **CV Assist**, a real-time visual assistance system designed to empower the visually impaired. Our system isn't just a detector; it's a spatial guide that bridges the gap between user intent and physical action using state-of-the-art AI.

[Slide 2: Problem Statement & Motivation]
**Speech**: For the visually impaired, simple tasks like finding a phone or a bottle can be a daily struggle. Existing tools often provide "what" is in the room, but rarely "where" it is in relation to the user's hand. To ensure maximum **Appropriateness** for our target users, we focused on providing intuitive, real-time spatial guidance that mimics a sighted assistant's instructions.

[Slide 3: System Architecture & Pipeline]
**Speech**: Our system architecture is built on a modular pipeline. We take a standard webcam feed and process it through three parallel detectors: OWL-ViT for object detection, MiDaS for depth, and MediaPipe for hand tracking. These inputs are fused by our Guidance Controller, which then communicates with the user via high-quality Text-to-Speech. This decoupled design ensures both **Soundness** and high performance.

[Slide 4: Zero-Shot Open-Vocabulary Detection (OWL-ViT)]
**Speech**: A major technical highlight is our use of **OWL-ViT v2**. Unlike traditional detectors that are limited to a fixed set of classes, our system is **Open-Vocabulary**. This means a user can ask for *anything*—a "blue mug," a "remote control," or a "wallet"—and the system can detect it zero-shot, without any retraining. This flexibility is a core "Excitement" point of our project.

[Slide 5: Monocular Depth Estimation (MiDaS)]
**Speech**: To provide spatial guidance, we need to know how far away things are. We use **MiDaS** for monocular depth estimation. By generating a depth map from a single RGB frame, we can estimate the relative distance between the user's hand and the target. This is a highly **Appropriate** choice as it works with standard webcams, making the system accessible and cost-effective.

[Slide 6: Hand Tracking & Gesture Recognition (MediaPipe)]
**Speech**: Precision is key. We use **MediaPipe** to track 21 hand landmarks in 3D. This allows us to know exactly where the user's palm is. We also recognize gestures: an open hand means the user is searching, while a closed hand indicates a successful grasp. This fine-grained tracking is essential for the **Soundness** of our spatial guidance.

[Slide 7: Hysteresis-Based Spatial Guidance]
**Speech**: Now, let's talk about the "brain" of the system: the **Hysteresis-Based Guidance** algorithm. A common problem in real-time guidance is "instruction jitter"—where the system flips between "left" and "right" rapidly at the boundary. To solve this, we implemented a dual-threshold hysteresis logic. This ensures that instructions are stable and consistent, providing a professional and reliable user experience. This is a prime example of the **Soundness** of our engineering.

[Slide 8: LLM-Vision Assisted Voice Interaction]
**Speech**: Perhaps the most **Exciting** innovation is our **LLM-Vision Multimodal Error Correction**. We use **DeepSeek-v3.2** to enhance voice recognition. If a user says "find the cut" in a noisy room, Whisper might transcribe it literally. However, our system sends that text along with **4 recent camera frames** to the LLM. The LLM "sees" the cup on the table and corrects the command to "find the cup." This multimodal fusion makes the system incredibly robust to real-world speech errors.

[Slide 9: Performance & Optimization]
**Speech**: To ensure a fluid experience, we've optimized the system using FP16 inference and asynchronous processing. The audio pipeline runs in the background, so the video feed never stutters. This technical rigor ensures that the system remains responsive, which is critical for a real-time assistive tool.

[Slide 10: Conclusion & Future Work]
**Speech**: In conclusion, CV Assist is a sound, appropriate, and exciting synthesis of computer vision and large language models. We've moved beyond simple detection to provide true spatial empowerment. In the future, we plan to deploy this on edge devices like smart glasses to further enhance the independence of visually impaired users. Thank you for your time.
