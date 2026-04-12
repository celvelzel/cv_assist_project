# Speech Transcript: CV Assist

[Slide 1: Title Slide]
**Icebreaker & Hook**: Imagine navigating a world where you can't see the objects around you. You know a cup is on the table, but where exactly? 
**Speech**: Good morning. I am excited to present **CV Assist**, a real-time visual assistance system. Our system isn't just a detector; it's a spatial guide bridging user intent and physical action.We designed a system highly relevant to the daily challenge of grasping objects, providing intuitive, real-time guidance that acts as a sighted assistant.

[Slide 2: System Architecture & Soundness]
**Speech**: We built a modular pipeline taking a standard webcam feed and processing it through parallel models: OWL-ViT for detection, MiDaS for monocular depth, and MediaPipe for hand tracking. This well-organized architecture replaces expensive LiDAR setups with a cost-effective, logical flow from pixel to spatial audio.
These inputs are fused by our Guidance Controller, which then communicates with the user via high-quality Text-to-Speech.

[Slide 4: Zero-Shot Open-Vocabulary Detection (Excitement/Innovation)]
**Speech**: A key point of **Excitement and Innovation** is our use of OWL-ViT v2. Traditional detectors are limited to fixed classes. Our system is **Open-Vocabulary**. A user can ask for *anything*—a "blue mug" or a "wallet"—and the system detects it zero-shot. It's incredibly engaging and flexible.
[Slide 5: Monocular Depth Estimation (MiDaS)]
**Speech**: To provide spatial guidance, we need to know how far away things are. We use **MiDaS** for monocular depth estimation. By generating a depth map from a single RGB frame, we can estimate the relative distance between the user's hand and the target.

[Slide 6: Hand Tracking & Gesture Recognition (MediaPipe)]
**Speech**: We use **MediaPipe** to track 21 hand landmarks in 3D. This allows us to know exactly where the user's palm is. We also recognize gestures: an open hand means the user is searching, while a closed hand indicates a successful grasp.

[Slide 7: Excitement/Innovation: LLM-Vision Error Correction]
**Speech**: Our most **Exciting** feature is the **LLM-Vision Multimodal Error Correction**. In a noisy room, speech recognition might transcribe "find the cup" as "find the mop". Our system intercepts this, sending the text and 4 recent camera frames to an LLM. The LLM "sees" the cup and corrects the command.This multimodal fusion makes the system incredibly robust to real-world speech errors.
 
[Slide 8: Hysteresis-Based Spatial Guidance]
**Speech**: Let's look at the guidance algorithm. Real-time guidance often suffers from "instruction jitter"—rapidly flipping between "left" and "right". We implemented a dual-threshold hysteresis logic to solve this, ensuring stable, consistent audio instructions without confusing oscillations. 



[Slide 9: Soundness: Real-Time Performance Optimizations]
**Speech**: To ensure our solution is practical, we applied rigorous performance optimizations. We use FP16 half-precision inference cutting memory use by 50%, run Voice and TTS asynchronously, and intelligently skip frames—running detection every 3 frames while keeping hand tracking real-time. This logical approach keeps the system highly responsive.

[Slide 10: Conclusion & Live Demonstration]
**Speech**: CV Assist is a sound, highly appropriate, and exciting synthesis of computer vision and language models. We have moved beyond simple detection to provide true spatial empowerment. We will now transition to a live demonstration to showcase the system's effectiveness. Thank you.
