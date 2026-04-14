# Speech Transcript: CV Assist

[Slide 1: Title Slide]
**Speech**: Good morning. Today, I am pleased to present CV Assist, a real-time visual assistance. We designed it around a very practical daily challenge: helping users grasp objects more safely and confidently by providing guidance that works like a sighted assistant.

[Slide 2: System Architecture Overview]
**Speech**: Our system uses a modular pipeline built on a standard webcam feed. In parallel, we run OWL-ViT for object detection, MiDaS for monocular depth estimation, and MediaPipe for hand tracking. Together, these components give us a cost-effective alternative to expensive LiDAR-based solutions, turning visual input into spatial audio guidance.
All of these signals are then fused by our Guidance Controller, which delivers clear instructions to the user through high-quality text-to-speech.

[Slide 4: Zero-Shot Open-Vocabulary Detection (Excitement/Innovation)]
**Speech**: One of the most exciting parts of our project is the use of OWL-ViT v2. Traditional detectors are limited to a fixed set of classes, but our system is open-vocabulary. That means the user can ask for almost anything, such as a blue mug or a wallet, and the system can detect it zero-shot. This makes the interaction much more flexible and much more natural.
[Slide 5: Monocular Depth Estimation (MiDaS)]
**Speech**: To provide spatial guidance, we also need to estimate distance. For that, we use MiDaS for monocular depth estimation. From a single RGB frame, it generates a depth map, which allows us to estimate the relative distance between the user's hand and the target object.

[Slide 6: Hand Tracking & Gesture Recognition (MediaPipe)]
**Speech**: We use MediaPipe to track 21 hand landmarks in 3D, so we can locate the user's palm very precisely. On top of that, we recognize simple gestures as part of the interaction design. An open hand means the user is still searching, while a closed hand indicates that the object has been successfully grasped.

[Slide 7: Excitement/Innovation: LLM-Vision Error Correction]
**Speech**: Our most exciting feature is the LLM-Vision multimodal error correction module. In a noisy room, speech recognition may mishear a command, for example turning "find the cup" into "find the mop." When that happens, our system sends both the transcribed text and the four most recent camera frames to a multimodal large language model. With that visual context, the model can identify the cup and correct the command. This multimodal design makes the system much more robust to real-world speech errors.
 
[Slide 8: Hysteresis-Based Spatial Guidance]
**Speech**: Let us now look at the guidance algorithm. Real-time guidance often suffers from instruction jitter, where directions rapidly flip between left and right. To solve this, we implemented a dual-threshold hysteresis strategy. This keeps the audio instructions stable and consistent, and avoids confusing oscillations for the user.



[Slide 9: Soundness: Real-Time Performance Optimizations]
**Speech**: To make sure the system is practical, we also focused on performance optimization. We use FP16 half-precision inference to reduce memory usage by about 50 percent, run voice processing and TTS asynchronously, and intelligently skip frames by running detection every three frames while keeping hand tracking real-time. These optimizations help the system stay responsive during live use.

[Slide 10: Conclusion & Live Demonstration]
**Speech**: In conclusion, CV Assist brings together computer vision and language models into a practical and meaningful assistive system. We go beyond simple detection to provide real spatial guidance for everyday tasks. Next, we will move on to a live demonstration to show the system in action. Thank you.
