
**Slide 1: Title**
*Duration: 10 seconds*

Today, I am proud to present our real-time visual assistance system designed to empower the visually impaired.

[Click for Next Slide]

**Slide 2: Complete Data Flow Architecture**
*Duration: 10 seconds*

Here is our Complete Data Flow Architecture. This pipeline seamlessly processes webcam input through our logic processing layer, ultimately delivering real-time audio guidance to the user.

[Click for Next Slide]

**Slide 3: Open-Vocabulary Detection**
*Duration: 10 seconds*

For target recognition, we utilize OWL-ViT v2. This open-vocabulary Vision Transformer allows users to search for any object using free-text queries without the limitations of fixed, pre-trained categories.

[Click for Next Slide]

**Slide 4: Accurate Distance Sensing**
*Duration: 10 seconds*

To determine distance, we pair this with MiDaS, a lightweight monocular depth estimation model. This generates per-pixel relative depth maps, completely eliminating the need for expensive LiDAR cameras.

[Click for Next Slide]

**Slide 5: High-Fidelity 3D Hand Landmarks**
*Duration: 10 seconds*

Finally, Google MediaPipe tracks twenty-one 3D hand landmarks in real-time. By calculating the spatial offset between the hand and the object, we achieve high accuracy for spatial alignment.

[Click for Next Slide]

**Slide 6: Spatial Guidance and Control Logic – [Deep Dive]**
*Duration: 55 seconds*

We make a key contribution in the Guidance Controller. Rather than processing every module at the same frame rate, we designed a Dynamic Frame Skipping mechanism.

Our hand tracking runs smoothly at a high frequency of 30 Hz. However, the object detection and depth estimation inferences are running in a low frequency of 5 Hz. It dynamically skips expensive visual inference, which ensures the system maintain high smoothness without overloading standard edge devices.

[Click for Next Slide]

**Slide 7: Voice Interaction & LLM Correction – [Deep Dive / Highlight]**
*Duration: 1 minute*

Our most exciting feature is the LLM-Assisted Keyword Extraction for error correction. In a noisy room, speech recognition may mishear a command, for example turning "find the cup" into "find the mop." When that happens, our system sends both the transcribed text and the four most recent camera frames to a multimodal large language model. With that visual context, the model can correct the command. This multimodal design makes the system tightly aligns with true user intent.

[Click for Next Slide]

**Slide 8: Dual-Threshold Logic – [Deep Dive]**
*Duration: 55 seconds*

Let us now look at the decision algorithm. Real-time guidance often suffers from instruction jitter, where directions rapidly flip between left and right. To solve this, we designed a Dual-Threshold hysteresis strategy, which is similar to a Buffer Zone. Instead of a single strict line, user must move significantly out of the center to trigger a 'left' command. This keeps the audio instructions stable and consistent.
