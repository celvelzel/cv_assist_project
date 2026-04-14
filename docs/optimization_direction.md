# Optimization Directions

This document organizes the next optimization directions into a consistent Markdown structure. It focuses on computation efficiency, interaction capability, model lightweighting, tracking, reasoning, and safety.

## 1. Adaptive Computation Pipeline

Dynamic model switching that automatically selects between OWL-ViT base/large variants or lightweight alternatives such as MobileViT based on real-time scene complexity and available GPU/CPU resources to maintain the target FPS.

## 2. Multi-Gesture and Interaction Intent Recognition

### Current State

Only recognizes two basic gestures (open/closed), so the interaction modality is limited.

### Optimization Direction

Define compound gesture sequences such as:

- Grasp preparation pose
- Arm rotation search pattern
- Fine alignment adjustment

## 3. Cross-Modal Knowledge Distillation and Lightweight Model Family

### Current State

Depends on larger models such as OWL-ViT-base and MiDaS, which makes edge-device deployment difficult.

### Optimization Direction

#### Teacher-Student Model Collaboration

- Offline pre-train a large teacher model, for example OWL-ViT-large plus high-resolution MiDaS.
- Distill knowledge into lightweight student models, such as MobileViT or DistilBERT for vision-language tasks.
- Run student models on edge devices such as Raspberry Pi or mobile devices, and call the cloud teacher for high-precision requirements.

#### Adaptive Precision Scheduling

- Dynamically select model size and input resolution according to device computational capacity.
- Balance real-time performance and accuracy.

## 4. Object Tracking Algorithm: Multi-Object Tracking (MOT)

### Current State

Each frame is independently detected. After an object disappears, the target must be set again via voice.

### Innovation Direction

Integrate DeepSORT or OC-SORT for persistent object tracking.

### Technical Advantage

Automatically re-identifies lost targets without repeated voice commands.

### Functionality Enhancement

Supports continuous interactions such as “continue finding that cup”.

### Implementation

- Add a new object_tracker.py module.
- Enable enable_tracking: true in config.

## 5. Scene Understanding and Semantic Reasoning: LLM Scene Analysis

### Current State

Rule-based target queries such as “a cup”.

### Innovation Direction

Integrate a lightweight LLM for scene reasoning.

### Technical Advantage

- “Find my usual cup” combines user historical habits for inference.
- “Get things from the table” understands spatial relationships.
- “Find the closest one” automatically selects the nearest target among multiple candidates.

### Functionality Enhancement

- Enhanced natural language understanding, such as “next to the book” or “the red one”.
- Scene awareness that adapts to study rooms, kitchens, and living rooms.

### Implementation

- Extend scene reasoning prompts in llm_vision.py.
- Add local LLM support, for example TinyLlama.

## 6. Obstacle Detection and Avoidance: Safety Enhancement

### Current State

Only guides movement toward the target, without obstacle awareness.

### Innovation Direction

Integrate real-time obstacle detection.

### Technical Advantage

Detects common obstacles such as table corners, chairs, and pets.

### Functionality Enhancement

- Voice announcement: “obstacle ahead”.
- Automatically guides the path to avoid dangerous areas.