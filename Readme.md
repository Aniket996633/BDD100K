# 🚗 BDD100K ADAS Object Detection Pipeline

An end-to-end computer vision pipeline developed on **Berkeley DeepDrive (BDD100K)** to detect critical road-user classes for Advanced Driver Assistance Systems (ADAS). 

This project isolates, preprocesses, trains, and evaluates a lightweight, high-speed model targeting **five essential road-user categories**: `car`, `bus`, `truck`, `pedestrian`, and `cyclist`.

---

## 🛠️ Pipeline Architecture & Design Choices

### 1. Data Engineering & Class Mapping
The raw BDD100K labels were mapped and compressed from 10 native classes into 5 unified structural categories to fit an ADAS profile:
*   `pedestrian` ← `person`
*   `cyclist` ← `rider` + `bike`
*   `car`, `bus`, `truck` ← Extracted directly.
*   *Note: Extraneous classes like traffic lights, signs, and trains were programmatically stripped during data preparation.*

### 2. Model Selection Rationale
*   **Architecture:** **YOLOv8 Nano** (Fused: 72 layers, 3,006,623 parameters, 8.1 GFLOPs).
*   **Reasoning:** Autonomous driving systems require extreme real-time execution. YOLOv8n delivers state-of-the-art edge performance, processing frames at an average inference speed of **2.2 ms** on an NVIDIA Tesla T4 GPU (~243 FPS capability), matching strict low-latency embedded safety profiles.

### 3. Targeting Pipeline Weaknesses
Our baseline experiments revealed a standard driving-data issue: poor detection of Vulnerable Road Users (VRUs) due to severe class imbalance and far-away, tiny pixel profiles. To fix this, we implemented structural modifications:
*   `scale=0.6`: Enhanced spatial scaling augmentations to force the model to learn small object features.
*   `box=10.5` & `cls=2.0`: Increased bounding box and classification loss penalties to enforce stricter anchor alignment.

---

## 📊 Quantitative Evaluation Results

The final enhanced pipeline was validated using **2,000 completely unseen road images** (comprising 24,553 independent object annotations):

### Speed Performance Metrics

| Stage | Speed Per Image (ms) |
| :--- | :--- |
| **Preprocessing** | 0.7 ms |
| **Inference Latency** | 2.2 ms |
| **Postprocessing** | 1.2 ms |
| **Total Pipeline Latency** | **4.1 ms** |

### Object Detection Performance (mAP@50)

| Class | Labeled Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Combined** | 24,553 | 0.567 | 0.415 | **0.438** | 0.258 |
| **Car 🚗** | 20,289 | 0.668 | 0.637 | **0.670** | 0.398 |
| **Truck 🚛** | 872 | 0.571 | 0.446 | **0.472** | 0.310 |
| **Pedestrian 🚶‍♂️** | 2,748 | 0.655 | 0.376 | **0.437** | 0.199 |
| **Bus 🚌** | 311 | 0.441 | 0.396 | **0.382** | 0.279 |
| **Cyclist 🚴‍♂️** | 333 | 0.498 | 0.221 | **0.229** | 0.105 |

---

## 🔍 Failure & Scenario Slicing Analysis

Despite structural pipeline improvements, a deep dive into the output charts (found in your results folder) highlights critical system limitations:

1. **The Cyclist Edge Case (Low Recall: 0.221):** Cyclists make up less than 1.5% of the validation targets. Their dynamic spatial configurations change drastically between side profiles and rear views, causing the model to completely miss them (high false negatives).
2. **Heavy Vehicle Classification Ambiguity:** The lower precision metric for buses (`0.441`) indicates a semantic domain overlap. Under poor lighting, rainy conditions, or partial occlusions, the model routinely confuses municipal passenger buses with commercial box trucks.

---


