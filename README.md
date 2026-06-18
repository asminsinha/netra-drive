# Netra-Drive AI Engine

A telemetry-driven driver monitoring system designed for real-time fatigue detection, attention analysis, microsleep identification, and fleet safety reporting.

Netra-Drive combines computer vision, adaptive statistical modeling, deep learning, asynchronous backend processing, and real-time web visualization to generate structured driver safety insights from live video streams.

The platform consists of three primary components:

* A Deep Learning Training Pipeline built with PyTorch
* A FastAPI-based Real-Time Inference and Telemetry Engine
* A Next.js Real-Time Monitoring Dashboard

---

## Table of Contents

1. Overview
2. Problem Statement
3. System Objectives
4. High-Level Architecture
5. Core Technologies
6. Deep Learning Training Pipeline
7. Dataset Processing Strategy
8. Multi-Task Learning Architecture
9. Spatial Vision Network
10. Temporal Sequence Modeling
11. Backend Inference Engine
12. Driver Risk Assessment Engine
13. Adaptive Baseline Calibration
14. Fatigue Detection Methodology
15. Attention Tracking Methodology
16. Explainable Rule-Based Safety Layer
17. Real-Time Backend Architecture
18. API Endpoints
19. WebSocket Telemetry Pipeline
20. Frontend Dashboard
21. Telemetry Visualization
22. PDF Report Generation
23. Project Structure
24. Containerization
25. Local Development Setup
26. Deployment Notes
27. Future Improvements
28. License

---

# Overview

Netra-Drive is a driver monitoring platform that analyzes facial behavior and driving attention indicators from live camera feeds.

The system continuously processes:

* Eye closure patterns
* Mouth movement patterns
* Head pose changes
* Gaze direction
* Driver attentiveness
* Fatigue accumulation
* Microsleep events

These signals are converted into structured telemetry data and streamed to a monitoring dashboard through a low-latency WebSocket pipeline.

The objective is to provide interpretable safety metrics rather than black-box predictions.

---

# Problem Statement

Driver fatigue remains one of the most significant contributors to transportation incidents.

Many conventional monitoring systems rely on:

* Fixed thresholds
* Static geometric assumptions
* Hardware-specific calibration
* Limited explainability
* Isolated device-side processing

Such approaches often struggle when:

* Drivers have different facial structures
* Camera placement changes
* Cabin lighting varies
* Long-term fatigue develops gradually
* Human operators require auditable reports

---

# System Objectives

Netra-Drive was designed with the following goals:

* Real-time operation
* Low-latency inference
* Driver-specific calibration
* Explainable risk assessment
* Continuous telemetry streaming
* Fleet compliance reporting
* Containerized deployment
* Cloud compatibility

---

# High-Level Architecture

```text
Camera Source
     │
     ▼
Frame Acquisition Layer
     │
     ▼
MediaPipe Face Mesh
     │
     ▼
Feature Extraction
(EAR, MAR, Pose, Gaze)
     │
     ▼
Adaptive Risk Engine
     │
     ▼
Telemetry State Store
     │
     ▼
WebSocket Broadcast Layer
     │
     ▼
Next.js Monitoring Dashboard
     │
     ▼
Analytics & PDF Reporting
```

---

# Core Technologies

## Backend

* Python 3.10
* FastAPI
* Uvicorn
* OpenCV
* MediaPipe
* NumPy
* PyTorch
* asyncio
* WebSockets

## Frontend

* Next.js
* React
* TypeScript
* Recharts
* jsPDF
* Tailwind CSS

## Deep Learning

* PyTorch
* MobileNetV3-Small
* Bidirectional GRU
* BCEWithLogitsLoss

## Deployment

* Docker
* Hugging Face Spaces
* Linux Containers

---

# Deep Learning Training Pipeline

The training subsystem supports multiple computer vision tasks simultaneously while handling partially labeled datasets.

The architecture is divided into:

1. Dataset Preparation
2. Multi-Task Classification Network
3. Spatial Vision Network
4. Temporal Sequence Network

---

# Dataset Processing Strategy

Many publicly available fatigue datasets are incomplete.

Examples:

* Eye-state datasets contain no fatigue labels.
* Yawn datasets contain no eye-state labels.
* Fatigue datasets contain no mouth-state labels.

To avoid discarding data, Netra-Drive uses a masked multi-task learning strategy.

## Dataset Splits

```text
Training      : 70%
Validation    : 15%
Testing       : 15%
```

## Classification Tasks

### Eye State

```text
0 → Open
1 → Closed
```

### Yawn Detection

```text
0 → No Yawn
1 → Yawn
```

### Fatigue Detection

```text
0 → Alert
1 → Fatigued
```

---

# Multi-Task Learning Architecture

## Shared Backbone

The primary model uses:

```text
MobileNetV3-Small
```

The final classification layer is removed and replaced with a shared feature extractor.

```python
backbone.classifier = nn.Identity()
```

This backbone generates feature embeddings that are consumed by multiple task-specific heads.

---

## Task Heads

Each task uses an independent classifier.

```text
Linear(in_features,64)
        ↓
ReLU
        ↓
Linear(64,1)
```

Separate heads are maintained for:

* Eye State
* Yawn Detection
* Fatigue Detection

This prevents interference between unrelated classification objectives.

---

# Masked Loss Training

Missing labels are encoded using:

```python
-1
```

During optimization, only valid labels contribute to loss calculation.

```python
mask = (labels != -1)

loss =
(
 criterion(outputs, labels)
 * mask
).sum()
/
(mask.sum() + 1e-6)
```

This approach prevents invalid gradients from affecting model updates.

---

# Spatial Vision Network

File:

```text
spatial_net.py
```

The spatial network is a lightweight CNN designed for edge inference.

---

## Architecture

The model uses depthwise separable convolutions to reduce computational cost.

```text
Input
  ↓
SeparableConv Blocks
  ↓
Feature Compression
  ↓
Global Average Pooling
  ↓
Task Heads
```

---

## Outputs

### Facial Landmark Regression

```text
136 Outputs
```

Represents:

```text
68 facial landmarks × 2 coordinates
```

---

### Head Pose Estimation

```text
3 Outputs
```

Represents:

* Pitch
* Yaw
* Roll

---

### Illumination Assessment

```text
1 Output
```

Produces a normalized estimate of cabin lighting quality.

---

# Temporal Sequence Modeling

File:

```text
temporal_gru.py
```

Driver fatigue is inherently temporal.

Single frames often fail to capture gradual behavioral changes.

---

## Input Sequence

A sliding window of six-dimensional telemetry vectors:

```text
EAR
MAR
Yaw
Pitch
Fatigue Probability
Attention Score
```

---

## Recurrent Architecture

```text
Bi-GRU
Layers: 2
Dropout: 0.2
```

The final hidden representation is passed into a classification layer.

---

## Output States

```text
0 → Focused
1 → Distracted
2 → Drowsy
3 → Active Microsleep
```

---

# Backend Inference Engine

The backend operates as a real-time processing service built with FastAPI.

---

# Geometric Feature Extraction

MediaPipe Face Mesh is used to estimate facial landmarks.

Confidence threshold:

```text
0.60
```

---

## Eye Aspect Ratio (EAR)

Measures eye openness using geometric distances.

Used for:

* Blink detection
* Eye closure duration
* Microsleep detection

---

## Mouth Aspect Ratio (MAR)

Measures mouth opening.

Used for:

* Yawn detection
* Fatigue accumulation

---

## Head Pose Estimation

Key landmarks:

```text
Nose Tip      → 1
Chin          → 152
Left Eye      → 33
Right Eye     → 263
```

Used to estimate:

* Yaw
* Pitch
* Roll

---

# Adaptive Baseline Calibration

The first 150 frames are reserved for calibration.

Approximate duration:

```text
~5 seconds
```

During calibration:

* EAR baseline is learned
* Pose center is learned
* Camera mounting bias is estimated

---

## EAR Threshold

The eye closure threshold is personalized.

```math
Threshold = μEAR - (1.9 × σEAR)
```

Where:

* μEAR = mean EAR
* σEAR = standard deviation

This reduces sensitivity to facial geometry differences.

---

# Fatigue Detection Methodology

Multiple indicators contribute to fatigue estimation.

---

## Inputs

* PERCLOS
* Yawn Count
* Eye Closure Events
* Neural Network Outputs
* Temporal Patterns

---

## EMA Smoothing

Raw fatigue values are smoothed using an Exponential Moving Average.

```math
Fatigue(t)
=
α × Current
+
(1−α) × Previous
```

Where:

```text
α = 0.08
```

This reduces noise and false alarms.

---

# Microsleep Detection

Microsleep events are treated as high-risk incidents.

Criteria:

```text
EAR < Threshold
```

for

```text
≥ 0.75 seconds
```

When detected:

* Risk level increases
* Safety alerts are generated
* Telemetry logs are updated

---

# Attention Tracking Methodology

Driver attention is estimated from calibrated pose information.

---

## Safe Operating Range

### Yaw

```text
±25°
```

### Pitch

```text
±18°
```

Values outside these ranges initiate distraction tracking.

---

## Attention Decay

```math
Decay
=
0.05
+
(Time × 0.04)
+
(Deviation × 0.12)
```

Longer distraction periods produce progressively larger penalties.

---

# Explainable Rule-Based Safety Layer

The risk engine includes a deterministic rule system.

Purpose:

* Human-readable outputs
* Operational transparency
* Dispatcher auditing

Examples:

```text
Driver Focused

Temporary Roadside Distraction

Sustained Visual Inattention

Progressive Fatigue

Active Microsleep Event

High-Risk Driver State
```

The rules are evaluated using:

* Fatigue score
* Attention score
* Microsleep events
* Pose deviation
* Temporal behavior

---

# Real-Time Backend Architecture

The backend separates frame acquisition from telemetry broadcasting.

---

## Frame Capture Thread

Responsible for:

* Camera reads
* Buffer flushing
* Frame synchronization

Uses:

```python
ThreadPoolExecutor(max_workers=1)
```

---

## Processing Pipeline

```text
Frame
 ↓
Landmark Detection
 ↓
Metric Extraction
 ↓
Risk Scoring
 ↓
Global State Update
```

---

## Broadcast Loop

Frequency:

```text
Every 80 ms
```

Equivalent to:

```text
~12.5 Hz
```

---

# API Endpoints

## Webcam Frame Upload

```http
POST /api/v1/process_webcam_frame
```

Accepts:

```text
multipart/form-data
```

Used by browser webcam clients.

---

## WebSocket Telemetry

```http
/ws
```

Provides:

* Fatigue scores
* Attention scores
* EAR
* MAR
* Pose estimates
* Risk states

---

# Frontend Dashboard

The frontend is built using Next.js and React.

Its primary responsibility is visualization and reporting.

---

# Telemetry Visualization

The dashboard includes:

* Real-time gauges
* Trend charts
* Event logs
* Alert indicators

---

## Historical Buffers

### Chart Buffer

```text
20 Entries
```

Used for real-time visualization.

### Analytics Ledger

```text
50 Entries
```

Used for reporting and statistical analysis.

---

## Data Throttling

Telemetry is sampled every:

```text
3 Seconds
```

to maintain predictable frontend performance.

---

# Charting System

Implemented using Recharts.

Visualized metrics include:

* Fatigue Trend
* Attention Trend
* Risk Progression

Charts update dynamically as telemetry arrives.

---

# Stream Display Layer

The interface supports:

* Live camera streams
* Remote feeds
* Placeholder fallback views

If a stream fails:

* Error handlers activate
* Fallback visuals are displayed

---

# PDF Report Generation

The reporting engine is implemented entirely on the client side using jsPDF.

---

## Generated Contents

Reports include:

* Session information
* Safety statistics
* Risk summaries
* Event history
* Fatigue trends
* Attention trends

---

## Statistical Processing

Computed metrics:

* Mean
* Median
* Mode
* Minimum
* Maximum

---

## Pagination Logic

Large telemetry logs automatically generate additional pages.

Headers and footers remain consistent throughout the document.

---

# Project Structure

```text
Netra-Drive/
│
├── training-pipeline/
│   ├── dataset_prep.py
│   ├── train_model.py
│   ├── spatial_net.py
│   └── temporal_gru.py
│
├── web-backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── risk_scoring.py
│   │   └── config.py
│
├── web-frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   └── page.tsx
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Containerization

The backend is packaged as a Docker container.

Base image:

```dockerfile
python:3.10-slim
```

Additional dependencies are installed for headless OpenCV execution.

The container exposes:

```text
7860
```

for cloud deployment environments.

---

# Local Development Setup

## Backend

Navigate to the backend directory:

```bash
cd web-backend
```

Install dependencies:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run the server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Frontend

Navigate to the frontend directory:

```bash
cd web-frontend
```

Install packages:

```bash
npm install
```

Create:

```text
.env.local
```

Add:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

Start the development server:

```bash
npm run dev
```

---

# Deployment Notes

The system is designed to support:

* Local development environments
* Linux servers
* Docker containers
* Cloud-hosted inference services
* Hugging Face Spaces deployments

Deployment requirements primarily depend on:

* Camera throughput
* Frame resolution
* Desired inference frequency
* Concurrent client count

---


---

# Deployment & Distributed Hosting

To balance heavy AI model inference with a fast, responsive user experience, Netra-Drive is deployed across a distributed cloud architecture.

## 1. Frontend Hosting: Vercel (Next.js Edge Platform)

The Next.js frontend repository is connected directly to **Vercel** for continuous deployment.

- **Edge Optimization:** Vercel automatically deploys static dashboards, asset pools, and historical analytics routes to global edge networks, minimizing UI interaction latency.
- **Stream Synchronization:** Client-side environment configurations (`NEXT_PUBLIC_WS_URL`) seamlessly establish connections to the remote backend hosting environment.

## 2. Backend Engine: Hugging Face Spaces (Headless Docker Container)

The FastAPI engine and deep-learning weights (`driver_model.pth`) are deployed on **Hugging Face Spaces** using a specialized headless Docker container configuration.

This architecture separates computationally intensive AI inference workloads from the user-facing interface, providing:

- Independent frontend and backend scaling
- Simplified model deployment and updates
- Improved system reliability
- Reduced UI latency during inference workloads
- Better resource utilization across cloud services

# Future Improvements

Potential areas for extension include:

* Multi-driver cabin support
* Driver identification modules
* ONNX export pipeline
* TensorRT acceleration
* Fleet management integrations
* Long-term telemetry storage
* Historical analytics dashboards
* Mobile monitoring applications

---

# License

This repository is provided for research, educational, and engineering demonstration purposes.

Please review the project's license file for specific usage permissions and restrictions.

---

## Acknowledgements

Developed and maintained by **Asmin Sinha**.

Netra-Drive combines computer vision, adaptive statistical modeling, deep learning, and real-time telemetry systems to provide interpretable driver monitoring and safety analytics.
