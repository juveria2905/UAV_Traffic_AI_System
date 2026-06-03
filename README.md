# Hierarchical Agentic AI for Autonomous UAV Traffic Management

## Overview

Hierarchical Agentic AI for Autonomous UAV Traffic Management is an intelligent multi-agent system designed to monitor traffic, predict collisions, prioritize actions, and provide explainable decision-making through a real-time analytics dashboard.

The system combines Computer Vision, Multi-Agent AI, Trajectory Prediction, Collision Risk Analysis, and Explainable AI to simulate autonomous traffic management in Urban Air Mobility (UAM) environments.

Although the current implementation uses traffic video streams for development and testing, the architecture is designed to be extended to UAV and drone traffic scenarios using aerial datasets such as VisDrone, UAVDT, HighD, and NGSIM.

---

## Problem Statement

Develop a hierarchical agentic AI system capable of:

* Coordinating autonomous UAV traffic
* Predicting potential collisions
* Prioritizing traffic decisions
* Providing explainable AI reasoning
* Supporting urban air mobility analytics

---

## Key Features

### Real-Time Object Detection

* YOLOv8-based detection pipeline
* Vehicle and pedestrian recognition
* Confidence-based filtering

### Multi-Object Tracking

* DeepSORT integration
* Persistent object identities
* Motion history tracking

### Trajectory Prediction

* Future path estimation
* Velocity and direction analysis
* Motion forecasting

### Collision Prediction Engine

* Time-to-Collision (TTC) computation
* Distance-based risk assessment
* High-risk collision detection

### Hierarchical Agentic AI

Multiple AI agents collaborate to make autonomous decisions:

#### Detection Agent

Responsible for:

* Object detection
* Classification
* Confidence estimation

#### Prediction Agent

Responsible for:

* Motion forecasting
* Trajectory generation
* Future state estimation

#### Collision Agent

Responsible for:

* Risk analysis
* Collision prediction
* TTC evaluation

#### Priority Agent

Responsible for:

* Threat prioritization
* Resource allocation
* Emergency assessment

#### Conflict Resolution Agent

Responsible for:

* Resolving conflicting decisions
* Decision arbitration

#### Communication Agent

Responsible for:

* Inter-agent messaging
* State synchronization

#### Memory Agent

Responsible for:

* Historical decision storage
* Event logging
* Explainability support

---

## Decision Framework

The system generates autonomous actions:

| Action         | Description                    |
| -------------- | ------------------------------ |
| MONITOR        | Continue observation           |
| HOLD           | Temporarily restrict movement  |
| REROUTE        | Suggest alternate route        |
| PRIORITIZE     | Allocate priority handling     |
| EMERGENCY_STOP | Immediate collision prevention |

---

## Dashboard Features

### Live Feed

* Real-time annotated video
* Bounding boxes
* Object labels
* Decision overlays

### Collision Alerts

* Risk visualization
* TTC monitoring
* Emergency notifications

### Agent Intelligence

* Active agents
* Agent hierarchy
* Communication flow

### Analytics

* Detection statistics
* Class distribution
* Density heatmaps
* Performance metrics

### Memory System

* Historical decisions
* Event storage
* Decision traces

### Reasoning Chains

* Explainable AI outputs
* Agent reasoning logs
* Decision transparency

### Performance Monitoring

* FPS tracking
* Throughput metrics
* Resource utilization

---

## Project Architecture

```text
Video Stream
     │
     ▼
YOLOv8 Detection
     │
     ▼
DeepSORT Tracking
     │
     ▼
Trajectory Prediction
     │
     ▼
Collision Risk Analysis
     │
     ▼
Hierarchical Agent System
     │
     ▼
Decision Engine
     │
     ▼
FastAPI Backend
     │
     ▼
Streamlit Dashboard
```

---

## Tech Stack

### Languages

* Python

### Computer Vision

* OpenCV
* YOLOv8
* DeepSORT

### Machine Learning

* NumPy
* Pandas
* Scikit-Learn

### Backend

* FastAPI
* Uvicorn

### Dashboard

* Streamlit
* Plotly

### Data Processing

* Pandas
* NumPy

### Development Tools

* Git
* GitHub

---

## Folder Structure

```text
UAV_traffic_AI_system/

├── agentic_ai/
├── analytics/
├── backend/
├── dashboard/
├── detection/
├── learning/
├── memory/
├── prediction/
├── utils/
├── tests/
├── notebooks/
├── docs/
├── outputs/
├── main.py
├── config.py
└── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/UAV_Traffic_AI_System.git
cd UAV_Traffic_AI_System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### Start Backend

```bash
python main.py
```

Backend API:

```text
http://localhost:8000
```

### Start Dashboard

Open a second terminal:

```bash
streamlit run dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

---

## Datasets

### Current Development

* Urban traffic video streams

### Recommended UAV Datasets

#### UAVDT

Dataset for UAV-based object detection and tracking.

#### VisDrone

Large-scale drone vision benchmark dataset.

#### HighD

Aerial highway traffic dataset.

#### NGSIM

Traffic trajectory dataset.

---

## Future Enhancements

* Fine-tune YOLOv8 on VisDrone
* Real UAV traffic simulation
* Multi-UAV coordination
* Reinforcement Learning based route optimization
* Federated agent communication
* Distributed inference scheduling
* Edge deployment on UAV hardware
* Digital Twin simulation environment

---

## Research Areas Covered

* Computer Vision
* Multi-Agent Systems
* Explainable AI (XAI)
* Urban Air Mobility
* Collision Avoidance
* Autonomous Systems
* Real-Time Analytics
* Intelligent Transportation Systems

---

## Author

**Shaik Juveria Tabassum**

B.E. Artificial Intelligence & Machine Learning

University College of Engineering, Osmania University

---

## License

This project is intended for educational, research, and academic purposes.
