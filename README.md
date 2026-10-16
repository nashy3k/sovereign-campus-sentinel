# Sovereign Campus Sentinel: On-Premise Physical AI Copilot

> **Built for**: GIGABYTE × NVIDIA DGX Spark Hackathon (ACE Malaysia 2026)  
> **Target Hardware**: GIGABYTE AI TOP ATOM powered by NVIDIA DGX Spark (NVIDIA GB10 Grace Blackwell Superchip)  

---

## 1. Project Overview

**Sovereign Campus Sentinel** is an on-premise, air-gapped physical safety assistant engineered for local desktop supercomputers like the GIGABYTE AI TOP ATOM.

In public environments such as schools, commercial campuses, and community facilities, transmitting live camera video feeds or sensitive incident reports to public cloud APIs introduces significant latency, high bandwidth costs, and data privacy friction.

This project demonstrates how compact Grace Blackwell silicon solves this problem:
1. **Zero Cloud Egress**: 100% on-premise vision and language processing.
2. **Multimodal Incident Perception**: Combines motion-delta keyframe sampling with local vision-language reasoning.
3. **Local Vernacular Copilot**: Evaluates local Malaysian phrasing and code-switching (Manglish, colloquial dialect) to generate actionable standard operating procedure (SOP) recommendations.
4. **Human-in-the-Loop Gate**: Puts decision authority in the hands of on-duty safety operators with one-click verification.

---

## 2. Hardware Architecture

```
┌────────────────────────────────────────────────────────┐
│             GIGABYTE AI TOP ATOM APPLIANCE             │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │   NVIDIA GB10 Grace Blackwell Superchip          │  │
│  │   • Grace ARM64 CPU + Blackwell GPU              │  │
│  │   • Unified High-Bandwidth Memory Architecture   │  │
│  │   • Ultra-Low Latency Zero-Copy Interconnect     │  │
│  └──────────────────────────────────────────────────┘  │
│                           │                            │
│  ┌────────────────────────▼─────────────────────────┐  │
│  │   Local Inference Engine (Port 8000 / 11434)     │  │
│  │   (vLLM / Ollama / NVIDIA NIM Microservice)      │  │
│  │   • Local VLM (Visual Keyframe Perception)       │  │
│  │   • Local LLM (Dialect Context Normalization)    │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           │                            │
│  ┌────────────────────────▼─────────────────────────┐  │
│  │   Sentinel Dashboard UI (Streamlit: 8501)        │  │
│  │   • Video & Camera Motion Sampler                │  │
│  │   • Operator Copilot Recommendation Feed         │  │
│  │   • Telemetry: 0 KB Cloud Egress | Real-Time ms  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start

### Prerequisites
* Python 3.10+
* GIGABYTE AI TOP ATOM or local workstation running an OpenAI-compatible endpoint (vLLM, Ollama, or NVIDIA NIM).

### Installation
```bash
git clone https://github.com/nashy3k/sovereign-campus-sentinel.git
cd sovereign-campus-sentinel
pip install -r requirements.txt
```

### 1. Hardware Diagnostic & Port Probe
Verify connection to the local DGX Spark inference server:
```bash
python probe_spark.py
```

### 2. Launch Operator Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. Key Files

* `app.py`: Streamlit operator dashboard featuring live webcam ingest, motion keyframe extraction, and copilot triage.
* `probe_spark.py`: Hardware diagnostic and latency benchmark probe.
* `test_prompts.json`: Curated local vernacular scenarios for live interactive demonstrations.
* `requirements.txt`: Python package dependencies.
