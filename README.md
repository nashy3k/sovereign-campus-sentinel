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

### Air-Gapped Inference via NVIDIA Sync
The dashboard runs on your laptop. The DGX Spark needs no internet: only the CCTV clip (or its peak-motion keyframe) and the incident text go to the Spark's local vLLM server, through a port that NVIDIA Sync forwards over its SSH link.

1. **Connect the laptop to the Spark directly.** Put both on the same switch or Wi-Fi network, or use a direct Ethernet cable; no internet uplink is needed. In NVIDIA Sync, the device should show **Connected**. Sync finds the Spark by its mDNS name (`<hostname>.local`), so both machines must be on the same local network.
2. **Serve the vision-language model on the Spark.** Run Nemotron 3.5 VL under vLLM on the Spark (for example on port `9001`). For full-clip triage, the server must accept video input.
3. **Forward the inference port.** In NVIDIA Sync, open **Settings → your device → Custom ports/apps** and add a port forward for the vLLM port (for example `9001`). Sync then serves it on your laptop at `localhost:9001`. The equivalent manual tunnel uses the SSH alias that Sync writes to `~/.ssh/config`:
   ```bash
   ssh -f -N -o ServerAliveInterval=30 -L 9001:127.0.0.1:9001 <sync-device-alias>
   ```
4. **Verify:** `python probe_spark.py` should report the detected model.
5. **Launch:** `streamlit run app.py`. The sidebar shows **Connected to Spark** and lists the served models. It selects `SENTINEL_MODEL` if that model is served, otherwise the first model with "vl" in its ID. Use **Upload Video** or the clip picker, and **Run Triage on All Clips** for batch runs. Uploaded clips are stored on the laptop in `sample_videos/`; they never leave it.

Configuration (optional environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `SENTINEL_API_URL` | `http://localhost:9001/v1` | Forwarded OpenAI-compatible endpoint |
| `SENTINEL_MODEL` | `nvidia/nemotron-3.5-super-vl` | Preferred model ID served by vLLM |

**Send full video clip** (sidebar, on by default) sends the whole clip as a `video_url`, so the model sees motion such as shoving or climbing, not one frame. If the server rejects video, the app retries with the peak-motion keyframe and labels the result accordingly. Keep clips small (for example 640×360, under 30 s).

**Fast mode** (sidebar, on by default) sends `enable_thinking=False`. Nemotron reasoning models otherwise spend most of the token budget on hidden reasoning, which makes triage about 3× slower.

---

## 4. Key Files

* `app.py`: Streamlit operator dashboard featuring live webcam ingest, motion keyframe extraction, and copilot triage.
* `probe_spark.py`: Hardware diagnostic and latency benchmark probe.
* `test_prompts.json`: Curated local vernacular scenarios for live interactive demonstrations.
* `requirements.txt`: Python package dependencies.
