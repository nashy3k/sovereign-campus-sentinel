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

### Fight Alert on the Waveshare ESP32-S3-Touch-LCD-5 / 5B
When the model confirms a fight in a clip, the dashboard pushes an alert over the local network to a [Waveshare ESP32-S3-Touch-LCD-5 / 5B](https://www.waveshare.com/esp32-s3-touch-lcd-5.htm) (5-inch touch display). The screen flashes red and shows **"Fighting is detected. Please take immediate actions now!"** with the clip name, severity and time. Tap **ACKNOWLEDGE** on the screen to clear it.

```
 clip ──► Nemotron VL (DGX Spark) ──► "FIGHT_DETECTED: YES" ──► dashboard ──HTTP POST /alert──► ESP32-S3 5" display
```

**How the trigger works.** The system prompt asks the model to end every answer with `FIGHT_DETECTED: YES` or `FIGHT_DETECTED: NO`, judged from the footage only. The dashboard alerts only on an explicit `YES` for a result that included a clip, keyframe or snapshot. A missing verdict, `NO`, a text-only tip or an inference error never alerts, and the result shows *No verdict* so you can see the model skipped the line. A failed delivery shows the error with a **Resend Alert to Display** button. **Dismiss** also clears the display.

**1. Flash the display** (Arduino IDE 2.x or `arduino-cli`). Verified on a Waveshare **ESP32-S3-Touch-LCD-5B (1024×600)**.
1. Install the **esp32 by Espressif Systems** board package and these libraries: `ESP32_Display_Panel` 1.0.x, `ESP32_IO_Expander` 1.x, `esp-lib-utils` (0.2.x or 0.3.x) and `lvgl` **8.4.0** (not 9.x). The tested set is esp32 core **3.2.0**, `ESP32_Display_Panel` 1.0.4, `ESP32_IO_Expander` 1.1.1, `esp-lib-utils` 0.3.0 and `lvgl` 8.4.0.
2. **Use Espressif's high-performance SDK build (needed on the 5B).** With Wi-Fi running, the 1024×600 RGB panel flickers on the stock SDK, because radio traffic competes with the screen refresh for PSRAM bandwidth. Download [`esp32-3.2.0-h.zip`](https://dl.espressif.com/AE/esp-arduino-libs/esp32-3.2.0-h.zip) (see [arduino-esp32-sdk](https://github.com/esp-arduino-libs/arduino-esp32-sdk)), unzip it, and replace the `esp32s3` folder inside `~/Library/Arduino15/packages/esp32/tools/esp32-arduino-libs/idf-release_v5.4-*/` with the one from the zip (keep a backup). It must match core 3.2.0; the build is not published for 3.3.x. The firmware also uses double-buffered anti-tearing, a 30-line RGB bounce buffer and Wi-Fi modem sleep to keep the panel steady.
3. In `firmware/esp32s3_alert_display/`, copy `config.h.example` to `config.h` and set your 2.4 GHz Wi-Fi SSID/password and a shared token. `config.h` holds secrets: never commit it.
4. Open `esp32s3_alert_display.ino` and select **ESP32S3 Dev Module** with: PSRAM **OPI PSRAM**, Flash Size **16MB**, Flash Mode **QIO 80MHz**, Partition Scheme **16M Flash (3MB APP/9.9MB FATFS)**, and USB CDC On Boot **Enabled** if you use the USB port. The sketch enables the `BOARD_WAVESHARE_ESP32_S3_TOUCH_LCD_5_B` profile. For the standard 800×480 model, switch to `BOARD_WAVESHARE_ESP32_S3_TOUCH_LCD_5` in `esp_panel_board_supported_conf.h`; a wrong profile gives a white screen with stray lines.
5. Upload. If it fails or the USB drops mid-write, use a direct data cable (no hub), set Upload Speed to 115200, and if the board no longer connects, hold **BOOT**, press **RESET**, then release **BOOT** to enter download mode.
6. The serial monitor (115200 baud) prints the IP address, and the screen shows `http://sentinel-display.local`. If `.local` names don't resolve on your network, use the IP address in the dashboard instead.

**Demo mode (optional).** Set `#define DEMO_MODE 1` in `config.h` and the red flashing alert screen is always on from power-up, even before any fight is detected. After **ACKNOWLEDGE** (or `POST /clear`) the monitoring screen shows for `DEMO_REALERT_DELAY_MS` (default 60000 = one minute), then the alert comes back by itself. A real alert from the dashboard still replaces the standing one with its clip and severity. The standing demo alert does not count as a pending alert in `/status`. Leave `DEMO_MODE` at `0` for real use.

**2. Point the dashboard at it.** Put the laptop and the display on the same network, then set the values below. They can also be edited in the sidebar under **ESP32 Alert Display**, which shows whether the display is online and has **Test alert** and **Clear display** buttons.

| Variable | Default | Purpose |
|---|---|---|
| `SENTINEL_DISPLAY_URL` | `http://sentinel-display.local` | Display address (mDNS name or IP) |
| `SENTINEL_DISPLAY_TOKEN` | *(empty)* | Must equal `SENTINEL_TOKEN` in the firmware's `config.h` |

**3. Test without the AI.** From the command line:
```bash
python sentinel_alerts.py status   # is the display reachable?
python sentinel_alerts.py test     # show a test alert
python sentinel_alerts.py clear    # back to the monitoring screen
```
Run the unit tests (no hardware needed) with `python -m unittest test_sentinel_alerts -v`.

**Display HTTP API** (port 80). `POST` requests need the `X-Sentinel-Token` header:

| Endpoint | Purpose |
|---|---|
| `POST /alert` | Form fields `clip`, `severity`, `time` (all optional). Shows the alert. |
| `POST /clear` | Returns to the monitoring screen. |
| `GET /status` | `{"ok":true,"alert":false,"count":0,"uptime_s":123}` |

The traffic stays on your LAN, so the air-gap story is unchanged, but it is plain HTTP with a shared token. Keep the display on a trusted or isolated network.

---

## 4. Key Files

* `app.py`: Streamlit operator dashboard featuring live webcam ingest, motion keyframe extraction, and copilot triage.
* `probe_spark.py`: Hardware diagnostic and latency benchmark probe.
* `sentinel_alerts.py`: Fight-verdict parsing and the client/CLI for the ESP32 alert display.
* `test_sentinel_alerts.py`: Unit tests for the verdict parser and display client.
* `firmware/esp32s3_alert_display/`: Arduino sketch for the Waveshare ESP32-S3-Touch-LCD-5 alert screen.
* `test_prompts.json`: Curated local vernacular scenarios for live interactive demonstrations.
* `requirements.txt`: Python package dependencies.
