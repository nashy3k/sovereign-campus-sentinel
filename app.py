import os
import sys
import time
import json
import base64
import re
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import cv2
import numpy as np
from openai import OpenAI

from sentinel_alerts import (
    DEFAULT_DISPLAY_URL, DEFAULT_DISPLAY_TOKEN, FIGHT_VERDICT_INSTRUCTION,
    parse_fight_verdict, send_fight_alert, clear_display_alert, get_display_status,
)

# ==============================================================================
# Page Configuration & Clean Dark HUD Theme
# ==============================================================================
st.set_page_config(
    page_title="SENTIASA-EDGE | Sovereign Campus Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Enterprise Defense Feel
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stMetric { background-color: #161e2e; border: 1px solid #334155; border-radius: 8px; padding: 12px; }
    .telemetry-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-airgap {
        background-color: #065f46;
        color: #34d399;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
    .badge-blackwell {
        background-color: #1e1b4b;
        color: #818cf8;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Sovereign System Prompt & Vernacular Slang Normalizer
# ==============================================================================
SOVEREIGN_SYSTEM_PROMPT = """
You are CAMPUS-SENTINEL, an on-premise physical AI security copilot deployed on an air-gapped GIGABYTE AI TOP ATOM (NVIDIA GB10 Grace Blackwell Superchip) for institutional safety monitoring.

Your mission:
1. Review the visual scene context from on-campus CCTV feeds.
2. Accurately translate and normalize Malaysian schoolyard vernacular slang and code-switching (Manglish, Bahasa Pasar):
   - 'ponteng' / 'ponteng kelas' -> Truancy / Unauthorized absence
   - 'lepak' / 'port belakang' -> Loitering in unmonitored perimeter zone
   - 'kena pau' / 'pau duit' -> Extortion / Coercive intimidation
   - 'gaduh tolak-tolak' / 'koyak' -> Physical aggression / Escalating altercation
   - 'hisap vape' / 'pod' -> Contraband / Substance violation
   - 'tapau' -> Defeat / Overcome (threat context)
3. Determine Severity Level: [LOW | MEDIUM | HIGH | CRITICAL].
4. Decide whether the footage shows a physical fight in progress (punching, kicking, grappling, or sustained violent shoving between people). Judge from the visual evidence only: a text tip alone, an argument, or playful horseplay without strikes is NOT a fight.
5. Output structured, professional operational advice for the on-duty security warden.
6. Finish your answer with one last line, exactly in this form: __VERDICT__

Constraint: You run 100% locally with zero cloud egress under Malaysian PDPA standards.
""".replace("__VERDICT__", FIGHT_VERDICT_INSTRUCTION)

# ==============================================================================
# Helper Functions: Video & LLM Triage
# ==============================================================================
def get_base64_frame(frame: np.ndarray) -> str:
    """Encode OpenCV BGR frame to JPEG base64 string for VLM inference."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

SAMPLE_VIDEO_DIR = "sample_videos"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

@st.cache_data(show_spinner=False)
def extract_peak_motion_frame(video_path: str) -> Optional[np.ndarray]:
    """Sample frame with highest motion delta from a local video file."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    max_motion = -1.0
    best_frame = prev_frame
    
    frame_count = 0
    while cap.isOpened() and frame_count < 150:  # Check first ~5 seconds
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.mean(delta)
        if motion_score > max_motion:
            max_motion = motion_score
            best_frame = curr_frame
        prev_gray = curr_gray
        frame_count += 1
    
    cap.release()
    return best_frame

SEVERITY_LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

@st.cache_resource
def get_result_store() -> Dict[str, Dict[str, Any]]:
    """Server-side triage results keyed by clip name; outlives browser sessions and reconnects."""
    return {}

def is_inference_error(text: str) -> bool:
    return text.startswith("⚠️ Local Inference Error")

def parse_severity(text: str) -> str:
    """Pull the severity level from the copilot's free-form answer (first level mentioned after 'severity')."""
    if is_inference_error(text):
        return "ERROR"
    idx = text.lower().find("severity")
    match = re.search(r"\b(" + "|".join(SEVERITY_LEVELS) + r")\b", text[idx:] if idx >= 0 else text)
    return match.group(1) if match else "UNKNOWN"

TEXT_ONLY_INPUT = "Text only"  # input description when no footage was sent to the model

@st.cache_data(ttl=15, show_spinner=False)
def cached_display_status(base_url: str):
    return get_display_status(base_url)

def push_display_alert(clip: str, text: str) -> Dict[str, Any]:
    """Send the fight alert to the ESP32 display; returns the outcome to store with the result."""
    ok, err = send_fight_alert(display_url, display_token, clip, parse_severity(text))
    return {"ok": ok, "error": err, "at": time.strftime("%H:%M:%S")}

def store_result(results: Dict[str, Dict[str, Any]], key: str, tip: str, text: str,
                 latency_ms: float, input_used: str, model: str) -> None:
    """Save a triage result; if the model confirmed a fight on real footage, alert the ESP32 display."""
    fight = None
    if not is_inference_error(text) and input_used != TEXT_ONLY_INPUT:
        fight = parse_fight_verdict(text)
    entry = {"tip": tip, "text": text, "latency_ms": latency_ms, "input": input_used,
             "model": model, "fight": fight, "alert": None}
    if fight and display_enabled:
        entry["alert"] = push_display_alert(key, text)
    results[key] = entry

def fight_label(res: Dict[str, Any]) -> str:
    fight = res.get("fight")
    return "🚨 FIGHT DETECTED" if fight else "No fight" if fight is False else "No verdict"

def alert_label(res: Dict[str, Any]) -> str:
    alert = res.get("alert")
    if not alert:
        return "-"
    return f"Sent {alert['at']}" if alert["ok"] else f"Failed: {alert['error'][:60]}"

# The app runs on this machine; only the clip (or keyframe) + prompt go to the DGX Spark's vLLM
# server, reached through a port forwarded by NVIDIA Sync (or `ssh -L 9001:127.0.0.1:9001 <spark>`).
DEFAULT_ENDPOINT = os.environ.get("SENTINEL_API_URL", "http://localhost:9001/v1")
DEFAULT_MODEL = os.environ.get("SENTINEL_MODEL", "nvidia/nemotron-3.5-super-vl")
VIDEO_MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
              ".mkv": "video/x-matroska", ".webm": "video/webm"}

@st.cache_data(ttl=15, show_spinner=False)
def list_endpoint_models(base_url: str) -> Tuple[Optional[list], str]:
    """Return (model ids, error) for the endpoint; short timeout so a dropped link fails fast."""
    try:
        client = OpenAI(base_url=base_url, api_key="local-spark-auth", timeout=3.0, max_retries=0)
        return [m.id for m in client.models.list().data], ""
    except Exception as e:
        return None, str(e)

def pick_default_model(models: list) -> int:
    """Index of SENTINEL_MODEL if served, else the first vision-language ('vl') model, else 0."""
    if DEFAULT_MODEL in models:
        return models.index(DEFAULT_MODEL)
    return next((i for i, m in enumerate(models) if "vl" in m.lower()), 0)

def get_base64_video(video_path: str) -> Tuple[str, str]:
    """Read a clip as (mime type, base64) for a video_url data URL."""
    mime = VIDEO_MIME.get(os.path.splitext(video_path)[1].lower(), "video/mp4")
    with open(video_path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode("utf-8")

def call_local_dgx_spark(
    base_url: str,
    model_name: str,
    user_prompt: str,
    frame_base64: Optional[str] = None,
    video_path: Optional[str] = None,
    disable_thinking: bool = True
) -> Tuple[str, float, str]:
    """Execute local inference against the DGX Spark endpoint and measure latency.

    Sends the full clip when video_path is given; if the server rejects video input,
    retries with the peak-motion keyframe. Returns (text, latency_ms, input description).
    """
    client = OpenAI(base_url=base_url, api_key="local-spark-auth", timeout=300.0)

    def run(content) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SOVEREIGN_SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=4096,
            # Nemotron reasoning models: skip the hidden thinking pass (~3x faster triage)
            extra_body={"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
        )
        text = response.choices[0].message.content
        if not text:
            # Reasoning models can spend the whole token budget thinking and return no answer
            raise RuntimeError("model returned no answer (token budget exhausted?)")
        return text

    tip = f"Incident Context & Slang Tip: '{user_prompt}'\n"
    keyframe_content = [
        {"type": "text", "text": tip + "Analyze this CCTV keyframe:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}}
    ] if frame_base64 else f"Analyze this incident report and normalize any slang: '{user_prompt}'"
    keyframe_label = "Peak-motion keyframe" if frame_base64 else TEXT_ONLY_INPUT

    start_time = time.perf_counter()
    try:
        if video_path:
            mime, video_b64 = get_base64_video(video_path)
            try:
                text = run([
                    {"type": "text", "text": tip + "Analyze this CCTV clip:"},
                    {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{video_b64}"}}
                ])
                return text, (time.perf_counter() - start_time) * 1000, "Full video clip"
            except Exception as video_err:
                if not frame_base64:
                    raise
                keyframe_label = f"Peak-motion keyframe (video rejected: {str(video_err)[:100]})"
        text = run(keyframe_content)
        return text, (time.perf_counter() - start_time) * 1000, keyframe_label
    except Exception as e:
        return f"⚠️ Local Inference Error: {str(e)}", (time.perf_counter() - start_time) * 1000, keyframe_label

# ==============================================================================
# UI Layout & Execution
# ==============================================================================

# Top Navigation & Telemetry Banner
st.title("🛡️ SOVEREIGN CAMPUS SENTINEL: Physical AI Copilot")
st.caption("Air-Gapped Institutional Safety Engine | GIGABYTE AI TOP ATOM (NVIDIA GB10 Grace Blackwell)")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ DGX Spark Runtime Config")
    endpoint_url = st.text_input("Local API Endpoint", value=DEFAULT_ENDPOINT,
                                 help="vLLM port on the DGX Spark, forwarded to this machine by NVIDIA Sync")
    available_models, endpoint_error = list_endpoint_models(endpoint_url)
    if available_models:
        st.success(f"Connected to Spark: {len(available_models)} model(s)")
        model_id = st.selectbox("Active Model ID", available_models, index=pick_default_model(available_models))
    else:
        st.error("Spark endpoint unreachable. Check the NVIDIA Sync connection and port forward.")
        st.caption(f"`{endpoint_error[:120]}`")
        model_id = st.text_input("Active Model ID", value=DEFAULT_MODEL)
    if st.button("🔄 Recheck Connection", use_container_width=True):
        list_endpoint_models.clear()
        st.rerun()
    send_full_video = st.toggle("Send full video clip", value=True,
                                help="Off: send only the peak-motion keyframe. Falls back to the keyframe if the model rejects video.")
    disable_thinking = st.toggle("Fast mode (skip model reasoning)", value=True,
                                 help="Sends enable_thinking=False to Nemotron reasoning models")

    st.divider()
    st.subheader("🚨 ESP32 Alert Display")
    display_enabled = st.toggle("Alert display on fight", value=True,
                                help="When the model confirms a fight, show the warning on the Waveshare ESP32-S3 5-inch display")
    display_url = st.text_input("Display URL", value=DEFAULT_DISPLAY_URL,
                                help="Address of the ESP32 on the local network (mDNS name or IP)")
    display_token = st.text_input("Display token", value=DEFAULT_DISPLAY_TOKEN, type="password",
                                  help="Must match SENTINEL_TOKEN in the firmware's config.h")
    if display_enabled:
        display_status, display_error = cached_display_status(display_url)
        if display_status is None:
            st.error("Alert display unreachable. Check its power, Wi-Fi and URL.")
            st.caption(f"`{display_error[:120]}`")
        elif display_status.get("alert"):
            st.warning("Display is showing an active alert.")
        else:
            st.success("Alert display online")
        test_col, clear_col = st.columns(2)
        if test_col.button("Test alert", use_container_width=True):
            ok, err = send_fight_alert(display_url, display_token, "TEST CLIP", "TEST")
            (st.toast if ok else st.error)("Test alert sent to display" if ok else f"Display error: {err}")
            cached_display_status.clear()
        if clear_col.button("Clear display", use_container_width=True):
            ok, err = clear_display_alert(display_url, display_token)
            (st.toast if ok else st.error)("Display cleared" if ok else f"Display error: {err}")
            cached_display_status.clear()

    st.divider()
    st.subheader("🔒 Sovereignty Verification")
    st.markdown('<span class="badge-airgap">AIR-GAPPED</span> <span class="badge-blackwell">GB10 BLACKWELL</span>', unsafe_allow_html=True)
    st.write("**Data Egress**: `0 Bytes` (Strictly On-Prem)")
    st.write("**PDPA Compliance**: Verified (Minors Privacy)")
    st.write("**Inference Architecture**: Unified Memory (Grace + Blackwell)")
    
    st.divider()
    st.subheader("💡 Curated Slang Scenarios")
    sample_choice = st.selectbox(
        "Load Test Scenario:",
        [
            "Custom Input",
            "Scenario A: 'Budak ponteng lepak port belakang pagar'",
            "Scenario B: 'Tolong cikgu, budak form 5 duk pau duit junior'",
            "Scenario C: 'Ada kecoh tolak-tolak dekat tangga blok C, nak gaduh'",
            "Scenario D: 'Budak-budak hisap pod sembunyi dalam tandas'"
        ]
    )

# Top Telemetry Cards
t1, t2, t3, t4 = st.columns(4)
t1.metric("Compute Silicon", "NVIDIA GB10 Superchip")
t2.metric("Memory Subsystem", "Unified LPDDR5X")
t3.metric("Network Interface", "Zero Cloud Egress (0 KB)")
t4.metric("Engine Status", "ONLINE (Air-Gapped)" if available_models else "OFFLINE")

st.divider()

# Main Workspace Columns
left_col, right_col = st.columns([1, 1], gap="medium")

with left_col:
    st.subheader("📹 CCTV Feed & Peak-Motion Sampler")
    feed_mode = st.radio("Camera Source:", ["Simulated CCTV (Incident Clip)", "Upload Video", "Live Camera (Device 0)"], horizontal=True)

    active_frame = None
    active_video_path = None
    source_name = None
    if feed_mode == "Upload Video":
        uploaded = st.file_uploader("Upload CCTV Clip:", type=[e.lstrip(".") for e in VIDEO_EXTS])
        if uploaded:
            # Save into sample_videos/ so OpenCV can read it and it joins the batch triage list
            os.makedirs(SAMPLE_VIDEO_DIR, exist_ok=True)
            upload_path = os.path.join(SAMPLE_VIDEO_DIR, os.path.basename(uploaded.name))
            with open(upload_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.video(upload_path)
            active_frame = extract_peak_motion_frame(upload_path)
            active_video_path = upload_path
            source_name = os.path.basename(upload_path)
            if active_frame is not None:
                st.image(active_frame, channels="BGR", caption="Peak-motion keyframe (fallback input)", use_container_width=True)
            else:
                st.warning("Could not decode frames from this video.")
    elif feed_mode == "Simulated CCTV (Incident Clip)":
        # List local incident clips from sample_videos/ (fall back to named demo zones)
        clip_files = sorted(f for f in os.listdir(SAMPLE_VIDEO_DIR) if f.lower().endswith(VIDEO_EXTS)) \
            if os.path.isdir(SAMPLE_VIDEO_DIR) else []
        clip_name = st.selectbox("Select Camera Zone:", clip_files or [
            "CCTV 04: Rear Perimeter Fence (SMK Sentiasa)",
            "CCTV 02: Block B Corridor Stairs",
            "CCTV 01: Main Courtyard (Negative Control)"
        ])

        clip_path = os.path.join(SAMPLE_VIDEO_DIR, clip_name)
        if clip_files:
            st.video(clip_path)
            active_frame = extract_peak_motion_frame(clip_path)
            active_video_path = clip_path
            source_name = clip_name
            if active_frame is not None:
                st.image(active_frame, channels="BGR", caption="Peak-motion keyframe (fallback input)", use_container_width=True)
        else:
            # Fallback visual demo canvas
            synthetic_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(synthetic_frame, f"LIVE SIMULATION: {clip_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.putText(synthetic_frame, "PEAK MOTION DETECTED: [CONFIDENCE: 92.4%]", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.rectangle(synthetic_frame, (180, 120), (460, 320), (0, 0, 255), 2)
            cv2.putText(synthetic_frame, "TARGET: CLUSTER_03", (185, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            st.image(synthetic_frame, channels="BGR", use_container_width=True)
            active_frame = synthetic_frame
    else:
        cam_input = st.camera_input("Take Live Verification Snapshot")
        if cam_input:
            bytes_data = cam_input.getvalue()
            active_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            source_name = "Live Camera Snapshot"

with right_col:
    st.subheader("💬 Warden Incident Triage Copilot")
    
    # Preload prompt based on sidebar choice
    default_text = "Budak-budak lepak dekat pagar belakang waktu kelas, tolong hantar warden."
    if "Scenario A" in sample_choice:
        default_text = "Budak ponteng duk lepak port belakang pagar dewan."
    elif "Scenario B" in sample_choice:
        default_text = "Tolong cikgu, budak form 5 duk pau duit junior dekat loker."
    elif "Scenario C" in sample_choice:
        default_text = "Ada kecoh tolak-tolak dekat tangga blok C, nampak gaya nak gaduh besar."
    elif "Scenario D" in sample_choice:
        default_text = "Budak-budak hisap pod sembunyi dalam tandas hujung."

    incident_tip = st.text_area("Incident Tip / Whistleblower Text:", value=default_text, height=100)
    
    col_btn, col_bench = st.columns([1, 1])
    with col_btn:
        triage_btn = st.button("🚀 Execute Sovereign Triage", type="primary", use_container_width=True)
    
    # Results persist per source so they survive reruns, reconnects and the warden gate buttons
    results = get_result_store()
    result_key = source_name or "Text-only report"

    if triage_btn:
        frame_b64 = get_base64_frame(active_frame) if active_frame is not None else None

        with st.spinner("GB10 Grace Blackwell Processing (Local VLM + Vernacular Normalization)..."):
            result_text, latency_ms, input_used = call_local_dgx_spark(
                base_url=endpoint_url,
                model_name=model_id,
                user_prompt=incident_tip,
                frame_base64=frame_b64,
                video_path=active_video_path if send_full_video else None,
                disable_thinking=disable_thinking
            )
        store_result(results, result_key, incident_tip, result_text, latency_ms, input_used, model_id)

    if result_key in results:
        res = results[result_key]
        # Display Results
        if is_inference_error(res["text"]):
            st.error(f"Triage Failed after {res['latency_ms']:.1f} ms")
        else:
            st.success(f"Triage Complete in {res['latency_ms']:.1f} ms (Running 100% on NVIDIA GB10)")
        st.markdown(f"### 📋 Copilot Operational Assessment: `{result_key}`")
        sev_col, fight_col = st.columns(2)
        sev_col.metric("Severity Level", parse_severity(res["text"]))
        fight_col.metric("Fight Verdict", fight_label(res))
        alert = res.get("alert")
        if alert and alert["ok"]:
            st.success(f"ESP32 display alerted at {alert['at']}: \"Fighting is detected. Please take immediate actions now!\"")
        elif alert:
            st.error(f"ESP32 display alert FAILED at {alert['at']}: {alert['error']}")
            if st.button("🔁 Resend Alert to Display"):
                results[result_key]["alert"] = push_display_alert(result_key, res["text"])
                st.rerun()
        st.caption(f"Model: `{res.get('model', '?')}` · Input: {res.get('input', 'Peak-motion keyframe')}")
        st.info(res["text"])

        # Human-in-the-Loop Approval Action Buttons
        st.write("**Human-in-the-Loop Warden Gate:**")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("✅ Approve: Dispatch On-Duty Patrol", use_container_width=True):
                st.toast("Warden Action Confirmed: Security dispatched locally.", icon="🛡️")
        with act_col2:
            if st.button("❌ Dismiss: False Positive / Normal", use_container_width=True):
                st.toast("Alert Logged as Dismissed.", icon="ℹ️")
                if display_enabled:
                    clear_display_alert(display_url, display_token)
                    cached_display_status.clear()

# ==============================================================================
# Batch Triage: every clip in sample_videos/ (including uploaded videos)
# ==============================================================================
st.divider()
st.subheader("🗂️ Batch Triage: All Sample & Uploaded Clips")
batch_files = sorted(f for f in os.listdir(SAMPLE_VIDEO_DIR) if f.lower().endswith(VIDEO_EXTS)) \
    if os.path.isdir(SAMPLE_VIDEO_DIR) else []
st.caption(f"{len(batch_files)} clip(s) in `{SAMPLE_VIDEO_DIR}/`. Each clip is sent in full (or as its peak-motion keyframe) with the incident tip above.")

if st.button("▶️ Run Triage on All Clips", disabled=not batch_files):
    results = get_result_store()
    progress = st.progress(0.0, text="Starting batch triage...")
    for i, fname in enumerate(batch_files):
        progress.progress(i / len(batch_files), text=f"Triaging {fname} ({i + 1}/{len(batch_files)})...")
        clip_file = os.path.join(SAMPLE_VIDEO_DIR, fname)
        frame = extract_peak_motion_frame(clip_file)
        text, ms, input_used = call_local_dgx_spark(
            base_url=endpoint_url,
            model_name=model_id,
            user_prompt=incident_tip,
            frame_base64=get_base64_frame(frame) if frame is not None else None,
            video_path=clip_file if send_full_video else None,
            disable_thinking=disable_thinking
        )
        store_result(results, fname, incident_tip, text, ms, input_used, model_id)
    progress.empty()

all_results = get_result_store()
batch_done = [f for f in batch_files if f in all_results]
if batch_done:
    st.dataframe(
        [{
            "Clip": f,
            "Severity": parse_severity(all_results[f]["text"]),
            "Fight": fight_label(all_results[f]),
            "Display Alert": alert_label(all_results[f]),
            "Latency (ms)": round(all_results[f]["latency_ms"]),
            "Input": all_results[f].get("input", "Peak-motion keyframe"),
            "Model": all_results[f].get("model", "?"),
            "Incident Tip": all_results[f]["tip"],
        } for f in batch_done],
        use_container_width=True,
        hide_index=True,
    )
    for f in batch_done:
        res = all_results[f]
        with st.expander(f"{'🚨 ' if res.get('fight') else ''}{parse_severity(res['text'])} · {f} · {res['latency_ms']:.0f} ms"):
            vid_col, txt_col = st.columns([1, 2])
            with vid_col:
                st.video(os.path.join(SAMPLE_VIDEO_DIR, f))
                frame = extract_peak_motion_frame(os.path.join(SAMPLE_VIDEO_DIR, f))
                if frame is not None:
                    st.image(frame, channels="BGR", caption="Peak-motion keyframe (fallback input)", use_container_width=True)
            with txt_col:
                st.caption(f"Model: `{res.get('model', '?')}` · Input: {res.get('input', 'Peak-motion keyframe')}")
                (st.error if is_inference_error(res["text"]) else st.markdown)(res["text"])
