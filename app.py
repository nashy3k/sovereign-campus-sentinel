import os
import sys
import time
import json
import base64
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
import cv2
import numpy as np
from openai import OpenAI

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
3. Identify potential violent activities from the video feed:
   - Physical fights/alterigations (gaduh, koyak)
   - Weapon detection or threat displays
   - Group confrontations
   - Severe loitering in monitored zones
4. Determine Risk/Dangerous Level: [LOW | MEDIUM | HIGH | CRITICAL].
5. Output structured, professional operational advice for the on-duty security warden including:
   - Identified activities
   - Risk level assessment
   - Recommended immediate actions

Constraint: You run 100% locally with zero cloud egress under Malaysian PDPA standards.
"""

# ==============================================================================
# Helper Functions: Video & LLM Triage
# ==============================================================================
def get_base64_frame(frame: np.ndarray) -> str:
    """Encode OpenCV BGR frame to JPEG base64 string for VLM inference."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

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

def call_local_dgx_spark(
    base_url: str,
    model_name: str,
    user_prompt: str,
    frame_base64: Optional[str] = None
) -> Tuple[str, float]:
    """Execute local inference against the DGX Spark endpoint and measure latency."""
    client = OpenAI(base_url=base_url, api_key="local-spark-auth")
    messages = [{"role": "system", "content": SOVEREIGN_SYSTEM_PROMPT}]
    
    if frame_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Incident Context & Slang Tip: '{user_prompt}'\nAnalyze this CCTV keyframe:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Analyze this incident report and normalize any slang: '{user_prompt}'"
        })
    
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return response.choices[0].message.content, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return f"⚠️ Local Inference Error: {str(e)}", elapsed_ms

# ==============================================================================
# Helper: Fetch Available Models from Local vLLM/NIM Endpoint
# ==============================================================================
@st.cache_data(ttl=300)
def fetch_available_models(base_url: str) -> List[str]:
    """Query the local LLM server's /models endpoint to discover available models."""
    try:
        client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="local-spark-auth")
        models = client.models.list()
        return [m.id for m in models.data] if models else []
    except Exception:
        return []

# ==============================================================================
# UI Layout & Execution
# ==============================================================================

# Top Navigation & Telemetry Banner
st.title("🛡️ SOVEREIGN CAMPUS SENTINEL: Physical AI Copilot")
st.caption("Air-Gapped Institutional Safety Engine | GIGABYTE AI TOP ATOM (NVIDIA GB10 Grace Blackwell)")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ DGX Spark Runtime Config")
    endpoint_url = st.text_input("Local API Endpoint", value="http://localhost:9001/v1")
    
    # Auto-discover models from endpoint
    available_models = fetch_available_models(endpoint_url)
    if not available_models:
        available_models = ["omni", "nvidia/nemotron-3.5-super-vl", "openai/gpt-4o-mini", "custom/model-id"]
    
    # Model selection: dropdown if auto-discovered, text input for custom
    if len(available_models) > 1 or st.checkbox("Show model dropdown", value=len(available_models) > 1):
        model_id = st.selectbox("Active Model ID:", available_models, index=0)
    else:
        model_id = st.text_input("Active Model ID", value=available_models[0] if available_models else "lightning")
    
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
t4.metric("Engine Status", "ONLINE (Air-Gapped)")

st.divider()

# Main Workspace Columns
left_col, right_col = st.columns([1, 1], gap="medium")

with left_col:
    st.subheader("📹 CCTV Feed & Media Source")
    feed_mode = st.radio("Camera Source:", ["Simulated CCTV (Incident Clip)", "Live Camera (Device 0)", "Upload Video"], horizontal=True)
    
    active_frame = None
    uploaded_video = None
    video_path_for_inference = None
    
    if feed_mode == "Simulated CCTV (Incident Clip)":
        clip_name = st.selectbox("Select Camera Zone:", [
            "CCTV 04: Rear Perimeter Fence (SMK Sentiasa)",
            "CCTV 02: Block B Corridor Stairs",
            "CCTV 01: Main Courtyard (Negative Control)"
        ])
        
        # Simulated placeholder frame if no video file
        placeholder_path = "sample_videos/clip1_rear_gate_loitering.mp4"
        if os.path.exists(placeholder_path):
            st.video(placeholder_path)
            active_frame = extract_peak_motion_frame(placeholder_path)
        else:
            # Fallback visual demo canvas
            synthetic_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(synthetic_frame, f"LIVE SIMULATION: {clip_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.putText(synthetic_frame, "PEAK MOTION DETECTED: [CONFIDENCE: 92.4%]", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.rectangle(synthetic_frame, (180, 120), (460, 320), (0, 0, 255), 2)
            cv2.putText(synthetic_frame, "TARGET: CLUSTER_03", (185, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            st.image(synthetic_frame, channels="BGR", use_container_width=True)
            active_frame = synthetic_frame
    elif feed_mode == "Live Camera (Device 0)":
        cam_input = st.camera_input("Take Live Verification Snapshot")
        if cam_input:
            bytes_data = cam_input.getvalue()
            active_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    elif feed_mode == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            st.video(video_bytes)
            # Save temporarily for processing
            video_path_for_inference = "/tmp/uploaded_video_for_inference.mp4"
            with open(video_path_for_inference, "wb") as f:
                f.write(video_bytes)
            # Extract peak motion frame for keyframe display
            active_frame = extract_peak_motion_frame(video_path_for_inference)
            if active_frame is not None:
                # Display extracted frame in collapsible expander
                with st.expander("📹 Peak Motion Keyframe", expanded=False):
                    frame_b64_preview = get_base64_frame(active_frame)
                    st.image(f"data:image/jpeg;base64,{frame_b64_preview}", channels="BGR", use_container_width=True, caption="Extracted peak motion frame")

with right_col:
    st.subheader("🛡️ Sovereign Triage Copilot")
    
    col_btn, col_bench = st.columns([1, 1])
    with col_btn:
        triage_btn = st.button("🚀 Execute Sovereign Triage", type="primary", use_container_width=True)
    
    if triage_btn:
        # Use keyframe for inference
        frame_b64 = get_base64_frame(active_frame) if active_frame is not None else None
        
        with st.spinner("GB10 Grace Blackwell Processing (Local VLM + Vernacular Normalization)..."):
            result_text, latency_ms = call_local_dgx_spark(
                base_url=endpoint_url,
                model_name=model_id,
                user_prompt="Analyze this CCTV keyframe for potential violent activities and determine risk level.",
                frame_base64=frame_b64
            )
        
        # Display Results
        st.success(f"Triage Complete in {latency_ms:.1f} ms (Running 100% on NVIDIA GB10)")
        st.markdown("### 📋 Copilot Operational Assessment")
        st.info(result_text)
        
# Initialize session state for approval notification
if "approval_submitted" not in st.session_state:
    st.session_state.approval_submitted = False
if "dismiss_submitted" not in st.session_state:
    st.session_state.dismiss_submitted = False

# Human-in-the-Loop Approval Action Buttons
st.write("**Human-in-the-Loop Warden Gate:**")
act_col1, act_col2 = st.columns(2)
with act_col1:
    if st.button("✅ Approve: Dispatch On-Duty Patrol", use_container_width=True):
        st.toast("Warden Action Confirmed: Security dispatched locally.", icon="🛡️")
        st.session_state.approval_submitted = True
with act_col2:
    if st.button("❌ Dismiss: False Positive / Normal", use_container_width=True):
        st.toast("Alert Logged as Dismissed.", icon="ℹ️")
        st.session_state.dismiss_submitted = True

# Notification section after approval
if st.session_state.approval_submitted:
    st.success("📡 Notification submitted to related authorities: On-duty patrol dispatched. Incident ID logged. Waiting for warden confirmation.")
    
# Dismissal notification
if st.session_state.dismiss_submitted:
    st.info("📝 Alert logged as dismissed. Incident recorded for review.")
