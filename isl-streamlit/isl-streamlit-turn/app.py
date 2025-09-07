import os
import time
from datetime import datetime, timedelta
from typing import Optional

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase

from src.config import (
    FAST2SMS_API_KEY,
    SMS_DEFAULT_RECIPIENTS,
    MOCK_MODE,
    EMERGENCY_THRESHOLD,
    ALERT_COOLDOWN_SEC,
    build_rtc_configuration,
)
from src.models.factory import Models
from src.sms.fast2sms import send_sms_fast2sms
from src.storage.db import insert_alert, recent_alerts
from src.utils.fps import FpsTracker

st.set_page_config(page_title="ISL → Text (Streamlit)", layout="wide")

RTC_CONFIGURATION = build_rtc_configuration()
MODELS = Models(mock=MOCK_MODE)

# -----------------------------
# Session State
# -----------------------------
if "recognized_word" not in st.session_state:
    st.session_state.recognized_word = ""
if "recognized_conf" not in st.session_state:
    st.session_state.recognized_conf = 0.0
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_alert_at" not in st.session_state:
    st.session_state.last_alert_at = datetime.min
if "fps" not in st.session_state:
    st.session_state.fps = 0.0
if "sms_enabled" not in st.session_state:
    st.session_state.sms_enabled = False
if "sms_recipients" not in st.session_state:
    st.session_state.sms_recipients = SMS_DEFAULT_RECIPIENTS

fps_tracker = FpsTracker()

def debounce_ok(window_sec: int = ALERT_COOLDOWN_SEC) -> bool:
    return datetime.utcnow() - st.session_state.last_alert_at > timedelta(seconds=window_sec)

def mark_alert_sent(payload: dict):
    st.session_state.last_alert_at = datetime.utcnow()
    insert_alert(
        ts=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        cls=payload.get("class", "unknown"),
        score=float(payload.get("score", 0.0)),
        ok=bool(payload.get("ok", False)),
        recipients=st.session_state.sms_recipients,
        payload=str(payload),
    )

# -----------------------------
# Video transformer
# -----------------------------
class SignVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_skip = 0

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        # 1) Gesture recognition
        landmarks = MODELS.landmarks(img)
        word, conf = MODELS.gesture(landmarks if landmarks is not None else img)
        st.session_state.recognized_word = word
        st.session_state.recognized_conf = float(conf)

        # Append token if confident
        if conf >= 0.85 and word:
            st.session_state.sentence = (st.session_state.sentence + " " + word).strip()

        # 2) Emergency detection (downsampled)
        self.frame_skip = (self.frame_skip + 1) % 3
        if self.frame_skip == 0:
            emer_cls, emer_score = MODELS.emergency(img)
            if (
                st.session_state.sms_enabled
                and emer_cls != "none"
                and emer_score >= EMERGENCY_THRESHOLD
                and debounce_ok()
            ):
                snippet = st.session_state.sentence or st.session_state.recognized_word
                msg = f"Emergency '{emer_cls}' detected (score={emer_score:.2f}). Msg: {snippet}."
                if FAST2SMS_API_KEY and FAST2SMS_API_KEY != "REPLACE_ME":
                    ok, info = send_sms_fast2sms(FAST2SMS_API_KEY, st.session_state.sms_recipients, msg)
                else:
                    ok, info = False, "FAST2SMS_API_KEY not set; SMS dry-run."
                mark_alert_sent({"class": emer_cls, "score": emer_score, "ok": ok, "resp": info})

        # 3) FPS tracking
        st.session_state.fps = fps_tracker.tick()
        return img

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    st.checkbox("Enable SMS alerts", key="sms_enabled", value=st.session_state.sms_enabled)
    st.text_input("Recipients (comma-separated phone numbers)", key="sms_recipients", value=st.session_state.sms_recipients)
    st.slider("Emergency threshold", 0.5, 0.95, EMERGENCY_THRESHOLD, 0.01, key="emergency_threshold")
    st.caption("Set FAST2SMS_API_KEY in .env to actually send SMS.")

# -----------------------------
# Layout
# -----------------------------
st.title("Indian Sign Language → Text (Streamlit)")
col_video, col_right = st.columns([2, 1])

with col_video:
    st.subheader("Live")
    webrtc_streamer(
        key="isl",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=SignVideoTransformer,
        async_transform=True,
    )

with col_right:
    st.subheader("Recognition")
    st.metric("Top-1 sign", st.session_state.recognized_word, f"{st.session_state.recognized_conf:.2f}")
    st.metric("FPS", f"{st.session_state.fps:.1f}")

    st.write("---")
    st.subheader("Sentence")
    if st.button("Refine sentence (T5)"):
        tokens = st.session_state.sentence.split()
        st.session_state.sentence = MODELS.sentence(tokens)
    st.text_area("Output", value=st.session_state.sentence, height=120)

    st.write("---")
    st.subheader("Alerts (recent)")
    rows = recent_alerts(limit=10)
    if not rows:
        st.info("No alerts yet.")
    else:
        for a in rows:
            st.write(f"• [{a['ts']}] class={a['class']} score={a['score']:.2f} ok={a['ok']} recipients={a['recipients']}")

st.write("---")
st.caption("Mock mode is enabled by default. Set MOCK_MODE=false in .env and implement model loaders under src/models/* to use real models.")
