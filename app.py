import threading
import time
import uuid

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from src.emotion_system.config import (CONFIDENCE_THRESHOLD, DATABASE_URL, MODEL_PATH,
                                       QUALITY_THRESHOLD, STM_WINDOW)
from src.emotion_system.model import load_model
from src.emotion_system.pipeline import CognitiveEmotionSystem

st.set_page_config(page_title="Emotion Recognition System", page_icon="🙂", layout="wide")
st.title("Live Facial Expression Recognition")
st.caption(
    "Start the camera and show an expression. The label updates on the live video about once "
    "per second. This estimates visible facial expression; it cannot establish how someone feels."
)


@st.cache_resource
def load_system():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, emotions = load_model(MODEL_PATH, device)
    return CognitiveEmotionSystem(model, emotions, DATABASE_URL, QUALITY_THRESHOLD,
                                  CONFIDENCE_THRESHOLD, STM_WINDOW, device)


@st.cache_resource
def inference_lock():
    # The model and session memory are shared by Streamlit sessions.
    return threading.Lock()


def show_result(result):
    st.subheader("Latest camera result")
    st.metric("Decision", result["knowledge"]["label"])
    st.write("Action:", result["knowledge"]["action"])
    smoothed = result["memory"]["smoothed_prediction"]
    st.write("Smoothed expression:", smoothed["emotion"],
             f"({smoothed['confidence']:.1%} confidence)")
    st.write("Attention:", result["attention"]["reason"])
    st.write("Knowledge rules:", ", ".join(result["knowledge"]["rules"]))


with st.sidebar:
    st.session_state.setdefault("camera_session_id", f"live-{uuid.uuid4().hex[:8]}")
    session_id = st.text_input("Session ID", key="camera_session_id").strip()
    st.write(f"Quality threshold: {QUALITY_THRESHOLD:.2f}")
    st.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    st.info("Video is processed in memory. PostgreSQL stores prediction metadata only.")

live_tab, upload_tab = st.tabs(["Live camera", "Upload image"])

with live_tab:
    st.write("Click **START**, allow camera access, then change your facial expression.")
    st.caption(f"Short-term memory averages up to {STM_WINDOW} accepted frames (about 5 seconds).")

    if "live_camera_state" not in st.session_state:
        st.session_state.live_camera_state = {
            "last_analysis": 0.0,
            "latest_result": None,
            "error": None,
            "lock": threading.Lock(),
        }
    camera_state = st.session_state.live_camera_state
    system = load_system()
    shared_inference_lock = inference_lock()

    def video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        should_analyze = False
        with camera_state["lock"]:
            if now - camera_state["last_analysis"] >= 1.0:
                camera_state["last_analysis"] = now
                should_analyze = True

        if should_analyze:
            try:
                with shared_inference_lock:
                    result = system.process(image, session_id or "live-demo", require_face=True)
                with camera_state["lock"]:
                    camera_state["latest_result"] = result
                    camera_state["error"] = None
            except Exception as error:
                with camera_state["lock"]:
                    camera_state["error"] = str(error)

        with camera_state["lock"]:
            result = camera_state["latest_result"]
            error = camera_state["error"]

        if error:
            label = "INFERENCE ERROR"
            detail = error[:72]
        elif result is None:
            label = "ANALYZING FACE..."
            detail = "Keep your face visible to the camera"
        elif not result["attention"]["accepted"]:
            label = "INSUFFICIENT DATA"
            detail = result["attention"]["reason"]
        else:
            prediction = result["memory"]["smoothed_prediction"]
            label = f"{prediction['emotion'].upper()}  {prediction['confidence']:.0%}"
            detail = result["knowledge"]["label"].upper()

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        box_width = min(image.shape[1], label_width + 28)
        cv2.rectangle(image, (10, 10), (10 + box_width, 56), (20, 37, 49), -1)
        cv2.putText(image, label, (20, 39), font, scale, (255, 255, 255), thickness,
                    cv2.LINE_AA)
        detail_y = min(image.shape[0] - 10, 82)
        cv2.putText(image, detail, (16, detail_y), font, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(
        key="emotion-live-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20},
            },
            "audio": False,
        },
        async_processing=True,
    )

    latest = camera_state["latest_result"]
    if latest:
        show_result(latest)
    elif camera_state["error"]:
        st.error(camera_state["error"])

with upload_tab:
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        rgb = np.asarray(Image.open(uploaded).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        left, right = st.columns([1, 1.2])
        left.image(rgb, caption="Uploaded input", use_container_width=True)
        try:
            result = load_system().process(bgr, session_id or "upload-demo", require_face=True)
            with right:
                show_result(result)
                st.subheader("Perception")
                st.json({k: v for k, v in result["perception"].items()
                         if k != "probabilities"})
                st.bar_chart({"emotion": list(result["perception"]["probabilities"]),
                              "probability": list(result["perception"]["probabilities"].values())},
                             x="emotion", y="probability")
                st.subheader("Attention")
                st.json(result["attention"])
                st.subheader("Memory")
                st.json({"previous": result["memory"]["previous"],
                         "short_term_size": result["memory"]["short_term_size"]})
                st.subheader("Knowledge rules")
                for rule in result["knowledge"]["rules"]:
                    st.write("-", rule)
        except Exception as error:
            st.error(str(error))
