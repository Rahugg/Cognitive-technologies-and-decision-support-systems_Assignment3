import hashlib
import io
import threading
import time
import uuid
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
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


SAMPLE_DIR = Path(__file__).parent / "examples" / "open_source"
UPLOAD_SAMPLES = {
    "Laughing face": "laughing-face.jpg",
    "Fearful face": "fear-face.jpg",
    "Smiling face": "smiling-face.jpg",
    "Sad face": "sad-face.jpg",
    "Angry face": "angry-face.jpg",
    "Surprised face": "surprised-face.jpg",
}


def show_photo_result(result):
    perception = result["perception"]
    attention = result["attention"]
    decision = result["knowledge"]
    prediction_col, confidence_col, quality_col = st.columns(3)
    prediction_col.metric("Top expression", perception["emotion"].title())
    confidence_col.metric("Confidence", f"{perception['confidence']:.0%}")
    quality_col.metric("Photo quality", f"{perception['quality_score']:.2f} / 1.00")

    if not attention["accepted"]:
        st.error(f"Image not accepted: {attention['reason']}. Try a brighter, sharper photo with one face centered.")
    elif decision["label"] == "uncertain":
        st.warning("Image accepted, but the model is unsure. Try a clearer face photo.")
    else:
        st.success(f"Image accepted · result: **{decision['label'].title()}**")

    st.caption(f"Knowledge result: {decision['label']} · Action: {decision['action']}")
    probabilities = perception["probabilities"]
    st.subheader("Emotion scores")
    score_frame = pd.DataFrame({"Emotion": list(probabilities), "Confidence": list(probabilities.values())})
    score_frame = score_frame.sort_values("Confidence", ascending=True)
    st.bar_chart(score_frame, x="Emotion", y="Confidence", horizontal=True)

    with st.expander("Technical details", expanded=False):
        details_left, details_right = st.columns(2)
        with details_left:
            st.markdown("**Attention**")
            st.write(f"Status: {'Accepted' if attention['accepted'] else 'Rejected'}")
            st.write(f"Reason: {attention['reason']}")
            st.write(f"Relevance: {attention['relevance']:.2f}")
            st.markdown("**Memory**")
            st.write(f"Recent accepted images in this session: {result['memory']['short_term_size']}")
        with details_right:
            st.markdown("**Knowledge rules**")
            for rule in decision["rules"]:
                st.write(f"- {rule}")
            st.markdown("**Perception**")
            st.write(f"Face detected: {'Yes' if perception['face_found'] else 'No'}")
            st.write(f"Brightness: {perception['brightness']:.2f} · Sharpness: {perception['sharpness']:.1f}")


with st.sidebar:
    st.session_state.setdefault("camera_session_id", f"live-{uuid.uuid4().hex[:8]}")
    with st.expander("Session and privacy", expanded=False):
        session_id = st.text_input("Session ID", key="camera_session_id").strip()
        st.write(f"Quality threshold: {QUALITY_THRESHOLD:.2f}")
        st.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
        st.info("Photos and video are processed in memory. PostgreSQL stores prediction metadata only.")
    session_id = st.session_state.get("camera_session_id", "").strip()

live_tab, upload_tab = st.tabs(["Live camera", "Upload a photo"])

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
    st.subheader("Check a face photo")
    st.write("Choose a photo from your device or use one of the included demo images.")
    st.caption("Use a clear, front-facing image with one face. JPG and PNG are supported.")

    source = st.radio("Photo source", ["Upload a photo", "Try a demo image"], horizontal=True,
                      key="photo_source")
    image_bytes = None
    image_name = None

    if source == "Upload a photo":
        uploaded = st.file_uploader(
            "Choose or drop a photo here", type=["jpg", "jpeg", "png"],
            help="The photo is processed by this local app. Only prediction metadata are stored.",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            image_name = uploaded.name
    else:
        image_name = st.selectbox("Choose an example", list(UPLOAD_SAMPLES))
        sample_path = SAMPLE_DIR / UPLOAD_SAMPLES[image_name]
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            st.caption("These synthetic demo faces are CC0 examples. The model prediction can differ from the visual label.")
        else:
            st.error("Demo images are missing. Rebuild the app with `docker compose up --build -d app`.")

    if image_bytes:
        image_key = hashlib.sha256(image_bytes).hexdigest()
        try:
            rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as error:
            st.error(f"Could not read this image: {error}")
            rgb = None

        if rgb is not None:
            preview_col, result_col = st.columns([0.9, 1.1], gap="large")
            with preview_col:
                st.image(rgb, caption=image_name, use_container_width=True)
                analyze = st.button("Analyze photo", type="primary", use_container_width=True,
                                    key=f"analyze-{image_key[:12]}")
            saved = st.session_state.get("photo_analysis")
            if analyze:
                try:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # Keep still-image analysis separate from the camera session so old live
                    # frames cannot bias its short-term smoothed prediction.
                    upload_session = f"upload-{image_key[:12]}"
                    result = load_system().process(bgr, upload_session, require_face=True)
                    saved = {"key": image_key, "result": result}
                    st.session_state.photo_analysis = saved
                except Exception as error:
                    st.error(f"Could not analyze this image: {error}")
                    saved = None

            with result_col:
                if saved and saved["key"] == image_key:
                    st.markdown("### Result")
                    show_photo_result(saved["result"])
                else:
                    st.info("Your photo is ready. Select **Analyze photo** to see the result.")
