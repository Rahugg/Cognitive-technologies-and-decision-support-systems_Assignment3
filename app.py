import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.emotion_system.config import (CONFIDENCE_THRESHOLD, DATABASE_URL, MODEL_PATH,
                                       QUALITY_THRESHOLD, STM_WINDOW)
from src.emotion_system.model import load_model
from src.emotion_system.pipeline import CognitiveEmotionSystem

st.set_page_config(page_title="Emotion Recognition System", page_icon="🙂", layout="wide")
st.title("Emotion Recognition System")
st.caption("Probabilistic facial-expression prototype. It does not establish a person's internal emotional state.")

@st.cache_resource
def load_system():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, emotions = load_model(MODEL_PATH, device)
    system = CognitiveEmotionSystem(model, emotions, DATABASE_URL, QUALITY_THRESHOLD,
                                    CONFIDENCE_THRESHOLD, STM_WINDOW, device)
    return system

with st.sidebar:
    session_id = st.text_input("Session ID", value="demo-session")
    st.write(f"Quality threshold: {QUALITY_THRESHOLD:.2f}")
    st.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    st.info("Images are processed in memory. Only prediction metadata is stored.")

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if uploaded:
    rgb = np.asarray(Image.open(uploaded).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    left, right = st.columns([1, 1.2])
    left.image(rgb, caption="Uploaded input", use_container_width=True)
    try:
        system = load_system()
        result = system.process(bgr, session_id.strip() or "demo-session")
        with right:
            st.subheader("Output")
            st.metric("Decision", result["knowledge"]["label"])
            st.write("Action:", result["knowledge"]["action"])
            st.write("Adjusted confidence:", f"{result['knowledge']['confidence']:.3f}")
            st.subheader("Perception")
            st.json({k: v for k, v in result["perception"].items() if k != "probabilities"})
            probabilities = result["perception"]["probabilities"]
            st.bar_chart({"emotion": list(probabilities), "probability": list(probabilities.values())},
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
else:
    st.info("Upload an image to inspect the full Perception → Attention → Memory → Knowledge → Output flow.")
