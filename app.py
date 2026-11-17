import hashlib
import io
import json
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from psycopg import Error as DatabaseError
from streamlit_webrtc import webrtc_streamer

from dss.model import rank
from src.emotion_system import decision, learning
from src.emotion_system.config import (CONFIDENCE_THRESHOLD, DATABASE_URL, MODEL_PATH,
                                       QUALITY_THRESHOLD, STM_WINDOW)
from src.emotion_system.model import load_model
from src.emotion_system.pipeline import CognitiveEmotionSystem


SAMPLE_DIR = Path(__file__).resolve().parent / "examples" / "open_source"
UPLOAD_SAMPLES = {
    "Clear smiling face": "smiling-face.jpg",
    "Dark frame": "dark-smiling-face.jpg",
    "Sad face": "sad-face.jpg",
    "Angry face": "angry-face.jpg",
    "Fearful face": "fear-face.jpg",
    "Laughing face": "laughing-face.jpg",
    "Surprised face": "surprised-face.jpg",
}
ACTION_LABELS = {
    "continue": "Continue the lesson",
    "re_explain": "Explain it differently",
    "pause": "Take a short break",
    "easier_task": "Offer an easier task",
    "ask_directly": "Ask the student directly",
}
CRITERION_LABELS = {
    "fit": "Expression fit",
    "reliability": "Signal reliability",
    "disruption": "Lesson disruption",
    "time_cost": "Time cost",
}


@st.cache_resource(show_spinner="Loading the expression model…")
def load_system():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, emotions = load_model(MODEL_PATH, device)
    return CognitiveEmotionSystem(model, emotions, DATABASE_URL, QUALITY_THRESHOLD,
                                  CONFIDENCE_THRESHOLD, STM_WINDOW, device), threading.Lock()


def recommendation(result, alternatives):
    return alternatives[result["order"][0]]


def ranking_table(result, alternatives, criteria):
    return pd.DataFrame([
        {"Rank": result["ranks"][i], "Alternative": ACTION_LABELS[alternatives[i]],
         "Score": result["scores"][i],
         **{CRITERION_LABELS[name]: value for name, value in
            zip(criteria, result["contributions"][i])}}
        for i in result["order"]
    ])


def show_sensitivity(matrix, weights, types, alternatives, criterion_index, criteria):
    others = [weight if i != criterion_index else 0 for i, weight in enumerate(weights)]
    if sum(others) == 0:
        others = [int(i != criterion_index) for i in range(len(weights))]
        caption = "The other criteria share the remaining weight equally for this sweep."
    else:
        caption = "The other criteria keep their relative weights."
    total = sum(others)
    samples = []
    winners = []
    for percentage in range(0, 101, 2):
        share = percentage / 100
        scenario = [(1 - share) * weight / total for weight in others]
        scenario[criterion_index] = share
        result = rank(matrix, scenario, types)
        winners.append(recommendation(result, alternatives))
        samples.append({"Weight (%)": percentage, **{
            ACTION_LABELS[alternative]: score
            for alternative, score in zip(alternatives, result["scores"])
        }})
    st.caption(f"Sweeping {CRITERION_LABELS[criteria[criterion_index]]} from 0% to 100%. {caption}")
    st.line_chart(pd.DataFrame(samples).set_index("Weight (%)"), height=270)
    changes = [f"{i * 2}% · {ACTION_LABELS[winner]}" for i, winner in enumerate(winners)
               if i == 0 or winner != winners[i - 1]]
    st.caption("First place at sampled weights — " + " → ".join(changes))


def show_dss(state, prefix):
    data = state.criteria_matrix
    criteria, types = data["criteria"], data["types"]
    alternatives = state.alternatives
    matrix = [data["values"][alternative] for alternative in alternatives]
    defaults = [round(weight * 100) for weight in data["weights"]]
    check_in = decision.needs_check_in(state)
    if check_in:
        st.warning(f"Recommended: **{ACTION_LABELS[state.recommended]}**. "
                   "The frame gives insufficient evidence for a teaching intervention.")

    st.subheader("What-if weights")
    st.caption("Change the weights to compare teaching actions for this frame. We normalize the weights to 100%.")
    keys = [f"{prefix}-weight-{name}" for name in criteria]
    for key, default in zip(keys, defaults):
        st.session_state.setdefault(key, default)
    if st.button("Reset weights", key=f"{prefix}-reset"):
        for key, default in zip(keys, defaults):
            st.session_state[key] = default
    weights = [column.slider(f"{CRITERION_LABELS[name]} ({kind})", 0, 100, key=key)
               for column, name, kind, key in
               zip(st.columns(len(criteria)), criteria, types, keys)]
    if not any(weights):
        st.warning("Give at least one criterion a positive weight to calculate a ranking.")
        return None, weights

    result = rank(matrix, weights, types)
    best = recommendation(result, alternatives)
    if check_in:
        st.info(f"First in the WSM comparison: **{ACTION_LABELS[best]}**. "
                "The weights only change the score comparison for this frame.")
    elif best == state.recommended:
        st.success(f"Recommended with these weights: **{ACTION_LABELS[best]}**")
    else:
        st.success(f"What-if recommendation: **{ACTION_LABELS[best]}**")
    normalized = " · ".join(f"{CRITERION_LABELS[name]} {weight:.1%}"
                            for name, weight in zip(criteria, result["weights"]))
    st.caption(f"Active weights — {normalized}")
    st.caption(f"Recommendation at analysis time — {ACTION_LABELS[state.recommended]}")
    comparison = result["comparison"]
    runner = ACTION_LABELS[alternatives[comparison["runner_up"]]]
    if abs(comparison["margin"]) < 1e-12:
        st.write(f"The first-ranked action ties with “{runner}”. The model breaks ties by alternative order.")
    else:
        deltas = sorted(zip(criteria, comparison["contributions"]), key=lambda item: -item[1])
        name, advantage = deltas[0]
        why = f"{CRITERION_LABELS[name]} gives the largest advantage ({advantage:+.3f})"
        if deltas[-1][1] < -1e-12:
            name, disadvantage = deltas[-1]
            why += f", while {CRITERION_LABELS[name].lower()} favors the runner-up ({disadvantage:+.3f})"
        st.write(f"The first-ranked action leads “{runner}” by **{comparison['margin']:.3f}**. {why}.")

    table = ranking_table(result, alternatives, criteria)
    table["Rank"] = table["Rank"].astype(str)
    signed = {"Rank": "", "Alternative": "Winner − runner-up", "Score": comparison["margin"],
              **dict(zip((CRITERION_LABELS[name] for name in criteria), comparison["contributions"]))}
    table = pd.concat([table, pd.DataFrame([signed])], ignore_index=True)
    st.dataframe(table.round(3), hide_index=True, use_container_width=True)
    st.caption("The criterion columns show weighted contributions. They sum to each WSM score.")
    with st.expander("Raw criteria and scoring method"):
        values = pd.DataFrame(matrix, index=[ACTION_LABELS[alt] for alt in alternatives],
                              columns=[CRITERION_LABELS[c] for c in criteria])
        st.dataframe(values.round(3), use_container_width=True)
        st.write("Weighted Sum Model (WSM) scales every column to 0–1, rewards lower costs "
                 "and adds weighted contributions. Equal values in a column contribute equally. "
                 "Fit uses an expert utility matrix, which can be updated from tutor feedback.")
    with st.expander("Sensitivity chart", expanded=True):
        criterion = st.selectbox("Criterion to sweep", criteria, format_func=CRITERION_LABELS.get,
                                 key=f"{prefix}-sweep")
        show_sensitivity(matrix, weights, types, alternatives, criteria.index(criterion), criteria)
    return result, weights


def show_learning(state, current, weights, system, lock, prefix):
    st.caption(next(line for line in reversed(state.explanation) if line.startswith("Learning:")))
    report = state.report
    key = f"{prefix}-feedback-{state.session_id}-{report['decision_id']}"
    saved = st.session_state.get(f"{prefix}-latest-feedback")
    if saved and saved["key"] == key:
        summary = saved["summary"]
        version = summary["version"]
        if summary["updated"]:
            st.success(f"Feedback saved. Utility matrix version {version} recorded.")
        elif summary["unchanged"]:
            st.info("Feedback saved. The affected utility values were already at their limits.")
        else:
            st.info("Feedback saved. The expression was unconfirmed, so the utility matrix stayed the same.")
        st.write(f"**{saved['label']}** → **{saved['choice']}** chosen by the tutor")
        if saved["before"] != saved["after"]:
            st.caption(f"First place changed from {saved['before']} to {saved['after']}.")
        else:
            st.caption(f"First place stayed with {saved['after']}.")
        st.dataframe(saved["table"], hide_index=True, use_container_width=True)
        st.caption("Same frame, history and weights. Learning affects the fit for future analyses.")
        return

    if current is None:
        st.info("Set a positive DSS weight before recording a tutoring decision.")
        return
    recommended = (state.recommended if decision.needs_check_in(state)
                   else recommendation(current, state.alternatives))
    st.write(f"Current DSS recommendation: **{ACTION_LABELS[recommended]}**. Choose what you did as the tutor.")
    with st.form(key):
        selected = st.selectbox("Tutor's action", state.alternatives, index=None,
                                format_func=ACTION_LABELS.get)
        confirmed = st.checkbox(f"The predicted expression ({report['emotion']}) looked accurate.") if state.accepted else False
        if not state.accepted:
            st.caption("Attention rejected this frame. We can record the choice without learning an expression-to-action preference.")
        submitted = st.form_submit_button("Save tutor decision")
    if not submitted:
        return
    if selected is None:
        st.warning("Choose the action you took before saving.")
        return

    data = state.criteria_matrix
    try:
        with lock:
            # Compare the exact frame and weights on either side of this update.
            before = decision.run(replace(state, explanation=[]), system.memory)
            summary = learning.submit_feedback(system.memory, state.session_id, report["emotion"],
                                               recommended, selected, confirmed, report["decision_id"])
            after = decision.run(replace(state, explanation=[]), system.memory)
    except DatabaseError:
        st.error("The database is unavailable. Reconnect it and analyze a frame to process any queued feedback.")
        return
    before_rank, after_rank = [rank([snapshot.criteria_matrix["values"][alt] for alt in state.alternatives],
                                   weights, data["types"]) for snapshot in (before, after)]
    table = pd.DataFrame([
        {"Action": ACTION_LABELS[alternative], "Before rank": before_rank["ranks"][i],
         "After rank": after_rank["ranks"][i], "Before score": before_rank["scores"][i],
         "After score": after_rank["scores"][i], "Before fit": before.criteria_matrix["values"][alternative][0],
         "After fit": after.criteria_matrix["values"][alternative][0]}
        for i, alternative in enumerate(state.alternatives)
    ]).round(3).sort_values("After rank")
    st.session_state[f"{prefix}-latest-feedback"] = {
        "key": key, "summary": summary, "table": table, "label": ACTION_LABELS[recommended],
        "choice": ACTION_LABELS[selected], "before": ACTION_LABELS[recommendation(before_rank, state.alternatives)],
        "after": ACTION_LABELS[recommendation(after_rank, state.alternatives)],
    }
    st.rerun()


def show_cycle(result, prefix, system, lock):
    state = result["state"]
    if state.accepted:
        if state.label == "uncertain":
            st.warning("Frame accepted. The expression is unclear.")
        else:
            st.success("Frame accepted.")
    else:
        st.warning(f"Frame rejected ({state.reject_reason}). Check with the student before acting on the image.")
    tabs = st.tabs(["DSS", "Perception", "Attention", "Memory", "Reasoning", "Decision", "Action", "Learning"])
    with tabs[0]:
        current, weights = show_dss(state, prefix)
    with tabs[1]:
        columns = st.columns(4)
        columns[0].metric("CNN expression", state.emotion.capitalize())
        columns[1].metric("Confidence", f"{state.confidence:.0%}")
        columns[2].metric("Frame quality", f"{state.quality:.2f}")
        columns[3].metric("Face found", "Yes" if state.face_found else "No")
        st.bar_chart(pd.Series(state.probs, name="CNN probability"), horizontal=True, height=240)
        st.caption(f"Brightness {state.brightness:.2f} · Laplacian variance {state.sharpness:.1f}")
    with tabs[2]:
        attention = result["attention"]
        st.metric("Frame gate", "Accepted" if state.accepted else "Rejected")
        if state.reject_reason:
            st.write(state.reject_reason.capitalize())
        st.write(f"Face required · Quality ≥ {attention['quality_threshold']:.2f} · "
                 f"Confidence ≥ {attention['confidence_threshold']:.2f}")
        st.metric("Relevance", f"{state.relevance:.2f}")
    with tabs[3]:
        st.metric("Frames in short-term window", f"{result['memory']['short_term_size']} / {STM_WINDOW}")
        st.write(f"Previous accepted expression: {state.previous_emotion or 'none'}")
        st.write(state.reasoning_steps[0])
        st.caption("The window feeds Reasoning with probabilities from accepted frames.")
    with tabs[4]:
        st.write(f"**{state.label.replace('_', ' ').capitalize()}** · Trend {state.trend.replace('_', ' ')}")
        st.bar_chart(pd.Series(state.smoothed_probs, name="Smoothed probability"), horizontal=True, height=240)
        for step in state.reasoning_steps:
            st.write(step)
    with tabs[5]:
        st.write(f"**{ACTION_LABELS[state.recommended]}** · Reliability {state.criteria_matrix['reliability']:.2f}")
        st.caption("WSM ranking at the default weights. When evidence is insufficient, the app recommends checking in.")
        table = pd.DataFrame([{"Rank": row["rank"], "Alternative": ACTION_LABELS[row["alternative"]],
                               "Score": row["score"]} for row in state.ranking])
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
    with tabs[6]:
        st.write(state.action.replace(state.recommended, ACTION_LABELS[state.recommended], 1))
        st.write("Decision metadata saved to PostgreSQL." if state.report["db_written"] else
                 "Decision metadata kept in memory for this app run.")
        st.download_button("Download decision report", json.dumps(state.report, indent=2),
                           "tutor-decision.json", "application/json", key=f"{prefix}-report")
        st.caption("The tutor decides what action to take.")
    with tabs[7]:
        show_learning(state, current, weights, system, lock, prefix)


def photo_workspace(system, lock, session_id):
    source = st.radio("Frame source", ["Demo image", "Upload an image"], horizontal=True)
    image_bytes = None
    if source == "Demo image":
        name = st.selectbox("Demo frame", UPLOAD_SAMPLES)
        image_bytes = (SAMPLE_DIR / UPLOAD_SAMPLES[name]).read_bytes()
    else:
        uploaded = st.file_uploader("Choose a JPG or PNG frame", type=["jpg", "jpeg", "png"])
        if uploaded:
            name, image_bytes = uploaded.name, uploaded.getvalue()
    if not image_bytes:
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        st.error("Could not read this image. Choose a valid JPG or PNG.")
        return
    image_key = hashlib.sha256(image_bytes).hexdigest()
    preview, controls = st.columns([1, 3], gap="large")
    preview.image(rgb, caption=name, width=210)
    run = controls.button("Analyze frame", type="primary")
    controls.caption("Repeat analysis on the same frame to inspect short-term memory. Choose another frame for a fresh window.")
    if run:
        try:
            with lock:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                result = system.process(bgr, f"{session_id}-photo-{image_key}", require_face=True)
        except (cv2.error, DatabaseError, RuntimeError, ValueError) as error:
            st.error(f"Could not analyze this frame: {error}")
            st.session_state.pop("photo_analysis", None)
        else:
            st.session_state.photo_analysis = {"image_key": image_key, "result": result}
    saved = st.session_state.get("photo_analysis")
    if saved and saved["image_key"] == image_key:
        show_cycle(saved["result"], "photo", system, lock)


def camera_workspace(system, lock, session_id):
    st.write("Start the camera, then select **Inspect latest frame** to open its decision report.")
    if "live_camera_state" not in st.session_state:
        st.session_state.live_camera_state = {"last": 0.0, "result": None, "error": None,
                                             "lock": threading.Lock()}
    camera = st.session_state.live_camera_state

    def video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        with camera["lock"]:
            analyze = now - camera["last"] >= 1.0
            if analyze:
                camera["last"] = now
        if analyze:
            try:
                with lock:
                    result = system.process(image, f"{session_id}-camera", require_face=True)
            except Exception as error:
                with camera["lock"]:
                    camera["error"] = str(error)
            else:
                with camera["lock"]:
                    camera.update(result=result, error=None)
        with camera["lock"]:
            result, error = camera["result"], camera["error"]
        label = "Waiting for analysis"
        if error:
            label = "Inference error"
        elif result:
            state = result["state"]
            label = (f"{state.label.replace('_', ' ').title()} | {state.smoothed_confidence:.0%}"
                     if state.accepted else "Frame rejected")
        cv2.rectangle(image, (10, 10), (min(image.shape[1], 365), 50), (28, 30, 34), -1)
        cv2.putText(image, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(key="emotion-live-camera", video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    if st.button("Inspect latest frame"):
        with camera["lock"]:
            result, error = camera["result"], camera["error"]
        if error:
            st.error(f"Camera analysis failed: {error}")
        elif result is None:
            st.info("Start the camera and wait for the first frame to be analyzed.")
        else:
            st.session_state.camera_snapshot = result
    if "camera_snapshot" in st.session_state:
        show_cycle(st.session_state.camera_snapshot, "camera", system, lock)


st.set_page_config(page_title="Tutor Decision Support", layout="wide")
st.title("Tutor Decision Support")
st.caption("Compare teaching actions using visible facial expressions. Only the student can tell you how they feel.")
if not MODEL_PATH.is_file():
    st.error(f"CNN checkpoint missing at {MODEL_PATH}. Restore `models/emotion_cnn.pt` from the submission.")
    st.stop()
system, inference_lock = load_system()
if "tutor_session" not in st.session_state:
    st.session_state.tutor_session = uuid.uuid4().hex
session_id = st.session_state.tutor_session
with st.sidebar:
    st.subheader("Session storage")
    st.write("PostgreSQL" if system.memory.db_ok else "In memory · lasts until the app stops")
    st.caption("Decision and feedback metadata can be stored. Frame pixels are processed in memory.")
    st.caption("Use only consented images. Demo faces are CC0.")

photo_tab, live_tab = st.tabs(["Analyze a frame", "Live camera"])
with photo_tab:
    photo_workspace(system, inference_lock, session_id)
with live_tab:
    camera_workspace(system, inference_lock, session_id)
