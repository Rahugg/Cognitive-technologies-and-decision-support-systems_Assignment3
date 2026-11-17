import cv2
import numpy as np
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emotion_system.config import (CONFIDENCE_THRESHOLD, DATABASE_URL,
                                       MODEL_PATH, QUALITY_THRESHOLD, STM_WINDOW)
from src.emotion_system.model import load_model
from src.emotion_system.pipeline import CognitiveEmotionSystem


def show(title, result):
    print(f"\n{title}")
    print("Perception:", result["perception"])
    print("Attention:", result["attention"])
    print("Memory:", {"previous": result["memory"]["previous"],
                      "short_term_size": result["memory"]["short_term_size"]})
    print("Knowledge:", result["knowledge"])
    state = result["state"]
    print("Reasoning:", {"trend": state.trend, "smoothed_confidence": round(state.smoothed_confidence, 3),
                         "label": state.label})
    for step in state.reasoning_steps:
        print("   ", step)
    print("Decision (DSS):", [(r["alternative"], round(r["score"], 3)) for r in state.ranking],
          "-> recommended:", state.recommended)
    print("Action:", state.action)
    print(state.explanation[-1])
    print("Report: db_written =", state.report["db_written"])


def main():
    samples = ROOT / "examples" / "open_source"
    image = cv2.imread(str(samples / "smiling-face.jpg"))
    severe = cv2.imread(str(samples / "dark-smiling-face.jpg"))
    if image is None or severe is None:
        raise SystemExit(f"Could not read both demo images in {samples}")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, emotions = load_model(MODEL_PATH, device)
    system = CognitiveEmotionSystem(model, emotions, DATABASE_URL, QUALITY_THRESHOLD,
                                    CONFIDENCE_THRESHOLD, STM_WINDOW, device)
    session = f"assignment3-demo-{uuid.uuid4().hex[:8]}"
    first = system.process(image, session, require_face=True)
    show("Example 1: clear demo image", first)
    mild = cv2.GaussianBlur((image.astype(np.float32) * 0.90).clip(0, 255).astype(np.uint8), (3, 3), 0)
    second = system.process(mild, session, require_face=True)
    show("Example 2: mildly degraded image, same session", second)
    third = system.process(severe, session, require_face=True)
    show("Example 3: dark image, same session", third)
    if not first["attention"]["accepted"] or not second["attention"]["accepted"]:
        raise RuntimeError("The clear and mildly degraded images must pass Attention.")
    if third["attention"]["accepted"] or third["state"].recommended != "ask_directly":
        raise RuntimeError("The dark image must trigger a check-in, even with earlier accepted frames.")


if __name__ == "__main__":
    main()
