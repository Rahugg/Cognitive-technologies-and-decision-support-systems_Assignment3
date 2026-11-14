import cv2
import numpy as np
import pandas as pd
import uuid
from pathlib import Path

from src.emotion_system.config import (CONFIDENCE_THRESHOLD, DATA_DIR, DATABASE_URL,
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
    print("Report: db_written =", state.report["db_written"])


def main():
    index_path = DATA_DIR / "processed/index.csv"
    if index_path.exists():
        sample_path = pd.read_csv(index_path).query("split == 'test'").iloc[0].path
    else:  # clean machine: use the bundled demo image
        sample_path = str(Path(__file__).resolve().parents[1] / "examples/open_source/smiling-face.jpg")
    image = cv2.imread(sample_path)
    if image is None:
        raise SystemExit(f"Could not read sample image: {sample_path}")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, emotions = load_model(MODEL_PATH, device)
    system = CognitiveEmotionSystem(model, emotions, DATABASE_URL, QUALITY_THRESHOLD,
                                    CONFIDENCE_THRESHOLD, STM_WINDOW, device)
    session = f"assignment3-demo-{uuid.uuid4().hex[:8]}"
    first = system.process(image, session)
    show("Example 1: clear test image", first)
    mild = cv2.GaussianBlur((image.astype(np.float32) * 0.90).clip(0, 255).astype(np.uint8), (3, 3), 0)
    second = system.process(mild, session)
    show("Example 2: mildly degraded image, same session", second)
    severe = cv2.GaussianBlur((image.astype(np.float32) * 0.03).clip(0, 255).astype(np.uint8), (31, 31), 0)
    third = system.process(severe, session)
    show("Example 3: heavily degraded image", third)
    if third["attention"]["accepted"]:
        raise RuntimeError("Low-quality demo failed its acceptance check; inspect the quality calculation.")


if __name__ == "__main__":
    main()
