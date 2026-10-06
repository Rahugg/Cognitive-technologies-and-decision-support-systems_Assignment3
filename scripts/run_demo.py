import cv2
import numpy as np
import pandas as pd
import uuid

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


def main():
    index = pd.read_csv(DATA_DIR / "processed/index.csv")
    sample = index[index.split == "test"].iloc[0]
    image = cv2.imread(sample.path)
    if image is None:
        raise SystemExit(f"Could not read sample image: {sample.path}")
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
