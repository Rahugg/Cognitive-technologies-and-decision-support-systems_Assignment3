import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/emotion_cnn.pt"))
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://emotion:emotion@localhost:5433/emotiondb"
)
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.45"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
STM_WINDOW = int(os.getenv("STM_WINDOW", "5"))
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
