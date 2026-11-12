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

# --- Reasoning / Decision (Role A) ---
RECENCY_DECAY = float(os.getenv("RECENCY_DECAY", "0.6"))  # older frame weight = decay ** age
R2_BONUS = 0.05  # repeated-emotion bonus on the smoothed probability
UNCERTAIN_CONFIDENCE = 0.60  # R3
TREND_L1_THRESHOLD = 0.35  # newest frame vs. window mean => "changing"
UNSTABLE_RELIABILITY_FACTOR = 0.8  # reliability multiplier when trend == "changing"

ALTERNATIVES = ["continue", "re_explain", "pause", "easier_task", "ask_directly"]
CRITERIA = ["fit", "reliability", "disruption", "time_cost"]
CRITERIA_TYPES = ["benefit", "benefit", "cost", "cost"]
DEFAULT_WEIGHTS = (0.35, 0.25, 0.25, 0.15)
DISRUPTION = {"continue": 0.0, "ask_directly": 0.3, "re_explain": 0.4,
              "easier_task": 0.5, "pause": 0.7}
TIME_COST_MIN = {"continue": 0, "ask_directly": 1, "easier_task": 2,
                 "re_explain": 3, "pause": 5}
# Initial expert utility matrix M: rows = emotion, columns = ALTERNATIVES order.
UTILITY = {
    "angry": [0.1, 0.3, 0.8, 0.5, 0.6],
    "disgust": [0.2, 0.3, 0.4, 0.4, 0.7],
    "fear": [0.1, 0.4, 0.5, 0.6, 0.8],
    "happy": [0.9, 0.2, 0.1, 0.2, 0.2],
    "neutral": [0.8, 0.3, 0.1, 0.2, 0.3],
    "sad": [0.2, 0.3, 0.6, 0.5, 0.8],
    "surprise": [0.3, 0.8, 0.2, 0.2, 0.5],
}
