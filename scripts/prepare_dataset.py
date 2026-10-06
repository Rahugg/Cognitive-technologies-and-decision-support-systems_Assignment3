import argparse
import csv
import random
import re
from pathlib import Path

import pandas as pd

from src.emotion_system.config import DATA_DIR, EMOTIONS

LABEL_ALIASES = {
    # FER-2013 integer labels: angry, disgust, fear, happy, sad, surprise, neutral.
    "angry": "angry", "anger": "angry", "0": "angry",
    "disgust": "disgust", "1": "disgust",
    "fear": "fear", "scared": "fear", "2": "fear",
    "happy": "happy", "happiness": "happy", "joy": "happy", "3": "happy",
    "neutral": "neutral", "6": "neutral",
    "sad": "sad", "sadness": "sad", "4": "sad",
    "surprise": "surprise", "surprised": "surprise", "5": "surprise",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(value, source=None):
    key = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if source and source.lower().startswith("raf"):
        raf_labels = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
                      "5": "sad", "6": "angry", "7": "neutral"}
        if key in raf_labels:
            return raf_labels[key]
    return LABEL_ALIASES.get(key)


def rows_from_images(raw_root):
    rows = []
    for path in raw_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source = path.relative_to(raw_root).parts[0]
        label = next((normalized for part in reversed(path.parts)
                      if (normalized := normalize_label(part, source)) is not None), None)
        if label:
            rows.append({"path": str(path.resolve()), "label": label,
                         "source": source})
    return rows


def rows_from_fer_csv(raw_root):
    rows = []
    for path in raw_root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not {"emotion", "pixels"}.issubset(reader.fieldnames):
                    continue
                for index, row in enumerate(reader):
                    label = normalize_label(row["emotion"], "fer2013")
                    if label not in EMOTIONS:
                        continue
                    # Convert FER's pixel-string CSV records into portable PNG files.
                    import numpy as np
                    from PIL import Image
                    values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
                    side = int(len(values) ** 0.5)
                    if side * side != len(values):
                        continue
                    image_path = DATA_DIR / "processed" / "fer_csv_images" / f"{path.stem}_{index}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(values.reshape(side, side)).save(image_path)
                    rows.append({"path": str(image_path.resolve()), "label": label,
                                 "source": "fer2013"})
        except (OSError, UnicodeError):
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create normalized train/val/test index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class-source", type=int, default=1500,
                        help="Set 0 to keep all matching images")
    args = parser.parse_args()
    raw_root = DATA_DIR / "raw"
    rows = rows_from_images(raw_root) + rows_from_fer_csv(raw_root)
    if not rows:
        raise SystemExit(f"No labeled images found under {raw_root}. Check dataset layout and labels.")

    random.seed(args.seed)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["source"], row["label"]), []).append(row)
    split_rows = []
    for (source, label), group in sorted(grouped.items()):
        random.shuffle(group)
        if args.max_per_class_source:
            group = group[:args.max_per_class_source]
        count = len(group)
        train_end = min(max(1, int(count * 0.8)), max(1, count - 2))
        val_end = min(count - 1, train_end + max(1, int(count * 0.1)))
        for i, row in enumerate(group):
            row["split"] = "train" if i < train_end else "validation" if i < val_end else "test"
            split_rows.append(row)
    output = DATA_DIR / "processed" / "index.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(output, index=False)
    print(f"Indexed {len(split_rows)} images from {len(grouped)} source/class groups: {output}")
    print(pd.DataFrame(split_rows).groupby(["source", "split"]).size().to_string())


if __name__ == "__main__":
    main()
