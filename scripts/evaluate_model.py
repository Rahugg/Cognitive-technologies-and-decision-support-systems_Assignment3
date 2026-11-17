import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.emotion_system.config import DATA_DIR, EMOTIONS, MODEL_PATH
from src.emotion_system.model import load_model


class TestDataset(Dataset):
    """Test images with the same preprocessing used during CNN training."""

    def __init__(self, frame, emotions):
        self.frame = frame.reset_index(drop=True)
        self.emotions = emotions
        self.label_to_index = {name: i for i, name in enumerate(emotions)}

        unknown = sorted(set(self.frame["label"]) - set(emotions))
        if unknown:
            raise ValueError(f"Unknown emotion labels in test index: {unknown}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(row["path"])

        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("L")
            pixels = cv2.resize(
                np.asarray(image),
                (48, 48),
                interpolation=cv2.INTER_AREA,
            )

        # Equivalent to transforms.ToTensor() for a grayscale PIL image.
        tensor = torch.from_numpy(pixels.copy()).float().unsqueeze(0) / 255.0
        label = self.label_to_index[row["label"]]

        return tensor, label


def calculate_metrics(confusion, emotions):
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class = {}
    precisions, recalls, f1_scores = [], [], []

    for i, emotion in enumerate(emotions):
        true_positive = int(confusion[i, i])
        support = int(confusion[i, :].sum())
        predicted = int(confusion[:, i].sum())

        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        per_class[emotion] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "samples": total,
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }


def evaluate(model, loader, device, emotions):
    confusion = np.zeros((len(emotions), len(emotions)), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            for actual, predicted in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                confusion[int(actual), int(predicted)] += 1

    return calculate_metrics(confusion, emotions), confusion


def save_confusion_matrix(confusion, emotions, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *emotions])
        for emotion, row in zip(emotions, confusion):
            writer.writerow([emotion, *[int(value) for value in row]])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an existing emotion CNN checkpoint without training."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N test images (0 means all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be zero or a positive number")

    index_path = DATA_DIR / "processed" / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {index_path}. "
            "Prepare the dataset before running evaluation."
        )

    index = pd.read_csv(index_path)
    required_columns = {"path", "label", "split"}
    missing_columns = required_columns - set(index.columns)
    if missing_columns:
        raise ValueError(f"index.csv is missing columns: {sorted(missing_columns)}")

    test_frame = index[index["split"] == "test"].copy()
    if test_frame.empty:
        raise ValueError("No test samples found in index.csv")

    if args.limit:
        test_frame = test_frame.head(args.limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_emotions = load_model(MODEL_PATH, device)

    if list(checkpoint_emotions) != list(EMOTIONS):
        raise ValueError(
            "Checkpoint emotion order does not match config.EMOTIONS.\n"
            f"Checkpoint: {checkpoint_emotions}\n"
            f"Config:     {EMOTIONS}"
        )

    dataset = TestDataset(test_frame, EMOTIONS)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    overall, confusion = evaluate(model, loader, device, EMOTIONS)

    results = {
        "evaluation_type": "limited_test_subset" if args.limit else "full_test_split",
        "checkpoint": Path(MODEL_PATH).name,
        "dataset_index": "data/processed/index.csv",    
        "device": str(device),
        "emotions": EMOTIONS,
        "metrics": overall,
    }

    if "source" in test_frame.columns:
        results["by_source"] = {}
        for source in sorted(test_frame["source"].dropna().unique()):
            source_frame = test_frame[test_frame["source"] == source]
            source_dataset = TestDataset(source_frame, EMOTIONS)
            source_loader = DataLoader(
                source_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            source_metrics, _ = evaluate(
                model, source_loader, device, EMOTIONS
            )
            results["by_source"][str(source)] = source_metrics

    partial = ".partial" if args.limit else ""
    output_metrics = Path(MODEL_PATH).parent / (
        "emotion_cnn.evaluation.partial.json" if args.limit else "emotion_cnn.evaluation.metrics.json")
    output_confusion = Path(MODEL_PATH).parent / f"emotion_cnn.confusion_matrix{partial}.csv"

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(confusion, EMOTIONS, output_confusion)

    print("\n=== CNN Evaluation ===")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Test samples: {overall['samples']}")
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Macro precision: {overall['macro_precision']:.4f}")
    print(f"Macro recall: {overall['macro_recall']:.4f}")
    print(f"Macro F1: {overall['macro_f1']:.4f}")

    print("\nPer-class results:")
    for emotion, values in overall["per_class"].items():
        print(
            f"{emotion:10s} "
            f"support={values['support']:4d} "
            f"precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )

    print(f"\nSaved metrics: {output_metrics}")
    print(f"Saved confusion matrix: {output_confusion}")


if __name__ == "__main__":
    main()
