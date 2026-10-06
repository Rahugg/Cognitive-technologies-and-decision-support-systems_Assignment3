import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.emotion_system.config import DATA_DIR, EMOTIONS, MODEL_PATH
from src.emotion_system.model import EmotionCNN


class FaceDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = {label: i for i, label in enumerate(EMOTIONS)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row.path).convert("L").resize((48, 48))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).unsqueeze(0), self.label_to_index[row.label]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(loss_fn(logits, labels)) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
    return {"loss": total_loss / max(count, 1), "accuracy": correct / max(count, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train the seven-class face-expression CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    index_path = DATA_DIR / "processed" / "index.csv"
    if not index_path.exists():
        raise SystemExit("Dataset index missing. Run the dataset-downloader service first.")
    frame = pd.read_csv(index_path)
    train_frame = frame[frame.split == "train"]
    val_frame = frame[frame.split == "validation"]
    test_frame = frame[frame.split == "test"]
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise SystemExit("Train, validation, and test splits must all contain images.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(FaceDataset(train_frame), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(FaceDataset(val_frame), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(FaceDataset(test_frame), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    model = EmotionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step()
            running_loss += float(loss) * len(labels)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_frame),
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"]}
        history.append(row)
        print(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "emotions": EMOTIONS,
                        "seed": args.seed}, MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {
        "seed": args.seed, "epochs": args.epochs, "device": device,
        "dataset_counts": frame.groupby(["source", "split"]).size().to_dict(),
        "history": history,
        "test": evaluate(model, test_loader, loss_fn, device),
    }
    metrics["dataset_counts"] = {"/".join(map(str, key)): int(value)
                                 for key, value in metrics["dataset_counts"].items()}
    metrics_path = MODEL_PATH.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}\nTest metrics: {metrics['test']}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
