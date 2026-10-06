"""Reproducible, augmented and class-balanced training for the emotion CNN."""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.emotion_system.config import DATA_DIR, EMOTIONS, MODEL_PATH
from src.emotion_system.model import EmotionCNN


class FaceDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = {label: i for i, label in enumerate(EMOTIONS)}
        self.augment = augment
        self.training_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1), shear=5, fill=0),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), value=0),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as source:
            image = source.convert("L")
            # Match OpenCV INTER_AREA preprocessing used by the live camera.
            pixels = cv2.resize(np.asarray(image), (48, 48), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(pixels)
        if self.augment:
            tensor = self.training_transform(image)
        else:
            tensor = transforms.ToTensor()(image)
        return tensor, self.label_to_index[row.label]


def classification_metrics(model, loader, device, loss_fn=None):
    model.eval()
    confusion = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if loss_fn:
                total_loss += float(loss_fn(logits, labels)) * len(labels)
            predictions = logits.argmax(1)
            for actual, predicted in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                confusion[actual, predicted] += 1
            count += len(labels)
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0)
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted_count, out=np.zeros_like(true_positive, dtype=float), where=predicted_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=(precision + recall) > 0)
    return {
        "loss": total_loss / max(count, 1) if loss_fn else None,
        "accuracy": float(true_positive.sum() / max(count, 1)),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {name: {"support": int(support[i]), "recall": float(recall[i]),
                              "precision": float(precision[i]), "f1": float(f1[i])}
                        for i, name in enumerate(EMOTIONS)},
    }


def main():
    parser = argparse.ArgumentParser(description="Train the seven-class face-expression CNN")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "8")))
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))

    index_path = DATA_DIR / "processed" / "index.csv"
    if not index_path.exists():
        raise SystemExit("Dataset index missing. Run the dataset-downloader service first.")
    frame = pd.read_csv(index_path)
    train_frame = frame[frame.split == "train"].copy()
    val_frame = frame[frame.split == "validation"]
    test_frame = frame[frame.split == "test"]
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise SystemExit("Train, validation, and test splits must all contain images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(FaceDataset(train_frame, augment=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(FaceDataset(val_frame), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(FaceDataset(test_frame), batch_size=args.batch_size, shuffle=False, num_workers=0)

    label_indices = train_frame.label.map({label: i for i, label in enumerate(EMOTIONS)}).to_numpy()
    counts = np.bincount(label_indices, minlength=len(EMOTIONS)).astype(np.float32)
    # Square-root inverse frequency corrects imbalance without letting tiny classes dominate.
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights /= weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    model = EmotionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.04)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_name(MODEL_PATH.stem + ".candidate.pt")
    best_score, best_loss, stale_epochs = -1.0, float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss, total_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += float(loss.detach()) * len(labels)
            total_seen += len(labels)
        scheduler.step()
        val_metrics = classification_metrics(model, val_loader, device, loss_fn)
        row = {"epoch": epoch, "train_loss": total_train_loss / max(total_seen, 1),
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "validation_macro_f1": val_metrics["macro_f1"], "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = val_metrics["macro_f1"] > best_score + 1e-5 or (
            abs(val_metrics["macro_f1"] - best_score) <= 1e-5 and val_metrics["loss"] < best_loss
        )
        if improved:
            best_score, best_loss, stale_epochs = val_metrics["macro_f1"], val_metrics["loss"], 0
            torch.save({"model_state": model.state_dict(), "emotions": EMOTIONS,
                        "seed": args.seed, "epoch": epoch, "architecture": "residual_cnn_v1"}, temporary_model)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stop at epoch {epoch}; best validation macro-F1={best_score:.4f}", flush=True)
            break

    checkpoint = torch.load(temporary_model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = classification_metrics(model, test_loader, device, loss_fn)
    by_source = {}
    for source, source_rows in test_frame.groupby("source"):
        by_source[str(source)] = classification_metrics(
            model, DataLoader(FaceDataset(source_rows), batch_size=args.batch_size, shuffle=False, num_workers=0),
            device, loss_fn)
    metrics = {
        "seed": args.seed, "requested_epochs": args.epochs, "trained_epochs": len(history),
        "best_epoch": checkpoint["epoch"], "device": device,
        "architecture": "residual_cnn_v1", "augmentation": ["horizontal_flip", "mild_affine", "brightness_contrast", "random_erasing"],
        "class_weights": {label: float(weights[i]) for i, label in enumerate(EMOTIONS)},
        "dataset_counts": {"/".join(map(str, key)): int(value)
                           for key, value in frame.groupby(["source", "split"]).size().to_dict().items()},
        "history": history, "validation_best_macro_f1": best_score,
        "test": test_metrics, "test_by_source": by_source,
    }
    metrics_path = MODEL_PATH.with_suffix(".metrics.json")
    metrics_candidate = metrics_path.with_name(metrics_path.stem + ".candidate.json")
    metrics_candidate.write_text(json.dumps(metrics, indent=2))
    os.replace(temporary_model, MODEL_PATH)
    os.replace(metrics_candidate, metrics_path)
    print(f"Saved model: {MODEL_PATH}\nTest metrics: {json.dumps(test_metrics)}\nMetrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
