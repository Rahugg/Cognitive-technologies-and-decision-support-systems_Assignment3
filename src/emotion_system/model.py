from pathlib import Path

import torch
from torch import nn

from .config import EMOTIONS


class EmotionCNN(nn.Module):
    """Small CNN for normalized 48x48 grayscale face crops."""

    def __init__(self, num_classes: int = len(EMOTIONS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def load_model(model_path: str | Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run: "
            "docker compose --profile data run --rm dataset-downloader, then "
            "docker compose --profile train run --rm trainer"
        )
    model = EmotionCNN().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()
    return model, EMOTIONS
