"""Compact residual CNN for seven facial-expression classes."""

from pathlib import Path

import torch

from torch import nn

from .config import EMOTIONS


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class EmotionCNN(nn.Module):
    """Residual feature extractor; input is a normalized 48x48 grayscale face."""

    def __init__(self, num_classes=len(EMOTIONS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            ResidualBlock(24, 24),
            ResidualBlock(24, 48, 2),
            ResidualBlock(48, 96, 2),
            ResidualBlock(96, 128, 2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(128 * 4, 192),
            nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(model_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run: "
            "docker compose --profile data run --rm dataset-downloader, then "
            "docker compose --profile train run --rm trainer"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = EmotionCNN().to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()
    return model, checkpoint.get("emotions", EMOTIONS)
