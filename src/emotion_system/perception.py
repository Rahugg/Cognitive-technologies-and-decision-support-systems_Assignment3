from pathlib import Path

import cv2
import numpy as np
import torch


class Perception:
    def __init__(self, model, emotions, device="cpu"):
        self.model = model
        self.emotions = emotions
        self.device = device
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(cascade_path))

    def process(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input must be a non-empty image array")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Image must be grayscale, BGR, or BGRA")

        faces = self.face_detector.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
        )
        face_found = len(faces) > 0
        if face_found:
            x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
            margin = int(max(width, height) * 0.12)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(gray.shape[1], x + width + margin), min(gray.shape[0], y + height + margin)
            roi = gray[y1:y2, x1:x2]
        else:
            # Public FER corpora often contain already-cropped faces.
            side = min(gray.shape[:2])
            y0, x0 = (gray.shape[0] - side) // 2, (gray.shape[1] - side) // 2
            roi = gray[y0:y0 + side, x0:x0 + side]

        crop = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).view(1, 1, 48, 48).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        brightness = float(np.mean(crop) / 255.0)
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        # Smooth, bounded proxy based on brightness and edge detail.
        brightness_score = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
        sharpness_score = min(1.0, sharpness / 120.0)
        quality = float(0.55 * brightness_score + 0.45 * sharpness_score)
        index = int(np.argmax(probabilities))
        return {
            "emotion": self.emotions[index],
            "confidence": float(probabilities[index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(self.emotions)},
            "brightness": brightness,
            "sharpness": sharpness,
            "quality_score": quality,
            "face_found": face_found,
            "face_crop": crop,
        }
