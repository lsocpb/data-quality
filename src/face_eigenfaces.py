from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FACE_SIZE = (100, 100)
DEFAULT_THRESHOLD = 6000.0
NUM_COMPONENTS = 80


@dataclass(frozen=True)
class FaceVerificationResult:
    claimed_user: str
    matched: bool
    confidence: float
    threshold: float
    predicted_user: str | None


def _haar_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def _crop_face(frame: np.ndarray, detector: cv2.CascadeClassifier) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y : y + h, x : x + w]
    return gray


def load_faces(
    faces_dir: str | Path,
) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    """
    Loads face images from faces/<user>/ directories.
    Returns (images, integer_labels, label_map).
    label_map maps integer label → username string.
    """
    faces_dir = Path(faces_dir)
    detector = _haar_detector()
    images: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[int, str] = {}
    label_id = 0

    for user_dir in sorted(faces_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        user_images: list[np.ndarray] = []
        for img_path in sorted(user_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = _crop_face(img, detector)
            user_images.append(cv2.resize(face, FACE_SIZE))

        if user_images:
            label_map[label_id] = user_dir.name
            for face_img in user_images:
                images.append(face_img)
                labels.append(label_id)
            label_id += 1

    return images, labels, label_map


def train_eigenfaces(
    images: list[np.ndarray],
    labels: list[int],
    *,
    num_components: int = NUM_COMPONENTS,
) -> cv2.face.EigenFaceRecognizer:
    recognizer = cv2.face.EigenFaceRecognizer_create(num_components=num_components)
    recognizer.train(images, np.array(labels, dtype=np.int32))
    return recognizer


def verify_face(
    recognizer: cv2.face.EigenFaceRecognizer,
    label_map: dict[int, str],
    frame: np.ndarray,
    claimed_user: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> FaceVerificationResult:
    detector = _haar_detector()
    face = _crop_face(frame, detector)
    face_resized = cv2.resize(face, FACE_SIZE)

    predicted_label, confidence = recognizer.predict(face_resized)
    predicted_user = label_map.get(predicted_label)

    matched = (predicted_user == claimed_user) and (confidence < threshold)

    return FaceVerificationResult(
        claimed_user=claimed_user,
        matched=matched,
        confidence=round(float(confidence), 1),
        threshold=threshold,
        predicted_user=predicted_user,
    )
