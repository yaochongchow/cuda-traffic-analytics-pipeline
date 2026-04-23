from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_ASSETS_DIR = ROOT / "docs" / "assets"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def build_video_writer(path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    ensure_dir(path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def iter_video_frames(video_path: Path) -> Iterable[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to load image: {image_path}")
    return image

