from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import DATA_DIR, build_video_writer, ensure_dir, save_image


@dataclass(frozen=True)
class SampleDataPaths:
    image_path: Path
    video_path: Path


def _draw_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    half = height // 2
    for y in range(half):
        blend = y / max(half - 1, 1)
        color = (
            int(220 - 80 * blend),
            int(180 + 20 * blend),
            int(110 + 50 * blend),
        )
        frame[y, :] = color

    frame[half:, :] = (55, 55, 55)

    shoulder = np.array(
        [
            [0, height],
            [int(width * 0.1), int(height * 0.55)],
            [int(width * 0.18), int(height * 0.55)],
            [int(width * 0.02), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [shoulder], (45, 110, 45))

    shoulder = np.array(
        [
            [width, height],
            [int(width * 0.9), int(height * 0.55)],
            [int(width * 0.82), int(height * 0.55)],
            [int(width * 0.98), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [shoulder], (45, 110, 45))


def _lane_points(width: int, height: int, frame_index: int, total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    progress = frame_index / max(total_frames - 1, 1)
    sway = np.sin(progress * np.pi * 2.0) * width * 0.015
    left = np.array(
        [
            [int(width * 0.24 + sway), height],
            [int(width * 0.44 + sway * 0.4), int(height * 0.62)],
        ],
        dtype=np.int32,
    )
    right = np.array(
        [
            [int(width * 0.76 + sway), height],
            [int(width * 0.56 + sway * 0.4), int(height * 0.62)],
        ],
        dtype=np.int32,
    )
    return left, right


def generate_lane_frame(
    width: int = 1280,
    height: int = 720,
    frame_index: int = 0,
    total_frames: int = 90,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_background(frame)

    left_points, right_points = _lane_points(width, height, frame_index, total_frames)

    road = np.array(
        [
            [int(width * 0.08), height],
            [left_points[1, 0], left_points[1, 1]],
            [right_points[1, 0], right_points[1, 1]],
            [int(width * 0.92), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [road], (60, 60, 60))

    cv2.line(frame, tuple(left_points[0]), tuple(left_points[1]), (255, 255, 255), 16)
    cv2.line(frame, tuple(right_points[0]), tuple(right_points[1]), (255, 255, 255), 16)

    horizon_y = int(height * 0.58)
    cv2.line(frame, (0, horizon_y), (width, horizon_y), (100, 100, 100), 2)

    shade_strength = int(20 + 10 * np.cos(frame_index / max(total_frames, 1) * np.pi))
    overlay = np.full_like(frame, shade_strength)
    frame = cv2.subtract(frame, overlay)

    return frame


def generate_sample_assets(
    image_path: Path | None = None,
    video_path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    total_frames: int = 90,
    fps: float = 24.0,
    overwrite: bool = False,
) -> SampleDataPaths:
    ensure_dir(DATA_DIR)
    image_path = image_path or DATA_DIR / "sample_lane_frame.png"
    video_path = video_path or DATA_DIR / "sample_lane_video.mp4"

    first_frame = generate_lane_frame(width=width, height=height, frame_index=0, total_frames=total_frames)
    if overwrite or not image_path.exists():
        save_image(image_path, first_frame)

    if overwrite or not video_path.exists():
        writer = build_video_writer(video_path, (width, height), fps)
        try:
            for frame_index in range(total_frames):
                frame = generate_lane_frame(width=width, height=height, frame_index=frame_index, total_frames=total_frames)
                writer.write(frame)
        finally:
            writer.release()

    return SampleDataPaths(image_path=image_path, video_path=video_path)
