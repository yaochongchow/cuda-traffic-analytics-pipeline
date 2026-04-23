from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import DOCS_ASSETS_DIR, build_video_writer, ensure_dir, iter_video_frames, save_image
from .lane_detection import LaneDetectionResult, run_lane_detection


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _label_image(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 50), (20, 20, 20), -1)
    cv2.putText(labeled, label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def create_stage_collage(source_frame: np.ndarray, result: LaneDetectionResult) -> np.ndarray:
    panels = [
        _label_image(source_frame, "Input"),
        _label_image(_to_bgr(result.edges), "Canny Edges"),
        _label_image(_to_bgr(result.roi_edges), "ROI Mask"),
        _label_image(result.output, "Lane Overlay"),
    ]
    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    return np.vstack((top, bottom))


def create_hero_image(source_frame: np.ndarray, result: LaneDetectionResult) -> np.ndarray:
    before = _label_image(source_frame, "Original")
    after = _label_image(result.output, "CPU Lane Detection")
    return np.hstack((before, after))


def write_portfolio_assets(
    source_frame: np.ndarray,
    sample_video_path: Path,
    image_name: str = "lane_detection_hero.png",
    stages_name: str = "lane_detection_stages.png",
    video_name: str = "lane_detection_demo.mp4",
    max_frames: int = 72,
    fps: float = 24.0,
) -> tuple[Path, Path, Path]:
    ensure_dir(DOCS_ASSETS_DIR)
    result = run_lane_detection(source_frame)

    hero_path = DOCS_ASSETS_DIR / image_name
    stages_path = DOCS_ASSETS_DIR / stages_name
    video_path = DOCS_ASSETS_DIR / video_name

    save_image(hero_path, create_hero_image(source_frame, result))
    save_image(stages_path, create_stage_collage(source_frame, result))

    writer = build_video_writer(video_path, (source_frame.shape[1] * 2, source_frame.shape[0]), fps)
    try:
        for frame_index, frame in enumerate(iter_video_frames(sample_video_path)):
            if frame_index >= max_frames:
                break
            processed = run_lane_detection(frame)
            combined = create_hero_image(frame, processed)
            writer.write(combined)
    finally:
        writer.release()

    return hero_path, stages_path, video_path

