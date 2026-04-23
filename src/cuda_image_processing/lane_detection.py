from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import cv2
import numpy as np


@dataclass
class LaneDetectionResult:
    grayscale: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    roi_edges: np.ndarray
    line_segments: np.ndarray | None
    lanes: list[tuple[int, int, int, int]]
    line_overlay: np.ndarray
    output: np.ndarray
    timings_ms: dict[str, float]


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_roi_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    polygon = np.array(
        [
            [
                (int(width * 0.08), height),
                (int(width * 0.43), int(height * 0.60)),
                (int(width * 0.57), int(height * 0.60)),
                (int(width * 0.92), height),
            ]
        ],
        dtype=np.int32,
    )

    mask = np.zeros_like(image)
    fill_value = 255 if image.ndim == 2 else (255,) * image.shape[2]
    cv2.fillPoly(mask, polygon, fill_value)
    return cv2.bitwise_and(image, mask)


def detect_line_segments(masked_edges: np.ndarray) -> np.ndarray | None:
    height, width = masked_edges.shape[:2]
    min_length = max(int(width * 0.08), 40)
    max_gap = max(int(width * 0.02), 20)
    return cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )


def _weighted_average_line(points: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    total_weight = sum(weight for _, _, weight in points)
    slope = sum(slope * weight for slope, _, weight in points) / total_weight
    intercept = sum(intercept * weight for _, intercept, weight in points) / total_weight
    return slope, intercept


def _make_line_points(
    slope: float,
    intercept: float,
    width: int,
    height: int,
    y_bottom: int,
    y_top: int,
) -> tuple[int, int, int, int] | None:
    if abs(slope) < 1e-6:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)

    x_bottom = int(np.clip(x_bottom, 0, width - 1))
    x_top = int(np.clip(x_top, 0, width - 1))
    return x_bottom, y_bottom, x_top, y_top


def average_lane_lines(frame: np.ndarray, line_segments: np.ndarray | None) -> list[tuple[int, int, int, int]]:
    if line_segments is None:
        return []

    height, width = frame.shape[:2]
    left_fits: list[tuple[float, float, float]] = []
    right_fits: list[tuple[float, float, float]] = []

    for segment in line_segments:
        x1, y1, x2, y2 = segment[0]
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.45 or abs(slope) > 3.5:
            continue

        intercept = y1 - slope * x1
        length = float(np.hypot(x2 - x1, y2 - y1))
        if slope < 0:
            left_fits.append((slope, intercept, length))
        else:
            right_fits.append((slope, intercept, length))

    lanes: list[tuple[int, int, int, int]] = []
    y_bottom = height - 1
    y_top = int(height * 0.62)

    for fit in (_weighted_average_line(left_fits), _weighted_average_line(right_fits)):
        if fit is None:
            continue
        line = _make_line_points(fit[0], fit[1], width, height, y_bottom, y_top)
        if line is not None:
            lanes.append(line)

    return lanes


def draw_lane_lines(frame: np.ndarray, lanes: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = np.zeros_like(frame)
    for x1, y1, x2, y2 in lanes:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 12)
    return overlay


def run_lane_detection(frame: np.ndarray) -> LaneDetectionResult:
    timings_ms: dict[str, float] = {}
    total_start = perf_counter_ns()

    stage_start = perf_counter_ns()
    grayscale = to_grayscale(frame)
    timings_ms["grayscale"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    blurred = apply_blur(grayscale)
    timings_ms["blur"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    edges = detect_edges(blurred)
    timings_ms["edges"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    roi_edges = apply_roi_mask(edges)
    timings_ms["roi"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    line_segments = detect_line_segments(roi_edges)
    lanes = average_lane_lines(frame, line_segments)
    timings_ms["fit_lanes"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    line_overlay = draw_lane_lines(frame, lanes)
    output = cv2.addWeighted(frame, 0.85, line_overlay, 1.0, 0.0)
    timings_ms["overlay"] = _elapsed_ms(stage_start)

    timings_ms["total"] = _elapsed_ms(total_start)

    return LaneDetectionResult(
        grayscale=grayscale,
        blurred=blurred,
        edges=edges,
        roi_edges=roi_edges,
        line_segments=line_segments,
        lanes=lanes,
        line_overlay=line_overlay,
        output=output,
        timings_ms=timings_ms,
    )

