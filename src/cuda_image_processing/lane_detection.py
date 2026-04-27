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


def detect_color_lane_lines(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(hsv, np.array([12, 45, 70]), np.array([45, 255, 255]))
    white = cv2.inRange(hls, np.array([0, 165, 0]), np.array([255, 255, 150]))
    mask = cv2.bitwise_or(yellow, white)

    roi = np.zeros((height, width), np.uint8)
    polygon = np.array(
        [
            [
                (0, height - 1),
                (int(width * 0.30), int(height * 0.20)),
                (int(width * 0.78), int(height * 0.20)),
                (width - 1, height - 1),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, polygon, 255)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=max(int(height * 0.08), 35),
        maxLineGap=max(int(height * 0.12), 45),
    )
    if lines is None:
        return []

    y_bottom = height - 1
    y_top = int(height * 0.36)
    candidates: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}

    for segment in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, segment)
        vertical_span = abs(y2 - y1)
        if vertical_span < height * 0.12:
            continue

        dx_dy = (x2 - x1) / (y2 - y1)
        if abs(dx_dy) < 0.25 or abs(dx_dy) > 3.5:
            continue

        intercept = x1 - dx_dy * y1
        x_bottom = dx_dy * y_bottom + intercept
        x_top = dx_dy * y_top + intercept
        if not (-width * 0.1 <= x_bottom <= width * 1.1 and -width * 0.1 <= x_top <= width * 1.1):
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        reaches_bottom = max(y1, y2) >= height * 0.66
        if not reaches_bottom:
            continue
        score = length * 1.4

        if dx_dy < 0 and width * 0.08 <= x_bottom <= width * 0.72 and width * 0.35 <= x_top <= width * 1.05:
            candidates["left"].append((score, dx_dy, intercept))
        elif dx_dy > 0 and width * 0.35 <= x_bottom <= width * 0.92 and width * 0.35 <= x_top <= width * 0.92:
            candidates["right"].append((score, dx_dy, intercept))

    lanes: list[tuple[int, int, int, int]] = []
    for side in ("left", "right"):
        side_candidates = sorted(candidates[side], key=lambda item: item[0], reverse=True)[:8]
        if not side_candidates:
            continue
        total_score = sum(score for score, _, _ in side_candidates)
        dx_dy = sum(dx_dy * score for score, dx_dy, _ in side_candidates) / total_score
        intercept = sum(intercept * score for score, _, intercept in side_candidates) / total_score
        lanes.append(
            (
                int(np.clip(dx_dy * y_bottom + intercept, 0, width - 1)),
                y_bottom,
                int(np.clip(dx_dy * y_top + intercept, 0, width - 1)),
                y_top,
            )
        )

    return lanes


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

