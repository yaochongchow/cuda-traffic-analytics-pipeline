from __future__ import annotations

from collections import deque
from time import perf_counter_ns

import cv2
import numpy as np

from .gpu_numba import CudaFramePreprocessor, CudaUnavailableError, get_cuda_status, preprocess_frame_cuda
from .lane_detection import (
    LaneDetectionResult,
    average_lane_lines,
    detect_line_segments,
    draw_lane_lines,
)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _lanes_from_stats(
    lane_stats: np.ndarray | None,
    width: int,
    height: int,
    min_pixels: int = 400,
) -> list[tuple[int, int, int, int]]:
    if lane_stats is None:
        return []

    lanes: list[tuple[int, int, int, int]] = []
    y_bottom = height - 1
    y_top = int(height * 0.62)

    for side, expected_sign in ((0, -1), (1, 1)):
        count, sum_y, sum_x, sum_yy, sum_xy = lane_stats[side]
        if count < min_pixels:
            continue

        denom = count * sum_yy - sum_y * sum_y
        if abs(denom) < 1e-6:
            continue

        dx_dy = (count * sum_xy - sum_y * sum_x) / denom
        if dx_dy * expected_sign <= 0 or abs(dx_dy) < 0.2 or abs(dx_dy) > 2.5:
            continue

        intercept = (sum_x - dx_dy * sum_y) / count
        x_bottom = int(np.clip(dx_dy * y_bottom + intercept, 0, width - 1))
        x_top = int(np.clip(dx_dy * y_top + intercept, 0, width - 1))
        lanes.append((x_bottom, y_bottom, x_top, y_top))

    return lanes


class CudaLaneDetector:
    """Stateful CUDA lane detector that reuses device buffers across frames."""

    def __init__(
        self,
        sobel_threshold: int = 625,
        copy_intermediates: bool = False,
        copy_roi_edges: bool = False,
        use_gpu_lane_fit: bool = True,
    ):
        self.sobel_threshold = sobel_threshold
        self.copy_intermediates = copy_intermediates
        self.copy_roi_edges = copy_roi_edges
        self.use_gpu_lane_fit = use_gpu_lane_fit
        self.preprocessor: CudaFramePreprocessor | None = None
        self.last_left: tuple[int, int, int, int] | None = None
        self.last_right: tuple[int, int, int, int] | None = None
        self.bottom_widths: deque[float] = deque(maxlen=20)
        self.top_widths: deque[float] = deque(maxlen=20)
        self.smooth_alpha = 0.35

    def _split_lanes(
        self,
        lanes: list[tuple[int, int, int, int]],
        width: int,
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        left_candidates = [lane for lane in lanes if lane[0] < width * 0.58]
        right_candidates = [lane for lane in lanes if lane[0] > width * 0.42]
        left = max(left_candidates, key=lambda lane: lane[0], default=None)
        right = min(right_candidates, key=lambda lane: lane[0], default=None)
        return left, right

    def _average_widths(self, width: int) -> tuple[float, float]:
        bottom_width = float(np.mean(self.bottom_widths)) if self.bottom_widths else width * 0.36
        top_width = float(np.mean(self.top_widths)) if self.top_widths else width * 0.16
        return bottom_width, top_width

    def _infer_missing(
        self,
        left: tuple[int, int, int, int] | None,
        right: tuple[int, int, int, int] | None,
        width: int,
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        bottom_width, top_width = self._average_widths(width)
        if left is None and right is not None:
            left = (
                int(np.clip(right[0] - bottom_width, 0, width - 1)),
                right[1],
                int(np.clip(right[2] - top_width, 0, width - 1)),
                right[3],
            )
        elif right is None and left is not None:
            right = (
                int(np.clip(left[0] + bottom_width, 0, width - 1)),
                left[1],
                int(np.clip(left[2] + top_width, 0, width - 1)),
                left[3],
            )
        return left, right

    def _smooth(
        self,
        current: tuple[int, int, int, int] | None,
        previous: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if current is None:
            return previous
        if previous is None:
            return current
        return tuple(
            int(round((1.0 - self.smooth_alpha) * old + self.smooth_alpha * new))
            for old, new in zip(previous, current)
        )

    def _stabilize_lanes(
        self,
        lanes: list[tuple[int, int, int, int]],
        width: int,
    ) -> list[tuple[int, int, int, int]]:
        left, right = self._split_lanes(lanes, width)
        left, right = self._infer_missing(left, right, width)

        if left is None and self.last_left is not None:
            left = self.last_left
        if right is None and self.last_right is not None:
            right = self.last_right

        if left is not None and right is not None:
            bottom_width = right[0] - left[0]
            top_width = right[2] - left[2]
            if bottom_width <= width * 0.12 or top_width <= width * 0.03:
                left, right = self.last_left, self.last_right
            else:
                self.bottom_widths.append(float(bottom_width))
                self.top_widths.append(float(top_width))

        left = self._smooth(left, self.last_left)
        right = self._smooth(right, self.last_right)
        self.last_left = left
        self.last_right = right

        return [lane for lane in (left, right) if lane is not None]

    def process(self, frame: np.ndarray) -> LaneDetectionResult:
        total_start = perf_counter_ns()
        if self.preprocessor is None or self.preprocessor.frame_shape != frame.shape:
            self.preprocessor = CudaFramePreprocessor(frame.shape, sobel_threshold=self.sobel_threshold)

        gpu_result = self.preprocessor.process(
            frame,
            copy_intermediates=self.copy_intermediates,
            copy_roi_edges=self.copy_roi_edges or not self.use_gpu_lane_fit,
            compute_lane_stats=self.use_gpu_lane_fit,
        )
        timings_ms = dict(gpu_result.timings_ms)

        stage_start = perf_counter_ns()
        if self.use_gpu_lane_fit:
            line_segments = None
            lanes = self._stabilize_lanes(_lanes_from_stats(gpu_result.lane_stats, frame.shape[1], frame.shape[0]), frame.shape[1])
        else:
            line_segments = detect_line_segments(gpu_result.roi_edges)
            lanes = average_lane_lines(frame, line_segments)
        timings_ms["fit_lanes"] = _elapsed_ms(stage_start)

        stage_start = perf_counter_ns()
        line_overlay = draw_lane_lines(frame, lanes)
        output = cv2.addWeighted(frame, 0.85, line_overlay, 1.0, 0.0)
        timings_ms["overlay"] = _elapsed_ms(stage_start)
        timings_ms["total"] = _elapsed_ms(total_start)

        return LaneDetectionResult(
            grayscale=gpu_result.grayscale,
            blurred=gpu_result.blurred,
            edges=gpu_result.edges,
            roi_edges=gpu_result.roi_edges,
            line_segments=line_segments,
            lanes=lanes,
            line_overlay=line_overlay,
            output=output,
            timings_ms=timings_ms,
        )


def run_cuda_lane_detection(
    frame: np.ndarray,
    sobel_threshold: int = 625,
    copy_intermediates: bool = True,
    copy_roi_edges: bool = True,
    use_gpu_lane_fit: bool = True,
) -> LaneDetectionResult:
    total_start = perf_counter_ns()
    gpu_result = preprocess_frame_cuda(
        frame,
        sobel_threshold=sobel_threshold,
        copy_intermediates=copy_intermediates,
        copy_roi_edges=copy_roi_edges,
        compute_lane_stats=use_gpu_lane_fit,
    )
    timings_ms = dict(gpu_result.timings_ms)

    stage_start = perf_counter_ns()
    if use_gpu_lane_fit:
        line_segments = None
        lanes = _lanes_from_stats(gpu_result.lane_stats, frame.shape[1], frame.shape[0])
    else:
        line_segments = detect_line_segments(gpu_result.roi_edges)
        lanes = average_lane_lines(frame, line_segments)
    timings_ms["fit_lanes"] = _elapsed_ms(stage_start)

    stage_start = perf_counter_ns()
    line_overlay = draw_lane_lines(frame, lanes)
    output = cv2.addWeighted(frame, 0.85, line_overlay, 1.0, 0.0)
    timings_ms["overlay"] = _elapsed_ms(stage_start)
    timings_ms["total"] = _elapsed_ms(total_start)

    return LaneDetectionResult(
        grayscale=gpu_result.grayscale,
        blurred=gpu_result.blurred,
        edges=gpu_result.edges,
        roi_edges=gpu_result.roi_edges,
        line_segments=line_segments,
        lanes=lanes,
        line_overlay=line_overlay,
        output=output,
        timings_ms=timings_ms,
    )


def explain_cuda_unavailable() -> str:
    status = get_cuda_status()
    return (
        f"{status.message}\n"
        "CUDA mode requires an NVIDIA GPU, NVIDIA driver, CUDA runtime, and Numba CUDA support. "
        "Run this mode on the NVIDIA PC described in README_pc_cuda_handoff.md."
    )


__all__ = [
    "CudaLaneDetector",
    "CudaUnavailableError",
    "explain_cuda_unavailable",
    "get_cuda_status",
    "run_cuda_lane_detection",
]

