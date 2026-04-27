from __future__ import annotations

from time import perf_counter_ns

import cv2
import numpy as np

from .gpu_numba import CudaUnavailableError, get_cuda_status, preprocess_frame_cuda
from .lane_detection import (
    LaneDetectionResult,
    average_lane_lines,
    detect_line_segments,
    draw_lane_lines,
)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def run_cuda_lane_detection(frame: np.ndarray, sobel_threshold: int = 120) -> LaneDetectionResult:
    total_start = perf_counter_ns()
    gpu_result = preprocess_frame_cuda(frame, sobel_threshold=sobel_threshold)
    timings_ms = dict(gpu_result.timings_ms)

    stage_start = perf_counter_ns()
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


__all__ = ["CudaUnavailableError", "explain_cuda_unavailable", "get_cuda_status", "run_cuda_lane_detection"]

