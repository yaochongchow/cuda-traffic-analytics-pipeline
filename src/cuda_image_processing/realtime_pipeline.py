from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .advanced_lane_detection import AdvancedLaneDetector
from .gpu_numba import CudaUnavailableError
from .gpu_pipeline import run_cuda_lane_detection
from .lane_detection import run_lane_detection
from .object_detection import Detection, ObjectDetectionUnavailableError, TrafficObjectDetector
from .perspective import PerspectiveTransform


@dataclass
class TrafficAnalyticsResult:
    frame: np.ndarray
    lanes_detected: int
    objects_detected: int
    fps: float
    timings_ms: dict[str, float]


class TrafficAnalyticsPipeline:
    """Combined traffic analytics pipeline.

    This is the bridge between the CUDA project and traffic-monitor: lane
    detection, optional YOLO object detection, optional perspective display,
    and per-frame timing live in one reusable class.
    """

    def __init__(
        self,
        lane_mode: str = "advanced",
        execution_mode: str = "cpu",
        enable_lanes: bool = True,
        enable_objects: bool = False,
        enable_perspective: bool = False,
        show_fps: bool = True,
        confidence: float = 0.4,
        object_skip_frames: int = 3,
    ):
        if lane_mode not in {"baseline", "advanced"}:
            raise ValueError("lane_mode must be either 'baseline' or 'advanced'.")
        if execution_mode not in {"cpu", "cuda"}:
            raise ValueError("execution_mode must be either 'cpu' or 'cuda'.")

        self.lane_mode = lane_mode
        self.execution_mode = execution_mode
        self.enable_lanes = enable_lanes
        self.enable_objects = enable_objects
        self.enable_perspective = enable_perspective
        self.show_fps = show_fps
        self.advanced_lanes = AdvancedLaneDetector() if lane_mode == "advanced" else None
        self.perspective = PerspectiveTransform()
        self.prev_time = perf_counter()
        self.object_detector: TrafficObjectDetector | None = None

        if enable_objects:
            self.object_detector = TrafficObjectDetector(confidence=confidence, skip_frames=object_skip_frames)

    def _draw_status(self, frame: np.ndarray, fps: float) -> np.ndarray:
        height = frame.shape[0]
        active = []
        if self.enable_lanes:
            active.append(f"LANES:{self.lane_mode.upper()}")
        if self.execution_mode == "cuda":
            active.append("CUDA")
        if self.enable_objects:
            active.append("OBJECTS")
        if self.enable_perspective:
            active.append("PERSP")
        status = " | ".join(active) if active else "ALL OFF"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def _process_lanes(self, frame: np.ndarray) -> tuple[np.ndarray, int, dict[str, float]]:
        if self.execution_mode == "cuda":
            result = run_cuda_lane_detection(frame)
            return result.output, len(result.lanes), result.timings_ms

        if self.lane_mode == "baseline":
            result = run_lane_detection(frame)
            return result.output, len(result.lanes), result.timings_ms

        assert self.advanced_lanes is not None
        start = perf_counter()
        left_line, right_line = self.advanced_lanes.process(frame)
        output = self.advanced_lanes.draw(frame, left_line, right_line)
        lanes_detected = int(left_line is not None) + int(right_line is not None)
        return output, lanes_detected, {"advanced_lanes": (perf_counter() - start) * 1000.0}

    def process(self, frame: np.ndarray) -> TrafficAnalyticsResult:
        timings_ms: dict[str, float] = {}
        display = frame.copy()
        lanes_detected = 0
        objects_detected = 0

        if self.enable_lanes:
            display, lanes_detected, lane_timings = self._process_lanes(frame)
            timings_ms.update(lane_timings)

        if self.enable_objects:
            if self.object_detector is None:
                raise ObjectDetectionUnavailableError("Object detection was enabled but the detector was not initialized.")
            start = perf_counter()
            detections = self.object_detector.process(frame)
            display = self.object_detector.draw(display, detections)
            objects_detected = len(detections)
            timings_ms["objects"] = (perf_counter() - start) * 1000.0

        if self.enable_perspective:
            start = perf_counter()
            display = self.perspective.process(display)
            timings_ms["perspective"] = (perf_counter() - start) * 1000.0

        current = perf_counter()
        fps = 1.0 / max(current - self.prev_time, 1e-5)
        self.prev_time = current
        display = self._draw_status(display, fps)
        timings_ms["total"] = sum(timings_ms.values())

        return TrafficAnalyticsResult(
            frame=display,
            lanes_detected=lanes_detected,
            objects_detected=objects_detected,
            fps=fps,
            timings_ms=timings_ms,
        )


__all__ = [
    "CudaUnavailableError",
    "Detection",
    "ObjectDetectionUnavailableError",
    "TrafficAnalyticsPipeline",
    "TrafficAnalyticsResult",
]

