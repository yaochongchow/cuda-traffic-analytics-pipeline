from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .advanced_lane_detection import AdvancedLaneDetector
from .gpu_numba import CudaUnavailableError
from .gpu_pipeline import CudaLaneDetector
from .lane_detection import detect_color_lane_lines, draw_lane_lines, run_lane_detection
from .object_detection import Detection, ObjectDetectionUnavailableError, TrafficObjectDetector
from .perspective import PerspectiveTransform


LANE_OCCLUDER_CLASSES = {"car", "truck", "bus", "person", "bicycle", "motorcycle"}


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
        self.cuda_lanes = CudaLaneDetector() if execution_mode == "cuda" else None
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

    def _line_hits_detection(self, line: tuple[int, int, int, int], detection: Detection) -> bool:
        x1, y1, x2, y2 = line
        for step in range(21):
            ratio = step / 20.0
            x = int(x1 + (x2 - x1) * ratio)
            y = int(y1 + (y2 - y1) * ratio)
            if detection.x1 <= x <= detection.x2 and detection.y1 <= y <= detection.y2:
                return True
        return False

    def _clip_occluded_lane(
        self,
        lane: tuple[int, int, int, int],
        detections: list[Detection],
    ) -> tuple[int, int, int, int] | None:
        if not detections:
            return lane

        occluders = [d for d in detections if d.class_name in LANE_OCCLUDER_CLASSES]
        if not occluders:
            return lane

        x1, y1, x2, y2 = lane
        last_clear = (x1, y1)
        for step in range(1, 41):
            ratio = step / 40.0
            x = int(x1 + (x2 - x1) * ratio)
            y = int(y1 + (y2 - y1) * ratio)
            if any(detection.x1 <= x <= detection.x2 and detection.y1 <= y <= detection.y2 for detection in occluders):
                if abs(last_clear[1] - y1) < 45:
                    return None
                return x1, y1, last_clear[0], last_clear[1]
            last_clear = (x, y)
        return lane

    def _clip_occluded_lanes(
        self,
        lanes: list[tuple[int, int, int, int]],
        detections: list[Detection],
    ) -> list[tuple[int, int, int, int]]:
        clipped = [self._clip_occluded_lane(lane, detections) for lane in lanes]
        return [lane for lane in clipped if lane is not None]

    def _merge_lane_candidates(
        self,
        primary: list[tuple[int, int, int, int]],
        fallback: list[tuple[int, int, int, int]],
        width: int,
    ) -> list[tuple[int, int, int, int]]:
        if fallback:
            return fallback
        if self.enable_objects:
            return []

        def side(line: tuple[int, int, int, int]) -> str:
            x1, y1, x2, y2 = line
            return "left" if (x2 - x1) / max(abs(y2 - y1), 1) < 0 else "right"

        merged: dict[str, tuple[int, int, int, int]] = {}
        for line in primary:
            merged[side(line)] = line
        for line in fallback:
            line_side = side(line)
            if line_side not in merged or abs(line[1] - line[3]) > abs(merged[line_side][1] - merged[line_side][3]):
                merged[line_side] = line
        return [merged[key] for key in ("left", "right") if key in merged]

    def _draw_lane_result(
        self,
        frame: np.ndarray,
        lanes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        line_overlay = draw_lane_lines(frame, lanes)
        return cv2.addWeighted(frame, 0.85, line_overlay, 1.0, 0.0)

    def _process_lanes(
        self,
        lane_frame: np.ndarray,
        render_frame: np.ndarray,
        detections: list[Detection],
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        if self.execution_mode == "cuda":
            assert self.cuda_lanes is not None
            result = self.cuda_lanes.process(lane_frame)
            lanes = self._clip_occluded_lanes(result.lanes, detections)
            color_lanes = self._clip_occluded_lanes(detect_color_lane_lines(lane_frame), detections)
            lanes = self._merge_lane_candidates(lanes, color_lanes, render_frame.shape[1])
            return self._draw_lane_result(render_frame, lanes), len(lanes), result.timings_ms

        if self.lane_mode == "baseline":
            result = run_lane_detection(lane_frame)
            lanes = self._clip_occluded_lanes(result.lanes, detections)
            color_lanes = self._clip_occluded_lanes(detect_color_lane_lines(lane_frame), detections)
            lanes = self._merge_lane_candidates(lanes, color_lanes, render_frame.shape[1])
            return self._draw_lane_result(render_frame, lanes), len(lanes), result.timings_ms

        assert self.advanced_lanes is not None
        start = perf_counter()
        left_line, right_line = self.advanced_lanes.process(lane_frame)
        lanes = self._clip_occluded_lanes([line for line in (left_line, right_line) if line is not None], detections)
        output = self.advanced_lanes.draw(render_frame, lanes[0] if len(lanes) > 0 else None, lanes[1] if len(lanes) > 1 else None)
        lanes_detected = int(left_line is not None) + int(right_line is not None)
        return output, min(lanes_detected, len(lanes)), {"advanced_lanes": (perf_counter() - start) * 1000.0}

    def _mask_lane_occluders(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        if not detections:
            return frame

        height, width = frame.shape[:2]
        road_sample = frame[int(height * 0.84) : int(height * 0.96), int(width * 0.42) : int(width * 0.68)]
        road_color = tuple(int(value) for value in np.median(road_sample.reshape(-1, 3), axis=0)) if road_sample.size else (45, 45, 45)
        masked = frame.copy()

        for detection in detections:
            if detection.class_name not in LANE_OCCLUDER_CLASSES or detection.y2 < height * 0.45:
                continue
            margin = 10
            x1 = max(0, detection.x1 - margin)
            y1 = max(0, detection.y1 - margin)
            x2 = min(width - 1, detection.x2 + margin)
            y2 = min(height - 1, detection.y2 + margin)
            cv2.rectangle(masked, (x1, y1), (x2, y2), road_color, -1)

        return masked

    def process(self, frame: np.ndarray) -> TrafficAnalyticsResult:
        timings_ms: dict[str, float] = {}
        display = frame.copy()
        lanes_detected = 0
        objects_detected = 0
        detections: list[Detection] = []

        if self.enable_objects:
            if self.object_detector is None:
                raise ObjectDetectionUnavailableError("Object detection was enabled but the detector was not initialized.")
            start = perf_counter()
            detections = self.object_detector.process(frame)
            objects_detected = len(detections)
            timings_ms["objects"] = (perf_counter() - start) * 1000.0

        lane_frame = self._mask_lane_occluders(frame, detections) if self.enable_objects else frame
        if self.enable_lanes:
            display, lanes_detected, lane_timings = self._process_lanes(lane_frame, frame, detections)
            timings_ms.update(lane_timings)

        if self.enable_objects and self.object_detector is not None:
            display = self.object_detector.draw(display, detections)

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

