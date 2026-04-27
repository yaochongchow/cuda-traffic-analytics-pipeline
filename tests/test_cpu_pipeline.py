from __future__ import annotations

import unittest

from cuda_image_processing.lane_detection import run_lane_detection
from cuda_image_processing.gpu_numba import get_cuda_status
from cuda_image_processing.advanced_lane_detection import AdvancedLaneDetector
from cuda_image_processing.perspective import PerspectiveTransform
from cuda_image_processing.realtime_pipeline import TrafficAnalyticsPipeline
from cuda_image_processing.sample_data import generate_lane_frame


class CpuPipelineTests(unittest.TestCase):
    def test_lane_detection_returns_overlay_with_same_shape(self) -> None:
        frame = generate_lane_frame(width=960, height=540, total_frames=30)
        result = run_lane_detection(frame)

        self.assertEqual(result.output.shape, frame.shape)
        self.assertGreaterEqual(len(result.lanes), 2)
        self.assertGreater(result.timings_ms["total"], 0.0)

    def test_cuda_status_check_is_safe_on_cpu_machines(self) -> None:
        status = get_cuda_status()

        self.assertIsInstance(status.available, bool)
        self.assertIsInstance(status.message, str)

    def test_perspective_transform_preserves_shape(self) -> None:
        frame = generate_lane_frame(width=640, height=360, total_frames=30)
        warped = PerspectiveTransform().warp(frame)

        self.assertEqual(warped.shape, frame.shape)

    def test_advanced_lane_detector_can_process_synthetic_frame(self) -> None:
        frame = generate_lane_frame(width=960, height=540, total_frames=30)
        detector = AdvancedLaneDetector(calibration_frames=1)
        left_line, right_line = detector.process(frame)

        self.assertIsNotNone(left_line)
        self.assertIsNotNone(right_line)

    def test_traffic_analytics_pipeline_runs_without_yolo(self) -> None:
        frame = generate_lane_frame(width=640, height=360, total_frames=30)
        pipeline = TrafficAnalyticsPipeline(lane_mode="baseline", enable_objects=False)
        result = pipeline.process(frame)

        self.assertEqual(result.frame.shape, frame.shape)
        self.assertGreaterEqual(result.lanes_detected, 2)


if __name__ == "__main__":
    unittest.main()
