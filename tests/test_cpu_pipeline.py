from __future__ import annotations

import unittest

from cuda_image_processing.lane_detection import run_lane_detection
from cuda_image_processing.gpu_numba import get_cuda_status
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


if __name__ == "__main__":
    unittest.main()
