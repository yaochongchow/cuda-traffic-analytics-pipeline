from __future__ import annotations

import cv2
import numpy as np


class PerspectiveTransform:
    """Bird's-eye perspective transform adapted from traffic-monitor."""

    def __init__(self, src_ratios: np.ndarray | None = None, dst_ratios: np.ndarray | None = None):
        self.src_ratios = np.float32(
            src_ratios
            if src_ratios is not None
            else [
                [0.18, 0.62],
                [0.82, 0.62],
                [0.98, 0.98],
                [0.02, 0.98],
            ]
        )
        self.dst_ratios = np.float32(
            dst_ratios
            if dst_ratios is not None
            else [
                [0.28, 0.02],
                [0.72, 0.02],
                [0.72, 0.98],
                [0.28, 0.98],
            ]
        )

    def _scaled_points(self, frame_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame_shape[:2]

        src = self.src_ratios.copy()
        src[:, 0] *= width
        src[:, 1] *= height

        dst = self.dst_ratios.copy()
        dst[:, 0] *= width
        dst[:, 1] *= height
        return src, dst

    def get_matrices(self, frame_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        src, dst = self._scaled_points(frame_shape)
        matrix = cv2.getPerspectiveTransform(src, dst)
        inverse = cv2.getPerspectiveTransform(dst, src)
        return matrix, inverse

    def warp(self, frame: np.ndarray, flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        height, width = frame.shape[:2]
        matrix, _ = self.get_matrices(frame.shape)
        return cv2.warpPerspective(frame, matrix, (width, height), flags=flags)

    def unwarp(self, frame: np.ndarray, output_shape: tuple[int, ...], flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        height, width = output_shape[:2]
        _, inverse = self.get_matrices(output_shape)
        return cv2.warpPerspective(frame, inverse, (width, height), flags=flags)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.warp(frame)

