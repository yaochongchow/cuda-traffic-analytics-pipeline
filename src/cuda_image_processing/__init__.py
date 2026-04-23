"""CPU-first image-processing utilities for the CUDA handoff project."""

from .lane_detection import LaneDetectionResult, run_lane_detection

__all__ = ["LaneDetectionResult", "run_lane_detection"]

