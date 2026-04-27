from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.gpu_numba import CudaUnavailableError
from cuda_image_processing.gpu_pipeline import explain_cuda_unavailable, run_cuda_lane_detection
from cuda_image_processing.io_utils import OUTPUTS_DIR, ensure_dir, read_image, save_image
from cuda_image_processing.lane_detection import run_lane_detection


def _diff_stats(cpu_image: np.ndarray, gpu_image: np.ndarray) -> tuple[float, int, np.ndarray]:
    diff = cv2.absdiff(cpu_image, gpu_image)
    return float(np.mean(diff)), int(np.max(diff)), diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CPU lane preprocessing against the CUDA path.")
    parser.add_argument("--image", type=Path, required=True, help="Image path to validate.")
    args = parser.parse_args()

    frame = read_image(args.image)
    cpu_result = run_lane_detection(frame)

    try:
        gpu_result = run_cuda_lane_detection(frame)
    except CudaUnavailableError as exc:
        raise SystemExit(f"{exc}\n\n{explain_cuda_unavailable()}") from exc

    output_dir = ensure_dir(OUTPUTS_DIR / "validation")
    save_image(output_dir / "cpu_lane_overlay.png", cpu_result.output)
    save_image(output_dir / "gpu_lane_overlay.png", gpu_result.output)

    for name, cpu_image, gpu_image in (
        ("grayscale", cpu_result.grayscale, gpu_result.grayscale),
        ("edges", cpu_result.edges, gpu_result.edges),
        ("roi_edges", cpu_result.roi_edges, gpu_result.roi_edges),
    ):
        mae, max_diff, diff = _diff_stats(cpu_image, gpu_image)
        save_image(output_dir / f"diff_{name}.png", diff)
        print(f"{name}: mae={mae:.3f}, max_diff={max_diff}")

    print(f"CPU lanes detected: {len(cpu_result.lanes)}")
    print(f"GPU lanes detected: {len(gpu_result.lanes)}")
    print(f"Validation outputs: {output_dir}")


if __name__ == "__main__":
    main()

