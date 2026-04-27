from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.gpu_numba import CudaUnavailableError
from cuda_image_processing.gpu_pipeline import CudaLaneDetector, explain_cuda_unavailable, run_cuda_lane_detection
from cuda_image_processing.io_utils import OUTPUTS_DIR, build_video_writer, ensure_dir, iter_video_frames, read_image, save_image
from cuda_image_processing.lane_detection import run_lane_detection


def _detector_for_mode(mode: str, copy_intermediates: bool = True):
    if mode == "cpu":
        return run_lane_detection

    def detect_cuda(frame):
        return run_cuda_lane_detection(frame, copy_intermediates=copy_intermediates)

    return detect_cuda


def _run_on_image(image_path: Path, mode: str) -> None:
    image = read_image(image_path)
    detector = _detector_for_mode(mode)
    result = detector(image)
    output_dir = ensure_dir(OUTPUTS_DIR / "runs")
    output_path = output_dir / f"{image_path.stem}_{mode}_lanes.png"
    save_image(output_path, result.output)
    print(f"Saved processed image to {output_path}")
    print(f"Mode: {mode}")
    print(f"Detected lanes: {len(result.lanes)}")
    print(f"Total frame time: {result.timings_ms['total']:.3f} ms")


def _run_on_video(video_path: Path, write_video: bool, mode: str) -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "runs")
    writer = None
    output_path = output_dir / f"{video_path.stem}_{mode}_lanes.mp4"
    detector = _detector_for_mode(mode, copy_intermediates=False)
    cuda_detector = CudaLaneDetector() if mode == "cuda" else None

    for frame_index, frame in enumerate(iter_video_frames(video_path)):
        result = cuda_detector.process(frame) if cuda_detector is not None else detector(frame)
        if write_video and writer is None:
            writer = build_video_writer(output_path, (frame.shape[1], frame.shape[0]), 24.0)
        if writer is not None:
            writer.write(result.output)
        if frame_index == 0:
            print(f"Mode: {mode}")
            print(f"First frame detected lanes: {len(result.lanes)}")
            print(f"First frame total time: {result.timings_ms['total']:.3f} ms")

    if writer is not None:
        writer.release()
        print(f"Saved processed video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lane-detection pipeline.")
    parser.add_argument("--image", type=Path, help="Image path to process.")
    parser.add_argument("--video", type=Path, help="Video path to process.")
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu", help="Execution mode.")
    parser.add_argument("--write-video", action="store_true", help="Write processed video output when using --video.")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise SystemExit("Choose exactly one of --image or --video.")

    try:
        if args.image:
            _run_on_image(args.image, args.mode)
        else:
            _run_on_video(args.video, args.write_video, args.mode)
    except CudaUnavailableError as exc:
        raise SystemExit(f"{exc}\n\n{explain_cuda_unavailable()}") from exc


if __name__ == "__main__":
    main()
