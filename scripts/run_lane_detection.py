from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.io_utils import OUTPUTS_DIR, build_video_writer, ensure_dir, iter_video_frames, read_image, save_image
from cuda_image_processing.lane_detection import run_lane_detection


def _run_on_image(image_path: Path) -> None:
    image = read_image(image_path)
    result = run_lane_detection(image)
    output_dir = ensure_dir(OUTPUTS_DIR / "runs")
    output_path = output_dir / f"{image_path.stem}_lanes.png"
    save_image(output_path, result.output)
    print(f"Saved processed image to {output_path}")
    print(f"Detected lanes: {len(result.lanes)}")
    print(f"Total frame time: {result.timings_ms['total']:.3f} ms")


def _run_on_video(video_path: Path, write_video: bool) -> None:
    output_dir = ensure_dir(OUTPUTS_DIR / "runs")
    writer = None
    output_path = output_dir / f"{video_path.stem}_lanes.mp4"

    for frame_index, frame in enumerate(iter_video_frames(video_path)):
        result = run_lane_detection(frame)
        if write_video and writer is None:
            writer = build_video_writer(output_path, (frame.shape[1], frame.shape[0]), 24.0)
        if writer is not None:
            writer.write(result.output)
        if frame_index == 0:
            print(f"First frame detected lanes: {len(result.lanes)}")
            print(f"First frame total time: {result.timings_ms['total']:.3f} ms")

    if writer is not None:
        writer.release()
        print(f"Saved processed video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPU lane-detection baseline.")
    parser.add_argument("--image", type=Path, help="Image path to process.")
    parser.add_argument("--video", type=Path, help="Video path to process.")
    parser.add_argument("--write-video", action="store_true", help="Write processed video output when using --video.")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise SystemExit("Choose exactly one of --image or --video.")

    if args.image:
        _run_on_image(args.image)
    else:
        _run_on_video(args.video, args.write_video)


if __name__ == "__main__":
    main()

