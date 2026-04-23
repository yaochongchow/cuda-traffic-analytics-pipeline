from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.benchmarking import benchmark_image, benchmark_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CPU lane-detection baseline.")
    parser.add_argument("--image", type=Path, help="Single image to benchmark.")
    parser.add_argument("--video", type=Path, help="Video to benchmark.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional frame cap for quicker runs.")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise SystemExit("Choose exactly one of --image or --video.")

    if args.image:
        artifacts = benchmark_image(args.image)
    else:
        artifacts = benchmark_video(args.video, limit_frames=args.limit_frames)

    print(f"Frame rows: {artifacts.frame_count}")
    print(f"Average total time: {artifacts.average_total_ms:.3f} ms")
    print(f"Estimated FPS: {artifacts.fps:.2f}")
    print(f"Frame CSV: {artifacts.frame_csv_path}")
    print(f"Summary CSV: {artifacts.summary_csv_path}")


if __name__ == "__main__":
    main()

