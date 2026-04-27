from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.gpu_numba import CudaUnavailableError
from cuda_image_processing.gpu_pipeline import CudaLaneDetector, explain_cuda_unavailable, run_cuda_lane_detection
from cuda_image_processing.io_utils import OUTPUTS_DIR, ensure_dir, iter_video_frames, read_image


FIELDS = [
    "frame_index",
    "resolution",
    "mode",
    "copy_h2d_ms",
    "grayscale_ms",
    "blur_ms",
    "edges_ms",
    "roi_ms",
    "lane_stats_ms",
    "gpu_kernel_total_ms",
    "copy_d2h_ms",
    "gpu_preprocess_total_ms",
    "fit_lanes_ms",
    "overlay_ms",
    "total_ms",
    "fps",
    "lanes_detected",
]


def _row(frame_index: int, frame, result) -> dict[str, object]:
    total_ms = result.timings_ms["total"]
    return {
        "frame_index": frame_index,
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        "mode": "cuda",
        "copy_h2d_ms": round(result.timings_ms.get("copy_h2d", 0.0), 4),
        "grayscale_ms": round(result.timings_ms.get("grayscale", 0.0), 4),
        "blur_ms": round(result.timings_ms.get("blur", 0.0), 4),
        "edges_ms": round(result.timings_ms.get("edges", 0.0), 4),
        "roi_ms": round(result.timings_ms.get("roi", 0.0), 4),
        "lane_stats_ms": round(result.timings_ms.get("lane_stats", 0.0), 4),
        "gpu_kernel_total_ms": round(result.timings_ms.get("gpu_kernel_total", 0.0), 4),
        "copy_d2h_ms": round(result.timings_ms.get("copy_d2h", 0.0), 4),
        "gpu_preprocess_total_ms": round(result.timings_ms.get("gpu_preprocess_total", 0.0), 4),
        "fit_lanes_ms": round(result.timings_ms.get("fit_lanes", 0.0), 4),
        "overlay_ms": round(result.timings_ms.get("overlay", 0.0), 4),
        "total_ms": round(total_ms, 4),
        "fps": round(1000.0 / total_ms, 4) if total_ms else 0.0,
        "lanes_detected": len(result.lanes),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    numeric_fields = [field for field in FIELDS if field not in {"frame_index", "resolution", "mode"}]
    summary = [{"metric": field, "average": round(mean(float(row[field]) for row in rows), 4)} for field in numeric_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "average"])
        writer.writeheader()
        writer.writerows(summary)


def _benchmark_image(image_path: Path) -> list[dict[str, object]]:
    frame = read_image(image_path)
    result = run_cuda_lane_detection(frame, copy_intermediates=False)
    return [_row(0, frame, result)]


def _benchmark_video(video_path: Path, limit_frames: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    detector = CudaLaneDetector()
    for frame_index, frame in enumerate(iter_video_frames(video_path)):
        if limit_frames is not None and frame_index >= limit_frames:
            break
        result = detector.process(frame)
        rows.append(_row(frame_index, frame, result))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Numba CUDA lane-detection path.")
    parser.add_argument("--image", type=Path, help="Single image to benchmark.")
    parser.add_argument("--video", type=Path, help="Video to benchmark.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional frame cap for quicker runs.")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise SystemExit("Choose exactly one of --image or --video.")

    try:
        rows = _benchmark_image(args.image) if args.image else _benchmark_video(args.video, args.limit_frames)
    except CudaUnavailableError as exc:
        raise SystemExit(f"{exc}\n\n{explain_cuda_unavailable()}") from exc

    frame_csv_path = OUTPUTS_DIR / "benchmarks" / "gpu_lane_detection_frames.csv"
    summary_csv_path = OUTPUTS_DIR / "benchmarks" / "gpu_lane_detection_summary.csv"
    _write_csv(frame_csv_path, rows)
    _write_summary(summary_csv_path, rows)

    average_total_ms = mean(float(row["total_ms"]) for row in rows) if rows else 0.0
    fps = 1000.0 / average_total_ms if average_total_ms else 0.0
    print(f"Frame rows: {len(rows)}")
    print(f"Average total time: {average_total_ms:.3f} ms")
    print(f"Estimated FPS: {fps:.2f}")
    print(f"Frame CSV: {frame_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

