from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2

from .io_utils import OUTPUTS_DIR, ensure_dir, iter_video_frames, read_image
from .lane_detection import run_lane_detection


FRAME_FIELDS = [
    "frame_index",
    "grayscale_ms",
    "blur_ms",
    "edges_ms",
    "roi_ms",
    "fit_lanes_ms",
    "overlay_ms",
    "total_ms",
    "lanes_detected",
]


@dataclass
class BenchmarkArtifacts:
    frame_csv_path: Path
    summary_csv_path: Path
    frame_count: int
    average_total_ms: float
    fps: float


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(frame_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not frame_rows:
        return []

    numeric_fields = [field for field in FRAME_FIELDS if field not in {"frame_index", "lanes_detected"}]
    summary = {
        "metric": "average",
        **{field: round(mean(float(row[field]) for row in frame_rows), 4) for field in numeric_fields},
        "lanes_detected": round(mean(float(row["lanes_detected"]) for row in frame_rows), 4),
    }
    fps = 1000.0 / summary["total_ms"] if summary["total_ms"] else 0.0
    return [
        {
            "metric": "frame_count",
            "grayscale_ms": len(frame_rows),
            "blur_ms": "",
            "edges_ms": "",
            "roi_ms": "",
            "fit_lanes_ms": "",
            "overlay_ms": "",
            "total_ms": "",
            "lanes_detected": "",
        },
        summary,
        {
            "metric": "fps",
            "grayscale_ms": round(fps, 4),
            "blur_ms": "",
            "edges_ms": "",
            "roi_ms": "",
            "fit_lanes_ms": "",
            "overlay_ms": "",
            "total_ms": "",
            "lanes_detected": "",
        },
    ]


def benchmark_image(image_path: Path) -> BenchmarkArtifacts:
    image = read_image(image_path)
    result = run_lane_detection(image)
    row = {
        "frame_index": 0,
        "grayscale_ms": round(result.timings_ms["grayscale"], 4),
        "blur_ms": round(result.timings_ms["blur"], 4),
        "edges_ms": round(result.timings_ms["edges"], 4),
        "roi_ms": round(result.timings_ms["roi"], 4),
        "fit_lanes_ms": round(result.timings_ms["fit_lanes"], 4),
        "overlay_ms": round(result.timings_ms["overlay"], 4),
        "total_ms": round(result.timings_ms["total"], 4),
        "lanes_detected": len(result.lanes),
    }
    return _write_benchmark_outputs([row], output_prefix="cpu_lane_detection_image")


def benchmark_video(video_path: Path, limit_frames: int | None = None) -> BenchmarkArtifacts:
    frame_rows: list[dict[str, object]] = []
    for frame_index, frame in enumerate(iter_video_frames(video_path)):
        if limit_frames is not None and frame_index >= limit_frames:
            break
        result = run_lane_detection(frame)
        frame_rows.append(
            {
                "frame_index": frame_index,
                "grayscale_ms": round(result.timings_ms["grayscale"], 4),
                "blur_ms": round(result.timings_ms["blur"], 4),
                "edges_ms": round(result.timings_ms["edges"], 4),
                "roi_ms": round(result.timings_ms["roi"], 4),
                "fit_lanes_ms": round(result.timings_ms["fit_lanes"], 4),
                "overlay_ms": round(result.timings_ms["overlay"], 4),
                "total_ms": round(result.timings_ms["total"], 4),
                "lanes_detected": len(result.lanes),
            }
        )

    return _write_benchmark_outputs(frame_rows, output_prefix="cpu_lane_detection")


def _write_benchmark_outputs(frame_rows: list[dict[str, object]], output_prefix: str) -> BenchmarkArtifacts:
    benchmark_dir = ensure_dir(OUTPUTS_DIR / "benchmarks")
    frame_csv_path = benchmark_dir / f"{output_prefix}_frames.csv"
    summary_csv_path = benchmark_dir / f"{output_prefix}_summary.csv"

    _write_csv(frame_csv_path, frame_rows, FRAME_FIELDS)

    summary_fields = ["metric", "grayscale_ms", "blur_ms", "edges_ms", "roi_ms", "fit_lanes_ms", "overlay_ms", "total_ms", "lanes_detected"]
    _write_csv(summary_csv_path, _summary_rows(frame_rows), summary_fields)

    average_total_ms = mean(float(row["total_ms"]) for row in frame_rows) if frame_rows else 0.0
    fps = 1000.0 / average_total_ms if average_total_ms else 0.0

    return BenchmarkArtifacts(
        frame_csv_path=frame_csv_path,
        summary_csv_path=summary_csv_path,
        frame_count=len(frame_rows),
        average_total_ms=average_total_ms,
        fps=fps,
    )

