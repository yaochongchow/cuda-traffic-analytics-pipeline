from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.gpu_numba import CudaUnavailableError
from cuda_image_processing.gpu_pipeline import CudaLaneDetector, explain_cuda_unavailable
from cuda_image_processing.io_utils import DOCS_ASSETS_DIR, OUTPUTS_DIR, build_video_writer, ensure_dir
from cuda_image_processing.lane_detection import run_lane_detection


def _label(frame, text: str):
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(labeled, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def _average(rows: list[dict[str, float]], field: str) -> float:
    return mean(row.get(field, 0.0) for row in rows) if rows else 0.0


def _write_summary(path: Path, cpu_rows: list[dict[str, float]], gpu_rows: list[dict[str, float]]) -> None:
    ensure_dir(path.parent)
    cpu_total = _average(cpu_rows, "total")
    gpu_total = _average(gpu_rows, "total")
    rows = [
        {"metric": "frames", "cpu": len(cpu_rows), "cuda": len(gpu_rows)},
        {"metric": "total_ms", "cpu": round(cpu_total, 4), "cuda": round(gpu_total, 4)},
        {"metric": "fps", "cpu": round(1000.0 / cpu_total, 4), "cuda": round(1000.0 / gpu_total, 4)},
        {"metric": "speedup", "cpu": "", "cuda": round(cpu_total / gpu_total, 4) if gpu_total else 0.0},
        {"metric": "gpu_kernel_total_ms", "cpu": "", "cuda": round(_average(gpu_rows, "gpu_kernel_total"), 4)},
        {"metric": "gpu_lane_stats_ms", "cpu": "", "cuda": round(_average(gpu_rows, "lane_stats"), 4)},
        {"metric": "gpu_copy_h2d_ms", "cpu": "", "cuda": round(_average(gpu_rows, "copy_h2d"), 4)},
        {"metric": "gpu_copy_d2h_ms", "cpu": "", "cuda": round(_average(gpu_rows, "copy_d2h"), 4)},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "cpu", "cuda"])
        writer.writeheader()
        writer.writerows(rows)


def _write_charts(output_dir: Path, cpu_rows: list[dict[str, float]], gpu_rows: list[dict[str, float]]) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    cpu_total = _average(cpu_rows, "total")
    gpu_total = _average(gpu_rows, "total")
    cpu_fps = 1000.0 / cpu_total if cpu_total else 0.0
    gpu_fps = 1000.0 / gpu_total if gpu_total else 0.0

    fps_chart = output_dir / "phase7_fps_comparison.png"
    plt.figure(figsize=(7, 4), dpi=160)
    bars = plt.bar(["CPU OpenCV", "CUDA Optimized"], [cpu_fps, gpu_fps], color=["#4c78a8", "#54a24b"])
    plt.ylabel("Frames per second")
    plt.title("1080p Lane Detection Throughput")
    plt.bar_label(bars, fmt="%.1f FPS", padding=4)
    plt.ylim(0, max(cpu_fps, gpu_fps) * 1.25)
    plt.tight_layout()
    plt.savefig(fps_chart)
    plt.close()

    latency_chart = output_dir / "phase7_cuda_latency_breakdown.png"
    labels = ["H2D copy", "GPU kernels", "D2H copy", "Lane fit", "Overlay"]
    values = [
        _average(gpu_rows, "copy_h2d"),
        _average(gpu_rows, "gpu_kernel_total"),
        _average(gpu_rows, "copy_d2h"),
        _average(gpu_rows, "fit_lanes"),
        _average(gpu_rows, "overlay"),
    ]
    plt.figure(figsize=(8, 4), dpi=160)
    bars = plt.bar(labels, values, color=["#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"])
    plt.ylabel("Milliseconds per frame")
    plt.title("CUDA Path Latency Breakdown")
    plt.bar_label(bars, fmt="%.2f ms", padding=3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(latency_chart)
    plt.close()

    return fps_chart, latency_chart


def build_phase7_assets(video_path: Path, limit_frames: int, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    ensure_dir(output_dir)
    benchmark_dir = ensure_dir(OUTPUTS_DIR / "benchmarks")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    side_by_side_path = output_dir / "phase7_cpu_vs_cuda_demo.mp4"
    writer = build_video_writer(side_by_side_path, (width * 2, height), fps)

    cpu_rows: list[dict[str, float]] = []
    gpu_rows: list[dict[str, float]] = []
    cuda_detector = CudaLaneDetector()

    frame_index = 0
    try:
        ok, warmup_frame = capture.read()
        if ok:
            cuda_detector.process(warmup_frame)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while frame_index < limit_frames:
            ok, frame = capture.read()
            if not ok:
                break

            cpu_result = run_lane_detection(frame)
            gpu_result = cuda_detector.process(frame)

            cpu_rows.append(cpu_result.timings_ms)
            gpu_rows.append(gpu_result.timings_ms)

            cpu_frame = _label(cpu_result.output, f"CPU OpenCV | {cpu_result.timings_ms['total']:.1f} ms")
            gpu_frame = _label(gpu_result.output, f"CUDA Optimized | {gpu_result.timings_ms['total']:.1f} ms")
            writer.write(cv2.hconcat([cpu_frame, gpu_frame]))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    summary_path = benchmark_dir / "phase7_demo_summary.csv"
    _write_summary(summary_path, cpu_rows, gpu_rows)
    fps_chart, latency_chart = _write_charts(output_dir, cpu_rows, gpu_rows)
    return side_by_side_path, fps_chart, latency_chart, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 7 final demo and benchmark assets.")
    parser.add_argument("--video", type=Path, default=DOCS_ASSETS_DIR / "sample.avi", help="Input road video.")
    parser.add_argument("--limit-frames", type=int, default=300, help="Number of frames to include.")
    parser.add_argument("--output-dir", type=Path, default=DOCS_ASSETS_DIR, help="Directory for committed demo assets.")
    args = parser.parse_args()

    try:
        side_by_side_path, fps_chart, latency_chart, summary_path = build_phase7_assets(
            args.video,
            args.limit_frames,
            args.output_dir,
        )
    except CudaUnavailableError as exc:
        raise SystemExit(f"{exc}\n\n{explain_cuda_unavailable()}") from exc

    print(f"Side-by-side demo: {side_by_side_path}")
    print(f"FPS chart: {fps_chart}")
    print(f"Latency chart: {latency_chart}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
