# Benchmarking Notes

The current Mac benchmark is CPU-only and is intended to establish a clean baseline for later GPU comparison.

## What gets measured

Per frame:

- grayscale time
- blur time
- edge detection time
- ROI masking time
- line detection and lane fitting time
- overlay drawing time
- total pipeline time

Aggregate summary:

- frame count
- average milliseconds per stage
- average total frame time
- FPS

## Outputs

Running `scripts/benchmark_cpu.py` writes:

- `outputs/benchmarks/cpu_lane_detection_frames.csv`
- `outputs/benchmarks/cpu_lane_detection_summary.csv`

## Why this matters

These CPU numbers become the baseline for the future CUDA benchmark on the NVIDIA PC. The benchmark layout is already structured so a GPU version can later emit the same CSV schema.

