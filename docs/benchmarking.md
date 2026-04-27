# Benchmarking Notes

The project now includes CPU and CUDA benchmarks. The CPU path is the OpenCV baseline, while the CUDA path runs pixel preprocessing and lane-stat reduction on the GPU before drawing the final overlay on the CPU.

## What gets measured

CPU per frame:

- grayscale time
- blur time
- edge detection time
- ROI masking time
- line detection and lane fitting time
- overlay drawing time
- total pipeline time

CUDA per frame:

- host-to-device copy time
- CUDA grayscale kernel time
- CUDA blur kernel time
- CUDA Sobel edge threshold time
- CUDA ROI mask time
- CUDA lane-stat reduction time
- GPU kernel total time
- device-to-host copy time
- compact lane-line reconstruction time
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

Running `scripts/benchmark_gpu.py` writes:

- `outputs/benchmarks/gpu_lane_detection_frames.csv`
- `outputs/benchmarks/gpu_lane_detection_summary.csv`

Generated comparison clips from the latest run:

- `outputs/sample_cpu_gpufit_compare_300.mp4`
- `outputs/sample_gpu_gpufit_compare_300.mp4`

## Latest Results

Input:

```text
docs/assets/sample.avi
Resolution: 1920x1080
Benchmark frames: 300
```

Summary:

```text
CPU baseline: 27.900 ms/frame, ~35.84 FPS
CUDA lane path: 10.359 ms/frame, ~96.53 FPS
Speedup: ~2.69x
```

CUDA stage averages:

```text
copy_h2d_ms:                 2.1511
grayscale_ms:                0.0644
blur_ms:                     0.0302
edges_ms:                    0.0319
roi_ms:                      0.0728
lane_stats_ms:               0.1921
gpu_kernel_total_ms:         0.3913
copy_d2h_ms:                 0.0957
gpu_preprocess_total_ms:     2.6381
fit_lanes_ms:                0.0636
overlay_ms:                  4.3541
total_ms:                   10.3593
```

Validation:

```text
CPU lanes detected on sample frame: 2
GPU lanes detected on sample frame: 2
300-frame video compare: 1920x1080 at 59.94 FPS
Average visual diff: 1.355 pixel intensity
```

Phase 7 generated assets:

- `docs/assets/phase7_cpu_vs_cuda_demo.mp4`
- `docs/assets/phase7_fps_comparison.png`
- `docs/assets/phase7_cuda_latency_breakdown.png`
- `outputs/benchmarks/phase7_demo_summary.csv`

## Why this matters

The original CUDA version only accelerated preprocessing and still ran CPU Hough lane fitting, which made GPU mode slower than CPU. The optimized version reuses fixed-size GPU buffers, avoids downloading the full ROI image in fast mode, and reduces lane pixels into left/right regression statistics on the GPU. CPU Hough is no longer on the hot path. The remaining major costs are frame upload and CPU overlay rendering.

Next optimization targets:

- use pinned host memory to reduce transfer overhead
- fuse small preprocessing kernels
- move overlay rendering or perspective warp to CUDA if the project needs more headroom

