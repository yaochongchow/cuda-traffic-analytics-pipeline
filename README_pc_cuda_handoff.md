# NVIDIA PC CUDA Handoff

This file is the practical handoff for the phases that cannot be completed on the M1 MacBook Pro.

Use this when moving the project to a Windows or Linux machine with an NVIDIA GPU.

## What Comes From the Mac

Already completed:

- Python project structure
- CPU OpenCV lane detection baseline
- image and video processing scripts
- synthetic sample image and video
- CPU benchmark CSV generation
- portfolio screenshots and demo clip generation
- phased roadmap in `docs/project_phases.md`

The Mac version proves the project logic before CUDA enters the picture.

## What Happens on the PC

The NVIDIA PC work turns the project into the full CUDA lane detection pipeline:

```text
CPU baseline
        |
        v
CUDA preprocessing kernels
        |
        v
Hybrid CUDA lane detection
        |
        v
CPU vs GPU benchmark results
        |
        v
Final charts and demo video
```

## Recommended Setup

- OS: Ubuntu 22.04, Ubuntu 24.04, or Windows 11
- GPU: NVIDIA GTX, RTX, or workstation GPU
- CUDA Toolkit: 12.x
- Python: 3.11 or 3.12
- Packages: `numpy`, `opencv-python`, `numba`, `matplotlib`

## Phase PC-1: Verify NVIDIA Environment

Install the NVIDIA driver and CUDA Toolkit, then verify:

```bash
nvidia-smi
nvcc --version
```

Verify Numba CUDA:

```bash
python3 -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
```

Done when:

- `nvidia-smi` shows the GPU
- `nvcc --version` shows CUDA
- `cuda.is_available()` is `True`

## Phase PC-2: Add CUDA Preprocessing Kernels

Start with Numba CUDA because it fits the current Python repo.

Suggested files:

```text
src/cuda_image_processing/gpu_numba.py
src/cuda_image_processing/gpu_pipeline.py
```

Kernel order:

1. Grayscale
2. Blur
3. Sobel
4. Threshold
5. ROI mask

The first GPU target should be preprocessing, not the entire lane detector.

Done when:

- each kernel runs on one image
- GPU output can be copied back to CPU
- intermediate GPU outputs can be saved for inspection

## Phase PC-3: Validate CPU vs GPU Output

Create:

```text
scripts/validate_cpu_vs_gpu.py
```

Validation should compute:

- mean absolute error
- max pixel difference
- difference images
- visual side-by-side comparisons

Suggested outputs:

```text
outputs/validation/diff_grayscale.png
outputs/validation/diff_sobel.png
outputs/validation/diff_threshold.png
outputs/validation/cpu_vs_gpu_overlay.png
```

Done when:

- GPU preprocessing is close enough to CPU preprocessing
- differences are explainable through rounding, border handling, or algorithm choices

## Phase PC-4: Add Hybrid CUDA Mode

Recommended split:

```text
GPU:
- grayscale
- blur
- Sobel
- threshold
- optional ROI mask

CPU:
- perspective warp
- lane pixel search
- polynomial fit
- overlay rendering
```

Suggested command shape:

```bash
python3 scripts/run_lane_detection.py --video data/real_drive_clip.mp4 --mode cpu
python3 scripts/run_lane_detection.py --video data/real_drive_clip.mp4 --mode cuda --write-video
```

Done when:

- CPU mode still works
- CUDA mode produces a lane overlay
- processed CUDA video is saved

## Phase PC-5: Benchmark CPU vs GPU

Create:

```text
scripts/benchmark_gpu.py
scripts/plot_benchmarks.py
```

Benchmark both kernel-only and end-to-end time.

Important timing categories:

- CPU preprocessing time
- CUDA kernel time
- host-to-device copy time
- device-to-host copy time
- GPU end-to-end preprocessing time
- full frame time
- FPS

Suggested command shape:

```bash
python3 scripts/benchmark_cpu.py --video data/real_drive_clip.mp4
python3 scripts/benchmark_gpu.py --video data/real_drive_clip.mp4
python3 scripts/plot_benchmarks.py
```

Done when:

- CPU and GPU CSVs exist
- charts are generated
- FPS comparison is available
- speedup is calculated from real numbers

## Phase PC-6: Optimize

After the basic CUDA path works, improve it.

Optimization order:

1. Keep data on GPU across multiple kernels
2. Fuse Sobel and threshold
3. Add shared memory Sobel
4. Use pinned host memory
5. Add CUDA streams for video throughput
6. Move ROI mask to CUDA
7. Consider CUDA perspective warp as a stretch goal

Done when:

- optimized mode improves either preprocessing time or total FPS
- benchmark charts show the difference between naive and optimized GPU paths

## Phase PC-7: Final Demo

Final deliverables:

- original road frame
- CPU lane output
- CUDA lane output
- side-by-side demo video
- FPS overlay
- benchmark charts
- final README numbers

Suggested final outputs:

```text
docs/assets/original_frame.png
docs/assets/cpu_lane_output.png
docs/assets/cuda_lane_output.png
docs/assets/cpu_vs_cuda_demo.mp4
outputs/charts/fps_comparison.png
outputs/charts/per_stage_latency.png
outputs/charts/speedup_by_stage.png
```

Done when:

- the README includes real FPS and latency numbers
- the demo clearly shows CPU vs CUDA output
- the resume bullet includes measured improvement

