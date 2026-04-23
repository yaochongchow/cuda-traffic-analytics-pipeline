# PC CUDA Handoff Plan

This README is the handoff for the parts that should be completed on an NVIDIA Windows or Linux machine rather than on the M1 MacBook Pro.

## Current State

Already completed on the Mac:

- Python project structure
- OpenCV CPU baseline
- Lane detection logic
- Sample image and sample video generation
- CPU benchmark tooling
- Portfolio screenshot and demo-media generation

Still to do on the NVIDIA PC:

- Install NVIDIA driver
- Install CUDA Toolkit
- Verify `nvidia-smi`
- Add CUDA or Numba CUDA kernels
- Benchmark CPU vs GPU
- Collect FPS and latency numbers
- Generate final comparison charts
- Record a polished demo video

## Recommended PC Setup

- OS: Ubuntu 22.04 or Windows 11
- GPU: NVIDIA RTX or GTX card with current drivers
- Python: 3.11 or 3.12
- CUDA Toolkit: 12.x
- Libraries: `numpy`, `opencv-python`, `numba`, `matplotlib`

## Phase 1: Environment Setup

1. Install the latest NVIDIA driver for the GPU.
2. Install the CUDA Toolkit.
3. Verify the GPU is visible:

```bash
nvidia-smi
```

4. Verify Python can see CUDA through Numba:

```bash
python -c "from numba import cuda; print(cuda.is_available()); print(cuda.detect())"
```

Expected outcome:

- `nvidia-smi` works
- `cuda.is_available()` returns `True`

## Phase 2: Add Numba CUDA Kernels

Create a GPU module that mirrors the CPU stages:

- grayscale kernel
- Gaussian or box blur kernel
- Sobel or Canny-prep kernel
- threshold kernel
- optional lane-mask or ROI kernel

Recommended file additions:

```text
src/cuda_image_processing/gpu_numba.py
src/cuda_image_processing/gpu_pipeline.py
```

Recommended implementation order:

1. Grayscale kernel
2. Blur kernel
3. Sobel or threshold kernel
4. End-to-end GPU preprocessing path
5. Hybrid lane fitting on CPU if needed

## Phase 3: GPU Validation

For each operation:

- compare CPU and GPU output visually
- compute mean absolute error
- compute max absolute difference

Suggested validation script additions:

```text
scripts/validate_cpu_vs_gpu.py
```

## Phase 4: Benchmark CPU vs GPU

Use the current CPU benchmark output as the baseline.

Benchmark both:

- kernel-only time
- end-to-end GPU time including host/device transfer
- average frame time
- FPS

Suggested measurements:

- grayscale ms
- blur ms
- edge-detection ms
- full preprocessing ms
- lane-overlay ms
- total frame ms
- CPU FPS
- GPU FPS
- speedup

Suggested command shape:

```bash
python3 scripts/benchmark_cpu.py --video data/sample_lane_video.mp4
python3 scripts/benchmark_gpu.py --video data/sample_lane_video.mp4
```

## Phase 5: Final Charts

After CPU and GPU CSV files exist, generate:

- CPU vs GPU time per stage
- full-pipeline FPS comparison
- transfer time vs kernel-only time
- speedup by operation

Suggested output files:

```text
outputs/charts/cpu_vs_gpu_stage_time.png
outputs/charts/fps_comparison.png
outputs/charts/gpu_transfer_vs_kernel.png
outputs/charts/speedup_by_stage.png
```

## Phase 6: Final Demo Recording

Record a demo that shows:

1. Original sample video
2. CPU lane detection output
3. GPU lane detection output
4. Benchmark summary
5. Final charts

Suggested demo flow:

- show `nvidia-smi`
- run the benchmark command
- open generated comparison charts
- play the processed output video

## Definition of Done

The PC work is complete when all of these are true:

- CUDA environment is verified
- at least one Numba CUDA kernel runs successfully
- CPU vs GPU benchmark CSVs exist
- comparison charts are generated
- FPS and latency numbers are recorded
- a final demo clip or screen recording is exported

