# Full Project Phases

This document turns the current M1-friendly project into a full phased CUDA traffic analytics project. The project now combines this repo's CPU/CUDA benchmarking work with the real-time lane and traffic-object modules from `traffic-monitor`.

## Phase 0: M1 CPU Scaffold

Status: Done

Machine: M1 MacBook Pro

Goal:

Create a Python project that can run locally without CUDA.

Completed deliverables:

- Python package under `src/cuda_image_processing/`
- scripts for generation, inference, benchmarking, and portfolio assets
- optional GPU modules and CLI hooks
- advanced traffic-monitor lane and perspective modules
- optional YOLO traffic-object module
- combined traffic analytics runner
- sample image and video generation
- smoke test coverage
- local git repository

Key files:

- `pyproject.toml`
- `scripts/generate_sample_data.py`
- `scripts/run_lane_detection.py`
- `tests/test_cpu_pipeline.py`

## Phase 1: CPU Lane Detection Baseline

Status: Done

Machine: M1 MacBook Pro

Goal:

Build the baseline lane detection path before adding any GPU complexity.

Implemented pipeline:

```text
frame
  -> grayscale
  -> Gaussian blur
  -> Canny edges
  -> ROI mask
  -> Hough line segments
  -> averaged lane lines
  -> overlay
```

Current benchmark:

The synthetic sample benchmark runs through `scripts/benchmark_cpu.py` and writes CSV output to `outputs/benchmarks/`.

Key files:

- `src/cuda_image_processing/lane_detection.py`
- `src/cuda_image_processing/benchmarking.py`
- `docs/benchmarking.md`

Definition of done:

- image input works
- video input works
- processed video can be written
- CPU timings are recorded
- portfolio screenshots and demo media can be generated

## Phase 2: Real Road Data and Tuning

Status: Next

Machine: M1 MacBook Pro

Goal:

Move from synthetic sample data to real driving footage.

Tasks:

- Add clear daytime road images to `data/`
- Add a short real driving video to `data/`
- Tune ROI points for real camera framing
- Tune Canny thresholds
- Tune Hough transform parameters
- Save intermediate stage images for debugging
- Regenerate portfolio assets from real footage

Suggested files to add:

```text
data/real_lane_01.jpg
data/real_lane_02.jpg
data/real_drive_clip.mp4
docs/assets/real_lane_detection_hero.png
docs/assets/real_lane_detection_demo.mp4
```

Useful commands:

```bash
python3 scripts/run_lane_detection.py --image data/real_lane_01.jpg
python3 scripts/run_lane_detection.py --video data/real_drive_clip.mp4 --write-video
python3 scripts/benchmark_cpu.py --video data/real_drive_clip.mp4
```

Definition of done:

- real image output detects both lane boundaries
- real video output is stable enough for a portfolio demo
- CPU benchmark CSV exists for real footage

## Phase 3: Advanced Traffic Analytics Pipeline

Status: Code added, needs real road tuning

Machine: M1 MacBook Pro or NVIDIA PC

Goal:

Upgrade from straight Hough lines to the more advanced traffic-monitor lane pipeline and optional YOLO object detection.

Target pipeline:

```text
frame
  -> resize
  -> grayscale
  -> blur
  -> Sobel
  -> threshold
  -> ROI mask
  -> perspective warp
  -> sliding-window lane search
  -> polynomial curve fitting
  -> lane area overlay
  -> optional YOLO traffic object detection
  -> object tracking
  -> FPS and status overlay
```

Why this phase matters:

The current Hough-line approach is a good baseline, but the pasted full project describes a more realistic lane detection system that can handle curved lanes and bird's-eye lane analysis.

Files added:

```text
src/cuda_image_processing/advanced_lane_detection.py
src/cuda_image_processing/perspective.py
src/cuda_image_processing/object_detection.py
src/cuda_image_processing/realtime_pipeline.py
scripts/run_traffic_analytics.py
```

Definition of done:

- perspective warp works on real road frames
- sliding-window lane search finds left and right lane pixels
- polynomial lane curves are drawn back onto the original frame
- outputs include intermediate images for every major stage
- combined traffic analytics can write an annotated video
- optional YOLO mode works after installing `.[traffic]`

## Phase 4: CUDA Preprocessing

Status: Code added, needs NVIDIA PC testing

Machine: NVIDIA PC

Goal:

Move expensive pixel-level preprocessing stages from CPU OpenCV to CUDA.

Start with Numba CUDA if you want faster iteration in Python. Use C++ CUDA later if you want the final project to match the original C++/CUDA spec exactly.

CUDA kernels to implement:

- grayscale
- box blur or Gaussian-style blur
- Sobel edge detection
- binary threshold
- optional ROI mask

Files added:

```text
src/cuda_image_processing/gpu_numba.py
src/cuda_image_processing/gpu_pipeline.py
scripts/benchmark_gpu.py
scripts/validate_cpu_vs_gpu.py
```

The current CUDA path implements:

- BGR to grayscale
- 3x3 box blur
- Sobel edge thresholding
- trapezoid ROI masking
- CPU Hough lane fitting and overlay after GPU preprocessing

Definition of done:

- `nvidia-smi` works
- `from numba import cuda; cuda.is_available()` returns `True`
- at least grayscale, Sobel, and threshold run on GPU
- CPU and GPU preprocessing outputs are visually similar

## Phase 5: Hybrid CUDA Traffic Analytics

Status: Planned

Machine: NVIDIA PC

Goal:

Connect CUDA preprocessing to the traffic analytics system.

Recommended hybrid split:

```text
GPU:
- grayscale
- blur
- Sobel
- threshold
- optional ROI

CPU:
- perspective warp
- sliding-window search
- polynomial fit
- overlay rendering
```

Why this split is useful:

Pixel-level preprocessing maps naturally to CUDA. Higher-level lane logic is easier to keep on the CPU for the first GPU version.

Definition of done:

- CUDA mode processes images
- CUDA mode processes videos
- CPU and CUDA outputs are functionally similar
- both modes can be selected from scripts or CLI flags
- traffic analytics can compare CPU lane mode and CUDA preprocessing mode

## Phase 6: CPU vs GPU Benchmarking

Status: Planned

Machine: NVIDIA PC

Goal:

Measure whether CUDA improves the real system, not just isolated kernels.

Benchmark categories:

- CPU preprocessing time
- GPU kernel-only preprocessing time
- host-to-device copy time
- device-to-host copy time
- GPU end-to-end preprocessing time
- lane search time
- curve fitting time
- overlay time
- total frame time
- FPS

Recommended CSV schema:

```csv
frame,resolution,mode,grayscale_ms,blur_ms,sobel_ms,threshold_ms,copy_h2d_ms,copy_d2h_ms,lane_search_ms,curve_fit_ms,overlay_ms,total_ms,fps
```

Definition of done:

- CPU benchmark CSV exists
- GPU benchmark CSV exists
- per-stage timings exist
- FPS is reported for both modes
- benchmark results cover at least 720p and 1080p

## Phase 7: Optimization and Final Demo

Status: Planned

Machine: NVIDIA PC

Goal:

Turn the working GPU implementation into a polished portfolio project.

Optimization ideas:

- shared memory Sobel
- kernel fusion
- pinned host memory
- CUDA streams
- keep data on GPU longer
- move ROI mask to CUDA
- optionally move perspective warp to CUDA

Final assets:

- CPU output video
- CUDA output video
- side-by-side comparison video
- benchmark charts
- FPS comparison chart
- README screenshots
- final demo recording

Definition of done:

- final README includes real numbers
- final charts are generated from CSV data
- demo video shows CPU and CUDA modes
- resume bullets include measured speedup

## Final Portfolio Framing

Suggested title:

```text
CUDA-Accelerated Real-Time Traffic Analytics Pipeline
```

Suggested description:

```text
Real-time traffic analytics pipeline with advanced lane detection, optional YOLOv8 object detection, CUDA-accelerated preprocessing, per-stage benchmarking, and FPS comparison across road video inputs.
```

Suggested resume bullet after GPU numbers exist:

```text
Built a CUDA-accelerated real-time traffic analytics pipeline combining lane detection, YOLOv8 object tracking, and GPU preprocessing benchmarks, improving 1080p throughput from X FPS to Y FPS while measuring memory transfer overhead and end-to-end latency.
```
