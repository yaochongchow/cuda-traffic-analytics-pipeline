# CUDA-Accelerated Real-Time Traffic Analytics Pipeline

This project combines the CUDA image-processing work in this repo with the real-time perception modules from [`traffic-monitor`](https://github.com/yaochongchow/traffic-monitor).

The long-term goal is to build a real-time traffic analytics system that compares:

- CPU-only OpenCV lane detection
- advanced lane detection with perspective warp, color masks, RANSAC, and temporal tracking
- optional YOLOv8 traffic object detection and tracking
- GPU-accelerated preprocessing with CUDA or Numba CUDA
- optional optimized GPU versions using shared memory, kernel fusion, pinned memory, streams, and profiling

In plain terms:

```text
Current Mac phase:
Build and validate the traffic analytics baseline.

NVIDIA PC phase:
Move expensive pixel-level preprocessing to CUDA and benchmark the speedup.
```

## Current Status

Completed on the M1 MacBook Pro:

- Python project structure under `src/`
- OpenCV CPU baseline
- lane detection for images and videos
- synthetic sample image and video generation
- CPU benchmark scripts
- Numba CUDA kernel modules and GPU CLI hooks
- advanced traffic-monitor lane detector
- perspective transform module
- optional YOLOv8 traffic object detection module
- combined realtime traffic analytics runner
- portfolio screenshots and demo clip generation
- documentation for the full phased CUDA roadmap

Completed on the NVIDIA PC:

- NVIDIA driver setup
- CUDA runtime/NVVM setup for Numba CUDA
- `nvidia-smi` verification
- running and validating the CUDA path on an RTX 3080 Ti
- GPU lane-stat reduction replacing CPU Hough in CUDA mode
- CPU vs GPU benchmarks on 1080p sample footage
- Phase 7 side-by-side demo generation and benchmark charts
- local road-video workflow for `sample.avi` and `sample_cropped.avi`

Large input and demo videos are intentionally not committed. Keep local `.avi` and `.mp4` files under `docs/assets/` or `outputs/`; Git ignores them so pushes stay small.

## Project Phases

The project should be treated as one phased portfolio project, not as a random collection of scripts.

| Phase | Name | Machine | Status |
|---:|---|---|---|
| 0 | M1 CPU Scaffold | M1 Mac | Done |
| 1 | CPU Lane Detection Baseline | M1 Mac | Done |
| 2 | Real Road Data and Tuning | M1 Mac | Next |
| 3 | Advanced Lane Pipeline | M1 Mac or PC | Planned |
| 4 | CUDA Preprocessing | NVIDIA PC | Done |
| 5 | Hybrid CUDA Traffic Analytics | NVIDIA PC | Baseline Implemented |
| 6 | CPU vs GPU Benchmarking | NVIDIA PC | Initial 1080p Complete |
| 7 | Optimization and Final Demo | NVIDIA PC | Implemented |

The full roadmap is documented in [docs/project_phases.md](docs/project_phases.md).

The NVIDIA handoff plan is documented in [README_pc_cuda_handoff.md](README_pc_cuda_handoff.md).

## Current Traffic Analytics Pipeline

The project now has two lane paths:

```text
Baseline lane mode:
grayscale -> blur -> Canny -> ROI -> Hough lines -> overlay

Advanced lane mode:
road color calibration -> white/yellow masks -> perspective warp
-> sliding-window search -> RANSAC/tracking -> lane-area overlay
```

The combined traffic analytics path is:

```text
Input Frame
        |
        v
Optional YOLOv8 Object Detection
        |
        v
Mask detected vehicles from lane input
        |
        v
CPU/CUDA Lane Detection
        |
        v
Optional Perspective Display
        |
        v
Lane, object, FPS, and status overlay
```

The main baseline implementation is in [lane_detection.py](src/cuda_image_processing/lane_detection.py).

The advanced traffic-monitor lane detector is in [advanced_lane_detection.py](src/cuda_image_processing/advanced_lane_detection.py).

The optional YOLO traffic detector is in [object_detection.py](src/cuda_image_processing/object_detection.py).

## Optimized CUDA Pipeline

The optimized CUDA lane path keeps preprocessing and lane-pixel reduction on the GPU:

```text
Input Frame
        |
        v
Reusable GPU frame buffers
        |
        v
CUDA Grayscale
        |
        v
CUDA Blur
        |
        v
CUDA Sobel
        |
        v
CUDA Threshold
        |
        v
CUDA ROI Mask
        |
        v
CUDA left/right lane-stat reduction
        |
Copy compact lane stats back to CPU
        |
        v
Overlay Rendering
```

Latest 1080p benchmark on local `docs/assets/sample.avi`:

```text
CPU baseline: 27.900 ms/frame, ~35.84 FPS
CUDA optimized: 10.359 ms/frame, ~96.53 FPS
Speedup: ~2.69x
```

## Project Layout

```text
cuda_image_processing/
|-- README.md
|-- pyproject.toml
|-- data/
|   |-- sample_lane_frame.png
|   `-- sample_lane_video.mp4
|-- docs/
|   |-- architecture.md
|   |-- benchmarking.md
|   |-- project_phases.md
|   `-- assets/
|       |-- phase7_cuda_latency_breakdown.png
|       `-- phase7_fps_comparison.png
|-- scripts/
|   |-- benchmark_cpu.py
|   |-- benchmark_gpu.py
|   |-- generate_sample_data.py
|   |-- prepare_phase7_demo_assets.py
|   |-- run_lane_detection.py
|   |-- run_traffic_analytics.py
|   `-- validate_cpu_vs_gpu.py
|-- src/
|   `-- cuda_image_processing/
|       |-- advanced_lane_detection.py
|       |-- benchmarking.py
|       |-- gpu_numba.py
|       |-- gpu_pipeline.py
|       |-- lane_detection.py
|       |-- object_detection.py
|       |-- perspective.py
|       `-- realtime_pipeline.py
`-- tests/
    `-- test_cpu_pipeline.py
```

## Quick Start on M1 Mac

Generate sample data:

```bash
python3 scripts/generate_sample_data.py
```

Run lane detection on the sample image:

```bash
python3 scripts/run_lane_detection.py --image data/sample_lane_frame.png
```

Run lane detection on the sample video:

```bash
python3 scripts/run_lane_detection.py --video data/sample_lane_video.mp4 --write-video
```

Run the CUDA path on an NVIDIA PC:

```bash
python3 scripts/run_lane_detection.py --video data/sample_lane_video.mp4 --mode cuda --write-video
```

On Windows with the local CUDA Python wheel layout, set the CUDA runtime paths first:

```powershell
$env:CUDA_PATH = "$PWD\.venv\Lib\site-packages\nvidia\cuda_nvcc"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\nvvm\bin;$PWD\.venv\Lib\site-packages\nvidia\cuda_runtime\bin;$env:PATH"
```

Run the CPU benchmark:

```bash
python3 scripts/benchmark_cpu.py --video data/sample_lane_video.mp4
```

Run the GPU benchmark on an NVIDIA PC:

```bash
python3 scripts/benchmark_gpu.py --video data/sample_lane_video.mp4
```

Generate portfolio-ready screenshots and demo media:

```bash
python3 scripts/prepare_portfolio_assets.py
```

Run combined traffic analytics on the sample video:

```bash
python3 scripts/run_traffic_analytics.py --video data/sample_lane_video.mp4 --limit-frames 60
```

Run advanced lanes plus YOLOv8 object detection after installing optional traffic dependencies:

```bash
python3 -m pip install -e ".[traffic]"
python3 scripts/run_traffic_analytics.py --video data/real_drive_clip.mp4 --objects --display
```

Run the CUDA traffic analytics preview on a local real road clip:

```powershell
.\.venv\Scripts\python.exe scripts\run_traffic_analytics.py --video docs\assets\sample.avi --mode cuda --lane-mode baseline --objects --display
```

Generate the Phase 7 benchmark charts and side-by-side demo locally:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_phase7_demo_assets.py --video docs\assets\sample.avi
```

Run smoke tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Outputs

Generated runtime outputs are written to `outputs/` and are intentionally gitignored.

Large local video assets are also gitignored:

- `docs/assets/*.avi`
- `docs/assets/*.mp4`

Committed portfolio-ready assets in `docs/assets/` should be small images, charts, or other lightweight files.

## Next Development Step

The best next step is to keep tuning real-road lane stability on cropped traffic footage, especially where intersections, crosswalks, parked cars, and curb posts produce lane-like edges.
