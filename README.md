# CUDA-Accelerated Lane Detection Pipeline

This project is being developed in phases. The current repository runs on an M1 MacBook Pro as a CPU-only OpenCV baseline, and the later phases are designed for an NVIDIA PC with CUDA.

The long-term goal is to build a real-time lane detection system that compares:

- CPU-only OpenCV lane detection
- GPU-accelerated preprocessing with CUDA or Numba CUDA
- optional optimized GPU versions using shared memory, kernel fusion, pinned memory, streams, and profiling

In plain terms:

```text
Current Mac phase:
Build and validate the lane detection baseline.

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
- portfolio screenshots and demo clip generation
- documentation for the full phased CUDA roadmap

Deferred to the NVIDIA PC:

- NVIDIA driver setup
- CUDA Toolkit setup
- `nvidia-smi` verification
- CUDA or Numba CUDA kernels
- CPU vs GPU benchmarks
- GPU FPS and latency numbers
- final benchmark charts
- final CUDA demo video

## Project Phases

The project should be treated as one phased portfolio project, not as a random collection of scripts.

| Phase | Name | Machine | Status |
|---:|---|---|---|
| 0 | M1 CPU Scaffold | M1 Mac | Done |
| 1 | CPU Lane Detection Baseline | M1 Mac | Done |
| 2 | Real Road Data and Tuning | M1 Mac | Next |
| 3 | Advanced Lane Pipeline | M1 Mac or PC | Planned |
| 4 | CUDA Preprocessing | NVIDIA PC | Planned |
| 5 | Hybrid CUDA Lane Detection | NVIDIA PC | Planned |
| 6 | CPU vs GPU Benchmarking | NVIDIA PC | Planned |
| 7 | Optimization and Final Demo | NVIDIA PC | Planned |

The full roadmap is documented in [docs/project_phases.md](docs/project_phases.md).

The NVIDIA handoff plan is documented in [README_pc_cuda_handoff.md](README_pc_cuda_handoff.md).

## Current CPU Pipeline

The implemented baseline is intentionally simple and reliable:

```text
Input Frame
        |
        v
Grayscale
        |
        v
Gaussian Blur
        |
        v
Canny Edge Detection
        |
        v
Region of Interest Mask
        |
        v
Hough Line Segments
        |
        v
Averaged Left and Right Lane Lines
        |
        v
Lane Overlay
```

The main implementation is in [lane_detection.py](src/cuda_image_processing/lane_detection.py).

## Future CUDA Pipeline

The target CUDA pipeline will keep high-level lane logic on the CPU at first and move pixel-level preprocessing to the GPU:

```text
Input Frame
        |
        v
Copy frame to GPU
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
Optional CUDA ROI Mask
        |
        v
Copy binary image back to CPU
        |
        v
Perspective Warp
        |
        v
Lane Pixel Search
        |
        v
Polynomial Fit
        |
        v
Overlay Rendering
```

This gives the project the same shape as the full lane detection spec: first build the working vision system, then accelerate the expensive image preprocessing stages.

## Project Layout

```text
cuda_image_processing/
├── README.md
├── README_pc_cuda_handoff.md
├── pyproject.toml
├── data/
│   ├── sample_lane_frame.png
│   └── sample_lane_video.mp4
├── docs/
│   ├── architecture.md
│   ├── benchmarking.md
│   ├── project_phases.md
│   └── assets/
│       ├── lane_detection_demo.mp4
│       ├── lane_detection_hero.png
│       └── lane_detection_stages.png
├── scripts/
│   ├── benchmark_cpu.py
│   ├── generate_sample_data.py
│   ├── prepare_portfolio_assets.py
│   └── run_lane_detection.py
├── src/
│   └── cuda_image_processing/
│       ├── __init__.py
│       ├── benchmarking.py
│       ├── io_utils.py
│       ├── lane_detection.py
│       ├── portfolio.py
│       └── sample_data.py
└── tests/
    └── test_cpu_pipeline.py
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

Run the CPU benchmark:

```bash
python3 scripts/benchmark_cpu.py --video data/sample_lane_video.mp4
```

Generate portfolio-ready screenshots and demo media:

```bash
python3 scripts/prepare_portfolio_assets.py
```

Run smoke tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Outputs

Generated runtime outputs are written to `outputs/` and are intentionally gitignored.

Committed portfolio-ready assets are written to `docs/assets/`.

## Next Development Step

The best next step is Phase 2: replace the synthetic sample video with real road footage and tune the CPU detector against real lanes before starting CUDA work.

After that, move to the NVIDIA PC and follow [README_pc_cuda_handoff.md](README_pc_cuda_handoff.md).

