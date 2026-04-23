# CUDA Image Processing on M1: CPU Baseline

This repository is now structured so it works on an M1 MacBook Pro without CUDA, while keeping the project aligned with the original GPU-acceleration goal.

What is implemented here:

- Python project structure under `src/`
- OpenCV CPU baseline
- Lane detection pipeline for images and video
- Synthetic sample image and video generation
- CPU benchmark scripts
- Portfolio asset generation for screenshots and demo clips
- Documentation for the current Mac workflow
- A separate PC handoff plan for CUDA work in `README_pc_cuda_handoff.md`

What is intentionally deferred to a Windows/Linux NVIDIA machine:

- NVIDIA driver setup
- CUDA Toolkit setup
- `nvidia-smi`
- CUDA or Numba CUDA kernels
- CPU vs GPU benchmarks
- FPS and latency collection for GPU runs
- Final comparison charts
- GPU demo recording

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

## CPU Pipeline

The baseline lane-detection pipeline is:

1. Convert BGR input to grayscale
2. Apply Gaussian blur
3. Run Canny edge detection
4. Apply a trapezoidal region-of-interest mask
5. Detect line segments with probabilistic Hough transform
6. Fit averaged left and right lane lines
7. Overlay the lane estimate back onto the original frame

This is intentionally CPU-only so it can run cleanly on Apple Silicon today.

## Quick Start

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

Run the smoke tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Outputs

Generated runtime outputs are written to `outputs/` and are intentionally gitignored.

Committed portfolio-ready assets are written to `docs/assets/`.

## Next Step for PC Work

For the CUDA handoff and benchmark plan you want to run on your NVIDIA PC, use:

[`README_pc_cuda_handoff.md`](README_pc_cuda_handoff.md)

