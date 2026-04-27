# Architecture

The repository is organized around a CPU lane-detection baseline and an optimized CUDA lane path for NVIDIA PCs.

## Data Flow

```text
sample_data.py or user input
        |
        v
run_lane_detection.py / run_traffic_analytics.py
        |
        v
lane_detection.py
  - grayscale
  - blur
  - edges
  - ROI mask
  - Hough segments
  - averaged lane fit
  - overlay
        |
        v
benchmarking.py / portfolio.py
        |
        v
outputs/ and docs/assets/
```

The optimized CUDA path follows a different split:

```text
input frame
        |
        v
CudaLaneDetector
        |
        v
reusable GPU buffers
  - BGR to grayscale
  - 3x3 blur
  - Sobel threshold
  - ROI mask
  - left/right lane-stat reduction
        |
        v
compact lane stats copied to CPU
        |
        v
lane-line reconstruction and overlay
        |
        v
outputs/ and docs/assets/
```

## CPU Modules

- `sample_data.py`: Generates deterministic synthetic road images and videos.
- `lane_detection.py`: Contains the OpenCV CPU baseline and lane fitting logic.
- `benchmarking.py`: Measures stage timings and writes CSV benchmark data.
- `portfolio.py`: Builds portfolio screenshots and a demo clip from processed outputs.
- `io_utils.py`: Shared path and image/video helpers.

## CUDA Modules

- `gpu_numba.py`: Numba CUDA kernels plus reusable fixed-shape GPU frame buffers.
- `gpu_pipeline.py`: CUDA lane detector, GPU lane-stat conversion, and output overlay integration.
- `benchmark_gpu.py`: Per-stage CUDA benchmark CSV generation.
- `prepare_phase7_demo_assets.py`: Side-by-side CPU/CUDA demo video and benchmark chart generation.

The CUDA path keeps full-frame preprocessing and lane-pixel reduction on the GPU. The CPU receives compact left/right lane statistics, reconstructs two display lines, and draws the final overlay.

