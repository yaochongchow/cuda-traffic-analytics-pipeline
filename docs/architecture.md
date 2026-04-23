# Architecture

The current repository is organized around a CPU-first lane-detection pipeline that can later be mirrored by a GPU implementation.

## Data Flow

```text
sample_data.py or user input
        |
        v
run_lane_detection.py
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

## CPU Modules

- `sample_data.py`: Generates deterministic synthetic road images and videos.
- `lane_detection.py`: Contains the OpenCV CPU baseline and lane fitting logic.
- `benchmarking.py`: Measures stage timings and writes CSV benchmark data.
- `portfolio.py`: Builds portfolio screenshots and a demo clip from processed outputs.
- `io_utils.py`: Shared path and image/video helpers.

## Future GPU Hook

The CPU pipeline is intentionally decomposed into stages that can later be matched by Numba CUDA kernels on the PC handoff machine.

