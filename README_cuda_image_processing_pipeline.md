# CUDA-Accelerated Image Processing Pipeline

## 1. Project Overview

This project is a CUDA-accelerated image processing pipeline built to compare **CPU-based image processing** against **GPU-accelerated image processing**.

The goal is to implement common computer vision preprocessing operations using both:

1. A CPU baseline using C++ and OpenCV
2. A GPU version using custom CUDA C++ kernels

The project benchmarks both versions and measures:

- Execution time per image
- Execution time per video frame
- Frames per second
- Speedup from CPU to GPU
- GPU memory transfer overhead
- Performance differences across image resolutions

This project is designed to demonstrate low-level GPU programming fundamentals, real-time computer vision preprocessing, and performance optimization.

---

## 2. Why This Project Matters

Many computer vision and machine learning systems spend a large amount of time preprocessing images before inference.

Examples include:

- resizing images
- converting RGB to grayscale
- applying blur
- detecting edges
- normalizing pixel values
- thresholding
- running convolution filters

On the CPU, these operations can become slow when processing high-resolution images or real-time video.

A GPU can process thousands of pixels in parallel, making it well-suited for image processing workloads.

This project answers the question:

> How much faster can basic image processing operations run when implemented with CUDA instead of CPU code?

---

## 3. What This Project Demonstrates

This project demonstrates:

- CUDA C++ programming
- GPU thread and block organization
- CPU vs GPU benchmarking
- Image processing fundamentals
- Memory transfer between host and device
- Performance profiling
- OpenCV integration
- Real-time video frame processing
- Computer vision preprocessing acceleration

---

## 4. Core Features

The project implements the following image operations:

### CPU Baseline

Implemented using OpenCV and standard C++.

- RGB to grayscale
- Gaussian blur
- Sobel edge detection
- Image thresholding
- Image resizing
- Basic convolution filter

### CUDA GPU Version

Implemented using custom CUDA kernels.

- CUDA grayscale conversion
- CUDA box blur or Gaussian-style blur
- CUDA Sobel edge detection
- CUDA binary thresholding
- CUDA image resizing
- CUDA convolution filter

### Benchmarking

The project compares CPU and GPU versions using:

- single image benchmark
- batch image benchmark
- video frame benchmark
- resolution-based benchmark

Example resolutions:

- 640 x 480
- 1280 x 720
- 1920 x 1080
- 3840 x 2160, optional

---

## 5. Example Pipeline

The full image processing pipeline looks like this:

```text
Input Image / Video Frame
        |
        v
RGB to Grayscale
        |
        v
Blur Filter
        |
        v
Sobel Edge Detection
        |
        v
Thresholding
        |
        v
Output Image / Processed Frame
```

The same pipeline is implemented in two ways:

```text
CPU Pipeline:
OpenCV CPU functions

GPU Pipeline:
Custom CUDA kernels
```

Then the project compares their performance.

---

## 6. Example Output

After running the benchmark, the terminal output may look like this:

```text
Input image: data/road_1080p.jpg
Resolution: 1920 x 1080

Operation              CPU Time      GPU Time      Speedup
----------------------------------------------------------
Grayscale              4.82 ms       0.71 ms       6.79x
Blur                   12.45 ms      2.10 ms       5.93x
Sobel Edge Detection   16.30 ms      2.85 ms       5.72x
Threshold              3.95 ms       0.54 ms       7.31x
Full Pipeline          37.52 ms      6.20 ms       6.05x

CPU FPS: 26.65
GPU FPS: 161.29
```

The exact results will depend on your GPU, CPU, image resolution, kernel implementation, and memory transfer strategy.

---

## 7. Technologies Used

- C++
- CUDA C++
- OpenCV
- CMake
- NVIDIA GPU
- Linux or Windows with CUDA support

Optional:

- Python for plotting benchmark results
- Matplotlib for graphs
- Nsight Systems or Nsight Compute for profiling

---

## 8. Prerequisites

### Required

You need:

- NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit
- C++ compiler
- CMake
- OpenCV

### Important Note for macOS Users

CUDA does not run on Apple Silicon Macs or most modern macOS systems.

If you are using a Mac, you should use one of these options:

1. A desktop or laptop with an NVIDIA GPU
2. A remote Linux machine with an NVIDIA GPU
3. A cloud GPU instance
4. Google Colab for CUDA experiments, though C++ project setup is less convenient there

---

## 9. Recommended System

This project is best developed on:

```text
OS: Ubuntu 22.04 or 24.04
GPU: NVIDIA GTX / RTX GPU
CUDA: 11.x or 12.x
Compiler: g++ or nvcc-compatible compiler
Build system: CMake
```

---

## 10. Project Structure

Recommended repository structure:

```text
cuda-image-processing/
│
├── README.md
├── CMakeLists.txt
│
├── data/
│   ├── sample_720p.jpg
│   ├── sample_1080p.jpg
│   └── sample_video.mp4
│
├── include/
│   ├── cpu_ops.hpp
│   ├── cuda_ops.cuh
│   ├── benchmark.hpp
│   └── image_utils.hpp
│
├── src/
│   ├── main.cpp
│   ├── cpu_ops.cpp
│   ├── cuda_ops.cu
│   ├── benchmark.cpp
│   └── image_utils.cpp
│
├── outputs/
│   ├── grayscale_cpu.png
│   ├── grayscale_gpu.png
│   ├── sobel_cpu.png
│   ├── sobel_gpu.png
│   └── benchmark_results.csv
│
├── scripts/
│   └── plot_benchmarks.py
│
└── docs/
    ├── architecture.md
    └── benchmark_notes.md
```

---

## 11. High-Level Architecture

```text
main.cpp
   |
   | loads image/video
   v
image_utils.cpp
   |
   | sends image to CPU and GPU pipelines
   v
cpu_ops.cpp              cuda_ops.cu
   |                         |
   | OpenCV CPU operations   | CUDA kernels
   v                         v
benchmark.cpp
   |
   | records timing results
   v
outputs/
   |
   | saves processed images and CSV benchmark results
   v
scripts/plot_benchmarks.py
```

---

## 12. Step-by-Step Development Guide

This section explains how to build the project from scratch.

---

### Step 1: Create the Repository

Create a new folder:

```bash
mkdir cuda-image-processing
cd cuda-image-processing
```

Initialize Git:

```bash
git init
```

Create the folder structure:

```bash
mkdir src include data outputs scripts docs
touch README.md CMakeLists.txt
touch src/main.cpp src/cpu_ops.cpp src/cuda_ops.cu src/benchmark.cpp src/image_utils.cpp
touch include/cpu_ops.hpp include/cuda_ops.cuh include/benchmark.hpp include/image_utils.hpp
```

---

### Step 2: Install Dependencies

On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config
sudo apt install libopencv-dev
```

Check OpenCV:

```bash
pkg-config --modversion opencv4
```

Check CUDA:

```bash
nvcc --version
```

Check NVIDIA GPU:

```bash
nvidia-smi
```

If `nvidia-smi` works, your system can see the NVIDIA GPU.

---

### Step 3: Create the CPU Baseline

The CPU version should use OpenCV.

Example CPU operations:

```cpp
cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
cv::Sobel(blurred, sobel, CV_8U, 1, 1);
cv::threshold(sobel, output, 100, 255, cv::THRESH_BINARY);
```

The CPU baseline is important because it gives you something to compare against.

Without a CPU baseline, you cannot prove the GPU version is faster.

---

### Step 4: Implement CUDA Grayscale Conversion

The first CUDA kernel should be grayscale conversion.

Each GPU thread handles one pixel.

Formula:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Conceptually:

```text
Thread 0 processes pixel 0
Thread 1 processes pixel 1
Thread 2 processes pixel 2
...
Thread N processes pixel N
```

This is a good first kernel because each pixel can be processed independently.

#### CUDA Grayscale Kernel Concept

```cpp
__global__ void rgbToGrayKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int grayIndex = y * width + x;
    int rgbIndex = (y * width + x) * channels;

    unsigned char b = input[rgbIndex + 0];
    unsigned char g = input[rgbIndex + 1];
    unsigned char r = input[rgbIndex + 2];

    output[grayIndex] = static_cast<unsigned char>(
        0.299f * r + 0.587f * g + 0.114f * b
    );
}
```

This teaches:

- `blockIdx`
- `blockDim`
- `threadIdx`
- 2D grid layout
- bounds checking
- pixel indexing

---

### Step 5: Understand CUDA Thread Layout

CUDA runs many threads in parallel.

You organize threads into blocks, and blocks into a grid.

Example:

```cpp
dim3 blockSize(16, 16);
dim3 gridSize(
    (width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y
);
```

For a 1920 x 1080 image:

```text
Each block has 16 x 16 = 256 threads

Grid X = ceil(1920 / 16) = 120 blocks
Grid Y = ceil(1080 / 16) = 68 blocks

Total blocks = 120 x 68 = 8160 blocks
Total possible threads = 8160 x 256 = 2,088,960 threads
```

Each thread processes one pixel.

This is why GPUs are powerful for image processing.

---

### Step 6: Implement CUDA Blur

A blur kernel looks at neighboring pixels.

For each pixel, compute the average of nearby pixels.

Example 3 x 3 blur:

```text
p1 p2 p3
p4 p5 p6
p7 p8 p9

output = average(p1 through p9)
```

This teaches:

- neighbor access
- boundary handling
- memory access patterns
- convolution-style operations

Simple version:

```text
For each pixel:
    sum neighboring pixels
    divide by number of valid neighbors
```

Advanced version:

- use shared memory
- reduce global memory reads
- compare naive vs optimized kernel

Start with the naive version first.

---

### Step 7: Implement CUDA Sobel Edge Detection

Sobel edge detection finds strong changes in image intensity.

It uses two filters:

```text
Gx filter:

-1  0  1
-2  0  2
-1  0  1

Gy filter:

-1 -2 -1
 0  0  0
 1  2  1
```

The final edge strength is:

```text
magnitude = sqrt(Gx^2 + Gy^2)
```

Simplified version:

```text
magnitude = abs(Gx) + abs(Gy)
```

The simplified version is faster and common in real-time pipelines.

This operation is very good for benchmarking because it requires multiple memory reads per pixel.

---

### Step 8: Implement CUDA Thresholding

Thresholding converts an image into black and white.

Example:

```text
if pixel > threshold:
    output = 255
else:
    output = 0
```

This is simple but useful for image segmentation and edge maps.

It also teaches branch behavior on the GPU.

---

### Step 9: Implement CUDA Image Resize

Image resizing is useful because ML models often require fixed input sizes.

Examples:

- 1920 x 1080 to 640 x 640
- 1280 x 720 to 224 x 224
- 640 x 480 to 320 x 240

Start with nearest-neighbor resizing.

Later, implement bilinear interpolation.

Nearest-neighbor is easier:

```text
output_x maps to input_x
output_y maps to input_y
copy nearest input pixel
```

Bilinear interpolation is better quality but more complex.

---

### Step 10: Build the Benchmark System

Benchmark each operation separately.

Measure:

```text
CPU grayscale time
GPU grayscale time

CPU blur time
GPU blur time

CPU Sobel time
GPU Sobel time

CPU threshold time
GPU threshold time

CPU full pipeline time
GPU full pipeline time
```

Recommended output format:

```csv
operation,resolution,cpu_ms,gpu_ms,speedup
grayscale,1920x1080,4.82,0.71,6.79
blur,1920x1080,12.45,2.10,5.93
sobel,1920x1080,16.30,2.85,5.72
threshold,1920x1080,3.95,0.54,7.31
full_pipeline,1920x1080,37.52,6.20,6.05
```

Save the result as:

```text
outputs/benchmark_results.csv
```

---

### Step 11: Use Correct GPU Timing

For CPU timing, use:

```cpp
std::chrono::high_resolution_clock
```

For GPU timing, use CUDA events:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

kernel<<<gridSize, blockSize>>>(...);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

CUDA events are better for measuring GPU execution time because GPU kernels run asynchronously.

If you only use CPU timers, you may measure the kernel launch time incorrectly.

---

### Step 12: Account for Memory Transfer Time

GPU processing has three major phases:

```text
1. Copy input image from CPU memory to GPU memory
2. Run CUDA kernel
3. Copy output image from GPU memory back to CPU memory
```

Benchmark both:

```text
Kernel-only time
End-to-end GPU time including memory copy
```

This matters because sometimes the GPU kernel is fast, but memory transfer is expensive.

Example:

```text
GPU kernel only: 0.70 ms
Host to device copy: 1.20 ms
Device to host copy: 1.10 ms
Total GPU time: 3.00 ms
```

For real systems, end-to-end time is usually more important.

---

### Step 13: Build the Project

Create a build folder:

```bash
mkdir build
cd build
```

Run CMake:

```bash
cmake ..
```

Build:

```bash
make -j
```

Run:

```bash
./cuda_image_pipeline ../data/sample_1080p.jpg
```

Example output:

```text
Loaded image: ../data/sample_1080p.jpg
Resolution: 1920 x 1080

Running CPU pipeline...
Running CUDA pipeline...

Saved outputs:
outputs/grayscale_cpu.png
outputs/grayscale_gpu.png
outputs/sobel_cpu.png
outputs/sobel_gpu.png

Saved benchmark:
outputs/benchmark_results.csv
```

---

### Step 14: Add Video Processing

After image processing works, add video support.

Pipeline:

```text
Open video with OpenCV
Read frame
Run CPU pipeline
Run CUDA pipeline
Measure time per frame
Display or save output
Calculate FPS
```

Example command:

```bash
./cuda_image_pipeline --video ../data/sample_video.mp4
```

Example output:

```text
Video: sample_video.mp4
Resolution: 1920 x 1080
Frames processed: 300

CPU average frame time: 38.6 ms
CPU FPS: 25.9

GPU average frame time: 7.4 ms
GPU FPS: 135.1

Speedup: 5.2x
```

---

### Step 15: Plot Benchmark Results

Use Python to plot the CSV results.

Example graph ideas:

1. CPU vs GPU time by operation
2. Speedup by operation
3. FPS comparison
4. Resolution vs processing time
5. Kernel-only time vs end-to-end time

Example command:

```bash
python3 scripts/plot_benchmarks.py outputs/benchmark_results.csv
```

Output:

```text
outputs/cpu_vs_gpu_time.png
outputs/speedup_by_operation.png
outputs/fps_comparison.png
```

---

## 13. CUDA Concepts Explained

### What is a CUDA Kernel?

A CUDA kernel is a function that runs on the GPU.

CPU function:

```cpp
void processImage(...) {
    // runs on CPU
}
```

CUDA kernel:

```cpp
__global__ void processImageKernel(...) {
    // runs on GPU
}
```

The CPU launches the GPU kernel:

```cpp
processImageKernel<<<gridSize, blockSize>>>(...);
```

---

### What is a Thread?

A thread is one small unit of execution.

In image processing, one thread often processes one pixel.

For example:

```text
Image has 1920 x 1080 = 2,073,600 pixels

GPU launches around 2 million threads

Each thread handles one pixel
```

---

### What is a Block?

A block is a group of threads.

Example:

```cpp
dim3 blockSize(16, 16);
```

This means each block has:

```text
16 x 16 = 256 threads
```

---

### What is a Grid?

A grid is a group of blocks.

For a 1920 x 1080 image:

```text
Grid contains enough blocks to cover the whole image
```

---

### What is Global Memory?

Global memory is the main memory on the GPU.

It is large but slower than shared memory.

Most beginner CUDA projects use global memory first.

---

### What is Shared Memory?

Shared memory is a small, fast memory region shared by threads in the same block.

It can improve performance for operations like blur and Sobel because neighboring pixels are reused.

Beginner version:

```text
Use global memory only
```

Advanced version:

```text
Use shared memory tiling
```

---

## 14. Benchmarking Plan

The benchmark should answer these questions:

### Question 1

How much faster is CUDA than CPU OpenCV for each operation?

### Question 2

Which operation benefits most from GPU acceleration?

### Question 3

How much time is spent copying memory between CPU and GPU?

### Question 4

How does resolution affect speedup?

### Question 5

Does the full pipeline reach real-time performance?

Real-time targets:

```text
30 FPS = 33.3 ms per frame
60 FPS = 16.7 ms per frame
120 FPS = 8.3 ms per frame
```

---

## 15. Recommended Benchmark Table

| Operation | Resolution | CPU Time | GPU Kernel Time | GPU End-to-End Time | Speedup |
|---|---:|---:|---:|---:|---:|
| Grayscale | 1080p | TBD | TBD | TBD | TBD |
| Blur | 1080p | TBD | TBD | TBD | TBD |
| Sobel | 1080p | TBD | TBD | TBD | TBD |
| Threshold | 1080p | TBD | TBD | TBD | TBD |
| Full Pipeline | 1080p | TBD | TBD | TBD | TBD |

---

## 16. Validation

The GPU output should match the CPU output closely.

Because floating-point operations may differ slightly, exact pixel-perfect output is not always required.

Recommended validation methods:

### Method 1: Mean Absolute Error

Compare CPU and GPU output pixel values.

```text
MAE = average absolute difference between CPU image and GPU image
```

### Method 2: Maximum Pixel Difference

```text
max_diff = maximum absolute pixel difference
```

### Method 3: Visual Comparison

Save both outputs:

```text
outputs/sobel_cpu.png
outputs/sobel_gpu.png
```

Open both and visually inspect.

### Method 4: Difference Image

Create an image showing where CPU and GPU differ.

```text
difference = abs(cpu_output - gpu_output)
```

---

## 17. Common Bugs and How to Fix Them

### Bug 1: Output image is black

Possible causes:

- GPU output memory was not copied back to CPU
- kernel did not launch correctly
- wrong image indexing
- input image channels are incorrect
- output buffer not allocated correctly

Check:

```cpp
cudaGetLastError();
cudaDeviceSynchronize();
```

---

### Bug 2: CUDA illegal memory access

Possible causes:

- thread index goes outside image bounds
- wrong width or height
- wrong channel count
- incorrect memory allocation size

Always include bounds checking:

```cpp
if (x >= width || y >= height) {
    return;
}
```

---

### Bug 3: CPU and GPU images look different

Possible causes:

- OpenCV uses BGR, not RGB
- rounding differences
- different blur kernel size
- different border handling
- different Sobel formula

Remember:

```text
OpenCV images usually use BGR channel order
```

---

### Bug 4: GPU is slower than CPU

Possible causes:

- image is too small
- memory transfer dominates runtime
- kernel is not optimized
- too many separate kernel launches
- CPU OpenCV is already highly optimized
- benchmarking method is incorrect

Try:

- larger images
- batch processing
- kernel fusion
- pinned memory
- shared memory optimization
- measuring kernel-only vs end-to-end time

---

## 18. Optimization Ideas

After the basic version works, improve it.

### Optimization 1: Shared Memory Tiling

Use shared memory for blur and Sobel.

Why?

Neighboring pixels are reused many times.

Instead of reading from global memory repeatedly, load a tile into shared memory first.

---

### Optimization 2: Kernel Fusion

Instead of launching separate kernels:

```text
grayscale kernel
blur kernel
sobel kernel
threshold kernel
```

Try combining operations:

```text
grayscale + blur
sobel + threshold
```

This reduces kernel launch overhead and memory reads/writes.

---

### Optimization 3: Pinned Host Memory

Pinned memory can improve CPU-to-GPU transfer speed.

Useful for video pipelines.

---

### Optimization 4: CUDA Streams

CUDA streams allow overlapping memory transfer and kernel execution.

Useful for real-time video.

Example idea:

```text
Frame 1: GPU processing
Frame 2: CPU copying to GPU
Frame 0: GPU copying back to CPU
```

---

### Optimization 5: Batch Processing

Instead of processing one image at a time, process multiple images.

This improves GPU utilization.

---

## 19. Stretch Goals

Once the basic project works, add:

- CUDA shared memory Sobel
- CUDA bilinear resize
- CUDA streams for video
- TensorRT inference integration
- PyTorch model preprocessing
- real-time webcam support
- dashboard for FPS and latency
- side-by-side CPU vs GPU video output
- Nsight profiling screenshots
- GitHub Actions build check, if possible

---

## 20. Roadmap

### Phase 1: Basic CPU Pipeline

- Load image with OpenCV
- Run CPU grayscale
- Run CPU blur
- Run CPU Sobel
- Save output images

### Phase 2: Basic CUDA Pipeline

- Implement CUDA grayscale
- Implement CUDA threshold
- Add CUDA memory allocation and copy
- Save GPU output images

### Phase 3: More CUDA Kernels

- Implement CUDA blur
- Implement CUDA Sobel
- Implement CUDA resize
- Validate CPU vs GPU outputs

### Phase 4: Benchmarking

- Add CPU timers
- Add CUDA event timers
- Save benchmark results to CSV
- Test multiple image sizes

### Phase 5: Video Support

- Read video frames with OpenCV
- Run CPU and GPU pipelines on each frame
- Measure average FPS
- Save processed output video

### Phase 6: Optimization

- Add shared memory version
- Add kernel fusion
- Add CUDA streams
- Compare optimized vs naive kernels

---

## 21. Example Commands

Run single image:

```bash
./cuda_image_pipeline --image ../data/sample_1080p.jpg
```

Run video:

```bash
./cuda_image_pipeline --video ../data/sample_video.mp4
```

Run all benchmarks:

```bash
./cuda_image_pipeline --benchmark ../data/
```

Run only grayscale benchmark:

```bash
./cuda_image_pipeline --op grayscale --image ../data/sample_1080p.jpg
```

Run output comparison:

```bash
./cuda_image_pipeline --validate ../data/sample_1080p.jpg
```

Plot results:

```bash
python3 scripts/plot_benchmarks.py outputs/benchmark_results.csv
```

---

## 22. Example Results Section

After completing the project, replace this section with your real results.

| Operation | Resolution | CPU Time | GPU Time | Speedup |
|---|---:|---:|---:|---:|
| Grayscale | 1920 x 1080 | TBD | TBD | TBD |
| Blur | 1920 x 1080 | TBD | TBD | TBD |
| Sobel | 1920 x 1080 | TBD | TBD | TBD |
| Threshold | 1920 x 1080 | TBD | TBD | TBD |
| Full Pipeline | 1920 x 1080 | TBD | TBD | TBD |

---

## 23. What I Learned

This project helped me understand:

- how image processing maps naturally to GPU parallelism
- how CUDA threads process pixels independently
- how memory transfer affects real-world performance
- why kernel-only benchmarks can be misleading
- how CPU and GPU implementations differ
- how to validate GPU output against CPU output
- how to profile and optimize computer vision preprocessing

---

## 24. Resume Bullets

After completing the project, this project can be summarized on a resume as:

```text
CUDA-Accelerated Image Processing Pipeline — C++, CUDA, OpenCV

- Built custom CUDA kernels for grayscale conversion, blur, Sobel edge detection, thresholding, and image resizing to accelerate computer vision preprocessing.
- Benchmarked CPU OpenCV vs CUDA implementations across 720p and 1080p images, measuring kernel time, end-to-end latency, memory transfer overhead, FPS, and speedup.
- Validated GPU outputs against CPU baselines using pixel-difference metrics and visual comparison, then optimized bottlenecks with shared memory, kernel fusion, and CUDA event profiling.
```

After you have real numbers, make it stronger:

```text
- Accelerated a C++ image processing pipeline with custom CUDA kernels, reducing full-frame 1080p preprocessing latency from X ms to Y ms and improving throughput from X FPS to Y FPS.
```

---

## 25. Future Extension

This project can be extended into a full real-time perception project.

Possible next project:

```text
CUDA-Accelerated Real-Time Lane Detection Pipeline
```

The CUDA kernels from this project can be reused for:

- grayscale conversion
- blur
- Sobel edge detection
- thresholding
- perspective preprocessing
- real-time video acceleration

This turns the project from a GPU fundamentals project into an applied computer vision system.

---

## 26. Final Goal

The final goal is to show that I can take a computer vision workload, identify expensive preprocessing operations, implement GPU-accelerated alternatives, and prove the performance improvement through careful benchmarking.

This project is not only about using CUDA.

It is about understanding the full performance pipeline:

```text
CPU image loading
        |
CPU to GPU memory copy
        |
CUDA kernel execution
        |
GPU to CPU memory copy
        |
output validation
        |
benchmark analysis
```

The most important lesson is that GPU acceleration is only useful when the total system is designed carefully.

A fast kernel does not automatically mean a fast application.

Real performance comes from balancing computation, memory movement, kernel launch overhead, and system-level design.

---

## Suggested Repository Name

```text
cuda-image-processing-pipeline
```

## Suggested GitHub Short Description

```text
CUDA-accelerated computer vision preprocessing pipeline with CPU vs GPU benchmarks for grayscale, blur, Sobel, thresholding, and image resizing.
```
