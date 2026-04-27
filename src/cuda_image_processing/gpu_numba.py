from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

try:
    from numba import cuda
except Exception as exc:  # pragma: no cover - depends on local optional install
    cuda = None
    _NUMBA_IMPORT_ERROR: Exception | None = exc
else:
    _NUMBA_IMPORT_ERROR = None


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA code is requested on a machine without CUDA support."""


@dataclass
class CudaStatus:
    available: bool
    message: str


@dataclass
class CudaPreprocessResult:
    grayscale: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    roi_edges: np.ndarray
    timings_ms: dict[str, float]


def get_cuda_status() -> CudaStatus:
    if _NUMBA_IMPORT_ERROR is not None:
        return CudaStatus(False, f"Numba is not importable: {_NUMBA_IMPORT_ERROR}")
    if cuda is None:
        return CudaStatus(False, "Numba CUDA is not importable.")
    try:
        if not cuda.is_available():
            return CudaStatus(False, "Numba CUDA is installed, but no CUDA-capable NVIDIA GPU is available.")
    except Exception as exc:  # pragma: no cover - depends on local CUDA drivers
        return CudaStatus(False, f"CUDA availability check failed: {exc}")
    return CudaStatus(True, "CUDA is available.")


def require_cuda() -> None:
    status = get_cuda_status()
    if not status.available:
        raise CudaUnavailableError(status.message)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


if cuda is not None:

    @cuda.jit
    def _bgr_to_gray_kernel(frame, gray, width, height):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        gray[y, x] = int(0.114 * b + 0.587 * g + 0.299 * r)

    @cuda.jit
    def _box_blur_3x3_kernel(gray, blurred, width, height):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        total = 0
        count = 0
        for dy in range(-1, 2):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-1, 2):
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                total += gray[yy, xx]
                count += 1

        blurred[y, x] = total // count

    @cuda.jit
    def _sobel_threshold_kernel(blurred, edges, width, height, threshold):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            edges[y, x] = 0
            return

        tl = int(blurred[y - 1, x - 1])
        tc = int(blurred[y - 1, x])
        tr = int(blurred[y - 1, x + 1])
        ml = int(blurred[y, x - 1])
        mr = int(blurred[y, x + 1])
        bl = int(blurred[y + 1, x - 1])
        bc = int(blurred[y + 1, x])
        br = int(blurred[y + 1, x + 1])

        gx = -tl + tr - 2 * ml + 2 * mr - bl + br
        gy = -tl - 2 * tc - tr + bl + 2 * bc + br
        magnitude = abs(gx) + abs(gy)
        edges[y, x] = 255 if magnitude > threshold else 0

    @cuda.jit
    def _roi_mask_kernel(edges, roi_edges, width, height, top_y, left_bottom_x, left_top_x, right_top_x, right_bottom_x):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        if y < top_y:
            roi_edges[y, x] = 0
            return

        denom = (height - 1) - top_y
        ratio = 0.0
        if denom > 0:
            ratio = (y - top_y) / denom

        left_x = left_top_x + ratio * (left_bottom_x - left_top_x)
        right_x = right_top_x + ratio * (right_bottom_x - right_top_x)

        if x >= left_x and x <= right_x:
            roi_edges[y, x] = edges[y, x]
        else:
            roi_edges[y, x] = 0


def _kernel_time_ms(kernel, grid_dims, block_dims, *args) -> float:
    start = cuda.event()
    stop = cuda.event()
    start.record()
    kernel[grid_dims, block_dims](*args)
    stop.record()
    stop.synchronize()
    return cuda.event_elapsed_time(start, stop)


def preprocess_frame_cuda(frame: np.ndarray, sobel_threshold: int = 120) -> CudaPreprocessResult:
    require_cuda()

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("CUDA preprocessing expects a BGR frame with shape height x width x 3.")

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    block_dims = (16, 16)
    grid_dims = ((width + block_dims[0] - 1) // block_dims[0], (height + block_dims[1] - 1) // block_dims[1])

    timings_ms: dict[str, float] = {}

    start_ns = perf_counter_ns()
    d_frame = cuda.to_device(frame)
    d_gray = cuda.device_array((height, width), dtype=np.uint8)
    d_blurred = cuda.device_array((height, width), dtype=np.uint8)
    d_edges = cuda.device_array((height, width), dtype=np.uint8)
    d_roi_edges = cuda.device_array((height, width), dtype=np.uint8)
    cuda.synchronize()
    timings_ms["copy_h2d"] = _elapsed_ms(start_ns)

    timings_ms["grayscale"] = _kernel_time_ms(_bgr_to_gray_kernel, grid_dims, block_dims, d_frame, d_gray, width, height)
    timings_ms["blur"] = _kernel_time_ms(_box_blur_3x3_kernel, grid_dims, block_dims, d_gray, d_blurred, width, height)
    timings_ms["edges"] = _kernel_time_ms(_sobel_threshold_kernel, grid_dims, block_dims, d_blurred, d_edges, width, height, sobel_threshold)

    top_y = int(height * 0.60)
    left_bottom_x = int(width * 0.08)
    left_top_x = int(width * 0.43)
    right_top_x = int(width * 0.57)
    right_bottom_x = int(width * 0.92)
    timings_ms["roi"] = _kernel_time_ms(
        _roi_mask_kernel,
        grid_dims,
        block_dims,
        d_edges,
        d_roi_edges,
        width,
        height,
        top_y,
        left_bottom_x,
        left_top_x,
        right_top_x,
        right_bottom_x,
    )
    timings_ms["gpu_kernel_total"] = timings_ms["grayscale"] + timings_ms["blur"] + timings_ms["edges"] + timings_ms["roi"]

    start_ns = perf_counter_ns()
    grayscale = d_gray.copy_to_host()
    blurred = d_blurred.copy_to_host()
    edges = d_edges.copy_to_host()
    roi_edges = d_roi_edges.copy_to_host()
    cuda.synchronize()
    timings_ms["copy_d2h"] = _elapsed_ms(start_ns)

    timings_ms["gpu_preprocess_total"] = timings_ms["copy_h2d"] + timings_ms["gpu_kernel_total"] + timings_ms["copy_d2h"]

    return CudaPreprocessResult(
        grayscale=grayscale,
        blurred=blurred,
        edges=edges,
        roi_edges=roi_edges,
        timings_ms=timings_ms,
    )

