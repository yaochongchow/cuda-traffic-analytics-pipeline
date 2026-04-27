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
    lane_stats: np.ndarray | None = None


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

    @cuda.jit
    def _lane_stats_kernel(roi_edges, stats, width, height, center_x):
        x, y = cuda.grid(2)
        if x >= width or y >= height or roi_edges[y, x] == 0:
            return

        side = 0 if x < center_x else 1
        xf = float(x)
        yf = float(y)

        cuda.atomic.add(stats, (side, 0), 1.0)
        cuda.atomic.add(stats, (side, 1), yf)
        cuda.atomic.add(stats, (side, 2), xf)
        cuda.atomic.add(stats, (side, 3), yf * yf)
        cuda.atomic.add(stats, (side, 4), xf * yf)


def _kernel_time_ms(kernel, grid_dims, block_dims, *args) -> float:
    start = cuda.event()
    stop = cuda.event()
    start.record()
    kernel[grid_dims, block_dims](*args)
    stop.record()
    stop.synchronize()
    return cuda.event_elapsed_time(start, stop)


class CudaFramePreprocessor:
    """Reusable CUDA buffers for fixed-size video frames."""

    def __init__(self, frame_shape: tuple[int, int, int], sobel_threshold: int = 625):
        require_cuda()
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError("CUDA preprocessing expects a BGR frame with shape height x width x 3.")

        self.height, self.width = frame_shape[:2]
        self.frame_shape = frame_shape
        self.sobel_threshold = sobel_threshold
        self.block_dims = (16, 16)
        self.grid_dims = (
            (self.width + self.block_dims[0] - 1) // self.block_dims[0],
            (self.height + self.block_dims[1] - 1) // self.block_dims[1],
        )
        self.empty_image = np.empty((0, 0), dtype=np.uint8)
        self.zero_lane_stats = np.zeros((2, 5), dtype=np.float64)

        self.d_frame = cuda.device_array(frame_shape, dtype=np.uint8)
        self.d_gray = cuda.device_array((self.height, self.width), dtype=np.uint8)
        self.d_blurred = cuda.device_array((self.height, self.width), dtype=np.uint8)
        self.d_edges = cuda.device_array((self.height, self.width), dtype=np.uint8)
        self.d_roi_edges = cuda.device_array((self.height, self.width), dtype=np.uint8)
        self.d_lane_stats = cuda.device_array((2, 5), dtype=np.float64)

    def process(
        self,
        frame: np.ndarray,
        copy_intermediates: bool = True,
        copy_roi_edges: bool = True,
        compute_lane_stats: bool = False,
    ) -> CudaPreprocessResult:
        if frame.shape != self.frame_shape:
            raise ValueError(f"Expected frame shape {self.frame_shape}, got {frame.shape}.")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        timings_ms: dict[str, float] = {}

        start_ns = perf_counter_ns()
        self.d_frame.copy_to_device(frame)
        if compute_lane_stats:
            self.d_lane_stats.copy_to_device(self.zero_lane_stats)
        cuda.synchronize()
        timings_ms["copy_h2d"] = _elapsed_ms(start_ns)

        timings_ms["grayscale"] = _kernel_time_ms(
            _bgr_to_gray_kernel,
            self.grid_dims,
            self.block_dims,
            self.d_frame,
            self.d_gray,
            self.width,
            self.height,
        )
        timings_ms["blur"] = _kernel_time_ms(
            _box_blur_3x3_kernel,
            self.grid_dims,
            self.block_dims,
            self.d_gray,
            self.d_blurred,
            self.width,
            self.height,
        )
        timings_ms["edges"] = _kernel_time_ms(
            _sobel_threshold_kernel,
            self.grid_dims,
            self.block_dims,
            self.d_blurred,
            self.d_edges,
            self.width,
            self.height,
            self.sobel_threshold,
        )

        top_y = int(self.height * 0.60)
        left_bottom_x = int(self.width * 0.08)
        left_top_x = int(self.width * 0.43)
        right_top_x = int(self.width * 0.57)
        right_bottom_x = int(self.width * 0.92)
        timings_ms["roi"] = _kernel_time_ms(
            _roi_mask_kernel,
            self.grid_dims,
            self.block_dims,
            self.d_edges,
            self.d_roi_edges,
            self.width,
            self.height,
            top_y,
            left_bottom_x,
            left_top_x,
            right_top_x,
            right_bottom_x,
        )

        if compute_lane_stats:
            timings_ms["lane_stats"] = _kernel_time_ms(
                _lane_stats_kernel,
                self.grid_dims,
                self.block_dims,
                self.d_roi_edges,
                self.d_lane_stats,
                self.width,
                self.height,
                self.width // 2,
            )

        timings_ms["gpu_kernel_total"] = (
            timings_ms["grayscale"]
            + timings_ms["blur"]
            + timings_ms["edges"]
            + timings_ms["roi"]
            + timings_ms.get("lane_stats", 0.0)
        )

        start_ns = perf_counter_ns()
        if copy_intermediates:
            grayscale = self.d_gray.copy_to_host()
            blurred = self.d_blurred.copy_to_host()
            edges = self.d_edges.copy_to_host()
        else:
            grayscale = self.empty_image
            blurred = self.empty_image
            edges = self.empty_image

        roi_edges = self.d_roi_edges.copy_to_host() if copy_roi_edges else self.empty_image
        lane_stats = self.d_lane_stats.copy_to_host() if compute_lane_stats else None
        cuda.synchronize()
        timings_ms["copy_d2h"] = _elapsed_ms(start_ns)
        timings_ms["gpu_preprocess_total"] = (
            timings_ms["copy_h2d"] + timings_ms["gpu_kernel_total"] + timings_ms["copy_d2h"]
        )

        return CudaPreprocessResult(
            grayscale=grayscale,
            blurred=blurred,
            edges=edges,
            roi_edges=roi_edges,
            timings_ms=timings_ms,
            lane_stats=lane_stats,
        )


def preprocess_frame_cuda(
    frame: np.ndarray,
    sobel_threshold: int = 625,
    copy_intermediates: bool = True,
    copy_roi_edges: bool = True,
    compute_lane_stats: bool = False,
) -> CudaPreprocessResult:
    preprocessor = CudaFramePreprocessor(frame.shape, sobel_threshold=sobel_threshold)
    return preprocessor.process(
        frame,
        copy_intermediates=copy_intermediates,
        copy_roi_edges=copy_roi_edges,
        compute_lane_stats=compute_lane_stats,
    )

