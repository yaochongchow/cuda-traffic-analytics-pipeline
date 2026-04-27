from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .lane_detection import run_lane_detection
from .perspective import PerspectiveTransform


class AdvancedLaneDetector:
    """Stateful lane detector adapted from traffic-monitor.

    This detector is better suited to real driving video than the simple
    Hough baseline because it uses color masks, perspective warp, RANSAC
    fitting, and temporal smoothing.
    """

    def __init__(self, buffer_size: int = 10, calibration_frames: int = 30, use_fallback: bool = True):
        self.calibration_frames = calibration_frames
        self.calib_count = 0
        self.calibrated = False
        self.road_values: deque[np.ndarray] = deque(maxlen=calibration_frames)
        self.road_hsv_mean: np.ndarray | None = None
        self.use_fallback = use_fallback

        self.search_margin = 55
        self.nwindows = 8
        self.minpix = 35
        self.min_lane_px = 45
        self.ransac_iters = 40
        self.ransac_thresh = 14.0
        self.fit_alpha = 0.28
        self.max_missed = 10
        self.rng = np.random.default_rng(42)
        self.line_top_ratio = 0.32
        self.hough_threshold = 18
        self.hough_min_len_ratio = 0.18
        self.hough_gap_ratio = 0.10

        self.left_range = (0.18, 0.42)
        self.right_range = (0.56, 0.82)
        self.left_top_range = (0.32, 0.62)
        self.right_top_range = (0.42, 0.74)
        self.pair_bottom = (0.20, 0.56)
        self.pair_top = (0.06, 0.34)

        self.left_fit: np.ndarray | None = None
        self.right_fit: np.ndarray | None = None
        self.left_missed = 0
        self.right_missed = 0
        self.left_line: tuple[int, int, int, int] | None = None
        self.right_line: tuple[int, int, int, int] | None = None
        self.left_fit_q: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.right_fit_q: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.width_bot_q: deque[float] = deque(maxlen=buffer_size)
        self.width_top_q: deque[float] = deque(maxlen=buffer_size)

        self.perspective = PerspectiveTransform(
            src_ratios=np.float32([[0.34, 0.16], [0.78, 0.16], [0.96, 0.98], [0.18, 0.98]]),
            dst_ratios=np.float32([[0.34, 0.02], [0.68, 0.02], [0.68, 0.98], [0.34, 0.98]]),
        )

    def _calibrate(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.82) : int(h * 0.95), int(w * 0.35) : int(w * 0.65)]
        if roi.size:
            self.road_values.append(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), axis=(0, 1)))
        self.calib_count += 1
        if self.calib_count >= self.calibration_frames:
            if self.road_values:
                self.road_hsv_mean = np.mean(self.road_values, axis=0)
            self.calibrated = True

    def _roi_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        points = np.array(
            [[[int(w * 0.16), h - 1], [int(w * 0.38), int(h * 0.18)], [int(w * 0.80), int(h * 0.18)], [int(w * 0.98), h - 1]]],
            np.int32,
        )
        cv2.fillPoly(mask, points, 255)
        return mask

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(2.0, (8, 8)).apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

        hls = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        roi = self._roi_mask(image.shape)

        white_l_min = max(int(self.road_hsv_mean[2] + 35), 175) if self.road_hsv_mean is not None else 185
        white = cv2.inRange(hls, np.array([0, white_l_min, 0]), np.array([255, 255, 120]))
        yellow = cv2.inRange(hsv, np.array([12, 70, 90]), np.array([42, 255, 255]))

        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        scaled = np.uint8(255 * sobel_x / max(np.max(sobel_x), 1))
        gradient = cv2.inRange(scaled, 30, 255)

        close_kernel = np.ones((5, 5), np.uint8)
        open_kernel = np.ones((3, 3), np.uint8)

        def refine(mask: np.ndarray) -> np.ndarray:
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
            return cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

        yellow_gradient = cv2.bitwise_and(gradient, cv2.inRange(hsv, np.array([10, 30, 50]), np.array([50, 255, 255])))
        white_gradient = cv2.bitwise_and(gradient, cv2.inRange(hls, np.array([0, 140, 0]), np.array([255, 255, 150])))
        left_mask = cv2.bitwise_and(refine(cv2.bitwise_or(yellow, yellow_gradient)), roi)
        right_mask = cv2.bitwise_and(refine(cv2.bitwise_or(white, white_gradient)), roi)

        h, w = image.shape[:2]
        cut = int(h * 0.08)
        left_mask[:cut] = 0
        left_mask[:, : int(w * 0.18)] = 0
        left_mask[:, int(w * 0.54) :] = 0
        right_mask[:cut] = 0
        right_mask[:, : int(w * 0.50)] = 0
        right_mask[:, int(w * 0.72) :] = 0
        return left_mask, right_mask

    def _search_prior(self, warped: np.ndarray, fit: np.ndarray | None) -> np.ndarray:
        if fit is None:
            return np.empty((0, 2), np.float32)
        nonzero_y, nonzero_x = warped.nonzero()
        if not len(nonzero_x):
            return np.empty((0, 2), np.float32)
        mask = np.abs(nonzero_x - (fit[0] * nonzero_y + fit[1])) <= self.search_margin
        return np.column_stack((nonzero_x[mask], nonzero_y[mask])).astype(np.float32)

    def _sliding_window(self, warped: np.ndarray, x_range: tuple[float, float]) -> np.ndarray:
        hist = np.sum(warped[warped.shape[0] // 2 :], axis=0)
        start = int(len(hist) * x_range[0])
        end = int(len(hist) * x_range[1])
        if end <= start or hist[start:end].max() <= 0:
            return np.empty((0, 2), np.float32)

        center_x = int(np.argmax(hist[start:end]) + start)
        nonzero_y, nonzero_x = warped.nonzero()
        window_height = max(warped.shape[0] // self.nwindows, 1)
        indices = []

        for window_idx in range(self.nwindows):
            y_low = warped.shape[0] - (window_idx + 1) * window_height
            y_high = warped.shape[0] - window_idx * window_height
            window_indices = np.where(
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= center_x - self.search_margin)
                & (nonzero_x < center_x + self.search_margin)
            )[0]
            indices.append(window_indices)
            if len(window_indices) > self.minpix:
                center_x = int(np.mean(nonzero_x[window_indices]))

        merged = np.concatenate(indices) if indices else np.array([], np.int32)
        return np.column_stack((nonzero_x[merged], nonzero_y[merged])).astype(np.float32)

    def _fit_endpoints(self, x_bottom: float, x_top: float, y_bottom: float, y_top: float) -> np.ndarray | None:
        if abs(y_bottom - y_top) < 1:
            return None
        slope = (x_bottom - x_top) / (y_bottom - y_top)
        return np.array([slope, x_bottom - slope * y_bottom], np.float64)

    def _has_support(self, points: np.ndarray | None, height: int) -> bool:
        if points is None or len(points) < self.min_lane_px:
            return False
        y_values = points[:, 1]
        return (y_values.max() - y_values.min()) / max(height, 1) >= 0.30 and np.count_nonzero(y_values >= height * 0.72) >= 18

    def _ransac_fit(self, points: np.ndarray, shape: tuple[int, ...], side: str) -> np.ndarray | None:
        h, w = shape[:2]
        if not self._has_support(points, h):
            return None

        points = np.asarray(points, np.float64)
        x_values = points[:, 0]
        y_values = points[:, 1]
        best_mask = None
        best_score = -1
        threshold = max(self.ransac_thresh, w * 0.015)

        for _ in range(self.ransac_iters):
            i, j = self.rng.choice(len(points), 2, replace=False)
            if abs(y_values[i] - y_values[j]) < 6:
                continue
            coeffs = self._fit_endpoints(x_values[i], x_values[j], y_values[i], y_values[j])
            if coeffs is None:
                continue
            inliers = np.abs(x_values - (coeffs[0] * y_values + coeffs[1])) <= threshold
            if inliers.sum() < self.min_lane_px:
                continue
            score = np.sum(1 + y_values[inliers] / max(y_values.max(), 1))
            if score > best_score:
                best_score = score
                best_mask = inliers

        if best_mask is None:
            best_mask = np.ones(len(points), bool)

        inlier_x = x_values[best_mask]
        inlier_y = y_values[best_mask]
        weights = 1 + (inlier_y - inlier_y.min()) / max(np.ptp(inlier_y), 1)
        slope, intercept = np.polyfit(inlier_y, inlier_x, 1, w=weights)

        if abs(slope) > 0.22:
            return None
        bottom_x = slope * inlier_y.max() + intercept
        if side == "left" and (bottom_x > w * 0.5 or bottom_x < w * self.left_range[0] * 0.85):
            return None
        if side == "right" and (bottom_x < w * 0.5 or bottom_x > w * min(self.right_range[1] * 1.1, 0.98)):
            return None
        return np.array([slope, intercept], np.float64)

    def _smooth(self, fit: np.ndarray | None, queue: deque[np.ndarray]) -> np.ndarray | None:
        if fit is None:
            return None
        if not queue:
            queue.append(fit)
            return fit
        smoothed = (1 - self.fit_alpha) * queue[-1] + self.fit_alpha * fit
        queue.append(smoothed)
        return smoothed

    def _track(
        self,
        current: np.ndarray | None,
        previous: np.ndarray | None,
        queue: deque[np.ndarray],
        missed: int,
    ) -> tuple[np.ndarray | None, int]:
        if current is not None:
            return self._smooth(current, queue), 0
        if previous is None or missed >= self.max_missed:
            return None, missed + 1
        queue.append(previous)
        return previous, missed + 1

    def _project(
        self,
        fit: np.ndarray | None,
        y_bottom: int,
        y_top: int,
        road_shape: tuple[int, ...],
        crop_top: int,
    ) -> tuple[int, int, int, int] | None:
        if fit is None:
            return None
        bird = np.array([[fit[0] * y_bottom + fit[1], y_bottom, fit[0] * y_top + fit[1], y_top]], np.float32)
        _, inverse = self.perspective.get_matrices(road_shape)
        points = cv2.perspectiveTransform(bird.reshape(1, 2, 2), inverse)[0]
        return (
            int(points[0, 0]),
            int(points[0, 1] + crop_top),
            int(points[1, 0]),
            int(points[1, 1] + crop_top),
        )

    def _in_band(self, line: tuple[int, int, int, int] | None, width: int, side: str) -> bool:
        if line is None:
            return False
        bottom_x = line[0] / width
        top_x = line[2] / width
        if side == "left":
            return self.left_range[0] <= bottom_x <= self.left_range[1] and self.left_top_range[0] <= top_x <= self.left_top_range[1]
        return self.right_range[0] <= bottom_x <= self.right_range[1] and self.right_top_range[0] <= top_x <= self.right_top_range[1]

    def _plausible(self, left_line: tuple[int, int, int, int] | None, right_line: tuple[int, int, int, int] | None, width: int) -> bool:
        if left_line is None or right_line is None:
            return False
        bottom_width = (right_line[0] - left_line[0]) / width
        top_width = (right_line[2] - left_line[2]) / width
        return bottom_width > 0 and top_width > 0 and self.pair_bottom[0] <= bottom_width <= self.pair_bottom[1] and self.pair_top[0] <= top_width <= self.pair_top[1]

    def _infer_missing(
        self,
        left_fit: np.ndarray | None,
        right_fit: np.ndarray | None,
        y_bottom: int,
        y_top: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.width_bot_q:
            return left_fit, right_fit
        bottom_width = float(np.mean(self.width_bot_q))
        top_width = float(np.mean(self.width_top_q))
        if left_fit is None and right_fit is not None:
            right_bottom = right_fit[0] * y_bottom + right_fit[1]
            right_top = right_fit[0] * y_top + right_fit[1]
            left_fit = self._fit_endpoints(right_bottom - bottom_width, right_top - top_width, y_bottom, y_top)
        elif right_fit is None and left_fit is not None:
            left_bottom = left_fit[0] * y_bottom + left_fit[1]
            left_top = left_fit[0] * y_top + left_fit[1]
            right_fit = self._fit_endpoints(left_bottom + bottom_width, left_top + top_width, y_bottom, y_top)
        return left_fit, right_fit

    def _fit_line_from_slope_intercept(self, slope: float, intercept: float, height: int, crop_top: int) -> tuple[int, int, int, int]:
        y_bottom = height - 1
        y_top = int(height * self.line_top_ratio)
        return (
            int(slope * y_bottom + intercept),
            int(y_bottom + crop_top),
            int(slope * y_top + intercept),
            int(y_top + crop_top),
        )

    def _hough_line(self, mask: np.ndarray, crop_top: int, frame_width: int, side: str) -> tuple[int, int, int, int] | None:
        h, _ = mask.shape[:2]
        lines = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=max(int(h * self.hough_min_len_ratio), 25),
            maxLineGap=max(int(h * self.hough_gap_ratio), 12),
        )
        if lines is None:
            return None

        y_bottom = h - 1
        y_top = int(h * self.line_top_ratio)
        candidates = []
        for segment in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, segment)
            if y2 > y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if abs(y1 - y2) < 12:
                continue

            slope = (x1 - x2) / (y1 - y2)
            intercept = x1 - slope * y1
            bottom_x = slope * y_bottom + intercept
            top_x = slope * y_top + intercept
            bottom_ratio = bottom_x / frame_width
            top_ratio = top_x / frame_width

            if side == "left":
                valid = -3.5 < slope < -0.35 and self.left_range[0] <= bottom_ratio <= self.left_range[1] and self.left_top_range[0] <= top_ratio <= self.left_top_range[1]
                position_bonus = bottom_ratio
            else:
                valid = 0.35 < slope < 4.5 and self.right_range[0] <= bottom_ratio <= self.right_range[1] and self.right_top_range[0] <= top_ratio <= self.right_top_range[1]
                position_bonus = 1.0 - bottom_ratio

            if not valid:
                continue

            length = float(np.hypot(x2 - x1, y2 - y1))
            candidates.append((length * (1.0 + 0.7 * position_bonus), slope, intercept))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[: min(8, len(candidates))]
        score_sum = sum(item[0] for item in top)
        slope = sum(item[1] * item[0] for item in top) / score_sum
        intercept = sum(item[2] * item[0] for item in top) / score_sum
        line = self._fit_line_from_slope_intercept(slope, intercept, h, crop_top)
        return line if self._in_band(line, frame_width, side) else None

    def _refine_line(
        self,
        mask: np.ndarray,
        line: tuple[int, int, int, int] | None,
        crop_top: int,
        width: int,
        side: str,
    ) -> tuple[int, int, int, int] | None:
        nonzero_y, nonzero_x = mask.nonzero()
        if not len(nonzero_x):
            return None

        if line is not None:
            x_bottom, y_bottom, x_top, y_top = line
            y_bottom -= crop_top
            y_top -= crop_top
            if abs(y_bottom - y_top) < 1:
                return None
            slope = (x_bottom - x_top) / (y_bottom - y_top)
            intercept = x_bottom - slope * y_bottom
            mask_near_line = np.abs(nonzero_x - (slope * nonzero_y + intercept)) <= 42
        else:
            x_range = self.left_range if side == "left" else self.right_range
            mask_near_line = (nonzero_x >= width * x_range[0]) & (nonzero_x <= width * x_range[1])

        points = np.column_stack((nonzero_x[mask_near_line], nonzero_y[mask_near_line])).astype(np.float64)
        if not self._has_support(points, mask.shape[0]):
            return None

        bands = np.linspace(0, mask.shape[0], 22, dtype=np.int32)
        selected = []
        for idx in range(len(bands) - 1):
            band = points[(points[:, 1] >= bands[idx]) & (points[:, 1] < bands[idx + 1])]
            if len(band) < 6:
                continue
            percentile = 94 if side == "left" else 6
            selected.append([np.percentile(band[:, 0], percentile), np.median(band[:, 1])])
        if len(selected) < 8:
            return None

        points = np.array(selected, np.float64)
        x_values = points[:, 0]
        y_values = points[:, 1]
        best_mask = None
        best_score = -1

        for _ in range(self.ransac_iters):
            i, j = self.rng.choice(len(points), 2, replace=False)
            if abs(y_values[i] - y_values[j]) < 12:
                continue
            coeffs = self._fit_endpoints(x_values[i], x_values[j], y_values[i], y_values[j])
            if coeffs is None:
                continue
            if side == "left" and coeffs[0] >= -0.03:
                continue
            if side == "right" and coeffs[0] <= 0.03:
                continue
            inliers = np.abs(x_values - (coeffs[0] * y_values + coeffs[1])) <= 12
            if inliers.sum() < self.min_lane_px:
                continue
            score = np.sum(1 + y_values[inliers] / max(y_values.max(), 1))
            if score > best_score:
                best_score = score
                best_mask = inliers

        if best_mask is None:
            best_mask = np.ones(len(points), bool)

        inlier_x = x_values[best_mask]
        inlier_y = y_values[best_mask]
        weights = 1 + (inlier_y - inlier_y.min()) / max(np.ptp(inlier_y), 1)
        slope, intercept = np.polyfit(inlier_y, inlier_x, 1, w=weights)
        if side == "left" and slope >= -0.03:
            return None
        if side == "right" and slope <= 0.03:
            return None
        y_bottom = mask.shape[0] - 1
        y_top = int(mask.shape[0] * self.line_top_ratio)
        line = (
            int(slope * y_bottom + intercept),
            int(y_bottom + crop_top),
            int(slope * y_top + intercept),
            int(y_top + crop_top),
        )
        return line if self._in_band(line, width, side) else None

    def _fallback_lines(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if not self.use_fallback:
            return None, None
        result = run_lane_detection(frame)
        if len(result.lanes) < 2:
            return None, None
        left, right = sorted(result.lanes[:2], key=lambda line: line[0])
        return left, right

    def process(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        h, w = frame.shape[:2]
        if not self.calibrated:
            self._calibrate(frame)

        crop_top = int(h * 0.30)
        road = frame[crop_top:]
        road_height = road.shape[0]
        if road_height < 40:
            return self._fallback_lines(frame)

        left_mask, right_mask = self._preprocess(road)
        hough_left = self._hough_line(left_mask, crop_top, w, "left")
        hough_right = self._hough_line(right_mask, crop_top, w, "right")

        if self._in_band(hough_left, w, "left") and self._in_band(hough_right, w, "right") and self._plausible(hough_left, hough_right, w):
            self.left_line, self.right_line = hough_left, hough_right
            return self.left_line, self.right_line

        direct_left = self._refine_line(left_mask, None, crop_top, w, "left")
        direct_right = self._refine_line(right_mask, None, crop_top, w, "right")
        if self._in_band(direct_left, w, "left") and self._in_band(direct_right, w, "right") and self._plausible(direct_left, direct_right, w):
            self.left_line, self.right_line = direct_left, direct_right
            return self.left_line, self.right_line

        left_warped = self.perspective.warp(left_mask, cv2.INTER_NEAREST)
        right_warped = self.perspective.warp(right_mask, cv2.INTER_NEAREST)

        left_points = self._search_prior(left_warped, self.left_fit)
        right_points = self._search_prior(right_warped, self.right_fit)
        if len(left_points) < self.min_lane_px:
            left_points = self._sliding_window(left_warped, self.left_range)
        if len(right_points) < self.min_lane_px:
            right_points = self._sliding_window(right_warped, self.right_range)

        left_current = self._ransac_fit(left_points, left_warped.shape, "left")
        right_current = self._ransac_fit(right_points, right_warped.shape, "right")

        self.left_fit, self.left_missed = self._track(left_current, self.left_fit, self.left_fit_q, self.left_missed)
        self.right_fit, self.right_missed = self._track(right_current, self.right_fit, self.right_fit_q, self.right_missed)

        y_bottom = left_warped.shape[0] - 1
        y_top = int(left_warped.shape[0] * self.line_top_ratio)
        left_fit, right_fit = self._infer_missing(self.left_fit, self.right_fit, y_bottom, y_top)
        left_line = self._project(left_fit, y_bottom, y_top, road.shape, crop_top)
        right_line = self._project(right_fit, y_bottom, y_top, road.shape, crop_top)

        if self._in_band(left_line, w, "left") and self._in_band(right_line, w, "right") and self._plausible(left_line, right_line, w):
            if left_fit is not None and right_fit is not None:
                bottom_width = (right_fit[0] * y_bottom + right_fit[1]) - (left_fit[0] * y_bottom + left_fit[1])
                top_width = (right_fit[0] * y_top + right_fit[1]) - (left_fit[0] * y_top + left_fit[1])
                if bottom_width > 0 and top_width > 0:
                    self.width_bot_q.append(float(bottom_width))
                    self.width_top_q.append(float(top_width))
            self.left_line, self.right_line = left_line, right_line

        refined_left = self._refine_line(left_mask, self.left_line, crop_top, w, "left") or hough_left
        refined_right = self._refine_line(right_mask, self.right_line, crop_top, w, "right") or hough_right
        candidate_left = refined_left or self.left_line
        candidate_right = refined_right or self.right_line

        if self._in_band(candidate_left, w, "left") and self._in_band(candidate_right, w, "right") and self._plausible(candidate_left, candidate_right, w):
            self.left_line, self.right_line = candidate_left, candidate_right

        if self.left_line is None or self.right_line is None:
            return self._fallback_lines(frame)
        return self.left_line, self.right_line

    def draw(
        self,
        frame: np.ndarray,
        left_line: tuple[int, int, int, int] | None,
        right_line: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        output = frame.copy()
        if left_line and right_line:
            points = np.array(
                [
                    [left_line[0], left_line[1]],
                    [left_line[2], left_line[3]],
                    [right_line[2], right_line[3]],
                    [right_line[0], right_line[1]],
                ],
                np.int32,
            )
            overlay = output.copy()
            cv2.fillPoly(overlay, [points], (0, 120, 0))
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

        for line in (left_line, right_line):
            if line:
                cv2.line(output, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 4)

        if not self.calibrated:
            progress = int(self.calib_count / max(self.calibration_frames, 1) * 100)
            cv2.putText(output, f"Calibrating... {progress}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return output

