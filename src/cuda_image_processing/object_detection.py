from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]


class ObjectDetectionUnavailableError(RuntimeError):
    """Raised when YOLO object detection is requested without optional deps."""


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


class TrafficObjectDetector:
    """YOLOv8 traffic-object detector adapted from traffic-monitor."""

    def __init__(self, confidence: float = 0.4, iou: float = 0.5, skip_frames: int = 3):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ObjectDetectionUnavailableError(
                "Traffic object detection requires the optional traffic dependencies. "
                'Install them with: python3 -m pip install -e ".[traffic]"'
            ) from exc

        self.confidence = confidence
        self.iou = iou
        self.skip_frames = max(skip_frames, 1)
        self.frame_count = 0
        self.last_detections: list[Detection] = []
        self.tracks: list[dict[str, object]] = []
        self.next_track_id = 1
        self.track_iou_threshold = 0.30
        self.max_track_age = 4
        self.box_smooth_alpha = 0.35
        self.model = YOLO("yolov8n.pt")

        self.colors = {
            "stop sign": (0, 0, 255),
            "traffic light": (0, 255, 255),
            "car": (255, 200, 0),
            "truck": (255, 150, 0),
            "bus": (255, 100, 0),
            "person": (0, 255, 0),
            "bicycle": (0, 200, 100),
            "motorcycle": (200, 100, 0),
        }
        self.default_color = (0, 255, 0)

    def _bbox_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return float(inter_area / max(union, 1))

    def _tracks_to_detections(self) -> list[Detection]:
        detections: list[Detection] = []
        for track in self.tracks:
            x1, y1, x2, y2 = np.asarray(track["bbox"]).astype(int)
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(track["confidence"]),
                    class_id=int(track["cls_id"]),
                    class_name=str(track["class_name"]),
                    track_id=int(track["id"]),
                )
            )
        return detections

    def _update_tracks(self, detections: list[Detection]) -> None:
        matched_track_ids: set[int] = set()
        updated_tracks: list[dict[str, object]] = []

        for det in detections:
            det_box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
            best_idx = None
            best_iou = 0.0
            for idx, track in enumerate(self.tracks):
                if idx in matched_track_ids or int(track["cls_id"]) != det.class_id:
                    continue
                iou = self._bbox_iou(det_box, np.asarray(track["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= self.track_iou_threshold:
                track = self.tracks[best_idx]
                track["bbox"] = (1.0 - self.box_smooth_alpha) * np.asarray(track["bbox"]) + self.box_smooth_alpha * det_box
                track["confidence"] = max(det.confidence, 0.7 * float(track["confidence"]) + 0.3 * det.confidence)
                track["age"] = 0
                track["hits"] = int(track["hits"]) + 1
                updated_tracks.append(track)
                matched_track_ids.add(best_idx)
            else:
                updated_tracks.append(
                    {
                        "id": self.next_track_id,
                        "bbox": det_box,
                        "confidence": det.confidence,
                        "cls_id": det.class_id,
                        "class_name": det.class_name,
                        "age": 0,
                        "hits": 1,
                    }
                )
                self.next_track_id += 1

        for idx, track in enumerate(self.tracks):
            if idx in matched_track_ids:
                continue
            track["age"] = int(track["age"]) + 1
            if int(track["age"]) <= self.max_track_age:
                updated_tracks.append(track)

        self.tracks = updated_tracks
        self.last_detections = self._tracks_to_detections()

    def process(self, frame: np.ndarray) -> list[Detection]:
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return self.last_detections

        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
            scale = 1.0

        results = self.model(small, conf=self.confidence, iou=self.iou, classes=TRAFFIC_CLASSES, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                detections.append(
                    Detection(
                        x1=int(x1 / scale),
                        y1=int(y1 / scale),
                        x2=int(x2 / scale),
                        y2=int(y2 / scale),
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        class_name=self.model.names[class_id],
                    )
                )

        self._update_tracks(detections)
        return self.last_detections

    def draw(self, frame: np.ndarray, detections: list[Detection], thickness: int = 2) -> np.ndarray:
        output = frame.copy()
        for detection in detections:
            color = self.colors.get(detection.class_name, self.default_color)
            cv2.rectangle(output, (detection.x1, detection.y1), (detection.x2, detection.y2), color, thickness)
            suffix = f" #{detection.track_id}" if detection.track_id is not None else ""
            label = f"{detection.class_name}{suffix} {detection.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (detection.x1, detection.y1 - th - 8), (detection.x1 + tw, detection.y1), color, -1)
            cv2.putText(output, label, (detection.x1, detection.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return output

