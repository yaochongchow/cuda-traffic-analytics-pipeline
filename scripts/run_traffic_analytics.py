from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.gpu_numba import CudaUnavailableError
from cuda_image_processing.gpu_pipeline import explain_cuda_unavailable
from cuda_image_processing.io_utils import OUTPUTS_DIR, build_video_writer, ensure_dir
from cuda_image_processing.object_detection import ObjectDetectionUnavailableError
from cuda_image_processing.realtime_pipeline import TrafficAnalyticsPipeline


def _open_capture(video_path: Path | None) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path) if video_path else 0)
    if not capture.isOpened():
        source = str(video_path) if video_path else "webcam"
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(description="Run combined traffic analytics from this project and traffic-monitor.")
    parser.add_argument("--video", type=Path, help="Input video path. Defaults to webcam if omitted.")
    parser.add_argument("--output", type=Path, help="Optional output video path.")
    parser.add_argument("--loop", action="store_true", help="Loop input video continuously.")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional frame cap for scripted runs.")
    parser.add_argument("--lane-mode", choices=("baseline", "advanced"), default="advanced", help="Lane detector to use.")
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu", help="Execution mode for lane preprocessing.")
    parser.add_argument("--no-lanes", action="store_true", help="Disable lane detection.")
    parser.add_argument("--objects", action="store_true", help="Enable YOLOv8 traffic object detection.")
    parser.add_argument("--perspective", action="store_true", help="Apply bird's-eye perspective display.")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--object-skip-frames", type=int, default=3, help="Run object detection every N frames.")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None and not args.display:
        output_dir = ensure_dir(OUTPUTS_DIR / "traffic")
        stem = args.video.stem if args.video else "webcam"
        output_path = output_dir / f"{stem}_{args.mode}_{args.lane_mode}_traffic.mp4"

    try:
        pipeline = TrafficAnalyticsPipeline(
            lane_mode=args.lane_mode,
            execution_mode=args.mode,
            enable_lanes=not args.no_lanes,
            enable_objects=args.objects,
            enable_perspective=args.perspective,
            confidence=args.conf,
            object_skip_frames=args.object_skip_frames,
        )
    except ObjectDetectionUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    capture = _open_capture(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    writer = build_video_writer(output_path, (width, height), fps) if output_path else None

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop and args.video:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            try:
                result = pipeline.process(frame)
            except CudaUnavailableError as exc:
                raise SystemExit(f"{exc}\n\n{explain_cuda_unavailable()}") from exc

            if writer is not None:
                writer.write(result.frame)
            if args.display:
                cv2.imshow("Traffic Analytics", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.limit_frames is not None and frame_count >= args.limit_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if output_path:
        print(f"Saved traffic analytics video to {output_path}")
    print(f"Frames processed: {frame_count}")


if __name__ == "__main__":
    main()

