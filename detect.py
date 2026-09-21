"""Object Detector: YOLO + OpenCV.

Examples:
    python detect.py --source samples/street.jpg
    python detect.py --source samples/traffic.mp4
    python detect.py --source 0            # webcam
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [
    (46, 117, 182), (232, 93, 4), (56, 176, 0), (214, 40, 40),
    (142, 68, 173), (0, 150, 199), (247, 184, 1), (0, 109, 119),
]


def draw_detections(frame, result):
    """Draw bounding boxes with class names and confidence on the frame."""
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        color = PALETTE[class_id % len(PALETTE)]
        label = f"{names[class_id]} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + text_w + 6, top + text_h + baseline + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + text_h + 1), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def detect_image(model, source, conf, output_dir, show):
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    result = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = draw_detections(frame, result)

    output_path = output_dir / f"{source.stem}_detected{source.suffix}"
    cv2.imwrite(str(output_path), annotated)
    print(f"Objects found: {len(result.boxes)}. Saved to {output_path}")

    if show:
        cv2.imshow("Object Detector (press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_video(model, source, conf, output_dir, show):
    is_webcam = isinstance(source, int)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if not is_webcam:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_dir / f"{Path(source).stem}_detected.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = draw_detections(frame, result)
            frames += 1

            if writer is not None:
                writer.write(annotated)
            if show or is_webcam:
                cv2.imshow("Object Detector (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Processed {frames} frames. Saved to {output_path}")
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection with YOLO and OpenCV")
    parser.add_argument("--source", required=True, help="image path, video path, or webcam index (0)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (downloaded automatically)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (0-1)")
    parser.add_argument("--output", default="output", help="folder for results")
    parser.add_argument("--show", action="store_true", help="show a window with the result")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    if args.source.isdigit():
        detect_video(model, int(args.source), args.conf, output_dir, args.show)
        return

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if source.suffix.lower() in IMAGE_EXTENSIONS:
        detect_image(model, source, args.conf, output_dir, args.show)
    else:
        detect_video(model, str(source), args.conf, output_dir, args.show)


if __name__ == "__main__":
    main()