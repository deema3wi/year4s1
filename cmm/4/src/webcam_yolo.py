import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt", help="Model path or name, e.g. yolov8n.pt")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (0 is default)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--device", default="cpu", help="cpu or 0,1,... for GPU")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width")
    p.add_argument("--height", type=int, default=720, help="Requested capture height")
    p.add_argument("--mirror", action="store_true", help="Mirror image horizontally")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    # Try to set capture resolution (may be ignored by some cameras/drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "YOLO Webcam (press Q or ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # YOLO inference on current frame
            # Ultralytics returns a list of Results; plot() draws boxes + labels on the image
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

            annotated = results[0].plot()  # BGR image with rectangles + labels

            # Optional FPS overlay
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-9)
            prev_t = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):  # ESC or Q
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
