"""Draw shelf zones on a camera view or video frame and save them to zones.json.

Examples:
    python3 define_zones.py --source store.mp4
    python3 define_zones.py --source 0            # webcam: press SPACE to freeze a frame

Mouse: left-click adds a polygon point.
Keys:  n / Enter = close current polygon   u = undo last point
       c = clear everything                s = save & quit      q = quit without saving

Zones are saved in normalised (0..1) coordinates so they work at any resolution.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

WINDOW = "define_zones"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Define shelf zones.")
    p.add_argument("--source", default="0", help="Webcam index or video path.")
    p.add_argument("--output", default="zones.json")
    p.add_argument("--frame", type=int, default=0, help="Video frame to draw on (files only).")
    return p.parse_args()


def grab_frame(source: str, frame_no: int) -> np.ndarray:
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")
    if is_webcam:
        while True:  # live preview until SPACE
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Camera read failed.")
            preview = frame.copy()
            cv2.putText(preview, "Press SPACE to freeze a frame", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, preview)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Could not read frame {frame_no}.")
    cap.release()
    return frame


def main() -> None:
    args = parse_args()
    base = grab_frame(args.source, args.frame)
    h, w = base.shape[:2]
    polygons: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        view = base.copy()
        overlay = view.copy()
        for poly in polygons:
            cv2.fillPoly(overlay, [np.array(poly, np.int32)], (0, 200, 255))
        view = cv2.addWeighted(overlay, 0.3, view, 0.7, 0)
        for i, poly in enumerate(polygons):
            cv2.polylines(view, [np.array(poly, np.int32)], True, (0, 200, 255), 2)
            cv2.putText(view, f"shelf {i + 1}", poly[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
        for pt in current:
            cv2.circle(view, pt, 4, (0, 255, 0), -1)
        if len(current) > 1:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 255, 0), 2)
        cv2.putText(view, f"zones: {len(polygons)} | click points, n=close, u=undo, c=clear, s=save, q=quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), 13) and len(current) >= 3:
            polygons.append(current.copy())
            current.clear()
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("c"):
            polygons.clear()
            current.clear()
        elif key == ord("s"):
            if len(current) >= 3:
                polygons.append(current.copy())
            if not polygons:
                print("No zones drawn; nothing saved.")
                break
            data = {"shelves": [[[x / w, y / h] for x, y in poly] for poly in polygons]}
            with open(args.output, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"Saved {len(polygons)} zone(s) to {args.output}")
            break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
