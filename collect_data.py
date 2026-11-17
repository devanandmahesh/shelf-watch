"""Stage 1 - Dataset builder.

Extract YOLO11 pose keypoints from a webcam or video file and append them,
with a label, to a CSV file.

Examples:
    python collect_data.py --label idle                       # webcam, hold 'r' to record
    python collect_data.py --label waving --output dataset.csv
    python collect_data.py --label jumping --source clips/jump.mp4   # whole video, automatic

Keys: hold 'r' = record (webcam only), 'q' = quit.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO

from pose_utils import FEATURE_COLUMNS, LABEL_COLUMN, MODEL_NAME, extract_features

# OpenCV's waitKey only reports key *presses* (OS auto-repeat), not releases.
# So "held" = a recent 'r' event. The first repeat arrives after the OS initial
# repeat delay, hence the longer grace period before repeats have started.
FIRST_PRESS_GRACE_S = 0.7
REPEAT_GRACE_S = 0.15


class HoldKey:
    """Approximates 'is this key currently held down?' from OpenCV key events."""

    def __init__(self) -> None:
        self._last = 0.0
        self._repeating = False

    def press(self) -> None:
        now = time.monotonic()
        if self.is_active(now):
            self._repeating = True  # a second event inside the window => auto-repeat
        else:
            self._repeating = False
        self._last = now

    def is_active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        grace = REPEAT_GRACE_S if self._repeating else FIRST_PRESS_GRACE_S
        return (now - self._last) < grace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record labelled YOLO pose keypoints to a CSV.")
    p.add_argument("--label", required=True, help="Action name, e.g. 'idle' or 'waving'.")
    p.add_argument("--source", default="0", help="'0' for webcam (default) or a path to a video file.")
    p.add_argument("--output", default="dataset.csv", help="CSV file to append to (default: dataset.csv).")
    return p.parse_args()


def open_capture(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Return (capture, is_webcam). Digit strings are camera indices."""
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source!r}")
    return cap, is_webcam


def draw_hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 36
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 36


def main() -> None:
    args = parse_args()
    model = YOLO(MODEL_NAME)
    cap, is_webcam = open_capture(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    new_file = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    recorded = 0
    window = "collect_data  (hold 'r' = record, 'q' = quit)" if is_webcam else "collect_data  ('q' = quit)"
    hold = HoldKey()

    print(f"Label: {args.label!r} | source: {'webcam' if is_webcam else args.source} | output: {args.output}")
    if is_webcam:
        print("Hold 'r' to record. Press 'q' to quit.")

    with open(args.output, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow([LABEL_COLUMN, *FEATURE_COLUMNS])

        while True:
            ok, frame = cap.read()
            if not ok:  # end of video file or camera failure
                break

            result = model(frame, verbose=False)[0]
            features = extract_features(result)  # always 34 floats, or None if no person
            recording = hold.is_active() if is_webcam else True

            if recording and features is not None:
                writer.writerow([args.label, *(f"{v:.6f}" for v in features)])
                recorded += 1

            annotated = result.plot(boxes=False)
            if is_webcam:
                status = ("REC", (0, 0, 255)) if recording else ("PAUSED - hold 'r'", (0, 200, 255))
            else:
                done = f"{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}" if total_frames else ""
                status = (f"PROCESSING {done}", (0, 200, 255))
            draw_hud(annotated, [
                (status[0], status[1]),
                (f"Label: {args.label}", (255, 255, 255)),
                (f"Recorded frames: {recorded}", (0, 255, 0)),
            ] + ([] if features is not None else [("No person detected", (0, 0, 255))]))

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                hold.press()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Done. Wrote {recorded} rows for label {args.label!r} to {args.output}")
    if os.path.exists(args.output):
        with open(args.output, newline="") as fh:
            counts = Counter(row[0] for i, row in enumerate(csv.reader(fh)) if i > 0 and row)
        print("Dataset totals:", dict(counts))


if __name__ == "__main__":
    main()
