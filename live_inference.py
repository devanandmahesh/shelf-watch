"""Stage 3 - Real-time webcam activity recognition.

Example:
    python live_inference.py
    python live_inference.py --source 1 --model activity_model.pkl
    python live_inference.py --source clip.mp4 --model kaggle_model.pkl

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
from collections import deque

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from pose_utils import MODEL_NAME, NUM_FEATURES, extract_features

SMOOTHING_FRAMES = 5  # average probabilities over the last N frames to stop label flicker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live activity recognition.")
    p.add_argument("--model", default="activity_model.pkl", help="Trained classifier from train_model.py.")
    p.add_argument("--source", default="0", help="Webcam index (default 0) or a path to a video file.")
    p.add_argument("--alert-class", default="conceal",
                   help="Class name that means 'suspicious' (used for the verdict). Default: conceal.")
    p.add_argument("--alert-frames", type=int, default=8,
                   help="Consecutive frames of --alert-class needed to raise the alert (default 8).")
    p.add_argument("--no-display", action="store_true", help="Run without a window; just print the verdict.")
    return p.parse_args()


def draw_label(frame: np.ndarray, text: str, color: tuple[int, int, int], bottom: bool = False) -> None:
    """Large text with a dark backing box so it's readable on any background."""
    scale = max(1.0, frame.shape[1] / 640 * 1.2)
    thick = max(2, int(scale * 2))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = 15, (frame.shape[0] - 25 if bottom else 20 + h)
    cv2.rectangle(frame, (x - 10, y - h - 12), (x + w + 10, y + base + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    clf = joblib.load(args.model)
    if getattr(clf, "n_features_in_", NUM_FEATURES) != NUM_FEATURES:
        raise SystemExit(f"Model expects {clf.n_features_in_} features, not {NUM_FEATURES}. Retrain it.")
    yolo = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source!r}")

    history: deque[np.ndarray] = deque(maxlen=SMOOTHING_FRAMES)
    can_alert = args.alert_class in list(clf.classes_)
    if not can_alert:
        print(f"Note: model has no class {args.alert_class!r}; verdict disabled. Classes: {list(clf.classes_)}")
    streak, alert_frame, frame_no = 0, None, 0  # alert latches once raised and stays on
    print("Running." + ("" if args.no_display else " Press 'q' in the video window to quit."))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        result = yolo(frame, verbose=False)[0]
        features = extract_features(result)  # identical to collect_data.py; always 34 floats
        annotated = result.plot(boxes=False)  # skeleton overlay

        if features is None:
            history.clear()
            streak = 0
            draw_label(annotated, "No person detected", (0, 165, 255))
        else:
            history.append(clf.predict_proba(features.reshape(1, -1))[0])
            avg = np.mean(history, axis=0)
            best = int(np.argmax(avg))
            label, conf = str(clf.classes_[best]), float(avg[best])
            color = (0, 220, 0) if conf >= 0.7 else (0, 200, 255)
            draw_label(annotated, f"{label}  {conf * 100:.0f}%", color)
            streak = streak + 1 if (can_alert and label == args.alert_class) else 0
            if alert_frame is None and streak >= args.alert_frames:
                alert_frame = frame_no
                print(f"ALERT: '{args.alert_class}' held for {args.alert_frames} frames (frame {frame_no})")

        if alert_frame is not None:
            draw_label(annotated, "SHOPLIFTING SUSPECTED", (0, 0, 255), bottom=True)

        if not args.no_display:
            cv2.imshow("live_inference  ('q' = quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if can_alert:
        print("Verdict: SHOPLIFTING SUSPECTED" if alert_frame is not None else "Verdict: nothing suspicious detected")


if __name__ == "__main__":
    main()
