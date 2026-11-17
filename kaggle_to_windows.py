"""Build a time-window training CSV from the Kaggle CCTV shoplifting videos.

For each video it tracks people (ByteTrack), takes the main person (the track with the largest
total box area), and for every frame that has ~1.5 s of history builds one window row
(see window_utils.py). A window is labelled "conceal" if at least half of its time span lies
inside the concealment segment from the dataset's VLM JSON (shoplifting clips only), else "normal".

Example:
    python3 kaggle_to_windows.py                       # writes kaggle_windows.csv
    python3 train_window_model.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from kaggle_to_csv import conceal_ranges, find_dataset
from pose_utils import MODEL_NAME, FeatureVector, person_features
from window_utils import WINDOW_COLUMNS, WINDOW_S, PersonBuffer


def track_subject(yolo: Any, path: str, stride: int = 1) -> tuple[float, list[tuple[int, FeatureVector]]]:
    """Track people through one video; return (fps, [(frame, pose), ...]) for the main person.

    The main person is the track with the largest total box area. Only every ``stride``-th frame
    is processed (2 = half the work); windows are sampled by time, so the features barely change.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    per_id: dict[int, list[tuple[int, FeatureVector, float]]] = defaultdict(list)
    frame_no = 0
    while cap.grab():
        if frame_no % stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            res = yolo.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                for i, pid in enumerate(res.boxes.id.int().cpu().tolist()):
                    feats = person_features(res, xyxy, i)
                    if feats is not None:
                        area = float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]))
                        per_id[pid].append((frame_no, feats, area))
        frame_no += 1
    cap.release()
    yolo.predictor = None  # reset the tracker between videos

    if not per_id:
        return fps, []
    subject = max(per_id, key=lambda k: sum(a for _, _, a in per_id[k]))
    return fps, [(f, feats) for f, feats, _ in per_id[subject]]


def subject_windows(track: list[tuple[int, FeatureVector]], fps: float) -> Iterator[tuple[int, FeatureVector]]:
    """Yield (frame, window feature vector) for every frame of ``track`` with enough history."""
    buf = PersonBuffer()
    for frame, feats in track:
        t = frame / fps
        buf.add(t, feats)
        window = buf.window(t)
        if window is not None:
            yield frame, window


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle videos -> time-window training CSV.")
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default="kaggle_windows.csv")
    args = p.parse_args()

    from ultralytics import YOLO

    root = args.dataset or find_dataset()
    ranges = conceal_ranges(root)
    yolo = YOLO(MODEL_NAME)
    counts = {"conceal": 0, "normal": 0}

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", *WINDOW_COLUMNS, "clip", "frame"])
        for path in sorted(glob.glob(os.path.join(root, "videos", "*.mp4"))):
            clip = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"(not_)?shoplifting(\d+)$", clip)
            if m is None:
                print(f"  {clip}: skipped (name is not shoplifting<N> / not_shoplifting<N>)")
                continue
            positive, scene = not m.group(1), int(m.group(2))
            lo, hi = ranges.get(scene, (-1, -1)) if positive else (-1, -1)

            fps, track = track_subject(yolo, path)
            if not track:
                print(f"  {clip}: no people found")
                continue
            n = 0
            for frame, window in subject_windows(track, fps):
                start = frame - WINDOW_S * fps
                overlap = max(0.0, min(hi, frame) - max(lo, start)) / (frame - start)
                label = "conceal" if (positive and overlap >= 0.5) else "normal"
                writer.writerow([label, *(f"{v:.5f}" for v in window), clip, frame])
                counts[label] += 1
                n += 1
            print(f"  {clip}: {n} windows")

    print(f"Wrote {sum(counts.values())} windows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
