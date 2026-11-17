"""Convert the Kaggle CCTV shoplifting dataset into the CSV that train_model.py expects.

Each frame becomes one row: label + the same 34 bbox-normalised keypoint features that
collect_data.py produces, plus a `clip` column (e.g. "shoplifting1") so you can hold whole
clips/scenes out when evaluating. train_model.py ignores that extra column.

Keypoints are re-extracted with OUR yolo11n-pose, the same extractor live_inference.py uses.
(--use-labels reads the dataset's own keypoint labels instead, but those come from a different
pose model, so a classifier trained on them mispredicts on live keypoints.)

Per-frame labels come from the dataset's VLM JSON:
    "conceal" = frames inside the "transition" segment of a shoplifting clip
    "normal"  = every other frame (including all frames of the not_shoplifting clips)

Sources:
    photos  = dataset/images/*.png (456 frames, every 3rd frame of each clip)    [default]
    videos  = every frame of dataset/videos/*.mp4 (--videos)
The photos are a subset of the video frames, so with --videos a photo whose (clip, frame) is
already covered by a video frame is skipped instead of being counted twice.

Examples:
    python3 kaggle_to_csv.py                                   # photos only -> kaggle_dataset.csv
    python3 kaggle_to_csv.py --videos --output kaggle_full_dataset.csv
    python3 train_model.py --input kaggle_full_dataset.csv --output kaggle_full_model.pkl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import cv2
import numpy as np

from pose_utils import (
    FEATURE_COLUMNS, LABEL_COLUMN, MODEL_NAME, NUM_KEYPOINTS, ensure_feature_vector, extract_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_COLUMN = "clip"


def find_dataset() -> str:
    hits = glob.glob(os.path.join(HERE, "kagglehub", "**", "CCTV_Shoplifting_Dataset"), recursive=True)
    if not hits:
        raise SystemExit("Could not find CCTV_Shoplifting_Dataset under ./kagglehub. Pass --dataset PATH.")
    return hits[0]


def conceal_ranges(root: str) -> dict[int, tuple[int, int]]:
    """Frame range of the concealment ('transition') segment for each shoplifting clip."""
    ranges: dict[int, tuple[int, int]] = {}
    for path in glob.glob(os.path.join(root, "VLM-labels", "shoplifting*_vlm_meta.json")):
        name = os.path.basename(path)
        if name.startswith("not_"):
            continue
        n = int(re.match(r"shoplifting(\d+)_", name).group(1))  # type: ignore[union-attr]
        with open(path) as fh:
            seg = json.load(fh)["chronological_sequence_segments"][1]
        lo, hi = seg["frame_range"].split("-")
        ranges[n] = (int(lo), int(hi))
    return ranges


def label_for(clip: str, frame: int, ranges: dict[int, tuple[int, int]]) -> str:
    m = re.match(r"(not_)?shoplifting(\d+)$", clip)
    if not m or m.group(1):  # unknown or not_shoplifting -> always normal
        return "normal"
    lo, hi = ranges.get(int(m.group(2)), (-1, -1))
    return "conceal" if lo <= frame <= hi else "normal"


def row_to_features(row: np.ndarray) -> np.ndarray:
    """YOLO-pose label row -> 34 features, normalised to the person's box like pose_utils."""
    cx, cy, w, h = row[1:5]
    kpts = row[5:].reshape(NUM_KEYPOINTS, 3)
    feats = np.zeros((NUM_KEYPOINTS, 2))
    for i, (kx, ky, vis) in enumerate(kpts):
        if vis > 0:  # 0 = not labelled -> stays 0.0, same as a missing keypoint
            feats[i] = ((kx - (cx - w / 2)) / w, (ky - (cy - h / 2)) / h)
    return ensure_feature_vector(feats)


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle shoplifting dataset -> training CSV.")
    p.add_argument("--dataset", default=None, help="Path to the CCTV_Shoplifting_Dataset folder.")
    p.add_argument("--output", default="kaggle_dataset.csv")
    p.add_argument("--videos", action="store_true", help="Also extract every frame of the .mp4 videos.")
    p.add_argument("--use-labels", action="store_true",
                   help="Use the dataset's own keypoint labels instead of running yolo11n-pose (not recommended).")
    args = p.parse_args()
    if args.use_labels and args.videos:
        raise SystemExit("--videos needs our pose model; it can't be combined with --use-labels.")

    root = args.dataset or find_dataset()
    ranges = conceal_ranges(root)
    yolo = None
    if not args.use_labels:
        from ultralytics import YOLO  # imported lazily so --use-labels needs no YOLO

        yolo = YOLO(MODEL_NAME)

    rows: dict[tuple[str, int], tuple[str, np.ndarray]] = {}  # (clip, frame) -> (label, features)
    n_video = n_photo = n_dup = skipped = 0

    if args.videos:
        for path in sorted(glob.glob(os.path.join(root, "videos", "*.mp4"))):
            clip = os.path.splitext(os.path.basename(path))[0]
            cap = cv2.VideoCapture(path)
            frame_no = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                feats = extract_features(yolo(img, verbose=False)[0])  # type: ignore[misc]
                if feats is None:
                    skipped += 1
                else:
                    rows[(clip, frame_no)] = (label_for(clip, frame_no, ranges), feats)
                    n_video += 1
                frame_no += 1
            cap.release()
            print(f"  video {clip}: {frame_no} frames read")

    for path in sorted(glob.glob(os.path.join(root, "labels", "*.txt"))):
        base = os.path.basename(path)[:-4]
        m = re.match(r"((?:not_)?shoplifting\d+)_f(\d+)$", base)
        if not m:
            continue
        clip, frame = m.group(1), int(m.group(2))
        if (clip, frame) in rows:  # already covered by a video frame
            n_dup += 1
            continue
        if yolo is not None:
            image = cv2.imread(os.path.join(root, "images", base + ".png"))
            feats = extract_features(yolo(image, verbose=False)[0]) if image is not None else None
        else:
            with open(path) as lf:
                label_rows = [np.array(line.split(), dtype=float) for line in lf if line.strip()]
            # largest box = primary person
            feats = row_to_features(max(label_rows, key=lambda r: r[3] * r[4])) if label_rows else None
        if feats is None:
            skipped += 1
            continue
        rows[(clip, frame)] = (label_for(clip, frame, ranges), feats)
        n_photo += 1

    counts = {"conceal": 0, "normal": 0}
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([LABEL_COLUMN, *FEATURE_COLUMNS, CLIP_COLUMN])
        for (clip, _frame), (label, feats) in sorted(rows.items()):
            writer.writerow([label, *(f"{v:.6f}" for v in feats), clip])
            counts[label] += 1

    print(f"Wrote {sum(counts.values())} rows to {args.output}: {counts}")
    print(f"  from videos: {n_video} | from photos: {n_photo} | photos skipped as duplicates of video "
          f"frames: {n_dup} | frames with no person: {skipped}")


if __name__ == "__main__":
    main()
