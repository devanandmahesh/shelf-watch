"""Build a time-window training CSV from a folder of videos sorted into one sub-folder per class.

    archive/
        shoplifting/*.mp4   -> every window labelled "conceal"
        normal/*.mp4        -> every window labelled "normal"   (any other sub-folder too)

Labels are per VIDEO (there are no start/end times), so the whole of a shoplifting clip counts as
"conceal", including the seconds before and after the act. That is noisy, but fine for short clips
trimmed around the theft. Only the main person of each clip is used (largest total box area), the
same as kaggle_to_windows.py, so bystanders in a shoplifting clip are not labelled as thieves.

Each video is its own `group`, so train_window_model.py holds out whole videos when testing.

Examples:
    python3 folder_to_windows.py                               # archive/ -> archive_windows.csv
    python3 folder_to_windows.py --root my_clips --output my_windows.csv --stride 1
    python3 folder_to_windows.py --resume                      # only videos missing from the CSV
    python3 train_window_model.py --input archive_windows.csv kaggle_windows.csv

The CSV is written once at the end (temp file + rename), so a sync tool such as iCloud Desktop
can't corrupt a half-written file; a first run that streamed rows ended up with a 27 MB hole.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

from kaggle_to_windows import subject_windows, track_subject
from pose_utils import MODEL_NAME
from window_utils import WINDOW_COLUMNS

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
HEADER = ["label", *WINDOW_COLUMNS, "clip", "frame", "group"]


def read_existing(path: str) -> dict[str, list[list[str]]]:
    """Well-formed rows of an earlier output, by clip. Damaged lines (e.g. NUL bytes) are dropped."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as fh:
        text = fh.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
    rows: dict[str, list[list[str]]] = {}
    for row in csv.reader(text.splitlines()[1:]):
        if len(row) == len(HEADER) and row[0] in ("conceal", "normal"):
            rows.setdefault(row[-1], []).append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Folder of class sub-folders of videos -> time-window CSV.")
    p.add_argument("--root", default="archive", help="Folder holding one sub-folder per class.")
    p.add_argument("--output", default="archive_windows.csv")
    p.add_argument("--positive", default="shoplifting", help="Sub-folder whose videos are 'conceal'.")
    p.add_argument("--stride", type=int, default=2, help="Process every Nth frame (default 2: 30 fps -> 15).")
    p.add_argument("--limit", type=int, default=0, help="Only the first N videos per class (quick test).")
    p.add_argument("--resume", action="store_true",
                   help="Keep the good rows already in --output and only process videos missing from it.")
    args = p.parse_args()

    from ultralytics import YOLO

    classes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    if args.positive not in classes:
        raise SystemExit(f"No {args.positive!r} sub-folder in {args.root}; found {classes}.")
    yolo = YOLO(MODEL_NAME)
    existing = read_existing(args.output) if args.resume else {}
    if args.resume:
        print(f"Resuming: {len(existing)} videos already in {args.output}")
    rows: list[list[object]] = []
    t0 = time.monotonic()

    for cls in classes:
        label = "conceal" if cls == args.positive else "normal"
        paths = sorted(p for p in glob.glob(os.path.join(args.root, cls, "*"))
                       if p.lower().endswith(VIDEO_EXTS))
        if args.limit:
            paths = paths[:args.limit]
        print(f"{cls}/ -> '{label}': {len(paths)} videos")
        for k, path in enumerate(paths, 1):
            clip = f"{cls}/{os.path.splitext(os.path.basename(path))[0]}"
            if clip in existing:
                rows += existing[clip]
                continue
            fps, track = track_subject(yolo, path, args.stride)
            n = 0
            for frame, window in subject_windows(track, fps):
                rows.append([label, *(f"{v:.5f}" for v in window), clip, frame, clip])
                n += 1
            print(f"  [{k}/{len(paths)}] {clip}: {n} windows"
                  + ("" if track else " (no people found)") + f"   {time.monotonic() - t0:.0f}s")

    tmp = args.output + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)
    os.replace(tmp, args.output)
    counts = {lab: sum(1 for r in rows if r[0] == lab) for lab in ("conceal", "normal")}
    print(f"Wrote {len(rows)} windows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
