"""Turn the archive videos and the Kaggle videos into video-model clip embeddings.

For each video: pass 1 tracks people (ByteTrack) and keeps the main person's box over time (largest
total box area, as in kaggle_to_windows.py); pass 2 crops that person at the frames each clip needs.
Every STEP_S seconds a clip of the last CLIP_S seconds (see video_utils.py) is embedded by the
pretrained MViT-V2-S. Labels:
    archive: by sub-folder ("shoplifting" -> conceal, others normal), the whole clip
    Kaggle : "conceal" if at least half of the clip lies in the dataset's concealment segment

Output: video_embeddings.npz (emb, label, clip, group, t_end, dataset). Written once at the end.

Examples:
    python3 videos_to_embeddings.py
    python3 videos_to_embeddings.py --limit 2          # quick test: 2 videos per class
    python3 train_video_model.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np

from folder_to_windows import VIDEO_EXTS
from kaggle_to_csv import conceal_ranges, find_dataset
from pose_utils import MODEL_NAME
from video_utils import CLIP_S, MAX_GAP_S, Backbone, crop_person, sample_times

STEP_S = 0.5


def subject_boxes(yolo, path: str, stride: int) -> tuple[float, dict[int, np.ndarray]]:
    """(fps, {frame: xyxy box}) of the main tracked person in a video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    per_id: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    area: dict[int, float] = defaultdict(float)
    frame_no = 0
    while cap.grab():
        if frame_no % stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            res = yolo.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                for i, pid in enumerate(res.boxes.id.int().cpu().tolist()):
                    per_id[pid][frame_no] = xyxy[i]
                    area[pid] += float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]))
        frame_no += 1
    cap.release()
    yolo.predictor = None  # reset the tracker between videos
    if not per_id:
        return fps, {}
    return fps, per_id[max(area, key=area.get)]  # type: ignore[arg-type]


def plan_clips(fps: float, boxes: dict[int, np.ndarray]) -> list[tuple[float, list[tuple[int, int]]]]:
    """[(t_end, [(frame to crop, frame whose box to use) x CLIP_FRAMES]), ...] for every valid clip."""
    if not boxes:
        return []
    tracked = np.array(sorted(boxes))
    t_first, t_last = tracked[0] / fps, tracked[-1] / fps
    plans = []
    for t_end in np.arange(t_first + CLIP_S, t_last + 1e-6, STEP_S):
        frames = np.round(sample_times(t_end) * fps).astype(int)
        nearest = tracked[np.abs(tracked[None, :] - frames[:, None]).argmin(axis=1)]
        if np.abs(nearest - frames).max() / fps <= MAX_GAP_S:
            plans.append((float(t_end), list(zip(frames.tolist(), nearest.tolist()))))
    return plans


def crops_for(path: str, plans, boxes: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{frame: person crop} for every frame any clip needs (pass 2)."""
    need = {f: b for _, pairs in plans for f, b in pairs}
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    frame_no, last = 0, max(need, default=-1)
    while frame_no <= last and cap.grab():
        if frame_no in need:
            ok, img = cap.retrieve()
            if ok:
                out[frame_no] = crop_person(img, boxes[need[frame_no]])
        frame_no += 1
    cap.release()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Archive + Kaggle videos -> video-model clip embeddings.")
    p.add_argument("--root", default="archive", help="Folder with one sub-folder per class.")
    p.add_argument("--positive", default="shoplifting", help="Sub-folder whose videos are 'conceal'.")
    p.add_argument("--kaggle", default=None, help="CCTV_Shoplifting_Dataset path (default: find it).")
    p.add_argument("--output", default="video_embeddings.npz")
    p.add_argument("--stride", type=int, default=2, help="Track every Nth frame (default 2).")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Only the first N videos per class (quick test).")
    args = p.parse_args()

    from ultralytics import YOLO

    videos: list[tuple[str, str, str, str, tuple[int, int] | None]] = []  # path, clip, group, dataset, conceal range
    for cls in sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))):
        paths = sorted(q for q in glob.glob(os.path.join(args.root, cls, "*")) if q.lower().endswith(VIDEO_EXTS))
        for q in paths[:args.limit or None]:
            clip = f"{cls}/{os.path.splitext(os.path.basename(q))[0]}"
            videos.append((q, clip, clip, "archive", (0, 10**9) if cls == args.positive else None))
    kroot = args.kaggle or find_dataset()
    ranges = conceal_ranges(kroot)
    kpaths = sorted(glob.glob(os.path.join(kroot, "videos", "*.mp4")))
    for q in kpaths[:2 * args.limit or None]:
        clip = os.path.splitext(os.path.basename(q))[0]
        m = re.match(r"(not_)?shoplifting(\d+)$", clip)
        if m:
            videos.append((q, clip, f"scene{m.group(2)}", "kaggle", None if m.group(1) else ranges.get(int(m.group(2)))))

    yolo, net = YOLO(MODEL_NAME), Backbone()
    print(f"{len(videos)} videos | backbone on {net.device}")
    cols: dict[str, list] = defaultdict(list)
    t0 = time.monotonic()
    for k, (path, clip, group, dataset, conceal) in enumerate(videos, 1):
        fps, boxes = subject_boxes(yolo, path, args.stride)
        plans = plan_clips(fps, boxes)
        crops = crops_for(path, plans, boxes)
        plans = [(t, pairs) for t, pairs in plans if all(f in crops for f, _ in pairs)]
        for i in range(0, len(plans), args.batch):
            chunk = plans[i:i + args.batch]
            emb = net.embed(np.stack([np.stack([crops[f] for f, _ in pairs]) for _, pairs in chunk]))
            for (t_end, _), e in zip(chunk, emb):
                lo, hi = conceal if conceal else (-1, -1)
                start, end = (t_end - CLIP_S) * fps, t_end * fps
                inside = max(0.0, min(hi, end) - max(lo, start)) / (end - start)
                cols["emb"].append(e)
                cols["label"].append("conceal" if inside >= 0.5 else "normal")
                cols["clip"].append(clip)
                cols["group"].append(group)
                cols["t_end"].append(t_end)
                cols["dataset"].append(dataset)
        print(f"  [{k}/{len(videos)}] {clip}: {len(plans)} clips   {time.monotonic() - t0:.0f}s", flush=True)

    tmp = args.output + ".tmp.npz"
    np.savez_compressed(tmp, **{c: np.array(v) for c, v in cols.items()})
    os.replace(tmp, args.output)
    labels, counts = np.unique(cols["label"], return_counts=True)
    print(f"Wrote {len(cols['emb'])} clip embeddings to {args.output}: {dict(zip(labels, counts.tolist()))}")


if __name__ == "__main__":
    main()
