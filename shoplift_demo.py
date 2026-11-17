"""Rule-based retail "suspicious behaviour" demo: YOLO11-Pose + ByteTrack + shelf zones.

Per tracked person it scores "reach into shelf zone -> hand to pocket within a few
seconds". When a score crosses the threshold it spotlights that person, zooms in on
their head, saves a snapshot, and logs the event on a side dashboard + CSV.

Setup:
    python3 define_zones.py --source store.mp4       # draw shelf zones once
Run:
    python3 shoplift_demo.py --source store.mp4
    python3 shoplift_demo.py --source 0               # webcam

Keys: 'q' = quit. Alerts are prompts for human review, not proof of theft.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from pose_utils import MODEL_NAME
from risk_engine import Event, RiskConfig, RiskEngine

PANEL_W = 340
GREEN, ORANGE, RED, WHITE = (0, 200, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based pose + tracking risk demo.")
    p.add_argument("--source", default="0", help="Webcam index or video path.")
    p.add_argument("--zones", default="zones.json", help="Shelf zones from define_zones.py.")
    p.add_argument("--threshold", type=float, default=40.0, help="Alert score threshold.")
    p.add_argument("--window", type=float, default=4.0, help="Seconds allowed between reach and pocket.")
    p.add_argument("--pocket-ratio", type=float, default=0.3, help="Wrist-to-hip distance as fraction of torso.")
    p.add_argument("--decay", type=float, default=2.0, help="Score decay per second.")
    p.add_argument("--conceal-region", choices=["hip", "torso"], default="hip",
                   help="'hip': wrist near a hip; 'torso': also wrist inside the chest-to-hips area.")
    p.add_argument("--alerts-dir", default="alerts", help="Where alert snapshots are saved.")
    p.add_argument("--events-csv", default="events.csv", help="Event log (appended).")
    p.add_argument("--save-video", default=None, help="Optional output .mp4 of the annotated view.")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window).")
    p.add_argument("--no-face-zoom", action="store_true", help="Disable the head zoom inset.")
    return p.parse_args()


def load_zones(path: str, width: int, height: int) -> list[np.ndarray]:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run define_zones.py first to draw shelf zones.")
    with open(path) as fh:
        shelves = json.load(fh)["shelves"]
    return [(np.array(poly) * [width, height]).astype(np.int32) for poly in shelves]


def put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
        color: tuple[int, int, int] = WHITE, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def spotlight(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Dim everything except the box."""
    x1, y1, x2, y2 = box
    out = (frame * 0.25).astype(np.uint8)
    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out


def head_crop(raw: np.ndarray, xy: np.ndarray, conf: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Crop around the head keypoints (nose/eyes/ears); fall back to the top of the person box."""
    h, w = raw.shape[:2]
    pts = xy[:5][conf[:5] >= 0.3]
    if len(pts) >= 2:
        cx, cy = pts.mean(axis=0)
        half = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 20.0) * 1.2
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    else:
        bx1, by1, bx2, by2 = box
        x1, y1, x2, y2 = bx1, by1, bx2, by1 + (by2 - by1) * 0.2
    x1, y1, x2, y2 = (int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(raw[y1:y2, x1:x2], (150, 150))


def draw_panel(height: int, engine: RiskEngine, visible: list[int], log: deque[str],
               threshold: float, t: float) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), 28, np.uint8)
    put(panel, "RISK DASHBOARD", (12, 28), 0.7, WHITE, 2)
    put(panel, f"t = {t:6.1f}s   people tracked: {len(visible)}", (12, 52), 0.5)
    scored = sorted(((engine.states[i].score, i) for i in visible if i in engine.states), reverse=True)[:6]
    y = 82
    for score, tid in scored:
        ratio = score / threshold
        color = RED if ratio >= 1 else ORANGE if ratio >= 0.5 else GREEN
        put(panel, f"ID {tid}", (12, y), 0.5)
        cv2.rectangle(panel, (60, y - 12), (PANEL_W - 70, y + 2), (60, 60, 60), -1)
        cv2.rectangle(panel, (60, y - 12), (60 + int(min(ratio, 1) * (PANEL_W - 130)), y + 2), color, -1)
        put(panel, f"{score:3.0f}", (PANEL_W - 60, y), 0.5, color)
        y += 26
    y = max(y + 14, 250)
    put(panel, "EVENT LOG", (12, y), 0.6, WHITE, 2)
    for line in list(log)[-8:][::-1]:
        y += 22
        if y > height - 10:
            break
        put(panel, line, (12, y), 0.45, RED if "ALERT" in line else ORANGE)
    return panel


def main() -> None:
    args = parse_args()
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source!r}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if width == 0 or height == 0:
        raise SystemExit("Could not read the frame size from the source.")

    zones = load_zones(args.zones, width, height)
    engine = RiskEngine(RiskConfig(
        shelf_zones=zones, alert_threshold=args.threshold, pocket_window_s=args.window,
        pocket_ratio=args.pocket_ratio, decay_per_s=args.decay, conceal_region=args.conceal_region,
    ))
    model = YOLO(MODEL_NAME)
    os.makedirs(args.alerts_dir, exist_ok=True)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + PANEL_W, height))

    new_csv = not os.path.exists(args.events_csv) or os.path.getsize(args.events_csv) == 0
    log: deque[str] = deque(maxlen=50)
    frame_idx, t0, n_alerts = 0, time.monotonic(), 0

    with open(args.events_csv, "a", newline="") as csv_fh:
        csv_out = csv.writer(csv_fh)
        if new_csv:
            csv_out.writerow(["time_s", "track_id", "event", "score"])

        while True:
            ok, raw = cap.read()
            if not ok:
                break
            t = (time.monotonic() - t0) if is_webcam else frame_idx / fps
            frame_idx += 1

            result = model.track(raw, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            ids: list[int] = []
            boxes: dict[int, tuple[int, int, int, int]] = {}
            kp: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                ids = result.boxes.id.int().cpu().tolist()
                xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                kxy = result.keypoints.xy.cpu().numpy()
                kconf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
                         else np.ones(kxy.shape[:2], dtype=np.float32))
                for i, tid in enumerate(ids):
                    boxes[tid] = tuple(int(v) for v in xyxy[i])  # type: ignore[assignment]
                    kp[tid] = (kxy[i], kconf[i])

            events: list[Event] = []
            for tid in ids:
                events += engine.update(tid, kp[tid][0], kp[tid][1], t)
            engine.prune(set(ids), t)

            for ev in events:
                csv_out.writerow([f"{ev.t:.2f}", ev.track_id, ev.kind, f"{ev.score:.1f}"])
                if ev.kind in ("conceal", "alert"):
                    log.append(f"{ev.t:6.1f}s ID{ev.track_id} {ev.kind.upper()} {ev.score:.0f}")

            # ---- compose the annotated frame ----
            view = result.plot(boxes=False) if ids else raw.copy()
            for z in zones:
                cv2.polylines(view, [z], True, (0, 200, 255), 2)
            for tid in ids:
                st = engine.states.get(tid)
                if st is None:
                    continue
                ratio = st.score / args.threshold
                color = RED if ratio >= 1 else ORANGE if ratio >= 0.5 else GREEN
                x1, y1, x2, y2 = boxes[tid]
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                put(view, f"ID {tid}  risk {st.score:.0f}", (x1, max(14, y1 - 6)), 0.55, color, 2)

            suspect = engine.active_alert_id(t)
            if suspect is not None and suspect in boxes:
                bx1, by1, bx2, by2 = boxes[suspect]
                view = spotlight(view, (max(0, bx1), max(0, by1), min(width, bx2), min(height, by2)))
                put(view, f"ALERT: ID {suspect} - review", (12, 30), 0.9, RED, 2)
                if not args.no_face_zoom:
                    crop = head_crop(raw, *kp[suspect], boxes[suspect])
                    if crop is not None:
                        view[10:160, width - 160:width - 10] = crop
                        cv2.rectangle(view, (width - 160, 10), (width - 10, 160), RED, 2)
            for ev in events:
                if ev.kind == "alert":
                    n_alerts += 1
                    snap = os.path.join(args.alerts_dir, f"alert_{n_alerts:03d}_id{ev.track_id}_{ev.t:.1f}s.jpg")
                    cv2.imwrite(snap, view)
                    print(f"ALERT id={ev.track_id} score={ev.score:.0f} t={ev.t:.1f}s -> {snap}")

            out = np.hstack([view, draw_panel(height, engine, ids, log, args.threshold, t)])
            if writer is not None:
                writer.write(out)
            if not args.no_display:
                cv2.imshow("shoplift_demo  ('q' = quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {frame_idx} frames, {n_alerts} alert(s). Events in {args.events_csv}")


if __name__ == "__main__":
    main()
