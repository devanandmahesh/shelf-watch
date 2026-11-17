"""Camera demo: track every person, classify each one, and flag a suspect.

For each tracked person (ByteTrack IDs) it classifies their pose with your trained model,
smooths the result over a few frames, and raises a per-person alert when the alert class
(default "conceal") holds for several frames in a row. The alert stays on for that person.

With a fused model (train_fused_model.py) it scores each person from BOTH their skeleton motion and
the pixels around them, behind one threshold — one model file instead of three. With a video model
(train_video_model.py) it uses the pixels alone. Either way it checks each person's last 2 s every
0.5 s and flags them when their recent theft risk stays above the model's threshold. That verdict
is read off their last ~10 s only, so it clears again once the risk falls back: a flag means "this
person looks suspicious now", not "this person stole something". On a webcam the check runs on a
background thread so the view stays smooth; on a video file it runs in step with the video, so
playback is slower than real time.

Examples:
    python3 demo.py                                  # webcam, video_model.pkl
    python3 demo.py --source archive/shoplifting/shoplifting-20.mp4
    python3 demo.py --calibrate --source my_quiet_shop.mp4            # fit a threshold
    python3 demo.py --model fused_model.pkl          # pose + pixels, one threshold
    python3 demo.py --model kaggle_full_model.pkl
    python3 demo.py --source clip.mp4 --save-video out.mp4
    python3 demo.py --model my_model.pkl --alert-class waving    # your own classes

Keys: 'q' = quit. An alert is a prompt for a human to look, never proof of anything.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from engine import AlertTracker, PersonView, draw_overlay, put
from fused_utils import FusedScorer
from pose_utils import MODEL_NAME, NUM_FEATURES, person_features
from video_utils import CLIP_S, MAX_GAP_S, MIN_CLIPS, STEP_S, VideoScorer
from window_utils import N_STEPS, WINDOW_FEATURE_DIM, WINDOW_S, PersonBuffer

SMOOTHING_FRAMES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera demo: per-person pose classification + alerts.")
    p.add_argument("--source", default="0", help="Webcam index (default 0) or a video file path.")
    p.add_argument("--model", default="video_model.pkl", help="Trained classifier (.pkl).")
    p.add_argument("--alert-class", default="conceal", help="Class that counts as suspicious.")
    p.add_argument("--alert-frames", type=int, default=8, help="Consecutive frames needed to alert.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Video models only: alert threshold (default: the one saved with the model).")
    p.add_argument("--min-clips", type=int, default=MIN_CLIPS,
                   help=f"Video/fused models only: clip scores needed per person before a verdict "
                        f"(default {MIN_CLIPS}; lower reacts sooner and false-alarms more).")
    p.add_argument("--verbose", action="store_true",
                   help="Print each tracked person's progress towards a verdict, once a second.")
    p.add_argument("--calibrate", action="store_true",
                   help="Video models only: footage with NO theft in it; raise no alerts and print "
                        "the --threshold that would keep this camera quiet.")
    p.add_argument("--save-video", default=None, help="Optional output .mp4 of the annotated view.")
    p.add_argument("--no-display", action="store_true", help="Run without a window; print the result.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loaded = joblib.load(args.model)
    # Time-window and video models are saved as a dict; a frame model is the bare classifier.
    kind = loaded.get("kind") if isinstance(loaded, dict) else "frame"
    window_mode, fused_mode = kind == "window", kind == "fused"
    video_mode = kind in ("video", "fused")  # both score a person from their recent history
    scorer = (FusedScorer(loaded, args.threshold, args.min_clips) if fused_mode else
              VideoScorer(loaded, args.threshold, args.min_clips) if video_mode else None)
    clf = loaded.get("model") if isinstance(loaded, dict) else loaded
    window_s = float(loaded.get("window_s", WINDOW_S)) if window_mode else WINDOW_S
    n_steps = int(loaded.get("n_steps", N_STEPS)) if window_mode else N_STEPS
    expected = WINDOW_FEATURE_DIM if window_mode else NUM_FEATURES
    if not video_mode and getattr(clf, "n_features_in_", expected) != expected:
        raise SystemExit(f"Model expects {clf.n_features_in_} features, not {expected}. Retrain it.")
    if scorer is not None:
        streams = "fused pose + video" if fused_mode else "video"
        print(f"Model type: {streams} (MViT on {scorer.net.device}), threshold {scorer.threshold:.2f}")
        print(f"A verdict needs ~{CLIP_S + args.min_clips * STEP_S:.0f} s of UNBROKEN tracking per "
              f"person ({args.min_clips} clip scores); a lost or renumbered track starts that over.")
    else:
        print(f"Model type: {'time-window (' + str(window_s) + ' s)' if window_mode else 'single-frame'}")
    classes = ["conceal", "normal"] if video_mode else list(clf.classes_)
    can_alert = args.alert_class in classes
    if not can_alert:
        print(f"Note: model has no class {args.alert_class!r}; alerts disabled. Classes: {classes}")

    yolo = YOLO(MODEL_NAME)
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source!r}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = (cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width, height))
              if args.save_video else None)

    smoothing = 1 if window_mode else SMOOTHING_FRAMES  # a window already spans ~1.5 s
    buffers: dict[int, PersonBuffer] = {}        # per-person pose history (window models only)
    t_start = time.monotonic()
    history: dict[int, deque[np.ndarray]] = {}   # per-person recent probabilities
    tracker = AlertTracker(need=1 if video_mode else args.alert_frames, latch=not video_mode)
    calib: dict[int, float] = {}                 # calibration: highest windowed risk seen per person
    calib_thr: dict[int, float] = {}             # and the threshold that risk was judged against
    pool = ThreadPoolExecutor(max_workers=1) if video_mode else None
    pending: Future | None = None                # video model check running on the worker thread
    frame_no, t_prev, fps = 0, time.monotonic(), 0.0
    t_said, warned = time.monotonic(), False   # last progress line; slow-loop warning printed
    print("Running." + ("" if args.no_display else " Press 'q' in the video window to quit."))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        t_now = (time.monotonic() - t_start) if is_webcam else frame_no / fps_src  # seconds
        result = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        view = result.plot(boxes=False)  # skeletons

        people: list[PersonView] = []
        if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
            ids = result.boxes.id.int().cpu().tolist()
            xyxy = result.boxes.xyxy.cpu().numpy()
            for i, pid in enumerate(ids):
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                box = (x1, y1, x2, y2)
                if scorer is not None:
                    if fused_mode:  # the pose stream needs this frame's keypoints too
                        scorer.add(pid, t_now, frame, xyxy[i], person_features(result, xyxy, i))
                    else:
                        scorer.add(pid, t_now, frame, xyxy[i])
                    risk, suspicious = scorer.state(pid)
                    w, thr = scorer.score_and_threshold(pid)  # thr: fused, or video-only in fallback
                    if risk is None:  # not checked yet
                        people.append(PersonView(pid, box, "collecting", -1.0, thr, False))
                        continue
                    if w is not None and w > calib.get(pid, -1.0):
                        calib[pid], calib_thr[pid] = w, thr  # keep the pair, the rules differ
                    event = tracker.update(pid, suspicious and not args.calibrate, frame_no)
                    if event == "raised":
                        print(f"ALERT: person {pid} theft risk above {thr:.2f} (frame {frame_no})")
                    elif event == "cleared":
                        print(f"cleared: person {pid} back below {thr:.2f} (frame {frame_no})")
                    label = "conceal" if risk > thr else "normal"
                    people.append(PersonView(pid, box, label, risk, thr, tracker.flagged(pid),
                                             alerting=w is not None and risk > thr))
                    continue
                feats = person_features(result, xyxy, i)
                if feats is None:
                    continue
                vec = feats
                if window_mode:
                    buf = buffers.setdefault(pid, PersonBuffer(window_s, n_steps))
                    buf.add(t_now, feats)
                    vec = buf.window(t_now)
                    if vec is None:  # not enough history yet
                        people.append(PersonView(pid, box, "collecting", -1.0, 0.5, False))
                        continue
                hist = history.setdefault(pid, deque(maxlen=smoothing))
                hist.append(clf.predict_proba(vec.reshape(1, -1))[0])
                avg = np.mean(hist, axis=0)
                best = int(np.argmax(avg))
                label, conf = str(classes[best]), float(avg[best])
                hit = can_alert and label == args.alert_class
                if tracker.update(pid, hit, frame_no) == "raised":
                    print(f"ALERT: person {pid} '{args.alert_class}' for {args.alert_frames} frames (frame {frame_no})")
                people.append(PersonView(pid, box, label, conf, 0.5, tracker.flagged(pid),
                                         alerting=hit))

        if scorer is not None and pool is not None:
            # Webcam: collect a finished check, start the next. File: wait, so results follow video time.
            if pending is not None and (pending.done() or not is_webcam):
                scorer.apply(pending.result())
                pending = None
            if pending is None:
                due = scorer.due(t_now)
                if due:
                    pending = pool.submit(scorer.score, due)
                    if not is_webcam:
                        scorer.apply(pending.result())
                        pending = None

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        draw_overlay(view, people, fps, sorted(tracker.active), video_mode)
        if scorer is not None and now - t_said > 1.0:
            t_said = now
            if args.verbose:
                print(f"t={t_now:6.1f}s  fps {fps:4.1f}  people {len(people)}"
                      + "".join(f"\n    ID {p.pid}: {scorer.status(p.pid)}" for p in people))
            if not scorer.any_scores and t_now > 15.0 and not warned:
                warned = True
                print(f"No clip has been scored in {t_now:.0f} s, so nobody can be flagged yet. A "
                      f"{CLIP_S:.0f} s clip needs the loop above ~{1 / MAX_GAP_S:.1f} fps (it is at "
                      f"{fps:.1f}) and one person tracked with no gap longer than {MAX_GAP_S:.1f} s. "
                      f"A smaller camera frame, or standing fully in view, usually fixes it.")

        if writer is not None:
            writer.write(view)
        if not args.no_display:
            cv2.imshow("demo  ('q' = quit)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if pool is not None:
        pool.shutdown(wait=True)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if args.calibrate and video_mode:
        if not calib:
            print("Calibration: nobody was tracked long enough to be scored. Use longer footage.")
        else:
            hi = max(calib.values())
            over = sum(v > calib_thr[k] for k, v in calib.items())  # each by the rule that judged it
            print(f"\nCalibration over {len(calib)} tracked people, highest windowed risk {hi:.2f} "
                  f"(per person: {', '.join(f'{k}:{v:.2f}' for k, v in sorted(calib.items()))})")
            thr = scorer.threshold  # type: ignore[union-attr]
            print(f"  at the current threshold {thr:.2f} this theft-free footage would have flagged "
                  f"{over} of them")
            if hi <= thr:
                print(f"  {thr:.2f} already clears this footage; keep it, and add more footage to be sure")
            else:
                print(f"  --threshold {min(hi + 0.01, 0.999):.3f} would keep it quiet; raising it costs "
                      f"real detections too, so calibrate on several minutes, not one clip")
    elif can_alert:
        print("Verdict: SUSPECTED (person " + ", ".join(str(k) for k in tracker.ever) + ")" if tracker.ever
              else "Verdict: nothing suspicious detected")


if __name__ == "__main__":
    main()
