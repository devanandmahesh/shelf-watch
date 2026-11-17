"""ShelfWatch browser front end: live view, per-person risk, alerts, live controls.

Everything runs locally — the camera, YOLO and the model never leave this machine, and the page is
served to your own browser. One capture thread owns the models (they are not thread-safe); the slow
video network runs on a single worker thread, as in demo.py, so the view stays smooth. HTTP handlers
only read the latest annotated frame and a state snapshot under a lock.

    python3 web_app.py                                   # video model, webcam, port 8000
    python3 web_app.py --model fused_model.pkl           # pose + pixels, better on its own camera
    python3 web_app.py --source archive/shoplifting/shoplifting-20.mp4 --loop
    python3 web_app.py --host 0.0.0.0 --port 9000        # reachable from the network: see README

Open http://127.0.0.1:8000. The threshold and the warm-up are adjustable while it runs, so you can
calibrate on your own camera without restarting. An alert is a prompt for a human to look, never
proof of anything.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from engine import AlertTracker, PersonView, draw_overlay
from fused_utils import FusedScorer
from pose_utils import MODEL_NAME, NUM_FEATURES, person_features
from video_utils import CLIP_S, MIN_CLIPS, STEP_S, VideoScorer
from window_utils import N_STEPS, WINDOW_FEATURE_DIM, WINDOW_S, PersonBuffer

HERE = Path(__file__).parent
UI_FILE = HERE / "web_ui.html"
ALERT_LOG = 100          # alert/clear events kept for the page
JPEG_QUALITY = 80
SMOOTHING_FRAMES = 5     # single-frame models: average this many frames of probabilities


class Pipeline(threading.Thread):
    """Owns the capture loop and the models; publishes the latest frame and state snapshot."""

    daemon = True

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(name="pipeline")
        self.args = args
        loaded = joblib.load(args.model)
        kind = loaded.get("kind") if isinstance(loaded, dict) else "frame"
        self.window_mode, self.fused_mode = kind == "window", kind == "fused"
        self.stream_mode = kind in ("video", "fused")
        self.scorer = (FusedScorer(loaded, args.threshold, args.min_clips) if self.fused_mode else
                       VideoScorer(loaded, args.threshold, args.min_clips) if self.stream_mode else None)
        self.clf = loaded.get("model") if isinstance(loaded, dict) else loaded
        self.window_s = float(loaded.get("window_s", WINDOW_S)) if self.window_mode else WINDOW_S
        self.n_steps = int(loaded.get("n_steps", N_STEPS)) if self.window_mode else N_STEPS
        expected = WINDOW_FEATURE_DIM if self.window_mode else NUM_FEATURES
        if not self.stream_mode and getattr(self.clf, "n_features_in_", expected) != expected:
            raise SystemExit(f"Model expects {self.clf.n_features_in_} features, not {expected}.")
        self.classes = ["conceal", "normal"] if self.stream_mode else list(self.clf.classes_)
        self.can_alert = args.alert_class in self.classes
        self.alerts = AlertTracker(need=1 if self.stream_mode else args.alert_frames,
                                   latch=not self.stream_mode)

        self.yolo = YOLO(MODEL_NAME)
        self.is_webcam = args.source.isdigit()
        self.buffers: dict[int, PersonBuffer] = {}
        self.history: dict[int, deque[np.ndarray]] = {}
        self.calib: dict[int, tuple[float, float]] = {}   # person -> (highest risk, its threshold)
        self.log: deque[dict[str, object]] = deque(maxlen=ALERT_LOG)
        self.pool = ThreadPoolExecutor(max_workers=1) if self.stream_mode else None

        self.lock = threading.Lock()
        self.new_frame = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.people: list[PersonView] = []
        self.fps = 0.0
        self.paused = False
        self.suppress = False      # calibration mode: measure, do not alert
        self.stop_flag = False
        self.error: str | None = None
        self.source_done = False

    # ---- control, called from HTTP threads -------------------------------------------------
    def control(self, body: dict[str, object]) -> dict[str, object]:
        if "threshold" in body and self.scorer is not None:
            self.scorer.set_threshold(float(body["threshold"]))  # type: ignore[arg-type]
        if "min_clips" in body and self.scorer is not None:
            self.scorer.set_min_clips(int(body["min_clips"]))  # type: ignore[arg-type]
        if "paused" in body:
            self.paused = bool(body["paused"])
        if "suppress" in body:
            self.suppress = bool(body["suppress"])
        if body.get("clear_alerts"):
            with self.lock:
                self.alerts.reset()  # in place: the capture thread holds a reference to it
                self.log.clear()
        if body.get("reset_calibration"):
            with self.lock:
                self.calib.clear()
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            hi = max((v for v, _ in self.calib.values()), default=None)
            over = sum(v > t for v, t in self.calib.values())
            return {
                "model": {"path": self.args.model, "kind": self.kind_name,
                          "device": getattr(self.scorer, "net", None) and self.scorer.net.device,
                          "threshold": self.scorer.threshold if self.scorer else None,
                          "min_clips": getattr(self.scorer, "min_clips", None) or
                                       getattr(getattr(self.scorer, "video", None), "min_clips", None),
                          "warmup_s": self.warmup_s, "classes": self.classes,
                          "alert_class": self.args.alert_class if self.can_alert else None},
                "source": {"name": self.args.source, "webcam": self.is_webcam,
                           "done": self.source_done},
                "fps": round(self.fps, 1),
                "paused": self.paused,
                "suppress": self.suppress,
                "error": self.error,
                "people": [p.as_json() for p in self.people],
                "alerted": sorted(self.alerts.active),
                "ever": sorted(self.alerts.ever),
                "log": list(self.log),
                "calibration": {"people": len(self.calib), "highest": hi, "over_threshold": over,
                                "suggested": None if hi is None else round(min(hi + 0.01, 0.999), 3)},
            }

    @property
    def kind_name(self) -> str:
        if self.fused_mode:
            return "fused pose + video"
        if self.stream_mode:
            return "video"
        return f"time-window ({self.window_s} s)" if self.window_mode else "single-frame"

    @property
    def warmup_s(self) -> float | None:
        if self.scorer is None:
            return None
        n = getattr(self.scorer, "min_clips", None) or self.scorer.video.min_clips
        return round(CLIP_S + n * STEP_S, 1)

    # ---- capture loop ----------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # the page should say what broke, not just stop updating
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(int(self.args.source) if self.is_webcam else self.args.source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open source: {self.args.source!r}")
        if self.is_webcam and self.args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(self.args.width * 9 / 16))
        return cap

    def _loop(self) -> None:
        cap = self._open()
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t_start, t_prev = time.monotonic(), time.monotonic()
        frame_no = 0
        pending: Future | None = None

        while not self.stop_flag:
            if self.paused:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok:
                if self.is_webcam:
                    break
                if self.args.loop:  # a file plays again, which keeps the page alive to look at
                    cap.release()
                    cap = self._open()
                    self.reset_people()
                    t_start, frame_no = time.monotonic(), 0
                    continue
                with self.lock:
                    self.source_done = True
                break
            frame_no += 1
            t_now = (time.monotonic() - t_start) if self.is_webcam else frame_no / fps_src
            result = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            view = result.plot(boxes=False)
            people = self.score_frame(result, frame, t_now, frame_no)

            if self.scorer is not None and self.pool is not None:
                if pending is not None and pending.done():
                    self.scorer.apply(pending.result())
                    pending = None
                if pending is None:
                    due = self.scorer.due(t_now)
                    if due:
                        pending = self.pool.submit(self.scorer.score, due)

            now = time.monotonic()
            fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            draw_overlay(view, people, fps, sorted(self.alerts.active), self.stream_mode)
            ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            with self.lock:
                self.people, self.fps = people, fps
                if ok:
                    self.jpeg = buf.tobytes()
            with self.new_frame:
                self.seq += 1
                self.new_frame.notify_all()
        cap.release()

    def reset_people(self) -> None:
        """Start the per-person state over (a looped file is a new run, not the same people)."""
        with self.lock:
            self.buffers.clear()
            self.history.clear()
            for pid in list(self.alerts.active):
                self.alerts.forget(pid)
        self.yolo.predictor = None  # reset ByteTrack ids

    def score_frame(self, result, frame: np.ndarray, t_now: float, frame_no: int) -> list[PersonView]:
        """Score every tracked person in one frame. Mirrors demo.py, via the same AlertTracker."""
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []
        people: list[PersonView] = []
        ids = result.boxes.id.int().cpu().tolist()
        xyxy = result.boxes.xyxy.cpu().numpy()
        for i, pid in enumerate(ids):
            box = tuple(int(v) for v in xyxy[i])
            feats = person_features(result, xyxy, i)
            if self.scorer is not None:
                if self.fused_mode:
                    self.scorer.add(pid, t_now, frame, xyxy[i], feats)
                else:
                    self.scorer.add(pid, t_now, frame, xyxy[i])
                risk, suspicious = self.scorer.state(pid)
                score, thr = self.scorer.score_and_threshold(pid)
                if risk is None:
                    people.append(PersonView(pid, box, "collecting", -1.0, thr, False,
                                             status=self.scorer.status(pid)))
                    continue
                if score is not None and score > self.calib.get(pid, (-1.0, thr))[0]:
                    self.calib[pid] = (score, thr)
                self.note(pid, suspicious and not self.suppress, frame_no, risk, thr)
                decided = score is not None  # before that, risk is just the latest clip score
                people.append(PersonView(pid, box, "conceal" if risk > thr else "normal", risk, thr,
                                         self.alerts.flagged(pid), alerting=decided and risk > thr,
                                         status=self.scorer.status(pid)))
                continue
            if feats is None:
                continue
            vec = feats
            if self.window_mode:
                buf = self.buffers.setdefault(pid, PersonBuffer(self.window_s, self.n_steps))
                buf.add(t_now, feats)
                vec = buf.window(t_now)
                if vec is None:
                    people.append(PersonView(pid, box, "collecting", -1.0, 0.5, False,
                                             status="collecting pose history"))
                    continue
            smoothing = 1 if self.window_mode else SMOOTHING_FRAMES
            hist = self.history.setdefault(pid, deque(maxlen=smoothing))
            hist.append(self.clf.predict_proba(vec.reshape(1, -1))[0])
            avg = np.mean(hist, axis=0)
            best = int(np.argmax(avg))
            label, conf = str(self.classes[best]), float(avg[best])
            hit = self.can_alert and label == self.args.alert_class and not self.suppress
            self.note(pid, hit, frame_no, conf, 0.5)
            people.append(PersonView(pid, box, label, conf, 0.5, self.alerts.flagged(pid),
                                     alerting=hit, status=f"{label} {conf * 100:.0f}%"))
        return people

    def note(self, pid: int, hit: bool, frame_no: int, risk: float, thr: float) -> None:
        """Update the alert state for one person and log any change."""
        event = self.alerts.update(pid, hit, frame_no)
        if event is not None:
            with self.lock:
                self.log.appendleft({"event": event, "id": pid, "frame": frame_no,
                                     "risk": round(risk, 3), "threshold": round(thr, 3),
                                     "at": time.strftime("%H:%M:%S")})


class Handler(BaseHTTPRequestHandler):
    """Serves the page, the MJPEG stream, the state JSON and the control endpoint."""

    pipeline: Pipeline
    protocol_version = "HTTP/1.1"
    server_version = "shelfwatch"

    def log_message(self, fmt: str, *a: object) -> None:  # keep the console for the pipeline
        pass

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            if not UI_FILE.exists():
                self._send(b"web_ui.html is missing", "text/plain", HTTPStatus.NOT_FOUND)
                return
            self._send(UI_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(json.dumps(self.pipeline.snapshot()).encode(), "application/json")
        elif path == "/frame.jpg":
            with self.pipeline.lock:
                jpeg = self.pipeline.jpeg
            if jpeg is None:
                self._send(b"no frame yet", "text/plain", HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self._send(jpeg, "image/jpeg")
        elif path == "/stream.mjpg":
            self.stream()
        else:
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/control":
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(b'{"error": "bad json"}', "application/json", HTTPStatus.BAD_REQUEST)
            return
        self._send(json.dumps(self.pipeline.control(body)).encode(), "application/json")

    def stream(self) -> None:
        """multipart/x-mixed-replace: one JPEG per boundary, pushed as the pipeline produces them."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        p, last = self.pipeline, -1
        try:
            while not p.stop_flag:
                with p.new_frame:
                    if not p.new_frame.wait_for(lambda: p.seq != last, timeout=5.0):
                        continue
                    last = p.seq
                with p.lock:
                    jpeg = p.jpeg
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the tab or reloaded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShelfWatch browser front end.")
    p.add_argument("--model", default="video_model.pkl", help="Trained model (.pkl).")
    p.add_argument("--source", default="0", help="Webcam index (default 0) or a video file path.")
    p.add_argument("--loop", action="store_true", help="Replay a video file when it ends.")
    p.add_argument("--threshold", type=float, default=None, help="Starting alert threshold.")
    p.add_argument("--min-clips", type=int, default=MIN_CLIPS, help="Clip scores before a verdict.")
    p.add_argument("--alert-class", default="conceal", help="Pose models: class that counts as suspicious.")
    p.add_argument("--alert-frames", type=int, default=8, help="Pose models: frames needed to alert.")
    p.add_argument("--width", type=int, default=0, help="Ask the webcam for this frame width.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: this machine only).")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args)
    Handler.pipeline = pipeline
    pipeline.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("127.0.0.1", "localhost") else args.host
    print(f"Model: {args.model} ({pipeline.kind_name})"
          + (f", threshold {pipeline.scorer.threshold:.2f}" if pipeline.scorer else ""))
    if args.host not in ("127.0.0.1", "localhost"):
        print("Serving to the whole network, with no authentication: anyone who can reach this port "
              "sees your camera. Only do this on a network you trust.")
    print(f"Open http://{shown}:{args.port}   (ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop_flag = True
        with pipeline.new_frame:
            pipeline.new_frame.notify_all()
        server.server_close()
        if pipeline.pool is not None:
            pipeline.pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
