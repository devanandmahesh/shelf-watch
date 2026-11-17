"""Video-model helpers: person crops over ~2 s fed to a pretrained action-recognition network.

Unlike the pose models (body keypoints only), this looks at the PIXELS around a person, so it can
see the item, the bag and the hands. A "clip" is CLIP_FRAMES square crops of one tracked person,
sampled evenly over the last CLIP_S seconds. The backbone is torchvision's MViT-V2-S pretrained on
Kinetics-400 (400 human actions); its 768-number clip embedding feeds a small classifier.
Training (videos_to_embeddings.py) and live use (demo.py) share this code.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

CLIP_FRAMES = 16
CLIP_S = 2.0
CROP = 224          # MViT-V2-S input size
CROP_MARGIN = 1.15  # square crop side = this x the longer side of the person box
MAX_GAP_S = 0.4     # a sample time with no person box within this long makes the clip invalid
MEAN = np.array([0.45, 0.45, 0.45], np.float32)    # Kinetics normalisation (RGB)
STD = np.array([0.225, 0.225, 0.225], np.float32)
EMBED_DIM = 768
STEP_S = 0.5        # live: each person is re-checked at most this often
HISTORY = 20        # live: clip scores kept per person (~10 s), like one training video
MIN_CLIPS = 12      # live: scores needed before a person can be flagged (~6 s of them)
FORGET_S = 5.0      # live: a person unseen this long is dropped, scores and all


def crop_person(frame: NDArray[np.uint8], box: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Square CROP x CROP RGB crop around an xyxy box; areas outside the frame are black."""
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * CROP_MARGIN / 2
    sx1, sy1, sx2, sy2 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    h, w = frame.shape[:2]
    out = np.zeros((sy2 - sy1, sx2 - sx1, 3), np.uint8)
    fx1, fy1, fx2, fy2 = max(sx1, 0), max(sy1, 0), min(sx2, w), min(sy2, h)
    if fx2 > fx1 and fy2 > fy1:
        out[fy1 - sy1:fy2 - sy1, fx1 - sx1:fx2 - sx1] = frame[fy1:fy2, fx1:fx2]
    return cv2.cvtColor(cv2.resize(out, (CROP, CROP), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def sample_times(t_end: float) -> NDArray[np.float64]:
    """The CLIP_FRAMES timestamps (seconds) of a clip ending at t_end."""
    return np.linspace(t_end - CLIP_S, t_end, CLIP_FRAMES)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Backbone:
    """Pretrained MViT-V2-S with the classification head removed: clips -> 768-number embeddings."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        self.torch = torch
        self.device = device or pick_device()
        model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        model.head = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    def embed(self, clips: NDArray[np.uint8]) -> NDArray[np.float32]:
        """(B, CLIP_FRAMES, CROP, CROP, 3) uint8 RGB -> (B, 768) float32."""
        x = (clips.astype(np.float32) / 255.0 - MEAN) / STD
        x = self.torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()  # B, C, T, H, W
        with self.torch.inference_mode():
            return self.model(x.to(self.device)).float().cpu().numpy()


class VideoScorer:
    """Live per-person scoring with a model saved by train_video_model.py.

    add() stores each person's crops; due() returns the clips ready to check; score() runs the
    network (slow, ~0.45 s per person on an M-series GPU, so demo.py may call it on a worker thread)
    and returns results that apply() records. A person is suspicious, as in training, when the
    `quantile` of their recent clip probabilities is above `threshold`.

    That quantile only means what it meant in training if it spans enough clips, hence MIN_CLIPS:
    over 3 scores the 0.9 quantile is all but the maximum, so one odd 2-second clip raised an alert
    that training would never have counted. From 12 up it takes two clips over the line, as it did
    on a whole training video (~18 clips), at the price of ~8 s of tracking before any verdict.
    """

    def __init__(self, info: dict[str, Any], threshold: float | None = None,
                 min_clips: int | None = None) -> None:
        self.head = info["model"]
        self.threshold = float(info["threshold"] if threshold is None else threshold)
        self.quantile = float(info["quantile"])
        self.min_clips = int(MIN_CLIPS if min_clips is None else min_clips)
        self.net = Backbone()
        self.crops: dict[int, deque[tuple[float, NDArray[np.uint8]]]] = {}
        self.probs: dict[int, deque[float]] = {}
        self.last_check: dict[int, float] = {}

    def add(self, pid: int, t: float, frame: NDArray[np.uint8], box: NDArray[np.float32]) -> None:
        q = self.crops.setdefault(pid, deque())
        if q and t - q[-1][0] < CLIP_S / CLIP_FRAMES / 2:  # 16 crops/s is plenty for 8 samples/s
            return
        q.append((t, crop_person(frame, box)))
        while q[0][0] < t - CLIP_S - 2 * MAX_GAP_S:
            q.popleft()

    def due(self, t: float) -> list[tuple[int, NDArray[np.uint8]]]:
        """(person id, clip) for every person seen just now whose last check is STEP_S old."""
        for pid in [k for k, q in self.crops.items() if t - q[-1][0] > FORGET_S]:  # gone for a while
            del self.crops[pid]
            self.probs.pop(pid, None)       # scores must not outlive the person, nor carry over to
            self.last_check.pop(pid, None)  # whoever ByteTrack hands this id to next
        out = []
        targets = sample_times(t)
        for pid, q in self.crops.items():
            if t - self.last_check.get(pid, -np.inf) < STEP_S or t - q[-1][0] > MAX_GAP_S:
                continue
            times = np.fromiter((s for s, _ in q), dtype=np.float64)
            idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
            if np.abs(times[idx] - targets).max() > MAX_GAP_S:
                continue
            out.append((pid, np.stack([q[i][1] for i in idx])))
            self.last_check[pid] = t
        return out

    def score(self, items: list[tuple[int, NDArray[np.uint8]]]) -> list[tuple[int, float]]:
        """[(person id, P(conceal))]. Touches no shared state, so it is safe on a worker thread."""
        emb = self.net.embed(np.stack([clip for _, clip in items]))
        return list(zip([pid for pid, _ in items], self.head.predict_proba(emb)[:, 1].tolist()))

    def apply(self, results: list[tuple[int, float]]) -> None:
        for pid, p in results:
            self.probs.setdefault(pid, deque(maxlen=HISTORY)).append(p)

    def score_and_threshold(self, pid: int) -> tuple[float | None, float]:
        """(the number to compare, the threshold it must beat). FusedScorer overrides this."""
        return self.window_score(pid), self.threshold

    def state(self, pid: int) -> tuple[float | None, bool]:
        """(latest P(conceal), or None if not checked yet; is this person suspicious right now?)

        The answer can go back to False: it is read off the last HISTORY scores only, so evidence
        rolls out of the window as the person carries on doing nothing.
        """
        ps = self.probs.get(pid)
        if not ps:
            return None, False
        window = self.window_score(pid)
        return ps[-1], window is not None and window > self.threshold

    def set_threshold(self, value: float) -> None:
        """Change the alert threshold while running (web_app.py's slider)."""
        self.threshold = float(value)

    def set_min_clips(self, n: int) -> None:
        """Change how many clip scores a verdict needs while running."""
        self.min_clips = max(1, int(n))

    @property
    def any_scores(self) -> bool:
        """Has any person been scored at all? False for a long time means the loop is too slow."""
        return bool(self.probs)

    def status(self, pid: int) -> str:
        """One line on how far this person is from a verdict — for demo.py --verbose."""
        crops, scores = len(self.crops.get(pid, ())), len(self.probs.get(pid, ()))
        w = self.window_score(pid)
        return (f"{crops:3d} crops, {scores:2d}/{self.min_clips} clip scores, "
                + (f"video {w:.2f}" if w is not None else "no verdict yet"))

    def window_score(self, pid: int) -> float | None:
        """The number compared against `threshold`, or None before `min_clips` scores."""
        ps = self.probs.get(pid)
        if not ps or len(ps) < self.min_clips:
            return None
        return float(np.quantile(ps, self.quantile))
