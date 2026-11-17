"""The fused model: the pose-window stream and the video stream behind ONE threshold.

The two models see different things about the same person — the pose stream reads 340 numbers of
skeleton motion over ~1.5 s (window_utils.py), the video stream reads the pixels around them over
2 s (video_utils.py) — so demo.py used to need a different model file, code path and alert rule for
each. FusedScorer runs both for every tracked person, takes each stream's 90th-percentile score over
its recent history, and averages the two into one number compared against one threshold.

Measured per video with whole videos held out (train_fused_model.py): fusing is worth about +0.03
AUC on the camera the model was trained on, and nothing at all on a camera it has never seen, where
both streams and the fusion sit at chance. It buys tidier code and a small in-domain gain; it does
not buy generalisation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pose_utils import FeatureVector
from video_utils import FORGET_S, VideoScorer
from window_utils import PersonBuffer

POSE_HISTORY_S = 10.0  # live: pose scores kept per person, matching the video stream's ~10 s
MIN_POSE_WINDOWS = 12  # live: pose scores needed before the pose stream has an opinion


class FusedScorer:
    """Live per-person scoring with a model saved by train_fused_model.py.

    The call sequence is VideoScorer's, with the person's pose features passed to add() as well:
    add() feeds both streams, due()/score()/apply() run the slow video network (demo.py may put
    score() on a worker thread), and state() answers from both. A person is suspicious when the
    average of the two streams' recent quantiles is above `threshold` — which needs both streams to
    have enough history, so a verdict takes ~8 s of tracking.
    """

    def __init__(self, info: dict[str, Any], threshold: float | None = None,
                 min_clips: int | None = None) -> None:
        # An explicit threshold is a deliberate override, so it governs the fallback rule too;
        # left to itself each rule keeps the threshold calibrated for it.
        self.video = VideoScorer(info["video"], threshold, min_clips=min_clips)
        self.pose = info["pose"]["model"]
        self.window_s = float(info["pose"]["window_s"])
        self.n_steps = int(info["pose"]["n_steps"])
        self.threshold = float(info["threshold"] if threshold is None else threshold)
        self.quantile = float(info["quantile"])
        self.buffers: dict[int, PersonBuffer] = {}
        self.pose_probs: dict[int, deque[tuple[float, float]]] = {}
        self.last_seen: dict[int, float] = {}

    @property
    def net(self):  # demo.py prints which device the backbone runs on
        return self.video.net

    def add(self, pid: int, t: float, frame: NDArray[np.uint8], box: NDArray[np.float32],
            feats: FeatureVector | None = None) -> None:
        """Feed one frame's view of a person to both streams; feats None = no usable pose."""
        self.last_seen[pid] = t
        self.video.add(pid, t, frame, box)
        if feats is None:
            return
        buf = self.buffers.setdefault(pid, PersonBuffer(self.window_s, self.n_steps))
        buf.add(t, feats)
        vec = buf.window(t)
        if vec is None:  # not enough pose history for a window yet
            return
        p = float(self.pose.predict_proba(vec.reshape(1, -1))[0, 1])  # column 1 = P(conceal)
        q = self.pose_probs.setdefault(pid, deque())
        q.append((t, p))
        while q[0][0] < t - POSE_HISTORY_S:
            q.popleft()

    def due(self, t: float) -> list[tuple[int, NDArray[np.uint8]]]:
        for pid in [k for k, seen in self.last_seen.items() if t - seen > FORGET_S]:
            del self.last_seen[pid]
            self.buffers.pop(pid, None)
            self.pose_probs.pop(pid, None)
        return self.video.due(t)

    def score(self, items: list[tuple[int, NDArray[np.uint8]]]) -> list[tuple[int, float]]:
        return self.video.score(items)

    def apply(self, results: list[tuple[int, float]]) -> None:
        self.video.apply(results)

    def set_threshold(self, value: float) -> None:
        """Change both rules' thresholds together: one slider, one meaning."""
        self.threshold = float(value)
        self.video.set_threshold(value)

    def set_min_clips(self, n: int) -> None:
        self.video.set_min_clips(n)

    @property
    def any_scores(self) -> bool:
        return self.video.any_scores

    def status(self, pid: int) -> str:
        """One line on how far this person is from a verdict — for demo.py --verbose."""
        pose = self.pose_score(pid)
        fused = self.window_score(pid)
        if fused is not None:
            verdict = f", FUSED {fused:.2f}"
        elif self.video.window_score(pid) is not None:
            verdict = " (video stream only: no pose yet)"
        else:
            verdict = ""  # neither stream has enough history; video.status already says so
        return (self.video.status(pid) + f", {len(self.pose_probs.get(pid, ())):3d}/{MIN_POSE_WINDOWS}"
                + " pose windows" + (f", pose {pose:.2f}" if pose is not None else "") + verdict)

    def pose_score(self, pid: int) -> float | None:
        """The pose stream's quantile over its recent scores, or None before MIN_POSE_WINDOWS."""
        q = self.pose_probs.get(pid)
        if not q or len(q) < MIN_POSE_WINDOWS:
            return None
        return float(np.quantile([p for _, p in q], self.quantile))

    def window_score(self, pid: int) -> float | None:
        """The fused number compared against `threshold`, or None while either stream is short."""
        video, pose = self.video.window_score(pid), self.pose_score(pid)
        if video is None or pose is None:
            return None
        return (video + pose) / 2

    def score_and_threshold(self, pid: int) -> tuple[float | None, float]:
        """(the number to compare, the threshold it must beat).

        The fused pair when both streams have history. When the pose stream has none — its keypoints
        drop out whenever a person is partly out of frame — it falls back to the video stream alone
        at the video stream's OWN calibrated threshold, so a missing skeleton can only cost accuracy,
        never silence the model. Averaging in a stream that has nothing to say would do the latter:
        on a camera neither stream was trained on, the pose head scores everything low and drags the
        fused number under the threshold.
        """
        fused = self.window_score(pid)
        if fused is not None:
            return fused, self.threshold
        return self.video.window_score(pid), self.video.threshold

    def state(self, pid: int) -> tuple[float | None, bool]:
        """(risk, suspicious?) — fused where possible, the video stream alone where not."""
        score, threshold = self.score_and_threshold(pid)
        if score is None:
            return self.video.state(pid)[0], False
        return score, score > threshold
