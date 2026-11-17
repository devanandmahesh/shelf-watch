"""Time-window features: classify a person's MOTION over ~1.5 s, not a single pose.

A window is the person's 34-number pose sampled at N_STEPS evenly spaced times over the last
WINDOW_S seconds. The window feature vector is:
    8 frames x 34 pose values   (272)   what the body looked like over time
  + mean |frame-to-frame change| (34)   how fast each keypoint moved
  + std over the window          (34)   how much each keypoint varied
= 340 numbers. Sampling is by TIME (not frame count), so it behaves the same at 13 fps on a
webcam and 24 fps in a video file. Training and live use the same code below.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from pose_utils import FEATURE_COLUMNS, NUM_FEATURES, FeatureVector

N_STEPS = 8
WINDOW_S = 1.5
MAX_GAP_S = 0.4  # a sample time with no pose within this long makes the window invalid

WINDOW_FEATURE_DIM = N_STEPS * NUM_FEATURES + 2 * NUM_FEATURES  # 340
WINDOW_COLUMNS: list[str] = (
    [f"t{k}_{n}" for k in range(N_STEPS) for n in FEATURE_COLUMNS]
    + [f"vel_{n}" for n in FEATURE_COLUMNS]
    + [f"std_{n}" for n in FEATURE_COLUMNS]
)


def window_features(seq: NDArray[np.float32]) -> FeatureVector:
    """(N_STEPS, 34) pose sequence -> 340-float window feature vector."""
    seq = np.asarray(seq, dtype=np.float32).reshape(N_STEPS, NUM_FEATURES)
    vel = np.abs(np.diff(seq, axis=0)).mean(axis=0)
    std = seq.std(axis=0)
    return np.concatenate([seq.ravel(), vel, std]).astype(np.float32)


class PersonBuffer:
    """Recent (time, pose) samples for ONE tracked person."""

    def __init__(self, window_s: float = WINDOW_S, n_steps: int = N_STEPS) -> None:
        self.window_s, self.n_steps = window_s, n_steps
        self._t: deque[float] = deque()
        self._f: deque[FeatureVector] = deque()

    def add(self, t: float, feats: FeatureVector) -> None:
        self._t.append(t)
        self._f.append(feats)
        while self._t and self._t[0] < t - self.window_s - 2 * MAX_GAP_S:
            self._t.popleft()
            self._f.popleft()

    def window(self, now: float) -> FeatureVector | None:
        """Window feature vector ending at ``now``, or None if there isn't enough history."""
        if len(self._t) < 2:
            return None
        times = np.fromiter(self._t, dtype=np.float64)
        targets = np.linspace(now - self.window_s, now, self.n_steps)
        idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
        if np.abs(times[idx] - targets).max() > MAX_GAP_S:
            return None
        frames = list(self._f)
        return window_features(np.stack([frames[i] for i in idx]))
