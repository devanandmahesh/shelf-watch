"""Shared pose helpers used by collect_data.py and live_inference.py.

Keeping keypoint extraction in ONE place guarantees that the features the
classifier is trained on are computed exactly the same way at inference time.

Feature layout (always exactly 34 floats):
    [nose_x, nose_y, left_eye_x, left_eye_y, ..., right_ankle_x, right_ankle_y]

Coordinates are normalised to the person's bounding box (0..1), so the model
does not care where in the frame you stand or how far you are from the camera.
Keypoints that YOLO could not see (occluded / low confidence) are set to 0.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

MODEL_NAME = "yolo11n-pose.pt"
LABEL_COLUMN = "label"

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 17
NUM_FEATURES = NUM_KEYPOINTS * 2  # 34
FEATURE_COLUMNS: list[str] = [f"{n}_{axis}" for n in KEYPOINT_NAMES for axis in ("x", "y")]

# Keypoints below this YOLO confidence are treated as "missing" (-> 0.0).
KEYPOINT_CONF_THRESHOLD = 0.3

FeatureVector = NDArray[np.float32]


def ensure_feature_vector(values: Any) -> FeatureVector:
    """Coerce anything array-like into a float32 vector of exactly 34 elements.

    Short inputs are zero-padded, long ones truncated, NaN/inf become 0.0.
    """
    flat = np.asarray(values, dtype=np.float32).ravel()
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(NUM_FEATURES, dtype=np.float32)
    n = min(flat.size, NUM_FEATURES)
    out[:n] = flat[:n]
    return out


def missing_keypoint_count(features: Any) -> int:
    """Number of keypoints whose (x, y) are both exactly 0.0 (i.e. not detected)."""
    pairs = ensure_feature_vector(features).reshape(NUM_KEYPOINTS, 2)
    return int(np.sum(np.all(pairs == 0.0, axis=1)))


def extract_features(
    result: Any, conf_threshold: float = KEYPOINT_CONF_THRESHOLD
) -> FeatureVector | None:
    """Return the 34-float feature vector for the primary person, or None.

    ``result`` is a single ultralytics ``Results`` object. The "primary" person
    is the one with the largest bounding box (closest to the camera).
    """
    boxes, kpts = result.boxes, result.keypoints
    if boxes is None or kpts is None or len(boxes) == 0 or kpts.xy is None:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return person_features(result, xyxy, int(np.argmax(areas)), conf_threshold)


def person_features(
    result: Any, xyxy: NDArray[np.float32], idx: int, conf_threshold: float = KEYPOINT_CONF_THRESHOLD
) -> FeatureVector | None:
    """Feature vector (34 floats) for person number ``idx`` in ``result``, or None.

    ``xyxy`` is ``result.boxes.xyxy`` as a numpy array. Used for per-person classification;
    extract_features() above is the single-primary-person shortcut and uses the same maths.
    """
    kpts = result.keypoints
    if kpts is None or kpts.xy is None or idx >= len(kpts.xy):
        return None

    xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    raw = kpts.xy[idx].cpu().numpy().reshape(-1, 2)
    n = min(len(raw), NUM_KEYPOINTS)
    xy[:n] = raw[:n]

    conf = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    if kpts.conf is not None:
        raw_conf = kpts.conf[idx].cpu().numpy().ravel()
        m = min(len(raw_conf), NUM_KEYPOINTS)
        conf[:m] = raw_conf[:m]
        conf[m:] = 0.0

    x1, y1, x2, y2 = (float(v) for v in xyxy[idx])
    width, height = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

    # A keypoint is valid if YOLO is confident AND it isn't the (0, 0) "not found" marker.
    valid = (conf >= conf_threshold) & ~((xy[:, 0] <= 0) & (xy[:, 1] <= 0))

    norm = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    norm[valid, 0] = (xy[valid, 0] - x1) / width
    norm[valid, 1] = (xy[valid, 1] - y1) / height

    return ensure_feature_vector(norm)
