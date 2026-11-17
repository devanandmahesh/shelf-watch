"""Rule-based risk scoring from tracked pose keypoints (no ML, no camera needed).

Rule: a wrist entering a shelf zone ("reach") followed, within a few seconds, by
the same-or-other wrist staying close to a hip ("pocket") adds a large score.
Scores decay over time so isolated reaches never accumulate into an alert.

This is a HEURISTIC: arms hanging at the sides can look like "hand at pocket".
Tune the thresholds on your own footage and treat every alert as "needs a human
to look", never as proof of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class RiskConfig:
    shelf_zones: list[NDArray[np.int32]]  # pixel-space polygons
    reach_points: float = 10.0
    pocket_points: float = 40.0
    decay_per_s: float = 2.0
    # reach(10) + pocket(40) - worst-case decay over the 4 s window (8) = 42, so 40 always fires
    alert_threshold: float = 40.0
    pocket_window_s: float = 4.0  # pocket must follow a reach within this long
    pocket_hold_s: float = 0.3  # wrist must stay near the hip this long
    pocket_ratio: float = 0.3  # "near hip" = within this fraction of torso length
    # "hip": only a wrist near a hip counts as concealing.
    # "torso": a wrist near a hip OR inside the chest-to-hips body area also counts
    #          (jacket front, hoodie pouch, inside coat, bag held at the body).
    conceal_region: str = "hip"
    torso_margin: float = 1.2  # grow the shoulders-hips quad by this factor
    bag_hold_s: float = 0.5  # accumulated seconds with a wrist inside a bag box before "conceal"
    alert_cooldown_s: float = 10.0
    alert_display_s: float = 4.0
    kp_conf: float = 0.3
    stale_after_s: float = 5.0


@dataclass
class Event:
    t: float
    track_id: int
    kind: str  # "reach" | "conceal" | "alert"
    score: float


@dataclass
class PersonState:
    last_update: float
    score: float = 0.0
    in_zone: list[bool] = field(default_factory=lambda: [False, False])
    pocket_since: list[float | None] = field(default_factory=lambda: [None, None])
    reach_time: float = float("-inf")
    reach_pending: bool = False
    bag_dwell: float = 0.0
    last_alert: float = float("-inf")
    alert_until: float = float("-inf")


class RiskEngine:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.states: dict[int, PersonState] = {}

    # ---- geometry helpers -------------------------------------------------
    def _in_shelf(self, pt: NDArray[np.float32]) -> bool:
        p = (float(pt[0]), float(pt[1]))
        return any(cv2.pointPolygonTest(z, p, False) >= 0 for z in self.cfg.shelf_zones)

    @staticmethod
    def _torso_length(xy: NDArray[np.float32], vis: NDArray[np.bool_]) -> float | None:
        if not (vis[L_SHOULDER] and vis[R_SHOULDER] and vis[L_HIP] and vis[R_HIP]):
            return None
        length = float(np.linalg.norm((xy[L_SHOULDER] + xy[R_SHOULDER]) / 2 - (xy[L_HIP] + xy[R_HIP]) / 2))
        return length if length > 1.0 else None

    def _in_torso(self, pt: NDArray[np.float32], xy: NDArray[np.float32], vis: NDArray[np.bool_]) -> bool:
        """True if pt lies inside the (slightly enlarged) shoulders-to-hips quadrilateral."""
        idx = (L_SHOULDER, R_SHOULDER, R_HIP, L_HIP)
        if not all(vis[i] for i in idx):
            return False
        quad = xy[list(idx)].astype(np.float32)
        centre = quad.mean(axis=0)
        quad = (centre + (quad - centre) * self.cfg.torso_margin).astype(np.float32)
        return cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0

    # ---- main update ------------------------------------------------------
    @staticmethod
    def _in_any_box(pt: NDArray[np.float32], boxes: list[NDArray[np.float32]], grow: float = 0.1) -> bool:
        """True if pt is inside any (xyxy) box, each enlarged by `grow` of its size."""
        for x1, y1, x2, y2 in boxes:
            gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
            if x1 - gx <= pt[0] <= x2 + gx and y1 - gy <= pt[1] <= y2 + gy:
                return True
        return False

    def update(
        self,
        track_id: int,
        xy: NDArray[np.float32],
        conf: NDArray[np.float32],
        t: float,
        bags: list[NDArray[np.float32]] | None = None,
    ) -> list[Event]:
        """Feed one person's keypoints (17x2 pixels, 17 confidences) at time t (seconds).

        ``bags`` (optional) are xyxy boxes of detected handbags/backpacks in this frame;
        a wrist inside one counts as concealing, just like a wrist at the hip.
        """
        cfg = self.cfg
        st = self.states.setdefault(track_id, PersonState(last_update=t))
        dt = max(0.0, t - st.last_update)
        st.score = max(0.0, st.score - cfg.decay_per_s * dt)
        st.last_update = t
        vis = conf >= cfg.kp_conf
        events: list[Event] = []

        # 1) Reach: a wrist ENTERS a shelf zone. If the wrist is hidden (e.g. behind the
        #    shelf) we keep its previous state instead of guessing.
        for side, wrist in enumerate((L_WRIST, R_WRIST)):
            if not vis[wrist]:
                continue
            inside = self._in_shelf(xy[wrist])
            if inside and not st.in_zone[side]:
                st.reach_time, st.reach_pending = t, True
                st.score += cfg.reach_points
                events.append(Event(t, track_id, "reach", st.score))
            st.in_zone[side] = inside

        # 2) Pocket: a wrist stays near its hip, having arrived AFTER the reach.
        torso = self._torso_length(xy, vis)
        for side, (wrist, hip) in enumerate(((L_WRIST, L_HIP), (R_WRIST, R_HIP))):
            near = bool(
                torso is not None and vis[wrist] and vis[hip]
                and float(np.linalg.norm(xy[wrist] - xy[hip])) <= cfg.pocket_ratio * torso
            )
            if not near and cfg.conceal_region == "torso" and vis[wrist]:
                near = self._in_torso(xy[wrist], xy, vis)
            if not near:
                st.pocket_since[side] = None
                continue
            if st.pocket_since[side] is None:
                st.pocket_since[side] = t
            since = st.pocket_since[side]
            if (
                st.reach_pending
                and since > st.reach_time
                and t - since >= cfg.pocket_hold_s
                and t - st.reach_time <= cfg.pocket_window_s
            ):
                st.reach_pending = False
                st.score += cfg.pocket_points
                events.append(Event(t, track_id, "conceal", st.score))

        if st.reach_pending and t - st.reach_time > cfg.pocket_window_s:
            st.reach_pending = False

        # 2b) Bag cue: a wrist inside a detected handbag/backpack box. Detections flicker,
        #     so use accumulated dwell time (which drains while the wrist is outside).
        in_bag = bool(bags) and any(
            vis[w] and self._in_any_box(xy[w], bags or []) for w in (L_WRIST, R_WRIST)
        )
        st.bag_dwell = st.bag_dwell + dt if in_bag else max(0.0, st.bag_dwell - dt)
        if st.bag_dwell >= cfg.bag_hold_s:
            st.bag_dwell = 0.0
            st.score += cfg.pocket_points
            events.append(Event(t, track_id, "conceal", st.score))

        # 3) Alert
        if st.score >= cfg.alert_threshold and t - st.last_alert >= cfg.alert_cooldown_s:
            st.last_alert = t
            st.alert_until = t + cfg.alert_display_s
            events.append(Event(t, track_id, "alert", st.score))

        return events

    def prune(self, visible_ids: set[int], t: float) -> None:
        """Forget tracks not seen for a while."""
        for tid in [k for k, s in self.states.items()
                    if k not in visible_ids and t - s.last_update > self.cfg.stale_after_s]:
            del self.states[tid]

    def active_alert_id(self, t: float) -> int | None:
        """Track id of the highest-scoring person whose alert is still being displayed."""
        live = [(s.score, k) for k, s in self.states.items() if s.alert_until > t]
        return max(live)[1] if live else None
