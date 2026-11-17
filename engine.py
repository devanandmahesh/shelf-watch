"""What demo.py and web_app.py share: when to alert, and how to draw the result.

The models themselves live in pose_utils / window_utils / video_utils / fused_utils. This module
holds the two things both front ends must agree on:

* AlertTracker — when a person's alert is raised and when it is dropped again. Stream models
  (video, fused) judge a person from their last ~10 s, so their alert clears once the risk falls
  back; the older pose models alert on a streak of frames and latch, which is how they have always
  behaved. Both rules live here so a fix to one front end cannot miss the other.
* PersonView and draw_overlay — one drawing of the annotated frame, so the browser view and the
  OpenCV window show the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GREEN, ORANGE, RED, GREY, WHITE = (0, 200, 0), (0, 165, 255), (0, 0, 255), (160, 160, 160), (255, 255, 255)


@dataclass
class PersonView:
    """One tracked person as both front ends need them: to draw, and to send as JSON."""

    pid: int
    box: tuple[int, int, int, int]
    label: str
    risk: float          # -1.0 while the model is still collecting history for this person
    threshold: float
    flagged: bool        # an alert is standing for this person (drawn red)
    alerting: bool = False   # meets the alert condition right now (drawn orange)
    status: str = ""     # the per-person progress line, when the model can produce one

    @property
    def collecting(self) -> bool:
        return self.risk < 0

    def as_json(self) -> dict[str, object]:
        return {"id": self.pid, "box": list(self.box), "label": self.label,
                "risk": None if self.collecting else round(self.risk, 4),
                "threshold": round(self.threshold, 4), "flagged": self.flagged,
                "alerting": self.alerting, "status": self.status, "collecting": self.collecting}


class AlertTracker:
    """Raises and drops per-person alerts, and remembers everyone ever flagged.

    `need` is how many consecutive hits raise an alert (1 for the stream models, whose own window
    already spans seconds). `latch` keeps an alert on for the rest of the run once raised, which is
    what the pose models have always done; the stream models pass latch=False so an alert follows
    the evidence and goes away with it.
    """

    def __init__(self, need: int = 1, latch: bool = False) -> None:
        self.need, self.latch = max(1, need), latch
        self.streak: dict[int, int] = {}
        self.active: dict[int, int] = {}  # person id -> frame their current alert was raised
        self.ever: dict[int, int] = {}    # person id -> frame they were first ever flagged

    def update(self, pid: int, hit: bool, frame_no: int) -> str | None:
        """Feed one frame's verdict for one person; returns "raised", "cleared" or None."""
        self.streak[pid] = self.streak.get(pid, 0) + 1 if hit else 0
        if self.streak[pid] >= self.need and pid not in self.active:
            self.active[pid] = self.ever.setdefault(pid, frame_no)
            return "raised"
        if not hit and pid in self.active and not self.latch:
            del self.active[pid]
            return "cleared"
        return None

    def reset(self) -> None:
        """Forget everything, including who was ever flagged (the web UI's "clear alerts")."""
        self.streak.clear()
        self.active.clear()
        self.ever.clear()

    def forget(self, pid: int) -> None:
        """Drop a person who is gone, so a recycled track id starts clean. `ever` is kept."""
        self.streak.pop(pid, None)
        self.active.pop(pid, None)

    def flagged(self, pid: int) -> bool:
        return pid in self.active


def put(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int],
        thick: int = 2) -> None:
    """Text with a dark outline so it's readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(view: np.ndarray, people: list[PersonView], fps: float, alerted: list[int],
                 risk_label: bool) -> np.ndarray:
    """Boxes, labels and the footer, drawn on `view` in place (and returned for convenience)."""
    height = view.shape[0]
    for p in people:
        x1, y1, x2, y2 = p.box
        if p.collecting:
            cv2.rectangle(view, (x1, y1), (x2, y2), GREY, 1)
            put(view, f"ID {p.pid}  collecting...", (x1, max(20, y1 - 8)), 0.55, (200, 200, 200), 1)
            continue
        color = RED if p.flagged else (ORANGE if p.alerting else GREEN)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 3 if p.flagged else 2)
        text = f"theft risk {p.risk * 100:.0f}%" if risk_label else f"{p.label} {p.risk * 100:.0f}%"
        put(view, f"ID {p.pid}  {text}" + ("  SUSPECT" if p.flagged else ""),
            (x1, max(20, y1 - 8)), 0.6, color)
    put(view, f"people: {len(people)}   fps: {fps:.0f}", (10, height - 12), 0.6, WHITE, 1)
    if alerted:
        put(view, "SHOPLIFTING SUSPECTED: ID " + ", ".join(str(k) for k in alerted),
            (10, height - 42), 0.8, RED)
    return view
