"""Scenario model for the synthetic multi-camera CCTV simulator.

World coordinates are meters: x along the street, y = depth (0 far .. 10 near).
Each camera sees a world rectangle and maps it to its image with a simple oblique
projection (person scale grows with y). This is NOT a photorealistic simulator; it
exists so the full pipeline (tracking features → risk engine → incidents → dashboard
→ evaluation) can be exercised and measured with exact ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .skeleton import make_skeleton

PERSON_HEIGHT_M = 1.7
STRIDE_M = 1.4


@dataclass
class Keyframe:
    t: float
    x: float
    y: float


@dataclass
class Action:
    """A posture action over [t0, t1). kind: strike | fall (fall keeps lying until t1)."""
    t0: float
    t1: float
    kind: str
    target: Optional[str] = None


@dataclass
class Hold:
    t0: float
    t1: float
    label: str  # knife | bat | phone


@dataclass
class Actor:
    id: str
    color: tuple[int, int, int]  # BGR clothing color
    keyframes: list[Keyframe]
    actions: list[Action] = field(default_factory=list)
    holds: list[Hold] = field(default_factory=list)

    def __post_init__(self):
        self.keyframes.sort(key=lambda k: k.t)
        self._cum = [0.0]
        for a, b in zip(self.keyframes, self.keyframes[1:]):
            self._cum.append(self._cum[-1] + math.hypot(b.x - a.x, b.y - a.y))

    def _seg(self, t: float):
        ks = self.keyframes
        if t <= ks[0].t:
            return 0, 0.0
        for i in range(len(ks) - 1):
            if ks[i].t <= t < ks[i + 1].t:
                return i, (t - ks[i].t) / (ks[i + 1].t - ks[i].t)
        return len(ks) - 1, 0.0

    def pos(self, t: float) -> np.ndarray:
        i, f = self._seg(t)
        ks = self.keyframes
        if i >= len(ks) - 1:
            return np.array([ks[-1].x, ks[-1].y])
        a, b = ks[i], ks[i + 1]
        return np.array([a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f])

    def dist_travelled(self, t: float) -> float:
        i, f = self._seg(t)
        if i >= len(self.keyframes) - 1:
            return self._cum[-1]
        return self._cum[i] + (self._cum[i + 1] - self._cum[i]) * f

    def velocity(self, t: float, dt: float = 0.1) -> np.ndarray:
        return (self.pos(t + dt / 2) - self.pos(t - dt / 2)) / dt

    def action_at(self, t: float) -> Optional[Action]:
        for a in self.actions:
            if a.t0 <= t < a.t1:
                return a
        return None

    def hold_at(self, t: float) -> Optional[str]:
        for h in self.holds:
            if h.t0 <= t < h.t1:
                return h.label
        return None

    def active(self, t: float) -> bool:
        return self.keyframes[0].t <= t <= self.keyframes[-1].t + 1e-6 or any(
            a.t0 <= t < a.t1 for a in self.actions)


@dataclass
class GTEvent:
    type: str
    start: float
    end: float
    cameras: list[str]
    actors: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    title: str
    duration: float
    start_clock: str  # ISO8601 local time of t=0
    cameras: list[str]
    actors: list[Actor]
    events: list[GTEvent] = field(default_factory=list)
    hard_negative: bool = False
    note: str = ""

    def actor(self, aid: str) -> Actor:
        return next(a for a in self.actors if a.id == aid)


@dataclass
class SimCamera:
    id: str
    world: tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int = 960
    height: int = 540
    tint: tuple[int, int, int] = (90, 95, 100)

    def px_per_m(self, y: float) -> float:
        x0, y0, x1, y1 = self.world
        f = (y - y0) / (y1 - y0)
        return 35.0 + f * (60.0 - 35.0)

    def to_image(self, p: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.world
        u = (p[0] - x0) / (x1 - x0) * self.width
        f = (p[1] - y0) / (y1 - y0)
        v = self.height * (0.40 + f * 0.55)
        return np.array([u, v])


@dataclass
class ActorView:
    """Ground-truth rendering of one actor in one camera at one instant."""
    actor_id: str
    keypoints: np.ndarray  # (17, 2)
    height_px: float
    px_per_m: float
    color: tuple[int, int, int]
    hold: Optional[str]
    hand: np.ndarray  # wrist position holding the object
    facing: float
    lying: bool


def actor_view(scn: Scenario, actor: Actor, cam: SimCamera, t: float) -> Optional[ActorView]:
    if not actor.active(t):
        return None
    p = actor.pos(t)
    vel = actor.velocity(t)
    speed = float(np.linalg.norm(vel))
    foot = cam.to_image(p)
    ppm = cam.px_per_m(p[1])
    h = PERSON_HEIGHT_M * ppm
    if foot[0] < -0.3 * h or foot[0] > cam.width + 0.3 * h:
        return None

    act = actor.action_at(t)
    facing = 1.0 if vel[0] >= 0 else -1.0
    strike = 0.0
    fall_angle = 0.0
    if act is not None and act.kind == "strike":
        tgt = scn.actor(act.target) if act.target else None
        if tgt is not None:
            facing = 1.0 if tgt.pos(t)[0] >= p[0] else -1.0
        strike = max(0.0, math.sin(2 * math.pi * 2.5 * (t - act.t0)))
    elif act is not None and act.kind == "fall":
        fall_angle = min(90.0, (t - act.t0) / 0.7 * 90.0)
        # fall direction is fixed by velocity just before falling
        v0 = actor.velocity(act.t0 - 0.2)
        facing = 1.0 if v0[0] >= 0 else -1.0

    gait_amp = 0.0 if fall_angle > 0 else min(0.22, speed / 4.0 * 0.22)
    phase = 2 * math.pi * actor.dist_travelled(t) / STRIDE_M
    kps = make_skeleton(foot, h, gait_phase=phase, gait_amp=gait_amp, strike=strike,
                        strike_dir=facing, fall_angle_deg=fall_angle, fall_dir=facing)
    hand = kps[10] if facing > 0 else kps[9]
    return ActorView(actor.id, kps, h, ppm, actor.color, actor.hold_at(t), hand, facing,
                     fall_angle >= 60.0)
