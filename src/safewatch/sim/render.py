"""Renders synthetic CCTV frames and produces noisy 'perception' outputs from ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from ..types import SKELETON, FrameObs, ObjectObs, PersonObs
from .scenario import ActorView, Scenario, SimCamera, actor_view

OBJ_LEN_M = {"knife": 0.28, "bat": 0.8, "phone": 0.15}


def render_frame(cam: SimCamera, views: list[ActorView], t: float) -> np.ndarray:
    W, H = cam.width, cam.height
    img = np.empty((H, W, 3), np.uint8)
    img[:] = cam.tint
    horizon = int(H * 0.36)
    img[:horizon] = (np.array(cam.tint) * 0.6 + 60).astype(np.uint8)
    for i in range(6):  # ground perspective lines
        y = int(horizon + (H - horizon) * (i / 5) ** 1.4)
        cv2.line(img, (0, y), (W, y), (75, 80, 85), 1)
    # static props
    cv2.rectangle(img, (40, horizon - 90), (160, horizon), (70, 70, 80), -1)
    cv2.rectangle(img, (W - 220, horizon - 120), (W - 60, horizon), (60, 65, 75), -1)
    for v in sorted(views, key=lambda v: v.keypoints[15, 1]):
        _draw_actor(img, v)
    cv2.putText(img, f"{t:6.1f}s", (W - 110, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    return img


def _draw_actor(img: np.ndarray, v: ActorView) -> None:
    k = v.keypoints.astype(int)
    th = max(2, int(v.height_px / 18))
    limb = (50, 50, 55)
    for a, b in SKELETON:
        color = v.color if (a, b) in [(5, 6), (5, 11), (6, 12), (11, 12)] else limb
        cv2.line(img, tuple(k[a]), tuple(k[b]), color, th + (2 if color == v.color else 0),
                 cv2.LINE_AA)
    torso = np.array([k[5], k[6], k[12], k[11]])
    cv2.fillConvexPoly(img, torso, v.color, cv2.LINE_AA)
    cv2.circle(img, tuple(k[0]), max(3, int(v.height_px * 0.06)), (150, 180, 210), -1, cv2.LINE_AA)
    if v.hold:
        _draw_object(img, v)


def _draw_object(img: np.ndarray, v: ActorView) -> None:
    L = OBJ_LEN_M[v.hold] * v.px_per_m
    h = v.hand
    d = np.array([v.facing * 0.7, 0.7]) if v.hold != "bat" else np.array([v.facing * 0.3, -0.95])
    e = h + d / np.linalg.norm(d) * L
    if v.hold == "knife":
        cv2.line(img, tuple(h.astype(int)), tuple(e.astype(int)), (215, 215, 220), 2, cv2.LINE_AA)
    elif v.hold == "bat":
        cv2.line(img, tuple(h.astype(int)), tuple(e.astype(int)), (40, 90, 140), 4, cv2.LINE_AA)
    else:
        cv2.rectangle(img, tuple((h - 3).astype(int)), tuple((h + 3).astype(int)), (20, 20, 20), -1)


def object_bbox(v: ActorView) -> tuple[float, float, float, float]:
    L = OBJ_LEN_M[v.hold] * v.px_per_m
    h = v.hand
    d = np.array([v.facing * 0.7, 0.7]) if v.hold != "bat" else np.array([v.facing * 0.3, -0.95])
    e = h + d / np.linalg.norm(d) * L
    pad = 3
    return (min(h[0], e[0]) - pad, min(h[1], e[1]) - pad, max(h[0], e[0]) + pad, max(h[1], e[1]) + pad)


@dataclass
class NoiseModel:
    """Perception error model. Defaults are rough guesses meant to stress the risk engine,
    not measurements of any real detector (replace with measured rates in Phase 1)."""
    bbox_jitter_px: float = 2.0
    kp_jitter_px: float = 2.0
    miss_prob: float = 0.03
    knife_detect_prob: float = 0.55
    bat_detect_prob: float = 0.75
    phone_as_knife_prob: float = 0.08
    random_knife_fp_per_frame: float = 0.002


class SimulatedPerception:
    """Converts ground-truth ActorViews into noisy detector/tracker/pose outputs."""

    def __init__(self, noise: Optional[NoiseModel] = None, seed: int = 0):
        self.noise = noise or NoiseModel()
        self.rng = np.random.default_rng(seed)
        self._track_ids: dict[str, int] = {}
        self._last_seen: dict[str, float] = {}
        self._next_id = 1

    def _track_id(self, actor_id: str, t: float) -> int:
        if actor_id not in self._track_ids or t - self._last_seen.get(actor_id, -1e9) > 1.0:
            self._track_ids[actor_id] = self._next_id
            self._next_id += 1
        self._last_seen[actor_id] = t
        return self._track_ids[actor_id]

    def process(self, cam_id: str, frame_idx: int, t: float, size: tuple[int, int],
                views: list[ActorView]) -> FrameObs:
        n, rng = self.noise, self.rng
        W, H = size
        obs = FrameObs(cam_id, frame_idx, t, W, H)
        for v in views:
            kp = v.keypoints
            x1, y1 = kp[:, 0].min(), kp[:, 1].min() - v.height_px * 0.05
            x2, y2 = kp[:, 0].max(), kp[:, 1].max()
            vis_w = min(x2, W) - max(x1, 0)
            if vis_w <= 0.4 * (x2 - x1):
                continue
            if rng.random() < n.miss_prob:
                continue
            j = rng.normal(0, n.bbox_jitter_px, 4)
            bbox = (max(0, x1 + j[0]), max(0, y1 + j[1]), min(W, x2 + j[2]), min(H, y2 + j[3]))
            kps = np.zeros((17, 3))
            kps[:, :2] = kp + rng.normal(0, n.kp_jitter_px, (17, 2))
            kps[:, 2] = np.clip(rng.normal(0.85, 0.08, 17), 0.3, 1.0)
            obs.persons.append(PersonObs(self._track_id(v.actor_id, t), bbox,
                                         float(np.clip(rng.normal(0.85, 0.05), 0.5, 1)), kps))
            if v.hold == "knife" and rng.random() < n.knife_detect_prob:
                obs.objects.append(ObjectObs("knife", object_bbox(v), float(rng.uniform(0.35, 0.75))))
            elif v.hold == "bat" and rng.random() < n.bat_detect_prob:
                obs.objects.append(ObjectObs("bat", object_bbox(v), float(rng.uniform(0.4, 0.85))))
            elif v.hold == "phone" and rng.random() < n.phone_as_knife_prob:
                obs.objects.append(ObjectObs("knife", object_bbox(v), float(rng.uniform(0.25, 0.5))))
        if rng.random() < n.random_knife_fp_per_frame:
            cx, cy = rng.uniform(0, W), rng.uniform(H * 0.4, H)
            obs.objects.append(ObjectObs("knife", (cx - 8, cy - 8, cx + 8, cy + 8),
                                         float(rng.uniform(0.25, 0.45))))
        return obs


@dataclass
class SimFrame:
    camera_id: str
    frame_idx: int
    ts: float
    image: np.ndarray
    views: list[ActorView]


def iter_camera(scn: Scenario, cam: SimCamera, fps: float = 10.0,
                render: bool = True) -> Iterator[SimFrame]:
    n = int(scn.duration * fps)
    for i in range(n):
        t = i / fps
        views = [v for a in scn.actors if (v := actor_view(scn, a, cam, t)) is not None]
        img = render_frame(cam, views, t) if render else np.zeros((1, 1, 3), np.uint8)
        yield SimFrame(cam.id, i, t, img, views)
