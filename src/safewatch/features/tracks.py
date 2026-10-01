"""Per-track kinematic / posture / object-holding features.

Units: distances are normalized by the person's (upright) body height H in pixels, so
speeds are in H/s. Walking ≈ 0.8 H/s, running ≳ 2 H/s (≈1.4 m/s and ≈4 m/s for a 1.7 m person).
Vertical image distances are multiplied by `depth_factor` to compensate for ground-plane
foreshortening (a per-camera calibration value; a homography would replace this in deployment).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry import point_in_polygon
from ..types import KP, WEAPON_LABELS, ObjectObs, PersonObs

HISTORY_S = 12.0


@dataclass
class Sample:
    ts: float
    foot: np.ndarray
    bbox: tuple
    kps: Optional[np.ndarray]
    torso_angle: float  # deg from vertical (nan if unknown)
    lying: bool
    weapon_conf: float
    weapon_label: Optional[str]
    wrist_rel: Optional[np.ndarray]  # (2, 2) wrists relative to shoulder center, in H units
    truncated: bool = False  # bbox touches the image border (position/size unreliable)


@dataclass
class TrackState:
    camera_id: str
    track_id: int
    first_ts: float
    hist: deque = field(default_factory=lambda: deque())
    upright_heights: deque = field(default_factory=lambda: deque(maxlen=60))
    last_ts: float = 0.0
    fall_onset_ts: Optional[float] = None
    lying_since: Optional[float] = None
    flee_onset_ts: Optional[float] = None
    weapon_first_ts: Optional[float] = None
    weapon_frames: int = 0
    weapon_label: Optional[str] = None
    zone_entry: dict = field(default_factory=dict)  # zone_id -> entry ts
    zone_path: dict = field(default_factory=dict)  # zone_id -> list of foot points
    zone_last_in: dict = field(default_factory=dict)
    appearance: Optional[np.ndarray] = None
    color_name: Optional[str] = None
    entered_side: Optional[str] = None
    lost: bool = False

    # ---------------------------------------------------------------- basics
    @property
    def scale(self) -> float:
        if self.upright_heights:
            return float(np.median(self.upright_heights))
        b = self.hist[-1].bbox
        return float(max(b[3] - b[1], b[2] - b[0], 1.0))

    @property
    def foot(self) -> np.ndarray:
        return self.hist[-1].foot

    def window(self, seconds: float) -> list[Sample]:
        t0 = self.last_ts - seconds
        return [s for s in self.hist if s.ts >= t0]

    def velocity(self, span: float = 0.6, depth_factor: float = 1.6) -> np.ndarray:
        """Velocity in H/s (ground-compensated)."""
        if len(self.hist) < 2:
            return np.zeros(2)
        cur = self.hist[-1]
        ref = cur
        for s in reversed(self.hist):
            ref = s
            if cur.ts - s.ts >= span:
                break
        dt = cur.ts - ref.ts
        if dt <= 1e-6:
            return np.zeros(2)
        d = (cur.foot - ref.foot) / self.scale
        d[1] *= depth_factor
        return d / dt

    def speed(self, depth_factor: float = 1.6) -> float:
        return float(np.linalg.norm(self.velocity(depth_factor=depth_factor)))

    def limb_speed(self, seconds: float = 1.0) -> float:
        """Mean wrist speed relative to the shoulders (H/s) over the window."""
        w = [s for s in self.window(seconds) if s.wrist_rel is not None]
        if len(w) < 3:
            return 0.0
        sp = []
        for a, b in zip(w, w[1:]):
            dt = b.ts - a.ts
            if dt > 1e-6:
                sp.append(float(np.max(np.linalg.norm(b.wrist_rel - a.wrist_rel, axis=1))) / dt)
        return float(np.mean(sp)) if sp else 0.0

    def reach_towards(self, direction: np.ndarray, seconds: float = 1.0) -> tuple[float, float]:
        """(mean, std) of max wrist extension along `direction` (unit vector, image coords)."""
        w = [s for s in self.window(seconds) if s.wrist_rel is not None]
        if len(w) < 3:
            return 0.0, 0.0
        r = [float(np.max(s.wrist_rel @ direction)) for s in w]
        return float(np.mean(r)), float(np.std(r))

    def weapon_presence(self, seconds: float = 2.0) -> float:
        w = self.window(seconds)
        if not w:
            return 0.0
        return float(sum(s.weapon_conf for s in w) / len(w))

    def lying_duration(self) -> float:
        return 0.0 if self.lying_since is None else self.last_ts - self.lying_since

    @property
    def truncated(self) -> bool:
        return bool(self.hist) and self.hist[-1].truncated

    def is_lying(self) -> bool:
        return bool(self.hist) and self.hist[-1].lying


def _torso_angle(kps: Optional[np.ndarray]) -> float:
    if kps is None:
        return float("nan")
    idx = [KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]
    if np.any(kps[idx, 2] < 0.3):
        return float("nan")
    sh = (kps[KP["l_shoulder"], :2] + kps[KP["r_shoulder"], :2]) / 2
    hp = (kps[KP["l_hip"], :2] + kps[KP["r_hip"], :2]) / 2
    v = sh - hp
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1] + 1e-9)))


def _is_lying(kps: Optional[np.ndarray], ang: float, w: float, h: float, trunc: bool) -> bool:
    """Lying posture. Not judged on border-clipped bodies (e.g. only the upper body visible)."""
    if trunc:
        return False
    if not np.isnan(ang):
        return ang > 55
    if kps is not None:
        # torso keypoints unreliable: decide by box shape only if the legs are actually visible
        legs = kps[[KP["l_knee"], KP["r_knee"], KP["l_ankle"], KP["r_ankle"]], 2]
        return bool(np.any(legs >= 0.3)) and w > 1.25 * h
    return w > 1.25 * h


def _wrists_rel(kps: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    if kps is None:
        return None
    idx = [KP["l_shoulder"], KP["r_shoulder"], KP["l_wrist"], KP["r_wrist"]]
    if np.any(kps[idx, 2] < 0.3):
        return None
    sh = (kps[KP["l_shoulder"], :2] + kps[KP["r_shoulder"], :2]) / 2
    return np.stack([kps[KP["l_wrist"], :2] - sh, kps[KP["r_wrist"], :2] - sh]) / scale


def associate_weapon(p: PersonObs, objects: list[ObjectObs], scale: float) -> tuple[float, Optional[str]]:
    """Best weapon-candidate object attached to this person's hands (or upper body)."""
    best, label = 0.0, None
    hands = []
    if p.keypoints is not None:
        for i in (KP["l_wrist"], KP["r_wrist"]):
            if p.keypoints[i, 2] >= 0.3:
                hands.append(p.keypoints[i, :2])
    x1, y1, x2, y2 = p.bbox
    for o in objects:
        if o.label not in WEAPON_LABELS:
            continue
        ox1, oy1, ox2, oy2 = o.bbox
        if hands:
            # distance from either hand to the object's box
            d = min(float(np.hypot(max(ox1 - h[0], 0, h[0] - ox2), max(oy1 - h[1], 0, h[1] - oy2)))
                    for h in hands)
            ok = d <= 0.12 * scale
        else:
            c = o.center
            ok = x1 - 0.15 * scale <= c[0] <= x2 + 0.15 * scale and y1 <= c[1] <= y1 + 0.75 * (y2 - y1)
        if ok and o.conf > best:
            best, label = o.conf, o.label
    return best, label


class TrackRegistry:
    """Maintains TrackState for one camera."""

    def __init__(self, camera_id: str, depth_factor: float = 1.6, lost_after_s: float = 1.5):
        self.camera_id = camera_id
        self.depth_factor = depth_factor
        self.lost_after_s = lost_after_s
        self.tracks: dict[int, TrackState] = {}

    def active(self) -> list[TrackState]:
        return [t for t in self.tracks.values() if not t.lost]

    def update(self, ts: float, persons: list[PersonObs], objects: list[ObjectObs],
               zones=(), hour: int = 12, frame_size: Optional[tuple[int, int]] = None
               ) -> tuple[list[TrackState], list[TrackState]]:
        """Returns (new_tracks, lost_tracks) for this frame."""
        new, lost = [], []
        for p in persons:
            st = self.tracks.get(p.track_id)
            if st is None:
                st = TrackState(self.camera_id, p.track_id, ts)
                self.tracks[p.track_id] = st
                new.append(st)
            st.lost = False
            x1, y1, x2, y2 = p.bbox
            ang = _torso_angle(p.keypoints)
            w, h = x2 - x1, y2 - y1
            trunc = False
            if frame_size:
                W, H = frame_size
                mx, my = max(3, 0.012 * W), max(3, 0.012 * H)
                trunc = x1 <= mx or x2 >= W - mx or y2 >= H - my
            lying = _is_lying(p.keypoints, ang, w, h, trunc)
            if not lying and (np.isnan(ang) or ang < 25) and h > 1.6 * w:
                st.upright_heights.append(h)
            # provisional sample so that scale/window work on the first frame
            if not st.hist:
                st.hist.append(Sample(ts, p.foot, p.bbox, p.keypoints, ang, lying, 0.0, None, None))
            scale = st.scale
            wconf, wlabel = associate_weapon(p, objects, scale)
            foot = p.foot if not lying else np.array([(x1 + x2) / 2, y2])
            s = Sample(ts, foot, p.bbox, p.keypoints, ang, lying, wconf, wlabel,
                       _wrists_rel(p.keypoints, scale), trunc)
            if len(st.hist) == 1 and st.hist[0].ts == ts:
                st.hist[0] = s
            else:
                st.hist.append(s)
            while st.hist and st.hist[0].ts < ts - HISTORY_S:
                st.hist.popleft()
            st.last_ts = ts
            self._update_events(st, ts)
            self._update_zones(st, ts, zones, hour)
        for st in self.tracks.values():
            if not st.lost and ts - st.last_ts > self.lost_after_s:
                st.lost = True
                lost.append(st)
        # forget very old tracks
        for tid in [k for k, v in self.tracks.items() if v.lost and ts - v.last_ts > 60]:
            del self.tracks[tid]
        return new, lost

    def _update_events(self, st: TrackState, ts: float) -> None:
        s = st.hist[-1]
        # fall: currently lying and was upright within the last 2.5 s
        if s.lying:
            if st.lying_since is None:
                st.lying_since = ts
                upright = [x for x in st.window(2.5) if not x.lying and
                           (np.isnan(x.torso_angle) or x.torso_angle < 35)]
                if upright:
                    st.fall_onset_ts = ts
        else:
            st.lying_since = None
        # weapon holding
        if s.weapon_conf > 0:
            st.weapon_frames += 1
            st.weapon_label = s.weapon_label
            if st.weapon_first_ts is None and st.weapon_frames >= 2:
                st.weapon_first_ts = ts
        # sudden acceleration (flee onset candidate): slow → fast within 1.5 s
        if len(st.hist) >= 3:
            v_now = st.speed(self.depth_factor)
            if v_now > 1.6:
                older = [x for x in st.window(2.0) if ts - x.ts >= 1.0]
                if older:
                    ref = older[0]
                    dt = ts - ref.ts
                    d = (s.foot - ref.foot) / st.scale
                    d[1] *= self.depth_factor
                    prev_speed = float(np.linalg.norm(d)) / max(dt, 1e-6)
                    if prev_speed < 1.0 and (st.flee_onset_ts is None or ts - st.flee_onset_ts > 5):
                        st.flee_onset_ts = ts

    def _update_zones(self, st: TrackState, ts: float, zones, hour: int) -> None:
        for z in zones:
            inside = z.active_at(hour) and point_in_polygon(st.foot, z.polygon)
            if inside:
                if z.id not in st.zone_entry:
                    st.zone_entry[z.id] = ts
                    st.zone_path[z.id] = []
                st.zone_path[z.id].append(st.foot.copy())
                st.zone_last_in[z.id] = ts
            elif z.id in st.zone_entry and ts - st.zone_last_in.get(z.id, ts) > 1.5:
                # left the zone (short excursions under 1.5 s are tolerated)
                st.zone_entry.pop(z.id, None)
                st.zone_path.pop(z.id, None)
