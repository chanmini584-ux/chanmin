"""Per-camera analysis pipeline and runners (synthetic scenarios, video files, RTSP)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .annotate import annotate
from .config import Camera, Site
from .features.tracks import TrackRegistry
from .incidents.appearance import color_hist, color_name, torso_region
from .incidents.manager import IncidentManager
from .risk.engine import RiskEngine, SituationTracker
from .risk.policy import Policy
from .types import FrameObs

Perception = Callable[[str, int, float, np.ndarray, object], FrameObs]


@dataclass
class PipelineStats:
    frames: int = 0
    proc_ms_sum: float = 0.0
    proc_ms_max: float = 0.0
    last_ts: float = 0.0
    situations: int = 0

    def as_dict(self):
        return {"frames": self.frames,
                "proc_ms_avg": round(self.proc_ms_sum / self.frames, 2) if self.frames else 0,
                "proc_ms_max": round(self.proc_ms_max, 2), "last_ts": round(self.last_ts, 2)}


class CameraPipeline:
    def __init__(self, cam: Camera, policy: Policy, manager: IncidentManager, perception: Perception,
                 clock_base: datetime, blur_heads: bool = False, score_log: Optional[list] = None):
        self.cam = cam
        self.p = policy
        self.manager = manager
        self.perception = perception
        self.clock_base = clock_base
        self.blur_heads = blur_heads
        depth = float(cam.analysis.get("depth_factor", policy.f("depth_factor", 1.6)))
        self.registry = TrackRegistry(cam.id, depth)
        self.engine = RiskEngine(policy, cam, self.registry)
        self.tracker = SituationTracker(policy)
        self.stats = PipelineStats()
        self.latest_jpeg: Optional[bytes] = None
        self.score_log = score_log  # optional list for evaluation: (cam, ts, type, actors, score)
        self.lock = threading.Lock()

    def hour(self, ts: float) -> int:
        return (self.clock_base.hour + int((self.clock_base.minute * 60 + self.clock_base.second + ts) // 3600)) % 24

    def process(self, image: np.ndarray, ts: float, frame_idx: int, aux=None) -> FrameObs:
        t0 = time.perf_counter()
        obs = self.perception(self.cam.id, frame_idx, ts, image, aux)
        hour = self.hour(ts)
        H, W = image.shape[:2]
        new, lost = self.registry.update(ts, obs.persons, obs.objects, self.cam.zones, hour, (W, H))
        if frame_idx % 3 == 0 or new:
            self._update_appearance(image, obs)
        evals = self.engine.step(ts, hour)
        changed = self.tracker.update(ts, evals)
        active = self.tracker.active()
        if self.score_log is not None:
            for s in self.tracker.items.values():
                self.score_log.append((self.cam.id, ts, s.type, s.actors, s.score, s.incident_id))
        clock = (self.clock_base.timestamp() + ts)
        annotated = annotate(image, obs, self.registry, active, self.cam.zones,
                             self.cam.id,  # OpenCV fonts are ASCII-only; names are shown in the dashboard
                             datetime.fromtimestamp(clock, self.clock_base.tzinfo).strftime("%Y-%m-%d %H:%M:%S"),
                             self.blur_heads)
        self.manager.on_frame(self.cam.id, ts, annotated, changed, active,
                              self.registry.tracks, new, lost, W)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self.lock:
            self.latest_jpeg = buf.tobytes() if ok else self.latest_jpeg
        dt = (time.perf_counter() - t0) * 1000
        self.stats.frames += 1
        self.stats.proc_ms_sum += dt
        self.stats.proc_ms_max = max(self.stats.proc_ms_max, dt)
        self.stats.last_ts = ts
        self.stats.situations = len(active)
        return obs

    def _update_appearance(self, image, obs: FrameObs) -> None:
        for p in obs.persons:
            st = self.registry.tracks.get(p.track_id)
            if st is None or st.truncated:
                continue
            patch = torso_region(image, p)
            if patch is None:
                continue
            h = color_hist(patch)
            st.appearance = h if st.appearance is None else 0.7 * st.appearance + 0.3 * h
            st.color_name = color_name(patch)


# ---------------------------------------------------------------- synthetic scenario runner
def run_scenario(scn, site: Site, policy: Policy, manager: IncidentManager, *, seed: int = 0,
                 noise=None, fps: float = 10.0, realtime: bool = False, speed: float = 1.0,
                 stop: Optional[threading.Event] = None, pipelines_out: Optional[dict] = None,
                 score_log: Optional[list] = None, render: bool = True, blur_heads: bool = False):
    from .sim.render import SimulatedPerception, iter_camera
    from .sim.scenario import SimCamera

    base = datetime.fromisoformat(scn.start_clock)
    pipes, iters = {}, {}
    for i, cid in enumerate(scn.cameras):
        cam = site.camera(cid)
        sim_cam = SimCamera(cid, tuple(cam.sim["world"]))
        sp = SimulatedPerception(noise=noise, seed=seed * 100 + i)

        def perception(cam_id, idx, ts, image, aux, _sp=sp, _c=sim_cam):
            return _sp.process(cam_id, idx, ts, (_c.width, _c.height), aux)

        pipes[cid] = CameraPipeline(cam, policy, manager, perception, base, blur_heads, score_log)
        iters[cid] = iter_camera(scn, sim_cam, fps=fps, render=True)
        manager.clock_base[cid] = base
    if pipelines_out is not None:
        pipelines_out.update(pipes)
    t_start = time.monotonic()
    n = int(scn.duration * fps)
    for k in range(n):
        if stop is not None and stop.is_set():
            break
        for cid, it in iters.items():
            f = next(it)
            pipes[cid].process(f.image, f.ts, f.frame_idx, f.views)
        if realtime:
            target = t_start + (k + 1) / fps / max(speed, 1e-3)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    return pipes


# ---------------------------------------------------------------- video file / RTSP runner
def run_source(source: str, cam: Camera, policy: Policy, manager: IncidentManager, perception: Perception,
               *, analysis_fps: float = 10.0, realtime: Optional[bool] = None,
               stop: Optional[threading.Event] = None, pipelines_out: Optional[dict] = None,
               clock_base: Optional[datetime] = None, max_seconds: Optional[float] = None,
               blur_heads: bool = False, on_status: Optional[Callable[[str], None]] = None):
    from .ingest.source import VideoSource

    base = clock_base or datetime.now().astimezone()
    manager.clock_base[cam.id] = base
    pipe = CameraPipeline(cam, policy, manager, perception, base, blur_heads)
    if pipelines_out is not None:
        pipelines_out[cam.id] = pipe
    src = VideoSource(source, analysis_fps=analysis_fps, realtime=realtime, on_status=on_status)
    for frame_idx, ts, image in src.frames(stop):
        if max_seconds is not None and ts > max_seconds:
            break
        pipe.process(image, ts, frame_idx)
    return pipe
