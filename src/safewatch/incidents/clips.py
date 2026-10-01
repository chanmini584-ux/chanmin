"""Per-camera pre-event ring buffer and incident clip recording (H.264 MP4)."""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import imageio_ffmpeg
import numpy as np


class ClipRecorder:
    def __init__(self, camera_id: str, out_dir: Path, pre_s: float, post_s: float,
                 max_s: float, fps_hint: float = 10.0):
        self.camera_id = camera_id
        self.out_dir = Path(out_dir)
        self.pre_s, self.post_s, self.max_s = pre_s, post_s, max_s
        self.fps_hint = fps_hint
        self.ring: deque = deque()  # (ts, jpeg bytes)
        self.active: dict[str, dict] = {}  # incident_id -> {frames, start_ts, stop_ts}

    def push(self, ts: float, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        item = (ts, buf.tobytes())
        self.ring.append(item)
        while self.ring and self.ring[0][0] < ts - self.pre_s:
            self.ring.popleft()
        for rec in self.active.values():
            if rec["stop_ts"] is None or ts <= rec["stop_ts"]:
                if ts - rec["start_ts"] <= self.max_s:
                    rec["frames"].append(item)

    def start(self, incident_id: str) -> None:
        if incident_id in self.active:
            self.active[incident_id]["stop_ts"] = None
            return
        frames = list(self.ring)
        start = frames[0][0] if frames else 0.0
        self.active[incident_id] = {"frames": frames, "start_ts": start, "stop_ts": None}

    def stop(self, incident_id: str, at_ts: float) -> None:
        rec = self.active.get(incident_id)
        if rec and rec["stop_ts"] is None:
            rec["stop_ts"] = at_ts + self.post_s

    def poll_finished(self, ts: float) -> list[tuple[str, dict]]:
        done = [(k, v) for k, v in self.active.items() if v["stop_ts"] is not None and ts >= v["stop_ts"]]
        for k, _ in done:
            del self.active[k]
        return done

    def flush_all(self) -> list[tuple[str, dict]]:
        done = list(self.active.items())
        self.active.clear()
        return done


def encode_clip(frames: list[tuple[float, bytes]], path: Path, fps: Optional[float] = None,
                on_done: Optional[Callable[[Path, float, float], None]] = None,
                background: bool = True) -> None:
    """Encode JPEG frames into an H.264 MP4 playable in browsers."""
    if not frames:
        return

    def _run():
        path.parent.mkdir(parents=True, exist_ok=True)
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        span = frames[-1][0] - frames[0][0]
        f = fps or (max(1.0, (len(frames) - 1) / span) if span > 0 else 10.0)
        tmp = path.with_suffix(".tmp.mp4")
        gen = imageio_ffmpeg.write_frames(str(tmp), (w, h), fps=f, codec="libx264",
                                          pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                          macro_block_size=2, quality=6,
                                          output_params=["-movflags", "+faststart"])
        gen.send(None)
        for _, jpg in frames:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            gen.send(np.ascontiguousarray(img[:, :, ::-1]))
        gen.close()
        tmp.replace(path)
        if on_done:
            on_done(path, frames[0][0], frames[-1][0])

    if background:
        threading.Thread(target=_run, daemon=True).start()
    else:
        _run()
