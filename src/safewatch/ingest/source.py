"""Video ingest for recorded files and live RTSP streams (IP camera / NVR / VMS re-stream).

- Files: media timestamps from the frame index; optional real-time pacing.
- RTSP: timestamps from a monotonic clock; automatic reconnect with backoff.
- Frames are sub-sampled to `analysis_fps` (analysis does not need the full camera rate).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np


def is_stream(source: str) -> bool:
    return source.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://"))


class VideoSource:
    def __init__(self, source: str, analysis_fps: float = 10.0, realtime: Optional[bool] = None,
                 max_reconnects: int = 1000, on_status: Optional[Callable[[str], None]] = None,
                 rtsp_transport: str = "tcp"):
        self.source = source
        self.analysis_fps = analysis_fps
        self.stream = is_stream(source)
        self.realtime = (not self.stream) if realtime is None else realtime
        self.realtime = self.realtime and not self.stream  # live streams are paced by the camera
        self.max_reconnects = max_reconnects
        self.on_status = on_status or (lambda s: None)
        if self.stream:
            # TCP transport is more robust through NAT/firewalls than UDP for RTSP
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{rtsp_transport}")

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def frames(self, stop: Optional[threading.Event] = None) -> Iterator[tuple[int, float, np.ndarray]]:
        if not self.stream and not os.path.exists(self.source):
            raise FileNotFoundError(self.source)
        reconnects = 0
        out_idx = 0
        t0 = time.monotonic()
        while True:
            cap = self._open()
            if not cap.isOpened():
                if not self.stream:
                    raise RuntimeError(f"cannot open video: {self.source}")
                reconnects += 1
                self.on_status(f"연결 실패, 재시도 {reconnects}")
                if reconnects > self.max_reconnects:
                    return
                time.sleep(min(10.0, 0.5 * 2 ** min(reconnects, 5)))
                continue
            self.on_status("connected")
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 1 or fps > 120:
                fps = 25.0
            step = max(1, round(fps / self.analysis_fps))
            idx = 0
            next_emit_wall = time.monotonic()
            last_stream_emit = -1e9
            while True:
                if stop is not None and stop.is_set():
                    cap.release()
                    return
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if self.stream:
                    now = time.monotonic()
                    if now - last_stream_emit < 1.0 / self.analysis_fps * 0.95:
                        continue
                    last_stream_emit = now
                    ts = now - t0
                else:
                    if idx % step != 0:
                        idx += 1
                        continue
                    ts = idx / fps
                    idx += 1
                    if self.realtime:
                        next_emit_wall += step / fps
                        d = next_emit_wall - time.monotonic()
                        if d > 0:
                            time.sleep(d)
                yield out_idx, ts, frame
                out_idx += 1
            cap.release()
            if not self.stream:
                return
            reconnects += 1
            self.on_status(f"스트림 끊김, 재연결 {reconnects}")
            if reconnects > self.max_reconnects:
                return
            time.sleep(min(10.0, 0.5 * 2 ** min(reconnects, 5)))
