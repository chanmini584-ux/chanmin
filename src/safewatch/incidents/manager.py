"""Incident lifecycle: groups situations into incidents, builds the timeline, records clips,
suggests/links neighbouring cameras (multi-camera handoff) and notifies listeners."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import Site
from ..features.tracks import TrackState
from ..geometry import compass_from_vector
from ..risk.engine import Situation
from ..risk.policy import Policy
from ..types import Level, SituationType as ST
from .appearance import hist_similarity
from .clips import ClipRecorder, encode_clip

TYPE_PRIORITY = [ST.WEAPON_THREAT, ST.ASSAULT, ST.CHASE, ST.FALL, ST.INTRUSION, ST.LOITERING]
ROLE_KO = {"actor": "행위자(추정)", "target": "상대방", "subject": "당사자"}
EDGE = 0.07


@dataclass
class ExpectedArrival:
    incident_id: str
    from_cam: str
    from_key: str
    to_cam: str
    entry_side: str
    t_exit: float
    window: tuple
    hist: Optional[np.ndarray]
    role: str


class IncidentManager:
    def __init__(self, site: Site, policy: Policy, out_dir: Path,
                 clock_base: dict[str, datetime] | None = None, encode_background: bool = True,
                 record_clips: bool = True):
        self.site = site
        self.p = policy
        self.out_dir = Path(out_dir)
        (self.out_dir / "clips").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.clock_base = clock_base or {}
        self.encode_background = encode_background
        self.record_clips = record_clips
        self.incidents: dict[str, dict] = {}
        self.listeners: list[Callable[[str, dict], None]] = []
        self.recorders: dict[str, ClipRecorder] = {}
        self.expected: list[ExpectedArrival] = []
        self.actor_index: dict[str, str] = {}  # "cam:tid" -> incident id
        self.lock = threading.RLock()
        self._seq = 0
        cfg = policy.incident
        self.merge_gap = float(cfg.get("merge_gap_s", 8))
        self.close_after = float(cfg.get("close_after_s", 12))
        self.pre_s, self.post_s = float(cfg.get("pre_s", 8)), float(cfg.get("post_s", 5))
        self.max_clip = float(cfg.get("max_clip_s", 120))
        self.pending_encodes = 0
        self._rate_ref: dict[str, tuple[float, float]] = {}  # cam -> (media ts, wall time) at first frame
        self._now: dict[str, tuple[float, float]] = {}

    def processing_wall(self, cam: str, media_ts: float) -> datetime:
        """Wall-clock time at which the given media timestamp was (or would be) processed.

        Media time and wall time differ for replayed/accelerated streams, so latency KPIs are
        measured on the processing clock, not on the scenario/recording clock."""
        ts0, w0 = self._rate_ref.get(cam, (media_ts, time.time()))
        ts1, w1 = self._now.get(cam, (media_ts, time.time()))
        rate = (w1 - w0) / (ts1 - ts0) if ts1 - ts0 > 1e-6 else 1.0
        return datetime.fromtimestamp(w1 - (ts1 - media_ts) * rate).astimezone()

    # ------------------------------------------------------------ utils
    def wall(self, cam: str, ts: float) -> datetime:
        base = self.clock_base.get(cam) or self.clock_base.get("*") or datetime.now().astimezone()
        return base + timedelta(seconds=ts)

    def _emit(self, kind: str, inc: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(kind, inc)
            except Exception:  # listener errors must not break analysis
                pass

    def _recorder(self, cam: str) -> ClipRecorder:
        if cam not in self.recorders:
            self.recorders[cam] = ClipRecorder(cam, self.out_dir / "clips", self.pre_s, self.post_s,
                                               self.max_clip)
        return self.recorders[cam]

    def _new_incident(self, s: Situation, ts: float) -> dict:
        self._seq += 1
        cam = self.site.camera(s.camera_id)
        start = self.wall(s.camera_id, s.first_ts)
        iid = f"INC-{start:%Y%m%d}-{self._seq:04d}-{int(time.time() * 1000) % 100000:05d}"
        inc = {
            "id": iid, "status": "OPEN", "review": "NEW",
            "type": s.type, "type_ko": ST.KO[s.type],
            "level": s.level.name, "peak_level": s.level.name, "peak_score": round(s.score, 3),
            "start_ts": s.first_ts, "end_ts": ts, "start_time": start.isoformat(timespec="seconds"),
            "primary_camera": s.camera_id, "cameras": [s.camera_id],
            "location": {"camera": s.camera_id, "name": cam.name, "address": cam.address,
                         "lat": cam.lat, "lon": cam.lon},
            "actors": {}, "objects": [], "situations": {}, "timeline": [],
            "related_cameras": [], "route": [], "clips": {}, "snapshot": None,
            "summary": "", "report": None, "feedback": None,
            "kpi": {"detected_wall": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "event_start_wall": self.processing_wall(s.camera_id, s.first_ts).isoformat(
                        timespec="milliseconds"),
                    "event_start_media_ts": round(s.first_ts, 2)},
            "_last_active": ts, "_popup": False,
        }
        self.incidents[iid] = inc
        for nb in cam.neighbors:
            self._add_related(inc, nb, "인접 CCTV (카메라 토폴로지)", "nearby")
        return inc

    def _add_related(self, inc, cam_id, reason, status, score=None):
        for r in inc["related_cameras"]:
            if r["camera"] == cam_id:
                order = ["nearby", "predicted", "candidate", "linked"]
                if order.index(status) > order.index(r["status"]):
                    r.update(status=status, reason=reason, score=score)
                return
        c = self.site.camera(cam_id)
        inc["related_cameras"].append({"camera": cam_id, "name": c.name, "reason": reason,
                                       "status": status, "score": score})

    def _timeline(self, inc, cam, ts, text, level=None, key=None):
        if key and any(e.get("key") == key for e in inc["timeline"]):
            return
        inc["timeline"].append({"ts": round(ts, 2), "time": self.wall(cam, ts).isoformat(timespec="seconds"),
                                "camera": cam, "level": level, "text": text, "key": key})
        inc["timeline"].sort(key=lambda e: (e["time"], e["ts"]))

    # ------------------------------------------------------------ main entry per frame
    def on_frame(self, cam_id: str, ts: float, annotated: np.ndarray, changed, active: list[Situation],
                 tracks: dict[int, TrackState], new_tracks: list[TrackState],
                 lost_tracks: list[TrackState], frame_w: int) -> None:
        with self.lock:
            now = time.time()
            self._rate_ref.setdefault(cam_id, (ts, now))
            self._now[cam_id] = (ts, now)
            rec = self._recorder(cam_id)
            if self.record_clips:
                rec.push(ts, annotated)
            # cross-camera candidates: recently appeared tracks once their appearance is measurable
            if self.expected:
                for t in tracks.values():
                    if (not t.lost and ts - t.first_ts <= 4.0 and t.appearance is not None
                            and f"{cam_id}:{t.track_id}" not in self.actor_index):
                        self._match_arrival(cam_id, t, ts, frame_w)
            touched = set()
            for s in active:
                inc = self._assign(s, ts)
                touched.add(inc["id"])
                self._update_from_situation(inc, s, ts, tracks)
            for s, prev in changed:
                if s.incident_id and s.level > prev:
                    inc = self.incidents.get(s.incident_id)
                    if inc:
                        self._timeline(inc, cam_id, ts, f"{ST.KO[s.type]} 위험도 {prev.name} → {s.level.name}",
                                       s.level.name)
                        if Level[inc["peak_level"]] <= s.level:
                            self._snapshot(inc, cam_id, annotated)
            for t in lost_tracks:
                self._handle_exit(cam_id, t, ts, frame_w)
            # lifecycle
            for inc in list(self.incidents.values()):
                if inc["status"] != "OPEN":
                    continue
                if cam_id not in inc["cameras"]:
                    continue
                if inc["id"] in touched:
                    inc["_last_active"] = ts
                    inc["end_ts"] = ts
                    rec.start(inc["id"])
                    self._emit("incident_updated", inc)
                elif ts - inc["_last_active"] > self.close_after and inc["primary_camera"] == cam_id:
                    inc["level"] = Level.NORMAL.name
                    self._close(inc, ts)
                elif ts - inc["_last_active"] > self.close_after:
                    rec.stop(inc["id"], ts)
            for iid, data in rec.poll_finished(ts):
                self._encode(iid, cam_id, data)

    def _assign(self, s: Situation, ts: float) -> dict:
        if s.incident_id and s.incident_id in self.incidents and \
                self.incidents[s.incident_id]["status"] == "OPEN":
            return self.incidents[s.incident_id]
        keys = [f"{s.camera_id}:{a}" for a in s.actors]
        for k in keys:
            iid = self.actor_index.get(k)
            if iid and iid in self.incidents:
                inc = self.incidents[iid]
                if inc["status"] == "OPEN" and ts - inc["_last_active"] <= self.merge_gap:
                    s.incident_id = iid
                    return inc
        inc = self._new_incident(s, ts)
        s.incident_id = inc["id"]
        self._timeline(inc, s.camera_id, s.first_ts, f"{ST.KO[s.type]} 상황 감지 시작", s.level.name)
        self._emit("incident_created", inc)
        return inc

    def _update_from_situation(self, inc: dict, s: Situation, ts: float, tracks) -> None:
        cam = s.camera_id
        if cam not in inc["cameras"]:
            inc["cameras"].append(cam)
        skey = f"{s.type}|{cam}|{','.join(map(str, s.actors))}|{s.zone or ''}"
        rec = inc["situations"].get(skey)
        if rec is None:
            rec = inc["situations"][skey] = {"type": s.type, "type_ko": ST.KO[s.type], "camera": cam,
                                             "actors": [f"{cam}:{a}" for a in s.actors],
                                             "first_ts": round(s.first_ts, 2), "zone": s.zone}
        rec.update(level=s.level.name, score=round(s.score, 3), peak_level=s.peak_level.name,
                   peak_score=round(s.peak_score, 3), last_ts=round(ts, 2),
                   evidence=[e.as_dict() for e in (s.peak_evidence or s.evidence)])
        # incident-level aggregation
        cur = max((Level[r["level"]] for r in inc["situations"].values()
                   if r.get("last_ts", 0) >= ts - 0.5), default=s.level)
        inc["level"] = max(cur, s.level).name
        if s.peak_score > inc["peak_score"] or Level[inc["peak_level"]] < s.peak_level:
            inc["peak_score"] = round(max(inc["peak_score"], s.peak_score), 3)
            inc["peak_level"] = max(Level[inc["peak_level"]], s.peak_level).name
        best = max(inc["situations"].values(),
                   key=lambda r: (Level[r["peak_level"]], r["peak_score"], -TYPE_PRIORITY.index(r["type"])))
        inc["type"], inc["type_ko"] = best["type"], best["type_ko"]
        if Level[inc["peak_level"]] >= self.p.popup_min_level and not inc["_popup"]:
            inc["_popup"] = True
            inc["kpi"]["alert_wall"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
            inc["kpi"]["alert_media_ts"] = round(ts, 2)
            self._timeline(inc, cam, ts, f"관제자 알림 발송 ({inc['peak_level']})", inc["peak_level"], key="alert")
            self._emit("incident_alert", inc)
        # actors
        roles = ("actor", "target") if len(s.actors) == 2 else ("subject",)
        for tid, role in zip(s.actors, roles):
            key = f"{cam}:{tid}"
            self.actor_index[key] = inc["id"]
            t = tracks.get(tid)
            a = inc["actors"].setdefault(key, {"key": key, "camera": cam, "track_id": tid, "role": role,
                                                "first_ts": round(t.first_ts if t else ts, 2)})
            if a["role"] == "subject" and role != "subject":
                a["role"] = role
            if t is not None:
                a["last_ts"] = round(t.last_ts, 2)
                a["color"] = t.color_name
                v = t.velocity()
                a["direction"] = compass_from_vector(v, self.site.camera(cam).image_right_bearing) \
                    if np.linalg.norm(v) > 0.3 else a.get("direction", "정지")
                if t.weapon_first_ts is not None and t.weapon_frames >= 3:
                    label = "흉기(칼 등) 의심" if t.weapon_label in ("knife", "scissors") else "둔기 의심"
                    a["weapon"] = label
                    if label not in inc["objects"]:
                        inc["objects"].append(label)
        # key evidence → timeline
        for e in s.evidence:
            if e.key in ("approach", "victim_flee", "flee_onset", "chase", "victim_fall", "sudden",
                         "flee_after", "in_zone", "dwell", "weapon_near", "weapon", "strike", "lying",
                         "prior_conflict", "companion") and e.strength >= 0.5:
                if e.key == "companion":
                    continue
                self._timeline(inc, cam, e.ts if e.ts is not None else ts, e.text,
                               key=f"{skey}|{e.key}")

    def _snapshot(self, inc, cam, frame):
        p = self.out_dir / "snapshots" / f"{inc['id']}.jpg"
        cv2.imwrite(str(p), frame)
        inc["snapshot"] = f"snapshots/{p.name}"

    # ------------------------------------------------------------ multi-camera handoff
    def _handle_exit(self, cam_id: str, t: TrackState, ts: float, frame_w: int) -> None:
        key = f"{cam_id}:{t.track_id}"
        iid = self.actor_index.get(key)
        if not iid or iid not in self.incidents:
            return
        inc = self.incidents[iid]
        if inc["status"] != "OPEN" and ts - inc["end_ts"] > 30:
            return
        x = t.foot[0] / max(frame_w, 1)
        side = "left" if x < EDGE else "right" if x > 1 - EDGE else None
        if side is None:
            return
        cam = self.site.camera(cam_id)
        a = inc["actors"].get(key, {})
        for nb, info in cam.neighbors.items():
            if info.get("exit_side") != side:
                continue
            entry_side = "left" if side == "right" else "right"
            self._add_related(inc, nb, f"인물 #{t.track_id}({ROLE_KO.get(a.get('role'), '')}) "
                                       f"{cam_id} {'오른쪽' if side == 'right' else '왼쪽'}으로 이탈 → 이동 예상",
                              "predicted")
            self.expected.append(ExpectedArrival(iid, cam_id, key, nb, entry_side, t.last_ts,
                                                 tuple(info.get("travel_s", [0, 30])), t.appearance,
                                                 a.get("role", "subject")))
            self._timeline(inc, cam_id, t.last_ts,
                           f"인물 #{t.track_id} {a.get('direction', '')} 방향으로 화면 이탈 → {nb} 방향 이동 예상",
                           key=f"exit|{key}")
            inc["route"].append({"camera": cam_id, "track": key, "from_ts": a.get("first_ts"),
                                 "to_ts": round(t.last_ts, 2), "exit": side})
            self._emit("incident_updated", inc)

    def _match_arrival(self, cam_id: str, t: TrackState, ts: float, frame_w: int) -> None:
        x = t.hist[0].foot[0] / max(frame_w, 1)  # where the track entered the view
        side = "left" if x < 0.15 else "right" if x > 0.85 else None
        best, best_sim = None, 0.0
        for ea in self.expected:
            if ea.to_cam != cam_id or side != ea.entry_side:
                continue
            dt = t.first_ts - ea.t_exit
            if not (ea.window[0] <= dt <= ea.window[1]):
                continue
            sim = hist_similarity(ea.hist, t.appearance) if (ea.hist is not None and t.appearance is not None) else 0.5
            if sim > best_sim:
                best, best_sim = ea, sim
        # expire stale expectations
        self.expected = [e for e in self.expected if ts - e.t_exit <= e.window[1] + 5]
        if best is None or best_sim < 0.45:
            return
        inc = self.incidents.get(best.incident_id)
        if inc is None:
            return
        key = f"{cam_id}:{t.track_id}"
        self.actor_index[key] = inc["id"]
        inc["actors"][key] = {"key": key, "camera": cam_id, "track_id": t.track_id, "role": best.role,
                              "first_ts": round(ts, 2), "linked_from": best.from_key,
                              "link_score": round(best_sim, 2), "link_status": "candidate",
                              "color": t.color_name}
        if cam_id not in inc["cameras"]:
            inc["cameras"].append(cam_id)
        self._add_related(inc, cam_id, f"{best.from_key} 인물과 인상착의 유사({best_sim:.2f}) — 관제자 확인 필요",
                          "linked", round(best_sim, 2))
        self._timeline(inc, cam_id, ts, f"{cam_id} 진입 인물 #{t.track_id} — {best.from_key}와 동일인 후보 "
                                        f"(색상 유사도 {best_sim:.2f}, 관제자 확인 필요)", key=f"link|{key}")
        inc["route"].append({"camera": cam_id, "track": key, "from_ts": round(ts, 2), "to_ts": None,
                             "exit": None})
        if inc["status"] == "OPEN":
            inc["_last_active"] = max(inc["_last_active"], ts)
            self._recorder(cam_id).start(inc["id"])
        self.expected.remove(best)
        self._emit("incident_updated", inc)

    # ------------------------------------------------------------ closing & clips
    def _close(self, inc: dict, ts: float) -> None:
        inc["status"] = "CLOSED"
        for cam in inc["cameras"]:
            self._recorder(cam).stop(inc["id"], ts)
        self._emit("incident_closed", inc)

    def _encode(self, iid: str, cam: str, data: dict) -> None:
        inc = self.incidents.get(iid)
        if inc is None or not data["frames"]:
            return
        path = self.out_dir / "clips" / f"{iid}_{cam}.mp4"
        inc["clips"][cam] = {"path": f"clips/{path.name}", "status": "encoding",
                             "start_ts": round(data["frames"][0][0], 2),
                             "end_ts": round(data["frames"][-1][0], 2)}
        self.pending_encodes += 1

        def done(p, a, b):
            with self.lock:
                inc["clips"][cam]["status"] = "ready"
                self.pending_encodes -= 1
                self._emit("incident_updated", inc)

        encode_clip(data["frames"], path, on_done=done, background=self.encode_background)

    def finish(self, ts_by_cam: dict[str, float] | None = None) -> None:
        """End of all streams: close open incidents and flush recordings."""
        with self.lock:
            for inc in self.incidents.values():
                if inc["status"] == "OPEN":
                    inc["status"] = "CLOSED"
                    self._emit("incident_closed", inc)
            for cam, rec in self.recorders.items():
                for iid, data in rec.flush_all():
                    self._encode(iid, cam, data)

    def public(self, inc: dict) -> dict:
        return {k: v for k, v in inc.items() if not k.startswith("_")}
