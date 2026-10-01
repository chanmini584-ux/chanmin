"""Risk Engine: evidence accumulation over individuals, pairs and zones.

For each frame the engine evaluates candidate *situations* (type + involved tracks).
Every situation score has the form

    raw   = gate * (base + (1 - base) * noisy_or(w_k * e_k))
    score = calibrate(apply_context(raw, place/time factor))
    level = policy thresholds (capped per type)

where `gate` is the evidence without which the situation cannot exist (e.g. sustained
following-run for CHASE) and e_k are supporting evidences (e.g. prior conflict, sudden flight
of the other person). Each evaluation keeps the evidence list so operators see *why*.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..features.tracks import TrackRegistry, TrackState
from ..geometry import cos_sim, noisy_or, ramp, unit
from ..types import Level, SituationType as ST
from .policy import Policy


@dataclass
class Evidence:
    key: str
    text: str
    strength: float
    weight: Optional[float] = None  # None => gate
    ts: Optional[float] = None

    def as_dict(self):
        return {"key": self.key, "text": self.text, "strength": round(self.strength, 3),
                "weight": self.weight, "ts": None if self.ts is None else round(self.ts, 2)}


@dataclass
class SituationEval:
    type: str
    camera_id: str
    actors: tuple  # track ids; for directional situations (actor, target)
    raw: float
    score: float
    evidence: list[Evidence]
    ts: float
    zone: Optional[str] = None
    alone: bool = False

    @property
    def key(self):
        return (self.type, self.camera_id, self.actors, self.zone)


@dataclass
class PairMemory:
    strike: deque = field(default_factory=lambda: deque(maxlen=400))  # (ts, s_ij, s_ji)
    chase: deque = field(default_factory=lambda: deque(maxlen=400))   # (ts, c_ij, c_ji)
    strike_seconds: float = 0.0
    first_strike_ts: Optional[float] = None
    last_strike_ts: Optional[float] = None
    last_contact_ts: Optional[float] = None
    last_near_ts: Optional[float] = None
    approach_ts: dict = field(default_factory=dict)   # actor -> (ts, strength)
    weapon_near_ts: dict = field(default_factory=dict)  # holder -> ts
    chase_start_ts: dict = field(default_factory=dict)  # chaser -> ts
    chase_last_ts: dict = field(default_factory=dict)
    last_dist: Optional[float] = None
    last_ts: Optional[float] = None
    dist_hist: deque = field(default_factory=deque)
    apart_ts: Optional[float] = None
    co_move_since: Optional[float] = None
    co_move_last: float = 0.0

    def companion(self, ts: float) -> float:
        """Strength of the 'moving together as a group' pattern without prior conflict."""
        if self.co_move_since is None:
            return 0.0
        conflict_before = (self.first_strike_ts is not None or any(
            t <= self.co_move_since + 0.5 for t, _ in self.approach_ts.values()))
        if conflict_before:
            return 0.0
        return ramp(ts - self.co_move_since, 2.0, 5.0)


class RiskEngine:
    def __init__(self, policy: Policy, camera, registry: TrackRegistry):
        self.p = policy
        self.cam = camera
        self.reg = registry
        self.depth = float(camera.analysis.get("depth_factor", policy.f("depth_factor", 1.6)))
        self.pairs: dict[tuple[int, int], PairMemory] = {}
        self.weapon_enabled = bool(camera.analysis.get("weapon", True))

    # ------------------------------------------------------------ helpers
    def _dist_vec(self, a: TrackState, b: TrackState) -> tuple[float, np.ndarray]:
        s = (a.scale + b.scale) / 2
        d = (b.foot - a.foot) / s
        d[1] *= self.depth
        return float(np.linalg.norm(d)), d

    def _r(self, x, key):
        lo, hi = self.p.f(key)
        return ramp(x, lo, hi)

    def _strike(self, a: TrackState, b: TrackState, dist: float) -> float:
        img_dir = unit(b.foot - a.foot)
        _, rstd = a.reach_towards(img_dir)
        return (self._r(a.limb_speed(), "strike_limb_speed") * self._r(rstd, "strike_reach_std")
                * self._r(a.speed(self.depth), "strike_max_body_speed")
                * self._r(dist, "contact_dist"))

    def _weapon(self, t: TrackState) -> float:
        if not self.weapon_enabled:
            return 0.0
        return self._r(t.weapon_presence(2.0), "weapon_presence")

    def _finish(self, typ, actors, gate, base, supp, ts, zone=None, alone=False, hour=12):
        pos = [e for e in supp if e.weight is not None and e.weight > 0]
        neg = [e for e in supp if e.weight is not None and e.weight < 0]
        raw = gate * (base + (1 - base) * noisy_or([e.weight * e.strength for e in pos]))
        if neg:  # counter-evidence (e.g. companions walking together) attenuates the score
            raw *= 1.0 - max(-e.weight * e.strength for e in neg)
        factor = self.p.context_factor(self.cam.place_type, hour)
        score = self.p.calibrate(self.p.apply_context(raw, factor))
        return raw, score

    # ------------------------------------------------------------ main step
    def step(self, ts: float, hour: int) -> list[SituationEval]:
        tracks = self.reg.active()
        out: list[SituationEval] = []
        near_any = {t.track_id: False for t in tracks}
        ids = {t.track_id: t for t in tracks}

        # ---- pairwise features & memory
        pair_feats = {}
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                dist, d = self._dist_vec(a, b)
                if dist > 10:
                    continue
                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                m = self.pairs.setdefault(key, PairMemory())
                x, y = (a, b) if a.track_id == key[0] else (b, a)
                dist, d = self._dist_vec(x, y)
                s_xy = self._strike(x, y, dist)
                s_yx = self._strike(y, x, dist)
                vx, vy = x.velocity(depth_factor=self.depth), y.velocity(depth_factor=self.depth)
                u = unit(d)
                c_xy = (self._r(np.linalg.norm(vx), "run_speed") * self._r(np.linalg.norm(vy), "run_speed")
                        * self._r(cos_sim(vx, d), "chase_alignment") * ramp(cos_sim(vy, d), 0.3, 0.7)
                        * ramp(dist, 8.0, 6.0))
                c_yx = (self._r(np.linalg.norm(vx), "run_speed") * self._r(np.linalg.norm(vy), "run_speed")
                        * self._r(cos_sim(vy, -d), "chase_alignment") * ramp(cos_sim(vx, -d), 0.3, 0.7)
                        * ramp(dist, 8.0, 6.0))
                dt = ts - m.last_ts if m.last_ts is not None else 0.0
                trunc = x.truncated or y.truncated
                if trunc:
                    # positions of border-clipped boxes are unreliable: no contact/strike evidence
                    s_xy = s_yx = 0.0
                    dist = max(dist, 3.0)
                # distance change over ~1 s (frame-to-frame differences are too noisy)
                m.dist_hist.append((ts, dist))
                while m.dist_hist and m.dist_hist[0][0] < ts - 1.0:
                    m.dist_hist.popleft()
                t0, d0 = m.dist_hist[0]
                closing = (d0 - dist) / (ts - t0) if ts - t0 >= 0.5 else 0.0
                near6 = ramp(dist, 6.0, 4.0)
                app_xy = self._r(float(vx @ u), "approach_speed") * ramp(closing, 0.2, 0.6) * near6
                app_yx = self._r(float(vy @ -u), "approach_speed") * ramp(closing, 0.2, 0.6) * near6
                # co-moving (walking/running together): similar velocity, close, both moving
                sx, sy = float(np.linalg.norm(vx)), float(np.linalg.norm(vy))
                co = (min(sx, sy) > 0.35 and cos_sim(vx, vy) > 0.9
                      and abs(sx - sy) < 0.35 * max(sx, sy) and dist < 3.5)
                if co:
                    if m.co_move_since is None:
                        m.co_move_since = ts
                    m.co_move_last = ts
                elif m.co_move_since is not None and ts - m.co_move_last > 1.5:
                    m.co_move_since = None
                m.strike.append((ts, s_xy, s_yx))
                m.chase.append((ts, c_xy, c_yx))
                if max(s_xy, s_yx) > 0.5:
                    m.strike_seconds += dt
                    m.first_strike_ts = m.first_strike_ts or ts
                    m.last_strike_ts = ts
                if dist > 2.5 and not trunc:
                    m.apart_ts = ts
                if dist < 1.4 and m.apart_ts is not None:
                    # contact counts only when the two came together (not a pair seen together from the start)
                    m.last_contact_ts = ts
                if dist < 3.0:
                    m.last_near_ts = ts
                for actor, val in ((x.track_id, app_xy), (y.track_id, app_yx)):
                    if val > 0.5:
                        m.approach_ts[actor] = (ts, val)
                for holder in (x, y):
                    if self._weapon(holder) > 0.5 and dist < 4.0:
                        m.weapon_near_ts[holder.track_id] = ts
                for chaser, val in ((x.track_id, c_xy), (y.track_id, c_yx)):
                    if val > 0.5:
                        m.chase_start_ts.setdefault(chaser, ts)
                        m.chase_last_ts[chaser] = ts
                    elif chaser in m.chase_start_ts and ts - m.chase_last_ts.get(chaser, ts) > 3:
                        m.chase_start_ts.pop(chaser, None)
                m.last_dist, m.last_ts = dist, ts
                pair_feats[key] = dict(dist=dist, x=x, y=y, closing=closing)
                if dist < self.p.f("near_dist")[0]:
                    near_any[x.track_id] = near_any[y.track_id] = True

        for key, f in pair_feats.items():
            out += self._assault(key, f, ts, hour)
            out += self._weapon_threat(key, f, ts, hour)
            out += self._chase(key, f, ts, hour)

        for t in tracks:
            if self.weapon_enabled and not near_any[t.track_id]:
                out += self._weapon_alone(t, ts, hour)
            out += self._fall(t, ts, hour)
            out += self._zones(t, ts, hour)
        # forget stale pairs
        for k in [k for k, m in self.pairs.items() if m.last_ts is not None and ts - m.last_ts > 60]:
            del self.pairs[k]
        return [s for s in out if s.raw > 0.0]

    # ------------------------------------------------------------ situations
    def _assault(self, key, f, ts, hour):
        c = self.p.s(ST.ASSAULT)
        m = self.pairs[key]
        if m.first_strike_ts is None:
            return []
        both = [(t, max(a, b)) for t, a, b in m.strike]
        frac = _coverage(both, 1, ts, c["window_s"])
        gate = ramp(frac, *c["gate_fraction"])
        if gate <= 0:
            return []
        x, y = f["x"], f["y"]
        sx = sum(a for _, a, _ in m.strike)
        sy = sum(b for _, _, b in m.strike)
        actor, target = (x, y) if sx >= sy else (y, x)
        mutual = min(sx, sy) > 0.5 * max(sx, sy)
        ev = [Evidence("strike", f"타격 동작 반복 (최근 {c['window_s']:.0f}초 중 {frac*100:.0f}%)"
                       + (" — 쌍방" if mutual else ""), gate, None, ts)]
        w = c["weights"]
        dur = ramp(m.strike_seconds, *c["duration_s"])
        ev.append(Evidence("duration", f"폭력 행위 누적 {m.strike_seconds:.1f}초", dur, w["duration"]))
        fell = [t for t in (x, y) if t.fall_onset_ts and t.fall_onset_ts >= m.first_strike_ts - 1]
        if fell:
            ev.append(Evidence("victim_fall", f"인물 #{fell[0].track_id} 쓰러짐", 1.0,
                               w["victim_fall"], fell[0].fall_onset_ts))
        fled = [t for t in (x, y) if t.flee_onset_ts and t.flee_onset_ts >= m.first_strike_ts]
        if fled:
            ev.append(Evidence("flee_after", f"인물 #{fled[0].track_id} 급히 이탈(도주 추정)", 1.0,
                               w["flee_after"], fled[0].flee_onset_ts))
        wp = max(self._weapon(x), self._weapon(y))
        if wp > 0:
            ev.append(Evidence("weapon", "위험물 의심 물체 동반", wp, w["weapon"]))
        raw, score = self._finish(ST.ASSAULT, None, gate, c["base"], ev[1:], ts, hour=hour)
        return [SituationEval(ST.ASSAULT, self.cam.id, (actor.track_id, target.track_id), raw,
                              score, ev, ts)]

    def _weapon_threat(self, key, f, ts, hour):
        if not self.weapon_enabled:
            return []
        c = self.p.s(ST.WEAPON_THREAT)
        m = self.pairs[key]
        out = []
        for holder, other in ((f["x"], f["y"]), (f["y"], f["x"])):
            wp = self._weapon(holder)
            if holder.weapon_first_ts is None:
                continue
            near = self._r(f["dist"], "near_dist")
            gate = wp * near
            if gate <= 0.05:
                continue
            w, R = c["weights"], c["recent_s"]
            label = "흉기" if holder.weapon_label in ("knife", "scissors") else "둔기"
            ev = [Evidence("weapon_near", f"{label} 의심 물체 보유자(#{holder.track_id})와 "
                           f"인물 #{other.track_id} 거리 {f['dist']:.1f}H", gate, None, ts)]
            ap = m.approach_ts.get(holder.track_id)
            if ap and ts - ap[0] <= R:
                ev.append(Evidence("approach", f"#{holder.track_id}가 #{other.track_id}에게 접근",
                                   ap[1], w["approach"], ap[0]))
            if other.flee_onset_ts and ts - other.flee_onset_ts <= R and m.last_near_ts and \
                    other.flee_onset_ts - m.last_near_ts <= 3.0:
                ev.append(Evidence("victim_flee", f"#{other.track_id} 급가속 이탈 (도주 추정)", 1.0,
                                   w["victim_flee"], other.flee_onset_ts))
            if m.last_contact_ts and ts - m.last_contact_ts <= R:
                ev.append(Evidence("contact", "근접 접촉", 1.0, w["contact"], m.last_contact_ts))
            win = [s for s in m.strike if s[0] >= ts - R]
            st = max([(a if holder is f["x"] else b) for _, a, b in win], default=0.0)
            if st > 0.3:
                ev.append(Evidence("strike", f"#{holder.track_id}의 공격 동작", st, w["strike"]))
            cs = m.chase_start_ts.get(holder.track_id)
            if cs:
                ev.append(Evidence("chase", f"#{holder.track_id}가 #{other.track_id} 추격", 1.0,
                                   w["chase"], cs))
            comp = m.companion(ts)
            if comp > 0:
                ev.append(Evidence("companion", "함께 이동하는 일행 패턴 (갈등 정황 없음)", comp,
                                   -float(c.get("mitigation", {}).get("companion", 0.7))))
            raw, score = self._finish(ST.WEAPON_THREAT, None, gate, c["base"], ev[1:], ts, hour=hour)
            out.append(SituationEval(ST.WEAPON_THREAT, self.cam.id,
                                     (holder.track_id, other.track_id), raw, score, ev, ts))
        return out

    def _weapon_alone(self, t: TrackState, ts, hour):
        c = self.p.s(ST.WEAPON_THREAT)
        if t.weapon_first_ts is None:
            return []
        wp = self._weapon(t)
        if wp <= 0.05:
            return []
        label = "흉기" if t.weapon_label in ("knife", "scissors") else "둔기"
        ev = [Evidence("weapon", f"{label} 의심 물체 보유 (주변 인물 없음)", wp, None, ts)]
        raw, score = self._finish(ST.WEAPON_THREAT, None, wp, c["base_alone"], [], ts, hour=hour)
        return [SituationEval(ST.WEAPON_THREAT, self.cam.id, (t.track_id,), raw, score, ev, ts,
                              alone=True)]

    def _chase(self, key, f, ts, hour):
        c = self.p.s(ST.CHASE)
        m = self.pairs[key]
        out = []
        for idx, (chaser, runner) in enumerate(((f["x"], f["y"]), (f["y"], f["x"]))):
            frac = _coverage(m.chase, 1 + idx, ts, c["window_s"])
            gate = ramp(frac, *c["gate_fraction"])
            if gate <= 0:
                continue
            w = c["weights"]
            start = m.chase_start_ts.get(chaser.track_id, ts)
            ev = [Evidence("chase", f"#{chaser.track_id}가 달아나는 #{runner.track_id}를 뒤따라 달림",
                           gate, None, start)]
            P = c["prior_s"]
            prior = []
            ap = m.approach_ts.get(chaser.track_id)
            if ap and start - ap[0] <= P and ap[0] <= start:
                prior.append(("접근", ap[0]))
            if m.last_strike_ts and start - m.last_strike_ts <= P:
                prior.append(("폭행", m.last_strike_ts))
            wn = m.weapon_near_ts.get(chaser.track_id)
            if wn and start - wn <= P:
                prior.append(("흉기 근접", wn))
            if prior:
                ev.append(Evidence("prior_conflict", "추격 전 갈등 정황: " + ", ".join(p for p, _ in prior),
                                   1.0, w["prior_conflict"], min(t for _, t in prior)))
            if runner.flee_onset_ts and abs(runner.flee_onset_ts - start) <= P:
                ev.append(Evidence("flee_onset", f"#{runner.track_id} 갑작스러운 도주 시작", 1.0,
                                   w["flee_onset"], runner.flee_onset_ts))
            wp = self._weapon(chaser)
            if wp > 0 or (wn and ts - wn <= P * 2):
                ev.append(Evidence("weapon", f"추격자 #{chaser.track_id} 위험물 의심", max(wp, 0.8),
                                   w["weapon"]))
            if f["closing"] > 0.2:
                ev.append(Evidence("closing", "거리 좁혀짐", ramp(f["closing"], 0.2, 1.0), w["closing"]))
            comp = m.companion(ts)
            if comp > 0 and not prior:
                ev.append(Evidence("companion", "함께 달리는 일행 패턴 (갈등 정황 없음)", comp,
                                   -float(c.get("mitigation", {}).get("companion", 0.7))))
            raw, score = self._finish(ST.CHASE, None, gate, c["base"], ev[1:], ts, hour=hour)
            out.append(SituationEval(ST.CHASE, self.cam.id, (chaser.track_id, runner.track_id),
                                     raw, score, ev, ts))
        return out

    def _fall(self, t: TrackState, ts, hour):
        if not t.is_lying():
            return []
        c = self.p.s(ST.FALL)
        dur = t.lying_duration()
        gate = ramp(dur, *c["lying_s"])
        if gate <= 0:
            return []
        w = c["weights"]
        ev = [Evidence("lying", f"인물 #{t.track_id} 바닥에 누운 자세 {dur:.0f}초", gate, None,
                       t.lying_since)]
        if t.fall_onset_ts is not None and t.fall_onset_ts == t.lying_since:
            ev.append(Evidence("sudden", "서 있던 자세에서 급격히 쓰러짐", 1.0, w["sudden"],
                               t.fall_onset_ts))
        ll = ramp(dur, *c["long_lying_s"])
        if ll > 0:
            ev.append(Evidence("long_lying", f"{dur:.0f}초째 일어나지 못함", ll, w["long_lying"]))
        if t.speed(self.depth) < 0.25 and t.limb_speed() < 1.0 and dur > 2:
            ev.append(Evidence("motionless", "움직임 거의 없음", 1.0, w["motionless"]))
        for key, m in self.pairs.items():
            if t.track_id in key and t.lying_since:
                last = max([x for x in (m.last_strike_ts, m.last_contact_ts) if x], default=None)
                if last and last >= t.lying_since - 10 and m.strike_seconds > 0.5:
                    other = key[0] if key[1] == t.track_id else key[1]
                    ev.append(Evidence("after_conflict", f"인물 #{other}와의 폭력 행위 직후", 1.0,
                                       w["after_conflict"], last))
                    break
        raw, score = self._finish(ST.FALL, None, gate, c["base"], ev[1:], ts, hour=hour)
        return [SituationEval(ST.FALL, self.cam.id, (t.track_id,), raw, score, ev, ts)]

    def _zones(self, t: TrackState, ts, hour):
        out = []
        for z in self.cam.zones:
            if z.id not in t.zone_entry:
                continue
            dwell = ts - t.zone_entry[z.id]
            if z.type == "restricted":
                c = self.p.s(ST.INTRUSION)
                gate = ramp(dwell, *c["dwell_s"])
                if gate <= 0:
                    continue
                ev = [Evidence("in_zone", f"인물 #{t.track_id} '{z.name}' 진입, {dwell:.0f}초 체류",
                               gate, None, t.zone_entry[z.id])]
                ld = ramp(dwell, *c["long_dwell_s"])
                if ld > 0:
                    ev.append(Evidence("long_dwell", "장시간 체류", ld, c["weights"]["long_dwell"]))
                raw, score = self._finish(ST.INTRUSION, None, gate, c["base"], ev[1:], ts, hour=hour)
                out.append(SituationEval(ST.INTRUSION, self.cam.id, (t.track_id,), raw, score, ev, ts,
                                         zone=z.id))
            elif z.type == "loiter_watch":
                c = self.p.s(ST.LOITERING)
                gate = ramp(dwell, *c["dwell_s"])
                if gate <= 0:
                    continue
                path = np.array(t.zone_path.get(z.id) or [t.foot])
                seg = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / t.scale if len(path) > 1 else 0
                net = np.linalg.norm(path[-1] - path[0]) / t.scale
                ratio = seg / max(net, 0.5)
                rev = _reversals(path[:, 0], t.scale * 0.5)
                w = c["weights"]
                ev = [Evidence("dwell", f"인물 #{t.track_id} '{z.name}' 내 {dwell:.0f}초 체류", gate,
                               None, t.zone_entry[z.id]),
                      Evidence("path_ratio", f"이동거리/순변위 {ratio:.1f}배", ramp(ratio, *c["path_ratio"]),
                               w["path_ratio"]),
                      Evidence("reversals", f"방향 전환 {rev}회", ramp(rev, *c["reversals"]),
                               w["reversals"])]
                raw, score = self._finish(ST.LOITERING, None, gate, c["base"], ev[1:], ts, hour=hour)
                out.append(SituationEval(ST.LOITERING, self.cam.id, (t.track_id,), raw, score, ev, ts,
                                         zone=z.id))
        return out


def _coverage(samples, idx: int, ts: float, window: float, thr: float = 0.5) -> float:
    """Fraction of the last `window` seconds during which samples[idx] > thr (time-weighted)."""
    win = [s for s in samples if s[0] >= ts - window]
    cov = 0.0
    for a, b in zip(win, win[1:]):
        if b[idx] > thr:
            cov += b[0] - a[0]
    return min(1.0, cov / window)


def _reversals(xs: np.ndarray, hysteresis: float) -> int:
    if len(xs) < 3:
        return 0
    n, direction, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        if direction >= 0 and x < anchor - hysteresis:
            n += direction > 0
            direction, anchor = -1, x
        elif direction <= 0 and x > anchor + hysteresis:
            n += direction < 0
            direction, anchor = 1, x
        elif (direction > 0 and x > anchor) or (direction < 0 and x < anchor):
            anchor = x
    return n


# ---------------------------------------------------------------- situation state tracking
@dataclass
class Situation:
    type: str
    camera_id: str
    actors: tuple
    zone: Optional[str]
    alone: bool
    first_ts: float
    last_eval_ts: float
    score: float = 0.0
    peak_score: float = 0.0
    peak_ts: float = 0.0
    level: Level = Level.NORMAL
    peak_level: Level = Level.NORMAL
    evidence: list = field(default_factory=list)
    peak_evidence: list = field(default_factory=list)
    incident_id: Optional[str] = None
    attention_ts: Optional[float] = None

    @property
    def key(self):
        return (self.type, self.camera_id, self.actors, self.zone)


class SituationTracker:
    """Applies temporal decay (anti-flicker) and level mapping to per-frame evaluations."""

    def __init__(self, policy: Policy):
        self.p = policy
        self.items: dict[tuple, Situation] = {}

    def update(self, ts: float, evals: list[SituationEval]) -> list[tuple[Situation, Level]]:
        """Returns list of (situation, previous_level) whose level changed this step."""
        changed = []
        seen = set()
        for e in evals:
            s = self.items.get(e.key)
            if s is None:
                s = Situation(e.type, e.camera_id, e.actors, e.zone, e.alone, ts, ts)
                self.items[e.key] = s
            seen.add(e.key)
            decayed = s.score * 0.5 ** ((ts - s.last_eval_ts) / self.p.half_life)
            if e.score >= decayed:
                s.score = e.score
                s.evidence = e.evidence
            else:
                s.score = decayed
            s.last_eval_ts = ts
        for k, s in list(self.items.items()):
            if k not in seen:
                s.score *= 0.5 ** ((ts - s.last_eval_ts) / self.p.half_life)
                s.last_eval_ts = ts
            new_level = self.p.level(s.score, s.type, s.alone)
            if s.score > s.peak_score:
                s.peak_score, s.peak_ts, s.peak_evidence = s.score, ts, list(s.evidence)
            if new_level != s.level:
                prev = s.level
                s.level = new_level
                s.peak_level = max(s.peak_level, new_level)
                if new_level >= Level.ATTENTION and s.attention_ts is None:
                    s.attention_ts = ts
                changed.append((s, prev))
            if s.score < 0.02 and k not in seen:
                del self.items[k]
        return changed

    def active(self, min_level: Level = Level.ATTENTION) -> list[Situation]:
        return [s for s in self.items.values() if s.level >= min_level]
