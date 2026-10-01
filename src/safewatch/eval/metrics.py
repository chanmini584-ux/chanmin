"""Event-level evaluation (not frame-level).

Predicted events at threshold θ are the time intervals where a situation type's score on a camera
stays ≥ θ (intervals closer than `merge_gap` are merged). A prediction matches a ground-truth event
when the run, camera (∈ GT cameras) and — in strict mode — the type agree, and the intervals overlap
within `tol` seconds. Duplicate predictions of an already-matched event (e.g. on a second camera)
are neither TP nor FP; an unmatched prediction is a false positive (false alarm).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GT:
    run: str
    type: str
    cameras: list[str]
    start: float
    end: float


@dataclass
class Pred:
    run: str
    camera: str
    type: str
    start: float
    end: float
    peak: float


def segments(trace, theta: float, merge_gap: float = 3.0) -> list[Pred]:
    """trace: iterable of (run, camera, ts, type, score)."""
    series = defaultdict(lambda: defaultdict(float))
    for run, cam, ts, typ, score in trace:
        k = (run, cam, typ)
        series[k][ts] = max(series[k][ts], score)
    out = []
    for (run, cam, typ), pts in series.items():
        cur = None
        for ts in sorted(pts):
            s = pts[ts]
            if s >= theta:
                if cur and ts - cur.end <= merge_gap:
                    cur.end, cur.peak = ts, max(cur.peak, s)
                else:
                    if cur:
                        out.append(cur)
                    cur = Pred(run, cam, typ, ts, ts, s)
        if cur:
            out.append(cur)
    return out


def _overlap(p: Pred, g: GT, tol: float) -> bool:
    return p.start <= g.end + tol and p.end >= g.start - tol


@dataclass
class Result:
    theta: float
    n_gt: int
    n_pred: int
    tp: int
    fn: int
    fp: int
    precision: float
    recall: float
    f1: float
    fp_per_cam_hour: float
    latency: list = field(default_factory=list)
    fps: list = field(default_factory=list)
    misses: list = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latency) if self.latency else np.array([np.nan])
        return {"theta": round(self.theta, 3), "gt": self.n_gt, "pred": self.n_pred, "tp": self.tp,
                "fn": self.fn, "fp": self.fp, "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4),
                "fnr": round(1 - self.recall, 4) if self.n_gt else None,
                "fp_per_cam_hour": round(self.fp_per_cam_hour, 3),
                "latency_mean_s": None if np.isnan(lat).all() else round(float(np.nanmean(lat)), 2),
                "latency_p90_s": None if np.isnan(lat).all() else round(float(np.nanpercentile(lat, 90)), 2)}


def evaluate(preds: list[Pred], gts: list[GT], camera_hours: float, theta: float,
             strict: bool = True, tol: float = 2.0, types: set | None = None) -> Result:
    if types is not None:
        preds = [p for p in preds if p.type in types]
        gts = [g for g in gts if g.type in types]
    matched_gt = set()
    latency = {}
    fps = []
    for p in preds:
        hit = False
        for gi, g in enumerate(gts):
            if g.run != p.run or p.camera not in g.cameras:
                continue
            if strict and g.type != p.type:
                continue
            if _overlap(p, g, tol):
                hit = True
                matched_gt.add(gi)
                d = max(-tol, p.start - g.start)
                latency[gi] = min(latency.get(gi, 1e9), d)
        if not hit:
            fps.append(p)
    tp = len(matched_gt)
    n_pred_matched = len(preds) - len(fps)
    precision = n_pred_matched / len(preds) if preds else 1.0
    recall = tp / len(gts) if gts else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    misses = [g for i, g in enumerate(gts) if i not in matched_gt]
    return Result(theta, len(gts), len(preds), tp, len(gts) - tp, len(fps), precision, recall, f1,
                  len(fps) / camera_hours if camera_hours > 0 else 0.0, list(latency.values()), fps, misses)


def sweep(trace, gts, camera_hours, thetas=None, strict=True, tol=2.0, merge_gap=3.0, types=None):
    thetas = thetas if thetas is not None else np.round(np.arange(0.05, 1.0, 0.01), 3)
    trace = list(trace)
    return [evaluate(segments(trace, float(t), merge_gap), gts, camera_hours, float(t), strict, tol, types)
            for t in thetas]


def average_precision(results: list[Result]) -> float:
    pts = sorted(((r.recall, r.precision) for r in results if r.n_pred > 0), key=lambda x: x[0])
    if not pts:
        return 0.0
    rec = np.array([0.0] + [p[0] for p in pts])
    prec = np.array([pts[0][1]] + [p[1] for p in pts])
    for i in range(len(prec) - 2, -1, -1):  # precision envelope
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def fit_thresholds(results: list[Result], targets: dict, order: list[str]) -> dict:
    """Pick operating points on the PR sweep that satisfy policy targets.

    min_recall    → the highest θ whose recall ≥ target (fewest alarms that still meet recall)
    min_precision → the lowest θ whose precision ≥ target (most recall that still meets precision)
    Thresholds are forced to be non-decreasing along `order`.
    """
    rs = sorted(results, key=lambda r: r.theta)
    out, notes = {}, {}
    prev = 0.0
    for lvl in order:
        t = targets.get(lvl, {})
        chosen = None
        if "min_recall" in t:
            ok = [r for r in rs if r.recall >= t["min_recall"] and r.n_pred > 0]
            chosen = ok[-1] if ok else None
        elif "min_precision" in t:
            ok = [r for r in rs if r.precision >= t["min_precision"] and r.n_pred > 0]
            chosen = ok[0] if ok else None
        if chosen is None:
            notes[lvl] = "target not reachable on this data; kept monotone fallback"
            theta = prev + 0.05
        else:
            theta = chosen.theta
        theta = max(theta, prev + 0.01)
        out[lvl] = round(min(theta, 0.99), 3)
        prev = out[lvl]
    return {"thresholds": out, "notes": notes}


def isotonic_fit(scores: np.ndarray, labels: np.ndarray, bins: int = 20) -> tuple[list, list]:
    """Pool-adjacent-violators isotonic regression on binned scores → (x, y) lookup table."""
    if len(scores) == 0:
        return [], []
    order = np.argsort(scores)
    s, y = scores[order], labels[order].astype(float)
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s >= a) & (s <= b)
        if m.sum() == 0:
            continue
        xs.append(float(s[m].mean()))
        ys.append(float(y[m].mean()))
        ws.append(float(m.sum()))
    # PAV
    blocks = [[x, yv, w] for x, yv, w in zip(xs, ys, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1]:
            a, b = blocks[i], blocks[i + 1]
            w = a[2] + b[2]
            blocks[i] = [(a[0] * a[2] + b[0] * b[2]) / w, (a[1] * a[2] + b[1] * b[2]) / w, w]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [round(b[0], 4) for b in blocks], [round(b[1], 4) for b in blocks]
