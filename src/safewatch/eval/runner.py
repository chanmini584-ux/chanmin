"""Runs scenarios (or labelled videos) through the full pipeline and writes an evaluation report."""
from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..config import load_policy, load_site
from ..risk.policy import Policy
from ..types import Level, SituationType as ST
from .metrics import GT, average_precision, fit_thresholds, isotonic_fit, segments, sweep

NOISE_PROFILES = {
    "default": {},
    "hard": dict(miss_prob=0.10, knife_detect_prob=0.35, phone_as_knife_prob=0.15, kp_jitter_px=4.0,
                 bbox_jitter_px=4.0, random_knife_fp_per_frame=0.01),
}
LEVELS = ["ATTENTION", "WARNING", "HIGH_RISK", "EMERGENCY"]


def _run_one(args):
    name, seed, site_path, policy_cfg, noise_name = args
    from ..incidents.manager import IncidentManager
    from ..pipeline import run_scenario
    from ..sim import library
    from ..sim.render import NoiseModel

    site = load_site(site_path)
    pol = Policy(policy_cfg)
    scn = library.get(name)
    log: list = []
    with tempfile.TemporaryDirectory() as td:
        mgr = IncidentManager(site, pol, Path(td), record_clips=False)
        run_scenario(scn, site, pol, mgr, seed=seed, noise=NoiseModel(**NOISE_PROFILES[noise_name]),
                     score_log=log)
        mgr.finish()
        n_inc = len(mgr.incidents)
        peak = [i["peak_level"] for i in mgr.incidents.values()]
    run = f"{name}#{seed}"
    trace = [(run, cam, round(ts, 3), typ, float(score)) for cam, ts, typ, _a, score, _i in log]
    gts = [asdict(GT(run, e.type, e.cameras, e.start, e.end)) for e in scn.events]
    return {"run": run, "scenario": name, "seed": seed, "trace": trace, "gt": gts,
            "camera_hours": scn.duration * len(scn.cameras) / 3600.0, "incidents": n_inc,
            "incident_peaks": peak, "hard_negative": scn.hard_negative}


def run_eval(scenarios: list[str], seeds: list[int], out_dir: Path, site_path=None, policy_path=None,
             noise: str = "default", workers: int = 4, fit_policy: Optional[Path] = None,
             calibrate: bool = False, log=print) -> dict:
    from ..sim import library

    policy_cfg = load_policy(policy_path)
    pol = Policy(policy_cfg)
    jobs = [(n, s, str(site_path) if site_path else None, policy_cfg, noise) for n in scenarios for s in seeds]
    log(f"[eval] {len(jobs)} runs ({len(scenarios)} scenarios x {len(seeds)} seeds), noise={noise}")
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            runs = []
            for r in pool.imap_unordered(_run_one, jobs):
                runs.append(r)
                log(f"  done {r['run']:28s} incidents={r['incidents']} peaks={r['incident_peaks']}")
    else:
        runs = []
        for j in jobs:
            r = _run_one(j)
            runs.append(r)
            log(f"  done {r['run']:28s} incidents={r['incidents']} peaks={r['incident_peaks']}")
    runs.sort(key=lambda r: r["run"])
    trace = [t for r in runs for t in r["trace"]]
    gts = [GT(**g) for r in runs for g in r["gt"]]
    cam_hours = sum(r["camera_hours"] for r in runs)
    trace5 = [(run, cam, ts, typ, sc) for run, cam, ts, typ, sc in trace]

    strict = sweep(trace5, gts, cam_hours, strict=True)
    lenient = sweep(trace5, gts, cam_hours, strict=False)
    by_theta = {round(r.theta, 3): r for r in strict}
    ops = {}
    for lv in LEVELS:
        th = pol.threshold(Level[lv])
        r = min(strict, key=lambda x: abs(x.theta - th))
        rl = min(lenient, key=lambda x: abs(x.theta - th))
        ops[lv] = {"threshold": th, "strict": r.summary(), "lenient": rl.summary()}
    warn_th = pol.threshold(Level.WARNING)
    per_type = {}
    for typ in ST.ALL:
        rs = sweep(trace5, gts, cam_hours, thetas=[warn_th], strict=True, types={typ})[0]
        n_gt = sum(1 for g in gts if g.type == typ)
        per_type[typ] = {**rs.summary(), "n_gt": n_gt}
    # false alarms at WARNING for inspection
    rw = min(strict, key=lambda x: abs(x.theta - warn_th))
    fp_list = [{"run": p.run, "camera": p.camera, "type": p.type, "start": p.start, "end": p.end,
                "peak": round(p.peak, 3)} for p in rw.fps]
    misses = [asdict(g) for g in rw.misses]
    hard_neg = {}
    for r in runs:
        if r["hard_negative"]:
            mx = max((sc for *_x, sc in r["trace"]), default=0.0)
            hard_neg.setdefault(r["scenario"], []).append(round(mx, 3))

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data": {"kind": "synthetic", "scenarios": scenarios, "seeds": seeds, "noise": noise,
                 "runs": len(runs), "gt_events": len(gts), "camera_hours": round(cam_hours, 4)},
        "policy": {"path": str(policy_path or "configs/policy.default.yaml"),
                   "provenance": policy_cfg.get("provenance"), "levels": policy_cfg.get("levels")},
        "operating_points": ops,
        "per_type_at_warning": per_type,
        "ap": {"strict": round(average_precision(strict), 4), "lenient": round(average_precision(lenient), 4)},
        "pr_curve": [{"theta": r.theta, "precision": round(r.precision, 4), "recall": round(r.recall, 4),
                      "f1": round(r.f1, 4), "fp": r.fp, "fp_per_cam_hour": round(r.fp_per_cam_hour, 2)}
                     for r in strict],
        "false_alarms_at_warning": fp_list,
        "misses_at_warning": misses,
        "hard_negative_max_score": hard_neg,
        "caveats": [
            "합성(시뮬레이션) 데이터 결과입니다. 실제 CCTV 성능을 의미하지 않습니다.",
            "인식(탐지·추적·자세) 오류는 NoiseModel 가정값으로 모사되었습니다. 실제 모델 오류 분포로 교체 필요.",
            "카메라-시간이 매우 짧아 FP/cam-h는 참고용입니다. 실제 장시간 정상 영상으로 측정해야 합니다.",
        ],
    }
    if calibrate or fit_policy:
        # situation-level samples for calibration: positive if inside a GT event of same type & camera
        xs, ys = [], []
        gidx = {}
        for g in gts:
            gidx.setdefault((g.run, g.type), []).append(g)
        for run, cam, ts, typ, sc in trace5[::3]:
            pos = any(cam in g.cameras and g.start - 1 <= ts <= g.end + 1 for g in gidx.get((run, typ), []))
            xs.append(sc)
            ys.append(1 if pos else 0)
        cx, cy = isotonic_fit(np.array(xs), np.array(ys))
        report["calibration"] = {"method": "isotonic", "x": cx, "y": cy, "samples": len(xs)}
    if fit_policy:
        fitted = fit_thresholds(strict, policy_cfg.get("fit_targets", {}), LEVELS)
        new_cfg = json.loads(json.dumps(policy_cfg))
        new_cfg["levels"] = fitted["thresholds"]
        new_cfg["provenance"] = {
            "source": "fitted", "fitted_at": report["generated_at"],
            "fitted_on": f"{report['data']['kind']}: {len(scenarios)} scenarios x {len(seeds)} seeds, noise={noise}",
            "note": "fit_targets를 만족하는 PR 운영점. 합성 데이터 적합값이므로 실제 운영 전 실데이터로 재적합 필요",
            "unreached_targets": fitted["notes"],
        }
        Path(fit_policy).parent.mkdir(parents=True, exist_ok=True)
        Path(fit_policy).write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        report["fitted_policy"] = {"path": str(fit_policy), **fitted}
        log(f"[eval] fitted policy written to {fit_policy}: {fitted['thresholds']}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "pr_curve.svg").write_text(pr_svg(report["pr_curve"], pol), encoding="utf-8")
    (out_dir / "report.md").write_text(report_md(report), encoding="utf-8")
    log(f"[eval] report: {out_dir / 'report.md'}")
    return report


def pr_svg(curve, pol: Policy, w=420, h=320) -> str:
    pad = 40
    def X(r): return pad + r * (w - pad - 10)
    def Y(p): return h - pad - p * (h - pad - 10)
    pts = " ".join(f"{X(c['recall']):.1f},{Y(c['precision']):.1f}" for c in sorted(curve, key=lambda c: c["theta"]))
    marks = []
    for lv in LEVELS:
        th = pol.threshold(Level[lv])
        c = min(curve, key=lambda c: abs(c["theta"] - th))
        marks.append(f'<circle cx="{X(c["recall"]):.1f}" cy="{Y(c["precision"]):.1f}" r="4" fill="#d33"/>'
                     f'<text x="{X(c["recall"]) + 6:.1f}" y="{Y(c["precision"]) - 6:.1f}" font-size="10">{lv}</text>')
    grid = "".join(f'<line x1="{X(v)}" y1="{Y(0)}" x2="{X(v)}" y2="{Y(1)}" stroke="#ddd"/>'
                   f'<line x1="{X(0)}" y1="{Y(v)}" x2="{X(1)}" y2="{Y(v)}" stroke="#ddd"/>'
                   f'<text x="{X(v) - 6}" y="{h - pad + 14}" font-size="9">{v:.1f}</text>'
                   f'<text x="8" y="{Y(v) + 3}" font-size="9">{v:.1f}</text>' for v in [0, .2, .4, .6, .8, 1])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif">'
            f'<rect width="100%" height="100%" fill="white"/>{grid}'
            f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2"/>{"".join(marks)}'
            f'<text x="{w / 2 - 20}" y="{h - 6}" font-size="11">Recall</text>'
            f'<text x="2" y="14" font-size="11">Precision</text></svg>')


def report_md(r: dict) -> str:
    L = [f"# 평가 리포트 ({r['generated_at']})", "",
         f"- 데이터: {r['data']['kind']} / 시나리오 {len(r['data']['scenarios'])}종 × seed {len(r['data']['seeds'])} "
         f"= {r['data']['runs']}회, GT 사건 {r['data']['gt_events']}건, 카메라-시간 {r['data']['camera_hours']}h, 노이즈 {r['data']['noise']}",
         f"- 정책: `{r['policy']['path']}` (provenance: {r['policy']['provenance'].get('source')})",
         f"- AP(strict) {r['ap']['strict']}, AP(lenient: 유형 무관) {r['ap']['lenient']}", "",
         "## 운영점(위험 단계 경계)별 사건 단위 성능 — strict(유형 일치)", "",
         "| 단계 | θ | Precision | Recall | F1 | FNR | FP | FP/cam-h | 평균 지연(s) | P90 지연(s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lv, o in r["operating_points"].items():
        s = o["strict"]
        L.append(f"| {lv} | {o['threshold']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['fnr']} | {s['fp']} | "
                 f"{s['fp_per_cam_hour']} | {s['latency_mean_s']} | {s['latency_p90_s']} |")
    L += ["", "## 유형별 (WARNING 기준)", "", "| 유형 | GT | P | R | F1 | FP | 평균 지연(s) |", "|---|---|---|---|---|---|---|"]
    for t, s in r["per_type_at_warning"].items():
        L.append(f"| {ST.KO[t]} | {s['n_gt']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['fp']} | {s['latency_mean_s']} |")
    L += ["", "## 하드 네거티브(정상-유사 상황) 최고 점수", ""]
    for k, v in r["hard_negative_max_score"].items():
        L.append(f"- {k}: {v}")
    L += ["", f"## WARNING 기준 오탐 {len(r['false_alarms_at_warning'])}건 / 미탐 {len(r['misses_at_warning'])}건", ""]
    for f in r["false_alarms_at_warning"][:20]:
        L.append(f"- FP {f['run']} {f['camera']} {f['type']} {f['start']}–{f['end']}s peak {f['peak']}")
    for m in r["misses_at_warning"][:20]:
        L.append(f"- MISS {m['run']} {m['type']} {m['cameras']} {m['start']}–{m['end']}s")
    if "fitted_policy" in r:
        L += ["", "## 적합된 정책 (fit-policy)", "", f"- 파일: `{r['fitted_policy']['path']}`",
              f"- 임계값: {r['fitted_policy']['thresholds']}", f"- 미달 목표: {r['fitted_policy']['notes'] or '없음'}"]
    L += ["", "![PR curve](pr_curve.svg)", "", "## 주의", ""] + [f"- {c}" for c in r["caveats"]]
    return "\n".join(L) + "\n"
