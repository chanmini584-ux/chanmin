import numpy as np
import pytest

from safewatch.config import load_policy
from safewatch.eval.metrics import GT, evaluate, fit_thresholds, isotonic_fit, segments, sweep
from safewatch.features.tracks import TrackRegistry, associate_weapon
from safewatch.geometry import compass_from_vector, noisy_or, point_in_polygon, ramp
from safewatch.risk.policy import Policy
from safewatch.types import Level, ObjectObs, PersonObs


def test_ramp_increasing_and_decreasing():
    assert ramp(0, 1, 2) == 0 and ramp(3, 1, 2) == 1 and ramp(1.5, 1, 2) == pytest.approx(0.5)
    assert ramp(0.5, 1.8, 1.0) == 1 and ramp(2.0, 1.8, 1.0) == 0  # decreasing (distance-like)


def test_polygon_noisy_or_compass():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((11, 5), sq)
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert compass_from_vector(np.array([1.0, 0.0]), 90) == "동쪽"
    assert compass_from_vector(np.array([0.0, -1.0]), 90) == "북쪽"


def test_policy_levels_caps_context():
    p = Policy(load_policy())
    assert p.level(0.0) == Level.NORMAL
    assert p.level(0.99) == Level.EMERGENCY
    assert p.level(0.99, "LOITERING") == Level.WARNING  # capped per type
    assert p.level(0.99, "WEAPON_THREAT", alone=True) == Level.ATTENTION
    assert p.apply_context(0.5, 1.0) == pytest.approx(0.5)
    assert p.apply_context(0.5, 1.2) > 0.5
    night = p.context_factor("street", 23)
    day = p.context_factor("street", 13)
    assert night > day


def test_isotonic_calibration_monotone():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < s).astype(int)
    x, yy = isotonic_fit(s, y)
    assert all(b >= a for a, b in zip(yy, yy[1:]))


def _trace():
    # run r1: FALL GT 10-20 on cam01, score rises at 11; run r2: hard negative with a blip at 0.4
    tr = [("r1", "cam01", t / 10, "FALL", 0.9 if 110 <= t <= 200 else 0.0) for t in range(0, 300)]
    tr += [("r2", "cam01", t / 10, "CHASE", 0.4 if 50 <= t <= 60 else 0.0) for t in range(0, 300)]
    gts = [GT("r1", "FALL", ["cam01"], 10.0, 20.0)]
    return tr, gts


def test_segments_and_event_level_metrics():
    tr, gts = _trace()
    preds = segments(tr, 0.5)
    assert len(preds) == 1 and preds[0].start == pytest.approx(11.0)
    r = evaluate(preds, gts, camera_hours=600 / 3600, theta=0.5)
    assert (r.tp, r.fp, r.fn) == (1, 0, 0) and r.latency == [pytest.approx(1.0)]
    r2 = evaluate(segments(tr, 0.3), gts, camera_hours=600 / 3600, theta=0.3)
    assert r2.fp == 1 and r2.precision == pytest.approx(0.5) and r2.fp_per_cam_hour == pytest.approx(6.0)


def test_strict_vs_lenient_type_matching():
    tr = [("r", "c", t / 10, "ASSAULT", 0.9) for t in range(100, 150)]
    gts = [GT("r", "FALL", ["c"], 10, 15)]
    assert evaluate(segments(tr, 0.5), gts, 1, 0.5, strict=True).tp == 0
    assert evaluate(segments(tr, 0.5), gts, 1, 0.5, strict=False).tp == 1


def test_fit_thresholds_monotone_and_targets():
    tr, gts = _trace()
    res = sweep(tr, gts, 600 / 3600)
    fitted = fit_thresholds(res, {"ATTENTION": {"min_recall": 1.0}, "WARNING": {"min_recall": 1.0},
                                  "HIGH_RISK": {"min_precision": 1.0}, "EMERGENCY": {"min_precision": 1.0}},
                            ["ATTENTION", "WARNING", "HIGH_RISK", "EMERGENCY"])["thresholds"]
    vals = list(fitted.values())
    assert vals == sorted(vals)
    assert fitted["HIGH_RISK"] > 0.4  # must exclude the 0.4 false alarm to reach precision 1.0


def _person(tid, bbox, kps=None):
    return PersonObs(tid, bbox, 0.9, kps)


def test_truncated_upper_body_is_not_lying():
    reg = TrackRegistry("c")
    kps = np.zeros((17, 3))
    p = _person(1, (100, 300, 700, 538), kps)  # wide box touching the bottom border
    reg.update(0.0, [p], [], frame_size=(960, 540))
    assert not reg.tracks[1].is_lying()


def test_weapon_association_by_hand():
    kps = np.zeros((17, 3))
    kps[:, 2] = 0.9
    kps[10, :2] = (150, 200)  # right wrist
    p = _person(1, (100, 100, 180, 300), kps)
    near = ObjectObs("knife", (148, 198, 165, 215), 0.6)
    far = ObjectObs("knife", (400, 400, 420, 420), 0.9)
    phone = ObjectObs("cell phone", (148, 198, 165, 215), 0.9)
    assert associate_weapon(p, [far], 200)[0] == 0
    assert associate_weapon(p, [phone], 200)[0] == 0
    assert associate_weapon(p, [near, far], 200) == (0.6, "knife")
