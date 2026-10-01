"""Regression tests on the synthetic scenarios.

These check the *behaviour of the whole pipeline* (features → risk engine → incidents) under the
current hypothesis policy. They are not accuracy claims: the scenarios were also used to design
the features (see docs/prototype/README.md, '평가 해석 시 주의').
"""
from pathlib import Path

import pytest

from safewatch.config import load_policy, load_site
from safewatch.incidents.manager import IncidentManager
from safewatch.pipeline import run_scenario
from safewatch.risk.policy import Policy
from safewatch.sim import library
from safewatch.types import Level


def _run(name, tmp_path, seed=1, record=False):
    site, pol = load_site(), Policy(load_policy())
    mgr = IncidentManager(site, pol, Path(tmp_path), record_clips=record, encode_background=False)
    log = []
    run_scenario(library.get(name), site, pol, mgr, seed=seed, score_log=log)
    mgr.finish()
    return mgr, log


def _max_level(mgr, typ=None):
    lv = Level.NORMAL
    for inc in mgr.incidents.values():
        for s in inc["situations"].values():
            if typ is None or s["type"] == typ:
                lv = max(lv, Level[s["peak_level"]])
    return lv


@pytest.mark.parametrize("name,typ,min_level", [
    ("assault_fight", "ASSAULT", Level.HIGH_RISK),
    ("assault_fight", "FALL", Level.HIGH_RISK),
    ("knife_threat_chase", "WEAPON_THREAT", Level.HIGH_RISK),
    ("knife_threat_chase", "CHASE", Level.HIGH_RISK),
    ("collapse_alone", "FALL", Level.HIGH_RISK),
    ("night_intrusion", "INTRUSION", Level.WARNING),
])
def test_dangerous_situations_escalate(tmp_path, name, typ, min_level):
    mgr, _ = _run(name, tmp_path)
    assert _max_level(mgr, typ) >= min_level


@pytest.mark.parametrize("name", ["normal_walkers", "jogging_pair", "kitchen_knife_carry", "greeting_hug",
                                  "phone_users", "bat_sports"])
@pytest.mark.parametrize("seed", [1, 2])
def test_hard_negatives_do_not_popup(tmp_path, name, seed):
    mgr, _ = _run(name, tmp_path, seed=seed)
    assert _max_level(mgr) < Level.WARNING, [i["summary"] for i in mgr.incidents.values()]


def test_knife_chase_is_one_cross_camera_incident(tmp_path):
    mgr, _ = _run("knife_threat_chase", tmp_path, record=True)
    incs = [i for i in mgr.incidents.values() if Level[i["peak_level"]] >= Level.WARNING]
    assert len(incs) == 1
    inc = incs[0]
    assert {"cam01", "cam02"} <= set(inc["cameras"])
    linked = [a for a in inc["actors"].values() if a.get("linked_from")]
    assert linked and all(a["link_score"] >= 0.8 for a in linked)
    assert any(r["camera"] == "cam02" and r["status"] == "linked" for r in inc["related_cameras"])
    assert inc["objects"]  # weapon recorded
    # pre-event clip exists and is a playable mp4
    clip = Path(tmp_path) / inc["clips"]["cam01"]["path"]
    assert clip.exists() and clip.stat().st_size > 10_000
    assert inc["clips"]["cam01"]["start_ts"] < inc["start_ts"]  # includes frames before detection


def test_companion_counter_evidence_is_explained(tmp_path):
    mgr, log = _run("bat_sports", tmp_path)
    ev = [e for inc in mgr.incidents.values() for s in inc["situations"].values() for e in s["evidence"]]
    # either nothing was raised, or the counter-evidence is visible to the operator
    assert not ev or any(e["key"] == "companion" or e["weight"] is None for e in ev)
