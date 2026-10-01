"""Score calibration and level mapping. All thresholds come from the policy file."""
from __future__ import annotations

import numpy as np

from ..types import Level


class Policy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.features = cfg.get("features", {})
        self.situations = cfg.get("situations", {})
        self.context = cfg.get("context", {})
        cal = cfg.get("calibration", {}) or {}
        self.cal_method = cal.get("method", "identity")
        self.cal_x = np.array(cal.get("x") or [], float)
        self.cal_y = np.array(cal.get("y") or [], float)
        lv = cfg["levels"]
        self.thresholds = sorted(((Level.parse(k), float(v)) for k, v in lv.items()),
                                 key=lambda kv: kv[0])
        self.alerting = cfg.get("alerting", {})
        self.incident = cfg.get("incident", {})
        self.popup_min_level = Level.parse(self.alerting.get("popup_min_level", "WARNING"))
        self.half_life = float(self.alerting.get("decay_half_life_s", 4.0))

    def f(self, key, default=None):
        return self.features.get(key, default)

    def s(self, typ: str) -> dict:
        return self.situations.get(typ, {})

    def calibrate(self, raw: float) -> float:
        if self.cal_method == "isotonic" and len(self.cal_x) >= 2:
            return float(np.interp(raw, self.cal_x, self.cal_y))
        return float(raw)

    def context_factor(self, place_type: str, hour: int) -> float:
        f = float((self.context.get("place_factor") or {}).get(place_type, 1.0))
        nh = self.context.get("night_hours")
        if nh:
            s, e = nh
            night = (s <= hour < e) if s < e else (hour >= s or hour < e)
            if night:
                f *= float(self.context.get("night_factor", 1.0))
        return f

    @staticmethod
    def apply_context(score: float, factor: float) -> float:
        return 1.0 - (1.0 - min(max(score, 0.0), 1.0)) ** factor

    def level(self, score: float, typ: str | None = None, alone: bool = False) -> Level:
        lvl = Level.NORMAL
        for l, th in self.thresholds:
            if score >= th:
                lvl = l
        if typ:
            sc = self.s(typ)
            cap_key = "max_level_alone" if alone and "max_level_alone" in sc else "max_level"
            if cap_key in sc:
                lvl = min(lvl, Level.parse(sc[cap_key]))
        return lvl

    def threshold(self, level: Level) -> float:
        return dict(self.thresholds)[level]
