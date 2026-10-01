"""Site (cameras/zones/topology) and policy (risk thresholds) configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITE = ROOT / "configs" / "site.demo.yaml"
DEFAULT_POLICY = ROOT / "configs" / "policy.default.yaml"


@dataclass
class Zone:
    id: str
    name: str
    type: str  # restricted | loiter_watch
    polygon: list[tuple[float, float]]
    active_hours: Optional[tuple[int, int]] = None  # [start, end) local hour, may wrap midnight

    def active_at(self, hour: int) -> bool:
        if not self.active_hours:
            return True
        s, e = self.active_hours
        return s <= hour < e if s < e else (hour >= s or hour < e)


@dataclass
class Camera:
    id: str
    name: str
    address: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    place_type: str = "street"
    image_right_bearing: float = 90.0
    neighbors: dict[str, dict] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    zones: list[Zone] = field(default_factory=list)
    sim: dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None  # file path or rtsp:// URL


@dataclass
class Site:
    name: str
    timezone: str
    cameras: dict[str, Camera]

    def camera(self, cid: str) -> Camera:
        if cid not in self.cameras:
            # unknown camera (e.g. ad-hoc video file): create a minimal entry
            self.cameras[cid] = Camera(cid, cid)
        return self.cameras[cid]


def load_site(path: Optional[str | Path] = None) -> Site:
    data = yaml.safe_load(Path(path or DEFAULT_SITE).read_text(encoding="utf-8"))
    cams = {}
    for c in data.get("cameras", []):
        zones = [Zone(z["id"], z["name"], z["type"], [tuple(p) for p in z["polygon"]],
                      tuple(z["active_hours"]) if z.get("active_hours") else None)
                 for z in c.get("zones", [])]
        cams[c["id"]] = Camera(
            c["id"], c.get("name", c["id"]), c.get("address", ""), c.get("lat"), c.get("lon"),
            c.get("place_type", "street"), float(c.get("image_right_bearing", 90)),
            c.get("neighbors", {}) or {}, c.get("analysis", {}) or {}, zones, c.get("sim", {}) or {},
            c.get("source"))
    s = data.get("site", {})
    return Site(s.get("name", "site"), s.get("timezone", "Asia/Seoul"), cams)


def load_policy(path: Optional[str | Path] = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_POLICY).read_text(encoding="utf-8"))
