"""Operator-facing overlay: zones, tracks colored by risk level, weapon candidates, banner."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Level, SituationType as ST

LEVEL_COLOR = {  # BGR
    Level.NORMAL: (90, 200, 90), Level.ATTENTION: (0, 220, 255), Level.WARNING: (0, 140, 255),
    Level.HIGH_RISK: (0, 0, 235), Level.EMERGENCY: (200, 0, 200),
}
TYPE_SHORT = {ST.ASSAULT: "ASSAULT", ST.WEAPON_THREAT: "WEAPON", ST.CHASE: "CHASE", ST.FALL: "FALL",
              ST.INTRUSION: "INTRUSION", ST.LOITERING: "LOITER"}


def annotate(frame: np.ndarray, obs, registry, situations, zones, title: str, clock: str,
             blur_heads: bool = False) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    if blur_heads:
        for p in obs.persons:
            if p.keypoints is not None and p.keypoints[0, 2] > 0.3:
                h = p.bbox[3] - p.bbox[1]
                r = max(4, int(h * 0.12))
                x, y = int(p.keypoints[0, 0]), int(p.keypoints[0, 1])
                x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), r / 2 + 1)
    overlay = img.copy()
    for z in zones:
        pts = np.array(z.polygon, np.int32)
        color = (0, 0, 200) if z.type == "restricted" else (200, 160, 0)
        cv2.fillPoly(overlay, [pts], color)
    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    for z in zones:
        pts = np.array(z.polygon, np.int32)
        color = (0, 0, 200) if z.type == "restricted" else (200, 160, 0)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
        cv2.putText(img, z.id, tuple(pts[0] + [4, 14]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    per_track: dict[int, tuple[Level, list[str]]] = {}
    top = Level.NORMAL
    for s in situations:
        top = max(top, s.level)
        for i, tid in enumerate(s.actors):
            lvl, tags = per_track.get(tid, (Level.NORMAL, []))
            tag = TYPE_SHORT[s.type] + ("*" if i == 0 and len(s.actors) == 2 else "")
            per_track[tid] = (max(lvl, s.level), tags + [tag])
    for p in obs.persons:
        lvl, tags = per_track.get(p.track_id, (Level.NORMAL, []))
        c = LEVEL_COLOR[lvl]
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 if lvl > Level.NORMAL else 1)
        label = f"#{p.track_id}" + (" " + "/".join(dict.fromkeys(tags)) if tags else "")
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        st = registry.tracks.get(p.track_id)
        if st is not None and len(st.hist) > 2:
            pts = np.array([s.foot for s in list(st.hist)[-30:]], np.int32)
            cv2.polylines(img, [pts], False, c, 1, cv2.LINE_AA)
    for o in obs.objects:
        x1, y1, x2, y2 = map(int, o.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, f"{o.label} {o.conf:.2f}", (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(img, f"{title}  {clock}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, top.name, (W - 120, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEVEL_COLOR[top], 2, cv2.LINE_AA)
    return img
