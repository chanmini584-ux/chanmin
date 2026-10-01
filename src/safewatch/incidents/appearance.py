"""Appearance cues for cross-camera candidate matching and report descriptions.

Only coarse clothing color is used (no face, no biometric identifiers).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import KP, PersonObs


def torso_region(frame: np.ndarray, p: PersonObs) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    if p.keypoints is not None and np.all(p.keypoints[[5, 6, 11, 12], 2] >= 0.3):
        pts = p.keypoints[[KP["l_shoulder"], KP["r_shoulder"], KP["r_hip"], KP["l_hip"]], :2]
        x1, y1 = np.maximum(pts.min(0).astype(int), 0)
        x2, y2 = np.minimum(pts.max(0).astype(int) + 1, [W, H])
        # shrink towards the center to avoid background
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hw, hh = max(2, (x2 - x1) // 3), max(2, (y2 - y1) // 3)
        x1, x2, y1, y2 = cx - hw, cx + hw, cy - hh, cy + hh
    else:
        bx1, by1, bx2, by2 = map(int, p.bbox)
        h = by2 - by1
        x1, x2 = bx1 + (bx2 - bx1) // 4, bx2 - (bx2 - bx1) // 4
        y1, y2 = by1 + int(h * 0.2), by1 + int(h * 0.5)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def color_hist(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return h / (h.sum() + 1e-9)


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-based similarity in [0, 1]."""
    d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - d))


def color_name(patch: np.ndarray) -> str:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h, s, v = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if v < 60:
        return "검정"
    if s < 40:
        return "흰색" if v > 190 else "회색"
    hue = h * 2  # 0..360
    for limit, name in [(15, "빨강"), (40, "주황"), (70, "노랑"), (160, "초록"), (200, "청록"),
                        (260, "파랑"), (300, "보라"), (340, "분홍"), (361, "빨강")]:
        if hue < limit:
            return name
    return "기타"
