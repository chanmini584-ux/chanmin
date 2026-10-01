from __future__ import annotations

import numpy as np


def point_in_polygon(pt, poly) -> bool:
    """Ray casting. poly: list of (x, y)."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear soft membership: 0 at <=lo, 1 at >=hi. Works for lo > hi (decreasing)."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    v = (x - lo) / (hi - lo)
    return float(min(1.0, max(0.0, v)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v, dtype=float)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def noisy_or(values) -> float:
    p = 1.0
    for v in values:
        p *= 1.0 - max(0.0, min(1.0, v))
    return 1.0 - p


def compass_from_vector(v: np.ndarray, image_right_bearing: float = 90.0) -> str:
    """Map an image-plane direction to a compass label.

    image_right_bearing: compass bearing (deg) of the image +x direction for this camera.
    Image +y points down (towards camera) — treated as bearing + 90.
    """
    if np.linalg.norm(v) < 1e-9:
        return "정지"
    ang = np.degrees(np.arctan2(v[1], v[0]))  # 0 = image right, 90 = image down
    bearing = (image_right_bearing + ang) % 360
    names = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    return names[int(((bearing + 22.5) % 360) // 45)] + "쪽"
