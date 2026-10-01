"""Procedural COCO-17 skeleton generator for the synthetic CCTV simulator."""
from __future__ import annotations

import math

import numpy as np

# Standing pose, units of body height, origin at foot center, y up is negative (image coords).
_STAND = np.array([
    [0.00, -0.93],  # nose
    [-0.02, -0.95], [0.02, -0.95],  # eyes
    [-0.04, -0.94], [0.04, -0.94],  # ears
    [-0.11, -0.82], [0.11, -0.82],  # shoulders
    [-0.14, -0.65], [0.14, -0.65],  # elbows
    [-0.15, -0.50], [0.15, -0.50],  # wrists
    [-0.07, -0.50], [0.07, -0.50],  # hips
    [-0.07, -0.27], [0.07, -0.27],  # knees
    [-0.07, -0.02], [0.07, -0.02],  # ankles
], dtype=float)


def make_skeleton(
    foot: np.ndarray,
    height_px: float,
    *,
    gait_phase: float = 0.0,
    gait_amp: float = 0.0,
    strike: float = 0.0,
    strike_dir: float = 1.0,
    fall_angle_deg: float = 0.0,
    fall_dir: float = 1.0,
) -> np.ndarray:
    """Return (17, 2) pixel keypoints.

    gait_amp: 0 (standing) .. ~0.12 (walk) .. ~0.22 (run)
    strike: 0..1 arm extension towards strike_dir (+1 right / -1 left)
    fall_angle_deg: 0 upright .. 90 lying, rotating around foot towards fall_dir
    """
    k = _STAND.copy()
    s = math.sin(gait_phase)
    if gait_amp > 0:
        # legs swing opposite, arms swing opposite to legs
        k[13, 0] += gait_amp * 0.6 * s
        k[15, 0] += gait_amp * s
        k[14, 0] -= gait_amp * 0.6 * s
        k[16, 0] -= gait_amp * s
        k[7, 0] -= gait_amp * 0.4 * s
        k[9, 0] -= gait_amp * 0.8 * s
        k[8, 0] += gait_amp * 0.4 * s
        k[10, 0] += gait_amp * 0.8 * s
        # lean forward when running
        lean = min(gait_amp, 0.25) * 0.15
        k[:11, 0] += lean * np.sign(strike_dir)
    if strike > 0:
        # extend the arm on the strike side (right arm when striking right)
        sh, el, wr = (6, 8, 10) if strike_dir > 0 else (5, 7, 9)
        base = k[sh].copy()
        reach = 0.12 + 0.30 * strike
        k[el] = base + np.array([strike_dir * reach * 0.5, 0.02])
        k[wr] = base + np.array([strike_dir * reach, 0.0])
    if fall_angle_deg > 0:
        a = math.radians(fall_angle_deg) * fall_dir
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        k = k @ rot.T
    return foot[None, :] + k * height_px
