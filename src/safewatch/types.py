"""Core data types shared across perception, features, risk engine and incidents."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

# COCO-17 keypoint order (used by both simulator and YOLO-pose backend)
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

# Object labels treated as "dangerous object candidates".
WEAPON_LABELS = {"knife", "scissors", "bat"}


class Level(IntEnum):
    NORMAL = 0
    ATTENTION = 1
    WARNING = 2
    HIGH_RISK = 3
    EMERGENCY = 4

    @classmethod
    def parse(cls, s: str) -> "Level":
        return cls[s.upper()]


class SituationType:
    ASSAULT = "ASSAULT"            # 폭행/싸움
    WEAPON_THREAT = "WEAPON_THREAT"  # 흉기/위험물 위협
    CHASE = "CHASE"                # 추격
    FALL = "FALL"                  # 쓰러짐
    INTRUSION = "INTRUSION"        # 침입
    LOITERING = "LOITERING"        # 비정상 배회

    ALL = [ASSAULT, WEAPON_THREAT, CHASE, FALL, INTRUSION, LOITERING]
    KO = {
        ASSAULT: "폭행/싸움", WEAPON_THREAT: "흉기·위험물 위협", CHASE: "추격",
        FALL: "쓰러짐", INTRUSION: "침입", LOITERING: "비정상 배회",
    }


@dataclass
class PersonObs:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    conf: float
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, conf

    @property
    def foot(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2])

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


@dataclass
class ObjectObs:
    label: str
    bbox: tuple[float, float, float, float]
    conf: float

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


@dataclass
class FrameObs:
    camera_id: str
    frame_idx: int
    ts: float  # seconds since stream start (media time)
    width: int
    height: int
    persons: list[PersonObs] = field(default_factory=list)
    objects: list[ObjectObs] = field(default_factory=list)
