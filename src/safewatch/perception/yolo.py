"""Real perception backend: Ultralytics YOLO (pose + built-in ByteTrack/BoT-SORT) + COCO objects.

License note: Ultralytics is AGPL-3.0. This adapter is the only module importing it, so the
backend can be swapped (e.g. RF-DETR / RTMPose, Apache-2.0) without touching the rest of the system.

Model weights are downloaded by Ultralytics from its GitHub releases on first use.
COCO classes used as dangerous-object candidates: 34 baseball bat, 43 knife, 76 scissors.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import FrameObs, ObjectObs, PersonObs

COCO_WEAPONS = {34: "bat", 43: "knife", 76: "scissors"}


class YoloPerception:
    def __init__(self, pose_model: str = "yolo11n-pose.pt", det_model: Optional[str] = "yolo11n.pt",
                 tracker: str = "bytetrack.yaml", person_conf: float = 0.35, obj_conf: float = 0.25,
                 imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.pose = YOLO(pose_model)
        self.det = YOLO(det_model) if det_model else None
        self.tracker = tracker
        self.person_conf, self.obj_conf = person_conf, obj_conf
        self.imgsz, self.device = imgsz, device

    def __call__(self, cam_id: str, frame_idx: int, ts: float, image: np.ndarray, aux=None) -> FrameObs:
        H, W = image.shape[:2]
        obs = FrameObs(cam_id, frame_idx, ts, W, H)
        r = self.pose.track(image, persist=True, tracker=self.tracker, conf=self.person_conf,
                            imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kps = None
            if r.keypoints is not None and r.keypoints.data is not None:
                kps = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
            for i in range(len(ids)):
                k = kps[i].astype(float) if kps is not None and len(kps) > i else None
                obs.persons.append(PersonObs(int(ids[i]), tuple(map(float, boxes[i])), float(confs[i]), k))
        if self.det is not None:
            d = self.det.predict(image, classes=list(COCO_WEAPONS), conf=self.obj_conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
            if d.boxes is not None:
                for b, c, cl in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy(),
                                    d.boxes.cls.int().cpu().numpy()):
                    obs.objects.append(ObjectObs(COCO_WEAPONS[int(cl)], tuple(map(float, b)), float(c)))
        return obs
