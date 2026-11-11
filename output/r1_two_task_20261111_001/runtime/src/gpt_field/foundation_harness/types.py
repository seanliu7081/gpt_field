"""Shared runtime records. All scene geometry is predicted from public images."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass
class ObjectGeometry:
    object_id: str
    query: str
    points: np.ndarray
    normals: np.ndarray
    status: str
    masks: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def summary(self):
        from .surface import planning_surface

        points = np.asarray(self.points, float).reshape(-1, 3)
        planning_points, planning_evidence = planning_surface(self)
        return jsonable({"id": self.object_id, "query": self.query, "status": self.status,
                         "surface_points": len(points),
                         "visible_surface_centroid_m": np.mean(points, axis=0) if len(points) else None,
                         "visible_surface_bounds_m": [points.min(0), points.max(0)] if len(points) else None,
                         "visible_surface_values_are_raw_hypotheses": True,
                         "raw_surface_points": len(points),
                         "raw_visible_surface_centroid_m": np.mean(points, axis=0) if len(points) else None,
                         "raw_visible_surface_bounds_m": [points.min(0), points.max(0)] if len(points) else None,
                         "planning_surface_points": len(planning_points),
                         "planning_surface_centroid_m": planning_points.mean(0) if len(planning_points) else None,
                         "planning_surface_bounds_m": [planning_points.min(0), planning_points.max(0)] if len(planning_points) else None,
                         "planning_surface_evidence": planning_evidence,
                         "evidence": self.evidence,
                         "complete_object_shape": False})


@dataclass
class SceneSnapshot:
    frame_id: int
    captured_at_s: float
    generated_at_s: float
    objects: dict[str, ObjectGeometry]
    scene_points: np.ndarray
    geometry: Any
    frame: Any
    candidates: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    scene_point_object_ids: np.ndarray | None = None

    def summary(self, now_s=None):
        now = self.generated_at_s if now_s is None else now_s
        return jsonable({"frame_id": self.frame_id, "age_s": max(0., now-self.captured_at_s),
                         "objects": [obj.summary() for obj in self.objects.values()],
                         "candidates": self.candidates, "scene_surface_points": len(self.scene_points),
                         "evidence": self.evidence,
                         "geometry_source": "SAM masks and predicted DA3 depth with public camera calibration",
                         "limitations": ["visible surfaces only", "scores are not error probabilities",
                                         "unknown space is not observed free space"]})


@dataclass
class KeyPose:
    name: str
    pose: np.ndarray
    gripper: str = "keep"
    profile: str = "transit"
    allowed_contacts: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return jsonable({"name": self.name, "pose": self.pose, "gripper": self.gripper,
                         "profile": self.profile, "allowed_contacts": self.allowed_contacts,
                         "metadata": self.metadata})


@dataclass
class OperationPlan:
    operation: dict
    keyposes: list[KeyPose]
    joint_paths: list[list[list[float]]]
    score: float
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return jsonable({"operation": self.operation, "keyposes": [p.to_dict() for p in self.keyposes],
                         "joint_paths": self.joint_paths, "score": self.score, "evidence": self.evidence})
