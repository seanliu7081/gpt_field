"""Explicit local obstacle review and task-directed motion purpose.

Image regions and metric boxes remain model hypotheses. The client validates
their wire representation; the runtime decides whether a rotation may execute.
V7's bounded, accounted subscription retry and raw-history semantics are reused.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v7 import (
    VisualFieldSubscriptionClientV7, output_schema as schema_v7, shared_guidance as guidance_v7,
)

VERSION = "visual-field-rgb-joints-v8-motion-context-obstacle-review-v1"
HISTORY_VERSION = "visual-v8-motion-context-recent-two-turns-v1"
MOTION_PURPOSES = ("task_translation", "contact_correction", "articulation", "task_reorientation", "reference_repair")
OBSTACLE_ASSESSMENTS = ("present", "none_relevant", "unobservable")


def motion_context_schema():
    return _object({
        "purpose": {"type": "string", "enum": list(MOTION_PURPOSES)},
        "tool_external_roi_uv": {"anyOf": [_array(4, 0, 1), {"type": "null"}]},
        "nearby_obstacles": {"type": "array", "maxItems": 6, "items": _object({
            "id": {"type": "string", "minLength": 1, "maxLength": 80},
            "external_roi_uv": _array(4, 0, 1),
        })},
        "obstacle_assessment": {"type": "string", "enum": list(OBSTACLE_ASSESSMENTS)},
    })


def output_schema(controller="hybrid"):
    result = schema_v7(controller)
    result["properties"]["motion_context"] = motion_context_schema()
    result["required"].append("motion_context")
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v7(task, demonstration)
    start = text.index("REFERENCE HEALTH:")
    end = text.index("INTERACTION:", start)
    text = text[:start] + (
        "REFERENCE HEALTH: reliable patch matching is not proof of controlled-body motion. Read "
        "visual_feedback.visual.response_evidence and reference_health. Weakly observable or unvalidated "
        "response is inconclusive, not proof of a wrong feature or physical misalignment. Continue "
        "useful bounded Field movement or a visually justified task/contact correction. Do not rotate "
        "just to repair tracking. A genuinely contradicted/unavailable correspondence calls for a "
        "different visible distal tool/payload part or an informative bounded translation; retain or "
        "renaming cannot erase evidence. Use moving gripper housing/finger, not fixed robot base. "
        "Optional current-image labels M=claimed moving feature, T=stationary anchor, G=independent "
        "tool are hypotheses, not verified identities. Check against both original views. "
    ) + text[end:]
    return text + (
        " MOTION CONTEXT: every response supplies purpose=task_translation|contact_correction|articulation|"
        "task_reorientation|reference_repair. Choose the actual purpose; weak tracking alone never "
        "justifies yaw. tool_external_roi_uv bounds the WHOLE currently visible tool, open fingers "
        "and held payload, not a tracking patch; use null when unavailable. nearby_obstacles contains "
        "up to six relevant neighboring objects with stable id and external_roi_uv, including neighbors "
        "outside the wrist crop. Exclude intended contact surfaces from unintended-obstacle roles. "
        "All ROIs are ordered positive-area unit [u0,v0,u1,v1]. obstacle_assessment=present requires "
        "nonempty neighbors. For Field EVERY listed neighbor ID needs its metric box in obstacles "
        "or unknown_regions, including when other geometry is unobservable. none_relevant requires "
        "an empty list after reviewing the current route "
        "and tool extent. unobservable explicitly records uncertain geometry, permits listing known "
        "neighbors, and calls for short bounded progress/informative observation, not unlimited clear "
        "travel. With unobservable Field geometry, local scene_execution_limits cap max_travel_m at "
        "0.015 and command_limit_m at 0.005 while preserving your goal and boxes. Never invent "
        "obstacles/walls, suppress a visible neighbor or lower uncertainty to "
        "obtain acceptance. The local projected_rotation_risk screen may veto Direct rotation that "
        "is reference-only, lacks tool/neighbor assessment, or overlaps a neighboring image region. "
        "A fixed EEF center does not fix finger positions while rotating. Make any needed task "
        "reorientation only with current swept-tool clearance evidence; otherwise use model-selected "
        "Field repositioning or bounded nonrotating Direct task correction. The image screen is a "
        "risk heuristic, never a calibrated 3D swept-volume bound or collision certificate."
    )


def _ordered_roi(roi):
    if roi is not None and (roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise SchemaError("Motion-context ROIs require ordered positive-area unit bounds")


class VisualFieldSubscriptionClientV8(VisualFieldSubscriptionClientV7):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v8", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        context = data["motion_context"]
        _ordered_roi(context["tool_external_roi_uv"])
        neighbors = context["nearby_obstacles"]
        for neighbor in neighbors:
            _ordered_roi(neighbor["external_roi_uv"])
        identities = [neighbor["id"] for neighbor in neighbors]
        if any(not identity.strip() for identity in identities) or len(identities) != len(set(identities)):
            raise SchemaError("Motion-context neighbor IDs must be nonempty and unique")
        assessment = context["obstacle_assessment"]
        if assessment == "present" and not neighbors:
            raise SchemaError("Present obstacle assessment requires named neighboring image regions")
        if assessment == "none_relevant" and neighbors:
            raise SchemaError("No-relevant-obstacle assessment requires an empty neighbor list")
        if data["controller"] == "visual_field" and data["field_intent"] is not None:
            intent = data["field_intent"]
            box_ids = {box["id"] for group in ("obstacles", "unknown_regions") for box in intent[group]}
            if not set(identities).issubset(box_ids):
                raise SchemaError("Every listed image neighbor needs a matching Field obstacle or unknown-region ID")
        legacy = copy.deepcopy(data)
        legacy.pop("motion_context")
        result = super()._validate_v5_response(legacy, grid)
        result["motion_context"] = copy.deepcopy(context)
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
