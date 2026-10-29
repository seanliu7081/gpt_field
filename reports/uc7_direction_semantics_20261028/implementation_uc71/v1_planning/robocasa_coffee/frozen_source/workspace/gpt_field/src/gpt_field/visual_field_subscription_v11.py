"""Explicit translation purpose for current-image execution screening.

Motion labels express intent, never geometric clearance. V10 persistent goals,
V9 scene meaning and subscription retry accounting remain unchanged.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_feedback_compact_v11 import compact_payload_feedback
from .visual_field_subscription_v10 import (
    VisualFieldSubscriptionClientV10, geometry_semantics, output_schema as schema_v10,
    shared_guidance as guidance_v10,
)

VERSION = "visual-field-rgb-joints-v11-projected-translation-permission-v1"
HISTORY_VERSION = "visual-v11-projected-translation-recent-two-turns-v1"
TRANSLATION_MODES = ("transit", "separate", "interaction")


def output_schema(controller="hybrid"):
    result = schema_v10(controller)
    context = result["properties"]["motion_context"]
    context["properties"].update(
        translation_mode={"type": "string", "enum": list(TRANSLATION_MODES)},
        intended_contact_id={"anyOf": [{"type": "string", "minLength": 1, "maxLength": 80}, {"type": "null"}]},
    )
    context["required"].extend(("translation_mode", "intended_contact_id"))
    return result


def shared_guidance(task, demonstration=False):
    return guidance_v10(task, demonstration) + (
        " TRANSLATION PERMISSION V11: every controller supplies motion_context.translation_mode="
        "transit|separate|interaction and intended_contact_id. transit is noncontact task travel; "
        "separate requests bounded movement away from an unintended neighbor. Both require "
        "intended_contact_id=null. interaction identifies the actual intended contact object and "
        "requires intended_contact_id exactly equal to top-level target_id. These labels never "
        "override a conflicting neighbor or certify contact. The current whole-tool envelope includes "
        "all visible fingers and payload; supply accurate current neighbor regions. Current projected "
        "tool/neighbor conflict requires a changed corridor or bounded separating movement, not "
        "another deeper descent on the same line. Choose useful task movement away from ALL flagged "
        "neighbors, not gratuitous yaw or repeated unchanged descent. A z-only path cannot detour sideways: explicitly "
        "replace an incompatible saved subgoal/axis constraint using current task evidence. "
        "Keep stable IDs for each physical object across obstacle, target and contact roles: when a prior "
        "neighbor becomes intended contact, reuse its exact ID as target_id and intended_contact_id, "
        "never a renamed duplicate. Read visual_feedback.translation_guard before repeating a rejected direction. separate is "
        "not a bypass: local observation must show increased separation; unsupported escape remains "
        "finite and re-seeding/renaming does not renew permission. A weighted metric risk cost or "
        "supported tool tracking is not clearance evidence. Preserve unknown regions and uncertainty; "
        "do not shrink boxes or omit neighbors to force execution. Before close or deeper near-target "
        "descent, identify the intended feature between the jaws in both current views. If the wrist "
        "shows a neighbor under the jaws or the target is not aligned, change the approach corridor. "
        "Retain a target-matched wrist observer when its actual feature remains visible, seed a new "
        "visible feature if needed, and use null honestly when unavailable. Do not invent a centered "
        "[0.5,0.5] target, object pose, depth or camera calibration. Preserve Field for bulk translation; "
        "Direct5 remains necessary contact correction, with accepted commands unchanged."
    )


class VisualFieldSubscriptionClientV11(VisualFieldSubscriptionClientV10):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v11", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return compact_payload_feedback(payload)

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        context = data["motion_context"]
        mode, contact = context["translation_mode"], context["intended_contact_id"]
        if mode == "interaction":
            if contact is None or not contact.strip() or contact != data["target_id"]:
                raise SchemaError("Interaction intended_contact_id must exactly match the nonempty top-level target_id")
        elif contact is not None:
            raise SchemaError("Transit/separate requires intended_contact_id=null")
        legacy = copy.deepcopy(data)
        for key in ("translation_mode", "intended_contact_id"):
            legacy["motion_context"].pop(key)
        result = super()._validate_v5_response(legacy, grid)
        result["motion_context"].update(translation_mode=mode, intended_contact_id=contact)
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
