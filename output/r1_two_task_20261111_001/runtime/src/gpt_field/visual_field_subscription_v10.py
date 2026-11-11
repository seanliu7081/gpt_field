"""Opt-in persistent task subgoals across visual/contact repair requests.

The client validates explicit lifecycle operations. The policy ledger alone
resolves measured arrival and persistent world goals; raw model output is kept.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v9 import (
    VisualFieldSubscriptionClientV9, geometry_semantics, output_schema as schema_v9,
    shared_guidance as guidance_v9,
)

VERSION = "visual-field-rgb-joints-v10-persistent-subgoal-v1"
HISTORY_VERSION = "visual-v10-persistent-subgoal-recent-two-turns-v1"
SUBGOAL_ACTIONS = ("new", "retain", "replace", "finish", "none")


def output_schema(controller="hybrid"):
    result = schema_v9(controller)
    result["properties"]["subgoal_update"] = _object({
        "action": {"type": "string", "enum": list(SUBGOAL_ACTIONS)},
        "subgoal_id": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 80}, {"type": "null"}]},
        "reason": {"type": "string", "maxLength": 400},
    })
    result["required"].append("subgoal_update")
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v9(task, demonstration)
    text = text.replace(
        "After closing, use a short Direct test with independent payload/tool references; Field transport requires observed co-motion and payload moving_kind.",
        "After closing, use bounded useful translation with independent payload/tool references. "
        "Use Field for a measured diagnostic translation/test lift when the local execution conditions "
        "permit; Direct5 remains available for necessary contact correction or an initial short test. "
        "Field transport requires observed co-motion and payload moving_kind.")
    return text + (
        " SUBGOAL LIFECYCLE: every response includes subgoal_update={action,subgoal_id,reason}. "
        "The first Field intent uses new with a unique nonempty ID; the ledger stores its world goal, "
        "phase, target and active motion_axes. Field retain uses that exact ID/phase/target/axes and "
        "continues the SAME stored world goal. Fresh goal_offset_m on retain is logged but ignored; "
        "do not repeatedly add vertical clearance or recenter the subgoal after each repair. "
        "Direct/gripper reference, attachment or contact repair uses retain with the active ID; its "
        "interaction phase/target may differ without deleting the saved goal. Read "
        "visual_feedback.subgoal_ledger for measured remaining distance and arrival. After actual "
        "local arrival, choose the next task stage/new goal; arrival is NOT task completion. "
        "finish is Direct/gripper only and closes that ID only after measured arrival. new cannot "
        "discard an unfinished goal. To change an unfinished goal/target/axes, explicitly replace "
        "the SAME active ID with a nonempty reason grounded in new visual/task evidence. "
        "Do not invent poses or replace merely because correspondence or attachment needs repair. "
        "none has subgoal_id=null and never clears a saved goal. Field permits new/retain/replace; "
        "Direct/gripper permits retain/finish/none. Use one bounded informative task movement and "
        "evaluate its evidence; avoid endless micro-test or static-hold requests."
    )


class VisualFieldSubscriptionClientV10(VisualFieldSubscriptionClientV9):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v10", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        update = data["subgoal_update"]
        action, identity = update["action"], update["subgoal_id"]
        if action == "none":
            if identity is not None:
                raise SchemaError("Subgoal none requires subgoal_id=null")
        elif identity is None or not identity.strip():
            raise SchemaError("Subgoal new/retain/replace/finish requires a nonempty subgoal_id")
        if action == "replace" and not update["reason"].strip():
            raise SchemaError("Subgoal replace requires an explicit nonempty evidence-based reason")
        allowed = ("new", "retain", "replace") if data["controller"] == "visual_field" else ("retain", "finish", "none")
        if action not in allowed:
            raise SchemaError("Field subgoal action must be new/retain/replace; Direct/gripper must be retain/finish/none")
        legacy = copy.deepcopy(data)
        legacy.pop("subgoal_update")
        result = super()._validate_v5_response(legacy, grid)
        result["subgoal_update"] = copy.deepcopy(update)
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
