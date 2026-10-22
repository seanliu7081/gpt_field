"""V16 native inventory wire with local-sweep coverage and concise live feedback."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v16 import (
    VisualFieldSubscriptionClientV16, output_schema, geometry_semantics,
    shared_guidance as guidance_v16,
)
from .visual_feedback_compact_v17 import compact_payload_feedback

VERSION = "visual-field-rgb-joints-v17-candidate-sweep-partial-support-v1"
HISTORY_VERSION = "visual-v17-inventory-compact-recent-two-turns-v1"


def shared_guidance(task, demonstration=False):
    text = guidance_v16(task, demonstration)
    text = text.replace("Current/remembered inventory geometry and projected support constrain every controller;",
        "Current/remembered full world geometry and full projected extents remain constraints in every controller;")
    text = text.replace("Read scene_inventory feedback and choose meaningful bounded Field progress or informative task correction.",
        "Read scene_inventory and scene_coverage feedback. current_visible_roi_uv/partial_current_support means "
        "currently tracked visible texture, NOT complete object visibility or clearance; current_roi_uv is legacy "
        "declared-rectangle availability; extent_visibility=boundary_limited takes precedence even when "
        "current_roi_uv is nonnull and requires the full-uncertainty candidate sweep. Full projected extents and world hypotheses remain retained. "
        "Local coverage checks the actual candidate tool sweep, not every object anywhere in the planning cube. "
        "A remote or partly cropped object is not automatically absent or automatically blocking: follow the "
        "specific candidate conflict and support reason. Reuse a valid persistent Field goal for useful motion; "
        "do not repeatedly replace it or request tiny waypoint batches solely because a remote box crosses a view edge. "
        "A genuinely conflicting/unsupported candidate still requires a changed corridor or useful observed correction.")
    text += ("\n\nFEEDBACK PRESENTATION. Current scene_inventory.entries[].current_geometry retains every object's "
        "exact hypothesis relative to the current measured EEF; omitted world duplicates and old revision details "
        "remain in local traces. feedback_ref.path plus keys means exactly those equal keys in the named record "
        "within the SAME visual_feedback, with any differing snapshot values in overrides. An older observation may use feedback_patch: start from its named latest observation in this request and apply the path replacements/deletions to recover the exact older snapshot. The latest feedback is self-contained. Never treat old "
        "snapshot support as new evidence. All current failures, contact permissions, attachment, goals and "
        "remaining allowances still apply. No geometry or partial texture is a collision certificate.")
    return text


class VisualFieldSubscriptionClientV17(VisualFieldSubscriptionClientV16):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v17", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return compact_payload_feedback(payload)

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
