"""V9 scene semantics: hard nominal objects and separate uncertainty cost.

The V8 wire shape and role/geometry validators are unchanged. Neither an RGB
nominal box nor its uncertainty cost is a calibrated geometric measurement.
"""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v8 import (
    VisualFieldSubscriptionClientV8, output_schema, shared_guidance as guidance_v8,
)

VERSION = "visual-field-rgb-joints-v9-nominal-core-uncertainty-cost-v1"
HISTORY_VERSION = "visual-v9-nominal-core-recent-two-turns-v1"


def geometry_semantics():
    """Fresh observable metadata for policy feedback and experiment manifests."""
    return {
        "version": "nominal-core-uncertainty-traversal-v9",
        "obstacles": "Visually localized objects with approximate nominal metric boxes; tool/payload-inflated nominal cores remain hard excluded.",
        "visible_uncertainty": "Separate nonnegative traversal cost outside hard nominal cores; not calibrated collision probability or guaranteed distance.",
        "soft_geometry_permission": "Requires present/none_relevant assessment, current whole-tool image ROI, and a current named image ROI matching every obstacles entry. Missing, unobservable or inconsistent support retains full hard uncertainty inflation.",
        "unknown_regions": "Genuinely occluded, unseen or unlocalized regions; fully hard inflated by tool/payload radius and declared uncertainty.",
        "geometry_source": "RGB model hypotheses and measured robot state, without depth or supplied environment calibration",
        "obstacle_identity_rule": "Preserve every listed nearby obstacle ID and box; never reclassify, shrink or delete to evade rejection.",
        "unobservable_scene_limits": {"max_travel_m": .015, "command_limit_m": .005},
        "clearance_certified": False,
    }


def shared_guidance(task, demonstration=False):
    text = guidance_v8(task, demonstration)
    start = text.index("GEOMETRY:")
    end = text.index("REFERENCE CONTRACT:", start)
    text = text[:start] + (
        "GEOMETRY V9: obstacles are visually localized objects with approximate NOMINAL metric boxes "
        "estimated from RGB. Each world-axis-aligned box has a stable unique id, center_offset_m from "
        "the SAME measured EEF, positive half_extents_m and honest uncertainty_m. The nominal object "
        "core inflated by tool/payload radius is HARD excluded. Outside that core, visible-object "
        "uncertainty contributes a separate NONNEGATIVE traversal cost ONLY with present/none_relevant "
        "assessment, current whole-tool ROI and a current named image ROI matching EVERY obstacles "
        "entry. Missing, unobservable or inconsistent support retains FULL hard uncertainty inflation. "
        "The cost is not calibrated collision probability, measured distance or guaranteed "
        "clearance. unknown_regions means genuinely occluded, unseen or unlocalized regions; these "
        "remain fully HARD inflated by tool/payload radius and declared uncertainty. Lack of calibrated "
        "depth alone does not make every visible object an unknown region. Classify from actual image "
        "evidence. Preserve static world positions as EEF moves, every named nearby obstacle, nominal "
        "extent and uncertainty. Never reclassify, shrink or delete an object/region simply to evade "
        "rejection. Revise only from new contradictory visual evidence. The executor reports these "
        "semantics in visual_feedback.geometry_semantics; a feasible hypothesis route is never "
        "a collision certificate. "
    ) + text[end:]
    return text + (
        " TASK PROGRESS: use an appropriate bounded Field translation toward the current task from "
        "measured robot state; let local planning follow that goal while monitoring observations. "
        "Do not replace useful movement with endless static holds or calibration. At interaction "
        "range, inspect actual jaw/feature engagement in both views and use necessary bounded Direct5 "
        "contact correction. Request explicit gripper close/open when current engagement/support "
        "evidence justifies it; local arrival or tracking uncertainty alone does not justify closure. "
        "Maintain the whole-tool/payload envelope and relevant neighbor regions in motion_context. "
        "No task-specific poses, calibration, depth or environment truth may be invented."
    )


class VisualFieldSubscriptionClientV9(VisualFieldSubscriptionClientV8):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v9", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
