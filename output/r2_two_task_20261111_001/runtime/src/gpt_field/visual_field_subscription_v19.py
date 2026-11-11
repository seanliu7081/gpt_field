"""V18 wire with explicit full-margin candidate-sweep semantics.

Only request guidance and observable version metadata change here. Existing
inventory, candidate grounding, raw history and bounded subscription repairs
remain inherited; the policy/guard owns all actual motion admission.
"""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v18 import (
    VisualFieldSubscriptionClientV18, output_schema,
    shared_guidance as guidance_v18, geometry_semantics as semantics_v18,
    CANDIDATE_LABEL,
)

VERSION = "visual-field-rgb-joints-v19-full-margin-candidate-sweep-v1"
HISTORY_VERSION = "visual-v19-full-margin-recent-two-turns-v1"


def geometry_semantics():
    result = semantics_v18()
    result.update({
        "version": "full-margin-model-candidate-sweep-v19",
        "uncertainty_margin_semantics": "Declared spatial model margin, not a calibrated statistical standard deviation or confidence bound.",
        "candidate_sweep_semantics": "Check the current actual candidate segment against full-uncertainty configuration-space unions of known tool and declared payload geometry; preserve every scene hypothesis.",
        "model_separation_semantics": "A positive full-margin model-sweep separation is conditional model geometry evidence only, never physical clearance certification; image overlap alone does not override that evidence by multiplying the margin again.",
        "retained_constraints": "Actual near/unknown/unsupported geometry, target part-specific contact, current response and attachment checks, and all existing shared/directional travel debt remain enforced.",
        "clearance_certified": False,
    })
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v18(task, demonstration)
    text = text.replace(
        "Local coverage checks the actual candidate tool sweep, not every object anywhere in the planning cube.",
        "Local coverage and translation admission check the current actual candidate segment against the full-uncertainty "
        "configuration-space union for the known tool and declared payload. uncertainty_m is a declared spatial "
        "model margin, NOT a calibrated statistical standard deviation; do not treat it as a measured Gaussian sigma "
        "or repeatedly multiply an already included margin to infer proximity. An exact full-margin model-separated "
        "sweep may support useful Field travel despite 2D overlap or a remote object's view boundary. This is "
        "conditional on the scene/tool hypotheses, never physical clearance certification. Read the actual current "
        "admission result: genuine near, unknown or unsupported cases still need their existing observation, contact "
        "and changed-path checks. Preserve all nominal extents and uncertainty; never shrink, delete or rename "
        "geometry to make this test pass. A model-separated path does not reset response, attachment or shared/directional "
        "travel allowance. When the local checks admit the segment, reuse the persistent Field goal for useful task "
        "progress instead of adding calibration requests solely because its projected boxes overlap. Reference "
        "candidate review remains required when independently requested by actual motion evidence.")
    return text


class VisualFieldSubscriptionClientV19(VisualFieldSubscriptionClientV18):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v19", schema=output_schema())
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
