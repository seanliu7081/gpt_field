"""V19 wire with an explicit, narrowly gated metric-transit contract.

The local executor alone admits an exact current segment. Weak RGB response
never becomes validated motion evidence, and no permission is encoded by
altering the model's inventory, raw wire or inherited reference budgets.
"""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v19 import (
    VisualFieldSubscriptionClientV19, output_schema,
    shared_guidance as guidance_v19, geometry_semantics as semantics_v19,
    CANDIDATE_LABEL,
)

VERSION = "visual-field-rgb-joints-v20-proprioceptive-metric-transit-v1"
HISTORY_VERSION = "visual-v20-proprioceptive-transit-recent-two-turns-v1"


def geometry_semantics():
    result = semantics_v19()
    result.update({
        "version": "candidate-bound-proprioceptive-metric-transit-v20",
        "proprioceptive_transit": {
            "scope": "Only weak-image-response allowance exhaustion may request exact current full-margin model-separated empty-tool Field transit outside the target contact band, without payload, contact refinement or goal revision.",
            "control": "Measured robot state tracks the existing fixed world goal; image-based goal correction is not used by this branch.",
            "maximum_command_m": .005,
            "original_intent_action_travel_budgets_unchanged": True,
            "retained_constraints": "Lost, contradicted, stale or changed references remain blocking; scene, target/palm, near-contact, attachment and inherited physical allowances remain enforced.",
            "image_response_validated": False,
            "reference_allowance_reset": False,
            "clearance_certified": False,
        },
    })
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v19(task, demonstration)
    text = text.replace(
        "A model-separated path does not reset response, attachment or shared/directional travel allowance.",
        "A model-separated path does not reset response, attachment or shared/directional travel allowance. "
        "Only when weak image-response allowance exhaustion is the sole reference blocker may local checks "
        "admit a bounded, exact current Field candidate as proprioceptive metric transit. This is empty-tool "
        "motion only, outside the target contact band, with no payload, contact refinement or goal revision; "
        "it requires target_contact=none and translation_mode=transit|separate. "
        "tool and anchor tracks must remain reliable. Each admitted command is at most 5 mm and the original "
        "intent action/travel budgets remain unchanged. This branch tracks "
        "the existing fixed world goal from measured robot state; it does not apply image-based goal correction, "
        "validate the visual response, or replenish any reference allowance. Lost, contradicted, stale or changed "
        "references do not qualify. Full-margin scene sweeps, target/palm and near-contact checks, attachment "
        "requirements and all remaining physical allowances still apply. Read proprioceptive_transit and "
        "field.proprioceptive_transit_admission for the actual local admission/cap "
        "feedback; selecting a Field goal alone never authorizes this branch. Retain a useful task goal when "
        "admitted, and change the corridor or grounded reference when the specific evidence requires it; "
        "do not fabricate geometry, grasp or completion to bypass a refusal.")
    return text


class VisualFieldSubscriptionClientV20(VisualFieldSubscriptionClientV19):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v20", schema=output_schema())
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
