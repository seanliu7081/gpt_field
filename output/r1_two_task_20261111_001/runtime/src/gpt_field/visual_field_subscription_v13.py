"""Target-aware part contact and two-view local grasp alignment, opt-in V13."""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v12 import (
    VisualFieldSubscriptionClientV12, output_schema as schema_v12,
    shared_guidance as guidance_v12, geometry_semantics,
)

VERSION = "visual-field-rgb-joints-v13-part-contact-image-grasp-v1"
HISTORY_VERSION = "visual-v13-part-contact-recent-two-turns-v1"


def output_schema(controller="hybrid"):
    result = schema_v12(controller)
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    identifier = {"type": "string", "minLength": 1, "maxLength": 80}
    result["properties"]["target_geometry"] = nullable(_object({
        "id": copy.deepcopy(identifier), "center_offset_m": _array(3, -.5, .5),
        "half_extents_m": _array(3, .0001, .5),
        "uncertainty_m": {"type": "number", "minimum": 0, "maximum": .1},
        "allowed_contact": {"type": "string", "enum": ["none", "fingers", "payload_support"]},
    }))
    view = _object({"target_contact_point_uv": _array(2, 0, 1),
                    "tool_pinch_point_uv": _array(2, 0, 1)})
    result["properties"]["contact_alignment"] = nullable(_object({
        "update": {"type": "string", "enum": ["seed", "retain"]},
        "target_id": copy.deepcopy(identifier),
        "external": nullable(copy.deepcopy(view)), "wrist": nullable(copy.deepcopy(view)),
    }))
    result["required"].extend(("target_geometry", "contact_alignment"))
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v12(task, demonstration)
    begin, end = text.index("INTERACTION."), text.index("\n\nHISTORY.")
    return text[:begin] + """TARGET CONTACT. Every response includes target_geometry, independently of nearby_obstacles/obstacles. It describes the WHOLE current primary target using exact target_id, world-axis center_offset_m from this current measured EEF, positive half_extents_m, honest uncertainty_m and allowed_contact=none|fingers|payload_support. Include visible primary targets on all approaches, interactions and closures even when no other objects are nearby. No depth calibration is needed to state a visibly grounded approximate hypothesis; do not invent precision. Null means genuinely unavailable and permits only finite observation/hold/recovery, never approach, interaction or closure. Null does not erase remembered geometry. Preserve every other relevant object hypothesis; the target's small texture-tracking ROI is not its whole extent. Revise geometry from current evidence, never shrink/delete it to bypass refusal.

PART PERMISSION. none permits no intentional target contact. fingers permits only the known fingers/pads against this exact interaction target; the palm/hand is ALWAYS constrained. payload_support permits the independently observed carried object to meet a DIFFERENT destination only during explicit release/interaction; it exempts no robot part. Moving from grasp target to destination must not forget held-object geometry or authorize palm contact. Interaction labels never exempt a whole target. Read target/contact-geometry feedback for refused robot parts and choose a changed useful corridor; do not repeat deeper descent against the same palm conflict. Geometry remains an RGB hypothesis, not a collision certificate.

GRASP ALIGNMENT. Select a local graspable patch/edge/handle within jaw span, not the whole object's visual center. Use a coupled xyz Field approach when needed; high-plane centering followed by blind z descent can miss through parallax. Only in Field approach/align, optional contact_alignment specifies update=seed, exact target_id and BOTH external/wrist {target_contact_point_uv,tool_pinch_point_uv}. The pinch point is the actual open-pad midpoint projected in that view, not the wrist image center or tool tracking patch. Existing external references must observe tool and the same target; wrist also observes that target. Choose visible descriptors compatible with reliable tracked regions. Local Field uses the two-view contact-to-pinch error and measured motion response for bounded correction. Retain uses both view descriptors null with the same target and live references; it cannot resurrect lost tracks or survive changed finger/orientation geometry. Use contact_alignment=null honestly when occluded/unavailable; ordinary bounded Field remains available. Recovery/retreat, transport and release require contact_alignment=null so grasp correction cannot pull them toward contact. Alignment does not authorize contact or automatic closure. A measured strongly incompatible alignment calls for a changed approach; unavailable alignment alone is not proof of misalignment.

INTERACTION. Centering, Field arrival or stalled descent does not prove that a graspable feature is between the jaws. Inspect both views and use needed Direct5 contact correction before explicit gripper closure; no repeated holds or gratuitous yaw. After closing, obtain independent tool/payload co-motion with useful bounded test translation. Field transport requires observed attachment. A tiny/noisy inconclusive test may be extended once if current images/fingers still support grasp; visible miss/slip requires changed grasp geometry. Preserve payload identities when changing destination. For placement, use Field align/xy to align the payload with support using observed grasp offset, then held-XY Field release/z. Direct release has zero XYZ. Choose opening from observed support and reassess separation; no automatic stage/release or idle completion claims.""" + text[end:]


class VisualFieldSubscriptionClientV13(VisualFieldSubscriptionClientV12):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v13", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        geometry, alignment = data["target_geometry"], data["contact_alignment"]
        controller, phase = data["controller"], data["phase"]
        mode = data["motion_context"]["translation_mode"]
        refs = data["visual_references"]
        external, wrist = refs["external"], refs["wrist"]
        if geometry is None:
            closes = (controller == "gripper" and data["gripper_intent"] is not None
                      and data["gripper_intent"]["command"] == "close")
            if controller == "direct" and data["direct_actions"] is not None:
                closes |= any(action[6] > 0 for action in data["direct_actions"])
                moving = any(any(action[:6]) for action in data["direct_actions"])
            else:
                moving = controller == "visual_field"
            if phase == "approach" or mode == "interaction" or closes or (moving and phase != "recover"):
                raise SchemaError("Unavailable target geometry permits only held observation/opening or finite recovery; not approach, interaction or closure")
        else:
            if geometry["id"] != data["target_id"] or not geometry["id"].strip():
                raise SchemaError("Target geometry id must exactly match the nonempty top-level target_id")
            permission = geometry["allowed_contact"]
            if permission != "none" and mode != "interaction":
                raise SchemaError("Target contact permission requires explicit interaction with this target")
            if permission == "payload_support":
                if (phase != "release" or external is None or external["moving_kind"] != "payload"
                        or external["moving_id"] == geometry["id"]):
                    raise SchemaError("Payload support is only release to a distinct target with independent payload/tool references")
        if alignment is not None:
            if controller != "visual_field" or phase not in ("approach", "align"):
                raise SchemaError("Contact alignment descriptors are only selected with visual_field approach or align")
            if geometry is None or alignment["target_id"] != data["target_id"]:
                raise SchemaError("Contact alignment requires matching current target geometry")
            if (external is None or external["moving_kind"] != "tool" or wrist is None
                    or external["target_id"] != data["target_id"] or wrist["target_id"] != data["target_id"]):
                raise SchemaError("Contact alignment requires tool and matching target references in both views")
            seed = alignment["update"] == "seed"
            if any((alignment[view] is not None) != seed for view in ("external", "wrist")):
                raise SchemaError("Contact alignment seed requires both view descriptors; retain requires both null")
        legacy = copy.deepcopy(data)
        for key in ("target_geometry", "contact_alignment"):
            legacy.pop(key)
        result = super()._validate_v5_response(legacy, grid)
        result.update(target_geometry=copy.deepcopy(geometry), contact_alignment=copy.deepcopy(alignment))
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
