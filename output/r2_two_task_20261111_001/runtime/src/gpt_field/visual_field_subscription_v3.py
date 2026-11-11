"""V3 semantic references and explicit, bounded local information acquisition."""
from __future__ import annotations

import copy
import math

from .astra import SchemaError
from .visual_field_subscription import _array, _object, _validate, shared_guidance as guidance_v1
from .visual_field_subscription_v2 import (
    VisualFieldSubscriptionClientV2, output_schema as schema_v2, _text_record,
)

VERSION = "visual-field-rgb-joints-v3"
HISTORY_VERSION = "visual-v3-persistent-evidence-recent-two-turns-v1"


def output_schema(controller="hybrid"):
    result = schema_v2(controller)
    identifier = {"type": "string", "minLength": 1, "maxLength": 80}
    bounds = _object({"lower_uv": _array(2, 0, 1), "upper_uv": _array(2, 0, 1)})
    # V2 external/wrist references share a schema dictionary; mutate once.
    reference = result["properties"]["visual_references"]["properties"]["external"]["anyOf"][0]
    del reference["properties"]["moving_radius_uv"]
    reference["required"].remove("moving_radius_uv")
    reference["properties"]["moving_footprint_uv"] = _object({"lower_uv": _array(2, -1, 2), "upper_uv": _array(2, -1, 2)})
    reference["required"].append("moving_footprint_uv")
    reference["properties"]["blocked_boxes_uv"] = {"type": "array", "maxItems": 8,
        "items": _object({"object_id": identifier, **copy.deepcopy(bounds["properties"]),
                          "role": {"type": "string", "enum": ["obstacle", "contact_target"]}})}
    reference["properties"]["unknown_boxes_uv"] = {"type": "array", "maxItems": 8,
        "items": _object({"object_id": identifier, **copy.deepcopy(bounds["properties"]),
                          "source": {"type": "string", "enum": ["occluded", "out_of_view", "self_occlusion"]}})}
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    grant = _object({"blocker_codes": {"type": "array", "items": identifier, "minItems": 1, "maxItems": 8},
        "directions": {"type": "array", "items": _array(3, -.1, .1), "minItems": 1, "maxItems": 6},
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 10},
        "max_displacement_m": {"type": "number", "exclusiveMinimum": 0, "maximum": .02},
        "corridors_uv": _object({view: nullable(copy.deepcopy(bounds)) for view in ("external", "wrist")})})
    result["properties"]["observation_intent"] = nullable(grant)
    result["required"].append("observation_intent")
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v3 is zero-shot Hybrid only")
    return guidance_v1(task, False).replace(
        "Use conservative moving_radius_uv for visible tool/payload extent. ", "") + (
        "V3 REFERENCES: target_anchor_uv is a real textured target landmark; target_point_uv is the "
        "desired projection of the SAME moving point and may be virtual. Preserve landmark IDs and "
        "their physical meaning. moving_footprint_uv is the observed tool/payload image bounding box "
        "containing moving_point_uv, not an invented isotropic radius. Use the visible extent, with "
        "uncertain hidden extent in unknown regions. blocked_boxes_uv.role is obstacle or contact_target; "
        "the latter is an intended interaction region, not free space. Field hands off before entering "
        "contact regions; model-directed Direct5 handles contact. Object names do not grant exceptions. "
        "unknown_boxes_uv.source is occluded, out_of_view, or self_occlusion. self_occlusion is only "
        "inside the currently observed moving footprint; hidden background is occluded, not self. "
        "Unknown masks need no interior texture, remain uncertainty, and are not automatically cleared. "
        "V3 FEASIBILITY FIRST: inspect typed feasibility/blockers even after a Direct response. An "
        "invalid start, blocked/unknown goal, bad footprint or contradictory mask requires semantic "
        "reference/goal correction or a different visibly justified contact/observation strategy. "
        "Repeating calibration motions cannot repair such a field. Do not relabel actual obstacles, "
        "shrink the tool extent, drop an unavailable view or rename landmarks merely to bypass a blocker. "
        "The local controller preserves each healthy view's measured response across another view's "
        "reseed. Directional model support, prediction errors and explicit per-view uncertainty matter; "
        "mere robot movement, a phase label or a new hypothesis_id is not information or task progress. "
        "Use the persistent progress_ledger to avoid repeating probes that did not reduce the same "
        "uncertainty. Only comparable landmark/goal epochs support image-distance progress claims. "
        "V3 LOCAL ACQUISITION: choose visual_field with persistent field_intent and optional observation_intent. "
        "The grant identifies blocker_codes, up to six actual normalized translation directions with "
        "norm<=0.1, max_steps 1..10, max_displacement_m<=0.02 from the current measured EEF, and per-view "
        "corridors_uv (box or null). Local probes require BOTH your grant and measured predictive support "
        "plus candidate/corridor checks. Every currently reliable view requires an explicit corridor. "
        "A missing view's null corridor grants no permission or new evidence: only its retained "
        "constraints and measured prediction, for at most three lost frames and command norm<=0.05, "
        "may permit a probe checked within another reliable view's corridor. Initial unsupported "
        "directions require your explicit Direct5; there is no blind cold-start or automatic upward motion. "
        "Acquisition executes zero rotation and zero incremental gripper action, preserving measured "
        "orientation and finger state. The local controller selects "
        "informative granted actions, resumes the same field intent when supported, and requests new "
        "semantics only on contradiction, contact handoff or expired local grant/refresh budget. These "
        "budgets are not episode caps. Direct requires observation_intent=null and exactly five raw "
        "actions, including contact/reorientation actions; no local correction changes those actions. "
        "Field holds orientation, zeroing residual<=0.01 rad; use Direct for larger reorientation. "
        "Held-object transport/alignment requires independently visible payload/tool co-motion; closure "
        "alone is not grasp. Missing evidence requests a specific test/regrasp, not automatic transport. "
        "Post-release image alignment and tool separation are observable evidence only, not 3D support "
        "or completion. No supplied depth/calibration/object truth or evaluator outcome enters this policy. "
        "Two recent full request turns and persistent structured probe/blocker memory are retained; raw "
        "logs stay local. Return strict JSON with no task-completion flag."
    )


def _runtime_schema(value):
    result = copy.deepcopy(value)
    if isinstance(result, dict):
        if result.get("type") == "integer":
            result["type"] = "number"
        for key, child in result.items():
            result[key] = _runtime_schema(child)
    elif isinstance(result, list):
        result = [_runtime_schema(child) for child in result]
    return result


def _ordered(box):
    return all(a < b for a, b in zip(box["lower_uv"], box["upper_uv"]))


class VisualFieldSubscriptionClientV3(VisualFieldSubscriptionClientV2):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v3", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                record["progress_ledger"] = copy.deepcopy(self.observable_context.get("progress_ledger", {}))
                from .astra import _canonical
                message["content"][0]["text"] = _canonical(record)
        return payload

    def validate_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        field = data["controller"] == "visual_field"
        if (data["field_intent"] is not None) != field or (data["direct_actions"] is None) != field:
            raise SchemaError("Only the selected V3 controller branch may be populated")
        if field and abs(sum(x*x for x in data["field_intent"]["orientation_target_xyzw"])-1) > .03:
            raise SchemaError("Orientation target must be a unit quaternion")
        for ref in data["visual_references"].values():
            if ref is None:
                continue
            if (not ref["moving_id"].strip() or not ref["target_id"].strip()
                    or ref["moving_id"] == ref["target_id"]):
                raise SchemaError("Moving and target identities must be nonempty and distinct")
            regions = ref["blocked_boxes_uv"]+ref["unknown_boxes_uv"]
            identities = [box["object_id"] for box in regions]
            if any(not ident.strip() for ident in identities) or len(identities) != len(set(identities)):
                raise SchemaError("Region identities must be nonempty and unique within each view")
            footprint = ref["moving_footprint_uv"]
            if not _ordered(footprint) or any(not lo <= point <= hi for lo, point, hi in
                    zip(footprint["lower_uv"], ref["moving_point_uv"], footprint["upper_uv"])):
                raise SchemaError("Observed footprint must have positive area and contain the moving point")
            if any(not _ordered(box) for box in regions):
                raise SchemaError("Reference boxes require positive area")
        grant = data["observation_intent"]
        if grant is not None:
            if not field:
                raise SchemaError("Observation acquisition grant requires persistent Field intent; Direct5 is unchanged")
            if type(grant["max_steps"]) is not int or not 0 < grant["max_displacement_m"] <= .02:
                raise SchemaError("Acquisition bounds require integer steps and positive displacement")
            if any(not 0 < math.sqrt(sum(x*x for x in direction)) <= .1+1e-12 for direction in grant["directions"]):
                raise SchemaError("Acquisition directions must be nonzero normalized translations of norm<=0.1")
            if any(box is not None and not _ordered(box) for box in grant["corridors_uv"].values()):
                raise SchemaError("Observation corridors require positive area")
            if not any(grant["corridors_uv"].values()):
                raise SchemaError("Acquisition requires at least one explicit image corridor")
        return copy.deepcopy(data)

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_VERSION, text_history="recent_two_turns_and_persistent_typed_progress_ledger")
        return result
