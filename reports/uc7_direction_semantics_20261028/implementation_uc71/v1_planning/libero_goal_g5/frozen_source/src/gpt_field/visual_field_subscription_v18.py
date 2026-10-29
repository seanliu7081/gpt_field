"""Observation-bound, model-selected reference candidates over the V17 protocol.

A candidate is temporal RGB evidence, not semantic tool identity. Only the
model selects a physical distal feature and supplies its whole-tool envelope.
Exact local coordinates enter the internal validator; original wire/history
retain the candidate ID and null coordinates. No motion budget is reset here.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v5 import _validate_region
from .visual_field_subscription_v8 import _ordered_roi
from .visual_field_subscription_v15 import _replace_current_diagnostic, MOTION_LABEL
from .visual_field_subscription_v17 import (
    VisualFieldSubscriptionClientV17, output_schema as schema_v17,
    shared_guidance as guidance_v17, geometry_semantics,
)

VERSION = "visual-field-rgb-joints-v18-grounded-reference-candidate-v1"
HISTORY_VERSION = "visual-v18-candidate-grounding-recent-two-turns-v1"
CANDIDATE_LABEL = (
    "Current observation-bound reference review board. Numbered candidate IDs identify exact "
    "current external-image patch coordinates, not robot semantics. Prior/current RGB and measured "
    "motion show temporal evidence; changes can include shadows, background and proximal links. "
    "Choose an ID ONLY when both original views support a physical distal rigid gripper/finger "
    "feature, and supply a fresh whole-tool ROI. Never use board/crop coordinates as full-frame UV. "
    "Selection does not certify identity, attachment, clearance or replenish motion allowance."
)


def output_schema(controller="hybrid"):
    result = schema_v17(controller)
    result["properties"]["reference_candidate_id"] = {"anyOf": [
        {"type":"string", "minLength":1, "maxLength":120}, {"type":"null"}]}
    result["required"].append("reference_candidate_id")
    seed = result["properties"]["visual_references"]["properties"]["external"]["anyOf"][0]["properties"]["seed"]["anyOf"][0]
    seed["properties"]["moving_point_uv"] = {"anyOf": [_array(2,0,1), {"type":"null"}]}
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v17(task, demonstration)
    # The new review is triggered by actual support stasis with other observed
    # motion, not by a small/inconclusive image response alone.
    text = text.replace(
        "Weak response is not itself slip, semantic mismatch or permission to rotate.",
        "Weak response is not itself slip, semantic mismatch or permission to rotate. However, reference_repair.review_required "
        "means current support stayed static while other RGB patches responded to measured robot motion: "
        "perform semantic candidate review even if reference_health still says weakly_observable. "
        "That health label does not require retaining the same disputed physical feature.")
    begin, end = text.index("RECOVERY."), text.index("\n\nREFERENCES.")
    text = text[:begin] + """RECOVERY. Read reference_repair before spending further directional allowance. When review_required, inspect its observation-bound numbered before/current board and both original RGB views. Reliable texture matching can track a fixed support; changed pixels can also be shadows, background or proximal links. Select a candidate ONLY if it is visibly on the distal rigid gripper housing/finger, not merely the most moving patch. reference_candidate_id chooses that exact current candidate. Supply external.update=seed, moving_kind=tool, a seed object with moving_point_uv=null and moving_roi_uv=null, and tool_point_uv=tool_roi_uv=null. Keep explicit valid stationary target anchor/ROI; wrist rules are unchanged. Supply a freshly inspected motion_context.tool_external_roi_uv containing the chosen feature and full gripper/fingers, not the patch alone. Local code resolves exact full-frame point/patch UV; do not estimate coordinates from the board or crop. The ID is invalid if stale or absent from this observation. If no candidate is semantically identifiable, use reference_candidate_id=null and explain the limitation; do not invent a moving tool. Null ID preserves ordinary seed/retain rules, including a nonnull manually seeded moving point. Candidate selection is an explicit model claim, not an automatic semantic certificate, fitted response, new grasp, or debt reset.

Reuse the persistent Field goal after repairing the correspondence when still appropriate; use new/replace only under the existing ledger rules. The local candidate evidence may be checked against already recorded RGB/state, but no action is automatically added and shared/same-direction measured debt remains. Read current recovery_options: previewed candidates are proposals, actual segments are rechecked, and finite allowance bounds measured travel. Use a justified changed direction and offered local cap if needed, never gratuitous yaw or repeated calibration. A truly blocked response may use valid Direct five zero actions and an explicit reason; holds produce no new evidence. First perform any available grounded reference review rather than repeatedly retaining a disputed feature and emitting unchanged holds. Never invent gripper opening, clearance or completion.""" + text[end:]
    return text


def _append_candidate_board(payload, image_url):
    # Reuse the existing explicitly-labelled derived-image replacement; both
    # original RGB images are untouched. The board has its own truthful label.
    for message in reversed(payload.get("input", [])):
        record = _text_record(message)
        if message.get("role") != "user" or not record or record.get("type") != "observation":
            continue
        kept, skip = [], False
        for part in message["content"]:
            if part.get("type") == "input_text" and part.get("text") == CANDIDATE_LABEL:
                skip = True; continue
            if skip and part.get("type") == "input_image":
                skip = False; continue
            skip = False; kept.append(part)
        message["content"] = kept
        break
    result = _replace_current_diagnostic(payload, image_url)
    for message in reversed(result.get("input", [])):
        record = _text_record(message)
        if message.get("role") != "user" or not record or record.get("type") != "observation":
            continue
        for part in reversed(message["content"]):
            if part.get("type") == "input_text" and part.get("text") == MOTION_LABEL:
                part["text"] = CANDIDATE_LABEL
                return result
        break
    return result


class VisualFieldSubscriptionClientV18(VisualFieldSubscriptionClientV17):
    history_protocol = HISTORY_VERSION
    reference_candidate_resolver = None
    reference_candidate_provider = None
    _candidate_observation = None

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def predict(self, observation, grid=None, mode="hybrid"):
        with self._history_lock:
            self._candidate_observation = copy.deepcopy(observation)
            try:
                return super().predict(observation, grid, mode)
            finally:
                self._candidate_observation = None

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v18", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        repair = ((self.observable_context or {}).get("visual_feedback") or {}).get("reference_repair") or {}
        if repair.get("review_required") is True and callable(self.reference_candidate_provider):
            diagnostic = self.reference_candidate_provider(copy.deepcopy(observation))
            if isinstance(diagnostic,str) and diagnostic.startswith("data:image/"):
                payload = _append_candidate_board(payload, diagnostic)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        identity = data["reference_candidate_id"]
        legacy = copy.deepcopy(data)
        legacy.pop("reference_candidate_id")
        resolved = None
        if identity is not None:
            external = legacy["visual_references"]["external"]
            seed = None if external is None else external["seed"]
            if (not identity.strip() or external is None or external["update"] != "seed"
                    or external["moving_kind"] != "tool" or seed is None):
                raise SchemaError("Reference candidate selection requires a named current candidate and explicit external tool seed")
            if any(seed[name] is not None for name in ("moving_point_uv","moving_roi_uv","tool_point_uv","tool_roi_uv")):
                raise SchemaError("Reference candidate selection requires null moving point/ROI and auxiliary tool point/ROI; coordinates are resolved locally")
            whole_tool = legacy["motion_context"]["tool_external_roi_uv"]
            if whole_tool is None:
                raise SchemaError("Reference candidate selection requires a current whole-tool ROI")
            _ordered_roi(whole_tool)
            if self._candidate_observation is None or not callable(self.reference_candidate_resolver):
                raise SchemaError("Reference candidate resolution is unavailable for this exact observation")
            try:
                resolved = copy.deepcopy(self.reference_candidate_resolver(identity, copy.deepcopy(self._candidate_observation)))
            except (ValueError, KeyError):
                # Do not expose arbitrary callback bodies as repair instructions.
                raise SchemaError("Reference candidate is unknown, stale or unavailable for this exact observation") from None
            if not isinstance(resolved,dict) or resolved.get("candidate_id") != identity:
                raise SchemaError("Reference candidate resolver did not return the exact selected identity")
            point, roi = resolved.get("current_point_uv"), resolved.get("current_roi_uv")
            _validate(point, _array(2,0,1), "resolved_candidate.current_point_uv")
            _validate(roi, _array(4,0,1), "resolved_candidate.current_roi_uv")
            _ordered_roi(roi)
            _validate_region({"point":point,"roi":roi}, "roi", "point")
            _validate_region({"point":point,"roi":whole_tool}, "roi", "point")
            seed["moving_point_uv"], seed["moving_roi_uv"] = copy.deepcopy(point), copy.deepcopy(roi)
        result = super()._validate_v5_response(legacy, grid)
        result["reference_candidate_id"] = identity
        result["reference_candidate_resolution"] = resolved
        result["raw_wire_prediction"] = copy.deepcopy(data)
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
