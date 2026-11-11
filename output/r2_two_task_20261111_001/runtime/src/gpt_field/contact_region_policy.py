"""Opt-in region-based contact feedback over the unchanged operation executor.

The wire contract describes semantic visible regions, not measured contact
points. A private adapter lets the existing v4 executor validate its unchanged
motion contract; it never rewrites accepted model text. Apply and shadow use
one client/prompt/schema. Shadow computes proposals but executes nominal goals.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .action import _rotvec_quaternion
from .astra import SchemaError
from .contact_region_feedback import (ContactRegionFeedback,
    contact_region_reference_schema, validate_contact_region_reference)
from .operation_contract import OperationEvidenceLedger, validate_operation, validate_visual_intent
from .policy import Decision
from .speed_first_v4 import (ChatGPTPlanSpeedFirstV4Client, SpeedFirstV4Policy,
    speed_first_v4_schema)


CONTACT_REGION_POLICY_VERSION = "operation-contact-region-v1"
_VIEWS = ("external", "wrist")


def _legacy_visual(visual):
    """Validation/execution compatibility only; never model input or wire history."""
    result = copy.deepcopy(visual)
    reference = result["pregrasp_reference"]
    result["pregrasp_reference"] = None
    if reference is not None:
        for view in _VIEWS:
            available = ((result["tracking_reference"] is not None
                          and result["tracking_reference"][view] is not None)
                         or (view == "external" and result["placement_reference"] is not None))
            if reference[view] is not None and not available:
                result["unavailable_reasons"][view] = "Region handled by the separate contact-region observer"
    return result


def validate_contact_region_visual_intent(visual, operation):
    operation = validate_operation(operation)
    if not isinstance(visual, dict) or set(visual) != {
            "mode", "pregrasp_reference", "placement_reference", "tracking_reference", "unavailable_reasons"}:
        raise ValueError("visual_intent requires the complete operation visual contract")
    reference = validate_contact_region_reference(visual["pregrasp_reference"])
    if reference is not None:
        if operation["kind"] != "pregrasp" or visual["mode"] != "pregrasp":
            raise ValueError("Contact-region reference requires a pregrasp operation")
        if reference["target_id"] != operation["target_id"]:
            raise ValueError("Contact-region target_id must match its operation")
        reasons = visual["unavailable_reasons"]
        if not isinstance(reasons, dict) or set(reasons) != set(_VIEWS):
            raise ValueError("unavailable_reasons must contain external and wrist")
        for view in _VIEWS:
            if reference[view] is not None and reasons[view] is not None:
                raise ValueError("A supplied contact-region view requires a null unavailable reason")
    # This checks the other reference formats, mode, and all unavailable reasons.
    validate_visual_intent(_legacy_visual(visual), operation)
    return copy.deepcopy(visual)


class ContactRegionEvidenceLedger(OperationEvidenceLedger):
    """Keep semantic region provenance while sharing v4's dated action ledger."""

    def accept(self, operation, visual, observation_index, state8=None):
        checked = validate_contact_region_visual_intent(visual, operation)
        super().accept(operation, _legacy_visual(checked), observation_index, state8)
        reference = checked["pregrasp_reference"]
        current = self._operations[-1]
        current.update(
            visual_reference_supplied=any(checked[k] is not None for k in (
                "pregrasp_reference", "placement_reference", "tracking_reference")),
            visual_unavailability=copy.deepcopy(checked["unavailable_reasons"]),
            contact_region_reference=(None if reference is None else {
                "reference_id": reference["reference_id"], "target_id": reference["target_id"],
                "available_views": [v for v in _VIEWS if reference[v] is not None],
                "semantics": "model-selected image regions and desired relation, not measured contacts"}))
        return copy.deepcopy(current)


def contact_region_policy_schema(task_key="bowl_plate"):
    schema = speed_first_v4_schema(task_key)
    schema["properties"]["visual_intent"]["properties"]["pregrasp_reference"] = contact_region_reference_schema()
    return schema


def _region_instructions(original):
    start = original.index("Optional pregrasp_reference is ONLY")
    end = original.index("Optional placement_reference is ONLY", start)
    replacement = (
        "Optional visual_intent.pregrasp_reference now uses compact VISIBLE REGION feedback, "
        "ONLY for kind=pregrasp with an open gripper and fixed orientation. Its shape is "
        "{reference_id,target_id,external,wrist}; each available view contains exactly "
        "tool_region_uv,target_region_uv,desired_offset_uv. Regions are ordered "
        "[u_min,v_min,u_max,v_max], normalized to [0,1] with top-left origin. Select a visible "
        "finger/hand material feature and the intended target grasp region. Prefer enough "
        "same-body texture for local tracking; a clearly visible wrist-tool edge can also be "
        "used under the conditional fixed-anchor fallback below. Exclude unrelated background and other bodies. "
        "The region centers are IMAGE ANCHORS, not physical contact points or pinch centers. "
        "Do not substitute the whole object's center for a grasp region. desired_offset_uv "
        "is the intended TOOL-ANCHOR MINUS TARGET-ANCHOR relation in normalized image coordinates; "
        "choose it for the selected features and desired grasp geometry, never automatically zero. "
        "Use one view if only one is visible; both may be supplied independently. If the desired "
        "relation is not observable, keep the reference null with honest unavailable_reasons. "
        "Do not guess hidden contact points or depth. The local program chooses and tracks texture "
        "within the regions, transports these anchors, and learns the response to useful ordinary "
        "robot translations. A corner-poor wrist TOOL region may use a fixed image anchor only "
        "while its visible intensity/edge pattern is checked each frame and robot orientation "
        "and finger width remain stable. This is a conditional rigid-hand assumption, NOT a "
        "measured optical-flow displacement or contact proof. This fallback never fixes the "
        "target anchor: target regions still require visual tracking. A static wrong region "
        "cannot be ruled out by appearance checks alone. Keep the EXACT SAME reference_id, seed regions and desired offset "
        "while the selected semantic objective remains unchanged; local tracking transports the "
        "anchors, so do not replace seed boxes with current positions. A changed target, feature, "
        "desired relation or reseed requires a NEW reference_id. Select orientation first and "
        "normally use a short pregrasp Field goal within 3 cm, preserving open fingers. Ordinary "
        "Direct still contains exactly five unchanged model-selected actions. Only previously "
        "observed directions with successful subsequent held-out motion predictions may propose "
        "a Field-goal correction, at most 6 mm TOTAL from the immutable model nominal waypoint "
        "within a 4 cm local domain. Prediction mismatch or unavailable tracking restores the "
        "nominal goal; normal control continues without extra requests, waiting or probes. "
        "No automatic closure follows alignment. Feedback distinguishes image-error reduction "
        "from physical grasp improvement: it supplies neither depth/calibration nor proof of "
        "contact, object pose, force closure or task completion.\n\n")
    return CONTACT_REGION_POLICY_VERSION + ". " + original[:start] + replacement + original[end:]


class ChatGPTPlanContactRegionClient(ChatGPTPlanSpeedFirstV4Client):
    contact_region_feedback = True

    @property
    def prompt_version(self):
        return CONTACT_REGION_POLICY_VERSION + "+" + self.planner_mode

    def reset_history(self):
        super().reset_history()
        self.operation_ledger = ContactRegionEvidenceLedger()

    def build_payload(self, observation, grid=None, mode="hybrid"):
        with self._history_lock:
            payload = super().build_payload(observation, grid, mode)
            payload["instructions"] = _region_instructions(payload["instructions"])
            payload["text"]["format"] = {"type": "json_schema", "strict": True,
                "name": "astra_operation_contact_region_v1",
                "schema": contact_region_policy_schema(self.task_spec.key)}
            return payload

    def validate_response(self, data, grid):
        # All region errors are rejected before v4 commits any diagnostics.
        if not isinstance(data, dict):
            raise SchemaError("Contact-region response requires the typed operation envelope")
        try:
            visual = validate_contact_region_visual_intent(data.get("visual_intent"), data.get("operation"))
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(str(exc)) from exc
        adapted = copy.deepcopy(data)
        adapted["visual_intent"] = _legacy_visual(visual)
        result = super().validate_response(adapted, grid)
        self.last_visual_intent = copy.deepcopy(visual)
        self.last_pregrasp_reference = copy.deepcopy(visual["pregrasp_reference"])
        self.last_compact_prediction = copy.deepcopy(data)
        return result


class ContactRegionPolicy(SpeedFirstV4Policy):
    def __init__(self, client, planner_mode="field", *, apply_contact_correction=True):
        if not getattr(client, "contact_region_feedback", False):
            raise ValueError("ContactRegionPolicy requires the matching compact-region client")
        if type(apply_contact_correction) is not bool:
            raise ValueError("apply_contact_correction must be bool")
        self.apply_contact_correction = apply_contact_correction
        super().__init__(client, planner_mode=planner_mode)
        self.method = CONTACT_REGION_POLICY_VERSION

    def reset(self):
        super().reset()
        self.pregrasp = ContactRegionFeedback()
        self._contact_region_counts = dict(proposal_queries=0, accepted_proposals=0,
            rejected_proposals=0, unavailable_queries=0, shadow_proposals=0,
            application_errors=0, prediction_mismatch_events=0)
        self._contact_prediction_mismatch_keys = set()

    def _pregrasp_call(self, operation, *, update=False, measurement_step=None):
        result = super()._pregrasp_call(operation, update=update, measurement_step=measurement_step)
        diagnostic = result[1] if update else result
        statuses = [diagnostic.get("status", ""), diagnostic.get("post_application", {}).get("status", "")]
        statuses.extend(value.get("status", "") for value in diagnostic.get("views", {}).values())
        if any("mismatch" in status for status in statuses if isinstance(status, str)):
            key = (diagnostic.get("epoch"), diagnostic.get("reference_id"))
            if key not in self._contact_prediction_mismatch_keys:
                self._contact_prediction_mismatch_keys.add(key)
                self._contact_region_counts["prediction_mismatch_events"] += 1
        return result

    def contact_region_stats(self):
        return {"version": CONTACT_REGION_POLICY_VERSION,
            "correction_enabled": self.apply_contact_correction,
            **copy.deepcopy(self._pregrasp_counters), **copy.deepcopy(self._contact_region_counts),
            "core": copy.deepcopy(self.pregrasp.snapshot()),
            "last_feedback": copy.deepcopy(self.pregrasp_local_feedback),
            "grasp_certified": False, "task_success_certified": False}

    def pregrasp_stats(self):
        # Existing recorder/runner hooks retain the new module's true identity.
        return self.contact_region_stats()

    def _validate_envelope(self, value):
        result = super()._validate_envelope(value)
        if self._pregrasp_requested and self._nominal_goal is not None:
            # A fresh accepted response replaces the previous Field cache in the
            # parent executor. Keep image learning, but explicitly retire the old
            # target so its restoration can never overwrite this new model goal.
            key = f"accepted-observation-{self.client._observation_count}"
            self._pregrasp_call(lambda: self.pregrasp.begin_nominal_goal(self._nominal_goal, key))
        return result

    def _restore_pregrasp_nominal(self):
        active = self._pregrasp_goal_active
        super()._restore_pregrasp_nominal()
        if active:
            self._pregrasp_call(lambda: self.pregrasp.notify_nominal_restored(self._nominal_goal))
            self.pregrasp_local_feedback = {**self.pregrasp_local_feedback,
                "nominal_goal_restored": True, "correction_applied": False,
                "fallback": "resume_model_nominal_goal"}

    def _maybe_update_pregrasp_goal(self, observation):
        if (self.controller != "coarse_field" or self.field is None or self._nominal_goal is None
                or not self.operation or self.operation["kind"] != "pregrasp"
                or self._field_cache.effective_compact_field["motion_mode"] != "move"):
            return
        if (self.pregrasp_local_feedback.get("restore_nominal_requested")
                or not self.pregrasp_local_feedback.get("tracking_valid", False)):
            self._restore_pregrasp_nominal()
            return
        goal_now = np.asarray(self.field.center) + np.asarray(
            self._field_cache.effective_compact_field["goal_offset_m"])
        if (self.step_index - self._last_pregrasp_update_step < 3
                and np.linalg.norm(np.asarray(observation.state[:3]) - goal_now) > .005):
            return
        compact = self._field_cache.effective_compact_field
        current = _rotvec_quaternion(np.asarray(observation.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"], dtype=float)
        target /= np.linalg.norm(target)
        angle = 2 * math.acos(float(np.clip(abs(np.dot(current, target)), 0., 1.)))
        if angle > self.pregrasp.maximum_orientation_change_rad:
            self._pregrasp_call(lambda: self.pregrasp.invalidate("nominal_goal_changes_orientation"))
            self._pregrasp_counters["fallbacks"] += 1
            self._restore_pregrasp_nominal()
            return
        self._contact_region_counts["proposal_queries"] += 1
        goal, diagnostic = self._pregrasp_call(
            lambda: self.pregrasp.corrected_goal(observation, self._nominal_goal), update=True)
        self._last_pregrasp_update_step = self.step_index
        valid = (goal is not None and np.asarray(goal).shape == (3,)
            and np.isfinite(goal).all()
            and diagnostic.get("response_validated", False)
            and diagnostic.get("proposal_available", False)
            and not diagnostic.get("restore_nominal_requested", False)
            and np.linalg.norm(goal - self._nominal_goal) <= .006 + 1e-12
            and np.all(goal >= self.planner.config.workspace_min)
            and np.all(goal <= self.planner.config.workspace_max)
            and self.field.query(goal) is not None)
        if not valid:
            self._contact_region_counts["unavailable_queries" if goal is None else "rejected_proposals"] += 1
            self._pregrasp_counters["fallbacks"] += 1
            self.pregrasp_local_feedback = {**diagnostic, "proposal_available": False,
                "correction_applied": False, "fallback": "resume_model_nominal_goal"}
            if goal is not None:
                self.pregrasp_local_feedback["correction_rejected_reason"] = "unvalidated_or_invalid_goal_or_outside_nominal_6mm"
            self._restore_pregrasp_nominal()
            return
        self._contact_region_counts["accepted_proposals"] += 1
        if not self.apply_contact_correction:
            self._contact_region_counts["shadow_proposals"] += 1
            self._restore_pregrasp_nominal()
            self.pregrasp_local_feedback = {**diagnostic, "correction_applied": False,
                "proposed_goal_xyz": np.asarray(goal).tolist(), "applied_goal_xyz": self._nominal_goal.tolist()}
            return
        changed = self._set_pregrasp_goal(np.asarray(goal))
        self._pregrasp_goal_active = bool(np.linalg.norm(goal - self._nominal_goal) >= .0001)
        if changed:
            try:
                self.pregrasp.notify_correction_applied(observation, self._nominal_goal, goal)
            except Exception:
                self._contact_region_counts["application_errors"] += 1
                self._pregrasp_call(lambda: self.pregrasp.invalidate("correction_monitor_error"))
                self._restore_pregrasp_nominal()
                return
        self.pregrasp_local_feedback = {**diagnostic, "correction_applied": changed,
            "applied_goal_xyz": np.asarray(goal).tolist(),
            "local_goal_updates": self._pregrasp_counters["applied_goals"]}

    def act(self, observation):
        decision = super().act(observation)
        details = {**decision.diagnostics, "contact_region_feedback": True,
            "contact_region_diagnostics": self.contact_region_stats()}
        decision = Decision(decision.action, details)
        self._pending = (self._pending[0], decision)
        return decision
