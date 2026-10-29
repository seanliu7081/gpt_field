"""Bounded proprioceptive transit when an image response is not required.

This is a candidate-bound model-hypothesis admission, not physical clearance.
The image reference ledger and every original motion budget remain intact.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_v19 import MetricFieldExecutorV19
from .visual_scene_guard_v17 import SceneCoverageGuardV17


class MetricFieldExecutorV20(MetricFieldExecutorV19):
    TRANSIT_REQUEST_VERSION = "v20-current-proprioceptive-transit"
    WEAK_REFERENCE_REASON = "controlled_reference_uncertainty_budget_exhausted"
    TRANSIT_COMMAND_CAP_M = .005

    def __init__(self):
        super().__init__()
        self.proprioceptive_transit_request = None
        self.last_proprioceptive_transit_admission = None
        self._transit_materialization = None

    def configure(self, state8, intent, *, inventory_materialization=None, **kwargs):
        self.proprioceptive_transit_request = None
        self.last_proprioceptive_transit_admission = None
        self._transit_materialization = copy.deepcopy(inventory_materialization)
        return super().configure(state8, intent,
            inventory_materialization=inventory_materialization, **kwargs)

    @staticmethod
    def _same(left, right, shape):
        try:
            a, b = np.asarray(left, float), np.asarray(right, float)
            return (a.shape == shape and b.shape == shape and np.isfinite(a).all()
                and np.isfinite(b).all() and np.allclose(a, b, atol=1e-10, rtol=0.))
        except (TypeError, ValueError):
            return False

    def _request_reason(self, state, request):
        monitor, guard = self.visual_monitor, self.translation_guard
        if not isinstance(request, dict) or request.get("version") != self.TRANSIT_REQUEST_VERSION:
            return "proprioceptive_transit_invalid_request"
        if (request.get("reason") != self.WEAK_REFERENCE_REASON
                or request.get("anchor_shifted") is not False
                or request.get("raw_external_role") != "tool"
                or request.get("target_contact") != "none"):
            return "proprioceptive_transit_ineligible_role_or_reference"
        if (monitor is None or guard is None or self.scene_assessor is None
                or not self._same(request.get("state8"), state, (8,))
                or not self._same(getattr(guard, "state", None), state, (8,))
                or not self._same(getattr(monitor, "state", None), state, (8,))
                or request.get("observation_epoch") != getattr(guard, "sweep_observation_epoch", None)
                or request.get("observation_epoch") is None
                or not self._same(request.get("goal_xyz"), self.goal, (3,))):
            return "proprioceptive_transit_stale_state_epoch_or_goal"
        record = getattr(monitor, "_active_reference", None) or {}
        refs = getattr(monitor, "refs", None) or {}
        if (not record.get("id") or request.get("reference_id") != record["id"]
                or (refs.get("external") or {}).get("moving_kind") != "tool"
                or (refs.get("external") or {}).get("target_id") != request.get("target_id")
                or not monitor._external_reliable() or getattr(monitor, "missing", 2) >= 2
                or any(not getattr((monitor.tracks or {}).get(name), "reliable", False)
                       for name in ("moving", "target"))):
            return "proprioceptive_transit_current_reference_unavailable"
        if (monitor.measured_control_limit()[1] != self.WEAK_REFERENCE_REASON
                or monitor.cumulative_response_evidence().get("contradicted")
                or getattr(monitor, "status", None) in ("visual_response_contradiction", "target_anchor_shifted")):
            return "proprioceptive_transit_reference_contradiction_or_different_limit"
        target = self.target_configuration or {}
        if (self.contact_refinement_enabled or self.goal_revisions
                or self.goal_revision_travel_m > 0.
                or target.get("target_id") != request.get("target_id")
                or target.get("effective_allowed_contact") != "none"
                or not target.get("available")):
            return "proprioceptive_transit_contact_or_retarget_dependency"
        carried = self.contact_geometry.carried_summary(state) if self.contact_geometry else {}
        if carried.get("active") or carried.get("candidate_ids"):
            return "proprioceptive_transit_payload_or_grasp_candidate"
        material = self._transit_materialization
        if (not isinstance(material, dict) or not material.get("objects")
                or material.get("controlled_payload_excluded_id") is not None):
            return "proprioceptive_transit_incomplete_scene"
        return None

    def _candidate_admission(self, state, action, request):
        delta = np.asarray(action[:3], float)*.05
        details = {"version": self.TRANSIT_REQUEST_VERSION,
            "proposed_delta_xyz_m": delta.tolist(), "state8": state.tolist(),
            "observation_epoch": request["observation_epoch"],
            "goal_xyz": self.goal.tolist(), "reference_id": request["reference_id"],
            "command_cap_m": self.TRANSIT_COMMAND_CAP_M,
            "image_jacobian_used_for_control": False, "budgets_modified": False,
            "reference_validation_modified": False, "clearance_certified": False,
            "task_success_certified": False}

        def result(reason=None):
            return {"accepted": reason is None,
                "reason": reason or "proprioceptive_fixed_goal_model_remote_transit",
                "diagnostics": details}

        if (abs(action[6]) > 1e-12 or np.linalg.norm(delta) > self.TRANSIT_COMMAND_CAP_M+1e-10
                or np.linalg.norm(action[3:6])*.5 > self.MAX_DRIFT_RAD+1e-10):
            return result("proprioceptive_transit_command_outside_contract")
        scene = self.scene_assessor(state, delta)
        details["scene_coverage_admission"] = copy.deepcopy(scene)
        data = scene.get("diagnostics") or {}
        if (not scene.get("accepted") or data.get("coverage_scope") != "actual_commanded_whole_tool_sweep"
                or data.get("full_finger_range") is not True
                or not self._same(data.get("start_xyz_m"), state[:3], (3,))
                or not self._same(data.get("end_xyz_m"), state[:3]+delta, (3,))
                or not isinstance(data.get("orientation_margin_rad"), (int, float))
                or not np.isfinite(data["orientation_margin_rad"])
                or data["orientation_margin_rad"] < self.MAX_DRIFT_RAD
                or data.get("carried_payload_hypothesis_in_sweep") is not False):
            return result("proprioceptive_transit_current_full_sweep_unavailable")
        objects = self._transit_materialization["objects"]
        ids = [row.get("id") for row in objects]
        checks = data.get("sweep_checks") or []
        check_ids = [row.get("id") for row in checks]
        represented = {row.get("id") for row in self.region_geometry}
        if (any(not isinstance(identity, str) or not identity for identity in ids)
                or len(ids) != len(set(ids)) or len(check_ids) != len(set(check_ids))
                or set(check_ids) != set(ids) or set(data.get("checked_metric_ids", [])) != set(ids)
                or not set(ids).issubset(represented) or request["target_id"] not in ids
                or data.get("unresolved_far_image_ids") or data.get("coverage_issues")
                or any(not isinstance(row.get("world_geometry"), dict) for row in objects)):
            return result("proprioceptive_transit_incomplete_scene")
        if any(row.get("full_uncertainty_sweep_disjoint") is not True
               or int(row.get("configuration_part_count", 0)) < 6 for row in checks):
            return result("proprioceptive_transit_intersects_model_uncertainty")
        target = copy.deepcopy(next(row for row in objects if row["id"] == request["target_id"]))
        target["world_geometry"]["uncertainty_m"] = 0.
        intent = copy.deepcopy(self.intent); intent["uncertainty_m"] = 0.
        boxes = SceneCoverageGuardV17._sweep_boxes(target, state,
            {"field_intent": intent}, data["orientation_margin_rad"])
        if not boxes:
            return result("proprioceptive_transit_target_core_unavailable")
        # Distance between the enclosing segment AABB and each configuration
        # box is a conservative lower bound on the ENTIRE segment's distance.
        low, high = np.minimum(state[:3], state[:3]+delta), np.maximum(state[:3], state[:3]+delta)
        distance = min(float(np.linalg.norm(np.maximum(np.maximum(
            part["lower_xyz"]-high, low-part["upper_xyz"]), 0.))) for part in boxes)
        band = max(.03, float(self.target_configuration["sigma_m"]))
        details["target_contact_band"] = {"target_id": request["target_id"],
            "whole_segment_core_distance_lower_bound_m": distance, "band_m": band,
            "all_robot_parts_included": True}
        if distance <= band+1e-10:
            return result("proprioceptive_transit_near_target_contact_band")
        details["complete_scene_ids"] = ids
        return result()

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        request = self.proprioceptive_transit_request
        if request is None:
            self.last_proprioceptive_transit_admission = None
            return super().select(state8, joint_position, command_cap_m=command_cap_m,
                                  measured_cap_m=measured_cap_m)
        if self.pending is not None:
            raise RuntimeError("observe must acknowledge the previous accepted action")
        state = np.asarray(state8, float)
        reason = self._request_reason(state, request)
        if reason:
            admission = {"accepted": False, "reason": reason,
                         "diagnostics": {"budgets_modified": False, "clearance_certified": False}}
            self.last_proprioceptive_transit_admission = admission
            return self._stop(reason, {"proprioceptive_transit_admission": copy.deepcopy(admission)})
        count, rotation = self.orientation_goal_refresh_commands, self.last_orientation_command_rad.copy()
        cap = self.TRANSIT_COMMAND_CAP_M if command_cap_m is None else min(command_cap_m, self.TRANSIT_COMMAND_CAP_M)
        selected = super().select(state8, joint_position, command_cap_m=cap, measured_cap_m=measured_cap_m)
        if not selected["accepted"]:
            self.last_proprioceptive_transit_admission = {"accepted": False,
                "reason": selected["status"], "diagnostics": {"inherited_check_rejected": True}}
            return selected
        admission = self._candidate_admission(state, selected["action"], request)
        self.last_proprioceptive_transit_admission = copy.deepcopy(admission)
        if not admission["accepted"]:
            self.pending = None
            self.orientation_goal_refresh_commands = count
            self.last_orientation_command_rad = rotation
            return self._stop(admission["reason"], {"proprioceptive_transit_admission": admission})
        selected["diagnostics"]["proprioceptive_transit_admission"] = admission
        return selected

    def summary(self):
        result = super().summary()
        result.update(version="candidate-bound-proprioceptive-transit-field-v20",
            proprioceptive_transit_admission=copy.deepcopy(
                getattr(self, "last_proprioceptive_transit_admission", None)))
        return result
