"""Checked continuous Field routes for explicitly selected reverse recovery.

Normal travel retains the weighted graph planner. A short model-selected
observed-motion reversal has an additional heading constraint: its distance
function is arc distance to its fixed goal along one continuously checked
segment. The same route and tracker are used for offers and real selection.
No image, geometry, travel or acknowledgement guard is removed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

from .action import _rotvec_quaternion
from .planner import CandidateRecord, PlanResult
from .visual_field_v21 import MetricFieldExecutorV21


class MetricFieldExecutorV22(MetricFieldExecutorV21):
    RECOVERY_EXTENT_M = .008

    def __init__(self):
        super().__init__()
        self.heading_recovery = None

    def configure(self, *args, **kwargs):
        self.heading_recovery = None
        return super().configure(*args, **kwargs)

    def nominal_regions_for_segment(self, start, end):
        # V19's per-object proof and V21's reversal contract ask the same
        # current scene assessor for the same segment. Share ONLY this one
        # invocation; no geometry/result survives another proposal or frame.
        assessor = self.scene_assessor
        if assessor is None:
            return super().nominal_regions_for_segment(start, end)
        cache = {}

        def once(state, delta):
            key = (tuple(np.asarray(state, float)), tuple(np.asarray(delta, float)))
            if key not in cache:
                cache[key] = copy.deepcopy(assessor(state, delta))
            return copy.deepcopy(cache[key])

        try:
            self.scene_assessor = once
            return super().nominal_regions_for_segment(start, end)
        finally:
            self.scene_assessor = assessor

    def _recovery_requested(self):
        guard = self.translation_guard
        return bool(guard is not None and self.intent is not None
            and (guard.context or {}).get("translation_mode") == "separate"
            and (guard.worsening or getattr(guard, "_retreat_committed", None)))

    def _prepare_heading_route(self, state):
        """Bind the model's goal, never synthesize or enlarge a recovery goal."""
        guard = self.translation_guard
        if (not self._same(state, self.expected_state, (8,))
                or not self._same(state, guard.state, (8,))):
            return "heading_recovery_unacknowledged_state"
        delta = self.goal-self.origin
        extent = float(np.linalg.norm(delta))
        if extent <= self.intent["goal_tolerance_m"]:
            return None  # The ordinary measured-arrival check owns this case.
        if extent > self.RECOVERY_EXTENT_M+1e-10:
            return "heading_recovery_goal_exceeds_bounded_route"
        histories = [guard.worsening_motion.get(identity) for identity in sorted(guard.worsening)]
        if not histories and getattr(guard, "_retreat_committed", None):
            histories = [guard.worsening_motion.get(identity)
                         for identity in guard._retreat_committed["episode_ids"]]
        if not histories or any(row is None for row in histories):
            return "heading_recovery_actual_motion_unavailable"
        if any(guard._angle(delta/extent, -np.asarray(row["measured_direction_xyz"]))
               > guard.reverse_tolerance_rad for row in histories):
            return "heading_recovery_goal_does_not_reverse_observed_motion"
        if (not self.checked.segment_valid(state[:3], self.goal)
                or not self.checked.segment_valid(self.origin, self.goal)):
            return "heading_recovery_continuous_segment_blocked"
        # The complete remaining route, not just its endpoints or first action,
        # must retain every full-margin scene object and target contact band.
        records = self.nominal_regions_for_segment(state[:3], self.goal)
        contract = guard._contract(records, self.goal-state[:3], 0.)
        if contract is None:
            return "heading_recovery_complete_route_contract_unavailable"
        if self.heading_recovery is not None:
            if (not self._same(self.heading_recovery["goal_xyz_m"], self.goal, (3,))
                    or self.heading_recovery["geometry_sha256"] != contract["geometry_sha256"]):
                return "heading_recovery_fixed_route_changed"
            self.heading_recovery["last_checked_observation_epoch"] = contract["observation_epoch"]
            return None
        positions = (tuple(float(x) for x in self.origin), tuple(float(x) for x in self.goal))
        identity = hashlib.sha256(json.dumps({"positions": positions,
            "geometry_sha256": contract["geometry_sha256"]}, sort_keys=True).encode()).hexdigest()
        self.heading_recovery = {"version": "continuous-checked-reverse-route-v22",
            "route_id": identity, "origin_xyz_m": list(positions[0]), "goal_xyz_m": list(positions[1]),
            "unit_direction_xyz": (delta/extent).tolist(), "length_m": extent,
            "geometry_sha256": contract["geometry_sha256"],
            "last_checked_observation_epoch": contract["observation_epoch"],
            "scene_ids": contract["scene_ids"], "clearance_certified": False,
            "guard_heading_tolerance_changed": False, "physical_debt_replenished": False}
        candidate = CandidateRecord(("checked_continuous_recovery",), positions, extent, True, 0., 0.)
        self.last_plan = PlanResult(tuple(float(x) for x in delta), "move", 1, 1, (candidate,), 0,
            extent, extent, search_metadata={"method": "heading_constrained_continuous_distance",
                "distance_semantics": "remaining_arc_length_on_checked_fixed_recovery_segment",
                "cost_unit": "m", "segment_count": 1, "cache_hit": False,
                "goal_xyz": self.goal.tolist(), "path_termination": "goal",
                "route_id": identity, "normal_weighted_graph_planner_unchanged": True,
                "scene_geometry_is_model_hypothesis": True})
        self.tracker.reset(positions)
        self.plan_calls += 1
        return None

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        if self.pending is not None:
            raise RuntimeError("observe must acknowledge the previous accepted action")
        if self.stop_reason is None and self._recovery_requested():
            reason = self._prepare_heading_route(np.asarray(state8, float))
            if reason:
                return self._stop(reason, {"heading_recovery": copy.deepcopy(self.heading_recovery)})
        result = super().select(state8, joint_position, command_cap_m=command_cap_m,
                                measured_cap_m=measured_cap_m)
        result["diagnostics"]["heading_recovery"] = copy.deepcopy(self.heading_recovery)
        return result

    def preview_recovery(self, state8, joint_position, goal_offset_m, *,
                         goal_tolerance_m=.003, command_limit_m=.002, max_travel_m=.008,
                         command_cap_m=None, measured_cap_m=None, scene_assessor=None):
        """Run the actual prospective Field replacement and select transaction.

        ``scene_assessor(executor, state8, delta_xyz)`` should be a pure callback
        using the prospective executor and the current complete inventory. It
        avoids a policy callback accidentally inspecting the old prediction.
        Monitor/track/response identities are preserved, and no observation is
        replayed. The model must still explicitly choose the offered goal.
        """
        def refusal(reason):
            return {"accepted": False, "status": reason, "action": [0.]*7,
                    "diagnostics": {"heading_recovery": None}, "preview": True,
                    "requires_explicit_model_choice": True, "physical_debt_replenished": False}
        guard = self.translation_guard
        if self.intent is None or guard is None:
            return refusal("recovery_preview_unconfigured")
        if self.pending is not None:
            return refusal("recovery_preview_unacknowledged_action")
        if command_cap_m is not None and command_cap_m <= 0.:
            return refusal("recovery_preview_policy_motion_limit")
        original, guard_original = self.__dict__, guard.__dict__
        working, guard_working = original.copy(), guard_original.copy()
        guard_working["_command_exhausted_experiments"] = set(guard._command_exhausted_experiments)
        guard_working["context"] = dict(guard.context or {})
        guard_working["context"].update(translation_mode="separate", intended_contact_id=None)
        state = np.asarray(state8, float)
        intent = copy.deepcopy(self.intent)
        origin_shift = self.origin-state[:3]
        # Unknowns retain their original world position in a re-centered intent.
        for region in intent["unknown_regions"]:
            for key in ("lower_offset_m", "upper_offset_m"):
                region[key] = (np.asarray(region[key])+origin_shift).tolist()
        intent.update(goal_offset_m=list(goal_offset_m), goal_tolerance_m=goal_tolerance_m,
            command_limit_m=command_limit_m, max_travel_m=max_travel_m, motion_axes="xyz",
            max_actions=40, orientation_target_xyzw=_rotvec_quaternion(state[3:6]).tolist())
        material = copy.deepcopy(self._transit_materialization)
        try:
            self.__dict__, guard.__dict__ = working, guard_working
            if scene_assessor is not None:
                self.scene_assessor = lambda current, delta: scene_assessor(self, current, delta)
            self.configure(state, intent, scene_assessment=guard.context,
                           inventory_materialization=material)
            selected = copy.deepcopy(self.select(state, joint_position, command_cap_m=command_cap_m,
                                                 measured_cap_m=measured_cap_m))
            if selected["accepted"] and selected["diagnostics"].get("heading_recovery") is None:
                selected = refusal("recovery_preview_no_heading_constrained_route")
        finally:
            self.__dict__, guard.__dict__ = original, guard_original
        selected.update(preview=True, requires_explicit_model_choice=True,
            physical_debt_replenished=False, intent_replacement_is_prospective=True)
        return selected

    def summary(self):
        result = super().summary()
        result.update(version="checked-continuous-recovery-field-v22",
            heading_recovery=copy.deepcopy(getattr(self, "heading_recovery", None)))
        return result
