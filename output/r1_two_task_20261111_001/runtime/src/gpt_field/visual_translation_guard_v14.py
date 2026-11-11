"""Finite directional RGB experiments for opt-in V14 translation admission.

The robot/model geometry remains a hypothesis, not observed free space. A
finite cold experiment need not already have an image Jacobian. Measured
travel, including residual hold motion, is charged persistently; a new model
label, box or reference never replenishes the shared allowance.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .visual_translation_guard_v11 import _gap
from .visual_translation_guard_v12 import ObservedRiskTranslationGuardV12


class ObservedRiskTranslationGuardV14(ObservedRiskTranslationGuardV12):
    version = "directional-evidence-recovery-v14"
    experiment_limit_m = .008
    information_checkpoint_m = .004
    maximum_experiments = 3
    compatible_direction_rad = math.radians(30.)
    fresh_direction_rad = math.radians(45.)
    changed_support_minimum_px = 8.

    def __init__(self):
        super().__init__()
        self.experiments = []
        self._experiment_pending = None
        self._motion_classification_pending = None
        self._last_charged_experiment = None
        self._support_generation = 0
        self._support_change = None
        self._recovery_reason = None
        self._command_exhausted_experiments = set()

    def configure(self, images, motion_context, state8, *, target_id, monitor=None):
        old = self.tool_track
        previous = None if old is None else old.position.copy()
        current = self._tool(monitor)
        if current is not None and old is not None and current is not old:
            distance = float(np.linalg.norm(current.position-previous))
            changed = distance >= self.changed_support_minimum_px
            if changed:
                self._support_generation += 1
            self._support_change = {
                "new_track": True, "displacement_from_previous_current_point_px": distance,
                "qualifies_as_changed_support": changed,
                "is_observed_motion": False,
            }
        self._experiment_pending = None
        self._motion_classification_pending = None
        super().configure(images, motion_context, state8, target_id=target_id, monitor=monitor)
        return self.summary()

    @staticmethod
    def _angle(first, second):
        return float(math.acos(np.clip(np.dot(first, second), -1., 1.)))

    def _experiment_for(self, direction, *, nominal_away):
        """Return a provisional ledger choice; only observe commits a trial."""
        if not self.experiments:
            return {"index": None, "direction": direction.tolist(),
                    "support_generation": self._support_generation}, None
        last = self.experiments[-1]
        changed = self._support_generation > last["support_generation"]
        angle = self._angle(direction, last["direction"])
        if not changed and angle <= self.compatible_direction_rad+1e-9:
            if (last["measured_travel_m"] >= self.experiment_limit_m-1e-9
                    or len(self.experiments)-1 in self._command_exhausted_experiments):
                return None, "translation_recovery_direction_exhausted"
            return {"index": len(self.experiments)-1, "direction": direction.tolist(),
                    "support_generation": last["support_generation"]}, None
        if self.context.get("translation_mode") != "separate" or not nominal_away:
            return None, "translation_new_experiment_requires_nominal_separation"
        if len(self.experiments) >= self.maximum_experiments:
            return None, "translation_recovery_experiment_limit"
        if not changed:
            if (last["measured_travel_m"] < self.experiment_limit_m-1e-9
                    and len(self.experiments)-1 not in self._command_exhausted_experiments):
                return None, "translation_recovery_direction_change_requires_review"
            if any(self._angle(direction, row["direction"]) < self.fresh_direction_rad-1e-9
                   for row in self.experiments):
                return None, "translation_recovery_direction_not_distinct"
        return {"index": None, "direction": direction.tolist(),
                "support_generation": self._support_generation}, None

    def observe(self, images, state8, action, *, monitor=None):
        pending = self._experiment_pending
        classification = self._motion_classification_pending
        self._experiment_pending = None
        self._motion_classification_pending = None
        command = np.asarray(action[:3], float)*.05
        norm = float(np.linalg.norm(command))
        before = None if self.state is None else self.state[:3].copy()
        matched = False
        if classification is not None and norm > 1e-8:
            aligned = self._angle(command/norm, classification["direction"]) < 1e-5
            if not aligned or norm > classification["command_cap_m"]+1e-9:
                self.domain_invalid = "translation_experiment_command_mismatch"
            else:
                matched = True
                self._charge_uncertain_motion = classification["uncertain"]
        # Downstream target/payload gates can replace a provisional command by
        # a hold. Such a hold must preserve the LAST EXECUTED motion class, not
        # adopt the classification of an action that never ran.
        charge = self._charge_uncertain_motion
        if pending is not None and matched:
            index = pending["index"]
            if index is None:
                index = len(self.experiments)
                self.experiments.append({
                    "direction": pending["direction"],
                    "support_generation": pending["support_generation"],
                    "measured_travel_m": 0., "acknowledged_actions": 0,
                })
            self._last_charged_experiment = index
        if charge and before is not None and self._last_charged_experiment is not None:
            row = self.experiments[self._last_charged_experiment]
            row["measured_travel_m"] += float(np.linalg.norm(np.asarray(state8[:3])-before))
            row["acknowledged_actions"] += 1
        super().observe(images, state8, action, monitor=monitor)
        # V12 trains on 2mm aggregates. Its three training samples plus two
        # held-out predictions can need 10mm, exceeding our 8mm experiment.
        # Complete an otherwise unconsumed aggregate at 1mm instead. If V12
        # already consumed it, its origin equals current and this does nothing.
        # The predictor is still evaluated BEFORE adding each training sample.
        current, start = self._previous_snapshot, self._response_origin
        if current is not None and start is not None and not self.domain_invalid:
            displacement = current["state"]-start["state"]
            identities = start["neighbors"].keys() & current["neighbors"].keys()
            if identities and np.linalg.norm(displacement) >= .001-1e-12:
                common = np.median([current["neighbors"][i]["position"]-start["neighbors"][i]["position"]
                                    for i in identities], axis=0)
                flow = current["tool"]-start["tool"]-common
                prediction = self.cumulative_response.predict(displacement)
                if prediction is not None:
                    self._response_error.append((float(np.sum((flow-prediction)**2)), float(np.sum(flow**2))))
                self.cumulative_response.add(displacement, flow)
                self._response_extent_m = max(self._response_extent_m, float(np.linalg.norm(displacement)))
                self._response_origin = current
                self._response_age = 0
        return self.summary()

    def assess(self, delta_xyz, *, monitor=None, nominal_away=False, nominal_clear=False,
               nominal_regions=None, translation_response=None, rotation_rad=0.,
               orientation_hold=False, direct_observation=False):
        command = np.asarray(delta_xyz, float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command translation must be three finite metres")
        size = float(np.linalg.norm(command))
        self._experiment_pending = None
        self._motion_classification_pending = None
        if nominal_regions is not None:
            self._metric = {}
            for region in nominal_regions:
                try:
                    distance, sigma = float(region["core_distance_m"]), float(region["uncertainty_m"])
                    if np.isfinite(distance) and np.isfinite(sigma) and distance >= 0 and sigma >= 0:
                        self._metric[region["id"]] = (distance, sigma)
                except (TypeError, KeyError, ValueError):
                    continue
        self._last_nominal_clear = bool(nominal_clear)
        result = {"accepted": True, "cap_m": None, "need_refresh": False,
                  "reason": "projected_risk_no_near_conflict", "diagnostics": {}}

        def done(reason=None, cap=None, uncertain=False, experiment=None):
            if reason:
                result.update(accepted=False, cap_m=0., need_refresh=True, reason=reason)
                self.events += 1
            else:
                result["cap_m"] = cap
                if size > 1e-12:
                    self._motion_classification_pending = {
                        "uncertain": uncertain, "direction": (command/size).tolist(),
                        "command_cap_m": size if cap is None else min(size, cap),
                    }
                if experiment is not None:
                    self._experiment_pending = {**experiment, "command_cap_m": cap}
            self._recovery_reason = result["reason"]
            self._v12_last_result = {k: copy.deepcopy(v) for k, v in result.items() if k != "diagnostics"}
            result["diagnostics"] = self.summary()
            return result

        if size < 1e-12:
            result["reason"] = "pose_hold_no_clearance_claim"
            return done()
        if self.context_invalid:
            return done(self.context_invalid)
        if self.domain_invalid:
            return done(self.domain_invalid)
        if abs(rotation_rad) > (.020 if orientation_hold else .015)+1e-9:
            return done("translation_requested_rotation_outside_tracking_domain")
        tool = self._tool(monitor) if monitor is not None else self.tool_track
        if tool is None or tool is not self.tool_track or not tool.reliable:
            return done("translation_tool_correspondence_lost")
        if np.any(self.tool_box[:2] < -.5) or np.any(self.tool_box[2:] > self.scale+.5):
            return done("translation_tool_envelope_outside_view")
        risks = [r for r in self._risks() if not r["intended_contact"]]
        flow, response = self._image_prediction(command, translation_response)
        near = []
        for region in risks:
            if not self._near_metric(region["id"]):
                continue
            threshold = max(region["margin_px"], 2.+self._gap_noise(region))
            gap = region["gap_px"] if flow is None else _gap(self.tool_box+np.tile(flow, 2), region["box_px"])
            if min(region["gap_px"], gap) <= threshold or region["id"] in self.worsening:
                near.append(region)
        if not near:
            if not nominal_clear:
                if not direct_observation:
                    return done("translation_metric_segment_clearance_unavailable")
                result["reason"] = "projected_direct_current_observation_bounded"
                return done(cap=min(size, self.uncertain_command_limit_m if flow is not None else .002))
            return done()
        if any(not row["track_reliable"] for row in near):
            return done("translation_nearby_obstacle_track_unavailable")
        if not nominal_clear:
            return done("translation_metric_segment_clearance_unavailable")
        if any(row["id"] in self.worsening for row in near):
            return done("translation_observed_neighbor_approach")
        predicted_away = bool(flow is not None and all(
            _gap(self.tool_box+np.tile(flow, 2), row["box_px"]) > row["gap_px"]+1e-6
            for row in near))
        established_away = predicted_away and all(row["id"] in self.improving for row in near)
        conditional_away = (predicted_away and nominal_away
                            and self.context.get("translation_mode") == "separate")
        if established_away or conditional_away:
            result["reason"] = ("projected_risk_observed_separation_supported" if established_away
                                else "projected_separation_conditional_empirical_evidence")
            return done(cap=min(size, self.uncertain_command_limit_m))
        remaining = max(0., self.observation_limit_m-self.observation_travel_m)
        if remaining <= 1e-9:
            return done("translation_observation_budget_exhausted")
        experiment, error = self._experiment_for(command/size, nominal_away=nominal_away)
        if error:
            return done(error)
        used = 0. if experiment["index"] is None else self.experiments[experiment["index"]]["measured_travel_m"]
        remaining = min(remaining, max(0., self.experiment_limit_m-used))
        requested = min(size, self.uncertain_command_limit_m)
        if translation_response is not None:
            limit = translation_response.command_cap_for_measured_limit(command/size, remaining, requested, self.state)
            cap = min(requested, float(limit["command_cap_m"]))
        else:
            cap = min(requested, remaining)
        if cap < self.minimum_command_m-1e-9:
            if experiment["index"] is not None:
                self._command_exhausted_experiments.add(experiment["index"])
            return done("translation_measured_allowance_below_useful_command_floor")
        result["reason"] = "projected_risk_finite_directional_experiment"
        return done(cap=cap, uncertain=True, experiment=experiment)

    def summary(self):
        result = super().summary()
        for key in ("cold_separation_travel_m", "cold_separation_remaining_m", "cold_separation_contradicted"):
            result.pop(key, None)
        if not hasattr(self, "experiments"):
            return result
        trials = []
        for index, trial in enumerate(self.experiments):
            travel = trial["measured_travel_m"]
            trials.append({**copy.deepcopy(trial), "index": index,
                "remaining_m": max(0., self.experiment_limit_m-travel),
                "information_checkpoint_reached": travel >= self.information_checkpoint_m,
                "exhausted": travel >= self.experiment_limit_m-1e-9 or index in self._command_exhausted_experiments,
                "remaining_cannot_fit_useful_command": index in self._command_exhausted_experiments})
        result["recovery_experiments"] = {
            "shared_limit_m": self.observation_limit_m,
            "shared_remaining_m": max(0., self.observation_limit_m-self.observation_travel_m),
            "per_direction_limit_m": self.experiment_limit_m,
            "information_checkpoint_m": self.information_checkpoint_m,
            "image_response_aggregate_displacement_m": .001,
            "maximum_experiments": self.maximum_experiments,
            "trials": trials, "pending_reason": self._recovery_reason,
            "selection_pending_observation": self._experiment_pending is not None,
            "motion_classification_pending_observation": self._motion_classification_pending is not None,
            "last_executed_motion_charged_as_uncertain": self._charge_uncertain_motion,
            "support_change": copy.deepcopy(self._support_change),
            "accounting": "Only actual observations commit experiments and charge measured path; residual holds remain charged to the last uncertain experiment.",
            "conditional_evidence_is_clearance_certificate": False,
        }
        return result


ProjectedTranslationGuardV14 = ObservedRiskTranslationGuardV14
