"""Metric Field tracking with explicitly separated measured/command limits."""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_v10 import MetricFieldExecutorV10
from .visual_field_v11 import MetricFieldExecutorV11


class MetricFieldExecutorV12(MetricFieldExecutorV11):
    def __init__(self):
        super().__init__()
        self.robot_response = None
        self.last_measured_allowance = None

    def bind_translation_guard(self, guard, monitor, robot_response=None):
        super().bind_translation_guard(guard, monitor)
        self.robot_response = robot_response

    def nominal_regions_for_segment(self, start, end):
        start, end = np.asarray(start, float), np.asarray(end, float)
        segment_low, segment_high = np.minimum(start, end), np.maximum(start, end)
        result = []
        for region in self.region_geometry:
            distances = []
            for box in region["tool_core_parts"]:
                distance = np.maximum(np.maximum(np.asarray(box["lower_xyz"])-segment_high,
                                                segment_low-np.asarray(box["upper_xyz"])), 0.)
                distances.append(float(np.linalg.norm(distance)))
            # Box enclosing the segment gives a LOWER bound on segment distance.
            # This may overstate risk but cannot invent remote clearance.
            result.append({"id": region["id"], "core_distance_m": min(distances),
                "uncertainty_m": region["margins_m"]["combined_uncertainty"],
                "kind": region["kind"], "geometry_is_model_hypothesis": True})
        return result

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        if self.pending is not None:
            raise RuntimeError("observe must acknowledge the previous accepted action")
        count = self.orientation_goal_refresh_commands
        rotation = self.last_orientation_command_rad.copy()
        self.last_measured_allowance = {"remaining_measured_m": measured_cap_m,
                                       "conversion": None}
        cap = command_cap_m
        result = None
        admission = None
        # Selection is reversible: only observe() consumes a real action.
        # Recheck the exact smaller command after any supported-scale reduction.
        for attempt in range(3):
            self.pending = None
            self.orientation_goal_refresh_commands = count
            self.last_orientation_command_rad = rotation.copy()
            result = MetricFieldExecutorV10.select(self, state8, joint_position,
                                                   command_cap_m=cap)
            if not result["accepted"]:
                return result
            delta = np.asarray(result["action"][:3], float)*.05
            norm = float(np.linalg.norm(delta))
            if measured_cap_m is not None:
                conversion = self.robot_response.command_cap_for_measured_limit(
                    delta/norm, measured_cap_m, norm, state8, joint_position=joint_position)
                self.last_measured_allowance["conversion"] = copy.deepcopy(conversion)
                allowed = float(conversion["command_cap_m"])
                if allowed < norm-1e-10:
                    cap = allowed if cap is None else min(cap, allowed)
                    if cap < self.MIN_USEFUL_COMMAND_M:
                        admission = {"accepted": False, "reason": "measured_allowance_has_no_useful_supported_command",
                                     "diagnostics": copy.deepcopy(self.last_measured_allowance)}
                        break
                    continue
            if self.translation_guard is None:
                return result
            point = np.asarray(state8[:3], float)
            admission = self.translation_guard.assess(delta, monitor=self.visual_monitor,
                nominal_away=self._nominal_away(point, delta),
                nominal_clear=bool(self.checked.segment_valid(point, point+delta)),
                nominal_regions=self.nominal_regions_for_segment(point, point+delta),
                translation_response=self.robot_response,
                rotation_rad=float(np.linalg.norm(result["action"][3:6])*.5),
                orientation_hold=True)
            allowed = admission.get("cap_m")
            if not admission["accepted"]:
                break
            if allowed is not None and allowed < norm-1e-10:
                cap = allowed if cap is None else min(cap, allowed)
                if cap < self.MIN_USEFUL_COMMAND_M:
                    admission = {**admission, "accepted": False,
                                 "reason": "translation_allowance_has_no_useful_supported_command"}
                    break
                continue
            self.last_translation_admission = copy.deepcopy(admission)
            result["diagnostics"].update(translation_admission=copy.deepcopy(admission),
                measured_control_allowance=copy.deepcopy(self.last_measured_allowance))
            return result
        self.pending = None
        self.orientation_goal_refresh_commands = count
        self.last_orientation_command_rad = rotation
        if admission is None or admission.get("accepted"):
            admission = {"accepted": False, "reason": "translation_supported_scale_requires_replan"}
        self.last_translation_admission = copy.deepcopy(admission)
        return self._stop(admission["reason"], {"translation_admission": copy.deepcopy(admission)})

    def summary(self):
        result = super().summary()
        result.update(version="measured-response-risk-field-v12",
                      measured_control_allowance=copy.deepcopy(getattr(self, "last_measured_allowance", None)))
        return result
