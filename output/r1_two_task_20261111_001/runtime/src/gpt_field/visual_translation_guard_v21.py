"""Finite reversal of an acknowledged motion that increased projected risk.

Worsening remains recorded. This does not certify metric or image separation:
it permits a small reverse experiment only through fresh whole-scene model
geometry, then requires actual image progress at the measured checkpoint.
"""
from __future__ import annotations

from collections import deque
import copy
import math
import numpy as np

from .visual_translation_guard_v19 import ProjectedTranslationGuardV19
from .visual_monitor_v4 import _orientation_distance


class ProjectedTranslationGuardV21(ProjectedTranslationGuardV19):
    version = "proposal-conditioned-observed-motion-retreat-v21"
    retreat_command_limit_m = .003
    reverse_tolerance_rad = math.radians(30.)

    def __init__(self):
        super().__init__()
        self.worsening_motion = {}
        self._actual_motion_history = deque(maxlen=128)
        self._current_retreat_geometry = None
        self._retreat_pending = None
        self._retreat_committed = None
        self.retreat_progress = None
        self.last_directional_retreat = None

    def configure(self, *args, **kwargs):
        self._retreat_pending = None
        self._current_retreat_geometry = None
        return super().configure(*args, **kwargs)

    def recovery_directions(self):
        records = [self.worsening_motion.get(identity) for identity in sorted(self.worsening)]
        if not records or any(record is None for record in records):
            return []
        direction = -np.asarray(records[0]["measured_direction_xyz"])
        if any(self._angle(direction, -np.asarray(record["measured_direction_xyz"]))
               > self.reverse_tolerance_rad for record in records):
            return []
        return [{"direction_xyz": direction.tolist(), "source": "reverse_observed_worsening_motion",
            "nominal_away_required": False, "maximum_measured_extent_m": self.experiment_limit_m,
            "information_checkpoint_m": self.information_checkpoint_m,
            "worsening_ids": sorted(self.worsening), "requires_explicit_model_choice": True,
            "clearance_certified": False}]

    def _contract(self, records, command, rotation_rad):
        contracts = [r.get("directional_retreat_contract") for r in records]
        if not contracts or any(not isinstance(c, dict) for c in contracts):
            return None
        c = contracts[0]
        try:
            ids = c["scene_ids"]
            good = (c["version"] == "v21-current-empty-tool-retreat"
                and c["observation_epoch"] == self.sweep_observation_epoch
                and np.allclose(c["state8"], self.state, atol=1e-10, rtol=0.)
                and np.allclose(c["start_xyz_m"], self.state[:3], atol=1e-10, rtol=0.)
                and np.allclose(c["end_xyz_m"], self.state[:3]+command, atol=1e-10, rtol=0.)
                and c["field_controller"] is True and c["full_scene_sweep_accepted"] is True
                and c["outside_target_contact_band"] is True and c["empty_tool"] is True
                and c["target_contact"] == "none" and c["contact_or_visual_goal_dependency"] is False
                and c["target_id"] == self.target_id and self.target_id in ids
                and c["remaining_actions"] > 0 and c["remaining_measured_travel_m"] > 0.
                and len(ids) == len(set(ids)) and set(ids) == {r["id"] for r in records}
                and all(item == c for item in contracts)
                and all(self._proof_status(r, command, rotation_rad) == "disjoint" for r in records))
        except (KeyError, TypeError, ValueError):
            good = False
        return copy.deepcopy(c) if good else None

    def assess(self, delta_xyz, **kwargs):
        command = np.asarray(delta_xyz, float)
        events = self.events
        self._retreat_pending = None
        base = super().assess(delta_xyz, **kwargs)
        records = kwargs.get("nominal_regions") or []
        contract = self._contract(records, command, float(kwargs.get("rotation_rad", 0.)))
        # Every new selection replaces the provenance, including a rejected
        # or metadata-free proposal. A later observation cannot borrow a proof
        # from a different action or an older observation epoch.
        self._current_retreat_geometry = copy.deepcopy(contract)
        active = bool(self._retreat_committed is not None and self.context.get("translation_mode") == "separate")
        if (base.get("reason") != "translation_observed_neighbor_approach"
                and not (active and base.get("accepted") and np.linalg.norm(command) > 1e-12)):
            return base
        info = {"historical_worsening_ids": sorted(self.worsening), "nominal_away": bool(kwargs.get("nominal_away")),
            "candidate_delta_xyz_m": command.tolist(), "geometry_contract_current": contract is not None,
            "physical_budget_replenished": False, "worsening_erased": False,
            "image_separation_certified": False, "clearance_certified": False}

        def result(reason, accepted=False, cap=0., experiment=None):
            self.last_directional_retreat = {"accepted": accepted, "reason": reason, "diagnostics": copy.deepcopy(info)}
            if not accepted:
                self._motion_classification_pending = self._experiment_pending = None
            else:
                self.events = events
                self._motion_classification_pending = {"uncertain": True,
                    "direction": direction.tolist(), "command_cap_m": cap}
                self._experiment_pending = {**experiment, "command_cap_m": cap}
                self._retreat_pending = {"direction": direction.tolist(), "command_cap_m": cap,
                    "episode_ids": episode_ids, "geometry_sha256": contract["geometry_sha256"]}
            self._recovery_reason = reason
            self._v12_last_result = {"accepted": accepted, "cap_m": cap, "need_refresh": not accepted, "reason": reason}
            value = {**self._v12_last_result, "diagnostics": self.summary()}
            value["diagnostics"]["checked_proposal"] = copy.deepcopy(base.get("diagnostics", {}).get("checked_proposal"))
            value["diagnostics"]["directional_retreat"] = copy.deepcopy(self.last_directional_retreat)
            return value

        if (contract is None or kwargs.get("nominal_clear") is not True
                or self.context.get("translation_mode") != "separate"):
            return result("translation_reverse_requires_current_empty_tool_full_sweep")
        monitor = kwargs.get("monitor")
        if (monitor is None or (monitor.refs.get("external") or {}).get("moving_kind") != "tool"
                or not monitor._external_reliable()
                or monitor.cumulative_response_evidence().get("contradicted")
                or monitor.status in ("visual_response_contradiction", "target_anchor_shifted")):
            return result("translation_reverse_reference_unavailable_or_contradicted")
        direction = command/np.linalg.norm(command)
        episode_ids = sorted(self.worsening) if self.worsening else list(self._retreat_committed["episode_ids"])
        histories = [self.worsening_motion.get(identity) for identity in episode_ids]
        if not histories or any(history is None for history in histories):
            return result("translation_reverse_actual_worsening_motion_unavailable")
        for history in histories:
            if (history["geometry_sha256"] != contract["geometry_sha256"]
                    or np.max(np.abs(np.asarray(history["state8"])[6:]-self.state[6:])) > .002
                    or _orientation_distance(np.asarray(history["state8"])[3:6], self.state[3:6]) > .035):
                return result("translation_reverse_worsening_geometry_or_domain_changed")
            if self._angle(direction, -np.asarray(history["measured_direction_xyz"])) > self.reverse_tolerance_rad:
                return result("translation_reverse_does_not_reverse_observed_worsening_motion")
        info["reversed_measured_motion"] = copy.deepcopy(histories)
        progress = self.retreat_progress
        if progress is not None and progress.get("further_worsening"):
            return result("translation_reverse_observed_further_worsening")
        if (progress is not None and progress["measured_travel_m"] >= self.information_checkpoint_m-1e-9
                and not progress.get("all_trigger_gaps_improved")):
            return result("translation_reverse_checkpoint_requires_observed_gap_progress")
        # This True selects only the existing finite *experiment* accounting.
        # Nominal-away input remains false in diagnostics: the admission is a
        # observed-motion reversal, not a new claim of metric separation.
        experiment, error = self._experiment_for(direction, nominal_away=True)
        if error:
            return result(error)
        used = 0. if experiment["index"] is None else self.experiments[experiment["index"]]["measured_travel_m"]
        remaining = min(max(0., self.observation_limit_m-self.observation_travel_m),
                        max(0., self.experiment_limit_m-used), contract["remaining_measured_travel_m"])
        if progress is None or not progress.get("all_trigger_gaps_improved"):
            remaining = min(remaining, max(0., self.information_checkpoint_m-(0. if progress is None else progress["measured_travel_m"])))
        if remaining <= 1e-9:
            return result("translation_reverse_measured_allowance_exhausted")
        requested = min(float(np.linalg.norm(command)), self.retreat_command_limit_m)
        response = kwargs.get("translation_response")
        if response is not None:
            conversion = response.command_cap_for_measured_limit(direction, remaining, requested, self.state)
            cap = min(requested, float(conversion["command_cap_m"]))
            info["measured_to_command_conversion"] = copy.deepcopy(conversion)
        else:
            cap = min(requested, remaining)
        info.update(measured_remaining_m=remaining, checkpoint_m=self.information_checkpoint_m,
                    shared_remaining_m=max(0., self.observation_limit_m-self.observation_travel_m))
        if cap < self.minimum_command_m-1e-9:
            return result("translation_reverse_checkpoint_has_no_useful_command")
        return result("projected_worsening_bounded_observed_motion_reverse", True, cap, experiment)

    def observe(self, images, state8, action, *, monitor=None):
        before = None if self.state is None else self.state.copy()
        first = self._window[0] if self._window else self._previous_snapshot
        window_start = None if first is None else np.asarray(first["state"]).copy()
        previous_worsening = set(self.worsening)
        geometry = copy.deepcopy(self._current_retreat_geometry)
        pending = self._retreat_pending
        self._retreat_pending = None
        command = np.asarray(action[:3], float)*.05
        size = float(np.linalg.norm(command))
        geometry_acknowledged = bool(geometry is not None and before is not None
            and size > 1e-8 and geometry.get("observation_epoch") == self.sweep_observation_epoch
            and np.allclose(geometry["state8"], before, atol=1e-10, rtol=0.)
            and np.allclose(np.asarray(geometry["end_xyz_m"])-geometry["start_xyz_m"], command,
                            atol=1e-10, rtol=0.))
        matched = bool(pending is not None and size > 1e-8
            and self._angle(command/size, pending["direction"]) < 1e-5
            and size <= pending["command_cap_m"]+1e-9)
        baseline = self._snapshot()
        if before is not None and not self._actual_motion_history:
            self._actual_motion_history.append({"state8": before.tolist(),
                "tool_box_px": None if self.tool_box is None else self.tool_box.tolist()})
        super().observe(images, state8, action, monitor=monitor)
        now = np.asarray(state8, float)
        self._actual_motion_history.append({"state8": now.tolist(),
            "tool_box_px": None if self.tool_box is None else self.tool_box.tolist()})
        newly_worse = self.worsening-previous_worsening
        if newly_worse and window_start is not None and geometry_acknowledged:
            delta = now[:3]-window_start
            distance = float(np.linalg.norm(delta))
            if distance >= .001 and not self.domain_invalid:
                for identity in newly_worse:
                    self.worsening_motion[identity] = {"measured_direction_xyz": (delta/distance).tolist(),
                        "measured_net_translation_m": distance, "window_start_xyz_m": window_start.tolist(),
                        "window_end_xyz_m": now[:3].tolist(), "state8": now.tolist(),
                        "geometry_sha256": geometry["geometry_sha256"],
                        "observation_epoch": self.sweep_observation_epoch,
                        "risk_evidence": copy.deepcopy(self.risk_windows.get(identity))}
        if matched:
            if self.retreat_progress is None:
                self.retreat_progress = {"episode_ids": pending["episode_ids"], "measured_travel_m": 0.,
                    "acknowledged_commands": 0, "baseline": copy.deepcopy(baseline),
                    "all_trigger_gaps_improved": False, "further_worsening": False, "per_neighbor": {}}
            self._retreat_committed = copy.deepcopy(pending)
        charged = matched or (size <= 1e-8 and self._retreat_committed is not None and self._charge_uncertain_motion)
        if charged and before is not None and self.retreat_progress is not None:
            progress = self.retreat_progress
            progress["measured_travel_m"] += float(np.linalg.norm(now[:3]-before[:3]))
            progress["acknowledged_commands"] += int(matched)
            current, original = self._snapshot(), progress["baseline"]
            statuses = []
            for identity in progress["episode_ids"]:
                a = None if original is None else original["neighbors"].get(identity)
                b = None if current is None else current["neighbors"].get(identity)
                if a is None or b is None:
                    statuses.append(False); continue
                change = b["gap"]-a["gap"]
                noise = max(.75, 3*np.hypot(a["noise"], b["noise"]))
                improved = progress["measured_travel_m"] >= .001 and change > noise
                if progress["measured_travel_m"] >= .001 and change < -noise:
                    progress["further_worsening"] = True
                statuses.append(improved)
                progress["per_neighbor"][identity] = {"gap_change_px": float(change),
                    "threshold_px": float(noise), "improved": bool(improved)}
            progress["all_trigger_gaps_improved"] = bool(statuses and all(statuses))
        return self.summary()

    def summary(self):
        result = super().summary()
        if not hasattr(self, "worsening_motion"):
            return result
        progress = copy.deepcopy(self.retreat_progress)
        if progress is not None:
            progress.pop("baseline", None)
        result["directional_retreat"] = {"last_admission": copy.deepcopy(self.last_directional_retreat),
            "actual_worsening_motion": copy.deepcopy(self.worsening_motion),
            "candidate_directions": self.recovery_directions(), "progress": progress,
            "command_limit_m": self.retreat_command_limit_m,
            "uses_existing_shared_and_heading_debt": True, "worsening_erased_by_admission": False,
            "extent_scale_removed": False, "clearance_certified": False}
        return result


ObservedRiskTranslationGuardV21 = ProjectedTranslationGuardV21
