"""Event-triggered projected-neighborhood screening for opt-in V11.

Model rectangles remain hypotheses, not segmentation or calibrated geometry.
A tracked distal-tool part transports the *declared* whole-tool rectangle while
orientation/fingers stay fixed. This is not a full-tool or 3-D clearance proof.
Only RGB and measured robot motion enter this module.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_field import _gray, SupportedLinearModel
from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import LocalFeatureBundle
from .visual_rotation_guard_v8 import _box


def _corners(box):
    a, b = np.asarray(box).reshape(2, 2)
    return np.array([a, [b[0], a[1]], b, [a[0], b[1]]])


def _bounds(points):
    return np.r_[np.min(points, axis=0), np.max(points, axis=0)]


def _gap(a, b):
    """Positive axis separation, negative minimum rectangular penetration."""
    return float(max(b[0]-a[2], a[0]-b[2], b[1]-a[3], a[1]-b[3]))


def _iou(a, b):
    overlap = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    area = float(np.prod(overlap))
    return area/max(1e-9, np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area)


class ProjectedTranslationGuardV11:
    """Cheap local admission; far paths keep their existing command horizon.

    ``assess`` consumes metric translation and returns a cap; it never executes
    or clips a command. Caller must reselect and reassess if a cap is smaller.
    ``observe`` alone accounts acknowledged measured travel. Cold separation is
    an explicit nominal-away model prior with a lifetime 5 mm allowance, not
    evidence of clearance. Configure/IDs/reseeds never replenish that allowance.
    """
    version = "tracked-projected-translation-guard-v11"
    cold_separation_limit_m = .005
    minimum_command_m = .001
    boundary_margin_px = 2.

    def __init__(self):
        self.context = None
        self.target_id = None
        self.state = None
        self.scale = None
        self.tool_track = None
        self.tool_source = None
        self.tool_box = None
        self.neighbors = []
        self.response = self._model()
        self.validations = 0
        self.prediction_errors = []
        self.scene_motion_px = 0.
        self.domain_invalid = None
        self.context_invalid = "translation_context_missing"
        self.cold_separation_travel_m = 0.
        self._pending_cold = False
        self.last_result = None
        self._tool_previous = None
        self._last_box = None
        self._regressing_escape = False
        self.cold_separation_contradicted = False
        self._pending_gaps = None
        self.events = 0

    @staticmethod
    def _model():
        return SupportedLinearModel(residual_limit=1.5, min_samples=4, min_rank=1,
                                    max_samples=24, minimum_excitation=.001)

    @staticmethod
    def _tool(monitor):
        if monitor is None:
            return None
        ext = getattr(monitor, "refs", {}).get("external") or {}
        return monitor.tracks.get("tool" if ext.get("moving_kind") == "payload" else "moving")

    @staticmethod
    def _noise(track):
        return max(.08, float(track.last_residual_px)/np.sqrt(max(3, track.last_inliers)))

    @staticmethod
    def _seed_neighbor(pixels, box, identity):
        # Bounded sampling INSIDE one model-named ROI, never whole-scene search.
        candidates = []
        for fx, fy in ((.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            try:
                track = LocalFeatureBundle(pixels, box[:2]+(box[2:]-box[:2])*[fx, fy],
                    identity, support_radius_px=12., maximum_features=8, roi_uv=box)
                candidates.append(track)
            except ValueError:
                pass
        return max(candidates, key=lambda t: t.last_inliers) if candidates else None

    @staticmethod
    def _transport(record):
        track = record["track"]
        if track is None or not track.reliable:
            return record["box"]
        return _bounds(record["source"]@track.matrix+track.translation)

    def configure(self, images, motion_context, state8, *, target_id, monitor=None):
        state = np.asarray(state8, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state8 must contain eight finite values")
        pixels = _gray(images["external"])
        self.scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], float)
        self.context, self.target_id = copy.deepcopy(motion_context), target_id
        self.context_invalid = None
        self._pending_cold = False
        self._regressing_escape = False
        context = motion_context if isinstance(motion_context, dict) else {}
        tool = _box(context.get("tool_external_roi_uv"))
        neighbors = context.get("nearby_obstacles")
        assessment, mode = context.get("obstacle_assessment"), context.get("translation_mode")
        if tool is None:
            self.context_invalid = "translation_tool_envelope_unavailable"
        elif (mode not in ("transit", "separate", "interaction") or
              assessment not in ("present", "none_relevant") or not isinstance(neighbors, list)
              or (assessment == "present" and not neighbors)
              or (assessment == "none_relevant" and neighbors)):
            self.context_invalid = "translation_neighborhood_unobservable_or_inconsistent"
        elif ((mode == "interaction" and context.get("intended_contact_id") != target_id)
              or (mode != "interaction" and context.get("intended_contact_id") is not None)):
            self.context_invalid = "translation_contact_role_inconsistent"
        if self.context_invalid:
            self.state = state.copy()
            return self.summary()
        incoming = []
        for n in neighbors:
            if not isinstance(n, dict) or not isinstance(n.get("id"), str) or not n["id"].strip():
                self.context_invalid = "translation_neighbor_invalid"
                break
            box = _box(n.get("external_roi_uv"))
            if box is None:
                self.context_invalid = "translation_neighbor_invalid"
                break
            incoming.append((n["id"], box))
        if self.context_invalid:
            self.state = state.copy()
            return self.summary()
        # Retain scene entries, including renamed or omitted overlapping objects.
        # A refreshed smaller rectangle cannot erase a previously observed risk.
        for identity, box in incoming:
            pixel_box = box*np.tile(self.scale, 2)
            old = next((n for n in self.neighbors if n["id"] == identity or
                        _iou(self._transport(n), pixel_box) >= .5), None)
            if old is not None and old["track"] is not None and old["track"].reliable:
                prior = self._transport(old)
                combined = np.r_[np.minimum(prior[:2], pixel_box[:2]), np.maximum(prior[2:], pixel_box[2:])]
                old["box"] = combined
                old["source"] = (_corners(combined)-old["track"].translation)@np.linalg.inv(old["track"].matrix)
                old["aliases"].add(identity)
                continue
            track = self._seed_neighbor(pixels, box, identity)
            record = {"id": identity, "aliases": {identity}, "box": pixel_box,
                      "track": track, "source": _corners(pixel_box)}
            if old is None:
                self.neighbors.append(record)
            else:
                record["aliases"] |= old["aliases"]
                # Never silently shrink a lost/renamed region on a new request.
                prior = old["box"]
                record["box"] = np.r_[np.minimum(prior[:2], pixel_box[:2]), np.maximum(prior[2:], pixel_box[2:])]
                record["source"] = _corners(record["box"])
                self.neighbors[next(i for i, value in enumerate(self.neighbors) if value is old)] = record
        track = self._tool(monitor)
        changed = track is not self.tool_track
        changed_configuration = self.domain_invalid == "translation_tool_configuration_changed"
        invalidated_mapping = self.domain_invalid is not None
        self.tool_track = track
        proposed = tool*np.tile(self.scale, 2)
        if track is not None and track.reliable:
            padding = max(.5, 3*self._noise(track))
            if np.any(track.position < proposed[:2]-padding) or np.any(track.position > proposed[2:]+padding):
                self.context_invalid = "translation_tool_reference_outside_declared_envelope"
        if self.tool_box is not None and not changed and not changed_configuration and track is not None and track.reliable:
            proposed = np.r_[np.minimum(self.tool_box[:2], proposed[:2]), np.maximum(self.tool_box[2:], proposed[2:])]
        self.tool_box = proposed
        self.tool_source = None
        if track is not None and track.reliable:
            self.tool_source = (_corners(proposed)-track.translation)@np.linalg.inv(track.matrix)
        if changed or invalidated_mapping:
            self.response, self.validations, self.prediction_errors = self._model(), 0, []
        self.domain_invalid = None
        self.scene_motion_px = 0.
        self._tool_previous = None if track is None or not track.reliable else track.position.copy()
        self.state = state.copy()
        self._last_box = None if self.tool_box is None else self.tool_box.copy()
        return self.summary()

    @property
    def response_ready(self):
        if self.response.matrix is None or self.validations < 2 or np.linalg.norm(self.response.matrix) < 20:
            return False
        rows = self.prediction_errors[-8:]
        return bool(rows and sum(x[0] for x in rows) < .75*sum(x[1] for x in rows))

    def observe(self, images, state8, action, *, monitor=None):
        state = np.asarray(state8, float)
        if self.state is None:
            raise RuntimeError("configure guard before observe")
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state8 must contain eight finite values")
        delta = state[:3]-self.state[:3]
        distance = float(np.linalg.norm(delta))
        was_cold = self._pending_cold
        if was_cold:
            self.cold_separation_travel_m += distance
        self._pending_cold = False
        rotation = _orientation_distance(state[3:6], self.state[3:6])
        if rotation > .015 or np.max(np.abs(state[6:]-self.state[6:])) > .002:
            self.domain_invalid = "translation_tool_configuration_changed"
        pixels = _gray(images["external"])
        neighbor_flows = []
        for n in self.neighbors:
            track = n["track"]
            previous = None if track is None or not track.reliable else track.position.copy()
            if track is not None and track.reliable and track.track(pixels):
                neighbor_flows.append(track.position-previous)
                n["box"] = self._transport(n)
        common = np.median(neighbor_flows, axis=0) if neighbor_flows else np.zeros(2)
        self.scene_motion_px = float(np.linalg.norm(common))
        if self.scene_motion_px > 1.:
            self.domain_invalid = "translation_scene_motion_unresolved"
        track = self._tool(monitor)
        if track is not self.tool_track or track is None or not track.reliable or self.tool_source is None:
            self.domain_invalid = "translation_tool_correspondence_lost"
        else:
            next_box = _bounds(self.tool_source@track.matrix+track.translation)
            if self._tool_previous is not None and not self.domain_invalid:
                flow = track.position-self._tool_previous-common
                predicted = self.response.predict(delta)
                if predicted is not None and distance > 1e-6:
                    err = float(np.linalg.norm(flow-predicted))
                    self.prediction_errors.append((err**2, float(np.dot(flow, flow))))
                    self.prediction_errors = self.prediction_errors[-8:]
                    if err <= max(1., 3*self._noise(track)):
                        self.validations += 1
                    else:
                        self.validations = 0
                if distance > 1e-6:
                    self.response.add(delta, flow)
            self._tool_previous = track.position.copy()
            self.tool_box = next_box
        if was_cold and self._pending_gaps is not None and not self.domain_invalid:
            now = {r["id"]: r for r in self._risks()}
            if any(identity in now and now[identity]["gap_px"] < gap-max(.5, 3*self._noise(track))
                   for identity, gap in self._pending_gaps.items()):
                self.cold_separation_contradicted = True
        self._pending_gaps = None
        self.state = state.copy()
        return self.summary()

    def _risks(self):
        if self.tool_box is None:
            return []
        noise = self._noise(self.tool_track) if self.tool_track is not None else .5
        rows = []
        for n in self.neighbors:
            track = n["track"]
            box = self._transport(n)
            neighbor_noise = self._noise(track) if track is not None and track.reliable else 1.
            margin = self.boundary_margin_px+3*np.hypot(noise, neighbor_noise)
            # Exact intended target may be contacted; every other object remains.
            intended = (self.context and self.context.get("translation_mode") == "interaction"
                        and n["id"] == self.target_id and n["aliases"] == {self.target_id})
            rows.append({"id": n["id"], "aliases": sorted(n["aliases"]), "box_px": box.tolist(),
                         "gap_px": _gap(self.tool_box, box), "margin_px": float(margin),
                         "track_reliable": bool(track is not None and track.reliable),
                         "intended_contact": bool(intended)})
        return rows

    def assess(self, delta_xyz, *, monitor=None, nominal_away=False, rotation_rad=0.):
        delta = np.asarray(delta_xyz, float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("delta_xyz must be a finite metric 3-vector")
        norm = float(np.linalg.norm(delta))
        self._pending_cold = False
        result = {"accepted": True, "cap_m": None, "need_refresh": False,
                  "reason": "projected_translation_no_conflict", "diagnostics": {}}
        def finish(reason=None, cap=None):
            if reason:
                result.update(accepted=False, cap_m=0., need_refresh=True, reason=reason)
                self.events += 1
            else:
                result["cap_m"] = cap
            self.last_result = copy.deepcopy(result)
            result["diagnostics"] = self.summary()
            self.last_result = {k: copy.deepcopy(v) for k, v in result.items() if k != "diagnostics"}
            return result
        if norm < 1e-12:
            return finish()
        if self.context_invalid:
            return finish(self.context_invalid)
        if self.domain_invalid:
            return finish(self.domain_invalid)
        if abs(rotation_rad) > .015:
            return finish("translation_requested_rotation_outside_tracking_domain")
        track = self._tool(monitor) if monitor is not None else self.tool_track
        if track is None or track is not self.tool_track or not track.reliable:
            return finish("translation_tool_correspondence_lost")
        risks = [r for r in self._risks() if not r["intended_contact"]]
        if np.any(self.tool_box[:2] < 0) or np.any(self.tool_box[2:] > self.scale):
            return finish("translation_tool_envelope_outside_view")
        # Predict only inside the actually supported scale/subspace. A requested
        # command outside that scale is *not* linearly extrapolated as certainty.
        predicted = self.response.predict(delta) if self.response_ready else None
        swept = None if predicted is None else np.r_[np.minimum(self.tool_box[:2], self.tool_box[:2]+predicted),
                                                      np.maximum(self.tool_box[2:], self.tool_box[2:]+predicted)]
        current = [r for r in risks if r["gap_px"] <= r["margin_px"]]
        imminent = [r for r in risks if swept is not None and _gap(swept, r["box_px"]) <= r["margin_px"]]
        if current and self.context.get("translation_mode") != "separate":
            if any(not r["track_reliable"] for r in current):
                return finish("translation_nearby_obstacle_track_unavailable")
            return finish("translation_projected_neighbor_conflict")
        if predicted is None and self.response_ready and self.response.samples:
            # This empirical envelope only TRIGGERS a restriction. It does not
            # turn extrapolated flow into positive evidence of free space.
            gain = max(np.linalg.norm(y)/max(1e-9, np.linalg.norm(x))
                       for x, y in self.response.samples)
            possible_reach_px = gain*norm + 3*max(.08, self.response.residual or 0.)
            near_horizon = any(r["gap_px"] <= r["margin_px"]+possible_reach_px for r in risks)
            support = 1.5*max(np.linalg.norm(row[0]) for row in self.response.samples)
            supported_delta = delta*min(1., support/max(norm, 1e-12))
            if near_horizon and self.response.predict(supported_delta) is not None and norm > support+1e-9:
                if support < self.minimum_command_m:
                    return finish("translation_response_scale_below_useful_motion_floor")
                result["reason"] = "projected_translation_response_scale_cap"
                return finish(cap=float(support))
            if near_horizon and not current and self.response.predict(supported_delta) is None:
                return finish("translation_nearby_direction_unobserved")
        if not current and not imminent:
            # Missing obstacle texture is not interpreted as disappearance.
            # Far declared rectangles may be used under the existing policy's
            # bounded unvalidated-motion gates; near lost texture forces review.
            return finish()
        if any(not r["track_reliable"] for r in current+imminent):
            return finish("translation_nearby_obstacle_track_unavailable")
        if self.context.get("translation_mode") != "separate":
            return finish("translation_projected_neighbor_conflict")
        if imminent and not current:
            return finish("translation_separation_would_enter_new_neighbor")
        if predicted is not None:
            moved = self.tool_box+np.tile(predicted, 2)
            # All current conflicts must improve beyond the measured noise, and
            # the swept box may not introduce a conflict with another neighbor.
            improving = all(_gap(moved, r["box_px"])-r["gap_px"] >=
                            max(.15, self._noise(track)) for r in current)
            entering = any(r not in current and _gap(swept, r["box_px"]) <= r["margin_px"] for r in risks)
            if improving and not entering:
                result["reason"] = "projected_separation_supported"
                return finish()
            return finish("translation_separation_not_supported_away")
        if self.response_ready and self.response.samples:
            support = 1.5*max(np.linalg.norm(row[0]) for row in self.response.samples)
            if norm > support+1e-9 and self.response.predict(delta*support/norm) is not None:
                if support < self.minimum_command_m:
                    return finish("translation_response_scale_below_useful_motion_floor")
                result["reason"] = "projected_separation_response_scale_cap"
                return finish(cap=float(support))
        remaining = max(0., self.cold_separation_limit_m-self.cold_separation_travel_m)
        if self.cold_separation_contradicted:
            return finish("translation_cold_separation_contradicted")
        if not nominal_away:
            return finish("translation_separation_direction_unobserved")
        if remaining < self.minimum_command_m-1e-9:
            return finish("translation_cold_separation_budget_exhausted")
        result["reason"] = "projected_separation_bounded_nominal_prior"
        self._pending_cold = True
        self._pending_gaps = {r["id"]: r["gap_px"] for r in current}
        return finish(cap=min(.002, remaining))

    def summary(self):
        risks = self._risks()
        conflicts = [r["id"] for r in risks if not r["intended_contact"] and r["gap_px"] <= r["margin_px"]]
        return {"version": self.version, "configured": self.context_invalid is None,
                "translation_mode": None if not self.context else self.context.get("translation_mode"),
                "tool_box_px": None if self.tool_box is None else self.tool_box.tolist(),
                "conflicting_obstacle_ids": conflicts, "neighbors": risks,
                "response": self.response.summary(), "response_validated": self.response_ready,
                "held_out_validations": self.validations, "scene_motion_px": self.scene_motion_px,
                "domain_invalid": self.domain_invalid, "cold_separation_travel_m": self.cold_separation_travel_m,
                "cold_separation_contradicted": self.cold_separation_contradicted,
                "cold_separation_remaining_m": max(0., self.cold_separation_limit_m-self.cold_separation_travel_m),
                "last_admission": copy.deepcopy(self.last_result), "event_count": self.events,
                "clearance_certified": False, "semantic_identity_certified": False,
                "geometry_source": "model whole-tool/neighbor rectangles transported by RGB correspondences",
                "limitations": ["2D overlap is ambiguity, not proof of 3D collision",
                    "declared tool extent and unlisted objects are not independently certified",
                    "part similarity transform only approximates whole-tool perspective",
                    "cold separation is a finite nominal-geometry prior, not observed free space"]}
