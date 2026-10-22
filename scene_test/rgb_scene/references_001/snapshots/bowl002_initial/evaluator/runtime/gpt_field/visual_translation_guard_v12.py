"""Observed projected-risk progress, distinct from model occupancy (opt-in V12).

A 2-D bounding-box overlap is ambiguous. Known/model hard geometry is checked
by the caller; this observer watches fresh relative image motion and admits a
finite, explicitly uncertain observation trajectory. No camera calibration,
depth, scene poses, contact signal or clearance certificate is consumed.
"""
from __future__ import annotations

import copy
from collections import deque
import numpy as np

from .visual_field import SupportedLinearModel
from .visual_translation_guard_v11 import ProjectedTranslationGuardV11, _box, _corners, _gap


class ObservedRiskTranslationGuardV12(ProjectedTranslationGuardV11):
    version = "displacement-adaptive-projected-risk-v12"
    observation_limit_m = .020
    uncertain_command_limit_m = .005
    minimum_command_m = .001
    window_max_frames = 64

    def __init__(self):
        super().__init__()
        self.observation_travel_m = 0.
        self._charge_uncertain_motion = False
        self._metric = {}
        self._last_nominal_clear = False
        self._window = deque(maxlen=self.window_max_frames)
        self.risk_windows = {}
        self.worsening = set()
        self.improving = set()
        self.cumulative_response = self._cumulative_model()
        self._response_origin = None
        self._response_error = deque(maxlen=8)
        self._response_extent_m = 0.
        self._response_age = 0
        self._response_track = None
        self.envelope_revision = {"accepted": False, "reason": "not_configured"}
        self._v12_last_result = None
        self._last_counterfactual_observation = None
        self._previous_snapshot = None

    @staticmethod
    def _cumulative_model():
        return SupportedLinearModel(residual_limit=1.5, min_samples=3, min_rank=1,
                                    max_samples=24, minimum_excitation=.001)

    def _template_contained(self, box, track):
        if track is None or not track.reliable:
            return False
        pixel_box = box*np.tile(track.scale, 2)
        points = track.points
        if len(points) < 3:
            return False
        # Include live descriptor support, not only its claimed seed center.
        return bool(np.all(points.min(axis=0)-3 >= pixel_box[:2]-.5)
                    and np.all(points.max(axis=0)+3 <= pixel_box[2:]+.5)
                    and np.all(track.position >= pixel_box[:2]-.5)
                    and np.all(track.position <= pixel_box[2:]+.5))

    def configure(self, images, motion_context, state8, *, target_id, monitor=None):
        old_box = None if self.tool_box is None else self.tool_box.copy()
        old_track = self.tool_track
        incoming = _box(motion_context.get("tool_external_roi_uv")) if isinstance(motion_context, dict) else None
        current_track = self._tool(monitor)
        revision_ok = incoming is not None and self._template_contained(incoming, current_track)
        if revision_ok:
            # The wire's CURRENT envelope is a fresh geometry claim. Updating
            # it does not replace the physical feature, erase neighbors, spend
            # or replenish observation debt, or assert an improvement in gap.
            self.tool_box = None
        super().configure(images, motion_context, state8, target_id=target_id, monitor=monitor)
        self.envelope_revision = {"accepted": revision_ok,
            "reason": "current_roi_contains_live_descriptor_support" if revision_ok else "current_roi_missing_live_descriptor_support",
            "previous_box_px": None if old_box is None else old_box.tolist(),
            "current_box_px": None if self.tool_box is None else self.tool_box.tolist(),
            "feature_identity_preserved": current_track is old_track,
            "is_observed_motion": False}
        if not revision_ok and self.context_invalid is None:
            self.context_invalid = "translation_current_envelope_support_inconsistent"
        if current_track is not self._response_track:
            self._response_track = current_track
            self.cumulative_response = self._cumulative_model()
            self._response_error.clear()
            self._response_origin = None
            self._response_extent_m = 0.
            self._response_age = 0
        self._window.clear()  # A box revision is NEVER measured gap progress.
        self.risk_windows = {}
        self._metric = {}
        self._last_nominal_clear = False
        self._previous_snapshot = self._snapshot()
        # Keep _charge_uncertain_motion through holds/refresh, until genuinely
        # clear/reliably supported execution replaces the uncertain episode.
        return self.summary()

    def _gap_noise(self, record):
        def transported_noise(track, box):
            if track is None or not track.reliable:
                return 2.
            span = np.linalg.norm(np.ptp(track.points, axis=0)) if len(track.points) else 0.
            extent = np.linalg.norm(np.asarray(box)[2:]-np.asarray(box)[:2])
            # Extrapolating a small part's affine transform to a full envelope
            # magnifies localization noise. Never treat part residual as whole
            # tool boundary accuracy.
            lever = max(1., extent/max(8., span))
            return max(.15, self._noise(track)*lever)
        neighbor = next((r for r in self.neighbors if r["id"] == record["id"]), None)
        t = transported_noise(self.tool_track, self.tool_box)
        n = transported_noise(None if neighbor is None else neighbor["track"], record["box_px"])
        return float(np.hypot(t, n))

    def _snapshot(self):
        tool = self.tool_track
        if tool is None or not tool.reliable or self.state is None or self.tool_box is None:
            return None
        return {"state": self.state[:3].copy(), "tool": tool.position.copy(),
                "neighbors": {r["id"]: {"gap": r["gap_px"], "noise": self._gap_noise(r),
                    "position": n["track"].position.copy(), "near": self._near_metric(r["id"])}
                    for r in self._risks() for n in self.neighbors if n["id"] == r["id"]
                    and n["track"] is not None and n["track"].reliable}}

    def _near_metric(self, identity):
        value = self._metric.get(identity)
        # Missing metric identity remains uncertain, never automatically remote.
        if value is None:
            return True
        distance, sigma = value
        return distance <= 3*sigma or identity in self.worsening

    @property
    def cumulative_response_ready(self):
        model = self.cumulative_response
        if model.matrix is None or np.linalg.norm(model.matrix) < 20 or len(self._response_error) < 2:
            return False
        error = sum(v[0] for v in self._response_error)
        zero = sum(v[1] for v in self._response_error)
        return bool(zero > .25 and error < .75*zero)

    def observe(self, images, state8, action, *, monitor=None):
        previous = self._previous_snapshot
        previous_state = self.state.copy() if self.state is not None else None
        charge = self._charge_uncertain_motion
        # Legacy tracking/scene-motion checks only. V12 owns its spatial debt.
        self._pending_cold = False
        super().observe(images, state8, action, monitor=monitor)
        if charge and previous_state is not None:
            self.observation_travel_m += float(np.linalg.norm(np.asarray(state8[:3])-previous_state[:3]))
        current = self._snapshot()
        self._previous_snapshot = current
        if previous is None or current is None or self.domain_invalid:
            self._window.clear()
            self._response_origin = None
            self._response_age = 0
            return self.summary()
        if not self._window:
            self._window.append(previous)
        self._window.append(current)
        first = self._window[0]
        path = float(sum(np.linalg.norm(b["state"]-a["state"]) for a,b in zip(self._window,list(self._window)[1:])))
        net = current["state"]-first["state"]
        self.risk_windows = {}
        resolved_any = False
        for identity in first["neighbors"].keys() & current["neighbors"].keys():
            start, end = first["neighbors"][identity], current["neighbors"][identity]
            change = float(end["gap"]-start["gap"])
            threshold = max(.75, 3*np.hypot(start["noise"], end["noise"]))
            relative = (current["tool"]-first["tool"])-(end["position"]-start["position"])
            near = self._near_metric(identity)
            attention = min(start["gap"], end["gap"]) <= max(6., threshold+2.)
            state = "inconclusive"
            if path >= .001 and attention and near and change < -threshold:
                self.worsening.add(identity);self.improving.discard(identity)
                state = "worsening";resolved_any = True
            elif path >= .001 and near and change > threshold:
                self.worsening.discard(identity);self.improving.add(identity)
                state = "improving";resolved_any = True
            self.risk_windows[identity] = {"state": state, "gap_change_px": change,
                "threshold_px": threshold, "measured_travel_m": path,
                "net_measured_translation_m": float(np.linalg.norm(net)),
                "relative_tool_neighbor_flow_px": relative.tolist(),
                "model_near": near, "transitions": len(self._window)-1}
        if resolved_any:
            self._window.clear();self._window.append(current)
        # Fit independently observed displacement aggregates. Subpixel control
        # transitions accumulate before training/held-out testing the Jacobian.
        if self._response_origin is None or self._response_age >= self.window_max_frames:
            self._response_origin = previous
            self._response_age = 0
        self._response_age += 1
        start = self._response_origin
        displacement = current["state"]-start["state"]
        common_ids = start["neighbors"].keys() & current["neighbors"].keys()
        if common_ids and np.linalg.norm(displacement) >= .002:
            anchor_flow = np.median([current["neighbors"][i]["position"]-start["neighbors"][i]["position"] for i in common_ids],axis=0)
            flow = current["tool"]-start["tool"]-anchor_flow
            predicted = self.cumulative_response.predict(displacement)
            if predicted is not None:
                self._response_error.append((float(np.sum((flow-predicted)**2)),float(np.sum(flow**2))))
            self.cumulative_response.add(displacement,flow)
            self._response_extent_m = max(self._response_extent_m,float(np.linalg.norm(displacement)))
            self._response_origin = current
            self._response_age = 0
        return self.summary()

    def _image_prediction(self, command, translation_response):
        if not self.cumulative_response_ready or translation_response is None:
            return None, None
        response = translation_response.predict(command, self.state)
        if not response.get("supported") or not response.get("responsive"):
            return None, response
        actual = np.asarray(response["mean_delta_m"],float)
        return self.cumulative_response.predict(actual), response

    def assess(self, delta_xyz, *, monitor=None, nominal_away=False, nominal_clear=False,
               nominal_regions=None, translation_response=None, rotation_rad=0.,
               orientation_hold=False, direct_observation=False):
        command = np.asarray(delta_xyz,float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command translation must be three finite metres")
        size = float(np.linalg.norm(command))
        if nominal_regions is not None:
            self._metric = {}
            for r in nominal_regions:
                try:
                    distance=float(r["core_distance_m"]);sigma=float(r["uncertainty_m"])
                    if np.isfinite(distance) and np.isfinite(sigma) and distance>=0 and sigma>=0:
                        self._metric[r["id"]] = (distance,sigma)
                except (TypeError,KeyError,ValueError):
                    continue
        self._last_nominal_clear = bool(nominal_clear)
        result = {"accepted":True,"cap_m":None,"need_refresh":False,
                  "reason":"projected_risk_no_near_conflict","diagnostics":{}}
        def done(reason=None, cap=None, uncertain=False):
            if reason:
                result.update(accepted=False,cap_m=0.,need_refresh=True,reason=reason)
                self.events += 1
            else:
                result["cap_m"] = cap
                if size>1e-12:
                    self._charge_uncertain_motion = uncertain
            self._v12_last_result={k:copy.deepcopy(v) for k,v in result.items() if k!="diagnostics"}
            result["diagnostics"]=self.summary()
            return result
        if size < 1e-12:
            return done()  # Residual physical motion remains charged if needed.
        if self.context_invalid:
            return done(self.context_invalid)
        if self.domain_invalid:
            return done(self.domain_invalid)
        rotation_limit = .020 if orientation_hold else .015
        if abs(rotation_rad)>rotation_limit+1e-9:
            return done("translation_requested_rotation_outside_tracking_domain")
        tool=self._tool(monitor) if monitor is not None else self.tool_track
        if tool is None or tool is not self.tool_track or not tool.reliable:
            return done("translation_tool_correspondence_lost")
        if np.any(self.tool_box[:2]<-.5) or np.any(self.tool_box[2:]>self.scale+.5):
            return done("translation_tool_envelope_outside_view")
        risks=[r for r in self._risks() if not r["intended_contact"]]
        flow,response=self._image_prediction(command,translation_response)
        near=[]
        for r in risks:
            if not self._near_metric(r["id"]):
                continue
            threshold=max(r["margin_px"], 2.+self._gap_noise(r))
            predicted_gap=r["gap_px"] if flow is None else _gap(self.tool_box+np.tile(flow,2),r["box_px"])
            if min(r["gap_px"],predicted_gap)<=threshold or r["id"] in self.worsening:
                near.append(r)
        if not near:
            if not nominal_clear:
                if not direct_observation:
                    return done("translation_metric_segment_clearance_unavailable")
                # Current-observation Direct remains a bounded image-based
                # contact correction; it inherits no stale metric path license.
                # Unsupported motion has only the explicit 2mm local prior.
                result["reason"] = "projected_direct_current_observation_bounded"
                cap = self.uncertain_command_limit_m if flow is not None else .002
                return done(cap=min(size, cap))
            return done()
        if any(not r["track_reliable"] for r in near):
            return done("translation_nearby_obstacle_track_unavailable")
        if not nominal_clear:
            return done("translation_metric_segment_clearance_unavailable")
        supported_away=bool(flow is not None and all(
            _gap(self.tool_box+np.tile(flow,2),r["box_px"])>r["gap_px"]+1e-6 for r in near)
            and all(r["id"] in self.improving for r in near))
        if any(r["id"] in self.worsening for r in near) and not supported_away:
            return done("translation_observed_neighbor_approach")
        if supported_away:
            result["reason"]="projected_risk_observed_separation_supported"
            return done(cap=min(size,self.uncertain_command_limit_m))
        remaining=max(0.,self.observation_limit_m-self.observation_travel_m)
        if remaining<=1e-9:
            return done("translation_observation_budget_exhausted")
        requested=min(size,self.uncertain_command_limit_m)
        if translation_response is not None:
            limit=translation_response.command_cap_for_measured_limit(command/size,remaining,requested,self.state)
            cap=min(requested,float(limit["command_cap_m"]))
        else:
            # Explicit unit-gain assumption for the finite cold trajectory;
            # actual overshoot remains visible and stops the next admission.
            cap=min(requested,remaining)
        if cap<self.minimum_command_m-1e-9:
            return done("translation_measured_allowance_below_useful_command_floor")
        result["reason"]="projected_risk_bounded_nominal_clear_observation"
        return done(cap=cap,uncertain=True)

    def summary(self):
        result=super().summary()
        if not hasattr(self,"observation_travel_m"):
            return result
        error=sum(r[0] for r in self._response_error);zero=sum(r[1] for r in self._response_error)
        states=[r["state"] for r in self.risk_windows.values()]
        riskstate="worsening" if self.worsening else "improving" if self.improving else "inconclusive" if states else "unobserved"
        result.update(version=self.version,last_admission=copy.deepcopy(self._v12_last_result),
            observation_travel_m=self.observation_travel_m,
            observation_travel_remaining_m=max(0.,self.observation_limit_m-self.observation_travel_m),
            observation_travel_limit_m=self.observation_limit_m,
            envelope_revision=copy.deepcopy(self.envelope_revision),
            risk_progress={"state":riskstate,"worsening_ids":sorted(self.worsening),
                "improving_ids":sorted(self.improving),"per_neighbor":copy.deepcopy(self.risk_windows)},
            cumulative_response={"fit":self.cumulative_response.summary(),
                "validated":self.cumulative_response_ready,"held_out_prediction_sse":error,
                "held_out_zero_motion_sse":zero,"held_out_ratio":None if zero<=0 else error/zero,
                "supported_spatial_extent_m":self._response_extent_m},
            nominal_regions=[{"id":identity,"core_distance_m":d,"uncertainty_m":s,
                "model_remote":d>3*s,"clearance_certified":False} for identity,(d,s) in self._metric.items()],
            nominal_segment_clear_reported=self._last_nominal_clear,
            budget_semantics="Persistent measured trajectory while near-risk observation is uncertain, including residual holds; configure/rename/reseed cannot replenish.",
            limitations=["2D overlap is ambiguity, not occupied3Dspace",
                "remote filtering uses revisable modelgeometry, not observed clearance",
                "current envelope revision verifies tracked support, not every physical toolpart",
                "finite unvalidated observation and actuator intervals remain conditional assumptions"])
        return result


ProjectedTranslationGuardV12 = ObservedRiskTranslationGuardV12
