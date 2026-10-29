"""Local two-view contact-image goals for Field retargeting (opt-in V13).

The model marks a physical target contact location and the open-pad midpoint.
Those semantic points are transported by existing bounded RGB correspondences;
no tiny pad detector, camera calibration, scene pose or depth is introduced.
A measured local image response estimates a contact goal in its supported
translation subspace. This module proposes goals only: no actions, closure,
attachment, clearance, or task completion are authorized by image alignment.
"""
from __future__ import annotations

import copy
from collections import deque
import numpy as np

from .visual_field import SupportedLinearModel
from .visual_monitor_v4 import _orientation_distance


class VisualContactAlignmentV13:
    version = "two-view-semantic-contact-field-v13"
    maximum_goal_change_m = .006
    maximum_local_estimate_m = .030
    aggregate_distance_m = .001
    aggregate_max_frames = 64
    orientation_domain_rad = .035
    finger_domain_m = .002

    def __init__(self):
        self.active = False
        self.status = "contact_alignment_not_supplied"
        self.descriptor = None
        self.target_id = None
        self.tracks = {}
        self.points = {}
        self.state = self.origin = None
        self.current = self.previous = None
        self.model = self._new_model()
        self.held_out = deque(maxlen=8)
        self.aggregate = None
        self.aggregate_frames = 0
        self.last_correction = None
        self.observed_updates = 0
        self.aligned_updates = 0

    @staticmethod
    def _new_model():
        return SupportedLinearModel(residual_limit=2.5, min_samples=3,
            min_rank=1, max_samples=24, minimum_excitation=.0008)

    @staticmethod
    def _state(value):
        v = np.asarray(value, float)
        if v.shape != (8,) or not np.isfinite(v).all():
            raise ValueError("contact state must be eight finite measured values")
        return v

    @staticmethod
    def _uv(value):
        v = np.asarray(value, float)
        if v.shape != (2,) or not np.isfinite(v).all() or np.any(v<0) or np.any(v>1):
            raise ValueError("contact point must be a normalized image coordinate")
        return v

    @staticmethod
    def _track_set(monitor):
        refs = getattr(monitor, "refs", {})
        ext, wrist = refs.get("external") or {}, refs.get("wrist") or {}
        tracks = getattr(monitor, "tracks", {})
        return {"tool": tracks.get("moving"), "external_target": tracks.get("target"),
                "wrist_target": tracks.get("wrist_target")}, ext, wrist

    def _invalidate(self, reason):
        self.active = False
        self.status = reason
        self.current = self.previous = self.aggregate = None
        self.aligned_updates = 0
        self.model = self._new_model()
        self.held_out.clear()
        return self.summary()

    @staticmethod
    def _bind(track, uv, bounds=None):
        pixel = VisualContactAlignmentV13._uv(uv)*track.scale
        source = np.linalg.solve(track.matrix.T, pixel-track.translation)
        if bounds is not None:
            box = np.asarray(bounds, float)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box<0)
                    or np.any(box>1) or np.any(box[:2]>=box[2:])):
                raise ValueError("contact point bounds must be an ordered image ROI")
            low, high = box.reshape(2,2)*track.scale
            inside = np.all(pixel>=low) and np.all(pixel<=high)
        elif track.roi is not None:
            inside = np.all(source>=track.roi[0]) and np.all(source<=track.roi[1])
        else:
            inside = np.linalg.norm(source-track.seed)<=track.support_radius_px
        if not inside:
            raise ValueError("contact_point_outside_declared_body_support")
        return source

    def configure(self, descriptor, monitor, state8, *, tool_roi_uv=None):
        state = self._state(state8)
        if descriptor is None:
            self.descriptor = None
            return self._invalidate("contact_alignment_not_supplied")
        tracks, ext, wrist = self._track_set(monitor)
        target = descriptor.get("target_id")
        if (not isinstance(target,str) or not target or ext.get("moving_kind")!="tool"
                or ext.get("target_id")!=target or wrist.get("target_id")!=target):
            return self._invalidate("contact_alignment_role_mismatch")
        if any(t is None or not t.reliable for t in tracks.values()):
            return self._invalidate("contact_alignment_correspondence_unavailable")
        mode = descriptor.get("update")
        if mode == "retain":
            if (descriptor.get("external") is not None or descriptor.get("wrist") is not None):
                raise ValueError("retained contact descriptors must not supply new coordinates")
            if (not self.active or target!=self.target_id
                    or any(tracks[k] is not self.tracks.get(k) for k in tracks)):
                return self._invalidate("contact_alignment_retain_identity_unavailable")
            domain = self._domain_reason(state)
            if domain:
                return self._invalidate(domain)
            self.state = state.copy()
            self.current = self._snapshot()
            if self.current is None:
                return self._invalidate("contact_alignment_point_outside_view")
            self.status = "contact_alignment_retained"
            return self.summary()
        if mode != "seed" or not isinstance(descriptor.get("external"),dict) or not isinstance(descriptor.get("wrist"),dict):
            raise ValueError("contact seed requires both view descriptors")
        self.tracks, self.target_id = tracks, target
        self.descriptor = copy.deepcopy(descriptor)
        try:
            self.points = {
                "tool": self._bind(tracks["tool"],descriptor["external"]["tool_pinch_point_uv"],tool_roi_uv),
                "external_target": self._bind(tracks["external_target"],descriptor["external"]["target_contact_point_uv"]),
                "wrist_target": self._bind(tracks["wrist_target"],descriptor["wrist"]["target_contact_point_uv"]),
                "wrist_pinch": self._uv(descriptor["wrist"]["tool_pinch_point_uv"])*tracks["wrist_target"].scale}
        except (ValueError,KeyError,np.linalg.LinAlgError):
            return self._invalidate("contact_alignment_point_outside_declared_support")
        self.active = True
        self.state = self.origin = state.copy()
        self.model = self._new_model()
        self.held_out.clear()
        self.current = self._snapshot()
        self.previous = self.aggregate = self.current
        self.aggregate_frames = self.observed_updates = self.aligned_updates = 0
        self.last_correction = None
        if self.current is None:
            return self._invalidate("contact_alignment_point_outside_view")
        self.status = "contact_alignment_seeded"
        return self.summary()

    def _domain_reason(self, state):
        if self.origin is None:
            return "contact_alignment_not_configured"
        if np.max(abs(state[6:]-self.origin[6:]))>self.finger_domain_m:
            return "contact_alignment_finger_configuration_changed"
        if _orientation_distance(state[3:6],self.origin[3:6])>self.orientation_domain_rad:
            return "contact_alignment_orientation_domain_changed"
        return None

    def _snapshot(self):
        if not self.active or any(t is None or not t.reliable for t in self.tracks.values()):
            return None
        xy, uncertainty = {}, {}
        for key,track in self.tracks.items():
            point = self.points[key]@track.matrix+track.translation
            if np.any(point<0) or np.any(point>track.scale):
                return None
            xy[key] = point
            span = max(8., float(np.linalg.norm(np.ptp(track.points,axis=0))))
            offset = float(np.linalg.norm(point-track.position))
            residual = max(.08,float(getattr(track,"last_residual_px",0.))/np.sqrt(max(1,track.last_inliers)))
            # Transporting a semantic pad midpoint outside a textured housing
            # is an uncertain rigid-part approximation, not a observed pad.
            uncertainty[key] = .25+residual*(1.+2.*offset/span)
        xy["wrist_pinch"] = self.points["wrist_pinch"].copy()
        uncertainty["wrist_pinch"] = .5  # Declared semantic coordinate, not a detector.
        error = np.r_[xy["external_target"]-xy["tool"],xy["wrist_target"]-xy["wrist_pinch"]]
        noise = np.r_[np.repeat(np.hypot(uncertainty["external_target"],uncertainty["tool"]),2),
                       np.repeat(np.hypot(uncertainty["wrist_target"],uncertainty["wrist_pinch"]),2)]
        return {"state":self.state[:3].copy(),"error":error,"noise":noise,
                "points":xy,"point_uncertainty_px":uncertainty}

    def observe(self, monitor, state8, action):
        state = self._state(state8)
        if not self.active:
            return self.summary()
        tracks, ext, wrist = self._track_set(monitor)
        if any(tracks.get(k) is not self.tracks[k] or not self.tracks[k].reliable for k in self.tracks):
            return self._invalidate("contact_alignment_correspondence_lost_or_changed")
        domain = self._domain_reason(state)
        if domain:
            return self._invalidate(domain)
        self.state = state.copy()
        current = self._snapshot()
        if current is None:
            return self._invalidate("contact_alignment_point_outside_view")
        self.current = current
        self.observed_updates += 1
        if self._within_tolerance():
            self.aligned_updates += 1
        else:
            self.aligned_updates = 0
        if self.aggregate is None or self.aggregate_frames>=self.aggregate_max_frames:
            self.aggregate = self.previous
            self.aggregate_frames = 0
        self.aggregate_frames += 1
        if self.aggregate is not None:
            delta = current["state"]-self.aggregate["state"]
            flow = current["error"]-self.aggregate["error"]
            if np.linalg.norm(delta)>=self.aggregate_distance_m:
                prediction = self.model.predict(delta)
                if prediction is not None:
                    error = float(np.linalg.norm(flow-prediction))
                    noise = float(np.linalg.norm(np.maximum(current["noise"],self.aggregate["noise"])))
                    self.held_out.append({"error2":error**2,"zero2":float(flow@flow),"noise":noise})
                self.model.add(delta,flow)
                self.aggregate = current
                self.aggregate_frames = 0
        self.previous = current
        self.status = "contact_alignment_response_supported" if self.response_ready else "contact_alignment_accumulating_useful_motion"
        return self.summary()

    @property
    def response_ready(self):
        if not self.active or self.model.matrix is None or len(self.held_out)<2:
            return False
        error=sum(r["error2"] for r in self.held_out)
        zero=sum(r["zero2"] for r in self.held_out)
        return bool(zero>.5 and error<.75*zero and np.linalg.norm(self.model.matrix)>20.)

    def _within_tolerance(self):
        if self.current is None:
            return False
        e,n=self.current["error"],self.current["noise"]
        return bool(all(np.linalg.norm(e[i:i+2])+np.linalg.norm(n[i:i+2])<=3.
                        for i in (0,2)))

    @property
    def ready_for_model_review(self):
        return bool(self.active and self.aligned_updates>=2 and self._within_tolerance())

    @staticmethod
    def _intersection(basis, allowed_axes):
        mask={"xyz":(0,1,2),"xy":(0,1),"z":(2,),"x":(0,),"y":(1,)}.get(allowed_axes)
        if mask is None:
            raise ValueError("unknown contact correction motion axes")
        observed=basis.T
        excluded=[i for i in range(3) if i not in mask]
        if not excluded:
            return observed
        _,s,vt=np.linalg.svd(observed[excluded],full_matrices=True)
        rank=int(np.sum(s>1e-7))
        return observed@vt[rank:].T

    def correction(self, goal_xyz, *, allowed_axes="xyz", max_correction_m=.006):
        goal=np.asarray(goal_xyz,float)
        if goal.shape!=(3,) or not np.isfinite(goal).all():
            raise ValueError("contact correction goal must be three finite metres")
        limit=float(max_correction_m)
        if not np.isfinite(limit) or limit<0:
            raise ValueError("contact correction limit must be finite and nonnegative")
        result={"accepted":False,"goal_xyz":None,"reason":self.status,"diagnostics":None}
        if self.ready_for_model_review:
            result["reason"]="contact_alignment_observed"
        elif self.response_ready:
            basis=self._intersection(self.model.basis,allowed_axes)
            if basis.shape[1]:
                jacobian=basis.T@self.model.matrix
                error=self.current["error"]
                weights=1./np.maximum(.75,self.current["noise"])
                a=jacobian*weights
                b=error*weights
                regularizer=max(1e-8,float(np.linalg.norm(a))**2*1e-5)
                estimate=-np.linalg.solve(a@a.T+regularizer*np.eye(len(a)),a@b)
                delta=basis@estimate
                length=float(np.linalg.norm(delta))
                delta*=min(1.,self.maximum_local_estimate_m/max(length,1e-12))
                # Estimate the contact location, do not repeatedly ADD current
                # image error to an already moving model goal. Unsupported goal
                # components remain unchanged.
                projector=basis@basis.T
                desired=self.state[:3]+delta+(np.eye(3)-projector)@(goal-self.state[:3])
                change=desired-goal
                size=float(np.linalg.norm(change))
                cap=min(limit,self.maximum_goal_change_m)
                change*=min(1.,cap/max(size,1e-12))
                if np.linalg.norm(change)>=1e-5:
                    result.update(accepted=True,goal_xyz=(goal+change).tolist(),reason="contact_alignment_goal_correction")
                    self.last_correction={"goal_before":goal.tolist(),"proposed_goal":result["goal_xyz"],
                        "goal_change_m":float(np.linalg.norm(change)),"supported_rank":basis.shape[1],
                        "estimated_contact_delta_m":delta.tolist(),"extrapolation_is_goal_hypothesis":True}
                else:
                    result["reason"]="contact_alignment_goal_already_consistent"
            else:
                result["reason"]="contact_alignment_axes_not_observed"
        result["diagnostics"]=self.summary()
        return result

    def summary(self):
        return {"version":self.version,"active":self.active,"status":self.status,"target_id":self.target_id,
                "image_error_px":None if self.current is None else self.current["error"].tolist(),
                "error_component_uncertainty_px":None if self.current is None else self.current["noise"].tolist(),
                "current_points_px":{} if self.current is None else {k:v.tolist() for k,v in self.current["points"].items()},
                "point_uncertainty_px":{} if self.current is None else self.current["point_uncertainty_px"],
                "response":self.model.summary(),"response_validated":self.response_ready,
                "held_out_samples":len(self.held_out),"held_out_error2":sum(r["error2"] for r in self.held_out),
                "held_out_zero_motion_error2":sum(r["zero2"] for r in self.held_out),
                "observed_updates":self.observed_updates,"aligned_observed_updates":self.aligned_updates,
                "ready_for_model_review":self.ready_for_model_review,"last_correction":copy.deepcopy(self.last_correction),
                "maximum_goal_change_m":self.maximum_goal_change_m,"clearance_certified":False,
                "grasp_certified":False,"task_success_certified":False,"automatic_close":False,
                "limitations":["Semantic contact and pinch locations are model hypotheses.",
                    "Rigid affine transport of a hand point approximates perspective and flex.",
                    "Online image response is local; goal estimates require Field geometry checks.",
                    "Two-view pixel agreement does not prove depth clearance, contact or attachment."]}
