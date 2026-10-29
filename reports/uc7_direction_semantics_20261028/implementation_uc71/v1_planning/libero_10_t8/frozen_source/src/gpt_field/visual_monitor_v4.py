"""Role-aware RGB evidence for a robot-space hypothesis field.

External RGB observes a controlled landmark. Wrist RGB observes target bearing;
it does not need to contain the entire gripper. Neither role supplies metric
obstacle geometry. Numerical response validation is not semantic identification.
"""
from __future__ import annotations

import copy
import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .visual_field import _Patch, _gray, SupportedLinearModel


def _orientation_distance(first_rotvec, second_rotvec):
    """Shortest SO(3) angle, independent of the rotation-vector branch at pi."""
    first = _rotvec_quaternion(np.asarray(first_rotvec, float))
    second = _rotvec_quaternion(np.asarray(second_rotvec, float))
    relative = _quaternion_product(first, np.r_[-second[:3], second[3]])
    # q and -q represent the same rotation. atan2 remains accurate for the tiny
    # motions that occur while the measured rotation vector crosses its branch.
    return float(2*np.arctan2(np.linalg.norm(relative[:3]), abs(relative[3])))


class RoleAwareVisualMonitor:
    def __init__(self):
        self.refs = {}
        self.tracks = {}
        self.state = None
        self.origin = None
        self.response = self._new_model()
        self.validation_count = 0
        self.bad_predictions = 0
        self.path_m = 0.
        self.intent_id = None
        self.intent_path_m = 0.
        self.unvalidated_path_m = 0.
        self.pixel_path = 0.
        self.maximum_feature_displacement_px = 0.
        self.missing = 0
        self.attachment_samples = 0
        self.attachment_evidence = False
        self.status = "unconfigured"
        self.epoch = 0
        self.last_prediction_error_px = None
        self.last_error = None
        self.best_error = None
        self.goal_progress_px = 0.
        self.correction_count = 0

    @staticmethod
    def _new_model():
        return SupportedLinearModel(residual_limit=1.5, min_samples=3, min_rank=1,
                                    minimum_excitation=.001, max_samples=24)

    @staticmethod
    def _patch(pixels, point, identity):
        if point is None:
            return None
        try:
            return _Patch(pixels, point, identity)
        except ValueError:
            return None

    def configure(self, images, references, state8, *, intent_id=None):
        """Bind references; only an explicit new local intent renews its budget.

        The caller supplies its model-request sequence as intent_id, rather than
        a model-authored identity. Omitted IDs conservatively retain the budget.
        Reference reseeding and response-model invalidation are different events
        and cannot authorize another 15 mm of unvalidated movement.
        """
        if intent_id is not None and intent_id != self.intent_id:
            self.intent_id = intent_id
            self.intent_path_m = self.unvalidated_path_m = 0.
        ref = references.get("external")
        old = self.refs.get("external")
        pixels = _gray(images["external"])
        # Attachment belongs to the payload/tool pair, independently of the
        # destination. A grasp-to-transport change of target must not erase an
        # observed test lift; changing the payload or tool correspondence must.
        attachment_keep = (ref is not None and old is not None
            and ref["moving_kind"] == old["moving_kind"] == "payload"
            and ref["moving_id"] == old["moving_id"] and self.state is not None
            and _orientation_distance(state8[3:6], self.state[3:6]) < .035
            and np.max(np.abs(np.asarray(state8[6:8])-self.state[6:8])) < .002)
        for key, seed in (("moving", "moving_point_uv"), ("tool", "tool_point_uv")):
            track = self.tracks.get(key)
            if (not attachment_keep or ref.get(seed) is None or track is None or not track.reliable
                    or np.linalg.norm((np.asarray(ref[seed])-track.uv)*track.scale) > 4
                    or not track.track(pixels)):
                attachment_keep = False
                break
        retained_attachment = self.attachment_evidence if attachment_keep else False
        retained_samples = self.attachment_samples if attachment_keep else 0
        preserve = (ref is not None and old is not None
                    and all(ref[k] == old[k] for k in ("moving_id", "target_id", "moving_kind"))
                    and self.state is not None
                    and _orientation_distance(state8[3:6], self.state[3:6]) < .035
                    and np.max(np.abs(np.asarray(state8[6:8])-self.state[6:8])) < .002)
        for key, seed in (("moving", "moving_point_uv"), ("target", "target_anchor_uv")):
            track = self.tracks.get(key)
            if (not preserve or track is None or not track.reliable
                    or np.linalg.norm((np.asarray(ref[seed])-track.uv)*track.scale) > 4
                    or not track.track(pixels)):
                preserve = False
                break
        if preserve:
            old_tool = self.tracks.get("tool")
            new_tool = ref.get("tool_point_uv")
            if ref["moving_kind"] == "payload" and (old_tool is None or new_tool is None
                    or not old_tool.reliable or np.linalg.norm((np.asarray(new_tool)-old_tool.uv)*old_tool.scale) > 4
                    or not old_tool.track(pixels)):
                preserve = False
        if not preserve:
            self.epoch += 1
            self.tracks = {}
            self.response = self._new_model()
            self.validation_count = self.bad_predictions = self.missing = 0
            self.path_m = self.pixel_path = self.maximum_feature_displacement_px = 0.
            self.attachment_samples = 0
            self.attachment_evidence = False
            self.last_prediction_error_px = None
            self.origin = np.asarray(state8, float).copy()
            if ref is not None:
                for key, seed, ident in (("moving", "moving_point_uv", ref["moving_id"]),
                                         ("target", "target_anchor_uv", ref["target_id"]),
                                         ("tool", "tool_point_uv", "independent_tool")):
                    self.tracks[key] = self._patch(pixels, ref.get(seed), ident)
            moving = self.tracks.get("moving")
            self.initial_feature = None if moving is None else moving.position.copy()
        self.attachment_evidence = retained_attachment
        self.attachment_samples = retained_samples
        # Wrist is an object-bearing observer, never a projected tool-volume gate.
        wrist = references.get("wrist")
        self.tracks["wrist_target"] = None if wrist is None else self._patch(
            _gray(images["wrist"]), wrist["target_anchor_uv"], wrist["target_id"])
        old_descriptor = None if old is None else np.subtract(old["target_point_uv"], old["target_anchor_uv"])
        new_descriptor = None if ref is None else np.subtract(ref["target_point_uv"], ref["target_anchor_uv"])
        same_goal = preserve and np.linalg.norm(new_descriptor-old_descriptor) < 1e-6
        self.refs = copy.deepcopy(references)
        self.state = np.asarray(state8, float).copy()
        self.last_error = self._error()
        if not same_goal:
            self.best_error = self.last_error
            self.goal_progress_px = 0.
        self.status = "observing" if self._external_reliable() else "external_reference_unavailable"
        return self.summary()

    def _external_reliable(self):
        return all(self.tracks.get(k) is not None and self.tracks[k].reliable for k in ("moving", "target"))

    def _error(self):
        ref = self.refs.get("external")
        if ref is None or not self._external_reliable():
            return None
        moving, target = self.tracks["moving"], self.tracks["target"]
        goal = target.position + np.subtract(ref["target_point_uv"], ref["target_anchor_uv"])*target.scale
        return float(np.linalg.norm(goal-moving.position))

    def observe(self, images, state8, action):
        state = np.asarray(state8, float)
        delta = state[:3]-self.state[:3]
        distance = float(np.linalg.norm(delta))
        rotation = _orientation_distance(state[3:6], self.state[3:6])
        grip_change = float(np.max(np.abs(state[6:8]-self.state[6:8])))
        response_was_ready = self.response_ready
        mapping_invalidated = rotation > .035 or grip_change > .002
        previous = {k: t.position.copy() for k, t in self.tracks.items() if t is not None and t.reliable}
        for name, track in self.tracks.items():
            if track is not None:
                track.track(_gray(images["wrist" if name == "wrist_target" else "external"]))
        self.path_m += distance
        self.intent_path_m += distance
        # Count the entire observed displacement if it started without a valid
        # response, or if a domain change invalidated that response during it.
        # This counter deliberately survives all evidence/mapping resets.
        if not response_was_ready or mapping_invalidated:
            self.unvalidated_path_m += distance
        self.missing = 0 if self._external_reliable() else self.missing+1
        self.status = "observing" if not self.missing else "external_track_lost"
        if mapping_invalidated:
            # Retain observed features, but invalidate a mapping across reorientation/contact.
            self.response = self._new_model()
            self.validation_count = self.bad_predictions = 0
            self.attachment_samples = 0
            self.attachment_evidence = False
            self.origin = state.copy()
            self.path_m = self.pixel_path = self.maximum_feature_displacement_px = 0.
            moving = self.tracks.get("moving")
            self.initial_feature = None if moving is None else moving.position.copy()
        if self._external_reliable() and all(k in previous for k in ("moving", "target")):
            moving_delta = self.tracks["moving"].position-previous["moving"]
            target_delta = self.tracks["target"].position-previous["target"]
            self.pixel_path += float(np.linalg.norm(moving_delta))
            if self.initial_feature is not None:
                self.maximum_feature_displacement_px = max(self.maximum_feature_displacement_px,
                    float(np.linalg.norm(self.tracks["moving"].position-self.initial_feature)))
            if rotation <= .035 and grip_change <= .002 and distance > .0001:
                predicted = self.response.predict(delta)
                if predicted is not None:
                    error = float(np.linalg.norm(predicted-(moving_delta-target_delta)))
                    self.last_prediction_error_px = error
                    self.bad_predictions = self.bad_predictions+1 if error > 3 else 0
                    if error <= 1.5:
                        self.validation_count += 1
                self.response.add(delta, moving_delta-target_delta)
            if self.path_m >= .006 and self.maximum_feature_displacement_px < self.feature_motion_threshold_px:
                self.status = "controlled_feature_unresponsive"
            elif self.bad_predictions >= 2:
                self.status = "visual_response_contradiction"
            elif np.linalg.norm(target_delta) > 6:
                self.status = "target_anchor_shifted"
            elif self.response_ready:
                self.status = "response_validated"
            ref = self.refs.get("external") or {}
            tool = self.tracks.get("tool")
            if ref.get("moving_kind") == "payload" and tool is not None and tool.reliable and "tool" in previous:
                separation = float(np.linalg.norm(tool.position-self.tracks["moving"].position))
                tool_delta = tool.position-previous["tool"]
                relative_change = float(np.linalg.norm(moving_delta-tool_delta))
                tool_norm, payload_norm = float(np.linalg.norm(tool_delta)), float(np.linalg.norm(moving_delta))
                cosine = float(np.dot(tool_delta, moving_delta)/max(1e-12, tool_norm*payload_norm))
                moving_pair = tool_norm >= .5 and payload_norm >= .5
                contradiction = (moving_pair and (cosine < .8
                    or relative_change > max(.5, .4*min(tool_norm, payload_norm))))
                if relative_change > 3 or separation < 8 or contradiction:
                    self.attachment_samples = 0
                    self.attachment_evidence = False
                elif (distance >= .0005 and moving_pair and grip_change <= .002):
                    self.attachment_samples += 1
                    self.attachment_evidence = self.attachment_samples >= 2
            elif ref.get("moving_kind") == "payload":
                self.attachment_samples = 0
                self.attachment_evidence = False
        else:
            self.attachment_samples = 0
            self.attachment_evidence = False
        if action[6] < 0:
            self.attachment_samples = 0
            self.attachment_evidence = False
        self.last_error = self._error()
        if self.last_error is not None:
            if self.best_error is None:
                self.best_error = self.last_error
            elif self.last_error < self.best_error-1.5:
                self.goal_progress_px += self.best_error-self.last_error
                self.best_error = self.last_error
        self.state = state.copy()
        return self.summary()

    @property
    def feature_motion_threshold_px(self):
        return 1.

    @property
    def response_ready(self):
        return bool(self.response.matrix is not None and self.validation_count >= 2
                and self.maximum_feature_displacement_px >= self.feature_motion_threshold_px
                and np.linalg.norm(self.response.matrix) >= 20.)

    def control_limit(self):
        """Finite bootstrap under the model intent, then measured feature validation.

        Absence of a wrist tool is not a veto. An unavailable external observer
        is never silently ignored; uncertain motion cannot exceed 15 mm.
        """
        if self.status in ("controlled_feature_unresponsive", "visual_response_contradiction", "target_anchor_shifted"):
            return 0., self.status
        if self.missing >= 2 or not self._external_reliable():
            return 0., "external_reference_unavailable"
        if not self.response_ready:
            remaining = .015-self.unvalidated_path_m
            if remaining <= .001:
                return 0., "visual_response_budget_exhausted"
            return min(.005, remaining), None
        return None, None

    def planar_correction(self, goal_xyz):
        """A small optional correction; caller still replans against all 3D boxes."""
        if (not self.response_ready or not self._external_reliable() or self.response.rank < 2
                or self.status != "response_validated" or self.origin is None
                or abs(self.state[2]-self.origin[2]) > .003
                or _orientation_distance(self.state[3:6], self.origin[3:6]) > .035
                or np.linalg.norm(np.asarray(goal_xyz)-self.state[:3]) > .025):
            return None
        matrix = self.response.matrix[:2]
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] < 20 or singular[0]/singular[-1] > 15:
            return None
        ref = self.refs["external"]
        moving, target = self.tracks["moving"], self.tracks["target"]
        error = target.position + np.subtract(ref["target_point_uv"], ref["target_anchor_uv"])*target.scale-moving.position
        if np.linalg.norm(error) <= 1.5:
            return None
        correction = np.linalg.solve(matrix.T, error)
        # Local planar goal estimate is expressed from actual EEF, not cumulatively added.
        proposed = np.asarray(goal_xyz, float).copy()
        desired_xy = self.state[:2]+correction
        shift = desired_xy-proposed[:2]
        norm = float(np.linalg.norm(shift))
        if norm > .003:
            shift *= .003/norm
        proposed[:2] += shift
        return proposed

    def summary(self):
        wrist = self.tracks.get("wrist_target")
        wrist_ref = self.refs.get("wrist")
        bearing_error = None
        if wrist is not None and wrist.reliable and wrist_ref:
            bearing_error = float(np.linalg.norm(wrist.position-np.asarray(wrist_ref["desired_target_uv"])*wrist.scale))
        return {"version": "role-aware-rgb-evidence-v4", "external_status": self.status,
            "external_reliable": self._external_reliable(), "response_validated": self.response_ready,
            "response": self.response.summary(), "held_out_validations": self.validation_count,
            "prediction_error_px": self.last_prediction_error_px, "observed_path_m": self.path_m,
            "intent_id": self.intent_id, "intent_observed_path_m": self.intent_path_m,
            "intent_unvalidated_path_m": self.unvalidated_path_m,
            "unvalidated_budget_remaining_m": max(0., .015-self.unvalidated_path_m),
            "controlled_feature_displacement_px": self.maximum_feature_displacement_px,
            "external_goal_error_px": self.last_error, "same_goal_image_improvement_px": self.goal_progress_px,
            "reference_epoch": self.epoch, "wrist_role": "target_bearing_observer",
            "wrist_target_reliable": bool(wrist is not None and wrist.reliable),
            "wrist_bearing_error_px": bearing_error, "attachment_co_motion_observed": self.attachment_evidence,
            "attachment_samples": self.attachment_samples,
            "clearance_certified": False, "task_success_certified": False,
            "semantic_correspondence_certified": False}
