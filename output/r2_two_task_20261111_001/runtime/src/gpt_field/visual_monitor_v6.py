"""V6 observed RGB correspondence and windowed payload attachment evidence.

This monitor consumes only images and measured robot feedback. Co-motion is an
observable consistency check, never a semantic identity, grasp, or clearance
certificate. No evaluator object state is used here.
"""
from __future__ import annotations

from collections import deque
import copy
import numpy as np

from .visual_field import _gray
from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import RoleAwareVisualMonitorV5


class WindowedAttachmentEvidence:
    """Finite-window co-motion with immediate release/correspondence rejection.

    Motion is relative to an independently tracked stationary anchor. Evidence
    needs measurable motion of both payload and tool, not just small residuals.
    Uncertainty preserves no positive evidence; a previously observed pair may
    receive at most 5 mm of additional measured travel while it is uncertain.
    New model requests do not reset this counter.
    """
    window_size = 4
    uncertain_travel_limit_m = .005

    def __init__(self):
        self.window = deque(maxlen=self.window_size)
        self.state = "unobserved"
        self.reason = "awaiting_measured_co_motion"
        self.samples = 0
        self.had_observed_evidence = False
        self.uncertain_travel_m = 0.
        self.metrics = {}

    @property
    def observed(self):
        return self.state == "observed"

    def reset(self, state="unobserved", reason="new_correspondence"):
        self.window.clear()
        self.state, self.reason = state, reason
        self.samples = 0
        self.had_observed_evidence = False
        self.uncertain_travel_m = 0.
        self.metrics = {}

    def _uncertain(self, distance, reason):
        self.state, self.reason = "uncertain", reason
        self.uncertain_travel_m += distance
        if self.uncertain_travel_m >= self.uncertain_travel_limit_m:
            self.reason = "uncertain_attachment_travel_exhausted"

    def observe(self, payload_delta, tool_delta, anchor_delta, *, measured_distance_m,
                separation_px, noise_px=(.08, .08, .08), reliable=True,
                distinct_features=True, opening=False, finger_change_m=0.,
                rotation_rad=0.):
        distance = float(measured_distance_m)
        if opening:
            self.reset("released", "open_command")
            return
        if finger_change_m > .002:
            self.reset("unobserved", "finger_configuration_changed")
            return
        if not reliable or payload_delta is None or tool_delta is None or anchor_delta is None:
            self.reset("correspondence_lost", "payload_tool_or_anchor_correspondence_lost")
            return
        if not distinct_features or separation_px < 8:
            self.reset("contradicted", "payload_and_tool_features_not_independent")
            return
        if rotation_rad > .035:
            self.reset("unobserved", "orientation_domain_changed")
            return
        payload = np.asarray(payload_delta, float)
        tool = np.asarray(tool_delta, float)
        anchor = np.asarray(anchor_delta, float)
        noise = np.maximum(np.asarray(noise_px, float), .08)
        pair_noise = float(np.sqrt(2*(noise[0]**2+noise[1]**2)))
        relative_change = float(np.linalg.norm(payload-tool))
        if relative_change > max(2.5, 5*pair_noise):
            self.reset("contradicted", "abrupt_payload_tool_differential_motion")
            return
        self.window.append((payload-anchor, tool-anchor, distance, noise, anchor))
        moving = np.sum([r[0] for r in self.window], axis=0)
        controlled = np.sum([r[1] for r in self.window], axis=0)
        anchor_motion = np.sum([r[4] for r in self.window], axis=0)
        moving_norm, controlled_norm = float(np.linalg.norm(moving)), float(np.linalg.norm(controlled))
        residual = float(np.linalg.norm(moving-controlled))
        window_noise = np.max([r[3] for r in self.window], axis=0)
        motion_noise = float(np.sqrt(2*(max(window_noise[:2])**2+window_noise[2]**2)))
        pair_noise = float(np.sqrt(2*(window_noise[0]**2+window_noise[1]**2)))
        resolved = max(.75, 3*motion_noise)
        measured = float(sum(r[2] for r in self.window))
        cosine = float(np.dot(moving, controlled)/max(1e-12, moving_norm*controlled_norm))
        self.metrics = {"window_transitions": len(self.window), "window_measured_path_m": measured,
                        "payload_motion_relative_anchor_px": moving.tolist(),
                        "tool_motion_relative_anchor_px": controlled.tolist(),
                        "window_anchor_motion_px": float(np.linalg.norm(anchor_motion)),
                        "payload_motion_px": moving_norm, "tool_motion_px": controlled_norm,
                        "payload_tool_differential_px": residual, "motion_threshold_px": resolved,
                        "pair_noise_px": pair_noise, "motion_cosine": cosine}
        # A common image translation cannot certify payload/tool attachment.
        # If the declared stationary anchor moves substantially, semantics are
        # uncertain even when the pair happens to have a similar optical flow.
        if np.linalg.norm(anchor_motion) > max(1., 3*np.sqrt(2)*window_noise[2]):
            self._uncertain(distance, "stationary_anchor_motion_uncertain")
            return
        shorter, longer = min(moving_norm, controlled_norm), max(moving_norm, controlled_norm)
        if len(self.window) >= 3 and longer >= resolved:
            # Resolved one-sided motion, opposite directions or growing relative
            # displacement contradict the rigid attachment hypothesis.
            if (shorter < .25*longer and residual > max(.7, 3*pair_noise)
                    or cosine < .2 and shorter >= resolved
                    or residual > max(1.2, 4*pair_noise, .5*shorter)):
                self.reset("contradicted", "windowed_payload_tool_motion_disagrees")
                return
        supported = (len(self.window) >= 3 and measured >= .001
                     and shorter >= resolved and cosine >= .85
                     and residual <= max(.7, 3*pair_noise, .35*shorter))
        if supported:
            self.samples += int(distance >= .0001)
            self.state, self.reason = "observed", "windowed_measured_co_motion"
            self.had_observed_evidence = True
            self.uncertain_travel_m = 0.
        elif distance < .0001 and longer < resolved and self.observed:
            # Stationary, unchanged correspondence does not itself invalidate a
            # preceding measured lift. It also creates no new evidence.
            self.reason = "stationary_correspondence_after_observed_motion"
        elif self.had_observed_evidence:
            self._uncertain(distance, "insufficient_resolved_window_motion")
        else:
            self.state, self.reason = "unobserved", "awaiting_measured_co_motion"

    def control_limit(self):
        if self.observed:
            return None, None
        if self.state == "uncertain" and self.had_observed_evidence:
            remaining = self.uncertain_travel_limit_m-self.uncertain_travel_m
            if remaining > .0001:
                return min(.003, remaining), None
            return 0., "uncertain_attachment_travel_exhausted"
        return 0., "payload_attachment_"+self.state

    def summary(self):
        cap, reason = self.control_limit()
        return {"state": self.state, "reason": self.reason, "co_motion_observed": self.observed,
                "supporting_windows": self.samples, "had_observed_evidence": self.had_observed_evidence,
                "uncertain_measured_travel_m": self.uncertain_travel_m,
                "uncertain_travel_limit_m": self.uncertain_travel_limit_m,
                "uncertain_travel_remaining_m": max(0., self.uncertain_travel_limit_m-self.uncertain_travel_m),
                "transport_command_cap_m": cap, "transport_limit_reason": reason,
                "metrics": copy.deepcopy(self.metrics), "physical_attachment_certified": False}


class RoleAwareVisualMonitorV6(RoleAwareVisualMonitorV5):
    """V5 feature/response checks plus windowed attachment and explicit reuse."""
    def __init__(self):
        super().__init__()
        self.attachment = WindowedAttachmentEvidence()
        self.reference_update_status = "unconfigured"

    @staticmethod
    def _current_roi(track):
        if track.roi is None:
            return None
        a, b = track.roi
        corners = np.array([a, [b[0], a[1]], b, [a[0], b[1]]])
        moved = (corners@track.matrix+track.translation)/track.scale
        return np.r_[moved.min(axis=0), moved.max(axis=0)].tolist()

    def _reset_external_mapping(self, state):
        self.epoch += 1
        self.response = self._new_model()
        self.validation_count = self.bad_predictions = self.missing = 0
        self.path_m = self.pixel_path = self.maximum_feature_displacement_px = 0.
        self.last_prediction_error_px = None
        self.origin = state.copy()
        self._held_out_motion = []

    def configure(self, images, references, state8, *, intent_id=None):
        """Resolve V6 seed/retain requests against actual live correspondences.

        Retain never invokes a seed constructor, invents coordinates, changes
        identity or renews attachment uncertainty. A failed retain makes the
        affected observer unavailable until an explicit fresh seed is supplied.
        """
        state = np.asarray(state8, float)
        if intent_id is not None and intent_id != self.intent_id:
            self.intent_id = intent_id
            self.intent_path_m = self.unvalidated_path_m = 0.
        old = self.refs.get("external")
        old_wrist = self.refs.get("wrist")
        ext = references.get("external")
        wrist = references.get("wrist")
        pixels, wrist_pixels = _gray(images["external"]), _gray(images["wrist"])
        flat, flat_wrist = None, None
        preserved = False
        compatible = (self.state is not None
            and np.linalg.norm(state[:3]-self.state[:3]) <= 1e-6
            and _orientation_distance(state[3:6], self.state[3:6]) < .035
            and np.max(np.abs(state[6:8]-self.state[6:8])) < .002)
        self.reference_update_status = "external_disabled"
        if ext is not None:
            mode = ext["update"]
            if mode == "seed":
                flat = {key: ext[key] for key in ("moving_id", "moving_kind", "target_id")}
                flat.update(copy.deepcopy(ext["seed"]))
                flat["target_point_uv"] = (np.asarray(flat["target_anchor_uv"])+ext["goal_offset_uv"]).tolist()
                self.reference_update_status = "external_seeded"
            elif mode == "retain":
                keys = ("moving", "target", "tool") if ext["moving_kind"] == "payload" else ("moving", "target")
                preserved = (old is not None and compatible
                    and all(ext[key] == old[key] for key in ("moving_id", "moving_kind", "target_id"))
                    and all(self.tracks.get(key) is not None and self.tracks[key].reliable for key in keys))
                if preserved:
                    preserved = all(self.tracks[key].track(pixels) for key in keys)
                if preserved:
                    flat = {key: ext[key] for key in ("moving_id", "moving_kind", "target_id")}
                    for key, point, region in (("moving", "moving_point_uv", "moving_roi_uv"),
                            ("target", "target_anchor_uv", "target_roi_uv"), ("tool", "tool_point_uv", "tool_roi_uv")):
                        track = self.tracks.get(key)
                        flat[point] = track.uv.tolist() if track is not None and track.reliable else None
                        flat[region] = self._current_roi(track) if track is not None and track.reliable else None
                    flat["target_point_uv"] = (np.asarray(flat["target_anchor_uv"])+ext["goal_offset_uv"]).tolist()
                    self.reference_update_status = "external_retained"
                else:
                    self.reference_update_status = "external_retain_rejected"
            else:
                raise ValueError("external update must be seed or retain")
        old_tracks = self.tracks
        if not preserved:
            self._reset_external_mapping(state)
            self.tracks = {}
            attachment_status = "correspondence_lost" if ext is not None and ext["update"] == "retain" else "unobserved"
            self.attachment.reset(attachment_status, self.reference_update_status)
        if wrist is not None:
            if wrist["update"] == "seed":
                flat_wrist = {"target_id": wrist["target_id"], "desired_target_uv": wrist["desired_target_uv"],
                              **copy.deepcopy(wrist["seed"])}
            elif wrist["update"] == "retain":
                track = old_tracks.get("wrist_target")
                if (old_wrist is not None and wrist["target_id"] == old_wrist["target_id"]
                        and track is not None and track.reliable and track.track(wrist_pixels)):
                    flat_wrist = {"target_id": wrist["target_id"], "desired_target_uv": wrist["desired_target_uv"],
                                  "target_anchor_uv": track.uv.tolist(), "target_roi_uv": self._current_roi(track)}
                    self.tracks["wrist_target"] = track
                else:
                    self.tracks["wrist_target"] = None
            else:
                raise ValueError("wrist update must be seed or retain")
        resolved = {"external": flat, "wrist": flat_wrist}
        self._patch_references = resolved
        self._patch_external_image, self._patch_wrist_image = pixels, wrist_pixels
        if not preserved and flat is not None:
            for key, point, identity in (("moving", "moving_point_uv", flat["moving_id"]),
                                         ("target", "target_anchor_uv", flat["target_id"]),
                                         ("tool", "tool_point_uv", "independent_tool")):
                self.tracks[key] = self._patch(pixels, flat.get(point), identity)
        if wrist is None:
            self.tracks["wrist_target"] = None
        elif wrist["update"] == "seed":
            self.tracks["wrist_target"] = self._patch(wrist_pixels, flat_wrist["target_anchor_uv"], flat_wrist["target_id"])
        if not preserved:
            moving = self.tracks.get("moving")
            self.initial_feature = None if moving is None else moving.position.copy()
        old_offset = None if old is None else np.subtract(old["target_point_uv"], old["target_anchor_uv"])
        same_goal = (preserved and np.linalg.norm(np.asarray(ext["goal_offset_uv"])-old_offset) < 1e-6)
        self.refs, self.state = resolved, state.copy()
        self.last_error = self._error()
        if not same_goal:
            self.best_error, self.goal_progress_px = self.last_error, 0.
        self.status = "observing" if self._external_reliable() else "external_reference_unavailable"
        self._sync_attachment()
        return self.summary()

    @staticmethod
    def _noise(track):
        return max(.08, track.last_residual_px/np.sqrt(max(3, track.last_inliers)))

    @staticmethod
    def _distinct(payload, tool):
        if np.linalg.norm(payload.position-tool.position) < 8:
            return False
        if not len(payload.points) or not len(tool.points):
            return False
        distances = np.linalg.norm(payload.points[:, None]-tool.points[None, :], axis=-1)
        shared = max(float(np.mean(distances.min(axis=0) < 1.5)),
                     float(np.mean(distances.min(axis=1) < 1.5)))
        return shared < .5

    def _sync_attachment(self):
        self.attachment_evidence = self.attachment.observed
        self.attachment_samples = self.attachment.samples

    def observe(self, images, state8, action):
        state = np.asarray(state8, float)
        previous_state = self.state.copy()
        previous = {key: track.position.copy() for key, track in self.tracks.items()
                    if track is not None and track.reliable}
        super().observe(images, state8, action)
        ref = self.refs.get("external") or {}
        tracks = [self.tracks.get(key) for key in ("moving", "tool", "target")]
        reliable = (ref.get("moving_kind") == "payload" and
                    all(track is not None and track.reliable for track in tracks) and
                    all(key in previous for key in ("moving", "tool", "target")))
        if ref.get("moving_kind") != "payload":
            self.attachment.reset("unobserved", "no_payload_reference")
        else:
            deltas = ([track.position-previous[key] for track, key in
                       zip(tracks, ("moving", "tool", "target"))] if reliable else [None]*3)
            self.attachment.observe(*deltas,
                measured_distance_m=float(np.linalg.norm(state[:3]-previous_state[:3])),
                separation_px=float(np.linalg.norm(tracks[0].position-tracks[1].position)) if reliable else 0.,
                noise_px=[self._noise(t) for t in tracks] if reliable else [.08]*3,
                reliable=reliable, distinct_features=self._distinct(*tracks[:2]) if reliable else False,
                opening=action[6] < 0,
                finger_change_m=float(np.max(np.abs(state[6:8]-previous_state[6:8]))),
                rotation_rad=_orientation_distance(state[3:6], previous_state[3:6]))
        self._sync_attachment()
        return self.summary()

    def attachment_control_limit(self):
        return self.attachment.control_limit()

    def summary(self):
        result = super().summary()
        result["version"] = "role-aware-rgb-windowed-attachment-v6"
        result["reference_update_status"] = self.reference_update_status
        result["attachment"] = self.attachment.summary()
        for key, track in self.tracks.items():
            if track is None:
                continue
            detail = result["tracking"][key]
            detail["tracked_point_uv"] = track.uv.tolist()
            detail["seed_point_uv"] = (track.seed/track.scale).tolist()
            detail["current_roi_uv"] = None
            detail["current_roi_corners_uv"] = None
            if track.roi is not None:
                a, b = track.roi
                corners = np.array([a, [b[0], a[1]], b, [a[0], b[1]]])
                moved = (corners@track.matrix+track.translation)/track.scale
                detail["current_roi_corners_uv"] = moved.tolist()
                detail["current_roi_uv"] = np.r_[moved.min(axis=0), moved.max(axis=0)].tolist()
            detail["point_currently_observed"] = bool(track.reliable)
            detail["coordinate_semantics"] = "current persistent feature estimate" if track.reliable else "last reliable estimate; correspondence lost"
        return result
