"""Passive, observation-bound moving-texture hypotheses for reference repair.

Raw changes and robot-correlated image motion do not identify a distal tool.
The model must inspect and select a candidate. This module neither executes
motion nor changes any motion, scene, attachment, or recovery allowance.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .observation import decode_image
from .visual_field import _gray
from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import LocalFeatureBundle


@dataclass
class _Frame:
    sequence: int
    rgb: np.ndarray
    state: np.ndarray
    joints: list | None
    anchor: np.ndarray | None
    anchor_reliable: bool
    path: float


class CausalReferenceCandidatesV18:
    version = "passive-causal-reference-candidates-v18"
    minimum_motion_m = .004
    maximum_frames = 32
    keyframe_spacing_m = .0005
    maximum_candidates = 6
    maximum_candidate_trials = 18
    change_threshold = 12

    def __init__(self):
        self._frames = deque(maxlen=self.maximum_frames-1)
        self.current = None
        self.path_m = 0.
        self.sequence = 0
        self.reference_generation = 0
        self._reference_track = None
        self._reference_origin_path = 0.
        self._reference_origin_state = None
        self._support_mask = self._support_pixels = None
        self._support_max_change = self._support_max_fraction = 0.
        self._declared_roi = None
        self._moving_kind = None
        self._anchor_identity = None
        self._candidates = []
        self._board_generation = 0
        self._review_id = None
        self._attempt_path = -1.
        self.status = "unconfigured"
        self._direction_segments = []
        self._image_cache = None
        self._board_prior = None
        self._build_count = 0

    @staticmethod
    def _data(observation):
        if isinstance(observation, dict):
            image = observation["observation/image"]
            state = observation["observation/state"]
            joints = observation.get("observation/joint_position")
            if joints is None:
                joints = observation.get("observation/arm_joint_position")
        else:
            image, state = observation.image, observation.state
            joints = getattr(observation, "joint_position", None)
        rgb = np.asarray(decode_image(image), dtype=np.uint8)
        state = np.asarray(state, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Candidate evidence requires finite measured state8")
        if joints is not None:
            joints = np.asarray(joints, float)
            joints = joints.tolist() if joints.shape == (7,) and np.isfinite(joints).all() else None
        return rgb, state, joints

    @staticmethod
    def _anchor(monitor):
        track = getattr(monitor, "tracks", {}).get("target")
        point = np.asarray(getattr(track, "position", []), float)
        reliable = bool(track is not None and getattr(track, "reliable", False)
                        and point.shape == (2,) and np.isfinite(point).all())
        return (point.copy() if reliable else None), reliable

    @staticmethod
    def _rgb_hash(rgb):
        return hashlib.sha256(rgb.tobytes()).hexdigest()

    @classmethod
    def _binding(cls, rgb, state):
        return hashlib.sha256(rgb.tobytes()+np.asarray(state, "<f8").tobytes()).hexdigest()

    def _history(self):
        frames = list(self._frames)
        if self.current is not None and (not frames or frames[-1].sequence != self.current.sequence):
            frames.append(self.current)
        return frames

    def _matches(self, observation):
        if self.current is None:
            return False
        rgb, state, _ = self._data(observation)
        return (np.array_equal(rgb, self.current.rgb)
                and np.array_equal(state, self.current.state))

    def configure(self, observation, monitor, motion_context):
        rgb, state, joints = self._data(observation)
        anchor, reliable = self._anchor(monitor)
        ext = (getattr(monitor, "refs", {}) or {}).get("external") or {}
        anchor_identity = ext.get("target_id")
        if self.current is None:
            self.current = _Frame(self.sequence, rgb.copy(), state.copy(), joints, anchor, reliable, self.path_m)
            self._frames.append(self.current)
        elif (not self._matches(observation) or anchor_identity != self._anchor_identity
              or reliable and self.current.anchor_reliable
              and np.linalg.norm(anchor-self.current.anchor) > .5):
            # No synthetic transitions across missing observations/anchor roles.
            self._frames.clear()
            self._candidates.clear()
            self.current = _Frame(self.sequence, rgb.copy(), state.copy(), joints, anchor, reliable, self.path_m)
            self._frames.append(self.current)
        else:
            # Model seed selection may refine the anchor at the same frame.
            self.current.anchor, self.current.anchor_reliable = anchor, reliable
        self._anchor_identity = anchor_identity
        self._moving_kind = ext.get("moving_kind")
        roi = np.asarray((motion_context or {}).get("tool_external_roi_uv"), float)
        self._declared_roi = (roi.copy() if roi.shape == (4,) and np.isfinite(roi).all()
                              and np.all(roi[:2] < roi[2:]) and np.all(roi >= 0) and np.all(roi <= 1) else None)
        track = getattr(monitor, "tracks", {}).get("moving")
        if track is not self._reference_track:
            # Root resolves/caches the selected history BEFORE configuring its
            # new monitor. The unused proposal bank then stops tracking; raw
            # acknowledged frames remain bounded and available for a later
            # evidence episode. Retain of the same track leaves IDs untouched.
            self._candidates.clear()
            self._board_prior = None
            self._reference_track = track
            self.reference_generation += 1
            self._reference_origin_path = self.path_m
            self._reference_origin_state = state.copy()
            self._review_id = None
            self._support_mask = self._support_pixels = None
            self._support_max_change = self._support_max_fraction = 0.
            self._direction_segments = []
            points = np.asarray(getattr(track, "points", []), float)
            if points.ndim == 2 and points.shape[1:] == (2,) and len(points) >= 3 and np.isfinite(points).all():
                mask = np.zeros(rgb.shape[:2], bool)
                for x, y in np.rint(points).astype(int):
                    if 3 <= x < rgb.shape[1]-3 and 3 <= y < rgb.shape[0]-3:
                        mask[y-3:y+4, x-3:x+4] = True
                if mask.any():
                    self._support_mask, self._support_pixels = mask, rgb[mask].copy()
        self.status = "observing"
        self._image_cache = None
        return self.summary()

    def _sample(self, frame, track):
        point = track.position.tolist()
        anchor = None if frame.anchor is None else frame.anchor.tolist()
        return {"sequence": frame.sequence, "state8": frame.state.tolist(),
                "joint_position": copy.deepcopy(frame.joints), "candidate_point_px": point,
                "candidate_reliable": bool(track.reliable), "anchor_point_px": anchor,
                "anchor_reliable": frame.anchor_reliable, "guard_anchor_point_px": anchor,
                "guard_anchor_reliable": frame.anchor_reliable,
                "noise_px": max(.08,float(track.last_residual_px)/np.sqrt(max(1,track.last_inliers))),
                "noise_semantics":"consensus_point_precision_proxy_not_per_feature_scatter",
                "feature_residual_px":float(track.last_residual_px),"feature_inliers":track.last_inliers,
                "raw_rgb_sha256": self._rgb_hash(frame.rgb)}

    def _direction_observation(self, delta):
        length = float(np.linalg.norm(delta))
        if length < .00005:
            return
        direction = delta/length
        if (self._direction_segments and
                np.dot(self._direction_segments[-1]["direction"], direction) >= np.cos(np.pi/4)):
            row = self._direction_segments[-1]
            row["delta"] += delta
            row["path_m"] += length
            row["direction"] = row["delta"]/np.linalg.norm(row["delta"])
        else:
            self._direction_segments.append({"direction": direction, "delta": delta.copy(), "path_m": length})
            self._direction_segments = self._direction_segments[-8:]

    def observe(self, before, after, monitor):
        if not self._matches(before):
            self.status = "unacknowledged_transition_gap"
            self._candidates.clear()
            return self.summary()
        rgb, state, joints = self._data(after)
        previous_rgb = self.current.rgb
        if rgb.shape != self.current.rgb.shape:
            self.status = "image_shape_changed"
            self._candidates.clear()
            return self.summary()
        delta = state[:3]-self.current.state[:3]
        self.path_m += float(np.linalg.norm(delta))
        self.sequence += 1
        anchor, reliable = self._anchor(monitor)
        self.current = _Frame(self.sequence, rgb.copy(), state.copy(), joints, anchor, reliable, self.path_m)
        self._direction_observation(delta)
        if self._support_mask is not None:
            difference = np.abs(rgb[self._support_mask].astype(np.int16)-self._support_pixels.astype(np.int16))
            self._support_max_change = max(self._support_max_change, float(difference.mean()))
            self._support_max_fraction = max(self._support_max_fraction, float((difference.max(1) > 6).mean()))
        domain_ok = (self._reference_origin_state is not None
                     and _orientation_distance(state[3:6], self._reference_origin_state[3:6]) <= .035
                     and np.max(abs(state[6:8]-self._reference_origin_state[6:8])) <= .002)
        pixels = _gray(rgb)
        if not domain_ok or not reliable:
            self._candidates.clear()
            self.status = "orientation_or_finger_domain_changed" if not domain_ok else "stationary_anchor_unavailable"
        else:
            alive = []
            for candidate in self._candidates:
                track = candidate["track"]
                # Identical rasters do not require another subpixel matcher.
                if (np.array_equal(rgb,previous_rgb) or track.track(pixels)) and self._candidate_inside(candidate):
                    sample = self._sample(self.current, track)
                    if self.path_m-candidate["last_key_path"] >= self.keyframe_spacing_m:
                        candidate["samples"].append(sample)
                        candidate["last_key_path"] = self.path_m
                    candidate["current_sample"] = sample
                    alive.append(candidate)
            self._candidates = alive
            self.status = "observing"
        if not self._frames or self.path_m-self._frames[-1].path >= self.keyframe_spacing_m:
            self._frames.append(self.current)
        used = self.path_m-self._reference_origin_path
        static = self._support_mask is not None and self._support_max_change <= 1.5 and self._support_max_fraction <= .05
        if (domain_ok and self._moving_kind == "tool" and static and used >= self.minimum_motion_m
                and not self._candidates and self.path_m-self._attempt_path >= .001):
            self._attempt_path = self.path_m
            self._build_candidates()
        if domain_ok and static and used >= self.minimum_motion_m and self._candidates and self._review_id is None:
            self._review_id = f"reference-{self.reference_generation}-review-{self.sequence}"
        self._image_cache = None
        return self.summary()

    @staticmethod
    def _corner_scores(gray):
        gy, gx = np.gradient(gray)
        def summed(a):
            return np.lib.stride_tricks.sliding_window_view(np.pad(a, 1), (3, 3)).sum((-1, -2))
        xx, yy, xy = summed(gx*gx), summed(gy*gy), summed(gx*gy)
        return .5*(xx+yy-np.sqrt(np.maximum((xx-yy)**2+4*xy*xy, 0)))

    def _build_candidates(self):
        self._build_count += 1
        frames = self._history()
        if len(frames) < 4 or any(not f.anchor_reliable for f in frames):
            return
        old, current = frames[0], frames[-1]
        if (_orientation_distance(old.state[3:6], current.state[3:6]) > .035
                or np.max(abs(old.state[6:8]-current.state[6:8])) > .002):
            return
        grays = [_gray(f.rgb) for f in frames]
        changed = np.abs(current.rgb.astype(np.int16)-old.rgb.astype(np.int16)).max(2) > self.change_threshold
        # A finite local neighborhood around actual changed raster pixels.
        nearby = np.lib.stride_tricks.sliding_window_view(np.pad(changed, 3), (7, 7)).sum((-1, -2)) >= 3
        scores = self._corner_scores(grays[0])
        h, w = changed.shape
        ys, xs = np.indices(changed.shape)
        valid = nearby & (xs >= 5) & (ys >= 5) & (xs < w-6) & (ys < h-6)
        scores = np.where(valid, scores, 0.)
        if self._declared_roi is not None:
            roi = self._declared_roi*np.array([w-1, h-1, w-1, h-1])
            inside = (xs >= roi[0]) & (ys >= roi[1]) & (xs <= roi[2]) & (ys <= roi[3])
        else:
            inside = np.zeros(changed.shape, bool)
        # Reserve hypotheses both inside and outside a possibly incorrect
        # declared tool rectangle. This does not classify any pixel as robot.
        choices = []
        for preferred in (inside, ~inside):
            local = np.where(preferred, scores, 0.).copy()
            for _ in range(self.maximum_candidate_trials//2):
                y, x = np.unravel_index(np.argmax(local), local.shape)
                if local[y, x] < 70.:
                    break
                choices.append((x, y))
                local[(xs-x)**2+(ys-y)**2 < 12.**2] = 0.
        found = []
        for x, y in choices:
            roi_px = np.array([max(0, x-10), max(0, y-10), min(w-1, x+10), min(h-1, y+10)], float)
            roi = roi_px/np.array([w-1,h-1,w-1,h-1])
            try:
                track = LocalFeatureBundle(grays[0], [x/(w-1), y/(h-1)], "candidate_hypothesis",
                    support_radius_px=10, maximum_features=8, roi_uv=roi)
            except ValueError:
                continue
            samples = [self._sample(old, track)]
            for frame, pixels in zip(frames[1:], grays[1:]):
                if not track.track(pixels):
                    break
                samples.append(self._sample(frame, track))
            if not track.reliable or len(samples) != len(frames):
                continue
            delta = track.position-track.seed
            anchor_delta = current.anchor-old.anchor
            relative = delta-anchor_delta
            magnitude = float(np.linalg.norm(relative))
            if magnitude < .5 or np.linalg.norm(delta) < .5:
                continue
            # Coherent global/camera translation is not controlled-body evidence.
            if np.linalg.norm(anchor_delta) > .5 and magnitude < max(.75, .35*np.linalg.norm(anchor_delta)):
                continue
            corners = np.array([roi_px[:2], [roi_px[2],roi_px[1]],roi_px[2:],[roi_px[0],roi_px[3]]])
            transformed = corners@track.matrix+track.translation
            current_roi = np.r_[transformed.min(0),transformed.max(0)]
            if np.any(current_roi[:2] < 0) or np.any(current_roi[2:] > [w-1,h-1]):
                continue
            source_mask = np.zeros((h,w),bool)
            for xx,yy in np.rint(track.initial_points).astype(int):
                source_mask[yy-3:yy+4,xx-3:xx+4] = True
            if not (changed & source_mask).any():
                continue
            quality = magnitude/(.25+float(np.median([s["noise_px"] for s in samples])))
            found.append({"track":track,"source_corners":corners,"samples":deque(samples,maxlen=self.maximum_frames),
                "current_sample":samples[-1],"last_key_path":self.path_m,"initial_raw_changed_pixels":int((changed&source_mask).sum()),
                "quality":quality,"initial_relative_displacement_px":relative.tolist()})
        found.sort(key=lambda c:c["quality"],reverse=True)
        chosen = []
        for candidate in found:
            if all(np.linalg.norm(candidate["track"].position-c["track"].position) >= 10 for c in chosen):
                chosen.append(candidate)
            if len(chosen) == self.maximum_candidates:
                break
        if chosen:
            self._board_generation += 1
            self._board_prior = old
            for index,candidate in enumerate(chosen):
                candidate["id"] = f"motion-{self._board_generation}-{index+1}"
            self._candidates = chosen

    @staticmethod
    def _candidate_inside(candidate):
        track=candidate['track']
        corners=candidate['source_corners']@track.matrix+track.translation
        return bool(track.reliable and np.isfinite(corners).all()
                    and np.all(corners>=0) and np.all(corners<=track.scale))

    def _candidate_summary(self, candidate):
        track = candidate["track"]
        corners = candidate["source_corners"]@track.matrix+track.translation
        roi = np.r_[corners.min(0),corners.max(0)]/np.tile(track.scale,2)
        samples = list(candidate["samples"])
        if samples[-1]["sequence"] != self.current.sequence:
            samples.append(candidate["current_sample"])
        first, last = samples[0], samples[-1]
        relative = (np.asarray(last["candidate_point_px"])-last["anchor_point_px"]
                    -np.asarray(first["candidate_point_px"])+first["anchor_point_px"])
        return {"candidate_id":candidate["id"],"current_point_uv":track.uv.tolist(),
                "current_roi_uv":roi.tolist(),"prior_point_uv":(np.asarray(first["candidate_point_px"])/track.scale).tolist(),
                "first_observation_step":first["sequence"],"last_observation_step":last["sequence"],
                "tracked_transition_count":len(samples)-1,"inliers":track.last_inliers,
                "residual_px":float(track.last_residual_px),"relative_displacement_px":relative.tolist(),
                "initial_changed_descriptor_pixels":candidate["initial_raw_changed_pixels"],
                "semantic_identity_certified":False,"distal_tool_identity_confirmed":False,
                "evidence_kind":"retrospective_candidate_hypothesis"}

    def summary(self):
        candidates = [self._candidate_summary(c) for c in self._candidates if self._candidate_inside(c)] if self.current is not None else []
        segments = [{"direction_xyz":r["direction"].tolist(),"net_delta_xyz_m":r["delta"].tolist(),
                     "measured_path_m":r["path_m"]} for r in self._direction_segments]
        informative = [r for r in self._direction_segments if np.linalg.norm(r["delta"]) >= self.minimum_motion_m]
        distinct = any(abs(float(a["direction"]@b["direction"])) <= np.cos(np.pi/4)
                       for i,a in enumerate(informative) for b in informative[i+1:])
        active = self._review_id is not None and bool(candidates) and self.status == "observing"
        return {"version":self.version,"status":self.status,"review_required":active,
            "review_recommended":active,"review_id":self._review_id,"board_generation":self._board_generation,
            "reason":"uninformative_reference_after_distinct_motion" if active and distinct else
                     "static_reference_with_observed_moving_alternatives" if active else "collecting_passive_evidence",
            "observation_binding":None if self.current is None else self._binding(self.current.rgb,self.current.state),
            "current_sequence":self.sequence,"measured_path_m":self.path_m,
            "measured_path_since_reference_m":self.path_m-self._reference_origin_path,
            "direction_segments":segments,"two_distinct_informative_directions":distinct,
            "support_max_mean_rgb_change":self._support_max_change,
            "support_max_changed_fraction":self._support_max_fraction,
            "candidates":candidates,"maximum_candidates":self.maximum_candidates,
            "maximum_history_frames":self.maximum_frames,"retained_history_frames":len(self._history()),
            "candidate_extraction_count":self._build_count,
            "selection_requires":"Inspect current RGB and explicitly choose a visible rigid distal tool feature; candidates may include proximal links, shadows or unrelated objects.",
            "interpretation":"Static support is uninformative for current control, not proof of wrong semantic identity. Candidate motion is retrospective and requires fresh predictive confirmation.",
            "automatic_reference_promotion":False,"control_debt_modified":False,
            "semantic_identity_certified":False,"clearance_certified":False,"environment_inputs":[]}

    def resolve_candidate(self, candidate_id, observation):
        if not self._matches(observation):
            raise ValueError("reference_candidate_observation_mismatch")
        candidate = next((c for c in self._candidates if c["id"] == candidate_id),None)
        if candidate is None or not self._candidate_inside(candidate) or self.status != "observing":
            raise ValueError("reference_candidate_unavailable")
        result = self._candidate_summary(candidate)
        samples = list(candidate["samples"])
        if samples[-1]["sequence"] != self.current.sequence:
            samples.append(candidate["current_sample"])
        # Samples are bounded including the current endpoint; no invented rows.
        samples = samples[-self.maximum_frames:]
        result.update(observation_binding=self._binding(self.current.rgb,self.current.state),
            evidence={"kind":"retrospective_candidate_conditioned","chronological_samples":copy.deepcopy(samples),
                "current_sequence":self.sequence,"image_shape":list(self.current.rgb.shape[:2]),
                "current_raw_rgb_sha256":self._rgb_hash(self.current.rgb),"current_state8":self.current.state.tolist(),
                "guard_anchor_source":"explicit_stationary_anchor","anchor_identity":self._anchor_identity,
                "selected_semantic_identity_certified":False,"validation_is_prospective":False,
                "fresh_held_out_confirmation_required":True,"control_debt_modified":False})
        result["evidence_sha256"] = hashlib.sha256(json.dumps(result["evidence"], sort_keys=True,
            separators=(",",":"), allow_nan=False).encode()).hexdigest()
        return copy.deepcopy(result)

    def image(self, observation=None):
        candidates=[c for c in self._candidates if self._candidate_inside(c)]
        if observation is None or not self._matches(observation) or not candidates or self.status!='observing':
            return None
        if self._image_cache is not None:
            return self._image_cache
        before = self._board_prior.rgb
        current = self.current.rgb
        h,w = current.shape[:2]
        board = Image.new("RGB",(w*4,h*2+108),(24,24,28))
        left,right=Image.fromarray(before).resize((w*2,h*2)),Image.fromarray(current).resize((w*2,h*2))
        board.paste(left,(0,32));board.paste(right,(w*2,32));draw=ImageDraw.Draw(board)
        draw.text((8,8),"RECORDED PRIOR RGB",fill="white");draw.text((w*2+8,8),"CURRENT RGB: choose only a visible distal TOOL",fill="white")
        for n,candidate in enumerate(candidates,1):
            summary=self._candidate_summary(candidate)
            for uv,offset in [(candidate["track"].seed/candidate["track"].scale,0),(summary["current_point_uv"],w*2)]:
                x,y=np.asarray(uv)*np.array([w-1,h-1])*2+[offset,32]
                draw.ellipse((x-6,y-6,x+6,y+6),outline=(255,220,30),width=2)
                draw.text((x+7,y-8),f"{n}",fill=(255,220,30))
        draw.text((8,h*2+38),"Numbers are motion hypotheses, NOT robot segmentation, identity, or clearance proof.",fill="white")
        draw.text((8,h*2+56),"Motion may be proximal links/shadows/background. Selection preserves all physical motion limits.",fill="white")
        draw.text((8,h*2+78),"IDs: "+"   ".join(f"{n} = {c['id']}" for n,c in enumerate(candidates,1)),fill=(255,220,30))
        stream=io.BytesIO();board.save(stream,format="PNG")
        self._image_cache="data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()
        return self._image_cache
