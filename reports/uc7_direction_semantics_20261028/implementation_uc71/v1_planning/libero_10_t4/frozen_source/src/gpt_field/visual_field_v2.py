"""Opt-in local visual control with landmark-relative goals and measured response.

Only RGB, joint feedback, end-effector/finger state and acknowledged OSC actions
are consumed. Fields have weighted-pixel units, not metric obstacle clearance.
A local feature bank follows an explicitly seeded identity; it never searches
outside a bounded neighborhood to assign a replacement object automatically.
"""
from __future__ import annotations

import math
import numpy as np

from .visual_field import (ImageDistanceField, PANDA_JOINT_LOWER, PANDA_JOINT_UPPER,
                           VIEWS, _box, _clear_segment, _gray, _uv, _vector)

VISUAL_FIELD_VERSION = "landmark-relative-image-distance-measured-eef-v2"


class FeatureBankTrack:
    """Immutable local patches voting for a semantic seed via feature offsets.

    No descriptor/template is replaced following a loss. Reacquisition requires
    the original appearance twice in the original bounded search neighborhood.
    A seed is still a model hypothesis, not an object-identity certificate.
    """
    def __init__(self, pixels, uv, ident, *, bounds=None, extent=6):
        self.ident = ident
        self.scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], float)
        self.position = _uv(uv)*self.scale
        self.features = []
        self.reliable, self.status = True, "tracking"
        self.score, self.consensus_count, self.loss_count = 1., 0, 0
        self._reacquire = None
        candidates = []
        low, high = ((np.zeros(2), self.scale) if bounds is None else
                     (np.asarray(bounds[0])*self.scale, np.asarray(bounds[1])*self.scale))
        center = np.rint(self.position).astype(int)
        for radius in (3, 4):
            for dy in range(-extent, extent+1, 3):
                for dx in range(-extent, extent+1, 3):
                    point = center+[dx, dy]
                    if np.any(point-radius < low) or np.any(point+radius > high):
                        continue
                    x, y = point
                    patch = pixels[y-radius:y+radius+1, x-radius:x+radius+1]
                    if patch.shape != (2*radius+1, 2*radius+1) or patch.std() < 6:
                        continue
                    template = patch-patch.mean()
                    candidates.append((float(patch.std())/(1+.035*np.linalg.norm(point-self.position)),
                                       point.astype(float), radius, template, float(np.linalg.norm(template))))
        for _, point, radius, template, norm in sorted(candidates, key=lambda x: -x[0]):
            if any(np.linalg.norm(point-f["seed"]) < 3 for f in self.features):
                continue
            self.features.append({"seed": point, "offset": point-self.position,
                                  "radius": radius, "template": template, "norm": norm})
            if len(self.features) == 5:
                break
        if not self.features:
            raise ValueError("no_local_textured_features")
        self.consensus_count = len(self.features)

    def _match(self, pixels, feature, *, recovering):
        expected = self.position+feature["offset"]
        x, y = np.rint(expected).astype(int)
        r, search = feature["radius"], 12
        left, right = max(r, x-search), min(pixels.shape[1]-r-1, x+search)
        top, bottom = max(r, y-search), min(pixels.shape[0]-r-1, y+search)
        if right < left or bottom < top:
            return None
        patches = np.lib.stride_tricks.sliding_window_view(
            pixels[top-r:bottom+r+1, left-r:right+r+1], feature["template"].shape)
        energy = np.maximum(np.sum(patches*patches, axis=(-2, -1))-
                            patches.mean(axis=(-2, -1))**2*feature["template"].size, 0.)
        denominator = np.sqrt(energy)*feature["norm"]
        scores = np.divide(np.einsum("ijkl,kl->ij", patches, feature["template"], optimize=False),
                           denominator, out=np.full(energy.shape, -1.), where=denominator > 1e-9)
        row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[row, col])
        second = scores.copy()
        second[max(0, row-2):row+3, max(0, col-2):col+3] = -1.
        point = np.array([left+col, top+row], float)
        threshold, uniqueness = (.84, .08) if recovering else (.73, .045)
        if (best < threshold or best-float(second.max()) < uniqueness
                or np.max(abs(point-expected)) >= search):
            return None
        return point-feature["offset"], best

    def track(self, pixels):
        recovering = not self.reliable
        votes = [v for f in self.features if (v := self._match(pixels, f, recovering=recovering)) is not None]
        required = min(2, len(self.features))
        consensus = []
        for point, _ in votes:
            group = [v for v in votes if np.linalg.norm(v[0]-point) <= 2.1]
            if len(group) > len(consensus):
                consensus = group
        if len(consensus) < required or (required == 1 and consensus[0][1] < .9):
            self.reliable, self.status = False, "track_lost"
            self.loss_count += 1
            self._reacquire = None
            self.consensus_count = len(consensus)
            return False
        point = np.median(np.asarray([v[0] for v in consensus]), axis=0)
        self.score = min(v[1] for v in consensus)
        self.consensus_count = len(consensus)
        if recovering:
            if self._reacquire is None or np.linalg.norm(point-self._reacquire) > 2.1:
                self._reacquire = point
                self.status = "reacquisition_pending"
                return False
            self._reacquire = None
            self.status = "reacquired_original_features"
        else:
            self.status = "tracking"
        self.position, self.reliable, self.loss_count = point, True, 0
        return True

    @property
    def uv(self):
        return self.position/self.scale

    def summary(self):
        return {"reliable": self.reliable, "status": self.status,
                "feature_count": len(self.features), "consensus_count": self.consensus_count,
                "minimum_consensus_ncc": float(self.score), "loss_count": self.loss_count}


class ValidatedResponse:
    """Short-window response fit: rank-one is allowed, unseen directions are not.

    A provisional model can support a small explicitly unverified bootstrap.
    Full local servo needs two successful pre-update (held-out) predictions.
    """
    def __init__(self, *, residual_limit, minimum_excitation, max_samples=24):
        self.residual_limit, self.minimum_excitation = residual_limit, minimum_excitation
        self.max_samples = max_samples
        self.samples, self.errors = [], []
        self.matrix = self.basis = None
        self.rank, self.validations, self.failures = 0, 0, 0
        self.residual, self.last_error = None, None
        self.status = "awaiting_samples"

    def predict(self, x, *, validated=True):
        if self.matrix is None or (validated and self.validations < 2):
            return None
        value = _vector(x, self.matrix.shape[0], "prediction input")
        norm = np.linalg.norm(value)
        unsupported = np.linalg.norm(value-self.basis.T@(self.basis@value))
        if unsupported > max(1e-8, .06*norm):
            return None
        if norm > 1.2*max(np.linalg.norm(item[0]) for item in self.samples)+1e-10:
            return None
        return value@self.matrix

    def add(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("response samples must be finite vectors")
        if self.samples and (x.shape != self.samples[0][0].shape or y.shape != self.samples[0][1].shape):
            raise ValueError("response dimensions changed")
        predicted = self.predict(x, validated=False)
        old_rank = self.rank
        if predicted is not None and np.linalg.norm(x) >= self.minimum_excitation*.5:
            error = float(np.linalg.norm(y-predicted))
            self.last_error = error
            self.errors = (self.errors+[error])[-8:]
            if error > self.residual_limit:
                self.validations = 0
                if error > 2*self.residual_limit:
                    self.samples = []
                    self.failures += 1
                    self.status = "prediction_failed_relearning"
            else:
                self.validations += 1
        self.samples = (self.samples+[(x.copy(), y.copy())])[-self.max_samples:]
        self.matrix = self.basis = None
        if len(self.samples) < 2:
            return
        xx, yy = np.asarray([p[0] for p in self.samples]), np.asarray([p[1] for p in self.samples])
        u, s, vt = np.linalg.svd(xx, full_matrices=False)
        self.rank = int(np.sum(s >= max(self.minimum_excitation, s[0]/25)))
        if self.rank < 1:
            self.status = "insufficient_excitation"
            return
        if old_rank and self.rank != old_rank:
            self.validations = 0
        s, u, vt = s[:self.rank], u[:, :self.rank], vt[:self.rank]
        matrix = vt.T@((s/(s*s+max(1e-12, s[-1]**2*1e-5)))[:, None]*(u.T@yy))
        self.residual = float(np.sqrt(np.mean(np.sum((xx@matrix-yy)**2, axis=1))))
        if self.residual > self.residual_limit:
            self.validations = 0
            self.status = "inconsistent_response"
            return
        self.matrix, self.basis = matrix, vt
        self.status = "ready" if self.validations >= 2 else "provisional"

    def summary(self):
        return {"status": self.status, "sample_count": len(self.samples), "rank": self.rank,
                "residual": self.residual, "last_prediction_error": self.last_error,
                "validation_count": self.validations, "failure_count": self.failures,
                "supported_basis": None if self.basis is None else self.basis.tolist(),
                "reliable": self.matrix is not None and self.validations >= 2,
                "provisional_available": self.matrix is not None}


def _response(kind):
    return ValidatedResponse(residual_limit={"eef": .0015, "joint": .025, "image": 1.5}[kind],
                             minimum_excitation={"eef": .015, "joint": .015, "image": .0008}[kind])


class VisualFieldControllerV2:
    """Landmark-relative two-view field controller, one checked action at a time."""
    def __init__(self, *, grid_size=40, joint_lower=None, joint_upper=None):
        if type(grid_size) is not int or not 8 <= grid_size <= 96:
            raise ValueError("grid_size must be an integer in [8,96]")
        if (joint_lower is None) != (joint_upper is None):
            raise ValueError("both joint bounds must be supplied together")
        self.grid_size = grid_size
        self.joint_lower = None if joint_lower is None else np.asarray(joint_lower, float)
        self.joint_upper = None if joint_upper is None else np.asarray(joint_upper, float)
        self.action_model, self.joint_model = _response("eef"), _response("joint")
        self.feature_model, self.anchor_model = _response("image"), _response("image")
        self.views, self.motion, self._attachment = {}, [], []
        self.q = self.state8 = self.projection_state = None
        self._status, self._epochs = "unconfigured", 0
        self._last_delta_feature = None
        self._last_gripper = None
        self._prediction_errors = {}
        self._projection_reset_reason = None
        self._attachment_epoch = 0

    def _joints(self, q):
        value = np.asarray(q, float)
        if value.ndim != 1 or not 1 <= len(value) <= 32 or not np.isfinite(value).all():
            raise ValueError("q must be a finite measured joint vector")
        if self.q is not None and value.shape != self.q.shape:
            raise ValueError("joint count changed")
        if self.joint_lower is not None:
            if (self.joint_lower.shape != value.shape or self.joint_upper.shape != value.shape
                    or not np.isfinite(self.joint_lower).all() or not np.isfinite(self.joint_upper).all()
                    or np.any(self.joint_lower >= self.joint_upper)):
                raise ValueError("joint bounds are invalid")
        return value.copy()

    @staticmethod
    def _identity(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError("track identity must be a short nonempty string")
        return value

    def _track(self, pixels, point, ident, old=None, *, bounds=None):
        point = _uv(point)
        if (old is not None and old.ident == ident and np.linalg.norm(point*old.scale-old.position) <= 5
                and old.track(pixels)):
            return old
        return FeatureBankTrack(pixels, point, ident, bounds=bounds)

    def _view(self, pixels, seed, previous):
        if seed is None:
            return {"status": "seed_unavailable", "reliable": False, "geometry_reliable": False}
        required = {"moving_id", "target_id", "moving_point_uv", "target_point_uv", "target_anchor_uv",
                    "moving_radius_uv", "blocked_boxes_uv", "unknown_boxes_uv"}
        if not isinstance(seed, dict) or not required <= set(seed) or set(seed)-required-{"moving_kind", "tool_point_uv"}:
            raise ValueError("V2 view seeds require target_anchor_uv and the documented fields only")
        ids = tuple(self._identity(seed[key]) for key in ("moving_id", "target_id"))
        if ids[0] == ids[1]:
            raise ValueError("moving and target identities must differ")
        kind, radius = seed.get("moving_kind", "tool"), seed["moving_radius_uv"]
        if kind not in ("tool", "payload") or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= .3:
            raise ValueError("invalid moving kind or image radius")
        target, anchor = _uv(seed["target_point_uv"]), _uv(seed["target_anchor_uv"])
        result = {"status": "tracking", "shape": pixels.shape, "tracks": {}, "tool": None,
                  "seed_ids": ids, "moving_kind": kind, "radius": float(radius), "boxes": [],
                  "goal_offset": (target-anchor)*np.array([pixels.shape[1]-1, pixels.shape[0]-1]),
                  "reliable": True, "geometry_reliable": True, "field": None, "field_key": None}
        for key, ident, point in (("moving", ids[0], seed["moving_point_uv"]), ("target", ids[1], anchor)):
            old = (previous or {}).get("tracks", {}).get(key)
            if (previous or {}).get("shape") != pixels.shape:
                old = None
            try:
                result["tracks"][key] = self._track(pixels, point, ident, old)
            except ValueError as exc:
                result.update(reliable=False, status=str(exc))
        if seed.get("tool_point_uv") is not None:
            try:
                tool_uv = _uv(seed["tool_point_uv"])
                moving = result["tracks"].get("moving")
                if moving is None or np.linalg.norm(tool_uv*moving.scale-moving.position) >= 10:
                    result["tool"] = self._track(pixels, tool_uv, "independent_tool", (previous or {}).get("tool"))
            except ValueError:
                pass
        seen = set()
        old_boxes = {b["id"]: b for b in (previous or {}).get("boxes", [])}
        for category in ("blocked", "unknown"):
            group = seed[category+"_boxes_uv"]
            if not isinstance(group, (list, tuple)) or len(group) > 8:
                raise ValueError("at most eight boxes per category")
            for box in group:
                if not isinstance(box, dict) or set(box) != {"object_id", "lower_uv", "upper_uv"}:
                    raise ValueError("boxes require identity and normalized bounds")
                ident = self._identity(box["object_id"])
                if ident in seen:
                    raise ValueError("obstacle identities must be unique in each view")
                seen.add(ident)
                low, high = _box(box)
                old = old_boxes.get(ident)
                try:
                    patch = self._track(pixels, (low+high)/2, ident, None if old is None else old["patch"], bounds=(low, high))
                except ValueError:
                    patch = None
                    result["geometry_reliable"] = False
                preserve = old is not None and patch is old["patch"] and old["kind"] == category
                result["boxes"].append({"id": ident, "kind": category, "low": low, "high": high,
                    "patch": patch, "anchor": None if patch is None else patch.position.copy(),
                    "response": old["response"] if preserve else _response("image")})
        self._build_field(result)
        return result

    def _goal(self, view):
        return view["tracks"]["target"].position+view["goal_offset"]

    def _build_field(self, view):
        view["field"] = None if not view.get("reliable") or not view.get("geometry_reliable") else view.get("field")
        if not view.get("reliable") or not view.get("geometry_reliable"):
            return
        groups = {"blocked": [], "unknown": []}
        for box in view["boxes"]:
            delta = (box["patch"].position-box["anchor"])/box["patch"].scale
            low, high = box["low"]+delta, box["high"]+delta
            if np.any(low < 0) or np.any(high > 1):
                view.update(geometry_reliable=False, status="obstacle_out_of_view", field=None)
                return
            groups[box["kind"]].append({"lower_uv": low.tolist(), "upper_uv": high.tolist()})
        goal = self._goal(view)/view["tracks"]["target"].scale
        if np.any(goal < 0) or np.any(goal > 1):
            view.update(reliable=False, status="virtual_goal_out_of_view", field=None)
            return
        key = (tuple(goal), tuple((k, tuple(b["lower_uv"]), tuple(b["upper_uv"])) for k in groups for b in groups[k]), view["radius"])
        if key != view["field_key"] or view["field"] is None:
            view["field"] = ImageDistanceField(view["shape"], goal, grid_size=self.grid_size,
                moving_radius_uv=view["radius"], blocked_boxes_uv=groups["blocked"], unknown_boxes_uv=groups["unknown"])
            view["field_key"] = key

    def _features(self):
        if not all(self.views.get(v, {}).get("reliable") for v in VIEWS):
            return None
        return np.concatenate([self.views[v]["tracks"]["moving"].position-self._goal(self.views[v]) for v in VIEWS])

    def _anchors(self):
        if self._features() is None:
            return None
        return np.concatenate([self.views[v]["tracks"]["target"].position for v in VIEWS])

    def _obstacles(self):
        return {(name, box["id"]): box["patch"].position-self._goal(view)
                for name, view in self.views.items() if view.get("reliable") and view.get("geometry_reliable")
                for box in view.get("boxes", []) if box["patch"] is not None and box["patch"].reliable}

    def _pairs(self):
        if not all(self.views.get(v, {}).get("moving_kind") == "payload"
                   and self.views[v].get("tool") is not None and self.views[v]["tool"].reliable
                   and self.views[v].get("tracks", {}).get("moving") is not None
                   and self.views[v]["tracks"]["moving"].reliable for v in VIEWS):
            return None
        return np.array([self.views[v]["tracks"]["moving"].position-self.views[v]["tool"].position for v in VIEWS])

    def _reset_projection(self, state, reason):
        self.feature_model, self.anchor_model = _response("image"), _response("image")
        for view in self.views.values():
            for box in view.get("boxes", []):
                box["response"] = _response("image")
        self.projection_state = state.copy()
        self._projection_reset_reason = reason

    def configure(self, images, seeds, q, *, state8):
        if not isinstance(images, dict) or set(images) != set(VIEWS) or not isinstance(seeds, dict) or set(seeds) != set(VIEWS):
            raise ValueError("images and seeds must have exactly external and wrist keys")
        joints, state = self._joints(q), _vector(state8, 8, "state8")
        previous, features = self.views, self._features()
        self.views = {name: self._view(_gray(images[name]), seeds[name], previous.get(name)) for name in VIEWS}
        preserve = (features is not None and self._features() is not None and self.state8 is not None
                    and np.linalg.norm(state[3:6]-self.state8[3:6]) <= .04
                    and all(self.views[v]["tracks"].get(k) is previous[v]["tracks"].get(k) for v in VIEWS for k in ("moving", "target")))
        if not preserve:
            self._reset_projection(state, "new_or_reseeded_landmarks")
        if not all(self.views[v].get("tracks", {}).get("moving") is previous.get(v, {}).get("tracks", {}).get("moving")
                   and self.views[v].get("tool") is previous.get(v, {}).get("tool") for v in VIEWS):
            self._attachment = []
            self._attachment_epoch += 1
        self.q, self.state8 = joints, state.copy()
        self._status, self._epochs = "configured", self._epochs+1
        return self.summary()

    def observe(self, images, q, *, state8, executed_action):
        if self.q is None:
            raise RuntimeError("configure before observe")
        if not isinstance(images, dict) or set(images) != set(VIEWS):
            raise ValueError("images must have exactly external and wrist keys")
        joints, state, action = self._joints(q), _vector(state8, 8, "state8"), _vector(executed_action, 7, "executed_action")
        if np.any(abs(action) > 1):
            raise ValueError("actions must be normalized")
        before, anchors, obstacles, pairs = self._features(), self._anchors(), self._obstacles(), self._pairs()
        attachment_tracked = {(name, key): bool(track and track.reliable)
                              for name, view in self.views.items()
                              for key, track in (("moving", view.get("tracks", {}).get("moving")), ("tool", view.get("tool")))}
        moving_before = {v: view["tracks"]["moving"].position.copy() for v, view in self.views.items() if "moving" in view.get("tracks", {})}
        for name in VIEWS:
            pixels, view = _gray(images[name]), self.views[name]
            if pixels.shape != view.get("shape", pixels.shape):
                raise ValueError("image dimensions changed within an epoch")
            for track in view.get("tracks", {}).values():
                track.track(pixels)
            if view.get("tool") is not None:
                view["tool"].track(pixels)
            for box in view.get("boxes", []):
                if box["patch"] is not None:
                    box["patch"].track(pixels)
            view["reliable"] = len(view.get("tracks", {})) == 2 and all(t.reliable for t in view["tracks"].values())
            view["geometry_reliable"] = view.get("shape") is not None and all(b["patch"] is not None and b["patch"].reliable for b in view.get("boxes", []))
            view["status"] = "tracking" if view["reliable"] else "track_lost"
            if not view["geometry_reliable"]:
                view["status"] = "obstacle_track_lost"
            self._build_field(view)
        if any(attachment_tracked.get((name, key)) and not bool(track and track.reliable)
               for name, view in self.views.items()
               for key, track in (("moving", view.get("tracks", {}).get("moving")), ("tool", view.get("tool")))):
            self._attachment_epoch += 1
        after, after_anchors, after_obstacles = self._features(), self._anchors(), self._obstacles()
        delta, dq = state-self.state8, joints-self.q
        translation_stable = np.linalg.norm(action[3:6]) <= .08 and np.linalg.norm(delta[3:6]) <= .08
        projection_stable = (translation_stable and np.linalg.norm(delta[3:6]) <= .04
            and np.linalg.norm(state[3:6]-self.projection_state[3:6]) <= .12
            and max(abs(delta[6:])) <= .004
            and (self._last_gripper is None or (action[6] >= 0) == (self._last_gripper >= 0)))
        # Measured EEF response survives feature reseeding and orientation noise.
        if translation_stable:
            self.action_model.add(action[:3], delta[:3])
            self.joint_model.add(action[:3], dq)
        if not projection_stable:
            self._reset_projection(state, "orientation_or_gripper_changed")
            self._status = "projection_relearning"
        elif before is not None and after is not None:
            self.feature_model.add(delta[:3], after-before)
            self.anchor_model.add(delta[:3], after_anchors-anchors)
            self._last_delta_feature = (after-before).tolist()
            for name, view in self.views.items():
                for box in view.get("boxes", []):
                    key = name, box["id"]
                    if key in obstacles and key in after_obstacles:
                        box["response"].add(delta[:3], after_obstacles[key]-obstacles[key])
            self._status = "observed"
        else:
            self._status = "reobserve_required"
        self._prediction_errors = {name: model.last_error for name, model in (
            ("eef_prediction", self.action_model), ("image_prediction", self.feature_model),
            ("joint_prediction", self.joint_model), ("target_anchor_prediction", self.anchor_model))}
        self.motion = (self.motion+[(float(np.linalg.norm(action[:3])*.05), float(np.linalg.norm(delta[:3])))])[-5:]
        after_pairs = self._pairs()
        if pairs is not None and after_pairs is not None and projection_stable and action[6] >= 0:
            displacement = max(np.linalg.norm(self.views[v]["tracks"]["moving"].position-moving_before[v]) for v in VIEWS)
            self._attachment = (self._attachment+[(float(np.linalg.norm(delta[:3])), after_pairs.copy(), float(displacement))])[-5:]
        else:
            self._attachment = []
        self.q, self.state8, self._last_gripper = joints, state.copy(), float(action[6])
        return self.summary()

    def summary(self):
        views, reseeds = {}, []
        for name, view in self.views.items():
            tracks = view.get("tracks", {})
            entry = {"status": view["status"], "reliable": bool(view.get("reliable")),
                     "geometry_reliable": bool(view.get("geometry_reliable")),
                     "moving_id": view.get("seed_ids", (None, None))[0],
                     "target_id": view.get("seed_ids", (None, None))[1], "moving_kind": view.get("moving_kind"),
                     "moving_track_reliable": bool(tracks.get("moving") and tracks["moving"].reliable),
                     "tool_track_reliable": bool(view.get("tool") and view["tool"].reliable),
                     "obstacle_track_count": len(view.get("boxes", [])), "distance_cost_px": None,
                     "target_anchor_track_reliable": bool(tracks.get("target") and tracks["target"].reliable),
                     "image_shape_hw": list(view["shape"]) if "shape" in view else None,
                     "image_scale_px": tracks["moving"].scale.tolist() if "moving" in tracks else None,
                     "tool_point_uv": view["tool"].uv.tolist() if view.get("tool") and view["tool"].reliable else None,
                     "tracks": {key: track.summary() for key, track in tracks.items()}}
            if "moving" in tracks and "target" in tracks:
                entry.update(moving_point_uv=tracks["moving"].uv.tolist(), target_anchor_uv=tracks["target"].uv.tolist(),
                    target_point_uv=(self._goal(view)/tracks["target"].scale).tolist(),
                    target_offset_px=view["goal_offset"].tolist(), goal_error_px=float(np.linalg.norm(tracks["moving"].position-self._goal(view))))
                if view.get("field") is not None:
                    entry["distance_cost_px"] = view["field"].cost(tracks["moving"].uv)
            entry["obstacle_tracks"] = [{"object_id": b["id"], "kind": b["kind"],
                "track_reliable": bool(b["patch"] and b["patch"].reliable),
                "response_reliable": b["response"].summary()["reliable"]} for b in view.get("boxes", [])]
            missing = [k for k in ("moving", "target") if k not in tracks or not tracks[k].reliable]
            missing += ["obstacle:"+b["id"] for b in view.get("boxes", []) if b["patch"] is None or not b["patch"].reliable]
            if missing:
                reseeds.append({"view": name, "features": missing, "request": "reseed the same visible identity with a nearby textured landmark; do not invent a replacement object"})
            views[name] = entry
        relative_change = (float(np.max(np.linalg.norm(np.array([v[1] for v in self._attachment])-self._attachment[0][1], axis=-1))) if self._attachment else None)
        co_motion = bool(len(self._attachment) >= 3 and sum(v[0] for v in self._attachment) >= .005
                         and sum(v[2] for v in self._attachment) >= 1 and relative_change <= 2.)
        models = {"action_to_eef": self.action_model.summary(), "eef_to_image": self.feature_model.summary(),
                  "action_to_joint": self.joint_model.summary(), "eef_to_target_anchor": self.anchor_model.summary()}
        missing_views = [v for v in VIEWS if not views.get(v, {}).get("reliable") or not views.get(v, {}).get("geometry_reliable")]
        return {"version": VISUAL_FIELD_VERSION, "status": self._status, "views": views,
            "tracks_reliable": bool(len(views) == 2 and all(v["reliable"] for v in views.values())),
            "geometry_reliable": bool(len(views) == 2 and all(v["geometry_reliable"] for v in views.values())),
            **models, "relative_feature_change_px": self._last_delta_feature,
            "object_tool_relative_change_px": relative_change, "co_motion_evidence": co_motion,
            "co_motion_sample_count": len(self._attachment), "attachment_epoch": self._attachment_epoch,
            "observed_prediction_errors": dict(self._prediction_errors),
            "prediction_failure": next((name+"_inconsistent" for name, m in models.items() if m["status"] in ("inconsistent_response", "prediction_failed_relearning")), None),
            "projection_reset_reason": self._projection_reset_reason,
            "no_progress": bool(len(self.motion) >= 5 and sum(v[0] for v in self.motion) >= .005 and sum(v[1] for v in self.motion) <= .0005),
            "recovery": {"missing_views": missing_views, "reseed_requests": reseeds,
                "suggested_action": "reobserve_and_reseed" if missing_views else "use a short model-directed Direct probe if the proposed direction lacks measured support"},
            "clearance_certified": False, "task_success_certified": False,
            "limitations": ["weighted image pixels, not metric depth or 3D clearance", "no camera calibration or object truth",
                "no full-arm geometry or self-collision check", "seed identity, attachment and task completion are not certified"]}

    feedback = summary

    def _evaluate(self, command, *, validated):
        def prediction(model, value):
            # Do not bootstrap through a freshly contradicted fit. A subsequent
            # acknowledged Direct probe must first restore predictive agreement.
            if not validated and model.last_error is not None and model.last_error > model.residual_limit:
                return None
            return model.predict(value, validated=validated)

        eef = prediction(self.action_model, command)
        dq = prediction(self.joint_model, command)
        if eef is None or (self.joint_lower is not None and dq is None):
            return None, "response_unsupported"
        if self.joint_lower is not None and (np.any(self.q+dq < self.joint_lower+.005) or np.any(self.q+dq > self.joint_upper-.005)):
            return None, "predicted_joint_limit"
        delta = prediction(self.feature_model, eef)
        anchors = prediction(self.anchor_model, eef)
        if delta is None or anchors is None:
            return None, "response_unsupported"
        total = 0.
        for index, name in enumerate(VIEWS):
            view, sl = self.views[name], slice(2*index, 2*index+2)
            moving, field = view["tracks"]["moving"], view["field"]
            point, end = moving.uv, moving.uv+delta[sl]/moving.scale
            absolute_end = end+anchors[sl]/moving.scale
            absolute_goal = field.target_uv+anchors[sl]/moving.scale
            if (np.any(np.array([absolute_end, absolute_goal]) < field.margin_uv)
                    or np.any(np.array([absolute_end, absolute_goal]) > 1-field.margin_uv)
                    or not field.segment_valid(point, end)):
                return None, "predicted_image_path_blocked"
            cost = field.cost(end)
            if cost is None:
                return None, "predicted_image_path_blocked"
            for box, (low, high) in zip(view["boxes"], field.boxes):
                drift = prediction(box["response"], eef)
                if drift is None:
                    return None, "obstacle_response_unsupported"
                shift = drift/moving.scale
                if not _clear_segment(point, end-shift, [(low, high)]):
                    return None, "predicted_image_path_blocked"
                absolute_shift = shift+anchors[sl]/moving.scale
                if np.any(low+field.margin_uv+absolute_shift < 0) or np.any(high-field.margin_uv+absolute_shift > 1):
                    return None, "predicted_obstacle_out_of_view"
            total += cost
        return float(total), None

    def select_action(self, proposed_action, *, q=None):
        proposed = _vector(proposed_action, 7, "proposed_action")
        if np.any(abs(proposed) > 1):
            raise ValueError("actions must be normalized")
        if q is not None and (self.q is None or not np.allclose(self._joints(q), self.q, atol=1e-10, rtol=0)):
            raise ValueError("selection must use the last acknowledged joint observation")
        info, held = self.summary(), proposed.copy()
        held[:6] = 0.

        def result(action, status, accepted, *, count=0, score=None, refresh=True, projected_unknown=True, reasons=None):
            return {"action": action.tolist(), "accepted": accepted, "status": status, "horizon": 1,
                "need_refresh": refresh, "diagnostics": {"candidate_count": count, "score": score,
                    "risk_unknown": True, "projected_risk_unknown": projected_unknown, "clearance_certified": False,
                    "rotation_gripper_preserved": bool(np.array_equal(action[3:], proposed[3:])),
                    "rotation_deadband_applied": bool(accepted and 0 < np.linalg.norm(proposed[3:6]) <= .02),
                    "rotation_deadband_threshold_normalized": .02,
                    "proposed_rotation_normalized": proposed[3:6].tolist(),
                    "rejection_reasons": reasons or {}, "feedback": info}}

        for condition, status in ((self.q is None, "unconfigured"), (not info["tracks_reliable"], "visual_reference_unavailable"),
                (not info["geometry_reliable"], "obstacle_tracking_unavailable"), (info["no_progress"], "no_progress")):
            if condition:
                return result(held, status, False)
        if self.joint_lower is not None and (np.any(self.q < self.joint_lower) or np.any(self.q > self.joint_upper)):
            return result(held, "measured_joint_limit", False)
        if np.linalg.norm(proposed[3:6]) > .02:
            return result(held, "rotation_requires_direct_probe", False)
        costs = [self.views[v]["field"].cost(self.views[v]["tracks"]["moving"].uv) if self.views[v].get("field") else None for v in VIEWS]
        if any(c is None for c in costs):
            return result(held, "image_route_unavailable", False)
        norm = float(np.linalg.norm(proposed[:3]))
        if norm < 1e-8:
            return result(held, "zero_translation", False)
        current, magnitude = sum(costs), min(.2, norm)
        candidates = [proposed[:3]*min(1., .2/norm), proposed[:3]*min(.5, .1/norm)]
        candidates += [np.eye(3)[axis]*magnitude*sign for axis in range(3) for sign in (-1, 1)]
        # Rank-one data can support useful diagonals not aligned to an OSC axis.
        if self.action_model.basis is not None:
            candidates += [basis*magnitude*sign for basis in self.action_model.basis for sign in (-1, 1)]
        unique = list({tuple(np.round(c, 10)): c for c in candidates}.values())
        best, checked, reasons = None, 0, {}
        for command in unique:
            score, reason = self._evaluate(command, validated=True)
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0)+1
                continue
            checked += 1
            objective = score+.1*np.linalg.norm(command-proposed[:3])
            if score < current-.05 and (best is None or objective < best[0]):
                best = objective, command
        if best is not None:
            action = proposed.copy(); action[:3] = best[1]; action[3:6] = 0.
            return result(action, "visual_distance_descent", True, count=len(unique), score=float(best[0]),
                          refresh=False, projected_unknown=False, reasons=reasons)
        # Bootstrap uses actual previously measured direction/scale and a checked
        # two-view path. A first-ever action with no image response is never blind.
        bootstrap = proposed.copy(); bootstrap[:3] *= min(1., .1/norm); bootstrap[3:6] = 0.
        score, reason = self._evaluate(bootstrap[:3], validated=False)
        if score is not None and score < current-.025 and checked == 0:
            return result(bootstrap, "bootstrap_empirical_image_checked", True, count=len(unique)+1,
                          score=score, projected_unknown=True, reasons=reasons)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0)+1
        status = "no_supported_progress" if checked else ("direct_probe_required" if reason in ("response_unsupported", "obstacle_response_unsupported") else reason or "no_supported_progress")
        return result(held, status, False, count=len(unique)+1, reasons=reasons)
