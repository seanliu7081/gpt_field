"""Local two-view image distance fields and measured-response OSC selection.

Inputs are RGB, robot joints, end-effector/finger state, model-supplied image
seeds and acknowledged actions. Nothing here reconstructs depth, uses camera
calibration, proves object identity, or certifies scene collision clearance.
Image distances are in pixels. Robot actions remain normalized OSC_POSE actions.
"""
from __future__ import annotations

import copy
import heapq
import math
from typing import Any

import numpy as np


VISUAL_FIELD_VERSION = "two-view-image-distance-measured-response-v1"
VIEWS = ("external", "wrist")
# Standard Panda joint-position limits (radians), supplied explicitly by callers
# that know the robot model. These are not environment geometry or IK solutions.
PANDA_JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973)
PANDA_JOINT_UPPER = (2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973)


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _uv(value, name="uv"):
    result = _vector(value, 2, name)
    if np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"{name} must be inside [0,1]")
    return result


def _gray(image):
    pixels = np.asarray(image)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
            or min(pixels.shape[:2]) < 24 or max(pixels.shape[:2]) > 1024):
        raise ValueError("images must be uint8 RGB arrays, 24..1024 pixels per axis")
    return pixels.astype(float).mean(axis=2)


def _box(value):
    if not isinstance(value, dict):
        raise ValueError("box must be an object")
    low, high = _uv(value["lower_uv"]), _uv(value["upper_uv"])
    if np.any(low > high):
        raise ValueError("box lower_uv must not exceed upper_uv")
    return low, high


def _clear_segment(start, end, boxes):
    """Closed segment against closed 2D boxes; touching is not clear."""
    delta = end - start
    for low, high in boxes:
        enter, leave = 0., 1.
        possible = True
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                if start[axis] < low[axis] or start[axis] > high[axis]:
                    possible = False
                    break
            else:
                a, b = (low[axis] - start[axis])/delta[axis], (high[axis] - start[axis])/delta[axis]
                enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
        if possible and enter <= leave + 1e-12:
            return False
    return True


class ImageDistanceField:
    """Bounded 8-neighbor Dijkstra field over predicted image constraints.

    Inflation is a conservative image-space envelope, not a metric 3D shape.
    Unknown boxes are hard exclusions. Risk is a dimensionless proximity cost;
    edge length and distance are in source-image pixels (weighted pixels).
    """

    def __init__(self, image_shape, target_uv, *, blocked_boxes_uv=(), unknown_boxes_uv=(),
                 moving_radius_uv=0., grid_size=56):
        if type(grid_size) is not int or not 8 <= grid_size <= 96:
            raise ValueError("grid_size must be an integer in [8,96]")
        if len(image_shape) < 2 or min(image_shape[:2]) < 2:
            raise ValueError("image_shape must have height and width")
        if not math.isfinite(moving_radius_uv) or not 0 <= moving_radius_uv <= .3:
            raise ValueError("moving_radius_uv must be in [0,.3]")
        self.size = grid_size
        self.scale = np.array([image_shape[1]-1, image_shape[0]-1], dtype=float)
        self.target_uv = _uv(target_uv).copy()
        self.margin_uv = moving_radius_uv + 1/(grid_size-1)
        self.boxes, self.unknown_boxes = [], []
        for group, values in ((self.boxes, blocked_boxes_uv), (self.unknown_boxes, unknown_boxes_uv)):
            if len(values) > 16:
                raise ValueError("at most sixteen boxes per group")
            for value in values:
                low, high = _box(value)
                group.append((low-self.margin_uv, high+self.margin_uv))
        self.boxes += self.unknown_boxes
        axis = np.linspace(0., 1., grid_size)
        xx, yy = np.meshgrid(axis, axis)
        self.points = np.stack((xx, yy), axis=-1)
        # A tool envelope extending outside an image has unobserved coverage.
        self.valid = np.all((self.points >= self.margin_uv) & (self.points <= 1-self.margin_uv), axis=-1)
        self.unknown = ~self.valid.copy()
        self.risk = np.zeros((grid_size, grid_size))
        for low, high in self.boxes:
            inside = np.all((self.points >= low) & (self.points <= high), axis=-1)
            self.valid &= ~inside
            separation = np.maximum(np.maximum(low-self.points, self.points-high), 0.)
            self.risk += 2*np.exp(-np.sum(separation**2, axis=-1)/(2*(2/(grid_size-1))**2))
        for low, high in self.unknown_boxes:
            self.unknown |= np.all((self.points >= low) & (self.points <= high), axis=-1)
        self.distance = np.full_like(self.risk, np.inf)
        self.successor = np.full((grid_size, grid_size, 2), -1, dtype=int)
        self._solve()

    def segment_valid(self, start_uv, end_uv):
        try:
            start, end = _uv(start_uv), _uv(end_uv)
        except (ValueError, TypeError):
            return False
        if np.any(np.asarray([start, end]) < self.margin_uv-1e-12) or np.any(np.asarray([start, end]) > 1-self.margin_uv+1e-12):
            return False
        return _clear_segment(start, end, self.boxes)

    def _corners(self, uv):
        x, y = uv*(self.size-1)
        return [(j, i) for j in sorted({int(np.floor(y)), int(np.ceil(y))})
                for i in sorted({int(np.floor(x)), int(np.ceil(x))})]

    def _solve(self):
        if not self.segment_valid(self.target_uv, self.target_uv):
            return
        heap = []
        for row, col in self._corners(self.target_uv):
            point = self.points[row, col]
            if self.valid[row, col] and self.segment_valid(point, self.target_uv):
                cost = float(np.linalg.norm((point-self.target_uv)*self.scale)*(1+self.risk[row, col]))
                self.distance[row, col] = cost
                heapq.heappush(heap, (cost, row, col))
        while heap:
            cost, row, col = heapq.heappop(heap)
            if cost != self.distance[row, col]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    nr, nc = row+dy, col+dx
                    if not (0 <= nr < self.size and 0 <= nc < self.size and self.valid[nr, nc]):
                        continue
                    if not self.valid[row, nc] or not self.valid[nr, col]:
                        continue
                    if not self.segment_valid(self.points[row, col], self.points[nr, nc]):
                        continue
                    length = float(np.linalg.norm(np.array([dx, dy])*self.scale/(self.size-1)))
                    proposed = cost + length*(1+.5*(self.risk[row, col]+self.risk[nr, nc]))
                    if proposed < self.distance[nr, nc]:
                        self.distance[nr, nc] = proposed
                        self.successor[nr, nc] = [row, col]
                        heapq.heappush(heap, (proposed, nr, nc))

    def cost(self, uv):
        try:
            point = _uv(uv)
        except (ValueError, TypeError):
            return None
        if not self.segment_valid(point, point):
            return None
        choices = [float(self.distance[r, c] + np.linalg.norm((self.points[r, c]-point)*self.scale)
                         *(1+self.risk[r, c])) for r, c in self._corners(point)
                   if np.isfinite(self.distance[r, c]) and self.segment_valid(point, self.points[r, c])]
        return min(choices) if choices else None

    def route(self, start_uv, *, max_vertices=256):
        point = _uv(start_uv)
        choices = [(self.distance[r, c]+np.linalg.norm((self.points[r, c]-point)*self.scale), r, c)
                   for r, c in self._corners(point)
                   if np.isfinite(self.distance[r, c]) and self.segment_valid(point, self.points[r, c])]
        if not choices or not self.segment_valid(point, point):
            return []
        _, row, col = min(choices)
        result = [point.tolist()]
        for _ in range(max_vertices):
            vertex = self.points[row, col]
            if np.linalg.norm(vertex-np.asarray(result[-1])) > 1e-12:
                result.append(vertex.tolist())
            nxt = self.successor[row, col]
            if nxt[0] < 0:
                if np.linalg.norm(self.target_uv-np.asarray(result[-1])) > 1e-12:
                    result.append(self.target_uv.tolist())
                break
            row, col = map(int, nxt)
        return result


class _Patch:
    def __init__(self, pixels, uv, ident, *, radius=4):
        self.ident, self.radius = ident, radius
        self.scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], dtype=float)
        self.position = np.rint(_uv(uv)*self.scale)
        self.template = self._crop(pixels, self.position)
        if self.template is None or self.template.std() < 7:
            raise ValueError("textureless_or_boundary_seed")
        self.template = self.template-self.template.mean()
        self.norm = float(np.linalg.norm(self.template))
        self.score = 1.
        self.reliable = True

    def _crop(self, pixels, point):
        x, y = map(int, np.rint(point)); r = self.radius
        if x-r < 0 or y-r < 0 or x+r >= pixels.shape[1] or y+r >= pixels.shape[0]:
            return None
        return pixels[y-r:y+r+1, x-r:x+r+1].copy()

    def track(self, pixels, search=16):
        x, y = map(int, self.position); r = self.radius
        left, right = max(r, x-search), min(pixels.shape[1]-r-1, x+search)
        top, bottom = max(r, y-search), min(pixels.shape[0]-r-1, y+search)
        if right < left or bottom < top:
            self.reliable = False
            return False
        patches = np.lib.stride_tricks.sliding_window_view(pixels[top-r:bottom+r+1, left-r:right+r+1], self.template.shape)
        energy = np.maximum(np.sum(patches*patches, axis=(-2, -1))-patches.mean(axis=(-2, -1))**2*self.template.size, 0.)
        denom = np.sqrt(energy)*self.norm
        score = np.divide(np.einsum("ijkl,kl->ij", patches, self.template, optimize=False), denom,
                          out=np.full(energy.shape, -1.), where=denom > 1e-9)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        self.score = float(score[row, col])
        second = score.copy()
        second[max(0, row-2):row+3, max(0, col-2):col+3] = -1.
        point = np.array([left+col, top+row], dtype=float)
        if self.score < .8 or self.score-float(second.max()) < .08 or np.max(abs(point-self.position)) >= search:
            self.reliable = False
            return False
        self.position = point
        self.reliable = True
        return True

    @property
    def uv(self):
        return self.position/self.scale


class SupportedLinearModel:
    """Regularized local linear response with explicit supported-subspace gates."""
    def __init__(self, *, residual_limit, min_samples=6, min_rank=2, max_samples=32,
                 minimum_excitation=1e-4):
        self.residual_limit = float(residual_limit)
        self.min_samples, self.min_rank, self.max_samples = min_samples, min_rank, max_samples
        self.minimum_excitation = minimum_excitation
        self.samples = []
        self.matrix = self.basis = None
        self.rank, self.residual = 0, None
        self.status = "awaiting_samples"

    def add(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("response samples must be finite vectors")
        if self.samples and (x.shape != self.samples[0][0].shape or y.shape != self.samples[0][1].shape):
            raise ValueError("response dimensions changed")
        self.samples.append((x.copy(), y.copy()))
        self.samples = self.samples[-self.max_samples:]
        self.matrix = self.basis = None
        if len(self.samples) < self.min_samples:
            return
        xx, yy = np.asarray([v[0] for v in self.samples]), np.asarray([v[1] for v in self.samples])
        u, s, vt = np.linalg.svd(xx, full_matrices=False)
        self.rank = int(np.sum(s >= max(self.minimum_excitation, s[0]/30)))
        if self.rank < self.min_rank:
            self.status = "insufficient_excitation"
            return
        s, u, vt = s[:self.rank], u[:, :self.rank], vt[:self.rank]
        regularizer = max(1e-12, s[-1]**2*1e-6)
        matrix = vt.T @ ((s/(s*s+regularizer))[:, None]*(u.T@yy))
        residual = float(np.sqrt(np.mean(np.sum((xx@matrix-yy)**2, axis=1))))
        self.residual = residual
        if residual > self.residual_limit:
            self.status = "inconsistent_response"
            return
        self.matrix, self.basis, self.status = matrix, vt, "ready"

    def predict(self, x):
        if self.matrix is None:
            return None
        value = _vector(x, self.matrix.shape[0], "prediction input")
        norm = float(np.linalg.norm(value))
        unsupported = float(np.linalg.norm(value-self.basis.T@(self.basis@value)))
        if unsupported > max(1e-7, norm*.15):
            return None
        # No unbounded extrapolation beyond the observed one-step scale.
        if norm > 1.5*max(np.linalg.norm(v[0]) for v in self.samples)+1e-9:
            return None
        return value@self.matrix

    def summary(self):
        return {"status": self.status, "sample_count": len(self.samples), "rank": self.rank,
                "residual": self.residual, "reliable": self.matrix is not None}


class VisualFieldController:
    """Stateful bounded local perception/planning, with explicit bootstrap risk.

    configure(images, seeds, q, state8=...) starts/reanchors an epoch. Seeds have
    external/wrist entries (or null): moving_id, target_id, moving_point_uv,
    target_point_uv, moving_radius_uv, blocked_boxes_uv, unknown_boxes_uv. Each
    box has object_id, lower_uv and upper_uv. observe() consumes one acknowledged
    action and measured q/state. select_action() never certifies clearance.
    """
    def __init__(self, *, grid_size=40, joint_lower=None, joint_upper=None):
        self.grid_size = grid_size
        self.joint_lower = None if joint_lower is None else np.asarray(joint_lower, dtype=float)
        self.joint_upper = None if joint_upper is None else np.asarray(joint_upper, dtype=float)
        if (joint_lower is None) != (joint_upper is None):
            raise ValueError("both joint bounds must be supplied together")
        self.action_model = SupportedLinearModel(residual_limit=.025, minimum_excitation=.01)
        self.feature_model = SupportedLinearModel(residual_limit=1.5, minimum_excitation=.002)
        self.views = {}
        self.q = self.state8 = self.initial_q = self.initial_state = None
        self.motion = []
        self._last_delta_feature = None
        self._status = "unconfigured"
        self._epochs = 0
        self._attachment = []
        self._prediction_failure = None
        self._prediction_errors = {}

    def _joints(self, q):
        values = np.asarray(q, dtype=float)
        if values.ndim != 1 or not 1 <= len(values) <= 32 or not np.isfinite(values).all():
            raise ValueError("q must be a finite measured joint vector")
        if self.q is not None and values.shape != self.q.shape:
            raise ValueError("joint count changed")
        if self.joint_lower is not None:
            if (self.joint_lower.shape != values.shape or self.joint_upper.shape != values.shape
                    or not np.isfinite(self.joint_lower).all() or not np.isfinite(self.joint_upper).all()
                    or np.any(self.joint_lower >= self.joint_upper)):
                raise ValueError("joint bounds are invalid")
        return values.copy()

    @staticmethod
    def _identity(value, name):
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError(f"{name} must be a nonempty short string")
        return value

    def _view(self, pixels, seed, previous):
        if seed is None:
            return {"status": "seed_unavailable", "reliable": False, "geometry_reliable": False}
        required = {"moving_id", "target_id", "moving_point_uv", "target_point_uv", "moving_radius_uv",
                    "blocked_boxes_uv", "unknown_boxes_uv"}
        if not isinstance(seed, dict) or not required <= set(seed) or set(seed)-required-{"moving_kind", "tool_point_uv"}:
            raise ValueError("view seed has missing or unexpected fields")
        kind = seed.get("moving_kind", "tool")
        if kind not in ("tool", "payload"):
            raise ValueError("moving_kind must be tool or payload")
        moving_id, target_id = [self._identity(seed[key], key) for key in ("moving_id", "target_id")]
        if moving_id == target_id:
            raise ValueError("moving and target track IDs must be distinct")
        radius = seed["moving_radius_uv"]
        if isinstance(radius, bool) or not isinstance(radius, (float, int)) or not math.isfinite(radius) or not 0 <= radius <= .3:
            raise ValueError("moving_radius_uv must be a number in [0,.3]")
        result = {"status": "tracking", "reliable": True, "geometry_reliable": True,
                  "shape": pixels.shape, "radius": float(radius), "tracks": {}, "boxes": [],
                  "seed_ids": (moving_id, target_id), "moving_kind": kind,
                  "field": None, "field_key": None, "tool": None}
        for name, ident in (("moving", moving_id), ("target", target_id)):
            point = _uv(seed[name+"_point_uv"])
            old = (previous or {}).get("tracks", {}).get(name)
            try:
                if (old is not None and old.ident == ident and pixels.shape == (previous or {}).get("shape")
                        and np.linalg.norm((point-old.uv)*old.scale) <= 4 and old.track(pixels)):
                    result["tracks"][name] = old
                else:
                    result["tracks"][name] = _Patch(pixels, point, ident)
            except ValueError as exc:
                result.update(status=str(exc), reliable=False)
        if len(result["tracks"]) == 2:
            a, b = result["tracks"].values()
            if np.max(abs(a.position-b.position)) <= 8:
                result.update(status="overlapping_anchors", reliable=False)
        if seed.get("tool_point_uv") is not None:
            point = _uv(seed["tool_point_uv"], "tool_point_uv")
            try:
                result["tool"] = _Patch(pixels, point, "independent_tool")
                if (kind == "payload" and "moving" in result["tracks"]
                        and np.max(abs(result["tool"].position-result["tracks"]["moving"].position)) <= 8):
                    result["tool"] = None
            except ValueError:
                pass
        seen = set()
        for kind in ("blocked", "unknown"):
            values = seed[kind+"_boxes_uv"]
            if not isinstance(values, (list, tuple)) or len(values) > 8:
                raise ValueError("at most eight boxes per seed group")
            for box in values:
                if set(box) != {"object_id", "lower_uv", "upper_uv"}:
                    raise ValueError("seed boxes require object_id and normalized bounds")
                ident = self._identity(box["object_id"], "object_id")
                if ident in seen:
                    raise ValueError("obstacle IDs must be unique within a view")
                seen.add(ident)
                low, high = _box(box)
                patch = None
                # Seed an actual textured interior patch, not an assumed center texture.
                candidates = []
                for fx in (.2, .5, .8):
                    for fy in (.2, .5, .8):
                        try:
                            trial = _Patch(pixels, low+(high-low)*[fx, fy], ident)
                            if np.all(trial.position-4 >= low*trial.scale) and np.all(trial.position+4 <= high*trial.scale):
                                candidates.append(trial)
                        except ValueError:
                            pass
                if candidates:
                    patch = max(candidates, key=lambda item: item.norm)
                else:
                    result["geometry_reliable"] = False
                result["boxes"].append({"id": ident, "kind": kind, "low": low.copy(), "high": high.copy(),
                                         "patch": patch, "anchor": None if patch is None else patch.position.copy(),
                                         "response": SupportedLinearModel(residual_limit=1.5, minimum_excitation=.002)})
        if not result["geometry_reliable"]:
            result["status"] = "obstacle_track_unavailable"
        self._build_field(result)
        return result

    def _build_field(self, view):
        if not view.get("reliable") or not view.get("geometry_reliable"):
            view["field"] = None
            return
        groups = {"blocked": [], "unknown": []}
        for box in view["boxes"]:
            delta = (box["patch"].position-box["anchor"])/box["patch"].scale
            low, high = box["low"]+delta, box["high"]+delta
            if np.any(low < 0) or np.any(high > 1):
                view.update(geometry_reliable=False, status="obstacle_out_of_view", field=None)
                return
            groups[box["kind"]].append({"lower_uv": low.tolist(), "upper_uv": high.tolist()})
        target = view["tracks"]["target"].uv
        key = (tuple(target), tuple((kind, tuple(b["lower_uv"]), tuple(b["upper_uv"]))
                                    for kind in groups for b in groups[kind]), view["radius"])
        if key != view["field_key"]:
            view["field"] = ImageDistanceField(view["shape"], target, grid_size=self.grid_size,
                moving_radius_uv=view["radius"], blocked_boxes_uv=groups["blocked"], unknown_boxes_uv=groups["unknown"])
            view["field_key"] = key

    def configure(self, images, seeds, q, *, state8):
        if not isinstance(images, dict) or set(images) != set(VIEWS) or not isinstance(seeds, dict) or set(seeds) != set(VIEWS):
            raise ValueError("images and seeds must have exactly external and wrist keys")
        joints, state = self._joints(q), _vector(state8, 8, "state8")
        old_features = self._features()
        old_ids = tuple(self.views.get(view, {}).get("seed_ids") for view in VIEWS)
        previous_views = self.views
        self.views = {view: self._view(_gray(images[view]), seeds[view], self.views.get(view)) for view in VIEWS}
        ids = tuple(self.views[view].get("seed_ids") for view in VIEWS)
        new_features = self._features()
        preserve = (ids == old_ids and old_features is not None and new_features is not None
                    and np.linalg.norm(new_features-old_features) <= 4 and self.initial_q is not None
                    and np.linalg.norm(joints-self.initial_q) <= .35)
        if not preserve:
            self.feature_model = SupportedLinearModel(residual_limit=1.5, minimum_excitation=.002)
            self.initial_q, self.initial_state = joints.copy(), state.copy()
            self._attachment = []
            self.action_model = SupportedLinearModel(residual_limit=.025, minimum_excitation=.01)
        else:
            for name, view in self.views.items():
                previous = {box["id"]: box for box in previous_views[name].get("boxes", [])}
                for box in view.get("boxes", []):
                    old = previous.get(box["id"])
                    if (old is not None and box["patch"] is not None and old["patch"] is not None
                            and np.linalg.norm(box["patch"].position-old["patch"].position) <= 4):
                        box["response"] = old["response"]
        self.q, self.state8 = joints, state.copy()
        self.motion = []
        self._last_delta_feature = None
        self._prediction_failure = None
        self._prediction_errors = {}
        self._status = "configured"
        self._epochs += 1
        return self.summary()

    def _features(self):
        if not all(self.views.get(v, {}).get("reliable") for v in VIEWS):
            return None
        return np.concatenate([self.views[v]["tracks"]["moving"].position-self.views[v]["tracks"]["target"].position for v in VIEWS])

    def observe(self, images, q, *, state8, executed_action):
        if self.q is None:
            raise RuntimeError("configure before observe")
        if not isinstance(images, dict) or set(images) != set(VIEWS):
            raise ValueError("images must have exactly external and wrist keys")
        joints, state = self._joints(q), _vector(state8, 8, "state8")
        action = _vector(executed_action, 7, "executed_action")
        if np.any(abs(action) > 1):
            raise ValueError("actions must be normalized")
        before = self._features()
        old_obstacle_features = self._obstacle_features()
        old_pairs = self._object_tool_pairs()
        old_moving = [self.views[v].get("tracks", {}).get("moving") for v in VIEWS]
        old_moving = [None if p is None else p.position.copy() for p in old_moving]
        for name in VIEWS:
            pixels = _gray(images[name]); view = self.views[name]
            if pixels.shape != view.get("shape", pixels.shape):
                raise ValueError("image dimensions changed within an epoch")
            if view.get("tracks"):
                for track in view["tracks"].values():
                    if track.reliable and not track.track(pixels):
                        view.update(reliable=False, status="track_lost")
                view["reliable"] = len(view["tracks"]) == 2 and all(track.reliable for track in view["tracks"].values())
            if view.get("geometry_reliable"):
                for box in view.get("boxes", []):
                    if box["patch"] is None or not box["patch"].track(pixels):
                        view.update(geometry_reliable=False, status="obstacle_track_lost")
                        break
            if view.get("tool") is not None and not view["tool"].track(pixels):
                view["tool"] = None
            self._build_field(view)
        after = self._features()
        dq = joints-self.q
        delta_state = state-self.state8
        self.motion = (self.motion+[(float(np.linalg.norm(action[:3])*.05), float(np.linalg.norm(delta_state[:3])))])[-5:]
        # The learned map is translation-only. Rotation and finger changes alter
        # silhouettes/grasp assumptions; require fresh image seeds afterwards.
        stable = (np.linalg.norm(action[3:6]) < 1e-8 and np.linalg.norm(delta_state[3:6]) < .025
                  and max(abs(delta_state[6:])) < .003)
        if stable:
            pairs = [("joint_prediction", self.action_model, action[:3], dq)]
            if before is not None and after is not None:
                pairs.append(("image_prediction", self.feature_model, dq, after-before))
                self._last_delta_feature = (after-before).tolist()
            obstacles = self._obstacle_features()
            for name in VIEWS:
                for box in self.views[name].get("boxes", []):
                    key = (name, box["id"])
                    if key in obstacles and key in old_obstacle_features:
                        pairs.append(("obstacle_prediction", box["response"], dq,
                                      obstacles[key]-old_obstacle_features[key]))
            for name, model, model_input, actual in pairs:
                prediction = model.predict(model_input)
                if prediction is not None:
                    error = float(np.linalg.norm(prediction-actual))
                    self._prediction_errors[name] = max(self._prediction_errors.get(name, 0.), error)
                    if error > 2*model.residual_limit:
                        self._prediction_failure = name+"_inconsistent"
            if self._prediction_failure is None:
                for _, model, model_input, actual in pairs:
                    model.add(model_input, actual)
        if np.linalg.norm(joints-self.initial_q) > .35 or not stable:
            self.feature_model = SupportedLinearModel(residual_limit=1.5, minimum_excitation=.002)
            self._status = "pose_or_grasp_changed"
            self.action_model = SupportedLinearModel(residual_limit=.025, minimum_excitation=.01)
        else:
            self._status = "observed"
        pairs = self._object_tool_pairs()
        if old_pairs is not None and pairs is not None and stable and action[6] >= 0:
            displacement = max(float(np.linalg.norm(self.views[v]["tracks"]["moving"].position-old_moving[i]))
                               for i, v in enumerate(VIEWS))
            self._attachment = (self._attachment+[(float(np.linalg.norm(delta_state[:3])),
                pairs.copy(), displacement)])[-5:]
        else:
            self._attachment = []
        self.q, self.state8 = joints, state.copy()
        return self.summary()

    def _obstacle_features(self):
        result = {}
        for name, view in self.views.items():
            if view.get("reliable") and view.get("geometry_reliable"):
                for box in view.get("boxes", []):
                    if box["patch"] is not None:
                        result[(name, box["id"])] = box["patch"].position-view["tracks"]["target"].position
        return result

    def _object_tool_pairs(self):
        if not all(self.views.get(v, {}).get("reliable") and self.views[v].get("moving_kind") == "payload"
                   and self.views[v].get("tool") is not None for v in VIEWS):
            return None
        return np.asarray([self.views[v]["tracks"]["moving"].position-self.views[v]["tool"].position for v in VIEWS])

    def summary(self):
        views = {}
        for name, view in self.views.items():
            tracks = view.get("tracks", {})
            entry = {"status": view["status"], "reliable": bool(view.get("reliable")),
                     "geometry_reliable": bool(view.get("geometry_reliable")),
                     "moving_id": view.get("seed_ids", (None, None))[0],
                     "target_id": view.get("seed_ids", (None, None))[1],
                     "moving_kind": view.get("moving_kind"),
                     "moving_track_reliable": bool(tracks.get("moving") is not None and tracks["moving"].reliable),
                     "tool_track_reliable": bool(view.get("tool") is not None and view["tool"].reliable),
                     "obstacle_track_count": len(view.get("boxes", [])), "distance_cost_px": None}
            entry["obstacle_tracks"] = [{"object_id": box["id"], "kind": box["kind"],
                "response_reliable": box["response"].matrix is not None} for box in view.get("boxes", [])]
            if "moving" in tracks and "target" in tracks:
                entry.update(moving_point_uv=tracks["moving"].uv.tolist(), target_point_uv=tracks["target"].uv.tolist(),
                             goal_error_px=float(np.linalg.norm(tracks["moving"].position-tracks["target"].position)))
                if view.get("field") is not None:
                    entry["distance_cost_px"] = view["field"].cost(tracks["moving"].uv)
            views[name] = entry
        relative_change = (float(np.max(np.linalg.norm(np.asarray([v[1] for v in self._attachment])
                                  -self._attachment[0][1], axis=-1))) if self._attachment else None)
        co_motion = bool(len(self._attachment) >= 3 and sum(v[0] for v in self._attachment) >= .005
                         and sum(v[2] for v in self._attachment) >= 1. and relative_change <= 2.)
        return {"version": VISUAL_FIELD_VERSION, "status": self._status, "views": views,
                "tracks_reliable": bool(len(views) == 2 and all(v["reliable"] for v in views.values())),
                "geometry_reliable": bool(len(views) == 2 and all(v["geometry_reliable"] for v in views.values())),
                "action_to_joint": self.action_model.summary(), "joint_to_image": self.feature_model.summary(),
                "relative_feature_change_px": self._last_delta_feature,
                "object_tool_relative_change_px": relative_change,
                "co_motion_evidence": co_motion,
                "co_motion_sample_count": len(self._attachment),
                "prediction_failure": self._prediction_failure,
                "observed_prediction_errors": dict(self._prediction_errors),
                "no_progress": bool(len(self.motion) >= 5 and sum(v[0] for v in self.motion) >= .005 and sum(v[1] for v in self.motion) <= .0005),
                "clearance_certified": False, "task_success_certified": False,
                "limitations": ["image constraints are predicted, not 3D geometry", "no depth or camera calibration",
                                "no full-arm scene collision check", "tracked IDs and co-motion do not prove grasp or success"]}

    feedback = summary

    def select_action(self, proposed_action, *, q=None):
        proposed = _vector(proposed_action, 7, "proposed_action")
        if np.any(abs(proposed) > 1):
            raise ValueError("actions must be normalized")
        if q is not None and (self.q is None or not np.allclose(self._joints(q), self.q, atol=1e-10, rtol=0)):
            raise ValueError("selection must use the last acknowledged joint observation")
        info = self.summary()

        def result(action, status, accepted, *, count=0, risk_unknown=True, need_refresh=True, score=None):
            return {"action": action.tolist(), "accepted": accepted, "status": status,
                    "horizon": 1, "need_refresh": need_refresh,
                    "diagnostics": {"candidate_count": count, "score": score,
                        "risk_unknown": risk_unknown, "clearance_certified": False,
                        "rotation_gripper_preserved": bool(np.array_equal(action[3:], proposed[3:])), "feedback": info}}

        held = proposed.copy(); held[:6] = 0.
        if self.q is None:
            return result(held, "unconfigured", False)
        if self.joint_lower is not None and (np.any(self.q < self.joint_lower) or np.any(self.q > self.joint_upper)):
            return result(held, "measured_joint_limit", False)
        if info["no_progress"]:
            return result(held, "no_progress", False)
        if not info["tracks_reliable"]:
            return result(held, "visual_reference_unavailable", False)
        if not info["geometry_reliable"]:
            return result(held, "obstacle_tracking_unavailable", False)
        if self._prediction_failure is not None:
            return result(held, self._prediction_failure, False)
        # A known-invalid start must not be bypassed by the bootstrap fallback.
        for view in self.views.values():
            if view.get("field") is not None and view["field"].cost(view["tracks"]["moving"].uv) is None:
                return result(held, "image_route_unavailable", False)
        norm = float(np.linalg.norm(proposed[:3]))
        bootstrap = proposed.copy()
        if norm > .1:
            bootstrap[:3] *= .1/norm
        usable = (info["tracks_reliable"] and info["geometry_reliable"]
                  and self.action_model.matrix is not None and self.feature_model.matrix is not None
                  and all(box["response"].matrix is not None for view in self.views.values() for box in view.get("boxes", []))
                  and np.linalg.norm(proposed[3:6]) < 1e-8 and self._status != "pose_or_grasp_changed")
        if not usable:
            return result(bootstrap, "bootstrap_unverified", True, count=1)
        magnitude = min(.2, norm)
        candidates = [proposed[:3]*min(1., .2/max(norm, 1e-12)), proposed[:3]*min(.5, .1/max(norm, 1e-12))]
        candidates += [np.eye(3)[axis]*magnitude*sign for axis in range(3) for sign in (-1, 1)]
        best, considered, supported = None, 0, 0
        current = sum(self.views[v]["field"].cost(self.views[v]["tracks"]["moving"].uv) for v in VIEWS)
        for command in candidates:
            dq = self.action_model.predict(command)
            if dq is None:
                continue
            delta = self.feature_model.predict(dq)
            if delta is None:
                continue
            supported += 1
            if self.joint_lower is not None and (np.any(self.q+dq < self.joint_lower) or np.any(self.q+dq > self.joint_upper)):
                continue
            total, valid = 0., True
            for index, name in enumerate(VIEWS):
                view = self.views[name]; moving = view["tracks"]["moving"]
                point = moving.uv; target = point+delta[2*index:2*index+2]/moving.scale
                cost = view["field"].cost(target)
                if cost is None or not view["field"].segment_valid(point, target):
                    valid = False
                    break
                # Work in target-relative image coordinates. Camera motion can
                # move obstacle projections even when world obstacles are fixed.
                # A supported local obstacle response is required; otherwise this
                # candidate cannot be treated as a checked image-space motion.
                for box, (low, high) in zip(view["boxes"], view["field"].boxes):
                    box_delta = box["response"].predict(dq)
                    if box_delta is None:
                        valid = False
                        break
                    # Relative moving-point/box translation, checked continuously.
                    drift = box_delta/moving.scale
                    if not _clear_segment(point, target-drift, [(low, high)]):
                        valid = False
                        break
                if not valid:
                    break
                total += cost
            if not valid:
                continue
            considered += 1
            score = total+.1*float(np.linalg.norm(command-proposed[:3]))
            if total < current-.05 and (best is None or score < best[0]):
                best = (score, command)
        if best is None:
            if supported == 0:
                return result(bootstrap, "unsupported_direction_bootstrap", True, count=len(candidates))
            return result(held, "no_supported_progress", False, count=considered)
        action = proposed.copy(); action[:3] = best[1]
        return result(action, "visual_distance_descent", True, count=considered,
                      risk_unknown=False, need_refresh=False, score=float(best[0]))
