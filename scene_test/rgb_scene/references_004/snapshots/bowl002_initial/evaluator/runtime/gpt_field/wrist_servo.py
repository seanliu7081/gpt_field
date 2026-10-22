"""Uncalibrated wrist-image alignment for agent_SpeedUp_v1 (spec section 6.3).

Align moves the hand in the wrist camera's image plane until a model-pointed
mark sits at the depth-free setpoint s* = c + kappa * P @ o: c is the
principal point, P the fixed mount mapping (upright image right = R[:,1],
image up = R[:,0]), o the lateral offset of the desired contact point from the
camera axis (the grip axis is o = (0, -0.05) m; robot.xml:239), and kappa
(px/m) is read from a 2x2 image Jacobian estimated online from the robot's own
measured motion (two small probe moves, then Broyden secant updates on clean
steps). No intrinsics, depth or camera projection are used. The mark is
tracked by subpixel NCC on the unmarked upright wrist image. Commands are
in-plane translations in the frame frozen at Align start, scaled by the
realization factor of the OSC plant, capped, with the approach coordinate and
the orientation held (never an exactly-zero rotation).
"""

from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from .contact_skills import (HoldingActionAdapter, Primitive, StepView, _capped, _count, _finite, _gripper, _json,
                             _real, _view_state, quaternion_angle_rad)

WRIST_SERVO_VERSION = "agent-speedup-v1-wrist-servo-2"
GRIP_AXIS_OFFSET_M = (0.0, -0.05)  # ROBOT (robot.xml:239): grip axis offset from the camera axis in (R1, R0)
P_MATRIX = np.array([[1.0, 0.0], [0.0, -1.0]])
P_MATRIX.setflags(write=False)
MAX_CONDITION = 50.0
PIXEL_NOISE_PX = 1.5  # design default (real-LIBERO smoke 006 fit residual ~1.5 px): per-frame mark noise for kappa weights
_TRACK_TOKENS = itertools.count(1)


def _vec2(value, name: str) -> np.ndarray:
    return _finite(value, 2, name)


def _gray(image) -> np.ndarray:
    gray = np.asarray(image, dtype=float)
    if gray.ndim != 2 or min(gray.shape) < 1 or not np.isfinite(gray).all():
        raise ValueError("gray must be a finite 2-D image")
    return gray


def principal_point(width, height) -> np.ndarray:
    """Image centre ((W-1)/2, (H-1)/2) in upright pixels (ROBOT: centred principal point)."""
    width, height = _real(width, "width"), _real(height, "height")
    return np.array([(width - 1) / 2.0, (height - 1) / 2.0])


def setpoint(kappa: float, offset_m, width: int, height: int) -> np.ndarray:
    """Depth-free setpoint c + kappa * P @ offset (offset in metres along R1, R0)."""
    kappa = _real(kappa, "kappa")
    return principal_point(width, height) + kappa * (P_MATRIX @ _vec2(offset_m, "offset_m"))


def inplane_displacement(before: StepView, after: StepView, R) -> np.ndarray:
    """Measured translation projected on (R[:,1], R[:,0]) of the given (frozen) frame."""
    _view_state(before), _view_state(after)
    frame = np.asarray(R, dtype=float)
    delta = after.position - before.position
    return np.array([float(delta @ frame[:, 1]), float(delta @ frame[:, 0])])


# ------------------------------------------------------------------ NCC track
class NCCTrack:
    """Single-template normalized cross-correlation track with a subpixel peak.

    The template is a (2r+1)^2 patch centred on the nearest pixel to ``uv``
    (shifted inward at the image border); the fractional / border offset to
    the requested point is kept, so ``uv`` always reports the tracked point.
    ``match`` searches +-``search`` px around the last match. A peak on the
    search-window edge (not the image edge) is treated as a miss. Optional
    ``scales`` re-acquire a mark whose apparent size changed (after a descent);
    a scaled hit re-extracts the template at the current scale.
    """

    min_peak: float = 0.6
    REACQUIRE_SCALES = (1.08, 1.16, 1.25, 0.93, 0.86)
    _QUADRATIC_PINV = np.linalg.pinv(np.array([[1.0, x, y, x * x, x * y, y * y]
                                               for y in (-1, 0, 1) for x in (-1, 0, 1)]))

    def __init__(self, gray: np.ndarray, uv, *, radius: int = 10, search: int = 32, min_std: float = 4.0,
                 min_peak: float | None = None):
        self.radius = _count(radius, "radius")
        self.search = _count(search, "search")
        self.min_std = _real(min_std, "min_std", strict=False)
        if min_peak is not None:
            self.min_peak = _real(min_peak, "min_peak", minimum=-1.0, maximum=1.0)
        self.last_peak: float | None = None
        self.last_scale = 1.0
        self.matches = self.misses = self.retemplates = 0
        self.token = next(_TRACK_TOKENS)  # identity of the tracked point; kept by retemplate and deepcopy
        error = self._extract(_gray(gray), _vec2(uv, "uv"))
        if error is not None:
            raise ValueError(error)

    def _extract(self, gray: np.ndarray, uv: np.ndarray) -> str | None:
        r = self.radius
        h, w = gray.shape
        if not (0.0 <= uv[0] <= w - 1 and 0.0 <= uv[1] <= h - 1) or w < 2 * r + 1 or h < 2 * r + 1:
            return "template_out_of_view"
        center = np.clip(np.rint(uv), [r, r], [w - 1 - r, h - 1 - r]).astype(int)
        x, y = int(center[0]), int(center[1])
        patch = gray[y - r:y + r + 1, x - r:x + r + 1]
        if float(patch.std()) < self.min_std:
            return "textureless_template"
        source_r = int(math.ceil(r / min(self.REACQUIRE_SCALES))) + 1
        source = np.full((2 * source_r + 1, 2 * source_r + 1), np.nan)
        y0, y1, x0, x1 = max(y - source_r, 0), min(y + source_r + 1, h), max(x - source_r, 0), min(x + source_r + 1, w)
        source[y0 - y + source_r:y1 - y + source_r, x0 - x + source_r:x1 - x + source_r] = gray[y0:y1, x0:x1]
        self.template, self.norm = self._normalized(patch)
        self._source = source
        self._center = center.astype(float)
        self._offset = uv - self._center
        self.uv = uv.copy()
        return None

    @staticmethod
    def _normalized(patch: np.ndarray) -> tuple[np.ndarray, float]:
        template = patch - patch.mean()
        return template, float(np.linalg.norm(template))

    def _scaled_template(self, scale: float):
        r, source_r = self.radius, (self._source.shape[0] - 1) // 2
        coords = np.arange(-r, r + 1) / scale + source_r
        x0 = np.floor(coords).astype(int)
        fx = coords - x0
        if x0.min() < 0 or x0.max() + 1 >= self._source.shape[0]:
            return None
        s = self._source
        rows = s[x0][:, x0] * np.outer(1 - fx, 1 - fx) + s[x0][:, x0 + 1] * np.outer(1 - fx, fx) \
            + s[x0 + 1][:, x0] * np.outer(fx, 1 - fx) + s[x0 + 1][:, x0 + 1] * np.outer(fx, fx)
        if not np.isfinite(rows).all() or float(rows.std()) < self.min_std:
            return None
        return self._normalized(rows)

    def _search(self, gray: np.ndarray, template: np.ndarray, norm: float):
        """(template-centre uv | None, peak) in a +-search window around the last centre."""
        r, s = self.radius, self.search
        h, w = gray.shape
        x, y = int(round(self._center[0])), int(round(self._center[1]))
        lo = np.array([max(r, x - s), max(r, y - s)])
        hi = np.array([min(w - 1 - r, x + s), min(h - 1 - r, y + s)])
        if np.any(hi < lo) or norm < 1e-9:
            return None, -1.0
        k, n = 2 * r + 1, template.size
        crop = gray[lo[1] - r:hi[1] + r + 1, lo[0] - r:hi[0] + r + 1]
        integral = np.zeros((crop.shape[0] + 1, crop.shape[1] + 1))
        integral[1:, 1:] = crop.cumsum(0).cumsum(1)
        squares = np.zeros_like(integral)
        squares[1:, 1:] = (crop * crop).cumsum(0).cumsum(1)
        box = lambda a: a[k:, k:] - a[:-k, k:] - a[k:, :-k] + a[:-k, :-k]
        sums, energy = box(integral), box(squares)
        energy = np.maximum(energy - sums * sums / n, 0.0)
        numerator = np.einsum("ijkl,kl->ij", np.lib.stride_tricks.sliding_window_view(crop, (k, k)), template)
        denominator = np.sqrt(energy) * norm
        score = np.full(numerator.shape, -1.0)
        valid = denominator > 1e-6 * norm * norm
        score[valid] = numerator[valid] / denominator[valid]
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        peak = float(np.clip(score[row, col], -1.0, 1.0))
        rows, cols = score.shape
        on_edge = ((col == 0 and lo[0] > r) or (col == cols - 1 and hi[0] < w - 1 - r)
                   or (row == 0 and lo[1] > r) or (row == rows - 1 and hi[1] < h - 1 - r))
        if on_edge:
            return None, peak
        return lo + np.array([col, row], dtype=float) + self._subpixel(score, row, col), peak

    @classmethod
    def _subpixel(cls, score: np.ndarray, row: int, col: int) -> np.ndarray:
        """2-D quadratic fit on the 3x3 peak neighbourhood; 1-D parabolas at window borders."""
        rows, cols = score.shape
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            c = cls._QUADRATIC_PINV @ score[row - 1:row + 2, col - 1:col + 2].ravel()
            hessian = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
            if hessian[0, 0] < -1e-9 and np.linalg.det(hessian) > 1e-12:
                offset = np.linalg.solve(hessian, -c[1:3])
                if np.all(np.abs(offset) <= 1.0):
                    return np.clip(offset, -0.5, 0.5)
        sub = np.zeros(2)
        if 0 < col < cols - 1:
            sub[0] = cls._parabola(score[row, col - 1], score[row, col], score[row, col + 1])
        if 0 < row < rows - 1:
            sub[1] = cls._parabola(score[row - 1, col], score[row, col], score[row + 1, col])
        return sub

    @staticmethod
    def _parabola(minus: float, peak: float, plus: float) -> float:
        minus, peak, plus = float(minus), float(peak), float(plus)
        curvature = minus - 2 * peak + plus
        return float(np.clip(0.5 * (minus - plus) / curvature, -0.5, 0.5)) if curvature < -1e-9 else 0.0

    def match(self, gray, *, scales=None) -> tuple[np.ndarray | None, float]:
        """Subpixel match; on success (peak >= min_peak) updates and returns ``uv``."""
        image = _gray(gray)
        center, peak = self._search(image, self.template, self.norm)
        scale = 1.0
        if (center is None or peak < self.min_peak) and scales:
            for candidate in scales:
                scaled = self._scaled_template(float(candidate))
                if scaled is None:
                    continue
                c, p = self._search(image, *scaled)
                if c is not None and (center is None or p > peak):
                    center, peak, scale = c, p, float(candidate)
        self.last_peak, self.last_scale = peak, scale
        if center is None or peak < self.min_peak:
            self.misses += 1
            return None, peak
        self.matches += 1
        self._center = center
        self.uv = center + scale * self._offset
        if scale != 1.0:
            self.retemplate(image)
        return self.uv.copy(), peak

    def retemplate(self, gray, uv=None) -> bool:
        """Re-extract the template at ``uv`` (default: current); False keeps the old one."""
        state = (self.template, self.norm, self._source, self._center, self._offset, self.uv)
        try:
            error = self._extract(_gray(gray), self.uv.copy() if uv is None else _vec2(uv, "uv"))
        except ValueError:
            error = "invalid"
        if error is not None:
            self.template, self.norm, self._source, self._center, self._offset, self.uv = state
            return False
        self.retemplates += 1
        return True

    def to_dict(self) -> dict:
        return _json({"token": self.token, "uv": self.uv, "radius": self.radius, "search": self.search,
                      "min_peak": self.min_peak, "last_peak": self.last_peak, "last_scale": self.last_scale,
                      "matches": self.matches, "misses": self.misses, "retemplates": self.retemplates})


# ------------------------------------------------------------ image Jacobian
class ImageJacobian:
    """2x2 map from in-plane hand translation (m along R1, R0) to mark pixel motion."""

    def __init__(self):
        self.J: np.ndarray | None = None
        self.source: str | None = None
        self.probe_samples = 0
        self.broyden_updates = 0
        self.broyden_rejected = 0
        self.kappa_rel_std: float | None = None  # heuristic relative uncertainty of kappa
        self.anchor: dict | None = None  # last (track token, uv, pose, kappa) left by an Align

    @property
    def ready(self) -> bool:
        return self.J is not None

    @staticmethod
    def _condition(matrix: np.ndarray) -> float:
        singular = np.linalg.svd(matrix, compute_uv=False)
        return float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf

    def set_from_probes(self, samples) -> None:
        """Least-squares J from >= 2 non-collinear (ds px, dp m) samples."""
        pairs = [(_vec2(ds, "ds"), _vec2(dp, "dp")) for ds, dp in samples]
        if len(pairs) < 2:
            raise ValueError("at least two probe samples are required")
        DS = np.array([ds for ds, _ in pairs])
        DP = np.array([dp for _, dp in pairs])
        if self._condition(DP) > MAX_CONDITION:
            raise ValueError("ill_conditioned_probes")
        J = np.linalg.lstsq(DP, DS, rcond=None)[0].T
        if not np.isfinite(J).all() or self._condition(J) > MAX_CONDITION:
            raise ValueError("ill_conditioned_jacobian")
        self.J, self.source, self.probe_samples = J, "probes", len(pairs)

    def broyden(self, ds, dp, *, min_step_m=0.0005, alpha: float = 1.0, max_kappa_ratio: float = 1.5,
                max_innovation_fraction: float = 0.5) -> bool:
        """Rank-one secant update J += alpha (ds - J dp) dp^T / |dp|^2 if |dp| >= min_step_m.

        Rejected (False, J unchanged) when the update is ill-conditioned, flips
        the orientation (det sign), moves kappa by more than ``max_kappa_ratio``,
        or the innovation exceeds a fraction of the predicted motion (a tracking
        jump rather than a model error).
        """
        if not self.ready:
            return False
        ds, dp = _vec2(ds, "ds"), _vec2(dp, "dp")
        norm2 = float(dp @ dp)
        if math.sqrt(norm2) < float(min_step_m):
            return False
        predicted = self.J @ dp
        innovation = ds - predicted
        candidate = self.J + float(alpha) * np.outer(innovation, dp) / norm2
        old_kappa = self.kappa
        innovation_limit = max(3.0, max_innovation_fraction * float(np.linalg.norm(predicted)))
        ok = (np.isfinite(candidate).all() and self._condition(candidate) <= MAX_CONDITION
              and np.sign(np.linalg.det(candidate)) == np.sign(np.linalg.det(self.J))
              and float(np.linalg.norm(innovation)) <= innovation_limit)
        if ok:
            ratio = float(np.mean(np.linalg.svd(candidate, compute_uv=False))) / old_kappa
            ok = 1.0 / max_kappa_ratio <= ratio <= max_kappa_ratio
        if not ok:
            self.broyden_rejected += 1
            return False
        self.J = candidate
        self.broyden_updates += 1
        return True

    @staticmethod
    def similarity_kappa(samples, *, reflection: bool | None = None) -> float | None:
        """Scale of the best similarity map ds = kappa Q dp (Q orthogonal).

        A pinhole with square pixels translating in its image plane obeys
        J = -kappa P, a similarity (a reflection: det < 0), so every sample
        constrains kappa whatever its direction; long baselines dominate the
        least-squares fit. ``reflection`` None picks the better-fitting handedness.
        """
        pairs = [(_vec2(ds, "ds"), _vec2(dp, "dp")) for ds, dp in samples]
        if not pairs:
            return None
        best = None
        for mirrored in ((True, False) if reflection is None else (bool(reflection),)):
            rows, targets = [], []
            for ds, dp in pairs:
                # rotation [[a,-b],[b,a]]; reflection [[a,b],[b,-a]]
                rows += ([[dp[0], dp[1]], [-dp[1], dp[0]]] if mirrored else [[dp[0], -dp[1]], [dp[1], dp[0]]])
                targets += [ds[0], ds[1]]
            A, b = np.array(rows), np.array(targets)
            if np.linalg.matrix_rank(A) < 2:
                return None
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            cost = float(np.sum((A @ params - b) ** 2))
            if best is None or cost < best[0]:
                best = (cost, float(np.hypot(*params)))
        return best[1] if best[1] > 0 and math.isfinite(best[1]) else None

    def rescale(self, kappa: float) -> None:
        """Scale J so that its mean singular value equals ``kappa`` (shape kept)."""
        kappa = _real(kappa, "kappa")
        self.J = self.J * (kappa / self.kappa)

    @property
    def kappa(self) -> float:
        """Mean singular value of J (px per metre)."""
        if not self.ready:
            raise RuntimeError("image Jacobian is not estimated yet")
        return float(np.mean(np.linalg.svd(self.J, compute_uv=False)))

    def solve(self, error_px) -> np.ndarray:
        """In-plane displacement (m along R1, R0) that cancels ``error_px``: J dp = -error."""
        if not self.ready:
            raise RuntimeError("image Jacobian is not estimated yet")
        return np.linalg.lstsq(self.J, -_vec2(error_px, "error_px"), rcond=None)[0]

    def model_deviation(self) -> float | None:
        """Relative Frobenius distance of J from the mount model -kappa P (0 = exact)."""
        if not self.ready:
            return None
        kappa = self.kappa
        return float(np.linalg.norm(self.J + kappa * P_MATRIX) / (kappa * math.sqrt(2.0)))

    def to_dict(self) -> dict:
        ready = self.ready
        return _json({"ready": ready, "J": self.J.tolist() if ready else None,
                      "kappa": self.kappa if ready else None,
                      "condition": self._condition(self.J) if ready else None,
                      "model_deviation": self.model_deviation(), "source": self.source,
                      "probe_samples": self.probe_samples, "broyden_updates": self.broyden_updates,
                      "broyden_rejected": self.broyden_rejected, "kappa_rel_std": self.kappa_rel_std,
                      "anchor": None if self.anchor is None else {
                          key: self.anchor[key] for key in ("token", "uv", "xyz", "kappa", "rel_std")}})


# --------------------------------------------------------------------- Align
class _Frame:
    """One observed wrist frame: measured pose in the frozen frame plus the mark match."""

    __slots__ = ("view", "inplane", "along", "quaternion", "uv", "peak")

    def __init__(self, view: StepView, R: np.ndarray, uv, peak):
        self.view = view
        self.inplane = np.array([float(view.position @ R[:, 1]), float(view.position @ R[:, 0])])
        self.along = float(view.position @ R[:, 2])
        self.quaternion = view.quaternion_xyzw
        self.uv = None if uv is None else np.asarray(uv, dtype=float).copy()
        self.peak = None if peak is None else float(peak)


def _cell_id(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= 64:
        raise ValueError("cell_id must be an integer in 1..64")
    return int(value)


def _fuse(estimates) -> tuple[float, float]:
    """Inverse-variance fusion of (kappa, relative std) estimates."""
    weights = [(kappa, 1.0 / (rel * kappa) ** 2) for kappa, rel in estimates]
    total = sum(w for _, w in weights)
    kappa = sum(k * w for k, w in weights) / total
    return kappa, 1.0 / (math.sqrt(total) * kappa)


class Align(Primitive):
    """Put a pointed mark at the depth-free setpoint by image-based visual servoing.

    Phases: point (coarse cell, then fine point; only without a track) ->
    probe (two small anchored moves along R1 then R0; least-squares J) -> servo.
    Track and Jacobian inputs are copied; read the updated ones back from
    ``align.track`` / ``align.jacobian``.

    Kappa (the only Jacobian quantity the setpoint uses) is estimated from all
    clean samples since a reference frame with the similarity model
    J = -kappa P, so long servo baselines refine it whatever their direction.
    Broyden secants are taken from that reference frame (long, low-noise
    baseline) and only on clean steps: both frames matched, relative rotation
    <= 0.2 deg, no approach motion. When the Jacobian carries the anchor left
    by a previous Align on the same track, kappa is first transferred to the
    new depth from the radial image scaling of the mark (valid with the
    orientation unchanged; uses only the centred principal point) and fused
    with the new probes/secants.
    """

    PROBE_COMMAND_M = 0.006
    PROBE_MAX_STEPS = 8
    PROBE_FRACTION = 0.8
    PROBE_MIN_FRACTION = 0.25
    LATERAL_HOLD_CAP_M = 0.003
    APPROACH_HOLD_CAP_M = 0.003
    CLEAN_ROTATION_RAD = 0.0035
    LOST_FRAMES = 2
    CONVERGED_FRAMES = 3
    MAX_ROTATION_RAD = 0.15
    MAX_MODEL_DEVIATION = 0.5
    RETEMPLATE_BELOW_PEAK = 0.85
    # Real-LIBERO smoke (2026-10-16): during a long lateral servo the rim's perspective changes and the
    # fixed template's peak decayed 0.97 -> 0.60 until the track was lost. While servoing, refresh the
    # template at the matched point whenever the match is still confident but decaying.
    RETEMPLATE_MIN_PEAK = 0.70
    TOLERANCE_MM = 3.0
    MIN_TOLERANCE_PX = 2.0
    MAX_KAPPA_STEP_RATIO = 1.1
    MIN_KAPPA_BASELINE_M = 0.005   # refit kappa only from baselines >= 5 mm (short ones are noise-dominated)
    MAX_PROBE_RESTARTS = 2   # smoke 006: kappa swung -23%..+37% around the truth with 1.5
    MIN_TRANSFER_OFFSET_M = 0.01
    SETTLE_ROTATION_RAD = math.radians(0.03)   # ladder settles ran 0.007-0.028 deg/step (live 001)
    SETTLE_TRANSLATION_M = 0.0003
    SETTLE_QUIET_STEPS = 2
    SETTLE_MAX_STEPS = 14                       # the live swing returned within 0.2 deg after 13-14 steps

    def __init__(self, view: StepView, *, offset_m=GRIP_AXIS_OFFSET_M, gripper: str = "open",
                 track: NCCTrack | None = None, jacobian: ImageJacobian | None = None, reprobe: bool = True,
                 allow_pointing: bool = True, max_repoints: int = 2, tolerance_px: float = 6.0,
                 gain: float = 0.5, realization: float = 0.3, max_command_m: float = 0.010,
                 max_servo_steps: int = 90, probe_m: float = 0.002, setpoint_row_limit: float | None = None,
                 far_command_m: float = 0.030, far_fraction: float = 0.25,
                 name: str = "align", min_step_m: float = 0.0005, track_radius: int = 10, track_search: int = 32,
                 kappa_transfer: bool = True):
        super().__init__(name)
        _view_state(view)
        if track is not None and not isinstance(track, NCCTrack):
            raise ValueError("track must be an NCCTrack")
        if jacobian is not None and not isinstance(jacobian, ImageJacobian):
            raise ValueError("jacobian must be an ImageJacobian")
        self.offset_m = _vec2(offset_m, "offset_m")
        self.gripper = _gripper(gripper)
        self.reprobe, self.allow_pointing = bool(reprobe), bool(allow_pointing)
        if (isinstance(max_repoints, (bool, np.bool_)) or not isinstance(max_repoints, (int, np.integer))
                or max_repoints < 0):
            raise ValueError("max_repoints must be a nonnegative integer")
        self.max_repoints = int(max_repoints)
        self.tolerance_px = _real(tolerance_px, "tolerance_px")
        self.gain = _real(gain, "gain", maximum=2.0)
        self.realization = _real(realization, "realization", maximum=1.0)
        self.max_command_m = _real(max_command_m, "max_command_m", maximum=0.04)
        # Design default (real-LIBERO smoke 2026-10-16): a 10 mm cap realizes ~2.5 mm/step, too slow for a
        # hover that is several cm off. The cap grows with the estimated remaining lateral error and returns
        # to max_command_m near the setpoint, so the final approach keeps the fine step.
        self.far_command_m = max(_real(far_command_m, "far_command_m", maximum=0.045), self.max_command_m)
        self.far_fraction = _real(far_fraction, "far_fraction", maximum=1.0)
        self.max_servo_steps = _count(max_servo_steps, "max_servo_steps")
        self.probe_m = _real(probe_m, "probe_m", maximum=0.02)
        self.min_step_m = _real(min_step_m, "min_step_m", strict=False)
        self.setpoint_row_limit = (None if setpoint_row_limit is None
                                   else _real(setpoint_row_limit, "setpoint_row_limit", minimum=-math.inf))
        self.track_radius = _count(track_radius, "track_radius")
        self.track_search = _count(track_search, "track_search")
        self.frame_R = view.R.copy()
        self.frame_R.setflags(write=False)
        self.hold_xyzw = view.quaternion_xyzw.copy()
        self.start_xyz = view.position.copy()
        self.approach_anchor = float(view.position @ self.frame_R[:, 2])
        self.adapter = HoldingActionAdapter(max_rotation_rad=self.MAX_ROTATION_RAD)
        self.height, self.width = view.wrist_gray_upright.shape
        self.track = copy.deepcopy(track)
        self.jacobian = copy.deepcopy(jacobian) if jacobian is not None else ImageJacobian()
        self.jacobian_input = None if jacobian is None else jacobian.to_dict()
        self.phase = "init"
        self.cell_id: int | None = None
        self.repoints = self.probe_steps = self.servo_steps = self.settle_steps = 0
        self.settle_record: list[dict] = []
        self._settle: dict | None = None
        self.lost_frames = self.lost_frames_total = 0
        self.converged_frames = 0
        self.probed = False
        self.probe_record: list[dict] = []
        self.pointed_uv: list[list[float]] = []
        self.kappa_transfer: dict | None = None
        self.kappa_estimates: dict = {}
        self.last_uv: np.ndarray | None = None
        self.last_setpoint: np.ndarray | None = None
        self.last_peak: float | None = None
        self.residual_history: list[float] = []
        self.realized: list[float] = []
        self.max_approach_drift_m = 0.0
        self._probe: dict | None = None
        self._probe_frames: list[_Frame] = []
        self._reference: _Frame | None = None
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._kappa_prior: tuple[float, float] | None = None
        self._good_frame: _Frame | None = None
        self._frame = self._observe_frame(view, scales=NCCTrack.REACQUIRE_SCALES, retemplate=True)
        if self.track is None:
            if self.allow_pointing:
                self.phase = "point_coarse"
            else:
                self._fail("mark_lost", detail="no_track_and_pointing_disabled")
        elif self._frame.uv is None:
            self._on_lost()
        else:
            if kappa_transfer and self.jacobian.ready:
                self._transfer_kappa(self._frame)
            self._track_ready()

    # ------------------------------------------------------------- pointing
    @property
    def pointing_request(self) -> dict | None:
        if self.done:
            return None
        if self.phase == "point_coarse":
            return {"stage": "coarse"}
        if self.phase == "point_fine":
            return {"stage": "fine", "cell_id": self.cell_id}
        return None

    def provide_coarse(self, cell_id) -> None:
        """Coarse pointing answer (cell id 1..64, or None = not visible)."""
        if self.done or self.phase != "point_coarse":
            raise RuntimeError("Align is not waiting for a coarse pointing answer")
        if cell_id is None:
            self._fail("mark_not_visible", detail="coarse_null")
            return
        self.cell_id = _cell_id(cell_id)
        self.phase = "point_fine"

    def provide_point(self, uv, view: StepView | None = None) -> None:
        """Fine pointing answer as an UPRIGHT pixel on ``view`` (default: the latest frame)."""
        if self.done or self.phase != "point_fine":
            raise RuntimeError("Align is not waiting for a fine pointing answer")
        if uv is None:
            self._fail("mark_not_visible", detail="fine_null")
            return
        uv = _vec2(uv, "uv")
        if view is not None:
            _view_state(view)
        view = self._frame.view if view is None else view
        self.pointed_uv.append([float(uv[0]), float(uv[1])])
        try:
            self.track = NCCTrack(view.wrist_gray_upright, uv, radius=self.track_radius, search=self.track_search)
        except ValueError as exc:
            reason = "textureless_mark" if str(exc) == "textureless_template" else "mark_not_visible"
            self._fail(reason, detail=str(exc))
            return
        self._frame = _Frame(view, self.frame_R, self.track.uv, 1.0)
        self.lost_frames = 0
        self.last_peak = 1.0
        self._note_good(self._frame)
        self._track_ready()

    # ------------------------------------------------------------- phases
    def _track_ready(self) -> None:
        self._reference, self._samples = self._frame, []
        if self.jacobian.ready and self._cutoff():
            return
        if not self.jacobian.ready or (self.reprobe and not self.probed):
            self._start_probe()
        else:
            self._start_servo()

    def _start_probe(self) -> None:
        # Live run 001 (probe_failed at steps 122/148): a WaypointReach ends goal_reached while the arm still
        # moves 2-3 mm and ~0.25 deg per step; the residual swing (up to 0.83 deg) rotated every probe frame
        # past CLEAN_ROTATION_RAD. Hold still until the arm has settled, then re-base the frozen frame and
        # the hold orientation on the measured pose before the probe origin is taken.
        self.phase = "probe_settle"
        self._settle = {"steps": 0, "quiet": 0, "anchor": self._frame.view.position.copy()}

    def _begin_probe(self) -> None:
        self.phase = "probe"
        self._probe_frames = [self._frame]
        self.probe_record = []
        self._probe = {"index": 0, "axis": self.frame_R[:, 1].copy(), "perp": self.frame_R[:, 0].copy(),
                       "start": self._frame.view.position.copy(), "steps": 0}

    def _observe_settle(self, previous: _Frame, frame: _Frame) -> None:
        settle = self._settle
        settle["steps"] += 1
        self.settle_steps += 1
        rotation = quaternion_angle_rad(previous.quaternion, frame.quaternion)
        motion = float(np.linalg.norm(frame.view.position - previous.view.position))
        quiet = rotation <= self.SETTLE_ROTATION_RAD and motion <= self.SETTLE_TRANSLATION_M
        settle["quiet"] = settle["quiet"] + 1 if quiet else 0
        if settle["quiet"] < self.SETTLE_QUIET_STEPS and settle["steps"] < self.SETTLE_MAX_STEPS:
            return
        view = frame.view
        self.settle_record.append({"steps": settle["steps"], "settled": settle["quiet"] >= self.SETTLE_QUIET_STEPS,
                                   "rebase_rotation_deg": math.degrees(quaternion_angle_rad(self.hold_xyzw,
                                                                                            view.quaternion_xyzw))})
        self.frame_R = view.R.copy()
        self.frame_R.setflags(write=False)
        self.hold_xyzw = view.quaternion_xyzw.copy()
        self.approach_anchor = float(view.position @ self.frame_R[:, 2])
        self._frame = _Frame(view, self.frame_R, frame.uv, frame.peak)
        if self._frame.uv is None:
            self._on_lost()
            return
        self._reference, self._samples = self._frame, []
        self._begin_probe()

    def _start_servo(self) -> None:
        self.phase = "servo"
        self.converged_frames = 0
        self._check_servo_frame(self._frame)

    def _on_lost(self) -> None:
        if self.allow_pointing and self.repoints < self.max_repoints:
            self.repoints += 1
            self.phase = "point_coarse"
            self.cell_id = None
            self.lost_frames = 0
            self.converged_frames = 0
            self._probe = None
        else:
            self._fail("mark_lost", detail="ncc_peak_below_threshold")

    def _fail(self, reason: str, **extra) -> None:
        self._finish("failed", reason, self._evidence_now(**extra))

    def _finish(self, status: str, reason: str, evidence: dict) -> None:
        super()._finish(status, reason, evidence)
        frame = self._good_frame
        kappa_is_current = self.probed or "transfer" in self.kappa_estimates or "samples" in self.kappa_estimates
        if frame is not None and self.track is not None and self.jacobian.ready and kappa_is_current:
            # (track, uv, pose, kappa at this depth) for a later Align's depth-free kappa transfer;
            # otherwise the incoming anchor (consistent with its own pose) is kept
            self.jacobian.anchor = {"token": self.track.token, "uv": frame.uv.copy(),
                                    "xyz": frame.view.position.copy(), "R": frame.view.R.copy(),
                                    "quaternion": frame.quaternion.copy(), "kappa": self.jacobian.kappa,
                                    "rel_std": self.jacobian.kappa_rel_std}

    # ------------------------------------------------------------- frames
    def _observe_frame(self, view: StepView, *, scales=None, retemplate=False, retemplate_floor=None) -> _Frame:
        uv = peak = None
        if self.track is not None:
            uv, peak = self.track.match(view.wrist_gray_upright, scales=scales)
            if (uv is not None and retemplate and peak < self.RETEMPLATE_BELOW_PEAK
                    and (retemplate_floor is None or peak >= retemplate_floor)):
                self.track.retemplate(view.wrist_gray_upright)
                uv = self.track.uv.copy()
        frame = _Frame(view, self.frame_R, uv, peak)
        self.max_approach_drift_m = max(self.max_approach_drift_m, abs(frame.along - self.approach_anchor))
        self.last_peak = frame.peak
        if self.track is not None:
            if uv is None:
                self.lost_frames += 1
                self.lost_frames_total += 1
            else:
                self.lost_frames = 0
                self._note_good(frame)
        return frame

    def _note_good(self, frame: _Frame) -> None:
        self.last_uv = frame.uv.copy()
        self._good_frame = frame

    def _tolerance(self) -> float:
        """Stop tolerance in pixels: tolerance_px, tightened to TOLERANCE_MM at the current kappa (>= 2 px).

        Real-LIBERO smoke: at a high hover (kappa ~900 px/m) 6 px is ~6.7 mm of lateral error."""
        kappa = self._kappa()
        if kappa is None:
            return self.tolerance_px
        return float(min(self.tolerance_px, max(self.MIN_TOLERANCE_PX, kappa * self.TOLERANCE_MM / 1000.0)))

    def _kappa(self) -> float | None:
        return self.jacobian.kappa if self.jacobian.ready else None

    def _setpoint(self) -> np.ndarray | None:
        kappa = self._kappa()
        if kappa is None:
            return None
        self.last_setpoint = setpoint(kappa, self.offset_m, self.width, self.height)
        return self.last_setpoint

    def _cutoff(self) -> bool:
        target = self._setpoint()
        if self.setpoint_row_limit is not None and target is not None and target[1] > self.setpoint_row_limit:
            self._finish("succeeded", "setpoint_cutoff", self._evidence_now())
            return True
        return False

    def _check_servo_frame(self, frame: _Frame) -> None:
        if self._cutoff():
            return
        if frame.uv is None:
            self.converged_frames = 0
            return
        residual = float(np.linalg.norm(frame.uv - self.last_setpoint))
        self.residual_history.append(residual)
        self.converged_frames = self.converged_frames + 1 if residual <= self._tolerance() else 0
        if self.converged_frames >= self.CONVERGED_FRAMES:
            self._finish("succeeded", "aligned", self._evidence_now())

    # ------------------------------------------------------------- kappa
    def _transfer_kappa(self, frame: _Frame) -> None:
        """Depth-free kappa at this pose from the previous Align's anchor on the same track."""
        anchor = self.jacobian.anchor
        if anchor is None or anchor.get("token") != self.track.token:
            return
        record = {"anchor_kappa": anchor["kappa"], "applied": False}
        self.kappa_transfer = record
        rotation = quaternion_angle_rad(anchor["quaternion"], frame.quaternion)
        record["rotation_rad"] = rotation
        if rotation > self.CLEAN_ROTATION_RAD:
            record["skipped"] = "orientation_changed"
            return
        center, R = principal_point(self.width, self.height), anchor["R"]
        delta = frame.view.position - anchor["xyz"]
        lateral_old = P_MATRIX @ (anchor["uv"] - center) / anchor["kappa"]
        lateral_new = lateral_old - np.array([float(delta @ R[:, 1]), float(delta @ R[:, 0])])
        radial_px = frame.uv - center
        record.update(approach_change_m=float(delta @ R[:, 2]), lateral_offset_m=float(np.linalg.norm(lateral_new)))
        if float(np.linalg.norm(lateral_new)) < self.MIN_TRANSFER_OFFSET_M:
            record["skipped"] = "mark_too_close_to_camera_axis"
            return
        kappa = float(radial_px @ (P_MATRIX @ lateral_new)) / float(lateral_new @ lateral_new)
        record["kappa"] = kappa
        if not (math.isfinite(kappa) and 1 / 2.0 <= kappa / anchor["kappa"] <= 2.0):
            record["skipped"] = "implausible_ratio"
            return
        anchor_rel = anchor.get("rel_std") or 0.1
        rel = math.sqrt(anchor_rel ** 2 + 2 * (PIXEL_NOISE_PX / max(float(np.linalg.norm(radial_px)), 1.0)) ** 2)
        self._kappa_prior = (kappa, rel)
        self.jacobian.rescale(kappa)
        self.jacobian.kappa_rel_std = rel
        self.jacobian.source = "kappa_transfer"
        record.update(applied=True, rel_std=rel)
        self.kappa_estimates["transfer"] = {"kappa": kappa, "rel_std": rel}

    def _rescale(self) -> None:
        """Fuse the similarity-model kappa of all clean samples since the reference with any prior."""
        estimates = [] if self._kappa_prior is None else [self._kappa_prior]
        if self._samples:
            baseline = max(float(np.linalg.norm(dp)) for _, dp in self._samples)
            kappa = None
            if baseline >= self.MIN_KAPPA_BASELINE_M:
                kappa = ImageJacobian.similarity_kappa(
                    self._samples, reflection=bool(np.linalg.det(self.jacobian.J) < 0))
            if kappa is not None:
                rel = math.sqrt(2.0) * PIXEL_NOISE_PX / (kappa * baseline)
                estimates.append((kappa, rel))
                self.kappa_estimates["samples"] = {"kappa": kappa, "rel_std": rel, "baseline_m": baseline,
                                                   "count": len(self._samples)}
        if not estimates:
            return
        kappa, rel = _fuse(estimates)
        ratio = kappa / self.jacobian.kappa
        if math.isfinite(ratio) and ratio > 0:
            # Clip, not skip: large corrections converge over a few updates while an outlier moves kappa
            # by at most MAX_KAPPA_STEP_RATIO per update.
            ratio = float(np.clip(ratio, 1 / self.MAX_KAPPA_STEP_RATIO, self.MAX_KAPPA_STEP_RATIO))
            self.jacobian.rescale(self.jacobian.kappa * ratio)
            self.jacobian.kappa_rel_std = rel
            self.kappa_estimates["fused"] = {"kappa": kappa, "rel_std": rel, "applied_ratio": ratio}

    # ------------------------------------------------------------- commands
    def _approach_hold(self, view: StepView) -> np.ndarray:
        error = self.approach_anchor - float(view.position @ self.frame_R[:, 2])
        return float(np.clip(error, -self.APPROACH_HOLD_CAP_M, self.APPROACH_HOLD_CAP_M)) * self.frame_R[:, 2]

    def _next(self, view):
        if self.phase in ("point_coarse", "point_fine"):
            raise RuntimeError("Align needs a pointing answer before the next action")
        if not np.array_equal(view.state, self._frame.view.state):
            self._frame = self._observe_frame(view)
        delta = self._approach_hold(view)
        if self.phase == "probe_settle":
            anchor = self._settle["anchor"]
            offset = view.position - anchor
            lateral = offset - float(offset @ self.frame_R[:, 2]) * self.frame_R[:, 2]
            delta = delta + _capped(-lateral, self.LATERAL_HOLD_CAP_M)
        elif self.phase == "probe":
            probe = self._probe
            perp_error = -float((view.position - probe["start"]) @ probe["perp"])
            delta = (delta + self.PROBE_COMMAND_M * probe["axis"]
                     + float(np.clip(perp_error, -self.LATERAL_HOLD_CAP_M, self.LATERAL_HOLD_CAP_M)) * probe["perp"])
        elif self.phase == "servo" and self._frame.uv is not None and self.last_setpoint is not None:
            remaining = self.jacobian.solve(self._frame.uv - self.last_setpoint)
            dp = (self.gain / self.realization) * remaining
            cap = float(np.clip(self.far_fraction * float(np.linalg.norm(remaining)),
                                self.max_command_m, self.far_command_m))
            norm = float(np.linalg.norm(dp))
            if norm > cap:
                dp *= cap / norm
            delta = delta + dp[0] * self.frame_R[:, 1] + dp[1] * self.frame_R[:, 0]
        return self.adapter.adapt(delta, self.hold_xyzw, self.gripper, view.state)

    # ------------------------------------------------------------- feedback
    def _observe(self, before, action, after):
        previous = self._frame
        commanded = np.asarray(action[:3], dtype=float) * 0.05
        commanded_inplane = np.array([commanded @ self.frame_R[:, 1], commanded @ self.frame_R[:, 0]])
        frame = self._observe_frame(after, retemplate=self.phase == "servo", retemplate_floor=self.RETEMPLATE_MIN_PEAK)
        self._frame = frame
        step = frame.inplane - previous.inplane
        if float(np.linalg.norm(commanded_inplane)) > 1e-4:
            self.realized.append(float(step @ commanded_inplane) / float(commanded_inplane @ commanded_inplane))
        phase = self.phase
        if self.lost_frames >= self.LOST_FRAMES:
            self._on_lost()
        elif phase == "probe_settle":
            self._observe_settle(previous, frame)
        elif phase == "probe":
            self._observe_probe(frame)
        elif phase == "servo":
            self._observe_servo(frame)
        return {"phase": phase, "mark_uv": frame.uv, "peak": frame.peak,
                "setpoint_uv": self.last_setpoint, "residual_px": self._residual(),
                "inplane_step_m": step, "approach_drift_m": frame.along - self.approach_anchor,
                "kappa": self._kappa(), "lost_frames": self.lost_frames, "repoints": self.repoints,
                "pointing_request": self.pointing_request}

    def _observe_probe(self, frame: _Frame) -> None:
        probe = self._probe
        probe["steps"] += 1
        self.probe_steps += 1
        if frame.uv is not None:
            self._probe_frames.append(frame)
        moved = float((frame.view.position - probe["start"]) @ probe["axis"])
        if moved < self.PROBE_FRACTION * self.probe_m and probe["steps"] < self.PROBE_MAX_STEPS:
            return
        self.probe_record.append({"axis": "R1" if probe["index"] == 0 else "R0", "moved_m": moved,
                                  "steps": probe["steps"]})
        if probe["index"] == 0:
            self._probe = {"index": 1, "axis": self.frame_R[:, 0].copy(), "perp": self.frame_R[:, 1].copy(),
                           "start": frame.view.position.copy(), "steps": 0}
            return
        self._fit_probes()

    def _fit_probes(self) -> None:
        origin = self._probe_frames[0]
        if min(record["moved_m"] for record in self.probe_record) < self.PROBE_MIN_FRACTION * self.probe_m:
            self._fail("probe_failed", detail="insufficient_probe_motion")
            return
        samples = [(f.uv - origin.uv, f.inplane - origin.inplane) for f in self._probe_frames[1:]
                   if quaternion_angle_rad(f.quaternion, origin.quaternion) <= self.CLEAN_ROTATION_RAD]
        try:
            self.jacobian.set_from_probes(samples)
        except ValueError as exc:
            # Residual arm rotation (e.g. right after an Orient) can leave too few clean probe frames;
            # restart the probe from the current frame instead of failing (bounded).
            restarts = getattr(self, "_probe_restarts", 0)
            if restarts < self.MAX_PROBE_RESTARTS:
                self._probe_restarts = restarts + 1
                self._start_probe()
                return
            self._fail("probe_failed", detail=str(exc))
            return
        self.kappa_estimates["probe_lstsq"] = {"kappa": self.jacobian.kappa}
        deviation = self.jacobian.model_deviation()
        if deviation > self.MAX_MODEL_DEVIATION:
            self._fail("probe_failed", detail="jacobian_inconsistent_with_mount", model_deviation=deviation)
            return
        self.probed = True
        self._reference, self._samples = origin, list(samples)
        self._rescale()
        self._start_servo()

    def _observe_servo(self, frame: _Frame) -> None:
        self.servo_steps += 1
        if frame.uv is not None:
            self._secant_update(frame)
        self._check_servo_frame(frame)
        if not self.done and self.servo_steps >= self.max_servo_steps:
            self._fail("align_no_convergence")

    def _secant_update(self, frame: _Frame) -> None:
        reference = self._reference
        if quaternion_angle_rad(frame.quaternion, reference.quaternion) > self.CLEAN_ROTATION_RAD:
            # Keep the long baseline: resetting on every slightly rotated frame refit kappa on short
            # baselines during fast moves (smoke 006). Rotated frames are simply skipped.
            return
        dp = frame.inplane - reference.inplane
        if abs(frame.along - reference.along) > max(self.min_step_m, 0.25 * float(np.linalg.norm(dp))):
            return
        ds = frame.uv - reference.uv
        if self.jacobian.broyden(ds, dp, min_step_m=self.min_step_m):
            self._samples.append((ds, dp))
            self._rescale()

    # ------------------------------------------------------------- evidence
    def _residual(self) -> float | None:
        if self.last_uv is None or self.last_setpoint is None:
            return None
        return float(np.linalg.norm(self.last_uv - self.last_setpoint))

    def _evidence_now(self, **extra) -> dict:
        kappa, residual = self._kappa(), self._residual()
        frame = getattr(self, "_frame", None)
        evidence = {
            "mark_uv": self.last_uv, "setpoint_uv": self.last_setpoint, "residual_px": residual,
            "residual_mm_est": None if residual is None or not kappa else residual / kappa * 1000.0,
            "kappa": kappa, "J": self.jacobian.J.tolist() if self.jacobian.ready else None,
            "repoints": self.repoints, "probe_steps": self.probe_steps, "servo_steps": self.servo_steps,
            "settle_steps": self.settle_steps, "settle_record": self.settle_record,
            "offset_m": self.offset_m, "phase": self.phase, "peak": self.last_peak,
            "tolerance_px": self.tolerance_px, "tolerance_px_effective": self._tolerance(),
            "setpoint_row_limit": self.setpoint_row_limit,
            "jacobian": self.jacobian.to_dict(), "jacobian_input": self.jacobian_input,
            "kappa_transfer": self.kappa_transfer, "kappa_estimates": self.kappa_estimates,
            "probe_record": self.probe_record, "pointed_uv": self.pointed_uv, "cell_id": self.cell_id,
            "lost_frames_total": self.lost_frames_total,
            "residual_history_px": self.residual_history[-12:],
            "realization_measured": float(np.median(self.realized)) if self.realized else None,
            "realization_assumed": self.realization, "gain": self.gain,
            "max_approach_drift_m": self.max_approach_drift_m,
            "start_xyz": self.start_xyz, "end_xyz": None if frame is None else frame.view.position,
            "track": self.track.to_dict() if self.track is not None else None,
        }
        evidence.update(extra)
        return evidence


# --------------------------------------------------------------- monitoring
class WristPatchMonitor:
    """Secondary hold evidence: a wrist patch next to the fingers must stay put.

    A carried object stays fixed in the wrist view; one left behind recedes.
    The patch defaults to (cu, 0.70 H) upright. A textureless patch makes the
    monitor unavailable for its lifetime.
    """

    def __init__(self, view: StepView, *, uv=None, radius=10, search=24, max_displacement_px=10.0, min_peak=0.6):
        _view_state(view)
        gray = view.wrist_gray_upright
        height, width = gray.shape
        self.uv0 = self._textured_default(gray, radius) if uv is None else _vec2(uv, "uv")
        self.max_displacement_px = _real(max_displacement_px, "max_displacement_px")
        self.track: NCCTrack | None = None
        self.unavailable_reason: str | None = None
        try:
            self.track = NCCTrack(gray, self.uv0, radius=radius, search=search, min_peak=min_peak)
        except ValueError as exc:
            self.unavailable_reason = str(exc)
        self.observations = 0

    @staticmethod
    def _textured_default(gray, radius) -> np.ndarray:
        """Highest-contrast patch in rows 0.55-0.72 H, columns cu +/- 60 (real table frames at (cu, 0.70 H)
        were textureless, smoke 006)."""
        height, width = gray.shape
        cu = (width - 1) / 2.0
        best, best_std = np.array([cu, 0.70 * height]), -1.0
        for v in range(int(0.55 * height), int(0.72 * height) + 1, 4):
            for u in range(int(cu - 60), int(cu + 60) + 1, 4):
                if radius <= v < height - radius and radius <= u < width - radius:
                    std = float(gray[v - radius:v + radius + 1, u - radius:u + radius + 1].std())
                    if std > best_std:
                        best, best_std = np.array([float(u), float(v)]), std
        return best

    @property
    def available(self) -> bool:
        return self.track is not None

    def observe(self, view) -> dict:
        self.observations += 1
        if self.track is None:
            return {"available": False, "ok": None, "peak": None, "displacement_px": None}
        uv, peak = self.track.match(view.wrist_gray_upright)
        if uv is None:
            return {"available": True, "ok": False, "peak": float(peak), "displacement_px": None}
        displacement = float(np.linalg.norm(uv - self.uv0))
        return {"available": True, "ok": bool(displacement <= self.max_displacement_px), "peak": float(peak),
                "displacement_px": displacement}
