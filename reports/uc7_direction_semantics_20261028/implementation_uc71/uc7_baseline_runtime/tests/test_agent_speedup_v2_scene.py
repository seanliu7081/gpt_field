"""BS scene tests (contract §C6 T1): fusion and D7 fills, registration on FakeTwoViewSim scenes, plan seeds,
provisional and partial registration, region masks, the hand-swept mask and the HeldFrameEstimator.

Geometry (BG) is replaced by ``STANDIN`` (contract-faithful re-implementations of the few functions the scene calls)
unless the real ``agent_speedup_v2_geometry`` module exists and exposes them (Wave-1 acceptance runs these tests with
the real module). The calibration products (camera model, external projector, hand locator) are always test
stand-ins built from the fake's own robot and camera geometry, and depth samples are synthesized from the fake's
depth map like a sparse kappa survey: one sample per textured grid patch (D15), patches across a depth step rejected
(D1 residual), 1.5 % depth noise (D4). Evaluator truth is used here only to build inputs and to score."""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from gpt_field import agent_speedup_v2_scene as scene_mod
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field.agent_speedup_v2_types import (DepthSample, FeatureRecord, HandPose, KeepOut, Meas, Plan, PlanPart,
                                              PlanStep, RegistrationFailure, RobotModel, Seed, SurveyResult,
                                              V2Calibration, WristCamera, poly_iou)
from gpt_field.contact_skills import GripperCalibration, StepView
from gpt_field.mark_prompting import coarse_cell_center, rotate_pixel, unrotate_pixel
from gpt_field.wrist_servo import NCCTrack

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
F_PX = 333.6
DEPTH_NOISE = 0.015          # D4-like relative depth sigma of a kappa sample
GRID_SPACING_PX = 36         # FT_GRID_SPACING_PX (geometry)
PATCH_HALF_PX = 10           # FT_RADIUS_PX
MIN_STD = 4.0                # D15
SEED_SEARCH_PX = 8           # RELIEF_SEARCH_PX (geometry)


# ------------------------------------------------------------------------------------------ geometry stand-in
def _structure_axis(gray, uv, half_px=12):
    g = np.asarray(gray, dtype=float)
    u0, v0 = int(round(uv[0])), int(round(uv[1]))
    r = int(half_px)
    patch = g[max(v0 - r, 0):v0 + r + 1, max(u0 - r, 0):u0 + r + 1]
    gy, gx = np.gradient(patch)
    t = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    w, v = np.linalg.eigh(t)
    coherence = float((w[1] - w[0]) / max(w[1] + w[0], 1e-9))
    across = v[:, 1]
    return np.array([-across[1], across[0]]), coherence, math.sqrt(max(float(w[1]), 0.0) / max(patch.size, 1))


def _to_lab255(rgb):
    return cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), cv2.COLOR_RGB2LAB)


def _refine_region_mask(external_rgb, affine_mask, seed_uv, *, sigma_px):
    """Lab region growing from the seed inside the affine mask dilated by 3 sigma (contract §C5 [C-51])."""
    lab = _to_lab255(external_rgb).astype(np.float32)
    allowed = cv2.dilate(affine_mask.astype(np.uint8), np.ones((2 * int(math.ceil(3 * sigma_px)) + 1,) * 2,
                                                               np.uint8)) > 0
    core = cv2.erode(affine_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ref = lab[core] if core.any() else lab[affine_mask]
    mean, std = np.median(ref, axis=0), np.maximum(ref.std(axis=0), 2.0)
    close = (np.abs(lab - mean[None, None, :]) <= 3.0 * std[None, None, :]).all(axis=-1) & allowed
    close = cv2.morphologyEx(close.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    n, comps = cv2.connectedComponents(close.astype(np.uint8), connectivity=4)
    u, v = int(round(seed_uv[0])), int(round(seed_uv[1]))
    label = comps[v, u] if 0 <= v < comps.shape[0] and 0 <= u < comps.shape[1] else 0
    mask = comps == label if label else affine_mask
    inter, union = np.count_nonzero(mask & affine_mask), np.count_nonzero(mask | affine_mask)
    return mask, {"iou_vs_affine": inter / max(union, 1), "pixels": int(mask.sum())}


def _reacquire(track, gray, predicted_uv, *, yaw_change_rad=None, kappa_ratio=None, sigma_px=None):
    t = __import__("copy").deepcopy(track)
    t._center = np.asarray(predicted_uv, float)
    uv, peak = t.match(gray, scales=NCCTrack.REACQUIRE_SCALES)
    return (None if uv is None else np.asarray(uv, float)), float(peak if peak is not None else -1.0)


# G-7 orientation kernels (design review 2 decision §2.2, BG): contract-faithful restatements of the axis lens's
# estimators (orientation_peak, mask_outline, edge_hist with the region / hull ROI forms), used while the geometry
# module does not define them yet (BG's parallel build); the scene calls vision.outline_orientation(mask, pose, z, f)
# and vision.edge_orientation(grey, roi_mask, exclude_mask, pose, z, f)
ORIENT_SMOOTH_DEG = 3.0              # ORIENT_SMOOTH_DEG (geometry, v7)
OUTLINE_TANGENT_PTS = 4              # OUTLINE_TANGENT_PTS
EDGE_TOP_FRACTION = 0.25             # D10_EDGE_TOP_FRACTION
EDGE_BLUR_PX = 1.0                   # D10_EDGE_BLUR_PX
PEAK_WINDOW_DEG = 7.5                # YAW_GRID_RAD / 2


def _backproject(pose, uv, z, f):
    return scene_mod._backproject_z(WristCamera(f), pose, np.asarray(uv, float).reshape(-1, 2), z)


def _orientation_peak(dirs_xy, weights):
    d = np.asarray(dirs_xy, float).reshape(-1, 2)
    w = np.asarray(weights, float).reshape(-1)
    ok = np.isfinite(d).all(axis=1) & np.isfinite(w)
    d, w = d[ok], w[ok]
    if len(d) < 3 or w.sum() <= 0:
        return None, 0.0, None
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    hist, _ = np.histogram(ang, bins=180, range=(0, 180), weights=w)
    x = np.arange(-30, 31)
    kern = np.exp(-0.5 * (x / ORIENT_SMOOTH_DEG) ** 2)
    sm = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(np.roll(np.pad(kern, (0, 180 - len(kern))), -30))))
    peak = int(np.argmax(sm))
    dd = (ang - peak + 90.0) % 180.0 - 90.0
    near = np.abs(dd) <= PEAK_WINDOW_DEG
    share = float(w[near].sum() / w.sum())
    mean = peak + float(np.average(dd[near], weights=w[near])) if near.any() else float(peak)
    far = np.abs(((np.arange(180) - peak + 90) % 180) - 90) > 2 * PEAK_WINDOW_DEG
    second = float(sm[far].max() / max(sm[peak], 1e-12)) if far.any() else 0.0
    a = math.radians(mean)
    return scene_mod._canonical_axis([math.cos(a), math.sin(a)]), share, second


def _outline_orientation(mask, pose, z, f):
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(c) < 12:
        return None
    w = _backproject(pose, c, z, f)[:, :2]
    k = OUTLINE_TANGENT_PTS
    dirs = np.roll(w, -k, axis=0) - np.roll(w, k, axis=0)
    weights = np.linalg.norm(dirs, axis=1) / 2.0
    dirs = dirs / np.maximum(weights[:, None] * 2.0, 1e-12)
    axis, share, second = _orientation_peak(dirs, weights)
    if axis is None:
        return None
    n = np.array([-axis[1], axis[0]])
    ang = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    a0 = math.degrees(math.atan2(axis[1], axis[0])) % 180.0
    near = np.abs((ang - a0 + 90) % 180 - 90) <= PEAK_WINDOW_DEG
    across = (w - np.nanmean(w, axis=0)) @ n
    two_sided = bool((near & (across > 0)).sum() >= 3 and (near & (across < 0)).sum() >= 3)
    return {"axis": axis, "share": share, "second": second, "two_sided": two_sided, "n": int(len(c))}


def _edge_orientation(grey, roi_mask, exclude_mask, pose, z, f):
    g = cv2.GaussianBlur(np.asarray(grey, np.float32), (0, 0), EDGE_BLUR_PX)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    m = np.asarray(roi_mask, bool) & ~np.asarray(exclude_mask, bool)
    if m.sum() < 20:
        return None
    thr = np.percentile(mag[m], 100 * (1 - EDGE_TOP_FRACTION))
    sel = m & (mag >= max(thr, 1e-6))
    vs, us = np.nonzero(sel)
    if len(us) < 10:
        return None
    ex, ey = -gy[vs, us], gx[vs, us]
    nrm = np.hypot(ex, ey) + 1e-12
    ex, ey = ex / nrm, ey / nrm
    p0 = _backproject(pose, np.column_stack([us - 3 * ex, vs - 3 * ey]), z, f)
    p1 = _backproject(pose, np.column_stack([us + 3 * ex, vs + 3 * ey]), z, f)
    axis, share, second = _orientation_peak(p1[:, :2] - p0[:, :2], mag[vs, us])
    if axis is None:
        return None
    return {"axis": axis, "share": share, "second": second, "n": int(len(us))}


KERNELS = {"outline_orientation": _outline_orientation, "edge_orientation": _edge_orientation,
           "orientation_peak": _orientation_peak}
STANDIN = SimpleNamespace(structure_axis=_structure_axis, to_lab255=_to_lab255, refine_region_mask=_refine_region_mask,
                          reacquire=_reacquire, RELIEF_OFFSET_M=0.015, **KERNELS)
_GEOMETRY_PATH = Path(th.PACKAGE_DIR) / "agent_speedup_v2_geometry.py"
_NEEDED = ("structure_axis", "to_lab255", "refine_region_mask", "reacquire", "RELIEF_OFFSET_M")


class _WithKernels:
    """The real geometry module, with the G-7 stand-in kernels for the ones it does not define yet."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        if name in KERNELS and not hasattr(self._module, name):
            return KERNELS[name]
        return getattr(self._module, name)


def _vision():
    if _GEOMETRY_PATH.is_file():
        try:
            from gpt_field import agent_speedup_v2_geometry as geometry
        except Exception:                                   # a half-built module during the parallel build
            return STANDIN
        if all(hasattr(geometry, name) for name in _NEEDED):
            return geometry if all(hasattr(geometry, name) for name in KERNELS) else _WithKernels(geometry)
    return STANDIN


VISION = _vision()
_VISION_KEYS = _NEEDED + tuple(KERNELS)


# --------------------------------------------------------------------------------- calibration stand-ins
class StandInCamera:
    """CameraModel duck type: f_used and wrist()."""

    def __init__(self, f_px=F_PX):
        self.f_used, self.sigma_f, self.source, self.f_hat, self.disagreement = f_px, 0.0, "R3", None, False

    def wrist(self):
        return WristCamera(self.f_used)


class AffineProjector:
    """ExternalProjector duck type: affine u = u_a + J (X - X_a), J = the fake external camera's Jacobian at the
    anchor (what a perfect R-1x/R-1v fit gives), sigma = sigma_px + 3 % of the predicted shift (the affine term)."""

    def __init__(self, camera, anchor_grip, sigma_px=2.0):
        self.camera = camera
        self.anchor_grip = np.asarray(anchor_grip, float)
        self.anchor_uv = np.asarray(camera.project(self.anchor_grip)[0][0], float)
        self.model = SimpleNamespace(J=camera.jacobian(self.anchor_grip), ray_dir=None)
        self.sigma_px = float(sigma_px)

    def project(self, xyz):
        shift = self.model.J @ (np.asarray(xyz, float) - self.anchor_grip)
        return self.anchor_uv + shift, self.sigma_px + 0.03 * float(np.linalg.norm(shift))

    def project_polygon(self, poly_xyz):
        pts = np.asarray(poly_xyz, float).reshape(-1, 3)
        uv = np.array([self.project(p)[0] for p in pts])
        return uv, max(self.project(p)[1] for p in pts)

    def mask(self, poly_xyz, shape=None, erode_px=None):
        shape = (512, 512) if shape is None else shape
        uv, _ = self.project_polygon(poly_xyz)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(uv).astype(np.int32)], 1)
        return mask > 0

    def reanchored(self, anchor_uv, anchor_grip, sigma_px):
        out = AffineProjector(self.camera, anchor_grip, sigma_px)
        out.anchor_uv = np.asarray(anchor_uv, float)
        return out


class StandInLocator:
    """HandLocator duck type: the fake external camera's projection of the grip (a perfect locator)."""

    def __init__(self, camera, sigma_px=2.0):
        self.camera, self.sigma_loc_px = camera, float(sigma_px)

    def locate(self, before_rgb, after_rgb, pose, *, after_pose=None):
        return np.asarray(self.camera.project(pose.grip)[0][0], float), self.sigma_loc_px


# ------------------------------------------------------------------------------------------------ builders
def make_world(scene, eef, **kw):
    return fakes.FakeTwoViewSim(scene, eef=eef, **kw)


def view_of(world):
    return StepView(world.observation())


def fingertip_band(world):
    """Free-air close difference at the current pose (as geometry.fingertip_band_mask would measure it)."""
    open_rgb = world.wrist_upright().astype(int)
    fakes.scripted_gripper(world, 1.0, 40)
    closed = world.wrist_upright().astype(int)
    fakes.scripted_gripper(world, -1.0, 40)
    diff = np.abs(open_rgb - closed).max(axis=-1) > 12
    return cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def synthetic_survey(world, view, *, step=0, spacing=GRID_SPACING_PX, seeds=(), noise=DEPTH_NOISE, rng=None,
                     exclude=None):
    """A sparse kappa survey of the view (SurveyRecorder stand-in): DepthSamples at the textured grid patches."""
    rng = np.random.default_rng(0) if rng is None else rng
    gray = np.asarray(view.wrist_gray_upright, float)
    depth = world.depth_map("wrist")
    pose = HandPose.from_view(view)
    cam = WristCamera(F_PX)
    h, w = gray.shape
    r = PATCH_HALF_PX
    pts = [(float(u), float(v), None) for v in range(spacing // 2, h - r, spacing)
           for u in range(spacing // 2, w - r, spacing)]
    for key, (u, v) in seeds:                 # a seed / relief point takes the highest-contrast patch within +-8 px
        best = None
        for dv in range(-SEED_SEARCH_PX, SEED_SEARCH_PX + 1, 2):
            for du in range(-SEED_SEARCH_PX, SEED_SEARCH_PX + 1, 2):
                ui, vi = int(round(u + du)), int(round(v + dv))
                if r <= ui < w - r and r <= vi < h - r:
                    std = gray[vi - r:vi + r + 1, ui - r:ui + r + 1].std()
                    if best is None or std > best[0]:
                        best = (std, ui, vi)
        if best is not None:
            pts.append((float(best[1]), float(best[2]), key))
    features, samples = [], []
    for fid, (u, v, key) in enumerate(pts):
        ui, vi = int(round(u)), int(round(v))
        if not (r <= ui < w - r and r <= vi < h - r):
            continue
        if exclude is not None and exclude[vi, ui]:
            continue
        patch = gray[vi - r:vi + r + 1, ui - r:ui + r + 1]
        zpatch = depth[vi - r:vi + r + 1, ui - r:ui + r + 1]
        if patch.std() < MIN_STD or not np.isfinite(zpatch).all():
            continue
        z = float(depth[vi, ui])
        if (zpatch.max() - zpatch.min()) / z > 0.01:              # straddles a depth step
            front = zpatch <= zpatch.min() * 1.01
            fg, bg = patch[front], patch[~front]
            # an occluding contour whose contrast dominates both sides' texture moves with the front surface (the
            # NCC peak follows the edge); otherwise the two motions disagree: D1 residual reject
            if abs(fg.mean() - bg.mean()) < 2.0 * max(fg.std(), bg.std()):
                continue
            z = float(zpatch.min())
        zn = z * (1.0 + noise * rng.standard_normal())
        xyz = cam.backproject(pose, (u, v), zn)
        samples.append(DepthSample(tuple(xyz), noise * z, noise * z * 0.5, "kappa", (u, v), fid, step, F_PX / zn))
        features.append(FeatureRecord(fid, key, (u, v), (u, v), 5, F_PX / zn, None, zn, noise * z, tuple(xyz), True,
                                      None))
    state = tuple(float(x) for x in view.state)
    return SurveyResult(state, state, {"probe": 5}, features, samples, None, [], {})


def make_cal(world, view, survey, *, band=None, quarter_turns=2):
    ext = world.external
    return V2Calibration(
        gripper=GripperCalibration.default(), sigma_close_mm=0.05, sigma_g_mm_per_step=0.002,
        free_yaw_rate_rad_per_step=0.0229, camera=StandInCamera(), external=None, locator=StandInLocator(ext),
        projector=AffineProjector(ext, HandPose.from_view(view).grip),
        fingertip_band=fingertip_band(world) if band is None else band, false_change={},
        sigma_px={"wrist": 0.35, "external": 0.35}, sigma_static_grey=0.35, quarter_turns=quarter_turns,
        start_pose=tuple(view.state), start_quaternion=tuple(view.quaternion_xyzw), survey=survey, mount={},
        thresholds=[])


def probe(world, scene, *, amplitude=0.02):
    """The D2 hover probe (+-a along R1 then R0, out and back), every reached pose shown to scene.on_view."""
    home = world.grip.copy()
    R = world.R
    for axis in (R[:, 1], R[:, 0]):
        for target in (home + amplitude * axis, home, home - amplitude * axis, home):
            fakes.scripted_goto(world, target, tol_m=0.0004)
            scene.on_view(view_of(world))
    return view_of(world)


def plan_of(parts, steps=()):
    return Plan("scene", tuple(parts), (), tuple(steps), "plan")


def started_scene(world, *, survey_seeds=(), band=None, quarter_turns=2, probe_first=None):
    """scene.start at the world's view; ``probe_first`` = amplitude of a Calibrate+-like probe run before start (its
    views cached by on_view, as the policy does during Calibrate+)."""
    scene = scene_mod.SceneModel(RobotModel(), vision=VISION)
    scene.on_view(view_of(world))
    if probe_first:
        probe(world, scene, amplitude=probe_first)
    view = view_of(world)
    scene.on_view(view)
    survey = synthetic_survey(world, view, seeds=survey_seeds)
    cal = make_cal(world, view, survey, band=band, quarter_turns=quarter_turns)
    info = scene.start(cal, view)
    return scene, view, cal, info


def seed_at(world, view, xyz, *, source="fine", quantization=8.0):
    uv, _ = world.project("wrist", xyz)
    return Seed("wrist", (float(uv[0]), float(uv[1])), 0, source, quantization, tuple(view.state))


def angle_deg(a, b):
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def register_at_hover(scene_fn, eef, target, pid, *, roles=("grip",), move="free", extra_seeds=(), amplitude=0.02,
                      world_kw=None, dense=False):
    """Start the scene at a hover, run the D2 probe (cached views for the parallax band test), register ``pid`` from
    a fine seed at the truth point ``target`` (the pointed grasp point). ``dense``: the hover survey also carries the
    probe tracker's dense lattice around the pointed pixel (``dense_survey``), as every v2 hover does."""
    sc = scene_fn()
    world = make_world(sc, eef, **(world_kw or {}))
    uv0 = world.project("wrist", target)[0]
    seeds = [(pid, tuple(uv0))] + [(key, tuple(world.project("wrist", xyz)[0])) for key, xyz in extra_seeds]
    scene, view, cal, info = started_scene(world, survey_seeds=seeds)
    after = probe(world, scene, amplitude=amplitude)
    if dense:
        uv = np.asarray(world.project("wrist", target)[0], float)
        scene.absorb_survey(dense_survey(world, after, uv, rng=np.random.default_rng(9)), after)
    scene.parts[pid] = scene_mod.PartRecord(pid, "part", move, frozenset(roles))
    rec = scene.register(pid, after, seed=seed_at(world, after, target), empty_hand=True)
    return SimpleNamespace(world=world, scene=scene, view=after, cal=cal, info=info, rec=rec, sc=sc)


# ============================================================================================== lint and constants
def test_scene_module_passes_the_t0_lint_audit_and_firewall():
    path = Path(scene_mod.__file__)
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(path, vocab) == []
    assert th.literal_audit(path) == []
    assert "agent_speedup_v2_scene.py" in th.CONTROL_MODULES and not th.LABEL_MERGE_ERRORS
    tree = ast.parse(path.read_text())
    lazy = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any("geometry" in a.name for a in n.names)]
    assert lazy and all(ast.unparse(n) == th.LAZY_IMPORT_STATEMENT for n in lazy)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("geometry" in ast.unparse(n) for n in top)


def test_contract_thresholds_and_restated_v1_constants():
    from gpt_field import wrist_servo
    owned = {"D7_PLANE_RADIUS_M": 0.02, "D7_COLOUR_SIGMA_K": 3.0, "D7_PLANE_SIGMA_K": 3.0, "D7_OBSTACLE_RADIUS_M": 0.02,
             "D7_OBSTACLE_SIGMA_K": 3.0, "D8_FLOOR_M": 0.003, "D8_SIGMA_K": 3.0, "D9_GRADIENT_RATIO": 4.0,
             "D9_DEPTH_SIGMA_K": 3.0, "D10_MIN_COHERENCE": 0.5, "D10_MIN_ASPECT": 1.5, "D11_RMS_FLOOR_PX": 1.5,
             "D11_RMS_FRACTION": 0.03, "M14_GAP_FLOOR_MM": 1.0, "M14_GAP_SIGMA_K": 6.0, "M14_IOU_MIN": 0.8,
             "M14_SHIFT_FLOOR_PX": 4.0, "M14_SHIFT_SIGMA_K": 3.0, "M17_FRACTION": 0.5, "TOP_PERCENTILE": 95,
             "RIGID_PATCH_MIN_DIST_M": 0.015}
    for name, value in owned.items():
        assert getattr(scene_mod, name) == value, name
        assert f"agent_speedup_v2_scene.{name}" in th.CONSTANT_LABELS, name
    assert scene_mod.TRACK_SIGMA_PX == wrist_servo.PIXEL_NOISE_PX
    assert scene_mod.D10_PATCH_MIN_HALF_PX == wrist_servo.BAR_CHECK_HALF_PX
    assert scene_mod.D10_MIN_COHERENCE == wrist_servo.BAR_CHECK_MIN_COHERENCE
    import inspect
    assert scene_mod.NCC_RADIUS_PX == inspect.signature(NCCTrack).parameters["radius"].default == 10
    from gpt_field.agent_speedup_v2_types import AXIS_AGREE_MAX_RAD, HF_CELL_M, YAW_GRID_RAD
    # design review 1: the 15 deg "region axis stands" rule is replaced by the S-4' consensus at YAW_GRID_RAD / 2
    assert not hasattr(scene_mod, "D10_AXIS_AGREE_RAD") and abs(AXIS_AGREE_MAX_RAD - YAW_GRID_RAD / 2) < 1e-12
    v6 = {"TOP_WINDOW_PX": 40, "TOP_GATE_M": 0.002, "TOP_BAND_K": 3, "TOP_BAND_SIGMA_K": 2.0, "TOP_BAND_SUPPORT_N": 10,
          "TOP_MODEL_M": 0.003, "TOP_MODEL_THIN_M": 0.005, "D10_MIN_REGION_CELLS": 4, "PB_REACH_M": 0.05,
          "D7_PARENT_SIGMA_CAP_M": 0.005}                      # decision §3 (THRESHOLDS_VERSION v6)
    for name, value in v6.items():
        assert getattr(scene_mod, name) == value, name
    assert scene_mod.TOP_BAND_K == scene_mod.BAND_MIN_SAMPLES and scene_mod.D7_PARENT_SIGMA_CAP_M == HF_CELL_M
    status = th.v6_constant_status()
    mine = {k: v for k, v in status.items() if v["owner"] == "agent_speedup_v2_scene"}
    assert set(mine) == set(v6) and all(v["state"] in ("labelled", "alias") for v in mine.values()), mine
    # r2close lead ruling 3 adds S15_SEED_SIGMA_K; B0 registers it and may bump THRESHOLDS_VERSION to v8
    assert th.THRESHOLDS_VERSION in ("agent-speedup-v2-thresholds-v7", "agent-speedup-v2-thresholds-v8")
    assert scene_mod.SCENE_VERSION == "agent-speedup-v2-scene-7"
    assert scene_mod.S15_SEED_SIGMA_K == 2.0
    entry = th.CONSTANT_LABELS["agent_speedup_v2_scene.S15_SEED_SIGMA_K"]
    assert entry["label"] == "DD" and "ruling 3" in entry["derivation"]
    assert "ruling 4 (a)" in th.CONSTANT_LABELS["agent_speedup_v2_scene.D10_OUTLINE_SECOND_MAX"]["derivation"]
    v7 = {"D10_OUTLINE_SHARE_MIN": 0.20, "D10_OUTLINE_SECOND_MAX": 0.8, "D10_EDGE_SHARE_MIN": 0.20,
          "D10_EDGE_REGION_DILATE_PX": 6, "D10_EDGE_HULL_DILATE_PX": 8, "REGION_COLLAPSE_RATIO": 2.0,
          "COLLAPSE_HULL_DILATE_PX": 2}                       # decision 2 §3 (THRESHOLDS_VERSION v7)
    for name, value in v7.items():
        assert getattr(scene_mod, name) == value, name
    status7 = {k: v for k, v in th.v7_constant_status().items() if v["owner"] == "agent_speedup_v2_scene"}
    assert set(v7) <= set(status7) and all(v["state"] == "labelled" for v in status7.values()), status7
    # the retired S-4' deciders stay defined (logged as diagnosis; the planner aliases two of them)
    assert all(hasattr(scene_mod, name) for name in th.V7_RETIRED_DECIDERS)
    observers = Path(th.PACKAGE_DIR) / "agent_speedup_v2_observers.py"
    if observers.is_file():
        o = ast.parse(observers.read_text())
        values = [n.value.value for n in o.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                  and any(isinstance(t, ast.Name) and t.id == "O1_HAND_DILATE_PX" for t in n.targets)]
        if values:
            assert scene_mod.SWEPT_DILATE_PX == values[0]


def test_height_bands_keep_a_small_band_beside_a_large_one():
    rng = np.random.default_rng(3)
    z = np.r_[0.900 + 0.0034 * rng.standard_normal(60), 0.918 + 0.003 * rng.standard_normal(7),
              0.9304 + 0.003 * rng.standard_normal(43), np.full(500, 0.8998), np.full(300, 0.9304)]
    s = np.r_[np.full(60, 0.0034), np.full(7, 0.003), np.full(43, 0.003), np.full(800, 0.001)]
    kind = np.r_[np.zeros(110, int), np.ones(800, int)]
    bands = scene_mod._height_bands(z, s, np.ones(len(z)), kind)
    heights = [b["z"] for b in bands]
    assert len(bands) == 3 and abs(heights[1] - 0.918) <= 0.003, heights
    assert bands[0]["plane"] and bands[2]["plane"] and not bands[1]["plane"]
    assert scene_mod._height_bands(np.zeros(0), np.zeros(0), np.zeros(0)) == []


# ================================================================================================== 1. fusion
@pytest.fixture(scope="module")
def aabb_scene():
    sc = fakes.textureless_dark_aabb_scene()
    world = make_world(sc, (-0.20, 0.019, 1.19))
    scene, view, cal, info = started_scene(world)
    return SimpleNamespace(world=world, scene=scene, view=view, cal=cal, info=info, sc=sc)


def _cells_in(hf, poly, shrink=0.0):
    poly = np.asarray(poly, float)
    if shrink:
        c = poly.mean(axis=0)
        poly = c + (poly - c) * (1.0 - shrink)
    return hf.cells_in_polygon(poly)


def _source(hf, cells):
    from gpt_field.agent_speedup_v2_types import HF_SOURCES
    return [HF_SOURCES[int(hf.source[iy, ix])] for iy, ix in cells]


def test_d7a_fills_a_uniform_patch_of_the_plane_colour_at_plane_z_plus_3_sigma(aabb_scene):
    scene, hf = aabb_scene.scene, aabb_scene.scene.hf
    assert scene.planes and abs(scene.planes[0]["z"].value - 0.900) <= 0.002
    plane = scene.planes[0]
    cells = _cells_in(hf, aabb_scene.sc.info["uniform_patch_polygon"], shrink=0.3)
    sources = _source(hf, cells)
    assert sources.count("plane") >= 0.9 * len(cells), sources
    expected = plane["z"].value + scene_mod.D7_PLANE_SIGMA_K * plane["z"].sigma
    filled = [hf.z[iy, ix] for (iy, ix), src in zip(cells, sources) if src == "plane"]
    assert np.allclose(filled, expected)
    names = {t["name"] for t in aabb_scene.info["fill"]["thresholds"]}
    assert {"D7_colour_band", "D7_fill_height_m"} <= names


def test_d7b_textureless_dark_box_is_an_obstacle_never_a_plane_fill_and_a_later_kappa_replaces_it(aabb_scene):
    scene, hf = aabb_scene.scene, aabb_scene.scene.hf
    box = np.array([[-0.175, 0.015], [-0.125, 0.015], [-0.125, 0.065], [-0.175, 0.065]])
    cells = _cells_in(hf, box, shrink=0.1)
    sources = _source(hf, cells)
    assert "plane" not in sources, sources                    # the spec §4.5 negative
    obstacles = [(iy, ix) for (iy, ix), src in zip(cells, sources) if src == "obstacle"]
    assert obstacles
    base = scene.planes[0]["z"].value
    assert all(hf.z[ij] >= base for ij in obstacles)
    iy, ix = obstacles[0]
    xy = hf.cell_xy((iy, ix))
    sample = DepthSample((float(xy[0]), float(xy[1]), 0.930), 0.004, 0.002, "kappa", None, None, 5, None)
    survey = SurveyResult(tuple(aabb_scene.view.state), tuple(aabb_scene.view.state), {}, [], [sample], None, [], {})
    assert scene.absorb_survey(survey, aabb_scene.view) == 1
    assert _source(hf, [(iy, ix)]) == ["kappa"] and abs(hf.z[iy, ix] - 0.930) < 1e-9
    assert scene.absorb_survey(survey, aabb_scene.view) == 0          # one survey is absorbed once


def test_touch_overrides_kappa_and_kappa_overrides_a_plane_fill(aabb_scene):
    scene, hf = aabb_scene.scene, aabb_scene.scene.hf
    plane_cells = np.argwhere(hf.source == scene_mod._PLANE_CODE)
    iy, ix = plane_cells[len(plane_cells) // 2]
    xy = hf.cell_xy((iy, ix))
    view = aabb_scene.view
    sample = DepthSample((float(xy[0]), float(xy[1]), 0.9012), 0.003, 0.0015, "kappa", None, None, 6, None)
    scene.absorb_survey(SurveyResult(tuple(view.state), tuple(view.state), {}, [], [sample], None, [], {}), view)
    assert _source(hf, [(iy, ix)]) == ["kappa"]
    foot = np.array([xy + [-0.002, -0.002], xy + [0.002, -0.002], xy + [0.002, 0.002], xy + [-0.002, 0.002]])
    surface, threshold = scene.touch_surface(0.9095, 0.001)
    assert threshold.name == "D5_fingertip_reach_m" and threshold.label == "ROBOT"
    assert abs(surface.value - (0.9095 - RobotModel().t_f_m)) < 1e-12 and surface.source == "touch"
    assert scene.stamp_touch(foot, surface.value, surface.sigma, 7) >= 1
    assert _source(hf, [(iy, ix)]) == ["touch"] and abs(hf.z[iy, ix] - surface.value) < 1e-9
    sample2 = DepthSample((float(xy[0]), float(xy[1]), 0.95), 0.003, 0.0015, "kappa", None, None, 8, None)
    scene.absorb_survey(SurveyResult(tuple(view.state), tuple(view.state), {}, [], [sample2], None, [], {}), view)
    assert _source(hf, [(iy, ix)]) == ["touch"]                         # kappa never replaces a touch


def test_fill_unknown_recomputes_and_samples_in_the_robot_mask_are_skipped(aabb_scene):
    scene, view = aabb_scene.scene, aabb_scene.view
    before = int((scene.hf.source == scene_mod._PLANE_CODE).sum())
    again = scene.fill_unknown(view)
    assert again["plane"] >= 0.95 * before and again["vacated"] <= 0.05 * before
    robot = scene._robot_mask((512, 512))
    v, u = np.argwhere(robot)[len(np.argwhere(robot)) // 2]
    pose = HandPose.from_view(view)
    xyz = WristCamera(F_PX).backproject(pose, (float(u), float(v)), 0.2)
    sample = DepthSample(tuple(xyz), 0.003, 0.0015, "kappa", (float(u), float(v)), None, 9, None)
    survey = SurveyResult(tuple(view.state), tuple(view.state), {}, [], [sample], None, [], {})
    assert scene.absorb_survey(survey, view) == 0


# ========================================================================================= 2. registration
LEVER_TOP = (-0.2008, 0.2074, 0.955)
LOOP_TOP = (0.030, -0.0865, 1.0345)


@pytest.fixture(scope="module")
def lever():
    sc = fakes.rotary_part_scene()
    return register_at_hover(fakes.rotary_part_scene, sc.info["hover_eef"]["knob"], LEVER_TOP, "A", move="hinged")


@pytest.fixture(scope="module")
def lever_dense():
    """The lever-like part at its hover with the probe tracker's dense lattice (every v2 hover's survey)."""
    sc = fakes.rotary_part_scene()
    return register_at_hover(fakes.rotary_part_scene, sc.info["hover_eef"]["knob"], LEVER_TOP, "A", move="hinged",
                             dense=True)


@pytest.fixture(scope="module")
def thin():
    sc = fakes.free_object_with_thin_part_scene()
    return register_at_hover(fakes.free_object_with_thin_part_scene, sc.info["hover_eef"]["moka_handle"], LOOP_TOP,
                             "C")


def _width_at(rec, yaw):
    return rec.width_at(yaw).value


def test_elongated_rotary_part_on_a_disk(lever):
    """The lever-like part's geometry at its hover (sparse grid survey): width across the part, top, support (the
    disk), circle, rigid patch. Decision 2 §2.3 S-11: its outline gives the axis (within 3 deg) and passes the outline
    gate, but without the dense lattice's top band only the region ROI carries edge evidence, and the mottled dark top
    out-votes the low-contrast silhouette there: D10-OE refuses honestly (AXIS_INCONSISTENT "no_edge_agreement", BR-1
    and P-H refine_axis); the dense hover accepts it (``test_d10_oe_accepts_lever_like_and_thin_parts``)."""
    rec, truth = lever.rec, lever.sc.info["truth"]
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", rec
    check = rec.axis_check
    oe = lever.scene.last_registration["A"]["d10_oe"]
    assert oe["outline_ok"] and angle_deg(oe["axis"], truth["elongation_axis_xy"]) <= 3.0, oe
    assert not rec.is_elongated() and not check["accepted"] and check["method"] == "outline_edges"
    assert check["source"] == scene_mod.AXIS_INCONSISTENT and check["reason"] == "no_edge_agreement", check
    assert set(oe["edges"]) == {"region"} and not oe["edges"]["region"]["gated"]
    assert check["region_collapsed"] is None                        # no top band: the S-13 guard is not evaluated
    axis = np.asarray(truth["elongation_axis_xy"], float)
    across = math.atan2(axis[1], axis[0]) + math.pi / 2
    assert abs(_width_at(rec, across) - truth["lever_width_m"]) <= 0.003
    assert abs(rec.top_z.value - truth["part_top_z"]) <= 0.003
    assert abs(rec.support_z.value - truth["disk_top_z"]) <= 0.006
    assert rec.circle is not None, lever.scene.last_registration["A"]["circle_fit"]
    centre, radius, rms = rec.circle
    assert np.linalg.norm(centre - np.asarray(truth["disk_centre_xy"])) <= 0.005
    assert abs(radius - truth["disk_radius_m"]) <= 0.003 and rms >= 0.0
    patch = rec.rigid_patch
    assert "track" in patch and patch["dist_m"] >= scene_mod.RIGID_PATCH_MIN_DIST_M
    assert isinstance(patch["track"], NCCTrack) and rec.track is not None
    assert rec.registered_empty_hand and not rec.partial and not rec.provisional
    assert rec.sigma_h_m > 0.0 and rec.cov.shape == (3, 3) and rec.region_cells is not None
    assert rec.external is not None and rec.external["template"] is not None
    # the retired S-4' quantities stay logged as diagnosis
    assert {"cells", "cell_ratio", "tensor", "coherence", "aspect", "pixel_axis"} <= set(check)


def test_d10_oe_accepts_lever_like_and_thin_parts(lever_dense):
    """Decision 2 §6.1 item 2 (S-11): the lever-like fake at a dense hover (and its uniform dark variant at a sparse
    one) and the thin-bar fake (dense and sparse) are accepted by D10-OE: outline gate (share >= 0.20, second peak <=
    0.8, two-sided) and a gated edge axis within AXIS_AGREE_MAX_RAD; elongation = (outline axis, outline share);
    the axis within 3 deg of the part (5 deg for the thin bar); the axis is remembered (S-12). Lead ruling 4 (a)
    (r2close) leaves them accepted: the agreeing edge ROI also passes its own second-peak gate (it counts), and no
    refusal is flagged ``edges_bimodal``."""
    sc = fakes.dark_rotary_part_scene()
    dark = register_at_hover(fakes.dark_rotary_part_scene, sc.info["hover_eef"]["knob"], LEVER_TOP, "A",
                             move="hinged")
    thin_sc = fakes.free_object_with_thin_part_scene()
    thin_sparse = register_at_hover(fakes.free_object_with_thin_part_scene, thin_sc.info["hover_eef"]["moka_handle"],
                                    LOOP_TOP, "C")
    t = _thin_dense_hover()
    thin_rec = t.scene.register("C", t.view, seed=t.seed, empty_hand=True)
    thin_dense = SimpleNamespace(scene=t.scene, rec=thin_rec, sc=t.sc)
    cases = [("A", lever_dense, lever_dense.sc.info["truth"]["elongation_axis_xy"], 3.0, True),
             ("A", dark, sc.info["truth"]["elongation_axis_xy"], 3.0, False),
             ("C", thin_dense, thin_sc.info["truth"]["loop_axis_xy"], 5.0, True),
             ("C", thin_sparse, thin_sc.info["truth"]["loop_axis_xy"], 5.0, False)]
    for pid, case, axis, tol_deg, banded in cases:
        rec = case.rec
        assert isinstance(rec, scene_mod.PartRecord) and rec.is_elongated(), (pid, rec)
        check, oe = rec.axis_check, case.scene.last_registration[pid]["d10_oe"]
        assert check["accepted"] and check["method"] == "outline_edges" and check["source"] == "outline_edges"
        assert angle_deg(rec.elongation[0], axis) <= tol_deg and angle_deg(check["image_axis"], rec.elongation[0]) < 1e-6
        assert rec.elongation[1] == pytest.approx(oe["outline"]["share"])
        assert oe["outline"]["share"] >= scene_mod.D10_OUTLINE_SHARE_MIN and oe["outline"]["two_sided"]
        assert oe["outline"]["second"] <= scene_mod.D10_OUTLINE_SECOND_MAX
        counting = {k: e for k, e in oe["edges"].items() if e["counts"]}
        assert all(e["counts"] == (e["gated"] and e["unimodal"]) for e in oe["edges"].values())
        assert all(e["unimodal"] == ((e["second"] or 0.0) <= scene_mod.D10_OUTLINE_SECOND_MAX)
                   for e in oe["edges"].values() if e["axis"] is not None)
        assert counting and min(e["agree_rad"] for e in counting.values()) <= scene_mod.AXIS_AGREE_MAX_RAD
        assert check["agree_rad"] == pytest.approx(min(e["agree_rad"] for e in counting.values()))
        assert not oe["edges_bimodal"] and check["edges_bimodal"] is False
        assert ("hull" in oe["edges"]) == banded and (check["region_collapsed"] is False) == banded
        assert case.scene.view_memory(pid)["axes"] and not case.scene.view_memory(pid)["conflicted"]
        names = {th_.name for th_ in case.scene.registration_thresholds(pid)}
        assert {"D10_outline_share", "D10_outline_second", "D10_edge_share", "D10_edge_second", "D10_axis_agree_rad",
                "S12_view_agree_rad"} <= names, names


def test_registration_event_and_thresholds_are_json_safe(lever):
    import json
    payload = lever.scene.last_registration["A"]
    json.dumps(payload, allow_nan=False)
    for key in ("pid", "source", "status", "xyz", "sigma_xyz", "top_z", "support_z", "width_mm", "elongation",
                "circle", "rigid_patch", "known_fraction", "registered_empty_hand", "thresholds"):
        assert key in payload, key
    names = {t.name for t in lever.scene.registration_thresholds("A")}
    assert {"D8_same_plane_m", "D10_coherence", "D10_aspect", "D11_rms_px"} <= names
    json.dumps(lever.rec.to_dict(), allow_nan=False)
    json.dumps(lever.scene.summary(), allow_nan=False)


def test_dark_rotary_part_registers_with_an_unmeasurable_rigid_patch():
    sc = fakes.dark_rotary_part_scene()
    out = register_at_hover(fakes.dark_rotary_part_scene, sc.info["hover_eef"]["knob"], LEVER_TOP, "A",
                            move="hinged")
    assert isinstance(out.rec, scene_mod.PartRecord), out.rec
    assert out.rec.rigid_patch == {"unmeasurable": "textureless"}
    assert abs(out.rec.top_z.value - 0.955) <= 0.003


def test_thin_part_of_a_free_object(thin):
    rec, truth = thin.rec, thin.sc.info["truth"]
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", rec
    assert rec.is_elongated() and angle_deg(rec.elongation[0], truth["loop_axis_xy"]) <= 5.0
    assert abs(_width_at(rec, 0.0) - truth["loop_width_m"]) <= 0.003
    assert abs(rec.top_z.value - truth["loop_top_z"]) <= 0.003
    assert abs(rec.support_z.value - 0.900) <= 0.003


def test_planar_region_registers_with_an_empty_hand():
    sc = fakes.planar_region_scene()
    truth = sc.info["truth"]
    out = register_at_hover(fakes.planar_region_scene, (-0.20, 0.08, 1.19), (-0.20, 0.08, truth["top_z"]), "B",
                            roles=("ref",), move="fixed", amplitude=0.04)
    rec = out.rec
    assert isinstance(rec, scene_mod.PartRecord), rec
    assert abs(rec.top_z.value - truth["top_z"]) <= 0.003 and abs(rec.support_z.value - truth["support_z"]) <= 0.003
    assert poly_iou(rec.region_xy, np.asarray(truth["polygon_xy"])) >= 0.8
    assert rec.registered_empty_hand and not rec.is_elongated()


# ============================================================================================ 3. plan seeds
def _plan_from_answer(answer, *, texts=None):
    texts = texts or {}
    parts = tuple(PlanPart(p["id"], texts.get(p["id"], p["text"]), p["move"], int(p["w"]), int(p["x"]))
                  for p in answer["parts"])
    return plan_of(parts)


def _roles(plan):
    roles = {"A": {"grip"}, "B": {"effect", "ref"}, "C": {"grip"}, "D": {"carried"}, "E": {"keepout"}}
    return {p.pid: frozenset(roles.get(p.pid, ())) for p in plan.parts}


@pytest.fixture(scope="module")
def task2_start():
    """The task-2-like start view (3.14 deg tilt, k = 2) with Calibrate+-like 4 cm probe views cached."""
    sc = fakes.task2_like_scene()
    world = make_world(sc, fakes.TASK2_START_STATE8[:3], start_tilt_deg=3.14)
    scene, after, cal, info = started_scene(world, probe_first=0.04)
    oracle = fakes.FakeOracleAnswers(world)
    answer = oracle.plan_answer(2)
    plan = _plan_from_answer(answer)
    merged = scene.merge_plan(plan, _roles(plan), after, 2)
    return SimpleNamespace(world=world, scene=scene, view=after, cal=cal, oracle=oracle, plan=plan, merged=merged,
                           answer=answer, sc=sc)


def test_merge_plan_maps_cells_through_the_request_quarter_turns(task2_start):
    t = task2_start
    for part in t.plan.parts:
        rec = t.merged[part.pid]
        assert rec.status == "seeded" and rec.text == part.text and rec.roles == _roles(t.plan)[part.pid]
        wrist = [s for s in rec.seeds if s.camera == "wrist"]
        if part.w_cell:
            expected = unrotate_pixel(coarse_cell_center(part.w_cell, 512), 2, 512)
            assert np.allclose(wrist[0].uv, expected) and wrist[0].quantization_px == 32.0
            truth = t.oracle.truth_uv(part.pid, "wrist")
            if truth is not None and 0 <= truth[0] < 512 and 0 <= truth[1] < 512:
                assert np.max(np.abs(np.asarray(wrist[0].uv) - truth)) <= 32.0 + 1e-6, part.pid
        if part.x_cell:
            ext = [s for s in rec.seeds if s.camera == "external"][0]
            assert np.allclose(ext.uv, coarse_cell_center(part.x_cell, 512))


def test_replan_uses_its_own_quarter_turns_and_keeps_unchanged_registrations(task2_start):
    t = task2_start
    scene = copy.copy(t.scene)
    scene.parts = dict(t.scene.parts)
    keep = scene.parts["B"] = dataclasses.replace(scene.parts["B"], status="registered", xyz=np.zeros(3),
                                                  sigma_h_m=0.01, registration_step=1)
    answer = t.oracle.plan_answer(1)
    plan = _plan_from_answer(answer, texts={"E": "another text"})
    merged = scene.merge_plan(plan, _roles(plan), t.view, 1)
    assert merged["B"] is not keep and merged["B"].xyz is keep.xyz and merged["B"].status == "registered"
    part_e = plan.part("E")
    seed = [s for s in merged["E"].seeds if s.camera == "wrist"][0]
    assert np.allclose(seed.uv, unrotate_pixel(coarse_cell_center(part_e.w_cell, 512), 1, 512))
    assert "E" in scene.retired and scene.retired["E"].text != "another text"


def test_provisional_grip_part_centroid_at_the_start_view(task2_start):
    t = task2_start
    rec = t.scene.register("A", t.view, empty_hand=True, provisional=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    assert rec.provisional and rec.status == "seeded" and rec.xyz is not None and rec.sigma_h_m is not None
    assert rec.region_xy is None and rec.width_by_yaw is None and rec.elongation is None
    cell = [s for s in rec.seeds if s.camera == "wrist" and s.source == "plan_cell"][0]
    uv, _ = t.world.project("wrist", rec.xyz)
    assert np.max(np.abs(np.asarray(uv) - np.asarray(cell.uv))) <= cell.quantization_px
    knob = np.asarray(fakes.TASK2_KNOB_CENTRE_XY)
    assert np.linalg.norm(rec.xyz[:2] - knob) <= 0.03
    planner = Path(th.PACKAGE_DIR) / "agent_speedup_v2_planner.py"
    if planner.is_file():
        from gpt_field import agent_speedup_v2_planner as planner_mod
        if hasattr(planner_mod, "plan_contact") and hasattr(planner_mod, "PlannerInputs"):
            pose = HandPose.from_view(t.view)
            inputs = planner_mod.PlannerInputs(robot=RobotModel(), g_open_mm=79.9, sigma_align_m=0.003,
                                               yaw_budget_left={1: 1.96, -1: 1.96}, pose=pose, mount={},
                                               sigma_loc_px=None)
            result = planner_mod.plan_contact(rec, t.scene.hf, inputs, [], purpose="turn")
            assert type(result).__name__ == "Infeasible" and result.reason == "geometry_unmeasured"


def _sigma_lat(rec):
    return math.sqrt(max(float(np.linalg.eigvalsh(np.asarray(rec.cov, float)[:2, :2]).max()), 0.0))


def test_provisional_and_border_cut_sigmas_cover_the_truth(task2_start):
    """Review BS-R2 / BS-R3 on the task-2-like start view: a provisional grip-only part needs no allow_partial even
    when the image border cuts it (C, partial=True), and every provisional or border-cut record's sigma covers the
    truth point (lateral |err| <= 3 sigma; height |err| <= 3 sigma_h, or for a lower-bound top (rulings R2: a side
    cut, PartRecord.top_bound "lower") top - 3 sigma_h <= truth <= top_upper_z). Decision 2 S-17 (BSR-4): a side cut is
    a lower bound only when the part band is not a plane band. Runs on a copy (the fixture's records stay seeded)."""
    t = task2_start
    scene = copy.copy(t.scene)
    scene.parts = dict(t.scene.parts)
    scene.last_registration = {}
    scene._registration_thresholds = {}
    scene._extents = dict(t.scene._extents)
    sides, plane_side = [], []
    for pid, provisional in (("A", True), ("C", True), ("B", False), ("D", False), ("E", False)):
        rec = scene.register(pid, t.view, empty_hand=True, provisional=provisional, allow_partial=not provisional)
        assert isinstance(rec, scene_mod.PartRecord), (pid, rec)
        event = scene.last_registration[pid]
        truth = np.asarray(t.oracle.truth_xyz(pid), float)
        top = rec.top_z.value if rec.top_z is not None else float(rec.xyz[2])
        assert np.linalg.norm(rec.xyz[:2] - truth[:2]) <= 3.0 * _sigma_lat(rec), (pid, rec.xyz, truth, event)
        if rec.top_is_lower_bound():
            assert top - 3.0 * rec.sigma_h_m <= truth[2] <= rec.top_upper_z, (pid, top, truth, rec.top_upper_z)
            # R2: no 1 / (1 - cut) on the top's sigma (the lower bound replaces it)
            assert abs(rec.sigma_h_m - max(event["top_sigma_raw"], event["hidden_sigma_h"],
                                           event["choice_sigma_h"])) <= 1e-12, (pid, event)
        elif not (rec.partial and event.get("part_band_plane")):
            assert abs(top - truth[2]) <= 3.0 * rec.sigma_h_m, (pid, top, truth, rec.sigma_h_m)
        else:
            # BSR-4 risk on this fake (reported): E's start region is the larger planar surface beside it (a plane
            # band), its truth top 17 mm above (3.3 sigma_h); without the lower bound only the partial view's survey
            # hover (needs_survey_hover) protects it
            plane_side.append((pid, float(truth[2] - top) / rec.sigma_h_m))
        if event.get("cut_kind") == "side" and event["cut_fraction"] >= scene_mod.PARTIAL_CUT_FRACTION:
            assert rec.partial and rec.elongation is None and rec.circle is None and scene.needs_survey_hover(pid)
            sides.append((pid, event["part_band_plane"]))
            if event["part_band_plane"]:
                # decision 2 S-17 (BSR-4): a plane band's side cut shows its top, so no lower bound (no P-F raise)
                assert event["top_bound"] is None and rec.top_bound is None and rec.top_upper_z is None, (pid, event)
            else:
                assert event["top_bound"] == "lower" and rec.top_bound == "lower" and rec.top_upper_z >= top
                assert abs(rec.top_upper_z - max(scene._lift_ceiling(), top)) <= 1e-12
        if provisional:
            assert rec.provisional and rec.status == "seeded"
            # the plan cell's quantization alone is a floor of the lateral sigma (+-32 px at the start kappa)
            kappa = F_PX / max(float(HandPose.from_view(t.view).camera[2] - top), 1e-6)
            assert _sigma_lat(rec) >= 32.0 / math.sqrt(3.0) / kappa - 1e-9
    assert scene.parts["C"].partial and scene.last_registration["C"]["cut_kind"] == "side"
    # BSR-4 both ways on one view: plane-band side cuts (B, E) keep their top, measured-band ones (C, D) are bounds
    assert {flag for _, flag in sides} == {True, False}, sides
    assert all(scene.needs_survey_hover(pid) for pid, _ in plane_side), plane_side


def test_external_seed_lifts_onto_its_surface_and_covers_unknown_cells(task2_start):
    """Review BS-R4: an external seed is lifted by marching down the external ray from above; on a fully known
    HeightField it lands on the surface within its quantization, through unknown cells its sigma covers the truth,
    and a part with only an external seed (plan w = 0) registers provisionally and is predicted."""
    t = task2_start
    scene = copy.copy(t.scene)
    scene.parts = dict(t.scene.parts)
    scene.last_registration = {}
    scene._registration_thresholds = {}
    pose = HandPose.from_view(t.view)
    # every part's truth point projected into the external camera: the lift's sigma covers it (Mahalanobis <= 3)
    for pid in "ABCDE":
        truth = np.asarray(t.oracle.truth_xyz(pid), float)
        uv, _ = scene.projector.project(truth)
        lifted = scene.estimate_seed_xyz(Seed("external", tuple(uv), 0, "fine", 1.0, tuple(t.view.state)), t.view)
        assert lifted is not None, pid
        err = lifted[0] - truth
        assert float(np.sqrt(err @ np.linalg.pinv(lifted[1]) @ err)) <= 3.0, (pid, lifted, scene.last_external_lift)
    # a fully known field: the lift is the surface hit, within its quantization
    known = copy.deepcopy(scene.hf)
    ny, nx = known.shape
    for iy in range(ny):
        xy = np.array([[known.cell_xy((iy, ix))[0], known.cell_xy((iy, ix))[1]] for ix in range(nx)])
        zs = t.world.heights(xy)
        for ix in range(nx):
            known.stamp_ij((iy, ix), float(zs[ix]), 0.001, "touch", 0)
    scene.hf = known
    point = np.r_[np.asarray(fakes.TASK2_BURNER_XY), 0.9305]
    uv, _ = scene.projector.project(point)
    lifted = scene.estimate_seed_xyz(Seed("external", tuple(uv), 0, "fine", 2.0, tuple(t.view.state)), t.view)
    assert lifted is not None and scene.last_external_lift["kind"] == "surface"
    err = lifted[0] - point
    sigma = np.sqrt(np.diag(lifted[1]))
    assert np.all(np.abs(err) <= 3.0 * sigma + 1e-6) and np.linalg.norm(err) <= 0.03, (err, sigma)
    scene.hf = t.scene.hf
    # w = 0: only the x cell seed
    cell = [s_ for s_ in t.scene.parts["C"].seeds if s_.camera == "external"][0]
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}), seeds=[cell])
    assert scene.predicted_uv("C", t.view, "external") is not None
    rec = scene.register("C", t.view, empty_hand=True, provisional=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.provisional, rec
    assert scene.last_registration["C"]["seed_from_external"]
    truth = np.asarray(t.oracle.truth_xyz("C"), float)
    err = np.asarray(rec.xyz) - truth
    assert float(np.sqrt(err @ np.linalg.pinv(rec.cov) @ err)) <= 3.0, (rec.xyz, truth)
    assert scene.predicted_uv("C", t.view) is not None and pose is not None


def test_partial_part_at_the_border_then_whole_from_a_later_view(task2_start):
    t = task2_start
    scene = t.scene
    refused = scene.register("D", t.view, empty_hand=True)
    assert isinstance(refused, RegistrationFailure) and refused.reason == "geometry_unmeasured"
    assert refused.evidence["why"] == "cut_by_border"
    part = scene.register("D", t.view, empty_hand=True, allow_partial=True)
    assert isinstance(part, scene_mod.PartRecord) and part.partial and part.status == "registered", part
    event = scene.last_registration["D"]
    cut = event["cut_fraction"]
    assert cut >= scene_mod.PARTIAL_CUT_FRACTION and event["cut_kind"] == "side" and event["top_bound"] == "lower"
    # rulings R2 (decision §2.2 S-9, remedy 2 test 13): the cut top is a LOWER BOUND carried in the record, with
    # top_upper_z = max(start fingertip height, highest measured surface); its sigma_h takes no 1 / (1 - cut) factor
    # (the lateral sigma keeps it)
    assert part.top_bound == "lower" and part.top_is_lower_bound() and scene.needs_survey_hover("D")
    assert abs(part.top_upper_z - max(scene._lift_ceiling(), part.top_z.value)) <= 1e-12
    assert abs(part.sigma_h_m - max(event["top_sigma_raw"], event["hidden_sigma_h"])) <= 1e-12
    assert part.to_dict()["top_bound"] == "lower" and part.to_dict()["top_upper_z"] == part.top_upper_z
    truth = np.asarray(t.oracle.truth_xyz("D"), float)
    assert part.top_z.value - 3.0 * part.sigma_h_m <= truth[2] <= part.top_upper_z
    assert np.linalg.norm(part.xyz[:2] - truth[:2]) <= 3.0 * math.sqrt(np.linalg.eigvalsh(part.cov[:2, :2]).max())
    assert part.elongation is None and part.circle is None          # a side-cut region stores neither
    # a view that sees the whole body: the hand moved 15 cm toward it
    world = t.world
    fakes.scripted_goto(world, world.grip + np.array([0.15, -0.02, 0.0]), tol_m=0.0005)
    view = view_of(world)
    scene.on_view(view)
    assert "D" in scene.partial_parts_in_view(view)
    survey = synthetic_survey(world, view, seeds=[("D", tuple(world.project("wrist", (0.030, -0.007, 1.0331))[0]))],
                              rng=np.random.default_rng(1))
    scene.absorb_survey(survey, view)
    after = probe(world, scene, amplitude=0.02)
    seed = seed_at(world, after, (0.030, -0.007, 1.0331))
    whole = scene.register("D", after, seed=seed, empty_hand=True, allow_partial=True)
    assert isinstance(whole, scene_mod.PartRecord) and not whole.partial, scene.last_registration["D"]
    assert abs(whole.top_z.value - 1.0331) <= 0.004 and scene.last_registration["D"]["top_bound"] is None
    # the whole re-registration clears the lower-bound pair (survey_hover support, rulings item 8)
    assert whole.top_bound is None and whole.top_upper_z is None and not scene.needs_survey_hover("D")


# ============================================================================================ 4. region masks
@pytest.fixture(scope="module")
def planar():
    sc = fakes.planar_region_scene()
    truth = sc.info["truth"]
    return register_at_hover(fakes.planar_region_scene, (-0.20, 0.08, 1.19), (-0.20, 0.08, truth["top_z"]), "B",
                             roles=("ref",), move="fixed", amplitude=0.04)


def _iou(a, b):
    return np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1)


def test_external_region_mask_affine_and_refined(planar):
    scene, view, world = planar.scene, planar.view, planar.world
    rec = scene.parts["B"]
    truth = world.truth_mask("external", "region")
    poly3 = np.column_stack([rec.region_xy, np.full(len(rec.region_xy), rec.top_z.value)])
    affine = scene.projector.mask(poly3, truth.shape)
    assert _iou(affine, truth) >= 0.6
    refined = scene.region_mask("B", "external", view)
    assert refined is not None and _iou(refined, truth) >= 0.8           # contract §C6 T1 item 4, the full truth mask
    log = scene.mask_log["B"]["external"]
    assert log["flag"] is None and log["sanity"] is None
    assert {"affine_px", "refined_px", "iou_affine_refined", "refine"} <= set(log)
    eroded = scene.region_mask("B", "external", view, erode_px=3.0)
    assert eroded.sum() < refined.sum() and not (eroded & ~refined).any()


def test_wrist_region_mask_matches_the_projected_top_face(planar):
    scene, view, world = planar.scene, planar.view, planar.world
    mask = scene.region_mask("B", "wrist", view)
    top = planar.sc.info["truth"]
    uv, _ = world.project("wrist", np.column_stack([np.asarray(top["polygon_xy"]), np.full(4, top["top_z"])]))
    truth = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(truth, [np.rint(uv).astype(np.int32)], 1)
    assert mask is not None and _iou(mask, truth > 0) >= 0.9
    with pytest.raises(ValueError):
        scene.region_mask("B", "side", view)


def _hand_truth_mask(world, yaw, shape):
    """Truth projection (fake external camera) of the fake's hand boxes (palm, upper hand, fingers; not the forearm
    or link, which lie above the hand) with the hand turned by ``yaw`` about its approach R2."""
    R = world.R
    turn = fakes._rot_axis(R[:, 2], yaw) @ R
    mask = np.zeros(shape, np.uint8)
    for name, c, half, _, _ in fakes._hand_boxes(world.q1, world.q2, external=True):
        if name in ("forearm",):
            continue
        signs = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
        corners = world.grip + (np.asarray(c, float)[None, :] + signs * np.asarray(half, float)[None, :]) @ turn.T
        uv, _ = world.external.project(corners)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
    return mask > 0


def _zero_sigma(projector):
    """A duck-typed projector with ``projector``'s projection and no sigma and no anchor (the raw hull)."""
    return SimpleNamespace(project=lambda xyz: (projector.project(xyz)[0], 0.0))


def _raw_hand_hull(scene, view, yaw_range):
    """The undilated S-16' hull projected with the scene's own hand projector (zero-sigma stand-in)."""
    saved = scene.projector
    hand, _ = scene.hand_projector(view)
    try:
        scene.projector = _zero_sigma(hand)
        return scene.hand_swept_mask("external", view, yaw_range, dilate_px=0.0)
    finally:
        scene.projector = saved


def _r7_slab_hull(scene, view, yaw_range):
    """The former S-16 model (diagnosis): the R7 rectangle from the fingertips (grip - t_f) up to the hand's z, one
    hull per consecutive footprint pair, projected with the scene's hand projector, undilated."""
    pose = HandPose.from_view(view)
    hand, _ = scene.hand_projector(view)
    shape = np.asarray(view.external_rgb_upright).shape[:2]
    lo, hi = sorted(yaw_range)
    count = max(1, int(math.ceil((hi - lo) / scene_mod.SWEEP_STEP_RAD)))
    jaw0 = pose.R[:2, 1] / np.linalg.norm(pose.R[:2, 1])
    out = np.zeros(shape, np.uint8)
    prisms = []
    for yaw in np.linspace(lo, hi, count + 1):
        c, s = math.cos(-yaw), math.sin(-yaw)
        jaw = np.array([c * jaw0[0] - s * jaw0[1], s * jaw0[0] + c * jaw0[1]])
        poly = scene.robot.hand_footprint_xy(pose.grip[:2], jaw)
        prisms.append(np.vstack([np.column_stack([poly, np.full(4, pose.grip[2] - scene.robot.t_f_m)]),
                                 np.column_stack([poly, np.full(4, pose.hand[2])])]))
    for a, b in (list(zip(prisms[:-1], prisms[1:])) or [(prisms[0], prisms[0])]):
        uv = np.array([hand.project(p)[0] for p in np.vstack([a, b])])
        cv2.fillConvexPoly(out, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
    return out > 0


def test_hand_swept_mask_contains_the_signed_sweep_and_not_the_opposite_sense(planar):
    """Decision §6 R8 (rulings wording): the mask covers >= 98 % of the known sense's truth sweep, equals the hull's
    sweep over that sense dilated by the S-16' kernel (effect lens fix (c)), and the +1 and -1 masks differ
    (over-coverage beyond the fake hand is conservative for O7)."""
    scene, view, world = planar.scene, planar.view, planar.world
    shape = np.asarray(view.external_rgb_upright).shape[:2]
    mask = scene.hand_swept_mask("external", view, (0.0, 0.6))
    assert mask is not None and mask.shape == shape
    sweep = np.zeros(shape, bool)
    for yaw in np.linspace(0.0, 0.6, 7):
        sweep |= _hand_truth_mask(world, yaw, shape)
    assert np.count_nonzero(sweep & mask) / np.count_nonzero(sweep) >= 0.98
    # the sweep is signed: the opposite range gives a different mask holding the opposite truth, each mask has its
    # own area where its sense's truth lies (a volume projected into one camera smears along the vertical, so the two
    # overlap near the hand), and both lie inside the symmetric range's mask
    back_truth = np.zeros(shape, bool)
    for yaw in np.linspace(-0.6, 0.0, 7):
        back_truth |= _hand_truth_mask(world, yaw, shape)
    back = scene.hand_swept_mask("external", view, (-0.6, 0.0))
    both = scene.hand_swept_mask("external", view, (-0.6, 0.6))
    assert np.count_nonzero(back_truth & back) / np.count_nonzero(back_truth) >= 0.98
    fwd_only, back_only = mask & ~back, back & ~mask
    assert fwd_only.sum() >= 0.05 * mask.sum() and back_only.sum() >= 0.05 * back.sum()
    assert (fwd_only & sweep & ~back_truth).any() and (back_only & back_truth & ~sweep).any()
    assert not ((mask | back) & ~both).any() and mask.sum() < both.sum()
    # S-16' item 2 with the BSBP amendment: near the hand projector's anchor (the calibration reading at this hover)
    # every hull vertex takes its own margin max((SWEPT_DILATE_PX - 1) / 2, ceil(2 max(sigma_grip, sigma_vertex)));
    # the raw hull = the same components through the same hand projector with zero sigma and no kernel
    raw = _raw_hand_hull(scene, view, (0.0, 0.6))
    mask = scene.hand_swept_mask("external", view, (0.0, 0.6))
    log = dict(scene.mask_log["hand_swept"])
    assert log["rule"] == "local" and log["per_vertex"] and log["grip_anchor_dist_m"] < scene_mod.HAND_MASK_LOCAL_RADIUS_M
    base = (scene_mod.SWEPT_DILATE_PX - 1) // 2
    assert log["sigma_proj_px"] == log["sigma_grip_px"] and log["sigma_far_px"] >= log["sigma_grip_px"]
    assert log["margin_px"] == max(base, math.ceil(2.0 * log["sigma_far_px"])) and log["side_px"] == 1
    assert log["margin_min_px"] >= max(base, math.ceil(2.0 * log["sigma_grip_px"]))
    assert log["components"] == ["palm", "finger1", "finger2"] and log["finger_outer_r1_mm"] == pytest.approx(26.4)
    _assert_between(mask, raw, int(log["margin_min_px"]), int(log["margin_px"]))
    plain = scene.hand_swept_mask("external", view, (0.0, 0.6), dilate_px=0.0)       # the sigma margin stays
    assert raw.sum() < plain.sum() <= mask.sum() and not (plain & ~mask).any()
    assert scene.hand_swept_mask("wrist", view, (-0.2, 0.2)).shape == (512, 512)


def _assert_between(mask, raw, inner_px, outer_px):
    """The per-vertex mask holds the raw hull grown by a disc of the smallest vertex margin (1 px rasterization) and
    lies inside it grown by a square of the largest (+ 1 px, the circumscribed 16-gon)."""
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (inner_px - 1) + 1,) * 2)
    inner = cv2.dilate(raw.astype(np.uint8), disc) > 0
    outer = cv2.dilate(raw.astype(np.uint8), np.ones((2 * (outer_px + 1) + 1,) * 2, np.uint8)) > 0
    assert not (inner & ~mask).any() and not (mask & ~outer).any()


def test_s16p_hand_model_is_the_palm_above_its_underside_and_the_finger_columns(planar):
    """Effect lens fix (c) S-16' item 1 [ROBOT]: the external hull is the palm prism (R7 rectangle from grip + p_min
    up to the hand) plus two finger columns (R1 +-q .. +-(q + 26.4 mm), R0 +-6.14 mm, grip - t_f .. grip + p_min),
    each hulled separately: it lies inside the former R7 slab (fingertips up to the hand) and is clearly smaller, and
    it never covers the space between the open fingers below the palm. Negative: a box between the columns below the
    palm projects into the former slab but not into the S-16' hull."""
    scene, view, world = planar.scene, planar.view, planar.world
    pose = HandPose.from_view(view)
    comps = scene._hand_components(pose)
    names = [c[0] for c in comps]
    assert names == ["palm", "finger1", "finger2"]
    (_, r1_palm, r0_palm, palm_lo, palm_hi), (_, f1_r1, f1_r0, f_lo, f_hi), (_, f2_r1, _, _, _) = comps
    q = pose.gap_mm / 2.0
    assert palm_lo == pytest.approx(pose.grip[2] + scene.robot.p_min_m) and palm_hi == pytest.approx(pose.hand[2])
    assert f_lo == pytest.approx(pose.grip[2] - scene.robot.t_f_m) and f_hi == pytest.approx(palm_lo)
    assert f1_r1 == pytest.approx((q, q + 26.4)) and f2_r1 == pytest.approx((-q - 26.4, -q))
    assert f1_r0 == pytest.approx((-6.14, 6.14)) and r0_palm == pytest.approx((-31.64, 31.62))
    for yaw_range in ((0.0, 0.0), (0.0, 0.6)):
        raw = _raw_hand_hull(scene, view, yaw_range)
        slab = _r7_slab_hull(scene, view, yaw_range)
        grown = cv2.dilate(slab.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        assert not (raw & ~grown).any(), yaw_range                      # inside the former slab (1 px rasterization)
        assert raw.sum() <= 0.9 * slab.sum(), (yaw_range, int(raw.sum()), int(slab.sum()))
    # the gap between the open fingers below the palm (R0 15-25 mm off the jaw line, the fingertips' height): the
    # former slab covers its projection, the S-16' hull leaves (most of) it out
    hand, _ = scene.hand_projector(view)
    gap = _box_xy_test(pose, (-q + 4.0, q - 4.0), (15.0, 25.0))
    pts = np.vstack([np.column_stack([gap, np.full(4, pose.grip[2] - 0.009)]),
                     np.column_stack([gap, np.full(4, pose.grip[2] + 0.020)])])
    probe_mask = np.zeros(raw.shape, np.uint8)
    uv = np.array([hand.project(p)[0] for p in pts])
    cv2.fillConvexPoly(probe_mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
    probe_mask = probe_mask > 0
    raw0, slab0 = _raw_hand_hull(scene, view, (0.0, 0.0)), _r7_slab_hull(scene, view, (0.0, 0.0))
    assert np.count_nonzero(probe_mask & slab0) == np.count_nonzero(probe_mask)
    assert np.count_nonzero(probe_mask & raw0) <= 0.5 * np.count_nonzero(probe_mask)


def _box_xy_test(pose, r1_mm, r0_mm):
    jaw = pose.R[:2, 1] / np.linalg.norm(pose.R[:2, 1])
    up = np.array([-jaw[1], jaw[0]])
    corners = np.array([[r1_mm[0], r0_mm[0]], [r1_mm[1], r0_mm[0]], [r1_mm[1], r0_mm[1]], [r1_mm[0], r0_mm[1]]]) / 1e3
    return pose.grip[:2][None, :] + np.outer(corners[:, 0], jaw) + np.outer(corners[:, 1], up)


def _anchored(scene, world, anchor, sigma_px=2.0, *, j_at=None):
    """A geometry ExternalProjector anchored at ``anchor`` on the fake external camera's true projection; its affine
    model J is the camera's Jacobian at ``j_at`` (None = the anchor): ONE fitted model shared by every anchor."""
    from gpt_field.agent_speedup_v2_geometry import ExternalProjector
    model = SimpleNamespace(J=world.external.jacobian(anchor if j_at is None else j_at))
    uv = np.asarray(world.external.project(anchor)[0][0], float)
    return ExternalProjector(model, uv, anchor, sigma_px), {"step": -1, "uv": uv, "grip": np.asarray(anchor, float),
                                                            "sigma": float(sigma_px), "source": "calibration"}


def test_hand_swept_mask_margin_covers_the_affine_projection_error(planar):
    """S-16' item 2, far rule (decision 2 S-16 on the S-16' hull): with the only kept reading 15 cm from the hand
    (beyond HAND_MASK_LOCAL_RADIUS_M) the geometry ExternalProjector there (sigma = hypot(sigma_anchor,
    PROJECTOR_AFFINE_FRACTION |delta uv|)) gives the far-vertex sigma, and the mask covers the projected hull shifted
    by PROJECTOR_AFFINE_FRACTION x the farthest vertex's |delta uv| in every direction; the former 4 px margin
    (SWEPT_DILATE_PX kernel) does not."""
    from gpt_field.agent_speedup_v2_geometry import PROJECTOR_AFFINE_FRACTION
    scene, view, world = planar.scene, planar.view, planar.world
    pose = HandPose.from_view(view)
    anchor = np.asarray(pose.grip, float) + np.array([0.15, 0.0, 0.0])
    projector, reading = _anchored(scene, world, anchor)
    saved, saved_loc = scene.projector, list(scene._locations)
    try:
        scene._locations = [reading]
        scene.projector = projector
        raw = _raw_hand_hull(scene, view, (0.0, 0.0)).astype(np.uint8)
        mask = scene.hand_swept_mask("external", view, (0.0, 0.0))
        log = dict(scene.mask_log["hand_swept"])
    finally:
        scene.projector, scene._locations = saved, saved_loc
    assert log["rule"] == "far" and log["grip_anchor_dist_m"] == pytest.approx(0.15, abs=1e-6)
    assert log["sigma_proj_px"] == log["sigma_far_px"] > log["sigma_grip_px"]
    vs, us = np.nonzero(raw)
    corners = np.column_stack([us, vs]).astype(float)
    duv = float(np.max(np.linalg.norm(corners - projector.anchor_uv[None, :], axis=1)))
    shift = PROJECTOR_AFFINE_FRACTION * duv
    assert shift > (scene_mod.SWEPT_DILATE_PX - 1) // 2 and log["margin_px"] >= shift
    old = cv2.dilate(raw, np.ones((scene_mod.SWEPT_DILATE_PX,) * 2, np.uint8)) > 0
    old_miss = 0
    for k in range(8):
        a = 2.0 * math.pi * k / 8
        M = np.float32([[1, 0, shift * math.cos(a)], [0, 1, shift * math.sin(a)]])
        moved = cv2.warpAffine(raw, M, (raw.shape[1], raw.shape[0]), flags=cv2.INTER_NEAREST) > 0
        assert not (moved & ~mask).any(), k
        old_miss += int((moved & ~old).sum())
    assert old_miss > 0


def test_s16p_hand_mask_anchors_on_the_reading_nearest_the_grip_not_the_region_projector(planar):
    """S-16' items 2-3: a region-mask projector re-anchored 0.3 m away ([C-51]) is not the hand mask's projector:
    the kept reading nearest the grip in 3-D anchors it, the local rule applies (margin max(4, ceil(2 sigma_grip)),
    25-26 px instead of 43-45 px live) and the mask still covers the fake hand's truth silhouette; with the far
    reading alone the far rule's mask is larger. Negative: a far kept reading never wins over a nearer one."""
    scene, view, world = planar.scene, planar.view, planar.world
    shape = np.asarray(view.external_rgb_upright).shape[:2]
    pose = HandPose.from_view(view)
    grip = np.asarray(pose.grip, float)
    far_projector, far_reading = _anchored(scene, world, grip + np.array([0.0, 0.3, -0.25]), j_at=grip)
    _, near_reading = _anchored(scene, world, grip + np.array([0.01, -0.01, 0.02]), 3.0, j_at=grip)
    near_reading["source"] = "reanchor_from"
    saved, saved_loc = scene.projector, list(scene._locations)
    try:
        scene.projector = far_projector
        scene._locations = [far_reading, near_reading]
        hand, info = scene.hand_projector(view)
        local = scene.hand_swept_mask("external", view, (0.0, 0.0))
        log_local = dict(scene.mask_log["hand_swept"])
        scene._locations = [far_reading]
        far = scene.hand_swept_mask("external", view, (0.0, 0.0))
        log_far = dict(scene.mask_log["hand_swept"])
    finally:
        scene.projector, scene._locations = saved, saved_loc
    assert info["source"] == "reanchor_from" and np.allclose(info["grip"], near_reading["grip"])
    assert np.allclose(hand.anchor_uv, near_reading["uv"]) and hand.sigma_px == 3.0
    assert log_local["rule"] == "local" and log_local["anchor"]["source"] == "reanchor_from"
    assert log_local["margin_px"] == max(4, math.ceil(2.0 * max(log_local["sigma_grip_px"],
                                                                 log_local["sigma_far_px"])))
    assert log_local["margin_min_px"] >= max(4, math.ceil(2.0 * log_local["sigma_grip_px"]))
    assert log_local["sigma_grip_px"] == pytest.approx(math.hypot(3.0, 0.2 * float(np.linalg.norm(
        hand.model.J @ (np.asarray(pose.grip, float) - near_reading["grip"])))))
    assert log_far["rule"] == "far" and log_far["margin_px"] > log_local["margin_px"]
    assert local.sum() < far.sum()
    truth = _hand_truth_mask(world, 0.0, shape)
    assert np.count_nonzero(truth & local) / np.count_nonzero(truth) >= 0.98


def test_s16p_local_margin_grows_with_each_vertex_sigma_rs9(planar):
    """BSBP amendment of S-16' item 2 (RS-9 real-sim gate: the spec's single max(4, ceil(2 sigma_grip)) margin covered
    only 96.5-98.9 % of the hand silhouette at 5 of 10 calibration-yaw hovers, both runs): the affine projector's
    scale error grows across the 20 cm hand even with the anchor AT the grip. On the fake (a perspective camera, the
    affine J fitted at the grip) the single-sigma mask misses the true hand's far end; the per-vertex mask covers it."""
    scene, view, world = planar.scene, planar.view, planar.world
    shape = np.asarray(view.external_rgb_upright).shape[:2]
    pose = HandPose.from_view(view)
    grip = np.asarray(pose.grip, float)
    projector, reading = _anchored(scene, world, grip, 2.0, j_at=grip)
    saved, saved_loc = scene.projector, list(scene._locations)
    try:
        scene.projector, scene._locations = projector, [reading]
        mask = scene.hand_swept_mask("external", view, (0.0, 0.0))
        log = dict(scene.mask_log["hand_swept"])
        raw = _raw_hand_hull(scene, view, (0.0, 0.0))
    finally:
        scene.projector, scene._locations = saved, saved_loc
    assert log["rule"] == "local" and log["grip_anchor_dist_m"] == pytest.approx(0.0, abs=1e-9)
    single = max(4, math.ceil(2.0 * log["sigma_grip_px"]))
    assert single == 4 and log["margin_px"] > single
    spec_mask = cv2.dilate(raw.astype(np.uint8), np.ones((2 * single + 1,) * 2, np.uint8)) > 0
    truth = _hand_truth_mask(world, 0.0, shape)
    cover = lambda m: np.count_nonzero(truth & m) / np.count_nonzero(truth)
    assert cover(mask) >= 0.98 and cover(spec_mask) < cover(mask)
    _assert_between(mask, raw, int(log["margin_min_px"]), int(log["margin_px"]))


# ======================================================================================= 5. HeldFrameEstimator
LID_TOP = (0.030, -0.007, 1.0331)


def _lowest(world, group="moka"):
    return float(world._bottom_points(group)[:, 2].min())


def held_setup(*, close_drag=(0.0, 0.0), yaw_sign=1.0):
    """Register the thin part C and its free object D at the C hover, turn the jaw across C (camera on the object's
    side for yaw_sign +1), descend to the height stop, close; every view goes through scene.on_view."""
    sc = fakes.free_object_with_thin_part_scene()
    hover = np.asarray(sc.info["hover_eef"]["moka_handle"], float)
    world = make_world(sc, hover, close_drag_m=close_drag)
    seeds = [("C", tuple(world.project("wrist", LOOP_TOP)[0])), ("D", tuple(world.project("wrist", LID_TOP)[0]))]
    scene, _, cal, _ = started_scene(world, survey_seeds=seeds)
    after = probe(world, scene)
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    scene.parts["D"] = scene_mod.PartRecord("D", "free object", "free", frozenset({"carried"}))
    rec_c = scene.register("C", after, seed=seed_at(world, after, LOOP_TOP), empty_hand=True)
    rec_d = scene.register("D", after, seed=seed_at(world, after, LID_TOP), empty_hand=True, allow_partial=True)
    assert isinstance(rec_c, scene_mod.PartRecord) and isinstance(rec_d, scene_mod.PartRecord), (rec_c, rec_d)
    q = fakes._matrix_quat(fakes._rot_axis((0.0, 0.0, 1.0), yaw_sign * math.pi / 2) @ world.R)
    fakes.scripted_goto(world, hover, quat_xyzw=q, max_steps=300)
    scene.on_view(view_of(world))
    fakes.scripted_goto(world, [hover[0], hover[1], 1.020], quat_xyzw=q, tol_m=0.0005)
    pre = view_of(world)
    scene.on_view(pre)
    fakes.scripted_gripper(world, 1.0, 25)
    post = view_of(world)
    scene.on_view(post)
    est = scene_mod.HeldFrameEstimator(scene, RobotModel())
    held = est.prior_at_close("D", "C", post, support_z_pick=rec_d.support_z, gap_mm=post.gap_mm)
    return SimpleNamespace(world=world, scene=scene, est=est, held=held, pre=pre, post=post, q=q, rec_c=rec_c,
                           rec_d=rec_d)


def _lift(t, *, tilt_deg=None):
    recorder = t.est.lift_recorder(t.held, t.post)
    if tilt_deg:
        fakes.scripted_goto(t.world, t.world.grip + [0.0, 0.0, 0.010], gripper=1.0, quat_xyzw=t.q, tol_m=0.0005)
        t.world.inject("tilt", deg=tilt_deg)
    fakes.scripted_goto(t.world, t.post.position + [0.0, 0.0, 0.025], gripper=1.0, quat_xyzw=t.q, tol_m=0.0005)
    lifted = view_of(t.world)
    t.scene.on_view(lifted)
    recorder.on_transition("lift", t.post, lifted)
    return recorder, lifted


@pytest.fixture(scope="module")
def held_rigid():
    return held_setup()


def test_prior_at_close_b_low_is_the_close_height_above_the_pick_support(held_rigid):
    t = held_rigid
    held, world = t.held, t.world
    assert held.pid == "D" and held.grip_pid == "C" and t.scene.held is held
    truth = float(world.grip[2]) - _lowest(world)
    assert abs(held.b_low.value - truth) <= 0.001
    assert abs(held.g_close - float(world.grip[2])) < 1e-6 and held.gap_ref_mm == t.post.gap_mm
    assert held.wrist_visible and held.offset_h[0] > 0.0          # P17: the object on the camera side (R0)
    assert held.extent_axes is not None and held.extent_axes.shape == (3, 2) and held.r_obj_m > 0.02
    import json
    json.dumps(held.to_dict(), allow_nan=False)


def test_lift_silhouette_of_a_rigid_hold(held_rigid):
    t = held_rigid
    recorder, lifted = _lift(t)
    new, evidence = t.est.after_lift(t.held, recorder, gap_before_mm=t.post.gap_mm, gap_after_mm=lifted.gap_mm)
    assert evidence["iou"] >= 0.8, evidence
    assert not evidence["touchdown"]["needed"], evidence
    assert new.silhouette is not None and new.vacated_cells is not None and len(new.vacated_cells) > 0
    names = {d["name"] for d in evidence["thresholds"]}
    assert {"M14_gap_change_mm", "M14_iou", "M14_shift_px"} <= names
    cells = new.vacated_cells
    assert t.scene.hf.known[cells[:, 0], cells[:, 1]].any()
    assert t.scene.vacate_held(new, 500) > 0
    assert not t.scene.hf.known[cells[:, 0], cells[:, 1]].any()
    import json
    json.dumps(evidence, allow_nan=False)


def test_relocalization_captures_a_close_drag():
    t = held_setup(close_drag=(0.0, 0.026))
    prior = np.asarray(t.held.offset_h, float)
    moved = t.est.relocalize(t.held, t.post)
    R = HandPose.from_view(t.post).R
    drag = R @ (np.asarray(moved.offset_h) - prior)
    assert np.linalg.norm(drag[:2] - np.array([0.0, 0.026])) <= 0.005, t.est.last_relocalize
    assert t.est.last_relocalize["status"] == "ok" and t.scene.held is moved


def test_tilt_in_the_pinch_triggers_m14_and_touchdown_sets_b_low_by_touch():
    t = held_setup()
    recorder, lifted = _lift(t, tilt_deg=15.0)
    new, evidence = t.est.after_lift(t.held, recorder, gap_before_mm=t.post.gap_mm, gap_after_mm=lifted.gap_mm)
    assert evidence["touchdown"]["needed"], evidence
    world = t.world
    for _ in range(60):
        z = world.grip[2]
        world.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
        if abs(world.grip[2] - z) < 2e-5:
            break
    g_touch = float(world.grip[2])
    touched = t.est.after_touchdown(new, g_touch, 0.001, pivot=False)
    truth = g_touch - _lowest(world)
    assert abs(touched.b_low.value - truth) <= 0.003 and touched.b_low.source == "touch"
    assert touched.touchdown["g_touch"] == g_touch and touched.touchdown["flags"] == []
    pivoted = t.est.after_touchdown(new, g_touch, 0.001, pivot=True)
    assert pivoted.touchdown["flags"] == ["touchdown_pivot"]
    assert pivoted.b_low.sigma >= scene_mod.M17_FRACTION * new.r_obj_m


def test_not_wrist_visible_uses_the_m17_offset_sigma():
    t = held_setup(yaw_sign=-1.0)
    assert not t.held.wrist_visible
    assert t.held.sigma_offset_m >= scene_mod.M17_FRACTION * t.held.r_obj_m
    recorder, lifted = _lift(t)
    new, evidence = t.est.after_lift(t.held, recorder, gap_before_mm=t.post.gap_mm, gap_after_mm=lifted.gap_mm)
    assert "not_wrist_visible" in evidence["touchdown"]["reasons"]
    m17 = [d for d in evidence["thresholds"] if d["name"] == "M17_sigma_offset_m"]
    assert m17 and abs(new.sigma_offset_m - m17[0]["value"]) < 1e-12
    assert new.sigma_offset_m >= scene_mod.M17_FRACTION * new.r_obj_m


def test_release_writes_the_held_object_at_its_new_place():
    """Review BS-R5: release_held(view) writes the hand-attached transforms into the records, so the placed object's
    xyz, parts_uv and region_mask follow it (here 8 cm from its pick place) instead of returning to the pick place."""
    t = held_setup()
    recorder, lifted = _lift(t)
    new, _ = t.est.after_lift(t.held, recorder, gap_before_mm=t.post.gap_mm, gap_after_mm=lifted.gap_mm)
    t.scene.vacate_held(new, 500)
    world = t.world
    pick_poly = np.asarray(t.scene.parts["D"].region_xy, float).copy()
    fakes.scripted_goto(world, world.grip + np.array([0.08, 0.0, 0.0]), gripper=1.0, quat_xyzw=t.q, tol_m=0.0005)
    carried = view_of(world)
    held_uv = t.scene.on_view(carried)["D"]["wrist"]
    held_xyz = t.scene._current_xyz("D", HandPose.from_view(carried))
    before = t.scene.parts["D"]           # a cut top's lower-bound pair (rulings R2) is cleared by the release
    t.scene.parts["D"] = dataclasses.replace(before, top_bound="lower", top_upper_z=before.top_z.value + 0.1)
    written = t.scene.release_held(carried)
    assert written.get("D") == "moved" and t.scene.held is None and "D" in t.scene.moved
    rec = t.scene.parts["D"]
    assert rec.top_bound is None and rec.top_upper_z is None and rec.top_band is None
    assert np.allclose(rec.xyz, held_xyz, atol=1e-9) and not rec.registered_empty_hand and rec.status == "registered"
    assert np.allclose(t.scene.on_view(carried)["D"]["wrist"], held_uv, atol=1e-6)
    assert np.linalg.norm(np.asarray(rec.region_xy).mean(axis=0) - pick_poly.mean(axis=0)) >= 0.07
    truth = world.truth_mask("wrist", "moka_body")
    mask = t.scene.region_mask("D", "wrist", carried)
    assert _iou(mask, truth) >= 0.6, _iou(mask, truth)
    pose = HandPose.from_view(carried)
    uv_old, _ = scene_mod._project(t.scene.camera, pose, np.column_stack([pick_poly, np.full(len(pick_poly),
                                                                                             rec.top_z.value)]))
    stale = np.zeros(truth.shape, np.uint8)
    cv2.fillPoly(stale, [np.rint(uv_old).astype(np.int32)], 1)
    assert _iou(stale > 0, truth) < 0.5 * _iou(mask, truth)
    assert t.scene.last_registration["D"]["source"] == "release"


def test_vacate_held_removes_the_object_cells_and_transit_no_longer_sees_them():
    t = held_setup()
    recorder, lifted = _lift(t)
    new, _ = t.est.after_lift(t.held, recorder, gap_before_mm=t.post.gap_mm, gap_after_mm=lifted.gap_mm)
    hf = t.scene.hf
    cells = new.vacated_cells
    own_top = float(np.nanmax(hf.z[cells[:, 0], cells[:, 1]]))
    assert own_top >= 1.02                                       # the object's top is in the field before
    assert t.scene.vacate_held(new, 600) > 0
    assert not hf.known[cells[:, 0], cells[:, 1]].any()
    assert scene_mod.SceneModel._current_xyz(t.scene, "D", HandPose.from_view(lifted)) is not None
    planner = Path(th.PACKAGE_DIR) / "agent_speedup_v2_planner.py"
    if not planner.is_file():
        pytest.skip("planner not built yet")
    from gpt_field import agent_speedup_v2_planner as planner_mod
    bare = dataclasses.replace(new, vacated_cells=None)         # the field alone must no longer hold the object
    pose = HandPose.from_view(lifted)
    path = planner_mod.transit_path(bare, pose.grip, (-0.0464, 0.2061), hf, [], RobotModel(), pose)
    assert path["swept_max_z"] < own_top - 0.03, path["swept_max_z"]


# ================================================================================================ API helpers
def test_on_view_parts_uv_prediction_survey_targets_and_bookkeeping(lever):
    scene, view, world = lever.scene, lever.view, lever.world
    out = scene.on_view(view, step=777)
    assert scene.step == 777 and set(out["A"]) == {"wrist", "external"}
    truth_uv = np.asarray(world.project("wrist", LEVER_TOP)[0])
    assert np.linalg.norm(np.asarray(out["A"]["wrist"]) - truth_uv) <= 6.0
    uv, sigma = scene.predicted_uv("A", view)
    assert np.allclose(uv, out["A"]["wrist"]) and 0.0 < sigma < 20.0
    euv, esigma = scene.predicted_uv("A", view, "external")
    assert np.linalg.norm(euv - world.project("external", LEVER_TOP)[0]) <= 3.0 * esigma + 6.0
    assert scene.predicted_uv("A", view, "side") is None and scene.predicted_uv("Q", view) is None
    targets = scene.survey_targets("A", view)
    assert targets["seeds"][0][0] == "A" and len(targets["relief"]) >= 2
    assert all(key.startswith("relief:A:") for key, _ in targets["relief"])
    pose = HandPose.from_view(view)
    for _, (u, v) in targets["relief"]:          # relief points lie off the part, beside it on its support
        assert not scene.region_mask("A", "wrist", view)[int(round(v)), int(round(u))]
    found = scene.reacquire_part("A", view)
    assert found["ok"] and np.linalg.norm(np.asarray(found["uv"]) - uv) <= 3.0
    assert scene.parts["A"].seeds[-1].source == "reacquire"
    keep = KeepOut("A", "articulated", np.array([[0, 0], [1, 0], [1, 1]], float), 0.9, 1.0, ())
    scene.add_keepout(keep)
    scene.add_keepout(dataclasses.replace(keep, z_high=1.1))
    assert list(scene.keepouts) == ["articulated:A"] and scene.keepouts["articulated:A"].z_high == 1.1
    import json
    json.dumps(scene.summary(), allow_nan=False)


def test_drop_failure_and_unknown_part_keep_the_records_consistent(thin):
    scene, view = thin.scene, thin.view
    assert isinstance(scene.register("H", view, empty_hand=True), RegistrationFailure)
    before = scene.parts["C"]
    plan_seed = Seed("wrist", (5.0, 5.0), 0, "plan_cell", 32.0, tuple(view.state))
    scene.parts["C"] = dataclasses.replace(before, seeds=list(before.seeds) + [plan_seed])
    failed = scene.register("C", view, seed=Seed("wrist", (2.0, 2.0), 0, "fine", 8.0, tuple(view.state)),
                            empty_hand=True)
    assert isinstance(failed, RegistrationFailure)
    assert scene.parts["C"].status == "registered" and scene.parts["C"].xyz is not None   # a failure keeps it
    scene.drop("C")
    dropped = scene.parts["C"]
    assert dropped.status == "dropped" and dropped.text == before.text and dropped.xyz is None
    assert [s.source for s in dropped.seeds] == ["plan_cell"]
    assert scene.predicted_uv("C", view) is None or len(scene.predicted_uv("C", view)) == 2


def test_estimate_seed_xyz_from_samples_and_from_the_height_field(aabb_scene):
    scene, view, world = aabb_scene.scene, aabb_scene.view, aabb_scene.world
    pose = HandPose.from_view(view)
    xyz, _, _ = scene._live_samples()
    uv, _ = scene_mod._project(scene.camera, pose, xyz)
    i = int(np.argmin(np.linalg.norm(uv - np.array([256.0, 200.0]), axis=1)))
    near = Seed("wrist", tuple(uv[i] + 2.0), 0, "fine", 8.0, tuple(view.state))
    point, cov = scene.estimate_seed_xyz(near, view)
    assert abs(point[2] - world.heights([point[:2]])[0]) <= 0.012 and cov.shape == (3, 3)
    plane_cell = np.argwhere(scene.hf.source == scene_mod._PLANE_CODE)[0]
    target = np.r_[scene.hf.cell_xy(plane_cell), scene.hf.z[tuple(plane_cell)]]
    far_uv = scene_mod._project(scene.camera, pose, target)[0][0]
    far = Seed("wrist", tuple(far_uv), 0, "coarse", 1e-3, tuple(view.state))
    lifted = scene.estimate_seed_xyz(far, view)
    assert lifted is not None and abs(lifted[0][2] - target[2]) <= 0.02


class MovingHandLocator(StandInLocator):
    """HandLocator duck type with the real one's precondition (geometry HandLocator.motion_check): a pair without hand
    motion gives no reading."""

    def locate(self, before_rgb, after_rgb, pose, *, after_pose=None):
        if after_pose is None or np.linalg.norm(after_pose.grip - pose.grip) < 0.002:
            self.last_locate = {"reason": "no_hand_motion"}
            return None, self.sigma_loc_px
        return super().locate(before_rgb, after_rgb, pose, after_pose=after_pose)


def test_external_mask_reanchors_from_an_earlier_moving_pair_when_the_hand_is_still():
    """Review BS-R6: at a grasp pose the hand is still, so the latest frame pair gives no HandLocator reading; the
    projector is re-anchored from the newest cached pair in which the hand moved (near the region)."""
    sc = fakes.planar_region_scene(centre_xy=(-0.20, 0.30))
    world = make_world(sc, (-0.20, 0.08, 1.19))
    truth = sc.info["truth"]
    scene, view, cal, _ = started_scene(world)
    scene.cal = dataclasses.replace(cal, locator=MovingHandLocator(world.external))
    poly = np.asarray(truth["polygon_xy"], float)
    scene.parts["B"] = scene_mod.PartRecord("B", "region", "fixed", frozenset({"ref"}), region_xy=poly,
                                            top_z=Meas(truth["top_z"], 0.002, "kappa"),
                                            xyz=np.r_[poly.mean(axis=0), truth["top_z"]], sigma_h_m=0.002,
                                            status="registered", registration_step=0)
    first = scene.projector
    fakes.scripted_goto(world, (-0.20, 0.23, 1.19), tol_m=0.0005)
    scene.on_view(view_of(world))
    for _ in range(3):                                   # moving toward the region, every view cached
        world.advance([0.0, 0.3, 0.0, 1e-9, 0.0, 0.0, -1.0])
        scene.on_view(view_of(world))
    for _ in range(4):                                   # then still (a close at the grasp pose)
        world.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
        scene.on_view(view_of(world))
    still = view_of(world)
    scene.on_view(still)
    mask = scene.region_mask("B", "external", still)
    log = scene.mask_log["projector"]["reanchor"]
    assert log["status"] == "reanchored" and scene.projector is not first and mask is not None, log
    assert log["reading_step"] < scene.step and np.linalg.norm(scene.projector.anchor_grip[:2] - poly.mean(axis=0)) < 0.15
    # a second mask needs no new reading: the region now lies within REANCHOR_RADIUS_M of the new anchor
    anchored = scene.projector
    scene.region_mask("B", "external", still)
    readings = [r for r in scene._locations if r.get("source") != "calibration"]      # the start anchor is kept too
    assert scene.projector is anchored and len(readings) == 1 and readings[0]["source"] == "cache_scan"


def test_a_flooding_external_refinement_falls_back_to_the_affine_mask(planar):
    """Review BS-R6: a refined external mask many times the affine one (here the whole image) is rejected; the affine
    mask stands, flagged in mask_log."""
    scene, view = planar.scene, planar.view
    flood = SimpleNamespace(**{k: getattr(VISION, k) for k in _VISION_KEYS})
    flood.refine_region_mask = lambda rgb, affine, seed, *, sigma_px: (np.ones_like(affine, bool), {"pixels": -1})
    saved = scene._vision
    scene._vision = flood
    try:
        mask = scene.region_mask("B", "external", view)
    finally:
        scene._vision = saved
    rec = scene.parts["B"]
    poly3 = np.column_stack([rec.region_xy, np.full(len(rec.region_xy), rec.top_z.value)])
    affine = scene.projector.mask(poly3, mask.shape)
    log = scene.mask_log["B"]["external"]
    assert np.array_equal(mask, affine) and log["flag"] == "refine_rejected" and log["sanity"]["refine_rejected"]


def test_external_mask_of_a_far_region_reanchors_the_projector_near_it():
    sc = fakes.planar_region_scene(centre_xy=(-0.20, 0.30))
    world = make_world(sc, (-0.20, 0.08, 1.19))
    truth = sc.info["truth"]
    scene, view, cal, _ = started_scene(world, survey_seeds=[("B", tuple(world.project("wrist", (-0.20, 0.30,
                                                                                               truth["top_z"]))[0]))])
    scene.parts["B"] = scene_mod.PartRecord("B", "region", "fixed", frozenset({"ref"}))
    poly = np.asarray(truth["polygon_xy"], float)
    scene.parts["B"] = dataclasses.replace(scene.parts["B"], region_xy=poly, top_z=Meas(truth["top_z"], 0.002, "kappa"),
                                           xyz=np.r_[poly.mean(axis=0), truth["top_z"]], sigma_h_m=0.002,
                                           status="registered", registration_step=0)
    first = scene.projector
    fakes.scripted_goto(world, (-0.20, 0.24, 1.19), tol_m=0.0005)
    scene.on_view(view_of(world))
    world.advance([0.2, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    moved = view_of(world)
    scene.on_view(moved)
    mask = scene.region_mask("B", "external", moved)
    log = scene.mask_log["projector"]["reanchor"]
    assert log["status"] == "reanchored" and scene.projector is not first and mask is not None
    assert np.linalg.norm(scene.projector.anchor_grip[:2] - poly.mean(axis=0)) < 0.15


# ============================================================================== real-data regression (BS review R1)
REAL_RUN = Path(th.PACKAGE_DIR).parents[1] / "runs" / "agent_speedup_v2_w1_bsfix_scene_t2_20261019T000812Z"


def _real_available():
    return all((REAL_RUN / f"{name}_ep{ep}.pkl").is_file() for ep in (0, 1, 2)
               for name in ("start", "hover_A", "hover_C"))


@functools.lru_cache(maxsize=None)
def _real_hover(ep):
    """Replay of the real task-2 dumps (BS fixer's scratch real-sim driver, zero model calls; ORACLE only for the
    scripted hover targets, the fine seed pixel and the scoring): Calibrate+-like start views -> scene.start ->
    the ORACLE plan cells -> start-view registrations, then the dumped hover views over A and C (ParallaxProbe-like
    +-2 cm survey) -> scene.register(seed = the ORACLE fine point). Returns the records and the dumped truth."""
    import pickle
    from gpt_field import agent_speedup_v2_geometry as geometry
    from gpt_field import agent_speedup_v2_plan as plan_mod
    start = pickle.loads((REAL_RUN / f"start_ep{ep}.pkl").read_bytes())
    scene = scene_mod.SceneModel(RobotModel())
    last = -1
    for step, obs in sorted(start["views"], key=lambda t: t[0]):
        if step > last:
            scene.on_view(StepView(obs), step=step)
            last = step
    view = StepView(start["view"])
    scene.on_view(view, step=last + 1)
    c = start["cal"]
    projector = None if start["anchor"] is None else geometry.ExternalProjector(c["external"], *start["anchor"])
    cal = V2Calibration(
        gripper=GripperCalibration.default(), sigma_close_mm=0.05, sigma_g_mm_per_step=0.002,
        free_yaw_rate_rad_per_step=0.0229, camera=c["camera"], external=c["external"], locator=c["locator"],
        projector=projector, fingertip_band=c["band"], false_change={},
        sigma_px={"wrist": start["sigma_px"], "external": start["sigma_px"]}, sigma_static_grey=start["sigma_px"],
        quarter_turns=start["k"], start_pose=tuple(view.state), start_quaternion=tuple(c["start_q"]),
        survey=c["result"], mount={}, thresholds=[])
    scene.start(cal, view)
    plan = plan_mod.parse_plan(start["plan"])
    roles = plan_mod.part_roles(plan)
    scene.merge_plan(plan, roles, view, start["k"])
    for pid in ("B", "D", "E", "A", "C"):
        provisional = "grip" in roles[pid] and not (roles[pid] - {"grip"})
        scene.register(pid, view, empty_hand=True, provisional=provisional, allow_partial=not provisional)
    out = {"truth": start["truth"]["widths"]}
    for pid in ("A", "C"):
        hover = pickle.loads((REAL_RUN / f"hover_{pid}_ep{ep}.pkl").read_bytes())
        for step, obs in sorted(hover["views"], key=lambda t: t[0]):
            if step > scene.step:
                scene.on_view(StepView(obs), step=step)
        after = StepView(hover["view"])
        scene.on_view(after, step=scene.step + 1)
        scene.absorb_survey(hover["result"], after)
        seed = Seed("wrist", tuple(hover["seed_uv"]), scene.step, "fine", 8.0,
                    tuple(StepView(hover["hover_view"]).state))
        rec = scene.register(pid, after, seed=seed, empty_hand=True)
        out[pid] = {"rec": rec, "event": dict(scene.last_registration[pid]),
                    "iou_visible": _iou(scene.region_mask(pid, "wrist", after), hover["truth_visible"])
                    if isinstance(rec, scene_mod.PartRecord) else None}
    return out


def _across_truth(rec, truth):
    axis = np.asarray(truth["axis"], float)
    return rec.width_at(math.atan2(axis[1], axis[0]) + math.pi / 2).value


@pytest.mark.skipif(not _real_available(), reason="real task-2 dumps absent (BS fixer run dir)")
@pytest.mark.parametrize("ep", [0, 1, 2])
def test_real_task2_lever_at_its_hover_is_elongated_with_its_width(ep):
    """Review BS-R1 (real LIBERO task 2, episodes 0-2): the lever registered at its grasp hover is the whole lever
    face (its bevels included, cut by the fingers at one end only), elongated (D10), with the width across its axis
    24.3 +- 3 mm (truth: the ORACLE region corners), its axis within 5 deg and its top within 5 mm of the collision
    top; the builder's region was 5-17 mm wide and failed D10 or the width in every episode."""
    r = _real_hover(ep)
    rec, event, truth = r["A"]["rec"], r["A"]["event"], r["truth"]["A"]
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered" and not rec.partial, event
    assert rec.is_elongated(), event
    assert abs(_across_truth(rec, truth) - truth["width_mm"] / 1000.0) <= 0.003, (_across_truth(rec, truth), truth)
    assert angle_deg(rec.elongation[0], truth["axis"]) <= 5.0
    assert abs(rec.top_z.value - truth["top_z"]) <= 0.005
    assert event["cut_kind"] in ("none", "end") and r["A"]["iou_visible"] >= 0.4
    assert rec.rigid_patch is not None and "track" in rec.rigid_patch


@pytest.mark.skipif(not _real_available(), reason="real task-2 dumps absent (BS fixer run dir)")
@pytest.mark.parametrize("ep", [0, 1, 2])
def test_real_task2_thin_part_at_its_hover_has_its_width(ep):
    """Review BS-R1: the thin part (13.3 mm across) registered at its hover keeps its width across its axis within
    3 mm (the builder's region took the part's descending curve too: 20-24 mm) and its region aspect."""
    r = _real_hover(ep)
    rec, event, truth = r["C"]["rec"], r["C"]["event"], r["truth"]["C"]
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", event
    assert abs(_across_truth(rec, truth) - truth["width_mm"] / 1000.0) <= 0.003, (_across_truth(rec, truth), truth)
    assert abs(rec.top_z.value - truth["top_z"]) <= 3.0 * rec.sigma_h_m + 0.005


@pytest.mark.skipif(not _real_available(), reason="real task-2 dumps absent (BS fixer run dir)")
@pytest.mark.parametrize("ep", [0, 1, 2])
def test_real_task2_accepted_axes_are_never_wrong_by_more_than_half_the_yaw_grid(ep):
    """Decision §2.2 S-4' on the real task-2 dumps (G-yaw, decision §4: 0 accepted axes wrong by more than
    YAW_GRID_RAD / 2): every accepted D10 axis lies within 7.5 deg of the truth axis and carries an accepted
    axis_check; the lever keeps D10 in every episode. The thin part's refusals are explicit: episode 0 aspect 1.36 and
    episode 2 coherence 0.44 (not_elongated, as before), and episode 1, whose deciding image tensor is 32.7 deg off
    (coherence 0.77, 45.8 deg from the region cells' PCA), is refused axis_inconsistent (the wave-1 rule had replaced
    the tensor by the region's pixel PCA and accepted it); BR-1's second hover / P-C refine it."""
    from gpt_field.agent_speedup_v2_types import AXIS_AGREE_MAX_RAD, AXIS_INCONSISTENT
    r = _real_hover(ep)
    for pid in ("A", "C"):
        rec, event, truth = r[pid]["rec"], r[pid]["event"], r["truth"][pid]
        assert isinstance(rec, scene_mod.PartRecord), event
        check = rec.axis_check
        assert check is not None and check["accepted"] == rec.is_elongated(), check
        if rec.is_elongated():
            assert math.radians(angle_deg(rec.elongation[0], truth["axis"])) <= AXIS_AGREE_MAX_RAD, (pid, check)
            assert check["agree_rad"] <= AXIS_AGREE_MAX_RAD and check["cells"] >= scene_mod.D10_MIN_REGION_CELLS
        else:
            assert check["source"] in (AXIS_INCONSISTENT, "not_elongated"), check
    assert r["A"]["rec"].is_elongated()
    if ep == 1:
        assert r["C"]["rec"].axis_check["source"] == AXIS_INCONSISTENT


# ================================================================ design review 1 (decision §2.2, §5.1 item 2)
def dense_survey(world, view, centre_uv, *, step_px=4, window_px=40, sigma_m=0.001, noise_m=0.0005, radius_px=5,
                 min_std=1.0, extra=(), extra_keys="dense", rng=None, step=0):
    """A probe-tracker-like hover survey (decision §2.3 G-1 stand-in): exact lattice seeds every ``step_px`` within
    ``window_px`` of ``centre_uv``, keyed "dense:<k>", 11 px templates (``radius_px``) that do not straddle a depth step
    and have std >= ``min_std``, depth from the fake's depth map with ``noise_m`` noise and a claimed ``sigma_m``.
    ``extra``: world points of mis-depthed samples (an occlusion layer), each placed at its own projected pixel, keyed
    "dense:<k>" (``extra_keys`` "dense") or as grid samples (None)."""
    rng = np.random.default_rng(7) if rng is None else rng
    gray = np.asarray(view.wrist_gray_upright, float)
    depth = world.depth_map("wrist")
    pose = HandPose.from_view(view)
    cam = WristCamera(F_PX)
    h, w = gray.shape
    r = radius_px
    features, samples = [], []
    k = 0
    for dv in range(-window_px, window_px + 1, step_px):
        for du in range(-window_px, window_px + 1, step_px):
            if du * du + dv * dv > window_px * window_px:
                continue
            u, v = int(round(centre_uv[0] + du)), int(round(centre_uv[1] + dv))
            if not (r <= u < w - r and r <= v < h - r):
                continue
            patch, zpatch = gray[v - r:v + r + 1, u - r:u + r + 1], depth[v - r:v + r + 1, u - r:u + r + 1]
            if patch.std() < min_std or not np.isfinite(zpatch).all() or (zpatch.max() - zpatch.min()) / depth[v, u] > 0.01:
                continue
            z = float(depth[v, u]) + noise_m * rng.standard_normal()
            xyz = cam.backproject(pose, (float(u), float(v)), z)
            fid = 20000 + k
            samples.append(DepthSample(tuple(xyz), sigma_m, sigma_m / 2, "kappa", (float(u), float(v)), fid, step, F_PX / z))
            features.append(FeatureRecord(fid, f"dense:{k}", (float(u), float(v)), (float(u), float(v)), 5, F_PX / z,
                                          None, z, sigma_m, tuple(xyz), True, None))
            k += 1
    for j, xyz in enumerate(extra):
        uv = world.project("wrist", np.asarray(xyz, float))[0]
        fid = 30000 + j
        key = f"dense:{k + j}" if extra_keys == "dense" else None
        samples.append(DepthSample(tuple(map(float, xyz)), sigma_m, sigma_m / 2, "kappa", (float(uv[0]), float(uv[1])),
                                   fid, step, None))
        features.append(FeatureRecord(fid, key, tuple(uv), tuple(uv), 5, None, None, None, sigma_m, tuple(xyz), True,
                                      None))
    state = tuple(float(x) for x in view.state)
    return SurveyResult(state, state, {"probe": 5}, features, samples, None, [], {})


def test_top_band_rule_ignores_strays_and_lower_layers_and_sets_the_model_term():
    """Remedy 1 test 6 (decision §2.2 S-1, S-3): 25 samples at z0 (sigma 1 mm), 30 at z0 - 15 mm and 2 strays at
    z0 + 20 mm: the band top lies within 1 mm of z0 (the strays have no TOP_BAND_K partners); a band of < 10 members
    takes TOP_MODEL_THIN_M; no band falls back to _top_cluster."""
    rng = np.random.default_rng(11)
    z0 = 1.0345
    z = np.r_[z0 + 0.0005 * rng.standard_normal(25), z0 - 0.015 + 0.0005 * rng.standard_normal(30), [z0 + 0.02] * 2]
    s = np.full(len(z), 0.001)
    top, members, info = scene_mod._top_band(z, s)
    assert info["rule"] == "band" and abs(top - z0) <= 0.001 and info["n"] == 25 and set(members) <= set(range(25))
    assert info["model_m"] == scene_mod.TOP_MODEL_M and info["stat_m"] <= 0.001 / math.sqrt(25) * 1.5 + 1e-12
    assert math.hypot(info["stat_m"], info["model_m"]) >= scene_mod.TOP_MODEL_M
    thin_top, thin_members, thin = scene_mod._top_band(z[18:], s[18:])          # 7 top members
    assert thin["rule"] == "band" and thin["n"] == 7 and thin["model_m"] == scene_mod.TOP_MODEL_THIN_M
    assert abs(thin_top - z0) <= 0.001
    spread = np.array([0.90, 0.95, 1.00, 1.05])                                    # no 3 samples within 3 mm
    top_c, keep_c, info_c = scene_mod._top_band(spread, np.full(4, 0.001))
    ref = scene_mod._top_cluster(spread, np.full(4, 0.001))
    assert info_c["rule"] == "top_cluster" and top_c == ref[0] and list(keep_c) == list(ref[2])
    assert scene_mod._top_band(np.zeros(0), np.zeros(0))[2]["rule"] is None


@pytest.fixture(scope="module")
def thin_hover():
    """The thin part's hover with a probe-tracker-like survey (dense lattice around the pointed pixel) plus a lower
    layer: 20 samples within 6 px of the seed placed 15 mm deeper along their rays (the layered dark loop whose small
    templates lock onto a lower segment), so the quantization median sits on the lower layer."""
    sc = fakes.free_object_with_thin_part_scene()
    world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
    uv0 = world.project("wrist", LOOP_TOP)[0]
    scene, view, cal, info = started_scene(world, survey_seeds=[("C", tuple(uv0))])
    after = probe(world, scene)
    uv_seed = np.asarray(world.project("wrist", LOOP_TOP)[0], float)
    rng = np.random.default_rng(5)
    pose, cam, depth = HandPose.from_view(after), WristCamera(F_PX), world.depth_map("wrist")
    lower = []
    for _ in range(20):
        u, v = uv_seed + rng.uniform(-6.0, 6.0, size=2)
        lower.append(cam.backproject(pose, (float(u), float(v)), float(depth[int(round(v)), int(round(u))]) + 0.015))
    survey = dense_survey(world, after, uv_seed, extra=lower, rng=rng)
    scene.absorb_survey(survey, after)
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    seed = seed_at(world, after, LOOP_TOP)
    return SimpleNamespace(world=world, scene=scene, view=after, seed=seed, uv=uv_seed, survey=survey, sc=sc)


def test_point_seed_lifts_onto_the_top_band_and_takes_an_honest_sigma_h(thin_hover):
    """Remedy 1 test 7 (decision §2.2 S-2, S-3): on the thin part with a lower layer under the seed, the fine seed is
    lifted onto the top band (the quantization median would put it on the lower layer: an area seed at the same pixel
    keeps that rule), top_z lies within 1 mm of the top with sigma_h >= TOP_MODEL_M = hypot(stat, model) and the
    record carries the top band (K-1); the event logs the band counts."""
    t = thin_hover
    scene, view = t.scene, t.view
    point, cov = scene.estimate_seed_xyz(t.seed, view)
    assert abs(point[2] - LOOP_TOP[2]) <= 0.001, point
    assert math.sqrt(cov[2, 2]) >= scene_mod.TOP_MODEL_M * 0.9               # the depth sigma is sigma_h along the ray
    area = Seed("wrist", t.seed.uv, t.seed.step, "coarse", t.seed.quantization_px, t.seed.pose)
    area_point, _ = scene.estimate_seed_xyz(area, view)
    assert area_point[2] < LOOP_TOP[2] - 0.008, area_point                   # the unchanged median rule: lower layer
    rec = scene.register("C", view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", scene.last_registration["C"]
    band = rec.top_band
    assert abs(rec.top_z.value - LOOP_TOP[2]) <= 0.001 and rec.sigma_h_m >= scene_mod.TOP_MODEL_M
    assert band["n"] >= scene_mod.TOP_BAND_SUPPORT_N and band["model_m"] == scene_mod.TOP_MODEL_M
    assert rec.sigma_h_m >= math.hypot(band["stat_m"], band["model_m"]) - 1e-12
    event = scene.last_registration["C"]
    assert event["seed_band"]["rule"] == "band" and event["top_band"]["n"] == band["n"]
    assert {"n_window", "n_gated", "n", "stat_m", "model_m"} <= set(event["seed_band"])
    names = {th_.name for th_ in scene.registration_thresholds("C")}
    assert "D4_top_model_m" in names and "D10_axis_agree_rad" in names
    import json
    json.dumps(event, allow_nan=False)
    json.dumps(rec.to_dict(), allow_nan=False)


def test_second_hover_fusion_raises_the_band_support_and_lowers_sigma_h(thin_hover):
    """Remedy 1 test 9 (decision §2.2 S-5 of remedy 1 = BR-1 fusion): two surveys of 6 + 6 top-band members fuse into
    one band of >= 10 members and sigma_h drops from ~5 mm (TOP_MODEL_THIN_M) to ~3 mm (TOP_MODEL_M)."""
    sc = fakes.free_object_with_thin_part_scene()
    world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
    scene, view, _, _ = started_scene(world)
    seed = seed_at(world, view, LOOP_TOP)
    rng = np.random.default_rng(3)

    def survey(offset):
        samples, features = [], []
        for j in range(6):
            xyz = (LOOP_TOP[0] + rng.uniform(-0.003, 0.003), LOOP_TOP[1] + offset + 0.004 * j,
                   LOOP_TOP[2] + 0.0004 * rng.standard_normal())
            uv = world.project("wrist", np.asarray(xyz))[0]
            fid = 100 + j
            samples.append(DepthSample(xyz, 0.001, 0.0005, "kappa", tuple(uv), fid, 0, None))
            features.append(FeatureRecord(fid, f"dense:{j}", tuple(uv), tuple(uv), 5, None, None, None, 0.001, xyz,
                                          True, None))
        return SurveyResult(tuple(view.state), tuple(view.state), {}, features, samples, None, [], {})
    assert (scene._seed_band(seed) or {}).get("rule") != "band"                  # the grid samples are not gated in
    scene.absorb_survey(survey(-0.012), view)
    first = scene._seed_band(seed)
    assert first["rule"] == "band" and first["n"] == 6 and first["model_m"] == scene_mod.TOP_MODEL_THIN_M
    assert 0.005 <= first["sigma_h"] <= 0.0052
    scene.absorb_survey(survey(-0.010), view)
    fused = scene._seed_band(seed)
    assert fused["n"] >= scene_mod.TOP_BAND_SUPPORT_N and fused["model_m"] == scene_mod.TOP_MODEL_M
    assert 0.003 <= fused["sigma_h"] <= 0.0031 and abs(fused["top"] - LOOP_TOP[2]) <= 0.001


def test_d10_oe_ignores_the_retired_structure_tensor():
    """Decision 2 §2.3 S-11: the structure tensor (and S-4''s cell arbiter) no longer decides D10. A 3 mm sliver of
    the 13 mm thin part with a sheared outline, registered once with a stub image tensor 30 deg off the part
    (coherence 1.0) and once 2 deg off, gets the same D10-OE verdict and axes; the tensor stays logged in axis_check
    as diagnosis (``tensor_axis``, ``coherence``)."""
    def run(tensor_deg):
        sc = fakes.free_object_with_thin_part_scene()
        world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
        scene, _, _, _ = started_scene(world, survey_seeds=[("C", tuple(world.project("wrist", LOOP_TOP)[0]))])
        view = probe(world, scene)
        pose = HandPose.from_view(view)
        # the sliver: 3 mm across the part (x), 44 mm along it (y), sheared by 4 mm along its length
        corners = [(-0.0015, -0.022, -0.002), (0.0015, -0.022, -0.002), (0.0015, 0.022, 0.002),
                   (-0.0015, 0.022, 0.002)]
        pts = np.array([[LOOP_TOP[0] + dx + shear, LOOP_TOP[1] + dy, LOOP_TOP[2]] for dx, dy, shear in corners])
        uv = scene_mod._project(scene.camera, pose, pts)[0]
        sliver = np.zeros((512, 512), np.uint8)
        cv2.fillConvexPoly(sliver, np.rint(uv).astype(np.int32), 1)
        a = math.radians(tensor_deg)
        stub = SimpleNamespace(**{k: getattr(VISION, k) for k in _VISION_KEYS})
        stub.structure_axis = lambda gray, uv_, half_px=12: (np.array([math.cos(a), math.sin(a)]), 1.0, 1.0)
        scene._vision = stub
        scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], sliver > 0)
        scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
        return scene.register("C", view, seed=seed_at(world, view, LOOP_TOP), empty_hand=True)
    off, along = run(30.0), run(2.0)
    assert isinstance(off, scene_mod.PartRecord) and isinstance(along, scene_mod.PartRecord), (off, along)
    a, b = off.axis_check, along.axis_check
    assert a["method"] == b["method"] == "outline_edges"
    assert (a["accepted"], a["source"], a["reason"]) == (b["accepted"], b["source"], b["reason"]), (a, b)
    assert angle_deg(a["image_axis"], b["image_axis"]) <= 1e-9 and off.is_elongated() == along.is_elongated()
    assert a["tensor"] == b["tensor"] == "image" and a["coherence"] == b["coherence"] == 1.0
    assert abs(angle_deg(a["tensor_axis"], b["tensor_axis"]) - 28.0) <= 1.0                   # logged, not decided


@pytest.fixture(scope="module")
def occlusion_layer():
    """The thin part's hover with an occlusion-layer fake: the dense lattice around the pointed pixel plus 12 dense
    samples at the part's height placed 15-25 mm beside its side (across, +x; the decision's live blockers), and 6 GRID
    samples at the same height farther out (a real raised neighbour's grid samples, the kind DSW never withdraws); then
    the point-seeded register."""
    sc = fakes.free_object_with_thin_part_scene()
    world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
    uv0 = world.project("wrist", LOOP_TOP)[0]
    scene, view, cal, info = started_scene(world, survey_seeds=[("C", tuple(uv0))])
    after = probe(world, scene)
    rng = np.random.default_rng(9)
    side = sc.info["truth"]["loop_width_m"] / 2.0
    phantom = [(LOOP_TOP[0] + side + d, LOOP_TOP[1] + along, LOOP_TOP[2] + 0.0005 * rng.standard_normal())
               for d, along in zip(np.linspace(0.015, 0.025, 12), np.tile([-0.012, -0.004, 0.004, 0.012], 3))]
    lattice = dense_survey(world, after, np.asarray(world.project("wrist", LOOP_TOP)[0], float), extra=phantom,
                           rng=rng)
    grid_neighbour = [(LOOP_TOP[0] + side + 0.040, LOOP_TOP[1] + along, LOOP_TOP[2]) for along in
                      (-0.015, -0.009, -0.003, 0.003, 0.009, 0.015)]
    grid = dense_survey(world, after, (0.0, 0.0), window_px=0, extra=grid_neighbour, extra_keys=None, rng=rng)
    scene.absorb_survey(grid, after)
    scene.absorb_survey(lattice, after)
    hf = scene.hf
    phantom_cells = {hf.index(p[:2]) for p in phantom} - {None}
    grid_cells = {hf.index(p[:2]) for p in grid_neighbour} - {None}
    measured_before = {c: bool(hf.measured[c]) for c in phantom_cells | grid_cells}
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    rec = scene.register("C", after, seed=seed_at(world, after, LOOP_TOP), empty_hand=True)
    return SimpleNamespace(world=world, scene=scene, view=after, rec=rec, phantom=phantom, phantom_cells=phantom_cells,
                           grid_cells=grid_cells, measured_before=measured_before)


def test_dense_probe_samples_off_the_part_are_withdrawn_and_refilled(occlusion_layer):
    """Decision §2.2 S-5 (DSW): after the point-seeded registration the dense-lattice samples above support + D8 floor
    outside the region dilated by one cell are withdrawn: their cells are vacated and re-filled as D7 fills (unverified),
    the samples stay registration markers (live, marker-only), the grid samples of the raised neighbour are never
    withdrawn and keep their cells measured, the part's own dense samples stay, and the log has the counts."""
    t = occlusion_layer
    scene, rec, hf = t.scene, t.rec, t.scene.hf
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", scene.last_registration["C"]
    assert all(t.measured_before[c] for c in t.phantom_cells | t.grid_cells)        # they were obstacles before
    log = scene.last_dsw["C"]
    assert scene.last_registration["C"]["dsw"] == log and log["skipped"] is None and log["surveys"] == 1, log
    assert log["withdrawn"] >= len(t.phantom) and log["affected"] >= len(t.phantom_cells) and log["fill"] is not None
    for c in t.phantom_cells:              # no measured cell keeps the phantom height (a real sample may rebuild it)
        assert not (hf.measured[c] and abs(hf.z[c] - LOOP_TOP[2]) <= 0.003), (c, hf.source[c], hf.z[c])
    refilled = [c for c in t.phantom_cells if hf.known[c] and not hf.measured[c]]
    assert len(refilled) >= 0.7 * len(t.phantom_cells), [hf.source[c] for c in t.phantom_cells]
    for c in t.grid_cells:
        assert hf.measured[c] and abs(hf.z[c] - LOOP_TOP[2]) <= 0.003, (c, hf.source[c], hf.z[c])
    s = scene._samples
    assert not (s["withdrawn"] & ~s["dense"]).any()                                   # grid / relief / start never
    xyz = s["xyz"]
    phantom_idx = [int(np.argmin(np.linalg.norm(xyz - np.asarray(p)[None, :], axis=1))) for p in t.phantom]
    assert all(s["withdrawn"][i] and s["marker"][i] for i in phantom_idx)
    live = scene._live_mask()
    assert all(live[i] for i in phantom_idx)                                         # still registration markers
    region = np.zeros(hf.shape, bool)
    region[rec.region_cells[:, 0], rec.region_cells[:, 1]] = True
    iy, ix, inside = scene._sample_cells()
    on_part = s["dense"] & inside
    on_part[inside] &= region[iy[inside], ix[inside]]
    assert on_part.any() and not (s["withdrawn"] & on_part).any()
    # a second point-seeded registration withdraws nothing new (the samples are withdrawn once)
    again = scene.register("C", t.view, seed=seed_at(t.world, t.view, LOOP_TOP), empty_hand=True)
    assert isinstance(again, scene_mod.PartRecord) and scene.last_dsw["C"]["withdrawn"] == 0


def test_part_band_fills_are_the_parts_halo_and_a_separate_raised_object_keeps_its_fills():
    """Decision §2.2 S-6 (PB): part_fill_cells returns the D7b fills whose parent is a measured cell in the part's top
    band, 8-connected to the registered region through raised cells and within PB_REACH_M of the record point; fills
    of a separate raised object at the same height (not connected) and of a band cell beyond the reach are not."""
    scene = scene_mod.SceneModel(RobotModel(), vision=VISION)
    hf = scene_mod.HeightField.empty((0.0, 0.0), half_extent_m=0.1)
    scene.hf = hf
    scene._fill_parent = np.full(hf.shape + (2,), -1, np.int32)
    scene._plane_member = np.full(hf.shape, -1, np.int16)
    top, support = 1.0345, 0.900
    for ij in np.ndindex(hf.shape):
        hf.stamp_ij(ij, support, 0.001, "kappa", 0)
    c = hf.index((0.0, 0.0))
    region = [(c[0] + dy, c[1]) for dy in range(-2, 3)]                          # a 1 x 5 cell bar along y
    halo = [(c[0], c[1] + 1), (c[0], c[1] + 2)]                                  # its measured height spread (+x)
    chain = [(c[0], c[1] + k) for k in range(3, 12)]                             # raised, connected, beyond 5 cm
    island = [(c[0] + 6, c[1] - 4), (c[0] + 6, c[1] - 5)]                        # separate object, same height
    for ij in region + halo + chain + island:
        hf.stamp_ij(ij, top - 0.001, 0.001, "kappa", 0)
    rec = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}), xyz=np.r_[hf.cell_xy(c), top],
                               cov=np.diag([1e-6, 1e-6, 9e-6]), top_z=Meas(top, 0.003, "kappa"),
                               support_z=Meas(support, 0.001, "kappa"), region_cells=np.array(region),
                               region_xy=np.array([[-0.003, -0.013], [0.003, -0.013], [0.003, 0.013], [-0.003, 0.013]]),
                               sigma_h_m=0.003, status="registered", registration_step=0)
    scene.parts["C"] = rec
    fills = {"from_region": (c[0] + 1, c[1] - 1), "from_halo": (c[0] - 1, c[1] + 2),
             "from_island": (c[0] + 7, c[1] - 5), "from_far_chain": (c[0] + 1, c[1] + 11)}
    parents = {"from_region": region[3], "from_halo": halo[1], "from_island": island[1], "from_far_chain": chain[-1]}
    for name, ij in fills.items():
        hf.vacate([ij], 1)
        hf.stamp_ij(ij, top + 0.009, 0.003, "obstacle", 1)
        scene._fill_parent[ij] = parents[name]
    got = {tuple(map(int, ij)) for ij in scene.part_fill_cells("C")}
    assert fills["from_region"] in got and fills["from_halo"] in got, got
    assert fills["from_island"] not in got and fills["from_far_chain"] not in got, got
    log = scene.part_fill_log["C"]
    assert log["fills"] == len(got) and log["reach_m"] == scene_mod.PB_REACH_M and log["skipped"] is None
    assert scene.part_fill_cells("H").shape == (0, 2) and scene.part_fill_log["H"]["skipped"] == "not_registered"


def test_part_band_fills_after_dense_withdrawal_on_the_thin_part(occlusion_layer):
    """S-5 + S-6 on the occlusion-layer fake: the refilled D7b cells beside the part that copy its height are the
    part's halo (returned by part_fill_cells), as the decision's live chain used them in P7."""
    t = occlusion_layer
    cells = t.scene.part_fill_cells("C")
    hf = t.scene.hf
    assert len(cells) and all(hf.source[tuple(ij)] == scene_mod._OBSTACLE_CODE for ij in cells)
    assert set(map(tuple, t.scene.derived_fills(t.rec.region_cells))) <= set(map(tuple, cells)) | set()


def test_support_must_lie_below_the_lowest_top_band_member():
    """Decision §2.2 S-7: a part band above a dark plane whose ring majority is a band level just below the part (its
    side samples) yields the plane below, not the band level; with no plane below the support stays unmeasured; and
    without a top band the wave-1 rule is unchanged."""
    scene = scene_mod.SceneModel(RobotModel(), vision=VISION)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    part = (xx - 100) ** 2 + (yy - 100) ** 2 <= 20 ** 2
    ring = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 75 ** 2) & ~part                  # covers the 15 mm support ring
    labels = np.where(part, 2, np.where(ring, 1, 0)).astype(np.int32)
    bands = [{"z": 0.918, "sigma": 0.001, "n": 40, "measured": 0, "plane": True, "members": np.zeros(0, int)},
             {"z": 0.950, "sigma": 0.001, "n": 9, "measured": 9, "plane": False, "members": np.zeros(0, int)},
             {"z": 0.955, "sigma": 0.001, "n": 30, "measured": 30, "plane": False, "members": np.zeros(0, int)}]
    seg = {"labels": labels, "bands": bands}
    pose = HandPose.from_state8((0.0, 0.0, 1.2, math.pi, 0.0, 0.0, 0.04, -0.04))
    kappa = 3000.0
    b, meas, _, check = scene._support(part, seg, 2, kappa, pose, floor=(0.954, 0.001))
    assert b == 0 and abs(meas.value - 0.918) < 1e-9, check
    assert check["rejected"] and abs(check["rejected"][0]["z"] - 0.950) < 1e-9 and check["fallback"] == "plane_below"
    b_old, meas_old, _, check_old = scene._support(part, seg, 2, kappa, pose)
    assert b_old == 1 and abs(meas_old.value - 0.950) < 1e-9 and not check_old["rejected"]
    no_plane = dict(seg, bands=[dict(bands[0], plane=False)] + bands[1:])
    b_none, meas_none, region_none, check_none = scene._support(part, no_plane, 2, kappa, pose, floor=(0.954, 0.001))
    assert b_none is None and meas_none is None and check_none["fallback"] == "unmeasured"


def test_obstacle_fill_parent_sigma_is_capped_at_the_cell_size(aabb_scene):
    """Remedy 2 test 11 (decision §2.2 S-8, RC-9a): a D7b parent with a 20 mm sigma fills at <= parent + 15 mm
    (D7_OBSTACLE_SIGMA_K x D7_PARENT_SIGMA_CAP_M), not + 60 mm; the cap is logged."""
    sc = fakes.textureless_dark_aabb_scene()
    world = make_world(sc, (-0.20, 0.019, 1.19))
    scene, view, _, _ = started_scene(world)
    hf = scene.hf
    obstacle = np.argwhere(hf.source == scene_mod._OBSTACLE_CODE)
    assert len(obstacle)
    o = tuple(obstacle[len(obstacle) // 2])
    p = tuple(scene._fill_parent[o])
    hf.sigma[p] = 0.020                                     # a parent measured with a 20 mm sigma
    fills = np.argwhere(hf.source == scene_mod._OBSTACLE_CODE)
    hf.vacate(fills, 2)
    info = scene.fill_unknown(view)
    near = [tuple(ij) for ij in np.argwhere(hf.source == scene_mod._OBSTACLE_CODE) if tuple(scene._fill_parent[tuple(ij)]) == p]
    assert near, "no fill derived from the 20 mm parent"
    for ij in near:
        assert hf.z[ij] <= hf.z[p] + scene_mod.D7_OBSTACLE_SIGMA_K * scene_mod.D7_PARENT_SIGMA_CAP_M + 1e-9
        assert hf.z[ij] <= hf.z[p] + 0.015 + 1e-9 and hf.sigma[ij] <= scene_mod.D7_PARENT_SIGMA_CAP_M + 1e-12
    cap = [d for d in info["thresholds"] if d["name"] == "D7_parent_sigma_cap_m"]
    assert cap and cap[0]["value"] == scene_mod.D7_PARENT_SIGMA_CAP_M and info["parent_sigma_capped"] >= len(near)


def test_refinement_absorb_stamps_only_the_target_region():
    """Remedy 2 test 12 (decision §2.2 S-8, RC-9b): absorb_survey(refine_pid) writes no cell outside the target's
    region dilated by 3 sigma_xy; the other samples stay registration markers."""
    sc = fakes.textureless_dark_aabb_scene()
    world = make_world(sc, (-0.20, 0.019, 1.19))
    scene, view, _, _ = started_scene(world)
    hf = scene.hf
    poly = np.array([[-0.16, 0.03], [-0.14, 0.03], [-0.14, 0.05], [-0.16, 0.05]])
    scene.parts["A"] = scene_mod.PartRecord("A", "part", "fixed", frozenset({"grip"}), xyz=np.array([-0.15, 0.04, 0.93]),
                                            cov=np.diag([1e-6, 1e-6, 1e-6]), region_xy=poly, sigma_h_m=0.001,
                                            top_z=Meas(0.93, 0.001, "kappa"), status="registered", registration_step=0)
    inside_xy, outside_xy = [(-0.15, 0.04), (-0.145, 0.035)], [(-0.25, -0.05), (-0.10, 0.10), (-0.20, 0.02)]
    samples = [DepthSample((x, y, 0.95), 0.001, 0.0005, "kappa", None, None, 3, None) for x, y in inside_xy + outside_xy]
    before = {hf.index(xy): (hf.source[hf.index(xy)], hf.z[hf.index(xy)], hf.stamp[hf.index(xy)]) for xy in outside_xy}
    survey = SurveyResult(tuple(view.state), tuple(view.state), {}, [], samples, None, [], {})
    scene.absorb_survey(survey, view, refine_pid="A")
    for xy in inside_xy:
        assert abs(hf.z[hf.index(xy)] - 0.95) < 1e-9
    for xy in outside_xy:
        ij = hf.index(xy)
        src, z, stamp = before[ij]
        assert hf.source[ij] == src and hf.stamp[ij] == stamp and (np.isnan(z) and np.isnan(hf.z[ij]) or hf.z[ij] == z)
    s = scene._samples
    assert s["marker"][-3:].all() and not s["marker"][-5:-3].any() and s["alive"][-5:].all()
    assert scene._survey_info[-1]["refine"] == {"pid": "A", "area": True, "stamped": 2, "markers_only": 3}


class SettledPairLocator(StandInLocator):
    """A HandLocator duck type with the real one's motion precondition: a pair whose hand moved less than 2 cm (a
    1-step pair of a descent: ~1 px in the external image) gives no reading."""

    def locate(self, before_rgb, after_rgb, pose, *, after_pose=None):
        if after_pose is None or np.linalg.norm(after_pose.grip - pose.grip) < 0.02:
            self.last_locate = {"reason": "motion_too_small"}
            return None, self.sigma_loc_px
        self.last_locate = {"reason": None}
        return super().locate(before_rgb, after_rgb, pose, after_pose=after_pose)


def _arrival_descent(world, scene, *, travel_m=0.03):
    """A settled hover (two views at one pose) then a guarded-like 3 cm vertical descent, every view cached; returns
    (settled top view, bottom view, largest 1-step motion)."""
    scene.on_view(view_of(world))
    for _ in range(2):
        world.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
        scene.on_view(view_of(world))
    top = scene._recent[-1][1]
    z0, largest = float(world.grip[2]), 0.0
    for _ in range(60):
        g = world.grip.copy()
        world.advance([0.0, 0.0, -0.3, 1e-9, 0.0, 0.0, -1.0])
        largest = max(largest, float(np.linalg.norm(world.grip - g)))
        scene.on_view(view_of(world))
        if z0 - float(world.grip[2]) >= travel_m:
            break
    return top, scene._recent[-1][1], largest


def test_reanchor_from_a_settled_three_cm_descent():
    """Decision §2.2 S-9 / rulings item 4: reanchor_from(settled top view, hover view after the 3 cm descent) takes a
    HandLocator reading on the pair, located on the hover view (the hand moved up the image from it: the locator's
    sharp case), keeps it and re-anchors the projector (its anchor lies beyond REANCHOR_RADIUS_M); a same-pose pair
    and a missing locator are refused with their reasons."""
    sc = fakes.planar_region_scene(centre_xy=(-0.20, 0.30))
    world = make_world(sc, (-0.20, 0.08, 1.19))
    scene, view, cal, _ = started_scene(world)
    scene.cal = dataclasses.replace(cal, locator=SettledPairLocator(world.external))
    fakes.scripted_goto(world, (-0.20, 0.28, 1.22), tol_m=0.0005)
    top, bottom, largest = _arrival_descent(world, scene)
    assert largest < 0.02 and float(top.position[2] - bottom.position[2]) >= 0.029
    first = scene.projector
    out = scene.reanchor_from(top, bottom)
    truth = world.external.project(HandPose.from_view(bottom).grip)[0][0]
    assert out["ok"] and out["status"] == "reanchored" and out["located"] == "after", out
    assert np.linalg.norm(np.asarray(out["uv"]) - truth) <= 1e-6
    assert scene.projector is not first and np.allclose(scene.projector.anchor_grip, HandPose.from_view(bottom).grip)
    assert out["step"] == [s for s, v in scene._recent if v is bottom][0]
    assert out["anchor_dist_m"] > scene_mod.REANCHOR_RADIUS_M and scene.last_reanchor == out
    assert any(r.get("source") == "reanchor_from" for r in scene._locations)
    again = scene.reanchor_from(bottom, bottom)
    assert not again["ok"] and again["status"] == "no_hand_motion"
    near = scene.reanchor_from(top, bottom)                  # the anchor is now at the reading: kept, not moved
    assert near["ok"] and near["status"] == "located" and len([r for r in scene._locations
                                                               if r.get("source") == "reanchor_from"]) == 1
    scene.cal = dataclasses.replace(cal, locator=None)
    assert scene.reanchor_from(top, bottom)["status"] == "no_locator"


def test_cache_scan_pairs_non_consecutive_settled_views():
    """Rulings item 4 (decision §6 4c): consecutive cached views of a descent move too little for the locator (300/300
    refusals on real hovers), so the cache scan of an external region mask pairs NON-consecutive views, a settled one
    first: the settled top of the 3 cm descent with a later view re-anchors the projector."""
    sc = fakes.planar_region_scene(centre_xy=(-0.20, 0.30))
    world = make_world(sc, (-0.20, 0.08, 1.19))
    truth = sc.info["truth"]
    scene, view, cal, _ = started_scene(world)
    scene.cal = dataclasses.replace(cal, locator=SettledPairLocator(world.external))
    poly = np.asarray(truth["polygon_xy"], float)
    scene.parts["B"] = scene_mod.PartRecord("B", "region", "fixed", frozenset({"ref"}), region_xy=poly,
                                            top_z=Meas(truth["top_z"], 0.002, "kappa"),
                                            xyz=np.r_[poly.mean(axis=0), truth["top_z"]], sigma_h_m=0.002,
                                            status="registered", registration_step=0)
    fakes.scripted_goto(world, (-0.20, 0.28, 1.22), tol_m=0.0005)
    top, bottom, largest = _arrival_descent(world, scene)
    assert largest < 0.02
    first = scene.projector
    scene.region_mask("B", "external", bottom)
    log = scene.mask_log["projector"]["reanchor"]
    assert log["status"] == "reanchored" and log["reading_source"] == "cache_scan" and scene.projector is not first, log
    top_step = [s for s, v in scene._recent if v is top][0]
    # the reading pairs the settled top with a descent frame >= 2 cm below it (non-consecutive: every 1-step pair
    # moved < 2 cm) and is located on that lower frame
    assert log["reading_step"] > top_step + 1 and log["tries"] <= scene_mod.REANCHOR_MAX_TRIES


# ================================================================ BS review of the r1 fixes (BSR-1, -4, -7, -8, -9)
def _thin_dense_hover(*, extra=(), extra_grid=()):
    """A fresh thin-part hover (D2 probe views cached) with a probe-tracker-like dense lattice around the pointed pixel
    (``dense_survey``); ``extra`` = mis-depthed dense samples, ``extra_grid`` = grid samples absorbed BEFORE the
    lattice (an earlier hover's grid features)."""
    sc = fakes.free_object_with_thin_part_scene()
    world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
    uv0 = world.project("wrist", LOOP_TOP)[0]
    scene, view, cal, info = started_scene(world, survey_seeds=[("C", tuple(uv0))])
    after = probe(world, scene)
    uv_seed = np.asarray(world.project("wrist", LOOP_TOP)[0], float)
    rng = np.random.default_rng(9)
    if extra_grid:
        scene.absorb_survey(dense_survey(world, after, (0.0, 0.0), window_px=0, extra=extra_grid, extra_keys=None,
                                         rng=rng), after)
    scene.absorb_survey(dense_survey(world, after, uv_seed, extra=extra, rng=rng), after)
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    return SimpleNamespace(world=world, scene=scene, view=after, uv=uv_seed, seed=seed_at(world, after, LOOP_TOP),
                           sc=sc)


def test_second_moment_axis_is_undefined_for_an_isotropic_cell_set():
    """BS review BSR-1: the S-4' arbiter's own aspect sqrt(lambda_major / lambda_minor). A 2 x 2 block of cell centres
    is isotropic (ratio 1, its eigenvector arbitrary), a 2 x 3 block is elongated along its long side (sqrt(8/3) >=
    D10_MIN_ASPECT) and collinear cells are unbounded (None) along their line."""
    c = 0.005
    _, ratio = scene_mod._second_moment_axis(np.array([[x * c, y * c] for x in range(2) for y in range(2)]))
    assert ratio is not None and abs(ratio - 1.0) <= 1e-9
    axis, ratio = scene_mod._second_moment_axis(np.array([[x * c, y * c] for x in range(3) for y in range(2)]))
    assert angle_deg(axis, (1.0, 0.0)) <= 1e-6 and abs(ratio - math.sqrt(8.0 / 3.0)) <= 1e-9
    assert ratio >= scene_mod.D10_MIN_ASPECT
    axis, ratio = scene_mod._second_moment_axis(np.array([[0.001 * k, 0.002 * k] for k in range(4)]))
    assert ratio is None and angle_deg(axis, (1.0, 2.0)) <= 1e-6


def test_d10_oe_outline_gate_refuses_an_isotropic_region_whatever_the_tensor():
    """Decision 2 §2.3 S-11 (BSR-1 superseded): a 6 x 6 mm square region on the thin part's top with a stub tensor of
    coherence 1.0 along either image axis. Its outline holds two equal orientation peaks (second peak 1.0 >
    D10_OUTLINE_SECOND_MAX): the outline gate fails, the axis is refused as NOT_ELONGATED with reason "outline_gate"
    (P-C's RC-7 may run; no BR-1 trigger), whatever the tensor; the cells stay logged."""
    def run(tensor_img_deg):
        sc = fakes.free_object_with_thin_part_scene()
        world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
        scene, _, _, _ = started_scene(world, survey_seeds=[("C", tuple(world.project("wrist", LOOP_TOP)[0]))])
        view = probe(world, scene)
        pose = HandPose.from_view(view)
        c0 = np.asarray(LOOP_TOP[:2], float)
        x0, x1, y0, y1 = c0[0] - 0.003, c0[0] + 0.003, c0[1] - 0.003, c0[1] + 0.003
        pts = np.array([[x0, y0, LOOP_TOP[2]], [x1, y0, LOOP_TOP[2]], [x1, y1, LOOP_TOP[2]], [x0, y1, LOOP_TOP[2]]])
        uv = scene_mod._project(scene.camera, pose, pts)[0]
        region = np.zeros((512, 512), np.uint8)
        cv2.fillConvexPoly(region, np.rint(uv).astype(np.int32), 1)
        a = math.radians(tensor_img_deg)
        stub = SimpleNamespace(**{k: getattr(VISION, k) for k in _VISION_KEYS})
        stub.structure_axis = lambda gray, uv_, half_px=12: (np.array([math.cos(a), math.sin(a)]), 1.0, 1.0)
        scene._vision = stub
        scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], region > 0)
        scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
        return scene, scene.register("C", view, seed=seed_at(world, view, LOOP_TOP), empty_hand=True)

    for tensor_img_deg in (0.0, 90.0):
        scene, rec = run(tensor_img_deg)
        assert isinstance(rec, scene_mod.PartRecord), rec
        check, oe = rec.axis_check, scene.last_registration["C"]["d10_oe"]
        assert not rec.is_elongated() and not check["accepted"] and check["method"] == "outline_edges"
        assert check["source"] == scene_mod.NOT_ELONGATED and check["reason"] == "outline_gate", check
        assert not oe["outline_ok"] and oe["outline"]["second"] > scene_mod.D10_OUTLINE_SECOND_MAX, oe["outline"]
        assert oe["edges"] == {} and check["coherence"] == 1.0 and check["cells"] >= 1      # logged, not decided


def test_region_too_small_point_seed_registers_through_the_band_hull():
    """BS review BSR-8 (decision §2.2 S-4' fallback, remedy 1 region_too_small): a point seed whose segment at the
    seed is below REGION_MIN_PX but whose top band holds >= TOP_BAND_SUPPORT_N members registers on the members'
    convex hull dilated by the probe template radius (logged as band_hull); with fewer members the same segment is
    refused region_too_small with no hull."""
    from gpt_field.agent_speedup_v2_types import poly_contains
    t = _thin_dense_hover()
    scene = t.scene
    u, v = (int(x) for x in np.rint(t.uv))
    tiny = np.zeros((512, 512), bool)
    tiny[v - 1:v + 2, u - 1:u + 2] = True                                          # 9 px < REGION_MIN_PX
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (None, tiny.copy())
    band = scene._seed_band(t.seed)
    assert band["rule"] == "band" and band["n"] >= scene_mod.TOP_BAND_SUPPORT_N, band
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", rec
    hull = scene.last_registration["C"]["band_hull"]
    assert hull is not None and hull["members"] == band["n"] and hull["pixels"] >= scene_mod.REGION_MIN_PX, hull
    assert hull["radius_px"] > 0 and poly_contains(rec.region_xy, [LOOP_TOP[:2]])[0]
    assert abs(rec.top_z.value - LOOP_TOP[2]) <= 0.001
    original = scene._seed_band
    scene._seed_band = lambda seed: dict(original(seed), n=scene_mod.TOP_BAND_SUPPORT_N - 1)
    refused = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(refused, RegistrationFailure) and refused.reason == "geometry_unmeasured", refused
    assert refused.evidence["why"] == "region_too_small" and refused.evidence["band_hull"] is None


def test_dense_withdrawal_rebuilds_a_mixed_cell_from_its_kept_grid_sample():
    """BS review BSR-8 (decision §2.2 S-5): a cell holding both a phantom dense sample (the part's height, 20 mm
    beside it) and an earlier grid sample of a real lower surface. The newer, disagreeing phantom replaced the grid
    value; DSW vacates the cell and re-stamps it from its kept samples in order, so the cell is measured at the grid
    height again and counted as rebuilt (not vacated)."""
    side = 0.013 / 2.0
    xy = [(LOOP_TOP[0] + side + 0.020, LOOP_TOP[1] + along) for along in (-0.010, 0.0, 0.010)]
    z_low = LOOP_TOP[2] - 0.080
    phantom = [(x, y, LOOP_TOP[2]) for x, y in xy]
    t = _thin_dense_hover(extra=phantom, extra_grid=[(x, y, z_low) for x, y in xy])
    hf = t.scene.hf
    cells = {hf.index(p[:2]) for p in phantom}
    assert len(cells) == 3 and all(hf.measured[c] and abs(hf.z[c] - LOOP_TOP[2]) <= 0.002 for c in cells)
    rec = t.scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    log = t.scene.last_dsw["C"]
    assert log["skipped"] is None and log["reference"] == "support" and log["rebuilt"] >= len(cells), log
    for c in cells:
        assert hf.measured[c] and abs(hf.z[c] - z_low) <= 0.001, (c, hf.source[c], hf.z[c])


def test_dense_withdrawal_without_a_support_uses_the_s7_floor():
    """BS review BSR-7: when S-7 leaves the support unmeasured, DSW takes the S-7 floor (lowest top-band member -
    max(D8_FLOOR_M, D8_SIGMA_K sigma), the highest admissible support) as its height reference instead of skipping:
    the phantoms at the part's height beside it are withdrawn, a dense sample below the floor stays (the decision's
    rule with any admissible support would withdraw it too, so this is a subset), and the reference is logged."""
    side = 0.013 / 2.0
    phantom = [(LOOP_TOP[0] + side + 0.020, LOOP_TOP[1] + along, LOOP_TOP[2]) for along in (-0.010, 0.0, 0.010)]
    low = (LOOP_TOP[0] - side - 0.020, LOOP_TOP[1], LOOP_TOP[2] - 0.050)               # above the table, below the floor
    t = _thin_dense_hover(extra=phantom + [low])
    scene = t.scene
    scene._support = lambda mask, seg, part_band, kappa, pose, floor=None: (
        None, None, None, {"floor_z": None if floor is None else float(floor[0]), "rejected": [],
                           "fallback": "unmeasured"})
    band = scene._seed_band(t.seed)
    expected = band["lowest_z"] - max(scene_mod.D8_FLOOR_M, scene_mod.D8_SIGMA_K * band["lowest_sigma"])
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.support_z is None, rec
    log = scene.last_dsw["C"]
    assert log["skipped"] is None and log["reference"] == "s7_floor", log
    assert abs(log["reference_z"] - expected) <= 1e-9 and low[2] < log["reference_z"] - scene_mod.D8_FLOOR_M
    s = scene._samples
    index = lambda p: int(np.argmin(np.linalg.norm(s["xyz"] - np.asarray(p)[None, :], axis=1)))
    assert all(s["withdrawn"][index(p)] for p in phantom) and not s["withdrawn"][index(low)]
    th_ = [x for x in scene.registration_thresholds("C") if x.name == "D8_dense_withdrawal_m"]
    assert th_ and "reference=s7_floor" in th_[-1].derivation, th_


def test_support_floor_applies_to_the_measured_support_height():
    """BS review BSR-9 a (decision §2.2 S-7): the floor test applies to the support's measured height too. A ring
    majority band whose level passes the floor but whose region median (the returned support) sits above it is
    rejected with its measured height logged, and the plane below is taken."""
    scene = scene_mod.SceneModel(RobotModel(), vision=VISION)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    part = (xx - 100) ** 2 + (yy - 100) ** 2 <= 20 ** 2
    ring = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 75 ** 2) & ~part
    labels = np.where(part, 2, np.where(ring, 1, 0)).astype(np.int32)
    bands = [{"z": 0.918, "sigma": 0.001, "n": 40, "measured": 0, "plane": True, "members": np.zeros(0, int)},
             {"z": 0.940, "sigma": 0.001, "n": 9, "measured": 9, "plane": False, "members": np.zeros(0, int)},
             {"z": 0.955, "sigma": 0.001, "n": 30, "measured": 30, "plane": False, "members": np.zeros(0, int)}]
    seg = {"labels": labels, "bands": bands}
    pose = HandPose.from_state8((0.0, 0.0, 1.2, math.pi, 0.0, 0.0, 0.04, -0.04))
    scene._region_height = lambda region, pose_, band: (Meas(0.951, 0.001, "kappa")
                                                        if abs(band["z"] - 0.940) < 1e-9 else None)
    b, meas, _, check = scene._support(part, seg, 2, 3000.0, pose, floor=(0.954, 0.001))
    assert b == 0 and abs(meas.value - 0.918) < 1e-9, check
    rejected = check["rejected"]
    assert len(rejected) == 1 and abs(rejected[0]["z"] - 0.940) < 1e-9 and abs(rejected[0]["measured_z"] - 0.951) < 1e-9
    assert check["fallback"] == "plane_below" and abs(check["support_z"] - 0.918) < 1e-9
    b_old, meas_old, _, check_old = scene._support(part, seg, 2, 3000.0, pose)          # no top band: unchanged
    assert b_old == 1 and abs(meas_old.value - 0.951) < 1e-9 and not check_old["rejected"]


def test_cache_scan_checks_the_located_frames_grip():
    """BS review BSR-9 c (rulings item 4): a cache-scan pair is a candidate when the frame the locator reads (for a
    descent the later, lower one, ``_locate_order``) is nearer the region than the anchor, not the older frame."""
    sc = fakes.planar_region_scene(centre_xy=(-0.20, 0.30))
    world = make_world(sc, (-0.20, 0.08, 1.19))
    scene, view, cal, _ = started_scene(world)
    high, low = np.array([-0.20, 0.20, 1.22]), np.array([-0.14, 0.20, 1.19])         # 6 cm apart, 3 cm down
    scene._recent.clear()
    scene._locations = []
    for target in (high, low):
        fakes.scripted_goto(world, target, tol_m=0.0005)
        scene.on_view(view_of(world))
    older, newer = scene._recent[0][1], scene._recent[1][1]
    assert scene._locate_order(older, newer)[0] is newer                              # a descent: the lower frame
    near_newer = list(scene._reanchor_pairs(HandPose.from_view(newer).grip, 0.04))
    assert len(near_newer) == 1 and near_newer[0][0][1] is older and near_newer[0][1][1] is newer
    assert list(scene._reanchor_pairs(HandPose.from_view(older).grip, 0.04)) == []


def test_flat_side_cut_plane_region_keeps_its_top_bsr4():
    """Decision 2 §2.3 S-17 (BSR-4 signed off): a flat planar region cut by the image border is registered partial
    with its top measured within 3 mm; its part band is a plane band, so the cut cannot hide a higher top: no lower
    bound (top_bound None, no P-F raise to top_upper_z); the partial view still asks for a survey hover. The measured
    band case (a lower bound) is in ``test_provisional_and_border_cut_sigmas_cover_the_truth``."""
    sc = fakes.planar_region_scene()
    truth = sc.info["truth"]
    target = (-0.20, 0.16, truth["top_z"])
    world = make_world(sc, (-0.20, 0.26, 1.19))
    scene, view, cal, _ = started_scene(world, survey_seeds=[("B", tuple(world.project("wrist", target)[0]))])
    after = probe(world, scene)
    scene.parts["B"] = scene_mod.PartRecord("B", "region", "fixed", frozenset({"ref"}))
    rec = scene.register("B", after, seed=seed_at(world, after, target, source="coarse", quantization=16.0),
                         empty_hand=True, allow_partial=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    event = scene.last_registration["B"]
    assert rec.partial and event["cut_kind"] == "side" and event["cut_fraction"] >= scene_mod.PARTIAL_CUT_FRACTION
    assert abs(rec.top_z.value - truth["top_z"]) <= 0.003 and event["part_band_plane"] is True
    assert rec.top_bound is None and rec.top_upper_z is None and not rec.top_is_lower_bound()
    assert scene.needs_survey_hover("B")                                    # partial: a whole view is still wanted


def test_end_cut_by_the_fingers_is_not_partial_and_its_top_is_no_lower_bound():
    """BS review BSR-8 ([C-38], decision §2.2 S-9): an elongated region whose short end meets the robot's own pixels
    (a grip part running under the fingers) is an END cut: registered whole, not partial, top_bound None and no survey
    hover. The same seed with a region whose long side meets the robot is a SIDE cut: refused without allow_partial,
    and a lower bound with it."""
    t = _thin_dense_hover()
    scene = t.scene
    robot = scene._images(t.view)["robot"]
    u, v = (int(x) for x in np.rint(t.uv))

    def region(rows, cols):
        box = np.zeros(robot.shape, bool)
        box[rows[0]:rows[1], cols[0]:cols[1]] = True
        box &= ~robot                                                    # the segmentation never takes robot pixels
        first = np.array([np.flatnonzero(robot[:, c]).min() for c in range(cols[0], cols[1])])
        assert (first < rows[1]).all() and (first > v).all()             # every column ends on the fingers
        return box
    bottom = int(np.flatnonzero(robot[:, u]).min())
    end = region((bottom - 58, bottom + 8), (u - 18, u + 18))           # 36 px wide, ~58 px toward the fingers
    side = region((v - 12, bottom + 8), (u - 44, u + 44))               # 88 px wide, ~52 px tall
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], end.copy())
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    event = scene.last_registration["C"]
    assert event["cut_kind"] == "end" and event["cut_all"] >= scene_mod.PARTIAL_CUT_FRACTION, event["cut_all"]
    assert event["cut_fraction"] == 0.0 and not rec.partial and rec.top_bound is None and rec.top_upper_z is None
    assert not scene.needs_survey_hover("C") and abs(rec.top_z.value - LOOP_TOP[2]) <= 0.001
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], side.copy())
    refused = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(refused, RegistrationFailure) and refused.evidence["why"] == "cut_by_border", refused
    assert refused.evidence["cut_kind"] == "side"
    scene.parts["E"] = scene_mod.PartRecord("E", "thin part", "free", frozenset({"grip"}))
    cut = scene.register("E", t.view, seed=t.seed, empty_hand=True, allow_partial=True)
    # decision 2 S-17 (BSR-4): the lower bound only for a measured (not a plane) part band; here the dense start
    # survey fitted a plane at the bar's top (>= PLANE_MIN_CELLS cells), so its band is a plane band
    plane = scene.last_registration["E"]["part_band_plane"]
    assert isinstance(cut, scene_mod.PartRecord) and cut.partial, cut
    assert cut.top_bound == (None if plane else "lower"), (plane, cut.top_bound)
    assert scene.last_registration["E"]["bands"][scene.last_registration["E"]["part_band"]]["plane"] == plane


# ============================================================================== design review 2 (decision 2 §2.3)
def _rotated(axis, deg):
    a = math.radians(deg)
    c, s_ = math.cos(a), math.sin(a)
    v = np.asarray(axis, float)
    return scene_mod._canonical_axis([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])


def _kernel_stub(outline=None, edge=None):
    """The vision with the G-7 kernels replaced: ``outline`` / ``edge`` = fixed kernel answers (None keeps the
    kernel)."""
    stub = SimpleNamespace(**{k: getattr(VISION, k) for k in _VISION_KEYS})
    if outline is not None:
        stub.outline_orientation = lambda mask, pose, z, f: dict(outline)
    if edge is not None:
        stub.edge_orientation = lambda grey, roi, exclude, pose, z, f: dict(edge)
    return stub


def test_s12_view_memory_agrees_disagrees_keeps_and_resets():
    """Decision 2 §6.1 item 2 (S-11 / S-12) on the thin-bar fake at a dense hover. Two agreeing views are kept (both
    remembered). A view whose outline carries a slanted lower edge with matching edges (the BR-1 side-wall case: the
    stub kernels put outline and edges 9 deg off the remembered axis) passes D10-OE on its own but is refused by S-12,
    AXIS_INCONSISTENT "views_disagree", and the memory is conflicted: a later clean view stays refused until
    ``part_moved``. After the reset a view is accepted alone; a refusal (outline gate) after an acceptance carries
    the remembered axis ("kept_earlier", its own verdict in ``refused_as``); after another reset the same refusal
    stands."""
    t = _thin_dense_hover()
    scene, axis_truth = t.scene, t.sc.info["truth"]["loop_axis_xy"]
    real = scene.vision

    def register(stub=None):
        scene._vision = real if stub is None else stub
        rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
        assert isinstance(rec, scene_mod.PartRecord), rec
        return rec, rec.axis_check
    first, check = register()
    assert check["accepted"] and check["source"] == "outline_edges" and len(check["views"]["axes"]) == 1
    second, check = register()                                          # two agreeing views: both remembered
    assert check["accepted"] and check["source"] == "outline_edges" and len(check["views"]["axes"]) == 2
    assert math.degrees(check["views"]["disagree_rad"]) <= 1.0 and not check["views"]["conflicted"]
    assert angle_deg(second.elongation[0], axis_truth) <= 5.0
    slanted = _rotated(first.elongation[0], 9.0)
    side_wall = _kernel_stub(outline={"axis": slanted, "share": 0.30, "second": 0.45, "two_sided": True},
                             edge={"axis": _rotated(slanted, 1.0), "share": 0.35, "second": 0.3})
    rec, check = register(side_wall)
    oe = scene.last_registration["C"]["d10_oe"]
    assert oe["accepted"] and oe["reason"] is None                       # the view alone passes D10-OE
    assert not rec.is_elongated() and not check["accepted"] and check["source"] == scene_mod.AXIS_INCONSISTENT
    assert check["reason"] == "views_disagree" and check["views"]["conflicted"] is True
    assert abs(math.degrees(check["views"]["disagree_rad"]) - 9.0) <= 0.5 and len(check["views"]["axes"]) == 2
    assert scene.view_memory("C")["conflicted"] is True
    rec, check = register()                                             # a clean view, but the memory is conflicted
    assert not rec.is_elongated() and check["reason"] == "views_disagree"
    assert scene.last_registration["C"]["d10_oe"]["accepted"]
    gate_off = _kernel_stub(outline={"axis": first.elongation[0], "share": 0.12, "second": 0.9, "two_sided": True})
    rec, check = register(gate_off)                                     # an own refusal stays the views' conflict
    assert not rec.is_elongated() and check["source"] == scene_mod.AXIS_INCONSISTENT
    assert check["reason"] == "views_disagree" and check["refused_as"] == {"source": scene_mod.NOT_ELONGATED,
                                                                           "reason": "outline_gate"}
    scene.part_moved("C")
    assert scene.view_memory("C") is None
    rec, check = register()
    assert rec.is_elongated() and check["source"] == "outline_edges" and len(check["views"]["axes"]) == 1
    remembered = np.asarray(rec.elongation[0], float)
    gate = _kernel_stub(outline={"axis": _rotated(remembered, 40.0), "share": 0.12, "second": 0.9, "two_sided": True})
    kept, check = register(gate)
    assert kept.is_elongated() and check["accepted"] and check["source"] == check["reason"] == "kept_earlier"
    assert check["refused_as"] == {"source": scene_mod.NOT_ELONGATED, "reason": "outline_gate"}
    assert angle_deg(kept.elongation[0], remembered) <= 1e-9 and kept.elongation[1] == pytest.approx(rec.elongation[1])
    assert angle_deg(check["image_axis"], _rotated(remembered, 40.0)) <= 1e-6       # this view's own evidence logged
    assert len(check["views"]["axes"]) == 1 and abs(math.degrees(check["views"]["disagree_rad"]) - 40.0) <= 0.5
    scene.part_moved("C")
    alone, check = register(gate)
    assert not alone.is_elongated() and check["source"] == scene_mod.NOT_ELONGATED and check["reason"] == "outline_gate"
    assert check["views"] is None and check["refused_as"] is None
    register()
    assert scene.view_memory("C") is not None
    scene.drop("C")                                                     # a repointed part starts a new memory
    assert scene.view_memory("C") is None


def _yawed_loop_hover(deg, *, only_top):
    """``_thin_dense_hover`` on a fake whose thin loop (or only its top segment) is yawed by ``deg`` about the loop's
    centre: the rendered silhouette, its edges, the depth and the dense lattice all carry the yaw. A pixel-level
    stand-in for decision 2 §1.2's BR-1 view of the loop's slanted side wall: outline and edges share one slanted
    silhouette, so their errors are correlated and D10-OE alone cannot see them."""
    base = fakes.free_object_with_thin_part_scene()
    a = math.radians(deg)
    bodies = []
    for body in base.bodies:
        if body.name == base.info["thin_part"]:
            cx, cy = body.centre[0], body.centre[1]
            top_z0 = max(c[2] for _, c, _, _ in body.parts)
            parts = []
            for kind, c, size, yaw in body.parts:
                if only_top and c[2] != top_z0:
                    parts.append((kind, c, size, yaw))
                    continue
                dx, dy = c[0] - cx, c[1] - cy
                parts.append((kind, (cx + math.cos(a) * dx - math.sin(a) * dy, cy + math.sin(a) * dx + math.cos(a) * dy,
                                     c[2]), size, yaw + a))
            body = dataclasses.replace(body, parts=tuple(parts))
        bodies.append(body)
    sc = fakes.FakeScene(bodies, support_z=base.support_z, texture_seed=0, support_texture_std=8.0, goal=base.goal,
                         info=base.info)
    world = make_world(sc, sc.info["hover_eef"][sc.info["thin_part"]])
    uv0 = world.project("wrist", LOOP_TOP)[0]
    scene, _, _, _ = started_scene(world, survey_seeds=[("C", tuple(uv0))])
    after = probe(world, scene)
    uv = np.asarray(world.project("wrist", LOOP_TOP)[0], float)
    scene.absorb_survey(dense_survey(world, after, uv, rng=np.random.default_rng(9)), after)
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    return SimpleNamespace(world=world, scene=scene, view=after, seed=seed_at(world, after, LOOP_TOP))


def test_s12_a_pixel_level_slanted_view_passes_d10_oe_and_is_refused_by_the_view_memory():
    """Decision 2 §6.1 item 2 (S-11 / S-12) at the pixel level (BS fix review t1-test-strength; the stub-kernel case
    is ``test_s12_view_memory_agrees_disagrees_keeps_and_resets``). An earlier view of the thin loop is accepted by
    D10-OE and remembered. A later view whose rendered loop is yawed 8.25 deg (its outline and edges carry the same
    slant, as the BR-1 side-wall silhouette does) passes D10-OE on its own pixels (outline gate, both edge ROIs gated,
    outline and edge axes agreeing), yet lies more than AXIS_AGREE_MAX_RAD from the remembered axis: S-12 refuses it,
    AXIS_INCONSISTENT "views_disagree", and conflicts the memory."""
    earlier = _yawed_loop_hover(-2.0, only_top=True)
    first = earlier.scene.register("C", earlier.view, seed=earlier.seed, empty_hand=True)
    assert isinstance(first, scene_mod.PartRecord) and first.axis_check["accepted"], first
    assert first.axis_check["source"] == "outline_edges" and first.is_elongated()
    later = _yawed_loop_hover(8.25, only_top=False)
    later.scene._views["C"] = copy.deepcopy(earlier.scene._views["C"])        # the same static part's memory
    rec = later.scene.register("C", later.view, seed=later.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    oe = later.scene.last_registration["C"]["d10_oe"]
    assert oe["accepted"] and oe["outline_ok"] and oe["reason"] is None, oe          # the view alone passes D10-OE
    assert all(e["gated"] for e in oe["edges"].values()) and set(oe["edges"]) == {"region", "hull"}
    assert oe["agree_rad"] <= scene_mod.AXIS_AGREE_MAX_RAD
    gap = math.radians(angle_deg(oe["axis"], first.elongation[0]))
    assert scene_mod.AXIS_AGREE_MAX_RAD < gap <= math.radians(10.0), math.degrees(gap)
    check = rec.axis_check
    assert not rec.is_elongated() and not check["accepted"] and check["source"] == scene_mod.AXIS_INCONSISTENT
    assert check["reason"] == "views_disagree" and check["views"]["conflicted"] is True
    assert later.scene.view_memory("C")["conflicted"] is True


def _collapsed_region(scene, t):
    """A disk at the pointed pixel holding a third of the top band's hull (dilated COLLAPSE_HULL_DILATE_PX)."""
    band = scene._seed_band(t.seed)
    hull = scene._members_hull(band, HandPose.from_view(t.view), scene_mod.COLLAPSE_HULL_DILATE_PX)
    radius = math.sqrt(float(hull.sum()) / 3.0 / math.pi)
    region = np.zeros(hull.shape, np.uint8)
    cv2.circle(region, (int(round(t.uv[0])), int(round(t.uv[1]))), int(round(radius)), 1, -1)
    return region > 0, int(hull.sum())


def test_s13_collapse_guard_refuses_the_axis_and_withholds_the_width():
    """Decision 2 §6.1 item 2 (S-13): a region holding a third of its top band's hull is collapsed: axis_check
    region_collapsed, the axis refused (the planner's P-H / BR-1 trigger), width_by_yaw == () (K-5, width unmeasured),
    top and xyz kept. With an earlier non-collapsed registration in the view memory, its width set is kept (and its
    axis carried, "kept_earlier")."""
    t = _thin_dense_hover()
    scene = t.scene
    scene.disturb("C", 0, "test_reset")     # gap round X-1: no MC-R2a whole record from an earlier test's registration
    region, hull_px = _collapsed_region(scene, t)
    assert hull_px > scene_mod.REGION_COLLAPSE_RATIO * region.sum() and region.sum() >= scene_mod.REGION_MIN_PX
    choose = scene._choose_component
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], region.copy())
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered" and not rec.provisional, rec
    check, event = rec.axis_check, scene.last_registration["C"]
    assert check["region_collapsed"] is True and not check["accepted"] and not rec.is_elongated(), check
    assert rec.width_by_yaw == () and rec.width_at(0.0) is None and event["width_source"] == "unmeasured"
    assert event["collapse"]["collapsed"] and event["collapse"]["ratio"] > scene_mod.REGION_COLLAPSE_RATIO
    assert abs(rec.top_z.value - LOOP_TOP[2]) <= 0.003 and rec.xyz is not None
    assert scene.view_memory("C")["widths_mm"] is None
    names = {th_.name for th_ in scene.registration_thresholds("C")}
    assert "region_collapse_ratio" in names
    from gpt_field import agent_speedup_v2_planner as planner_mod
    assert planner_mod.axis_refusal(rec) == "region_collapsed"
    # an earlier whole registration: its widths and axis are kept through the collapsed view
    scene.part_moved("C")
    scene._choose_component = choose
    whole = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert whole.is_elongated() and whole.width_by_yaw and not whole.axis_check["region_collapsed"]
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], region.copy())
    # gap round MC-R2a: in the same epoch the collapsed view keeps the whole record itself (kept_whole)
    same = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert same is whole and scene.last_registration["C"]["width_source"] == "kept_whole"
    assert scene.last_registration["C"]["flag"] == "kept_whole"
    assert scene.last_registration["C"]["collapse"]["stage"] in ("band", "edge")
    # after a disturbance (X-1) the S-13 view-memory rule decides as before
    scene.disturb("C", 0, "test_contact")
    kept = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert kept.axis_check["region_collapsed"] is True and kept.axis_check["source"] == "kept_earlier"
    assert kept.is_elongated() and angle_deg(kept.elongation[0], whole.elongation[0]) <= 1e-9
    assert len(kept.width_by_yaw) == len(whole.width_by_yaw)
    assert all(abs(a[1].value - b[1].value) <= 1e-12 for a, b in zip(kept.width_by_yaw, whole.width_by_yaw))
    assert scene.last_registration["C"]["width_source"] == "kept_earlier"


def test_s14_a_whole_registration_below_a_lower_bound_keeps_the_flag():
    """Decision 2 §6.1 item 2 (S-14): a whole registration reading 30 mm below an existing lower-bound top of the
    same part does not clear the flag: logged top_inconsistent, the lower bound (top_z) and top_upper_z kept. The
    inconsistent view's geometry (xyz and region back-projected at its low top, its widths, axis and support) is not
    the part's: the earlier record is kept whole (BS fix review, S14-kept-record-mixed-geometry) and the view's reading
    is only logged; the view memory takes nothing from it. One reading at the bound (or above it) clears it as
    before."""
    t = _thin_dense_hover()
    scene = t.scene
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and not rec.top_is_lower_bound()
    measured = rec.top_z.value
    for lift_m, kept in ((0.030, True), (-0.002, False)):
        bound = Meas(measured + lift_m, rec.top_z.sigma, rec.top_z.source)
        # the earlier (lower-bound) record: its grip point at its own top, 2 cm away, a region of its own
        shifted = np.asarray(rec.xyz, float) + np.array([0.02, 0.0, lift_m])
        earlier = dataclasses.replace(rec, top_z=bound, top_bound="lower", top_upper_z=measured + 0.20, xyz=shifted,
                                      region_xy=np.asarray(rec.region_xy, float) + np.array([0.02, 0.0]))
        scene.parts["C"] = earlier
        scene.part_moved("C")
        again = scene.register("C", t.view, seed=t.seed, empty_hand=True)
        event = scene.last_registration["C"]
        assert isinstance(again, scene_mod.PartRecord) and not again.partial, again
        assert event["top_check"]["kept"] is kept and abs(event["top_check"]["below_m"] - lift_m) <= 0.002
        if kept:
            assert again is earlier and again.top_is_lower_bound() and event["flag"] == "top_inconsistent"
            assert again.top_z.value == pytest.approx(bound.value) and again.top_upper_z == pytest.approx(measured + 0.2)
            assert np.allclose(again.xyz, shifted) and abs(float(again.xyz[2]) - again.top_z.value) <= 1e-9
            assert np.allclose(again.region_xy, earlier.region_xy) and event["kept_whole"] is True
            check = event["top_check"]
            assert check["kept_record"] == "earlier" and abs(check["measured_xyz"][2] - measured) <= 0.002
            assert abs(check["measured_top_z"] - measured) <= 0.002
            assert scene.view_memory("C") is None                       # the inconsistent view's axis is not remembered
            assert scene.needs_survey_hover("C")
            assert any(th_.name == "S14_lower_bound_m" for th_ in scene.registration_thresholds("C"))
        else:
            assert not again.top_is_lower_bound() and again.top_upper_z is None
            assert abs(again.top_z.value - measured) <= 0.001


def test_s15_a_seed_outside_its_region_is_refused():
    """Decision 2 §6.1 item 2 (S-15): a point seed 25 px from its chosen region (beyond q_now + S15_SEED_SIGMA_K x
    TRACK_SIGMA_PX, lead ruling 3) is refused, part_not_found "seed_outside_region", instead of being snapped to the
    nearest region pixel; a region 4 px away is still snapped (within the quantization)."""
    t = _thin_dense_hover()
    scene = t.scene
    u, v = (int(round(x)) for x in t.uv)
    for gap_px, refused in ((25, True), (4, False)):
        region = np.zeros((512, 512), np.uint8)
        cv2.circle(region, (u + gap_px + 8, v), 8, 1, -1)
        scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], region > 0)
        rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
        if refused:
            assert isinstance(rec, RegistrationFailure) and rec.reason == "part_not_found", rec
            assert rec.evidence["why"] == "seed_outside_region" and rec.evidence["gap_px"] >= 24.0
            assert rec.evidence["gap_px"] > rec.evidence["limit_px"] == pytest.approx(
                8.0 + scene_mod.S15_SEED_SIGMA_K * scene_mod.TRACK_SIGMA_PX, rel=0.1)
        else:
            assert isinstance(rec, scene_mod.PartRecord), rec
            assert any(th_.name == "seed_region_px" for th_ in scene.registration_thresholds("C"))


def test_s17_point_seed_on_a_sample_less_region_registers():
    """Decision 2 §6.1 item 2 (S-17, the D hover KeyError 'n'): a point-seeded registration whose region has no band
    and no samples of its own takes the lifted seed as its band, with the keys of ``_region_band``, and registers;
    the event's band list carries it (n 0)."""
    t = _thin_dense_hover()
    scene = t.scene
    u, v = (int(round(x)) for x in t.uv)
    region = np.zeros((512, 512), np.uint8)
    cv2.circle(region, (u, v), 6, 1, -1)
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (None, region > 0)
    scene._region_band = lambda mask, pose: None
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord), rec
    bands = scene.last_registration["C"]["bands"]
    part = bands[scene.last_registration["C"]["part_band"]]
    assert part["n"] == 0 and part["measured"] == 1 and part["plane"] is False
    assert scene_mod._bands_summary([{"z": 1.0, "sigma": 0.001, "members": np.arange(3)}])[0]["n"] == 3


def _support_ring_fake():
    """The S-17 (BSR-6) support fake: a 40 x 120 px region (top 0.955) whose SUPPORT_RING_M ring holds the table plane
    (band 0, 0.900), a measured disk band (1, 0.918) over most of the ring, the part's own ridge level next to it
    (2, 0.950: above the S-7 floor, the ring-pixel majority) and the part band (3)."""
    t = _thin_dense_hover()
    scene, pose = t.scene, HandPose.from_view(t.view)
    h = w = 512
    mask = np.zeros((h, w), bool)
    mask[236:276, 196:316] = True
    top = 0.955
    kappa = scene.camera.f_px / max(float(pose.camera[2] - top), 1e-6)
    ring_px = max(int(math.ceil(kappa * scene_mod.SUPPORT_RING_M)), scene_mod.MARKER_RADIUS_PX + 1)
    outer = cv2.dilate(mask.astype(np.uint8), scene_mod._disk_kernel(ring_px)) > 0
    near = cv2.dilate(mask.astype(np.uint8), scene_mod._disk_kernel(int(0.6 * ring_px))) > 0
    labels = np.zeros((h, w), np.int32)                                # 0 = the table plane everywhere
    labels[outer & (np.arange(w)[None, :] < 300)] = 1                  # the disk: most of the ring
    labels[near] = 2                                                   # the ridge level next to the part
    labels[mask] = 3
    bands = [{"z": 0.900, "sigma": 0.001, "n": 500, "measured": 0, "plane": True, "members": np.zeros(0, int)},
             {"z": 0.918, "sigma": 0.001, "n": 40, "measured": 40, "plane": False, "members": np.zeros(0, int)},
             {"z": 0.950, "sigma": 0.001, "n": 60, "measured": 60, "plane": False, "members": np.zeros(0, int)},
             {"z": top, "sigma": 0.001, "n": 80, "measured": 80, "plane": False, "members": np.zeros(0, int)}]
    ring = outer & ~(cv2.dilate(mask.astype(np.uint8), scene_mod._disk_kernel(scene_mod.MARKER_RADIUS_PX)) > 0)
    assert np.count_nonzero(ring & (labels == 2)) > np.count_nonzero(ring & (labels == 1)) > \
        np.count_nonzero(ring & (labels == 0)) > 0

    def samples(specs, rng):
        """(xyz, sigma) of ring samples: specs = [(band label, z, count, sigma_m, height noise_m)]."""
        pts, sig = [], []
        for b, z, n, s, noise in specs:
            vs, us = np.nonzero(ring & (labels == b))
            pick = rng.choice(len(us), size=n, replace=False)
            uv = np.column_stack([us[pick], vs[pick]]).astype(float)
            zz = z + noise * rng.standard_normal(n)
            pts.append(np.vstack([scene_mod._backproject_z(scene.camera, pose, uv[i:i + 1], zz[i]) for i in range(n)]))
            sig += [s] * n
        return np.vstack(pts), np.asarray(sig, float)
    return SimpleNamespace(scene=scene, pose=pose, mask=mask, kappa=kappa, labels=labels, bands=bands,
                           samples=samples, floor=(0.953, 0.001))


def test_s17_support_fallback_takes_the_measured_disk_not_the_table_plane():
    """Decision 2 §6.1 item 2 (S-17, BSR-6): the ring around a part holds a band at its own ridge level (rejected by
    S-7: not below the lowest top-band member), a measured disk band and the table plane. The fallback takes the
    next admissible surface in the ring, the measured disk (more ring pixels than the table), measured from the
    ring's own samples (fallback "ring_below"); before, it took the highest plane below the floor (the table)."""
    f = _support_ring_fake()
    scene, labels = f.scene, f.labels
    xyz, sigma = f.samples([(1, 0.9185, 12, 0.001, 0.0005), (0, 0.900, 12, 0.001, 0.0005),
                            (2, 0.950, 12, 0.001, 0.0005)], np.random.default_rng(3))
    scene._live_samples = lambda: (xyz, sigma, ["kappa"] * len(xyz))
    seg = {"labels": labels, "bands": f.bands}
    band, meas, region, check = scene._support(f.mask, seg, 3, f.kappa, f.pose, floor=f.floor)
    assert band == 1 and abs(meas.value - 0.9185) <= 0.001 and check["fallback"] == "ring_below", check
    assert [round(r["z"], 3) for r in check["rejected"]] == [0.950] and region is not None
    # without the disk band in the ring, the table plane is the fallback (as before)
    labels[labels == 1] = 0
    band, meas, _, check = scene._support(f.mask, seg, 3, f.kappa, f.pose, floor=f.floor)
    assert band == 0 and abs(meas.value - 0.900) <= 0.001 and check["fallback"] == "ring_below", check


def test_s17_support_fallback_measures_the_disk_through_noisy_ring_samples():
    """BS fix review (BSR6-ring-own-sigma, the live rings hold samples with sigma 6-22 mm): the disk band's 12 clean
    ring samples (sigma 1 mm) share the ring with 0-20 imprecise ones (sigma 15 mm, 10 mm scatter about 0.93) that
    agree with the disk only within their own wide tolerance. The fallback's ring-own measurement is inverse-variance
    weighted (``_ring_height``): the disk is measured at its height with the clean samples' sigma and passes S-7, so
    the support is the disk, never the table plane (the plain median of the claimed sigmas gave 15 mm and dropped to
    the table with 14 noisy samples). The negative: with only imprecise samples on the disk band, its ring-own sigma
    stays wide (no clean sample to measure it)."""
    for n_noisy in (0, 6, 14, 20):
        f = _support_ring_fake()
        rng = np.random.default_rng(3)
        specs = [(1, 0.9185, 12, 0.001, 0.0005), (0, 0.900, 12, 0.001, 0.0005), (2, 0.950, 12, 0.001, 0.0005)]
        if n_noisy:
            specs.append((1, 0.93, n_noisy, 0.015, 0.010))
        xyz, sigma = f.samples(specs, rng)
        f.scene._live_samples = lambda xyz=xyz, sigma=sigma: (xyz, sigma, ["kappa"] * len(xyz))
        band, meas, _, check = f.scene._support(f.mask, {"labels": f.labels, "bands": f.bands}, 3, f.kappa, f.pose,
                                                floor=f.floor)
        assert band == 1 and check["fallback"] == "ring_below", (n_noisy, check)
        assert abs(meas.value - 0.9185) <= 0.0015 and meas.sigma <= 0.0015, (n_noisy, meas)
    f = _support_ring_fake()
    xyz, sigma = f.samples([(1, 0.9185, 12, 0.015, 0.0005)], np.random.default_rng(4))
    f.scene._live_samples = lambda: (xyz, sigma, ["kappa"] * len(xyz))
    disk = f.scene._ring_height(f.labels == 1, f.pose, f.bands[1])
    assert disk is not None and disk.sigma >= 0.015 and abs(disk.value - 0.9185) <= 0.002


def test_top_band_on_the_base_support_registered_the_support_not_the_part():
    """Decision 2 §2.3 S-15 diagnosis (the band lift of live5 episode 17), extended by lead ruling 7 (r2close): a fine
    seed whose 40 px window holds only the base support's samples (the dark thin part has none of its own) has its top
    band ON the support plane (D8): the window measured the support, so the registration is refused, part_not_found
    "band_on_support" (source "top_band"), instead of lifting the seed onto the support plane. Every point-seeded
    part is refused on a non-provisional call (a part of another role too: a part registered at its support's height
    is a mis-registration); on a provisional call only a grip part is (the diagnosis rule unchanged)."""
    t = _thin_dense_hover()
    scene, world, view = t.scene, t.world, t.view
    table = np.array([LOOP_TOP[0] + 0.07, LOOP_TOP[1] + 0.05, scene.planes[0]["z"].value])
    uv = np.asarray(world.project("wrist", table)[0], float)
    scene.absorb_survey(dense_survey(world, view, uv, rng=np.random.default_rng(5)), view)
    seed = seed_at(world, view, table)
    assert scene._seed_band(seed)["rule"] == "band"

    def refused(rec):
        return isinstance(rec, RegistrationFailure) and rec.evidence.get("why") == "band_on_support"
    for pid, roles in (("C", {"grip"}), ("B", {"ref"}), ("D", {"carried"})):
        scene.parts[pid] = scene_mod.PartRecord(pid, "part", "free", frozenset(roles))
        rec = scene.register(pid, view, seed=seed, empty_hand=True)
        assert refused(rec) and rec.reason == "part_not_found", (pid, rec)
        assert rec.evidence["source"] == "top_band" and abs(rec.evidence["dz_m"]) <= rec.evidence["tol_m"]
        assert rec.evidence["tol_m"] >= scene_mod.D8_FLOOR_M
        assert [th_.name for th_ in rec.thresholds] == ["top_band_on_support_m"]
    # provisional calls: the grip part is refused, a part of another role is not (by this rule)
    scene.parts["C"] = scene_mod.PartRecord("C", "part", "free", frozenset({"grip"}))
    assert refused(scene.register("C", view, seed=seed, empty_hand=True, provisional=True))
    scene.parts["B"] = scene_mod.PartRecord("B", "region", "fixed", frozenset({"ref"}))
    assert not refused(scene.register("B", view, seed=seed, empty_hand=True, provisional=True))


def _unbanded_support_seed(t, *, lift_m):
    """Lead ruling 7 (ii) fake: a fine seed on the base support with no top band (only the sparse grid there), its
    region a 9 px disk at the seed whose part band is a measured band ``lift_m`` above the base support plane."""
    scene, world, view = t.scene, t.world, t.view
    base = scene.planes[0]["z"]
    table = np.array([LOOP_TOP[0] + 0.07, LOOP_TOP[1] + 0.05, base.value])
    seed = seed_at(world, view, table)
    assert scene._seed_band(seed)["rule"] != "band"                                # no top band at the seed
    u, v = (int(round(x)) for x in seed.uv)
    region = np.zeros((512, 512), np.uint8)
    cv2.circle(region, (u, v), 9, 1, -1)
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (None, region > 0)
    scene._region_band = lambda mask, pose: {"z": float(base.value) + lift_m, "sigma": 0.001, "measured": 5,
                                             "members": np.zeros(0, int), "n": 5, "plane": False}
    return seed


def test_measured_top_on_the_base_support_is_refused_r2close_ruling_7():
    """Lead ruling 7 (r2close) on the measured top (the review-2 object-body record 155.7 mm low: a seed with one gated
    sample, no top band, its region the support): a point-seeded non-provisional registration whose measured top lies
    on the base support plane by the D8 same-surface test is refused, part_not_found "band_on_support" (source
    "measured_top"), for a non-grip part too; a provisional call is not refused by it; a part whose measured top is
    20 mm above the support registers."""
    for pid, roles in (("B", {"ref"}), ("C", {"grip"})):
        t = _thin_dense_hover()
        seed = _unbanded_support_seed(t, lift_m=0.0)
        t.scene.parts[pid] = scene_mod.PartRecord(pid, "part", "free", frozenset(roles))
        rec = t.scene.register(pid, t.view, seed=seed, empty_hand=True)
        assert isinstance(rec, RegistrationFailure) and rec.reason == "part_not_found", (pid, rec)
        assert rec.evidence["why"] == "band_on_support" and rec.evidence["source"] == "measured_top", rec.evidence
        assert abs(rec.evidence["dz_m"]) <= rec.evidence["tol_m"]
        assert rec.evidence["tol_m"] == pytest.approx(max(scene_mod.D8_FLOOR_M, scene_mod.D8_SIGMA_K * math.hypot(
            t.scene.planes[0]["z"].sigma, rec.evidence["sigma_top_m"])))
        assert "top_on_support_m" in {th_.name for th_ in rec.thresholds}
    t = _thin_dense_hover()
    seed = _unbanded_support_seed(t, lift_m=0.0)
    t.scene.parts["B"] = scene_mod.PartRecord("B", "part", "fixed", frozenset({"ref"}))
    rec = t.scene.register("B", t.view, seed=seed, empty_hand=True, provisional=True)
    assert not (isinstance(rec, RegistrationFailure) and rec.evidence.get("why") == "band_on_support"), rec
    t = _thin_dense_hover()
    seed = _unbanded_support_seed(t, lift_m=0.020)
    t.scene.parts["B"] = scene_mod.PartRecord("B", "part", "fixed", frozenset({"ref"}))
    rec = t.scene.register("B", t.view, seed=seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", rec
    assert abs(rec.top_z.value - (t.scene.planes[0]["z"].value + 0.020)) <= 0.002
    # the rule itself: no new number, the D8 tolerance with the top's own sigma (never the model term)
    base = t.scene.planes[0]["z"]
    info, threshold = t.scene._top_on_support(Meas(base.value + 0.002, 0.001, "kappa"), None)
    assert info is not None and threshold.name == "top_on_support_m" and info["sigma_top_m"] == 0.001
    assert t.scene._top_on_support(Meas(base.value + 0.020, 0.001, "kappa"), None) == (None, None)
    band = {"spread_m": 0.0005, "top": base.value + 0.002}
    info, _ = t.scene._top_on_support(Meas(base.value + 0.002, 0.006, "kappa"), band)
    assert info["sigma_top_m"] == 0.0005                                 # a banded seed: one member's height scale


def test_s15_radius_is_q_now_plus_two_track_sigma_r2close_ruling_3():
    """Lead ruling 3 (r2close): the S-15 refusal radius is q_now + S15_SEED_SIGMA_K x TRACK_SIGMA_PX (2 sigma). With
    the region's nearest pixel at a measured gap G from the seed, a seed whose q_now puts G at q + 1.9 sigma is
    snapped and registers (its ``seed_check`` logged), one at q + 2.1 sigma is refused, part_not_found
    "seed_outside_region", with q_now, the factor and gap - limit in the evidence."""
    t = _thin_dense_hover()
    scene = t.scene
    u, v = (int(round(x)) for x in t.uv)
    region = np.zeros((512, 512), np.uint8)
    cv2.circle(region, (u + 20, v), 8, 1, -1)
    uv = np.asarray(t.seed.uv, float)
    cleaned = scene_mod._clean_mask(region > 0, uv)                      # the region as registration cleans it
    gap = float(np.linalg.norm(scene_mod._nearest_mask_pixel(cleaned, uv) - uv))
    assert 10.0 <= gap <= 14.0
    sigma = scene_mod.TRACK_SIGMA_PX
    scene._choose_component = lambda seg, seed_uv, q, point_seed, edge: (seg["seed_band"], region > 0)
    for k, admitted in ((1.9, True), (2.1, False)):
        q = gap - k * sigma
        seed = dataclasses.replace(t.seed, quantization_px=q)
        scene.part_moved("C")
        rec = scene.register("C", t.view, seed=seed, empty_hand=True)
        if admitted:
            assert isinstance(rec, scene_mod.PartRecord), rec
            check = scene.last_registration["C"]["seed_check"]
            assert check["inside"] is False and check["gap_px"] == pytest.approx(gap)
            assert check["limit_px"] == pytest.approx(q + scene_mod.S15_SEED_SIGMA_K * sigma)
            assert check["excess_px"] == pytest.approx((k - scene_mod.S15_SEED_SIGMA_K) * sigma)
            assert any(th_.name == "seed_region_px" for th_ in scene.registration_thresholds("C"))
        else:
            assert isinstance(rec, RegistrationFailure) and rec.reason == "part_not_found", rec
            e = rec.evidence
            assert e["why"] == "seed_outside_region" and e["gap_px"] == pytest.approx(gap)
            assert e["limit_px"] == pytest.approx(q + 2.0 * sigma) and e["q_now_px"] == pytest.approx(q)
            assert e["sigma_k"] == scene_mod.S15_SEED_SIGMA_K and e["excess_px"] == pytest.approx(0.1 * sigma)
    # a seed inside its region (the scene's own region choice) logs that it is inside
    del scene._choose_component
    scene.part_moved("C")
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord)
    assert scene.last_registration["C"]["seed_check"]["inside"] is True


def _near_isotropic_region(*, stripes, period=10, contrast=60):
    """Lead ruling 4 (a) pixel fake: a 50 x 40 px rectangle region (aspect 1.25, the near-isotropic chain region of
    review 2's 89 deg case) on a uniform grey image whose region ROI carries stripes: ``stripes`` "along" (edges along
    the rectangle's long side only: a unimodal edge histogram) or ("along", "across") (two orthogonal edge families
    of nearly equal weight: a bimodal histogram whose main peak agrees with the outline)."""
    h = w = 512
    mask = np.zeros((h, w), np.uint8)
    mask[280:320, 231:281] = 1
    yy, xx = np.mgrid[0:h, 0:w]
    texture = np.zeros((h, w))
    if "along" in stripes:
        texture += contrast * ((np.floor((yy - 300) / period) % 2) == 0)
    if "across" in stripes:
        texture += contrast * ((np.floor((xx - 256) / period) % 2) == 0)
    roi = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
    grey = np.clip(np.where(roi, 90.0 + texture, 90.0), 0, 255).astype(np.uint8)
    return mask > 0, grey


def test_d10_oe_bimodal_edges_are_no_agreement_r2close_ruling_4a():
    """Lead ruling 4 (a) (r2close) at the pixel level, real G-7 kernels: a near-isotropic region (aspect 1.25) passes
    the outline gate; its region-ROI edges, two orthogonal families of nearly equal weight, pass the share gate and
    their main peak agrees with the outline (the old rule accepted), but their second peak exceeds
    D10_OUTLINE_SECOND_MAX: the ROI does not count, D10-OE refuses AXIS_INCONSISTENT "no_edge_agreement" with
    ``edges_bimodal``. The same region with one edge family (a unimodal histogram) is accepted."""
    t = _thin_dense_hover()
    scene, pose = t.scene, HandPose.from_view(t.view)
    images = scene._images(t.view)
    for stripes, accepted in ((("along", "across"), False), (("along",), True)):
        mask, grey = _near_isotropic_region(stripes=stripes)
        oe = scene._d10_oe(mask, dict(images, gray=grey, robot=np.zeros(mask.shape, bool)), pose, LOOP_TOP[2], None)
        outline = oe["outline"]
        assert oe["outline_ok"] and outline["second"] > 0.5, outline            # near-isotropic, gate passed
        region = oe["edges"]["region"]
        assert set(oe["edges"]) == {"region"} and region["gated"], region
        assert region["agree_rad"] <= scene_mod.AXIS_AGREE_MAX_RAD, region       # the edges agree with the outline
        assert oe["accepted"] is accepted and region["counts"] is accepted and region["unimodal"] is accepted
        names = {th_.name: th_ for th_ in oe["thresholds"]}
        assert names["D10_edge_second"].value == scene_mod.D10_OUTLINE_SECOND_MAX
        if accepted:
            assert region["second"] <= scene_mod.D10_OUTLINE_SECOND_MAX and oe["reason"] is None
            assert not oe["edges_bimodal"] and oe["edge_roi"] == "region"
        else:
            assert region["second"] > scene_mod.D10_OUTLINE_SECOND_MAX, region
            assert oe["reason"] == "no_edge_agreement" and oe["edges_bimodal"] is True
            assert oe["edge_roi"] is None and oe["agree_rad"] is None


class _EdgeAnswers:
    """An edge kernel stub answering per call in ROI order (region first, then hull); None = the kernel's failure."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, grey, roi, exclude, pose, z, f):
        answer = self.answers[self.calls]
        self.calls += 1
        return None if answer is None else dict(answer)


def test_d10_oe_counts_only_edge_rois_that_pass_both_gates_r2close_ruling_4a():
    """Lead ruling 4 (a) decision matrix (stub kernels, a banded seed: both ROIs): the outline (share 0.23, second
    0.57: the review-2 89 deg chain stage's) is accepted only within AXIS_AGREE_MAX_RAD of an edge ROI that passes its
    share gate AND its own second-peak gate (second <= D10_OUTLINE_SECOND_MAX, the bound included; a missing second is
    unimodal); a bimodal agreeing ROI with the other ROI contradicting or failing refuses with ``edges_bimodal``; an
    ungated agreeing ROI refuses without it."""
    t = _thin_dense_hover()
    scene, pose = t.scene, HandPose.from_view(t.view)
    band = scene._seed_band(t.seed)
    assert band["rule"] == "band"
    axis = np.array([1.0, 0.0])
    near, far = _rotated(axis, 0.8), _rotated(axis, 82.0)
    region = np.zeros((512, 512), bool)
    u, v = (int(round(x)) for x in t.uv)
    region[v - 10:v + 10, u - 15:u + 15] = True
    second_max = scene_mod.D10_OUTLINE_SECOND_MAX
    cases = [  # (region answer, hull answer, accepted, edges_bimodal, counting roi)
        ({"axis": near, "share": 0.27, "second": 0.95}, {"axis": far, "share": 0.30, "second": 0.40}, False, True,
         "hull"),
        ({"axis": near, "share": 0.27, "second": 0.95}, {"axis": _rotated(axis, 2.0), "share": 0.30, "second": 0.40},
         True, False, "hull"),
        ({"axis": near, "share": 0.27, "second": second_max}, None, True, False, "region"),
        ({"axis": near, "share": 0.27, "second": None}, None, True, False, "region"),
        ({"axis": near, "share": 0.27, "second": 0.95}, None, False, True, None),
        ({"axis": near, "share": 0.15, "second": 0.30}, {"axis": far, "share": 0.10, "second": 0.20}, False, False,
         None)]
    for region_answer, hull_answer, accepted, bimodal, roi in cases:
        stub = _kernel_stub(outline={"axis": axis, "share": 0.23, "second": 0.57, "two_sided": True})
        stub.edge_orientation = _EdgeAnswers(region_answer, hull_answer)
        scene._vision = stub
        oe = scene._d10_oe(region, scene._images(t.view), pose, LOOP_TOP[2], band)
        assert stub.edge_orientation.calls == 2 and set(oe["edges"]) == {"region", "hull"}
        assert oe["accepted"] is accepted and oe["edges_bimodal"] is bimodal, (region_answer, hull_answer, oe)
        assert oe["edge_roi"] == roi and oe["reason"] == (None if accepted else "no_edge_agreement")
        for e in oe["edges"].values():
            assert e["counts"] == (e["gated"] and e["unimodal"])


def test_near_isotropic_acceptance_from_bimodal_edges_is_not_carried_r2close_ruling_4a():
    """The review-2 89 deg flow (gate episode 2, +1 cm x): the chain view's near-isotropic region passed the outline
    gate (share 0.23, second 0.57) and agreed only with bimodal edges (second 0.95); the second stage failed the
    outline gate and S-12 carried the chain axis ("kept_earlier"). Under lead ruling 4 (a) the chain view is refused
    AXIS_INCONSISTENT "no_edge_agreement" (``edges_bimodal``; BR-1's grip trigger) and remembers no axis, so the
    second stage stays NOT_ELONGATED "outline_gate". With a unimodal agreeing ROI the old flow is unchanged (accepted,
    then carried)."""
    t = _thin_dense_hover()
    scene = t.scene
    axis = np.array([1.0, 0.0])
    outline = {"axis": axis, "share": 0.23, "second": 0.57, "two_sided": True}
    gate_off = _kernel_stub(outline={"axis": _rotated(axis, 89.0), "share": 0.25, "second": 0.89, "two_sided": True})
    for second, kept in ((0.95, False), (0.40, True)):
        scene.part_moved("C")
        chain = _kernel_stub(outline=outline, edge={"axis": _rotated(axis, 0.8), "share": 0.27, "second": second})
        scene._vision = chain
        rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
        assert isinstance(rec, scene_mod.PartRecord), rec
        check = rec.axis_check
        if kept:
            assert rec.is_elongated() and check["source"] == "outline_edges" and check["edges_bimodal"] is False
        else:
            assert not rec.is_elongated() and check["source"] == scene_mod.AXIS_INCONSISTENT
            assert check["reason"] == "no_edge_agreement" and check["edges_bimodal"] is True
            assert not scene.view_memory("C")["axes"]
        scene._vision = gate_off
        second_stage = scene.register("C", t.view, seed=t.seed, empty_hand=True)
        check = second_stage.axis_check
        if kept:
            assert second_stage.is_elongated() and check["reason"] == "kept_earlier"
            assert angle_deg(second_stage.elongation[0], axis) <= 1.0
        else:
            assert not second_stage.is_elongated() and check["source"] == scene_mod.NOT_ELONGATED
            assert check["reason"] == "outline_gate" and check["refused_as"] is None


# ------------------------------------------------------------------------- Wave 2 entry gate (lead decisions F1-F3)
def _thin_hover_with(*, dense=(), dense_sigma=0.001, grid=(), grid_sigma=0.001):
    """``_thin_dense_hover`` with per-extra claimed sigmas: ``dense`` = extra dense-lattice samples (feature keys
    "dense:<k>") and ``grid`` = grid samples absorbed BEFORE the lattice, each at its own projected pixel."""
    sc = fakes.free_object_with_thin_part_scene()
    world = make_world(sc, sc.info["hover_eef"]["moka_handle"])
    uv0 = world.project("wrist", LOOP_TOP)[0]
    scene, view, cal, info = started_scene(world, survey_seeds=[("C", tuple(uv0))])
    after = probe(world, scene)
    uv_seed = np.asarray(world.project("wrist", LOOP_TOP)[0], float)
    rng = np.random.default_rng(9)

    def resigma(survey, sigma):
        samples = [dataclasses.replace(s, sigma_z=sigma, sigma_xy=sigma / 2.0) if (s.feature_id or 0) >= 30000 else s
                   for s in survey.samples]
        return dataclasses.replace(survey, samples=samples)
    if grid:
        scene.absorb_survey(resigma(dense_survey(world, after, (0.0, 0.0), window_px=0, extra=grid, extra_keys=None,
                                                 rng=rng), grid_sigma), after)
    scene.absorb_survey(resigma(dense_survey(world, after, uv_seed, extra=dense, rng=rng), dense_sigma), after)
    scene.parts["C"] = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}))
    return SimpleNamespace(world=world, scene=scene, view=after, uv=uv_seed, seed=seed_at(world, after, LOOP_TOP),
                           sc=sc)


def _sample_index(scene, xyz):
    return int(np.argmin(np.linalg.norm(scene._samples["xyz"] - np.asarray(xyz, float)[None, :], axis=1)))


def test_f1_width_sigma_is_the_pixel_term_with_the_part_level_model_term(monkeypatch):
    """Lead decision F1: every width of a registered record carries sigma_w = hypot(pixel sigma_w, W_MODEL_M) (the
    TOP_MODEL_M pattern, DD-labelled); the width values, the narrowest width and the axis decision are those of the
    pixel-only rule, and the registration logs the model term as a threshold."""
    assert scene_mod.LABELS["W_MODEL_M"][0] == "DD" and scene_mod.W_MODEL_M == pytest.approx(0.0015)
    with_model = _thin_dense_hover()
    rec = with_model.scene.register("C", with_model.view, seed=with_model.seed, empty_hand=True)
    monkeypatch.setattr(scene_mod, "W_MODEL_M", 0.0)
    pixel_only = _thin_dense_hover()
    ref = pixel_only.scene.register("C", pixel_only.view, seed=pixel_only.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and isinstance(ref, scene_mod.PartRecord), (rec, ref)
    assert rec.width_by_yaw and len(rec.width_by_yaw) == len(ref.width_by_yaw)
    for (yaw, w), (yaw0, w0) in zip(rec.width_by_yaw, ref.width_by_yaw):
        assert yaw == yaw0 and w.value == pytest.approx(w0.value, abs=1e-12)
        assert w.sigma == pytest.approx(math.hypot(w0.sigma, 0.0015), rel=1e-9) and w.sigma >= 0.0015
    assert rec.width_at(0.3).value == pytest.approx(ref.width_at(0.3).value, abs=1e-12)
    assert rec.is_elongated() == ref.is_elongated()
    names = [t.name for t in with_model.scene.registration_thresholds("C")]
    assert "width_model_m" in names


def test_f2_dense_samples_on_the_part_above_its_top_are_withdrawn_and_correct_ones_kept():
    """Lead decision F2 (S-5 extended): the pointed lattice's samples INSIDE the registered region dilated by one
    cell that read above the record's top + max(D8_FLOOR_M, D8_SIGMA_K sigma_h) are withdrawn and their cells
    re-made (vacated, re-stamped from the kept samples, refilled); the lattice's correct on-part samples, one 4 mm
    above the top included (inside the tolerance), are kept and the record's top is unchanged. The tolerance is the
    3 sigma_h one (9 mm here, BS r3 review): a sample 7.5 mm above the top is kept, one 12 mm above is withdrawn."""
    above = [(LOOP_TOP[0] + 0.001, LOOP_TOP[1] - 0.006, LOOP_TOP[2] + 0.020),
             (LOOP_TOP[0] - 0.001, LOOP_TOP[1] + 0.006, LOOP_TOP[2] + 0.030)]   # the occlusion layer on the part
    just_above = (LOOP_TOP[0], LOOP_TOP[1] - 0.0015, LOOP_TOP[2] + 0.012)      # beyond 3 sigma_h: withdrawn
    within = (LOOP_TOP[0], LOOP_TOP[1] + 0.002, LOOP_TOP[2] + 0.004)           # a correct sample inside the band
    edge = (LOOP_TOP[0], LOOP_TOP[1] + 0.0005, LOOP_TOP[2] + 0.0075)           # inside 3 sigma_h, above D8_FLOOR_M
    t = _thin_hover_with(dense=above + [just_above, within, edge], dense_sigma=0.004)
    hf, scene = t.scene.hf, t.scene
    cells = {hf.index(p[:2]) for p in above}
    assert all(hf.measured[c] and hf.z[c] > LOOP_TOP[2] + 0.015 for c in cells)          # phantom heights stamped
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and rec.status == "registered", scene.last_registration["C"]
    assert abs(rec.top_z.value - LOOP_TOP[2]) <= 0.002 and not rec.top_is_lower_bound()
    log = scene.last_dsw["C"]
    tol = max(scene_mod.D8_FLOOR_M, scene_mod.D8_SIGMA_K * rec.sigma_h_m)
    assert log["skipped"] is None and log["near_skipped"] is None, log
    assert log["near_reference_z"] == pytest.approx(rec.top_z.value + tol)
    assert log["withdrawn_near"] >= len(above) and log["withdrawn"] == log["withdrawn_off"] + log["withdrawn_near"]
    s = scene._samples
    assert all(s["withdrawn"][_sample_index(scene, p)] and s["marker"][_sample_index(scene, p)]
               for p in above + [just_above])
    assert tol == pytest.approx(scene_mod.D8_SIGMA_K * rec.sigma_h_m) and 0.0075 < tol - 0.0005 and tol < 0.0115
    for p in (within, edge):
        k = _sample_index(scene, p)
        assert not s["withdrawn"][k] and not s["marker"][k], p                     # inside the tolerance: kept
    for c in cells:                                                                # no measured phantom remains
        assert not (hf.measured[c] and hf.z[c] > rec.top_z.value + tol), (c, hf.source[c], hf.z[c])
    region = np.zeros(hf.shape, bool)
    region[rec.region_cells[:, 0], rec.region_cells[:, 1]] = True
    iy, ix, inside = scene._sample_cells()
    on_part = s["dense"] & inside & (s["xyz"][:, 2] <= log["near_reference_z"])
    on_part[inside] &= region[iy[inside], ix[inside]]
    assert on_part.sum() >= 10 and not (s["withdrawn"] & on_part).any()              # the part's own lattice stays
    th_ = [x for x in scene.registration_thresholds("C") if x.name == "D8_dense_near_part_m"]
    assert th_ and th_[-1].value == pytest.approx(tol) and th_[-1].label == "DD"


def test_f2_a_raised_neighbour_seen_by_grid_samples_beside_the_part_is_not_withdrawn():
    """F2 negative: GRID samples of a real raised neighbour one cell beside the part (inside the near zone), reading
    25 mm above the part's top, are never withdrawn (only the pointed lattice is); their cells stay measured at the
    neighbour's height, also where a dense twin of the same height was withdrawn (the cell is re-stamped from them)."""
    side = 0.013 / 2.0
    wall = [(LOOP_TOP[0] + side + 0.004, LOOP_TOP[1] + along, LOOP_TOP[2] + 0.025) for along in (-0.004, 0.004)]
    twin = [(x + 0.0005, y, z) for x, y, z in wall]
    t = _thin_hover_with(dense=twin, dense_sigma=0.004, grid=wall, grid_sigma=0.004)
    scene, hf = t.scene, t.scene.hf
    rec = scene.register("C", t.view, seed=t.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord), scene.last_registration["C"]
    s = scene._samples
    assert not (s["withdrawn"] & ~s["dense"]).any()
    for p in wall:
        k = _sample_index(scene, p)
        assert not s["dense"][k] and not s["withdrawn"][k]
        c = hf.index(p[:2])
        assert hf.measured[c] and abs(hf.z[c] - p[2]) <= 0.002, (c, hf.source[c], hf.z[c])
    assert all(s["withdrawn"][_sample_index(scene, p)] for p in twin)
    assert scene.last_dsw["C"]["withdrawn_near"] + scene.last_dsw["C"]["withdrawn_off"] >= len(twin)


def test_f2_never_withdraws_above_a_lower_bound_top():
    """F2 negative: a record whose top is only a lower bound (the part may reach higher) withdraws nothing above its
    top ("near_skipped" "top_lower_bound"); S-5's off-part rule is unchanged."""
    above = [(LOOP_TOP[0] + 0.001, LOOP_TOP[1] - 0.006, LOOP_TOP[2] + 0.020)]
    twin = _thin_hover_with(dense=above, dense_sigma=0.004)
    rec = twin.scene.register("C", twin.view, seed=twin.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and twin.scene.last_dsw["C"]["withdrawn_near"] >= 1
    t = _thin_hover_with(dense=above, dense_sigma=0.004)
    lower = dataclasses.replace(rec, top_bound="lower", top_upper_z=rec.top_z.value + 0.05, partial=True)
    assert lower.top_is_lower_bound()
    log = t.scene._withdraw_dense("C", lower, t.view)
    assert log["near_skipped"] == "top_lower_bound" and log["near_reference_z"] is None and log["withdrawn_near"] == 0
    k = _sample_index(t.scene, above[0])
    assert not t.scene._samples["withdrawn"][k]
    c = t.scene.hf.index(above[0][:2])
    assert t.scene.hf.measured[c] and t.scene.hf.z[c] > LOOP_TOP[2] + 0.015


def _s6_field():
    """A synthetic S-6 field (as the S-6 test): the support measured everywhere at 0.900, the part a 1 x 5 cell bar
    along y at the field centre at the top 1.0345 (sigma_h 3 mm), and a registered record of it."""
    scene = scene_mod.SceneModel(RobotModel(), vision=VISION)
    hf = scene_mod.HeightField.empty((0.0, 0.0), half_extent_m=0.1)
    scene.hf = hf
    scene._fill_parent = np.full(hf.shape + (2,), -1, np.int32)
    scene._plane_member = np.full(hf.shape, -1, np.int16)
    top, support = 1.0345, 0.900
    for ij in np.ndindex(hf.shape):
        hf.stamp_ij(ij, support, 0.001, "kappa", 0)
    c = hf.index((0.0, 0.0))
    region = [(c[0] + dy, c[1]) for dy in range(-2, 3)]
    for ij in region:
        hf.stamp_ij(ij, top - 0.001, 0.001, "kappa", 0)
    rec = scene_mod.PartRecord("C", "thin part", "free", frozenset({"grip"}), xyz=np.r_[hf.cell_xy(c), top],
                               cov=np.diag([1e-6, 1e-6, 9e-6]), top_z=Meas(top, 0.003, "kappa"),
                               support_z=Meas(support, 0.001, "kappa"), region_cells=np.array(region),
                               region_xy=np.array([[-0.003, -0.013], [0.003, -0.013], [0.003, 0.013], [-0.003, 0.013]]),
                               sigma_h_m=0.003, status="registered", registration_step=0)
    scene.parts["C"] = rec
    return scene, hf, c, top


def _s6_fill(scene, hf, ij, parent, z):
    hf.vacate([ij], 1)
    hf.stamp_ij(ij, z, 0.005, "obstacle", 1)
    scene._fill_parent[ij] = parent


def test_f3_part_band_fills_of_the_parts_own_body_joined_through_unmeasured_cells_are_its_halo():
    """Lead decision F3 (S-6 connectivity, BPRR2-1): the part's own body (measured cells at the part's height) joined
    to the region through one or two UNMEASURED cells (unknown or a D7b fill), each touching measured raised ground,
    is connected: the fills its band-height cells parent are the part's halo; the bridge cells are logged and never
    parents. The measured joins of the stored records spanned 1-2 such cells."""
    scene, hf, c, top = _s6_field()
    body = [(c[0] + dy, c[1] + 3) for dy in range(-2, 3)]                       # 2 unmeasured columns between
    near_body = [(c[0] + dy, c[1] - 2) for dy in range(-1, 2)]                  # 1 unmeasured column between
    for ij in body + near_body:
        hf.stamp_ij(ij, top - 0.004, 0.007, "kappa", 0)
    for dy in range(-2, 3):
        hf.vacate([(c[0] + dy, c[1] + 1)], 1)                                   # unknown
        _s6_fill(scene, hf, (c[0] + dy, c[1] + 2), body[2], top - 0.004 + 0.015)  # the body's own D7b fill
    for dy in range(-1, 2):
        hf.vacate([(c[0] + dy, c[1] - 1)], 1)
    far = {"body": ((c[0] - 3, c[1] + 4), body[0]), "near_body": ((c[0], c[1] - 3), near_body[1])}
    for ij, parent in far.values():
        _s6_fill(scene, hf, ij, parent, top + 0.011)
    got = {tuple(map(int, ij)) for ij in scene.part_fill_cells("C")}
    assert far["body"][0] in got and far["near_body"][0] in got, (got, scene.part_fill_log["C"])
    assert {(c[0] + dy, c[1] + 2) for dy in range(-2, 3)} <= got                 # the bridge fills, parent the body
    log = scene.part_fill_log["C"]
    assert log["bridged_cells"] >= 3 + 2 and log["skipped"] is None
    parent_of = {tuple(map(int, ij)): tuple(map(int, scene._fill_parent[tuple(ij)])) for ij in got}
    assert all(hf.measured[p] for p in parent_of.values())


@pytest.mark.parametrize("gap", ["three_unmeasured", "measured_support", "plane_fill"])
def test_f3_a_separate_raised_object_at_the_parts_height_keeps_its_fills(gap):
    """F3 negatives (S-6's deliberate negative kept): a separate raised object at the part's height whose gap to the
    part is observed support (a measured support cell or a D7a plane fill) or a run of 3 unmeasured cells (the
    middle one touches no raised measurement) is not connected, and the fills it parents stay P7 blockers."""
    scene, hf, c, top = _s6_field()
    island = [(c[0] + dy, c[1] + 4) for dy in range(-1, 2)]
    for ij in island:
        hf.stamp_ij(ij, top - 0.002, 0.004, "kappa", 0)
    between = [(c[0] + dy, c[1] + dx) for dy in range(-2, 3) for dx in (1, 2, 3)]
    if gap == "three_unmeasured":
        hf.vacate(between, 1)
    elif gap == "measured_support":
        hf.vacate([ij for ij in between if ij[1] != c[1] + 2], 1)              # one support column stays measured
    else:
        hf.vacate(between, 1)
        for ij in [ij for ij in between if ij[1] == c[1] + 2]:
            hf.stamp_ij(ij, 0.900 + 0.0015, 0.0005, "plane", 1)                  # the support seen (D7a)
    fill = (c[0], c[1] + 6)
    _s6_fill(scene, hf, fill, island[1], top - 0.002 + 0.012)
    got = {tuple(map(int, ij)) for ij in scene.part_fill_cells("C")}
    assert fill not in got, (gap, got, scene.part_fill_log["C"])
    assert all(not hf.measured[ij] for ij in between if gap == "three_unmeasured")


def test_f2_is_skipped_when_the_regions_own_lattice_contradicts_the_top():
    """F2 premise check (BS r3 review): the same correct thin-part lattice cleaned against a record whose top was read
    15 mm low (a lower band taken for the top; sigma_h 3 mm, so the part's true top stands 6 mm above the F2
    reference). Most of the lattice samples inside the region cells read above top + tolerance, so the top is in doubt
    and F2 withdraws nothing ("near_skipped" "top_contradicted", counts logged): the part's measured region cells and
    heights stay. Without the check the part's own lattice was withdrawn and its region cells became fills."""
    good = _thin_dense_hover()
    rec = good.scene.register("C", good.view, seed=good.seed, empty_hand=True)
    assert isinstance(rec, scene_mod.PartRecord) and good.scene.last_dsw["C"]["near_skipped"] is None
    t = _thin_dense_hover()
    low = dataclasses.replace(rec, top_z=Meas(rec.top_z.value - 0.015, rec.top_z.sigma, rec.top_z.source))
    hf = t.scene.hf
    region = np.zeros(hf.shape, bool)
    region[rec.region_cells[:, 0], rec.region_cells[:, 1]] = True
    measured0, z0 = hf.measured & region, hf.z.copy()
    assert measured0.sum() >= 10 and np.nanmax(np.where(measured0, z0, np.nan)) > low.top_z.value + 0.009
    log = t.scene._withdraw_dense("C", low, t.view)
    assert log["near_skipped"] == "top_contradicted" and log["withdrawn_near"] == 0, log
    assert 2 * log["near_region"]["above"] > log["near_region"]["samples"] > 0
    iy, ix, inside = t.scene._sample_cells()
    zone = np.zeros(len(inside), bool)
    zone[inside] = scene_mod._dilate_cells(region, 1)[iy[inside], ix[inside]]
    assert not (t.scene._samples["withdrawn"] & zone).any()                      # nothing on or beside the part
    assert np.array_equal(hf.measured & region, measured0)
    assert np.allclose(hf.z[measured0], z0[measured0])


# ======================================================================= live-first blockers: BS-1 shared regions
def _scene_copy(scene):
    """An independent copy of a module-scoped fixture's scene (records, HeightField, samples and BS-1 state)."""
    out = copy.copy(scene)
    out.parts = dict(scene.parts)
    out.last_registration, out._registration_thresholds = {}, {}
    out._extents = dict(scene._extents)
    out.hf = copy.deepcopy(scene.hf)
    out._plane_member, out._fill_parent = scene._plane_member.copy(), scene._fill_parent.copy()
    out._samples = dict(scene._samples)
    out._views, out.mask_log = dict(scene._views), dict(scene.mask_log)
    out._grip_pairs, out._record_kind, out._area_prior = set(scene._grip_pairs), {}, {}
    out.region_conflicts = []
    return out


@pytest.fixture(scope="module")
def bs1_start():
    """task2_start's construction, private to the BS-1 tests (the shared fixture's records and seed lists are
    mutated by earlier tests)."""
    return task2_start.__wrapped__()


def _with_grip_part(t, *, grips=True, step_obj="D"):
    """The task-2-like start scene with a part F whose plan cells are D's (its area seed lands on D's region) and,
    with ``grips``, a plan step that grips D through F."""
    scene = _scene_copy(t.scene)
    d = t.plan.part("D")
    parts = tuple(t.plan.parts) + (PlanPart("F", "part text", "free", d.w_cell, d.x_cell),)
    steps = (PlanStep("place_on", step_obj, ("F",), "B", "any", (), ()),) if grips else ()
    plan = plan_of(parts, steps)
    roles = dict(_roles(plan), F=frozenset({"grip"}))
    scene.merge_plan(plan, roles, t.view, 2)
    return scene


def test_bs1_an_area_seeded_grip_part_on_its_objects_region_is_refused_and_keeps_its_record(bs1_start):
    """Grip-part lens BS-1 (primary fix): D (the object) registers from its plan cell; F, the grip part of a step
    whose obj is D, registers from the SAME area seed and picks D's region (cell IoU >= SHARED_REGION_IOU_MIN): it
    is refused part_not_found "region_of_part" with D's id and the IoU, and keeps its previous record unchanged
    (seeded, not "failed"); provisional calls are refused alike. D itself is unaffected."""
    t = bs1_start
    scene = _with_grip_part(t)
    assert ("F", "D") in scene._grip_pairs
    d = scene.register("D", t.view, empty_hand=True, allow_partial=True)
    assert isinstance(d, scene_mod.PartRecord) and len(d.region_cells), d
    before = scene.parts["F"]
    for provisional in (False, True):
        out = scene.register("F", t.view, empty_hand=True, allow_partial=True, provisional=provisional)
        assert isinstance(out, scene_mod.RegistrationFailure), (provisional, out)
        assert out.reason == "part_not_found" and out.evidence["why"] == "region_of_part"
        assert out.evidence["other"] == "D" and out.evidence["iou"] >= scene_mod.SHARED_REGION_IOU_MIN
        assert out.evidence["rule"] == "grip_part_refused"
        assert any(th_.name == "shared_region_iou" and th_.value == scene_mod.SHARED_REGION_IOU_MIN
                   for th_ in out.thresholds)
        assert scene.parts["F"] is before and before.status == "seeded"
    assert scene.parts["D"] is d and scene.region_conflicts and scene.region_conflicts[-1]["other"] == "D"
    overlap = scene.region_overlap("D", "D")
    assert overlap["iou"] == 1.0 and scene.region_overlap("D", "F") is None


def test_bs1_without_a_grip_relation_the_earlier_registration_keeps_the_region(bs1_start):
    """BS-1: two unrelated parts (no plan step grips one through the other) on one region: the later area-seeded
    registration is refused ("earlier_keeps"). Negative: a part whose plan cell lies on another region (B, A)
    registers as before."""
    t = bs1_start
    scene = _with_grip_part(t, grips=False)
    assert not scene._grip_pairs
    assert isinstance(scene.register("D", t.view, empty_hand=True, allow_partial=True), scene_mod.PartRecord)
    out = scene.register("F", t.view, empty_hand=True, allow_partial=True)
    assert isinstance(out, scene_mod.RegistrationFailure) and out.evidence["rule"] == "earlier_keeps"
    b = scene.register("B", t.view, empty_hand=True, allow_partial=True)
    a = scene.register("A", t.view, empty_hand=True, provisional=True)
    assert isinstance(b, scene_mod.PartRecord) and isinstance(a, scene_mod.PartRecord), (b, a)
    assert "region_of_part" not in json_dumps(scene.last_registration["B"])


def json_dumps(value):
    import json
    return json.dumps(value, default=str)


def test_bs1_the_object_registered_after_its_grip_part_keeps_the_region_and_the_grip_part_yields(bs1_start):
    """BS-1, the other order: F (grip part of D) registered first from the area seed and took the region; D's
    area-seeded registration of that region is accepted and F returns to its record before that registration (its
    earlier, independent estimate), logged in D's payload and ``region_conflicts``. A point-seeded grip record is never
    reverted by the scene (BR-3's check): only logged."""
    t = bs1_start
    scene = _with_grip_part(t)
    seeded = scene.parts["F"]
    f = scene.register("F", t.view, empty_hand=True, provisional=True)
    assert isinstance(f, scene_mod.PartRecord) and f.provisional
    d = scene.register("D", t.view, empty_hand=True, allow_partial=True)
    assert isinstance(d, scene_mod.PartRecord), d
    assert scene.parts["F"] is seeded and scene.parts["F"].xyz is None
    entry = scene.last_registration["D"]["region_of_part"][0]
    assert entry["yielded"] == "F" and entry["reverted"] and entry["overlap"]["iou"] >= 0.5
    # a point-seeded grip record sharing the region is only logged
    scene2 = _with_grip_part(t)
    f2 = scene2.register("F", t.view, empty_hand=True, provisional=True)
    assert isinstance(f2, scene_mod.PartRecord)
    scene2._record_kind["F"] = "point"
    scene2.register("D", t.view, empty_hand=True, allow_partial=True)
    assert scene2.parts["F"] is f2 and not scene2.last_registration["D"]["region_of_part"][0]["reverted"]


# ================================================================== live-first blockers: BS-2 unconfirmed cells
def _point_survey(xy, z, *, n=1, step=0, sigma=0.001):
    samples = [DepthSample((float(xy[0]), float(xy[1]), float(z)), sigma, 0.001, "kappa", None, None, step, None)
               for _ in range(n)]
    return SimpleNamespace(samples=samples, features=[], pose0=None)


@pytest.fixture
def bs2_scene():
    sc = fakes.planar_region_scene()
    world = make_world(sc, (-0.20, 0.08, 1.19))
    scene, view, cal, info = started_scene(world)
    hf = scene.hf
    # a support cell whose 8 neighbours are known and level with it (within D8)
    ok = hf.known & np.isfinite(hf.z)
    for iy, ix in np.argwhere(hf.measured & ok):
        block = (slice(iy - 2, iy + 3), slice(ix - 2, ix + 3))
        if iy < 3 or ix < 3 or iy > hf.shape[0] - 4 or ix > hf.shape[1] - 4:
            continue
        if ok[block].all() and float(np.ptp(hf.z[block])) < 0.002 and not hf.unconfirmed[block].any():
            return SimpleNamespace(scene=scene, view=view, ij=(int(iy), int(ix)), z0=float(hf.z[iy, ix]))
    pytest.skip("no level 5 x 5 block in the start field")


def test_bs2_a_single_high_cell_is_unconfirmed_until_a_later_sample_confirms_it(bs2_scene):
    """Grip-part lens BS-2: one measured cell standing above every known 8-neighbour by more than max(D8_FLOOR_M,
    D8_SIGMA_K hypot(sigma_c, sigma_n)) (here +110 mm, the rs2 false cell) is exported in ``hf.unconfirmed``; many
    samples of the SAME survey do not confirm it; a LATER survey's sample within D8 does. The cell is never deleted
    (its height stays in the field). The export survives a deep copy (the planner's input snapshot)."""
    t = bs2_scene
    scene, hf, ij = t.scene, t.scene.hf, t.ij
    xy = hf.cell_xy(ij)
    start = {tuple(c) for c in scene.unconfirmed_cells().tolist()}             # the start field's own (fake) spikes
    scene.absorb_survey(_point_survey(xy, t.z0 + 0.110, n=6, step=5), t.view)
    assert hf.z[ij] == pytest.approx(t.z0 + 0.110, abs=1e-6)
    assert hf.unconfirmed[ij] and {tuple(c) for c in scene.unconfirmed_cells().tolist()} == start | {ij}
    assert copy.deepcopy(hf).unconfirmed[ij] and scene.summary()["unconfirmed_cells"] == len(start) + 1
    scene.absorb_survey(_point_survey(xy, t.z0 + 0.111, step=9), t.view)            # a later survey, within D8
    assert not hf.unconfirmed[ij] and hf.z[ij] == pytest.approx(t.z0 + 0.1105, abs=1e-3)


def test_bs2_negatives_level_bump_raised_block_touch_and_sparse_neighbours(bs2_scene):
    """BS-2 negatives: a cell within the D8 tolerance of its neighbours (+2 mm), a raised 2 x 2 block (each cell has
    a raised neighbour), a touched cell (a touch is authoritative) and a high cell with fewer than
    UNCONFIRMED_MIN_NEIGHBOURS known neighbours are never unconfirmed."""
    t = bs2_scene
    scene, hf, (iy, ix) = t.scene, t.scene.hf, t.ij
    start = hf.unconfirmed.copy()                                             # the start field's own (fake) spikes
    scene.absorb_survey(_point_survey(hf.cell_xy((iy, ix)), t.z0 + 0.002, step=5), t.view)
    assert not (hf.unconfirmed & ~start).any()
    survey = SimpleNamespace(samples=[], features=[], pose0=None)
    for dy in (0, 1):
        for dx in (0, 1):
            survey.samples += _point_survey(hf.cell_xy((iy + dy, ix + dx)), t.z0 + 0.05, step=6).samples
    scene.absorb_survey(survey, t.view)
    assert not (hf.unconfirmed & ~start).any()
    # a touch on a single high cell
    cell = hf.cell_xy((iy - 2, ix - 2))
    scene.stamp_touch(scene_mod.rect_polygon(cell, (1.0, 0.0), 0.001, 0.001), t.z0 + 0.08, 0.001, 7)
    assert hf.source[iy - 2, ix - 2] == scene_mod._TOUCH_CODE and not (hf.unconfirmed & ~start).any()
    # a single high cell whose neighbours are unknown but two
    far = (iy + 2, ix - 2)
    neighbours = [(far[0] + a, far[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    hf.vacate(neighbours[2:], 8)
    scene.absorb_survey(_point_survey(hf.cell_xy(far), t.z0 + 0.09, step=8), t.view)
    assert not hf.unconfirmed[far]


# ------------------------------------------------------------- grasp-refusal decision J: S-17 own-band ring
def _ring_scene():
    """A 5 x 3 cell part region (top 0.955) on a 0.918 base; its far-edge row measured at the top height but left out
    of the region (one cell under-segmented), a halo row two cells out, and in the ring: a touch, a D7b fill, a cell
    below the band, a real obstacle above the band and an unmeasured cell."""
    from gpt_field.agent_speedup_v2_types import HeightField, PartRecord, RobotModel, hf_code
    scene = scene_mod.SceneModel(RobotModel())
    hf = HeightField.empty((0.0, 0.0), half_extent_m=0.05)
    hf.z[:], hf.sigma[:], hf.known[:], hf.source[:] = 0.918, 0.001, True, hf_code("kappa")
    c = hf.index((0.0, 0.0))
    region = np.array([(c[0] + dy, c[1] + dx) for dy in (-1, 0, 1) for dx in range(-2, 3)])
    hf.z[region[:, 0], region[:, 1]] = 0.955
    edge = [(c[0] + 2, c[1] + dx) for dx in range(-2, 3)]
    for ij in edge:
        hf.z[ij] = 0.953
    hf.z[c[0] + 3, c[1]] = 0.954                                  # two rows out: never ring
    hf.z[c[0] - 2, c[1]], hf.source[c[0] - 2, c[1]] = 0.955, hf_code("touch")
    hf.z[c[0] - 2, c[1] + 1], hf.source[c[0] - 2, c[1] + 1] = 0.956, hf_code("obstacle")
    hf.z[c[0] - 2, c[1] - 1] = 0.930                              # below the band: never ring
    hf.z[c[0] - 2, c[1] + 2] = 0.975                              # a real obstacle above the band: never ring
    blank = (c[0] - 2, c[1] - 2)                                  # unmeasured: never ring
    hf.z[blank], hf.sigma[blank], hf.known[blank], hf.source[blank] = np.nan, np.nan, False, 0
    scene.hf = hf
    scene._fill_parent = np.full(hf.shape + (2,), -1, dtype=np.int32)
    scene._fill_parent[c[0] - 2, c[1] + 1] = (c[0] - 1, c[1] + 1)
    scene.parts["A"] = PartRecord(pid="A", text="part A", move_prior="hinged", roles=frozenset({"grip"}),
                                  xyz=np.array([0.0, 0.0, 0.955]), cov=np.eye(3) * 4e-6,
                                  top_z=Meas(0.955, 0.003, "kappa"), support_z=Meas(0.918, 0.001, "kappa"),
                                  region_cells=region, status="registered", registration_step=0, sigma_h_m=0.003)
    return scene, hf, c, region, edge


def test_s17_part_ring_cells_are_the_measured_top_band_cells_one_cell_out():
    """S-17: the ring holds the region's one-cell (8-connected) neighbours measured in the part's S-6 top band (the
    far-edge row at 0.953 and the band-height cells around); never a cell two rows out, a touch, a D7b fill, a cell
    below the band, a real obstacle above the band, an unmeasured cell or a region cell; empty without a registered
    region; logged with its band."""
    scene, hf, c, region, edge = _ring_scene()
    assert scene_mod.OWN_RING_CELLS == 1 and scene_mod.LABELS["OWN_RING_CELLS"][0] == "DD"
    assert "agent_speedup_v2_scene.OWN_RING_CELLS" in th.CONSTANT_LABELS
    ring = {tuple(map(int, ij)) for ij in scene.part_ring_cells("A")}
    assert set(edge) <= ring
    assert not ring & {tuple(map(int, ij)) for ij in region}
    for ij in ((c[0] + 3, c[1]), (c[0] - 2, c[1]), (c[0] - 2, c[1] + 1), (c[0] - 2, c[1] - 1), (c[0] - 2, c[1] + 2),
               (c[0] - 2, c[1] - 2)):
        assert ij not in ring, ij
    dilated = {(c[0] + dy, c[1] + dx) for dy in range(-2, 3) for dx in range(-3, 4)}
    assert ring <= dilated
    log = scene.part_ring_log["A"]
    assert log["cells"] == len(ring) and log["ring_cells"] == 1 and log["skipped"] is None
    lo = 0.955 - max(scene_mod.D8_FLOOR_M, scene_mod.TOP_BAND_SIGMA_K * 0.003) - scene_mod.D8_FLOOR_M
    hi = 0.955 + scene_mod.D8_SIGMA_K * 0.003 + scene_mod.D8_FLOOR_M
    assert log["band_m"] == pytest.approx([lo, hi])
    assert len(scene.part_ring_cells("missing")) == 0 and scene.part_ring_log["missing"]["skipped"] == "not_registered"
    scene.parts["A"] = dataclasses.replace(scene.parts["A"], region_cells=None)
    assert len(scene.part_ring_cells("A")) == 0 and scene.part_ring_log["A"]["skipped"] == "not_registered"


def test_s17_ring_is_pure_and_bp2b_parent_lookup_is_not_adopted():
    """S-17 reads the field without changing it (no cell, source, sigma or fill parent moves), and the BP-2b fill-
    parent helper of remedy 1 is not part of the scene (decision J)."""
    scene, hf, c, region, edge = _ring_scene()
    before = (hf.z.copy(), hf.sigma.copy(), hf.source.copy(), hf.known.copy(), scene._fill_parent.copy())
    scene.part_ring_cells("A")
    after = (hf.z, hf.sigma, hf.source, hf.known, scene._fill_parent)
    for a, b in zip(before, after):
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
    assert not hasattr(scene, "fill_parents")
