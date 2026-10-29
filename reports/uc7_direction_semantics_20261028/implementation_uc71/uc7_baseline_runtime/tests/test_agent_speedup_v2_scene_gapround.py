"""Gap round (judge decision §2.2 / §2.3 / §2.6), owner scene / geometry / observers: focused tests, one positive and
one negative per rule. X-1 disturbance epochs, SAL-R1 stale fills, SAL-R4 BS-2 neighbours, HS-H1 whole-body held
frame, HS-H2 template ownership, MC-R1 seed-band fallback guards, MC-R2a kept whole record, MC-R2b annulus (geometry
kernel and scene decision), MC-R3 PO3 / PO1s / PO5 / PO6, HS-H3 wrist arbitration (CoMotion carry scope, Upright)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gpt_field import agent_speedup_v2_geometry as geometry
from gpt_field import agent_speedup_v2_observers as O
from gpt_field import agent_speedup_v2_scene as scene_mod
from gpt_field.agent_speedup_v2_types import (DepthSample, HeightField, Meas, PartRecord, RegistrationFailure,
                                              RobotModel, Seed)
from gpt_field.wrist_servo import NCCTrack

ROBOT = RobotModel()


# ------------------------------------------------------------------------------------------------ helpers
def bare_scene():
    sc = scene_mod.SceneModel(ROBOT)
    sc.hf = HeightField.empty(np.zeros(2), half_extent_m=0.1)
    sc.hf.unconfirmed = np.zeros(sc.hf.shape, dtype=bool)
    sc._plane_member = np.full(sc.hf.shape, -1, dtype=np.int16)
    sc._fill_parent = np.full(sc.hf.shape + (2,), -1, dtype=np.int32)
    sc._fill_parent_z = np.full(sc.hf.shape, np.nan)
    return sc


def square(cx, cy, half):
    return np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]])


def record(sc, pid, text, *, centre, half, top, sigma=0.002, status="registered", partial=False):
    poly = square(centre[0], centre[1], half)
    return PartRecord(pid=pid, text=text, move_prior="free", roles=frozenset({"grip"}),
                      xyz=np.array([centre[0], centre[1], top]), cov=np.eye(3) * 0.002 ** 2,
                      top_z=Meas(top, sigma, "kappa"), region_xy=poly, region_cells=sc.hf.cells_in_polygon(poly),
                      status=status, registration_step=0, sigma_h_m=sigma, partial=partial)


def stamp(sc, ij, z, source="kappa"):
    sc.hf.stamp_ij(ij, z, 0.001, source, 0)


def obstacle(sc, ij, parent):
    sc.hf.stamp_ij(ij, float(sc.hf.z[parent]) + 0.003, 0.001, "obstacle", 0)
    sc._fill_parent[ij] = parent
    sc._fill_parent_z[ij] = float(sc.hf.z[parent])


# ------------------------------------------------------------------------------------------------ X-1
def test_x1_disturb_raises_the_epoch_of_the_object_parts_only_and_window_start_keeps_widths():
    sc = bare_scene()
    sc.parts["A"] = record(sc, "A", "the box: lid", centre=(0.0, 0.0), half=0.01, top=0.10)
    sc.parts["B"] = record(sc, "B", "the box: body", centre=(0.0, 0.02), half=0.02, top=0.08)
    sc.parts["C"] = record(sc, "C", "the cup: wall", centre=(0.06, 0.0), half=0.01, top=0.09)
    sc._grip_pairs.add(("A", "B"))
    sc._note_body("B", sc.parts["B"], False)
    assert sc.body_record("B") is not None
    sc._views["B"] = {"axes": [(np.array([1.0, 0.0]), 0.9, 0)], "conflicted": True, "widths": ((0.0, Meas(0.02, 0.001,
                      "kappa")),), "width_step": 0}
    sc.window_start("B")                                     # MC-R2a / L6: axis memory and conflict reset only
    assert sc._views["B"]["axes"] == [] and not sc._views["B"]["conflicted"] and sc._views["B"]["widths"]
    assert sc.body_record("B") is not None                   # the window start does not erase the body record
    assert sc.disturb("A", 5, "descent_touch") == 1
    assert (sc.epoch_of("A"), sc.epoch_of("B"), sc.epoch_of("C")) == (1, 1, 0)
    assert sc.body_record("B") is None and sc.disturbances[-1]["cause"] == "descent_touch"


# ------------------------------------------------------------------------------------------------ SAL-R1
def test_sal_r1_a_vacated_or_restamped_parent_vacates_its_fills_field_wide():
    sc = bare_scene()
    p1, p2 = (10, 10), (30, 30)
    stamp(sc, p1, 0.10)
    stamp(sc, p2, 0.05)
    obstacle(sc, (10, 11), p1)
    obstacle(sc, (30, 31), p2)
    sc._refresh_unconfirmed()
    assert sc.hf.source[10, 11] == scene_mod._OBSTACLE_CODE        # parents unchanged: fills live
    sc.hf.vacate([p1], 1)
    sc._refresh_unconfirmed()                                       # the post-write hook (no view involved)
    assert sc.hf.source[10, 11] == 0 and sc._fill_parent[10, 11, 0] == -1
    assert sc.hf.source[30, 31] == scene_mod._OBSTACLE_CODE        # a fill of another parent untouched
    stamp(sc, p2, 0.07)                                             # re-stamped at another height
    sc._refresh_unconfirmed()
    assert sc.hf.source[30, 31] == 0 and sc.last_fill_consistency["vacated"] == 1


def test_sal_r1_a_fill_never_parents_a_fill():
    sc = bare_scene()
    stamp(sc, (10, 10), 0.10)
    obstacle(sc, (10, 11), (10, 10))
    sc.hf.stamp_ij((10, 12), 0.2, 0.001, "obstacle", 0)
    sc._fill_parent[10, 12] = (10, 11)                              # parented to a fill: stale by construction
    sc._fill_parent_z[10, 12] = float(sc.hf.z[10, 11])
    sc._refresh_unconfirmed()
    assert sc.hf.source[10, 11] == scene_mod._OBSTACLE_CODE and sc.hf.source[10, 12] == 0


# ------------------------------------------------------------------------------------------------ SAL-R4
def _spike_field(sc):
    centre = (20, 20)
    stamp(sc, centre, 0.12)
    ring = [(19, 19), (19, 20), (19, 21), (20, 19), (20, 21), (21, 19), (21, 20), (21, 21)]
    for ij in ring[:3]:
        stamp(sc, ij, 0.02)                                         # three measured low neighbours
    for ij in ring[3:]:
        obstacle(sc, ij, centre)                                    # the spike's own fills at its height + 3 sigma
    return centre


def test_sal_r4_a_spike_surrounded_by_its_own_fills_is_unconfirmed():
    sc = bare_scene()
    centre = _spike_field(sc)
    sc._refresh_unconfirmed()
    assert sc.hf.unconfirmed[centre]                                # fills are no neighbours (SAL-R4)


def test_sal_r4_a_spike_confirmed_by_a_second_group_stays_confirmed():
    sc = bare_scene()
    centre = _spike_field(sc)
    xy = sc.hf.cell_xy(centre)
    for survey in (0, 1):
        sc._add_samples([DepthSample((float(xy[0]), float(xy[1]), 0.12), 0.001, 0.0025, "kappa", None, None, 0,
                                     None)], survey=survey)
    sc._refresh_unconfirmed()
    assert not sc.hf.unconfirmed[centre]


# ------------------------------------------------------------------------------------------------ MC-R3
def test_po3_refuses_a_region_inside_another_objects_part_at_the_same_surface():
    sc = bare_scene()
    sc.parts["B"] = record(sc, "B", "the tray: top", centre=(0.0, 0.0), half=0.03, top=0.10)
    cells = sc.hf.cells_in_polygon(square(0.0, 0.0, 0.01))
    refusal, _ = sc._part_object_check("A", np.array([0.0, 0.0, 0.10]), cells, Meas(0.101, 0.002, "kappa"))
    assert refusal is not None and refusal["why"] == "region_of_other_object" and refusal["other"] == "B"
    assert "region_of_other_object" in scene_mod.PART_NOT_FOUND_WHY if hasattr(scene_mod, "PART_NOT_FOUND_WHY") \
        else True


def test_po3_passes_an_object_resting_on_another_and_parts_of_the_same_object():
    sc = bare_scene()
    sc.parts["B"] = record(sc, "B", "the tray: top", centre=(0.0, 0.0), half=0.03, top=0.10)
    sc.parts["A"] = PartRecord("A", "the cup: wall", "free", frozenset({"grip"}))
    sc.parts["D"] = PartRecord("D", "the tray: rim", "free", frozenset({"grip"}))
    cells = sc.hf.cells_in_polygon(square(0.0, 0.0, 0.01))
    resting, _ = sc._part_object_check("A", np.array([0.0, 0.0, 0.16]), cells, Meas(0.16, 0.002, "kappa"))
    same, _ = sc._part_object_check("D", np.array([0.0, 0.0, 0.10]), cells, Meas(0.10, 0.002, "kappa"))
    assert resting is None and same is None


def test_po1s_and_po5_bound_a_grip_sub_part_by_its_objects_start_record():
    sc = bare_scene()
    sc.parts["B"] = record(sc, "B", "the box: body", centre=(0.0, 0.0), half=0.02, top=0.08)
    sc._start_record["B"] = sc.parts["B"]
    sc._grip_pairs.add(("A", "B"))
    small = sc.hf.cells_in_polygon(square(0.0, 0.01, 0.005))
    near, _ = sc._part_object_check("A", np.array([0.0, 0.05, 0.09]), small, Meas(0.09, 0.002, "kappa"))
    far, ths = sc._part_object_check("A", np.array([0.4, 0.0, 0.09]), small, Meas(0.09, 0.002, "kappa"))
    assert near is None and far["why"] == "far_from_object"
    assert any(t.name == "PO1s_object_reach_m" for t in ths)
    n_obj = len(sc.parts["B"].region_cells)
    big = sc.hf.cells_in_polygon(square(0.0, 0.0, 0.0225))
    small2 = sc.hf.cells_in_polygon(square(0.0, 0.0, 0.0195))
    over, _ = sc._part_object_check("A", np.array([0.0, 0.0, 0.09]), big, Meas(0.09, 0.002, "kappa"))
    under, _ = sc._part_object_check("A", np.array([0.0, 0.0, 0.09]), small2, Meas(0.09, 0.002, "kappa"))
    assert len(big) / n_obj > 1.1 and over["why"] == "larger_than_object"
    assert len(small2) / n_obj < 1.0 and under is None


def test_po6_keeps_an_undisturbed_sub_part_record_whose_top_jumps_up():
    sc = bare_scene()
    old = record(sc, "A", "the box: lid", centre=(0.0, 0.0), half=0.01, top=0.08)
    new = record(sc, "A", "the box: lid", centre=(0.0, 0.0), half=0.01, top=0.11)
    sc._grip_pairs.add(("A", "B"))
    sc._record_epoch["A"] = sc.epoch_of("A")
    kept, check = sc._po6_check("A", old, new)
    assert kept is old and check["kept"] and check["jump_m"] > check["tol_m"]
    sc.disturb("A", 1, "close")                                     # after a contact the view decides alone
    after, check2 = sc._po6_check("A", old, new)
    assert after is new and check2 is None
    sc._grip_pairs.clear()                                          # an object-level part: no PO6
    sc._record_epoch["A"] = sc.epoch_of("A")
    assert sc._po6_check("A", old, new) == (new, None)


# ------------------------------------------------------------------------------------------------ MC-R2a
def test_r2a_keeps_the_whole_record_in_the_same_epoch_and_not_after_contact():
    sc = bare_scene()
    whole = record(sc, "A", "the box: lid", centre=(0.0, 0.0), half=0.02, top=0.10)
    sc._whole["A"] = {"record": whole, "epoch": sc.epoch_of("A"), "step": 0}
    kept, log = sc._kept_whole("A", np.array([0.005, 0.0, 0.10]), Meas(0.101, 0.002, "kappa"), 0.002)
    assert kept is whole and log["kept"]
    outside, log2 = sc._kept_whole("A", np.array([0.08, 0.0, 0.10]), Meas(0.101, 0.002, "kappa"), 0.002)
    assert outside is None and log2["why"] == "grip_outside"
    lower, log3 = sc._kept_whole("A", np.array([0.005, 0.0, 0.07]), Meas(0.07, 0.002, "kappa"), 0.002)
    assert lower is None and log3["why"] == "top_differs"
    sc.disturb("A", 1, "descent_touch")
    gone, log4 = sc._kept_whole("A", np.array([0.005, 0.0, 0.10]), Meas(0.101, 0.002, "kappa"), 0.002)
    assert gone is None and log4["why"] in ("disturbed", "no_whole_record")


# ------------------------------------------------------------------------------------------------ MC-R2b
def _ring_points(arc_deg, *, r=0.04, wall=0.006, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    a = np.radians(rng.uniform(0.0, arc_deg, n))
    rr = r + rng.uniform(-wall / 2, wall / 2, n)
    return np.column_stack([rr * np.cos(a), rr * np.sin(a)])


@pytest.mark.parametrize("arc", [150.0, 270.0])
def test_annulus_fit_measures_arcs_walls_and_radius(arc):
    fit = geometry.annulus_fit(_ring_points(arc), outlier_k=3.0, min_points=12)
    assert fit["radius_m"] == pytest.approx(0.04, abs=0.002) and fit["wall_m"] == pytest.approx(0.8 * 0.006, abs=0.0015)
    assert fit["arc_deg"] == pytest.approx(arc, abs=6.0) and fit["rms_m"] <= fit["wall_m"]


def test_annulus_fit_on_a_quarter_arc_keeps_the_evaluated_algebraic_fit():
    """The kernel is the measure_consistency c3_eval fit (algebraic circle, robust refits): on a 90 deg arc with a
    6 mm band it under-reads the radius (the lane's recorded R2b centre-error risk) and keeps a thin wall."""
    fit = geometry.annulus_fit(_ring_points(90.0, n=3000, seed=3), outlier_k=3.0, min_points=12)
    assert 0.025 <= fit["radius_m"] <= 0.042 and fit["wall_m"] <= 0.5 * fit["radius_m"]


def test_annulus_fit_on_a_disk_and_a_bar_and_too_few_points():
    rng = np.random.default_rng(1)
    disk = rng.uniform(-0.03, 0.03, (2000, 2))
    disk = disk[np.linalg.norm(disk, axis=1) <= 0.03]
    fit = geometry.annulus_fit(disk, outlier_k=3.0, min_points=12)
    assert fit is None or fit["wall_m"] > 0.5 * fit["radius_m"]
    bar = np.column_stack([rng.uniform(-0.05, 0.05, 800), rng.uniform(-0.004, 0.004, 800)])
    fit_bar = geometry.annulus_fit(bar, outlier_k=3.0, min_points=12)
    assert fit_bar is None or fit_bar["arc_deg"] < scene_mod.ANNULUS_MIN_ARC_DEG or \
        fit_bar["wall_m"] > 0.5 * fit_bar["radius_m"]
    assert geometry.annulus_fit(np.zeros((5, 2)), outlier_k=3.0, min_points=12) is None


class _FitStub:
    """A vision stand-in whose annulus kernel returns a fixed fit (the scene decision alone is under test)."""

    def __init__(self, fit):
        self.fit = fit

    def annulus_fit(self, pts, *, outlier_k, min_points):
        return dict(self.fit)


def _annulus_decision(monkeypatch, pts, *, centre_px_inside, fit=None):
    sc = bare_scene()
    sc._vision = geometry if fit is None else _FitStub(fit)
    component = np.zeros((64, 64), dtype=bool)
    component[10:20, 10:20] = True
    monkeypatch.setattr(sc, "_region_points", lambda mask, pose, z: pts)
    uv = (15.0, 15.0) if centre_px_inside else (40.0, 40.0)
    monkeypatch.setattr(scene_mod, "_project", lambda cam, pose, xyz: (np.array([uv]), np.array([1.0])))
    return sc._annulus(component, None, Meas(0.1, 0.002, "kappa"), 1000.0)


def test_r2b_accepts_a_150_degree_ring(monkeypatch):
    out = _annulus_decision(monkeypatch, _ring_points(150.0), centre_px_inside=False)
    assert out["accepted"] and all(out["checks"].values())


@pytest.mark.parametrize("case", ["disk", "short_arc", "thick_wall"])
def test_r2b_refuses_a_disk_a_short_arc_and_a_thick_wall(monkeypatch, case):
    if case == "disk":
        out = _annulus_decision(monkeypatch, _ring_points(360.0, r=0.02, wall=0.03), centre_px_inside=True)
    elif case == "short_arc":
        fit = {"centre": np.zeros(2), "radius_m": 0.04, "wall_m": 0.004, "rms_m": 0.001, "arc_deg": 60.0, "n": 500}
        out = _annulus_decision(monkeypatch, _ring_points(60.0), centre_px_inside=False, fit=fit)
        assert out["checks"] == {"arc": False, "wall": True, "centre_outside": True, "rms": True}
    else:
        out = _annulus_decision(monkeypatch, _ring_points(200.0, r=0.03, wall=0.025), centre_px_inside=False)
    assert out is None or not out["accepted"]


def test_r2b_radial_width_is_the_wall():
    widths = tuple((k * math.pi / 12, Meas(0.05, 0.001, "kappa")) for k in range(12))
    decided = {"widths": widths, "width_source": "region"}
    out = scene_mod.SceneModel._annulus_widths(decided, {"centre": np.zeros(2), "wall_m": 0.006},
                                               np.array([0.0, 0.04, 0.1]), 1000.0)
    radial = [w for y, w in out["widths"] if abs(y - math.pi / 2) < 1e-9][0]
    assert radial.value == pytest.approx(0.006) and out["width_source"] == "annulus"
    assert sum(1 for _, w in out["widths"] if w.value == pytest.approx(0.05)) == 11


# ------------------------------------------------------------------------------------------------ MC-R1
def _band_scene(monkeypatch, *, s1_z=0.12, near_z=0.08, n_near=12):
    """S-1 samples (a raised band) 30 px from the seed and a lower band right at the seed: pixel = 1000 x world xy."""
    sc = bare_scene()
    sc.camera = object()
    monkeypatch.setattr(scene_mod, "_project", lambda cam, pose, xyz: (np.asarray(xyz, float).reshape(-1, 3)[:, :2]
                                                                        * 1000.0, np.ones(len(np.asarray(xyz)
                                                                                              .reshape(-1, 3)))))
    samples = []
    for k in range(12):
        samples.append(DepthSample((0.030 + 0.001 * (k % 4), 0.001 * (k // 4), s1_z), 0.001, 0.0025, "kappa", None,
                                   None, 0, None))
    for k in range(n_near):
        samples.append(DepthSample((0.001 * (k % 4), 0.001 * (k // 4), near_z), 0.001, 0.0025, "kappa", None, None,
                                   0, None))
    sc._add_samples(samples)
    monkeypatch.setattr(sc, "_live_mask", lambda: np.ones(len(sc._samples["xyz"]), bool))
    seed = Seed("wrist", (0.0, 0.0), 0, "fine", 8.0, (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0))
    s1 = {"rule": "band", "top": s1_z, "b_m": 0.003, "sigma_h": 0.002, "n": 12, "members": np.arange(12)}
    return sc, seed, s1


def test_r1_nearest_band_is_the_band_at_the_seed(monkeypatch):
    sc, seed, s1 = _band_scene(monkeypatch)
    near = sc._nearest_band(seed, s1)
    assert near["rule"] == "band" and near["top"] == pytest.approx(0.08, abs=0.001) and near["n"] == 12


@pytest.mark.parametrize("case,refused", [("same", "same_band"), ("support", "G6_support"), ("own_top", "G7_own_top")])
def test_r1_fallback_guards_refuse(monkeypatch, case, refused):
    kw = {"same": {"near_z": 0.1205}, "support": {"n_near": 6}, "own_top": {}}[case]
    sc, seed, s1 = _band_scene(monkeypatch, **kw)
    if case == "same":
        s1["top"] = 0.1205
    rec = PartRecord("A", "the box: lid", "free", frozenset({"grip"}))
    if case == "own_top":
        sc.parts["A"] = record(sc, "A", "the box: lid", centre=(0.03, 0.0), half=0.01, top=0.12)
    else:
        sc.parts["A"] = rec
    monkeypatch.setattr(sc, "_register_seed", lambda *a, **k: pytest.fail("no re-registration expected"))
    failure = RegistrationFailure("A", "part_not_found", {"why": "seed_outside_region"}, ())
    out = sc._seed_band_fallback(sc.parts["A"], None, seed, s1, failure, empty_hand=True, allow_partial=False)
    assert out is None and sc.last_seed_band_fallback["refused"] == refused


def test_r1_fallback_accepts_a_measured_record_and_restores_on_a_guard(monkeypatch):
    sc, seed, s1 = _band_scene(monkeypatch)
    sc.parts["A"] = PartRecord("A", "the box: lid", "free", frozenset({"grip"}))
    monkeypatch.setattr(sc, "_members_hull", lambda band, pose, d: None)
    monkeypatch.setattr(scene_mod.HandPose, "from_view", staticmethod(lambda view: None))
    good = record(sc, "A", "the box: lid", centre=(0.0, 0.0), half=0.01, top=0.08)
    good = PartRecord(**{**good.__dict__, "width_by_yaw": ((0.0, Meas(0.02, 0.001, "kappa")),)})

    def register(*a, **k):
        sc.parts["A"] = good
        sc.last_registration["A"] = {"collapse": {"collapsed": False}}
        return good
    monkeypatch.setattr(sc, "_register_seed", register)
    sc.cal = type("Cal", (), {"gripper": type("G", (), {"open_gap_mm": 77.4})()})()
    failure = RegistrationFailure("A", "geometry_unmeasured", {"why": "region_too_small"}, ())
    out = sc._seed_band_fallback(sc.parts["A"], None, seed, s1, failure, empty_hand=True, allow_partial=False)
    assert out is good and sc.last_registration["A"]["flag"] == "seed_band_fallback"
    wide = PartRecord(**{**good.__dict__, "width_by_yaw": ((0.0, Meas(0.5, 0.001, "kappa")),)})
    sc.parts["A"] = PartRecord("A", "the box: lid", "free", frozenset({"grip"}))
    before = sc.parts["A"]

    def register_wide(*a, **k):
        sc.parts["A"] = wide
        sc.last_registration["A"] = {"collapse": {"collapsed": False}}
        return wide
    monkeypatch.setattr(sc, "_register_seed", register_wide)
    out2 = sc._seed_band_fallback(sc.parts["A"], None, seed, s1, failure, empty_hand=True, allow_partial=False)
    assert out2 is None and sc.parts["A"] is before and "G2" in sc.last_seed_band_fallback["refused"]


# ------------------------------------------------------------------------------------------------ HS-H1 / HS-H2
def _held_scene():
    sc = bare_scene()
    sc.parts["D"] = record(sc, "D", "the box: body", centre=(0.0, 0.0), half=0.008, top=0.08)    # a sub-feature
    sc.parts["C"] = record(sc, "C", "the box: tab", centre=(0.0, 0.035), half=0.006, top=0.07)
    sc._grip_pairs.add(("C", "D"))
    body = record(sc, "D", "the box: body", centre=(0.0, 0.0), half=0.03, top=0.08)
    sc._note_body("D", body, False)
    return sc, scene_mod.HeldFrameEstimator(sc, ROBOT)


def _pose(xy):
    return type("Pose", (), {"grip": np.array([xy[0], xy[1], 0.1])})()


def test_h1_the_larger_body_record_of_the_epoch_replaces_the_footprint():
    sc, est = _held_scene()
    log = est._body_footprint("D", "C", _pose((0.0, 0.035)))
    assert log["footprint"] == "body" and log["body_area_m2"] > log["current_area_m2"]
    _, r_obj, _, _ = est._footprint("D")
    assert r_obj == pytest.approx(0.03 * math.sqrt(2), abs=0.002)
    assert len(sc.parts["D"].region_cells) > 100


@pytest.mark.parametrize("case", ["disturbed", "not_adjacent", "other_object", "current_largest"])
def test_h1_negatives_keep_the_current_record(case):
    sc, est = _held_scene()
    before = sc.parts["D"]
    if case == "disturbed":
        sc.disturb("D", 1, "descent_touch")
        why = "no_body_record"
    elif case == "not_adjacent":
        sc.parts["C"] = record(sc, "C", "the box: tab", centre=(0.0, 0.08), half=0.006, top=0.07)
        why = "not_adjacent_to_grip"
    elif case == "other_object":
        sc.parts["E"] = record(sc, "E", "the cup: wall", centre=(0.015, 0.0), half=0.005, top=0.09)
        why = "overlaps_other_object"
    else:
        sc.parts["D"] = before = record(sc, "D", "the box: body", centre=(0.0, 0.0), half=0.035, top=0.08)
        why = "current_is_largest"
    log = est._body_footprint("D", "C", _pose((0.0, 0.035)))
    assert log["footprint"] == "current" and log["why"] == why and sc.parts["D"] is before


def test_h1_whole_grip_needs_the_grip_inside_the_body():
    sc, est = _held_scene()
    sc._grip_pairs.clear()
    assert est._body_footprint("D", "D", _pose((0.2, 0.2)))["why"] == "grip_outside_body"
    assert est._body_footprint("D", "D", _pose((0.01, 0.0)))["footprint"] == "body"


def test_h2_a_template_on_the_hand_is_rejected_and_one_off_the_hand_kept(monkeypatch):
    sc, est = _held_scene()
    gray = np.random.default_rng(0).uniform(0, 255, (128, 128))
    track = NCCTrack(gray, (64.0, 64.0))
    hand = np.zeros((128, 128), bool)
    hand[70:80, 60:70] = True
    monkeypatch.setattr(sc, "hand_swept_mask", lambda camera, view, yaws: hand)
    out, info = est._owned_template(track, None)
    assert out is None and info["flag"] == "external_template_on_robot"
    far = NCCTrack(gray, (20.0, 20.0))
    kept, info2 = est._owned_template(far, None)
    assert kept is far and info2["flag"] is None


# ------------------------------------------------------------------------------------------------ HS-H3
import test_agent_speedup_v2_observers as OT   # noqa: E402  (the observer fixtures: fake carry views)


def test_h3_carry_external_displaced_with_the_wrist_holding_is_pending_unverified():
    w, views = OT._load_test_views(tilt_per_step_deg=0.6)
    carry = OT.feed(OT._load_observer(w, views[0], ext_point=OT.SUPPORT_POINT, rotation_verdict=False), views)
    last = carry[-1]
    assert (last.verdict, last.evidence["cause"], last.evidence["external_cause"]) == \
        ("pending", "external_unverified", "displaced")
    assert not any(r.verdict == "met" for r in carry if r.evidence["external"]["lag_with_wrist_held"])


def test_h3_load_test_and_no_wrist_model_keep_their_verdicts():
    w, views = OT._load_test_views(tilt_per_step_deg=0.6)
    load = OT.feed(OT._load_observer(w, views[0], ext_point=OT.SUPPORT_POINT), views)
    assert (load[-1].reason, load[-1].evidence["cause"]) == ("grip_unstable", "external_lag")
    nowrist = OT.feed(OT._load_observer(w, views[0], wrist=False, ext_point=OT.SUPPORT_POINT,
                                        rotation_verdict=False), views)
    assert (nowrist[-1].verdict, nowrist[-1].reason, nowrist[-1].evidence["cause"]) == \
        ("violated", "grip_not_on_obj", "displaced")


def _upright_with_wrist(estimate):
    w, views = OT._load_test_views()
    co = OT._load_observer(w, views[0], rotation_verdict=False)
    co.wrist_estimate_deg = estimate
    q0 = OT.matrix_to_quaternion_xyzw(OT.R_T2)
    start = OT.make_view(OT.state_with(OT.R_T2, 14.0), external=OT.bar_image(0.0))
    gray = cv2.cvtColor(start.external_rgb_upright, cv2.COLOR_RGB2GRAY).astype(float)
    obs = O.UprightObserver(robot=ROBOT, grasp_quaternion=q0, held=OT.held_frame(),
                            ext_track=NCCTrack(gray, (256.0, 256.0), radius=16, search=24),
                            axis_fn=OT.structure_axis_stand_in, tilt_fn=co.wrist_tilt_deg)
    obs.start(start)
    return obs, start


def test_h3_upright_wrist_estimate_decides_and_external_tilt_is_evidence():
    obs, start = _upright_with_wrist(2.0)
    rep = obs.update(start, None, OT.make_view(OT.state_with(OT.R_T2, 14.0), external=OT.bar_image(34.0)),
                     {"primitive": "held_reach"})
    assert rep.verdict == "met" and rep.evidence["tilt_source"] == "wrist_estimate"
    row = rep.evidence["external_axis"]
    # the external reading is evidence only: an axis tilt is logged, a lost template is not re-searched
    assert (row.get("external_tilt_evidence_deg") or 0.0) > O.O4_TILT_MAX_DEG or \
        (row["template_lost"] and row["reacquired_tilt_deg"] is None)
    obs2, start2 = _upright_with_wrist(25.0)
    rep2 = obs2.update(start2, None, OT.make_view(OT.state_with(OT.R_T2, 14.0), external=OT.bar_image(0.0)),
                       {"primitive": "held_reach"})
    assert (rep2.verdict, rep2.evidence["cause"]) == ("violated", "in_hand_tilt")


def test_h3_upright_without_a_wrist_estimate_the_external_axis_decides():
    obs, start = _upright_with_wrist(None)
    rep = obs.update(start, None, OT.make_view(OT.state_with(OT.R_T2, 14.0), external=OT.bar_image(32.0)),
                     {"primitive": "held_reach"})
    assert rep.verdict == "violated" and rep.evidence["tilt_source"] in ("external_axis", "external_template_rotation")
