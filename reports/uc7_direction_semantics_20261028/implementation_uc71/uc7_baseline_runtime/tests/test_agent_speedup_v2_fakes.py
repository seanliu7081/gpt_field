"""BF acceptance tests for the agent_speedup_v2 T1 fakes (contract §C15 items 1-8).

Everything here is test-side: the fakes expose evaluator truth through explicit accessors only, and these tests check
that the fake world is physically consistent with the ROBOT facts (robot_facts_v2.json), with the phase-A task-2
geometry and with the recorded probes (SP fits the hinge; CP is held out and checked non-circularly)."""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pytest

import agent_speedup_fakes as v1f
import agent_speedup_v2_fakes as F
from gpt_field import agent_speedup_v2_robot as robot
from gpt_field.contact_skills import (AnchoredMove, GripperCalibration, GripperSettlePrimitive, HoldPrimitive,
                                      StepView)
from gpt_field.external_axes import hand_image_motion


def world(scene=None, **kw):
    scene = scene or F.task2_like_scene()
    kw.setdefault("eef", F.TASK2_START_STATE8[:3])
    kw.setdefault("start_tilt_deg", 3.14)
    return F.FakeTwoViewSim(scene, **kw)


def run_primitive(w, primitive, max_steps=400):
    view = StepView(w.observation())
    n = 0
    while not primitive.done and n < max_steps:
        action = primitive.next_action(view)
        after = StepView(w.step(action))
        primitive.observe_transition(view, action, after)
        view, n = after, n + 1
    return primitive.outcome()


# ------------------------------------------------------------------------------------------------ 0. data + API
def test_data_files_are_verbatim_copies_with_manifest():
    manifest = json.loads((F.DATA_DIR / "MANIFEST.json").read_text())
    for rel, entry in manifest["files"].items():
        digest = hashlib.sha256((F.DATA_DIR / rel).read_bytes()).hexdigest()
        assert digest == entry["sha256"], rel
        if entry["verbatim"]:
            assert entry["source_sha256"] == digest, rel
    for name in ("sp_knob_sense_probe", "cp_knob_twist_probe", "cp_pot_lower_probe"):
        assert manifest["files"][f"{name}.json"]["verbatim"] is True
        assert F.load_probe(name)
    sp = F.load_probe("sp_knob_sense_probe")
    assert set(sp) == {"order_+1-1", "order_-1+1", "empty_closed_yaw"}
    records = F.load_probe("effect_probe_t2_records")
    index = records["frame_tags_to_record_index"]
    q = {tag: records["records"][i]["knob_q"] for tag, i in index.items()}
    assert q["pre_cross"] < 0.5 <= q["cross"] < q["cross_plus1"]
    region = F.effect_probe_region_mask()
    assert region.shape == (512, 512) and abs(int(region.sum()) - 10285) <= 2
    frame = F.effect_probe_frame("cross_plus1")
    assert frame.shape == (512, 512, 3) and frame.dtype == np.uint8
    assert F.survey_frame("t2_wrist").shape == (512, 512, 3)


def test_fakev2sim_compatibility_and_reset():
    w = world()
    sim = F.FakeV2Sim(prompt=w.prompt, wrist=w)
    obs0 = sim.reset(2, 0)
    assert obs0.prompt == F.TASK2_PROMPT and len(obs0.state) == 8
    obs, success = sim.step([0.2, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    assert success is False and obs.state[0] > obs0.state[0]
    view = StepView(obs)
    assert view.wrist_rgb_upright.shape == (512, 512, 3) and view.external_rgb_upright.shape == (512, 512, 3)
    np.testing.assert_array_equal(view.wrist_rgb_upright, w.wrist_upright())     # legacy orientation round trip
    w.reset()
    np.testing.assert_allclose(w.state, F.FakeTwoViewSim(F.task2_like_scene(), eef=F.TASK2_START_STATE8[:3],
                                                         start_tilt_deg=3.14).state)
    assert w.steps == 0 and w.held() is None


def test_start_pose_reproduces_task2_wrist_camera():
    w = world()
    R = w.R
    np.testing.assert_allclose(R[:, 2], (-0.0548, 0.0, -0.9985), atol=2e-3)           # 3.14 deg start tilt
    np.testing.assert_allclose(R[:, 1], (0.0, -1.0, 0.0), atol=2e-3)                  # jaw R1 = -y
    cam = w.wrist_camera()
    np.testing.assert_allclose(cam.position, (-0.1461, 0.019, 1.2834), atol=1.5e-3)  # t2_dump robot0_eye_in_hand
    ext = w.external
    assert abs(ext.pitch_deg - 38.9) < 0.2 and abs(ext.f_px - 618.0) < 1.0
    np.testing.assert_allclose(ext.right, (0.0, 1.0, 0.0), atol=1e-9)                 # agentview right = +y


# ------------------------------------------------------------------------------------------------ 1. rendering
def test_wrist_projection_equals_fakewristsim():
    rng = np.random.default_rng(3)
    for _ in range(50):
        eef = rng.uniform([-0.3, -0.1, 1.0], [0.0, 0.2, 1.25])
        rotvec = rng.normal(0, 0.3, 3) + [math.pi, 0.0, 0.0]
        w = F.FakeTwoViewSim(F.planar_region_scene(), eef=eef, rotvec=rotvec, f_wrist=v1f.FOCAL_PX, images=False)
        v1 = v1f.FakeWristSim(eef=eef, rotvec=rotvec)
        for point in rng.uniform([-0.4, -0.2, 0.9], [0.1, 0.3, 0.95], (5, 3)):
            uv_a, z_a = w.project("wrist", point)
            uv_b, z_b = v1.project(point)
            assert abs(z_a - z_b) < 1e-9
            if uv_b is not None:
                np.testing.assert_allclose(uv_a, uv_b, atol=1e-6)


def test_render_matches_projection_and_ids():
    w = world(F.planar_region_scene())
    mask = w.truth_mask("wrist", "region")
    corners = np.array([[x, y, 0.9305] for x, y in F.planar_region_scene().info["truth"]["polygon_xy"]])
    uv, _ = w.project("wrist", corners)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    rows, cols = np.nonzero(mask)
    assert abs(cols.min() - lo[0]) <= 1.5 and abs(cols.max() - hi[0]) <= 1.5
    assert abs(rows.min() - lo[1]) <= 1.5 and abs(rows.max() - hi[1]) <= 1.5
    depth = w.depth_map("wrist")
    r, c = int(np.median(rows)), int(np.median(cols))
    centre = w.wrist_camera()
    assert abs(float(depth[r, c]) - (centre.position[2] - 0.9305) / abs(w.R[2, 2])) < 0.01


def test_texture_moves_rigidly_with_the_view():
    """World-anchored textures: after a hand move, the image sampled at the re-projected world points equals the image
    before at the old projections (residual << the texture std); a stale image does not."""
    import cv2
    w = world(F.planar_region_scene(), pixel_noise_grey=0.0)
    g0 = w.wrist_upright().astype(np.float32).mean(axis=2)
    xs, ys = np.meshgrid(np.linspace(-0.28, -0.13, 40), np.linspace(0.0, 0.16, 40))
    region = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 0.9305)]
    xs, ys = np.meshgrid(np.linspace(-0.12, -0.02, 30), np.linspace(-0.14, -0.04, 30))
    support = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 0.900)]
    uv0 = {k: w.project("wrist", P)[0] for k, P in (("region", region), ("support", support))}
    w.advance([0.2, 0.1, 0.05, 1e-9, 0.0, 0.08, -1.0])
    g1 = w.wrist_upright().astype(np.float32).mean(axis=2)

    def sample(img, uv):
        return cv2.remap(img, uv[:, 0].astype(np.float32).reshape(1, -1), uv[:, 1].astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR)[0]
    for name, P in (("region", region), ("support", support)):
        uv1 = w.project("wrist", P)[0]
        a, b, stale = sample(g0, uv0[name]), sample(g1, uv1), sample(g1, uv0[name])
        assert np.abs(a - b).mean() < 0.15 * a.std() < np.abs(a - stale).mean(), name


def test_start_view_texture_resembles_the_real_start_view():
    """D15 grid patches (36 px spacing, radius 10, std >= 4 grey) above the fingertip band: the fake start view and the
    real t2 start view (54/143 per the spec) pass a comparable share (reported; within 20 points)."""
    def share(gray):
        stds = [float(gray[v - 10:v + 11, u - 10:u + 11].std()) for v in range(18, 380, 36) for u in range(18, 494, 36)]
        return sum(s >= 4.0 for s in stds), len(stds)
    w = world()
    fake = share(w.wrist_upright().astype(float) @ [0.299, 0.587, 0.114])
    real = share(F.survey_frame("t2_wrist").astype(float) @ [0.299, 0.587, 0.114])
    print("D15 patches passing: fake", fake, "real", real)
    assert abs(fake[0] / fake[1] - real[0] / real[1]) <= 0.20
    # a uniform support patch has std ~3 grey: it fails D15 (the spec's 54/143 condition)
    gray = w.wrist_upright().astype(float) @ [0.299, 0.587, 0.114]
    uv, _ = w.project("wrist", [-0.12, 0.0, 0.900])
    u, v = int(uv[0]), int(uv[1])
    assert 1.5 <= gray[v - 10:v + 11, u - 10:u + 11].std() < 4.0


def test_fingertip_band_at_the_bottom_changes_with_the_gap():
    w = world()
    opened = w.wrist_upright().astype(int)
    F.scripted_gripper(w, 1.0, 30)
    closed = w.wrist_upright().astype(int)
    diff = np.abs(opened - closed).max(axis=2) > 12
    rows = np.nonzero(diff)[0]
    assert rows.size > 500 and rows.min() >= 330 and np.median(rows) > 400           # fingertips at the bottom
    finger = w.truth_mask("wrist", "robot")
    assert finger[480:, :].mean() > 0.5 and finger[:300, :].mean() == 0.0


def test_external_hand_motion_and_fingertip_extreme():
    w = world()
    before = w.external_upright()
    tip0 = w.grip + F.FINGERTIP_REACH_M * w.R[:, 2]
    for _ in range(8):
        w.step([0.0, 0.3, 0.0, 1e-9, 0.0, 0.0, -1.0])
    after = w.external_upright()
    shift = hand_image_motion(before, after)
    assert shift is not None
    J = w.external.jacobian(w.grip)
    moved = w.grip - F.TASK2_START_STATE8[:3]
    predicted = J @ moved
    assert np.linalg.norm(np.asarray(shift["image_shift_px"]) - predicted) <= 0.2 * np.linalg.norm(predicted) + 1.0
    robot_mask = w.truth_mask("external", "robot")
    v_hat = F._unit(J @ [0.0, 0.0, -1.0])
    rows, cols = np.nonzero(robot_mask)
    extreme = np.array([cols, rows]).T[np.argmax(np.c_[cols, rows] @ v_hat)]
    tip_uv, _ = w.project("external", w.grip + F.FINGERTIP_REACH_M * w.R[:, 2])
    assert abs((extreme - tip_uv) @ v_hat) <= 3.0                     # the extreme along v_hat is at fingertip depth
    R = w.R
    tips = [w.grip + R @ [0.0, sgn * (abs(q) + 0.007), F.FINGERTIP_REACH_M] for sgn, q in ((1, w.q1), (-1, w.q2))]
    assert min(np.linalg.norm(extreme - w.project("external", p)[0]) for p in tips) <= 6.0   # ... on a fingertip
    assert tip0[2] > 0.9


def test_render_budget_both_cameras():
    """Contract §C15 item 1: <= 50 ms per step for both 512-px renders. The best of three 15-frame medians is gated
    (a loaded shared machine inflates single medians; review BF-R12); every median is printed as the metric.
    AGENT_SPEEDUP_V2_RENDER_BUDGET_MS overrides the bound on a machine known to be slower."""
    import os
    budget = float(os.environ.get("AGENT_SPEEDUP_V2_RENDER_BUDGET_MS", "50"))
    w = world()
    medians = []
    for _ in range(3):
        times = []
        for k in range(15):
            w.advance([0.15, -0.1, -0.05, 1e-9, 0.0, 0.1, -1.0])
            t0 = time.perf_counter()
            w.wrist_upright()
            w.external_upright()
            times.append(time.perf_counter() - t0)
        medians.append(1000 * float(np.median(times)))
        w.reset()
    print("render both 512 px views: medians %s ms" % ", ".join("%.1f" % m for m in medians))
    assert min(medians) <= budget


# ------------------------------------------------------------------------------------------------ 2. contact
def _descend(w, steps=40, command=-0.2, gripper=-1.0):
    z, gaps, inward = [], [], []
    for _ in range(steps):
        before = w.state.copy()
        w.advance([0.0, 0.0, command, 1e-9, 0.0, 0.0, gripper])
        z.append(float(w.grip[2]))
        gaps.append(w.gap_mm)
        inward.append(((before[6] - w.state[6]) * 1000, (w.state[7] - before[7]) * 1000))
    return np.asarray(z), gaps, inward


def test_open_fingers_land_on_the_highest_surface_in_their_column():
    """Straddling columns over the knob disk land on the disk top: grip = 0.918 + t_f (9.25 mm, robot fact R5)."""
    w = world(F.rotary_part_scene(), eef=(-0.2008, 0.2074, 1.0), start_tilt_deg=0.0)
    z, gaps, inward = _descend(w)
    assert abs(z[-1] - (0.918 + F.FINGERTIP_REACH_M)) < 3e-4
    ratios = -np.diff(np.r_[1.0, z]) / 0.01
    k = int(np.argmax(ratios < 0.6 * 0.26))
    assert ratios[k - 1] > 0.2 and max(ratios[k:k + 2].mean(), 0) < 0.6 * 0.26     # collapse at the contact
    landing = next(e["step"] for e in w.contact_log if e["kind"] == "finger_landing")
    first = inward[landing]
    assert 0.7 - 1e-9 <= max(first) <= 2.0 + 1e-9                                     # first-landing deflection


def test_palm_contact_at_grip_plus_31mm():
    box = F.FakeBody("post", "aabb", (-0.20, 0.08, 0.900), (0.02, 0.02, 0.10), colour=(90, 90, 90))
    w = world(F.FakeScene([box]), eef=(-0.20, 0.08, 1.08), start_tilt_deg=0.0, images=False)
    z, _, _ = _descend(w, steps=60)
    assert abs(z[-1] - (1.0 - robot.PALM_UNDERSIDE_MM / 1000.0)) < 1.5e-3
    assert any(e["kind"] == "palm" for e in w.contact_log)


def test_retreat_under_an_overhang_stalls_and_held_side_contact_opens_the_pinch():
    shelf = F.FakeBody("shelf", "aabb", (-0.20, 0.08, 1.12), (0.30, 0.30, 0.02), colour=(120, 100, 80))
    w = world(F.FakeScene([shelf]), eef=(-0.20, 0.08, 0.98), start_tilt_deg=0.0, images=False)
    for _ in range(20):
        w.advance([0.0, 0.0, 0.2, 1e-9, 0.0, 0.0, -1.0])
    assert abs(w.grip[2] + F.GRIP_ALONG_R2_M - 1.12) < 1e-3                                # hand top meets the shelf
    assert any(e["kind"] == "hand_top" for e in w.contact_log)
    post = F.FakeBody("post", "aabb", (0.10, 0.04, 0.900), (0.04, 0.04, 0.20), colour=(90, 90, 90))
    scene = F.free_object_with_thin_part_scene()
    v = world(F.FakeScene(scene.bodies + [post], info=scene.info), images=False)
    F.scripted_grasp_thin_part(v)
    gap = v.gap_mm
    for _ in range(12):
        v.advance([0.3, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])                                   # push the pot into the post
    assert any(e["kind"] == "held_obstacle" for e in v.contact_log)
    assert v.gap_mm - gap >= 0.4


def test_heights_truth_ignores_the_robot():
    w = world()
    xy = np.array([F.TASK2_KNOB_CENTRE_XY, F.TASK2_BURNER_XY, (-0.20, 0.0), F.TASK2_MOKA_XY])
    h = w.heights(xy)
    np.testing.assert_allclose(h, [0.955, 0.9305, 0.900, 1.0331], atol=1e-6)
    w2 = world(eef=(-0.20, 0.0, 0.91))                                                  # the hand right above the xy
    assert abs(w2.heights([[-0.20, 0.0]])[0] - 0.900) < 1e-9


# ------------------------------------------------------------------------------------------------ 3. grasp
def test_empty_close_follows_the_recorded_robot_curve():
    w = world(images=False)
    gaps = []
    for _ in range(35):
        w.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])
        gaps.append(w.gap_mm)
    np.testing.assert_allclose(gaps[:30], F.CLOSE_CURVE_MM[1:], atol=0.01)
    assert 1.0 <= gaps[-1] <= 1.15


def test_close_on_parts_gives_width_plus_one_mm():
    w = world(F.rotary_part_scene(), eef=F.rotary_part_scene().info["start_eef"], images=False)
    F.scripted_grasp_rotary(w)
    assert w.held()["kind"] == "rotary"
    assert abs(w.gap_mm - (24.3 + 1.013)) < 0.05
    w2 = world(F.free_object_with_thin_part_scene(), images=False)
    F.scripted_grasp_thin_part(w2, lift_m=0.0)
    assert w2.held()["kind"] == "free" and abs(w2.gap_mm - (13.0 + 1.013)) < 0.05


def test_calibrate_style_close_classification_and_stuck_gripper():
    w = world(images=False)
    settle = GripperSettlePrimitive("close", StepView(w.observation()))
    run_primitive(w, settle)
    curve = settle.gap_curve_mm
    assert 77.4 - min(curve) >= 20.0 and curve[14] < 3.0
    stuck = world(images=False, gripper_stuck=True)
    s2 = GripperSettlePrimitive("close", StepView(stuck.observation()))
    run_primitive(stuck, s2)
    assert 77.4 - min(s2.gap_curve_mm) < 1.0


def test_free_object_attaches_with_offset_drag_and_tilt():
    w = world(images=False)
    F.scripted_grasp_thin_part(w)
    held = w.held()
    assert held["group"] == "moka" and held["lifted"]
    pot = w.body_pose("moka_body")["xyz"]
    np.testing.assert_allclose(pot[:2] - w.grip[:2], (0.0, 0.0795), atol=2e-3)      # CP pot - grip
    b_low = w.grip[2] - pot[2]
    assert abs(b_low - (1.020 - 0.900)) < 2e-3                                          # g_close - support
    F.scripted_goto(w, w.grip + [0.05, 0.0, 0.03], gripper=1.0)
    np.testing.assert_allclose(w.body_pose("moka_body")["xyz"][:2] - w.grip[:2], (0.0, 0.0795), atol=2e-3)
    dragged = world(images=False, close_drag_m=(0.0, 0.026))
    F.scripted_grasp_thin_part(dragged)
    shift = dragged.body_pose("moka_body")["xyz"][:2] - np.array(F.TASK2_MOKA_XY)
    assert abs(shift[1] - 0.026) < 1e-6
    tilted = world(images=False, held_tilt_deg=15.0)
    F.scripted_grasp_thin_part(tilted)
    assert abs(tilted.held()["tilt_deg"] - 15.0) < 0.5


def test_palm_on_a_free_overhang_tips_it():
    w = world(images=False)
    q = F._matrix_quat(F._rot_axis((0, 0, 1), math.pi / 2) @ w.R)
    F.scripted_goto(w, (0.030, -0.0865, 1.10), quat_xyzw=q)
    for _ in range(40):
        w.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, -1.0])
        if w.body_pose("moka_body")["tilt_deg"] > 3.0:
            break
    assert w.body_pose("moka_body")["tilt_deg"] > 3.0
    assert any(e["kind"] == "palm_tip" for e in w.contact_log)


# ------------------------------------------------------------------------------------------------ 4. held lowering
def _carry_over_burner(w):
    F.scripted_grasp_thin_part(w)
    off = w.body_pose("moka_body")["xyz"][:2] - w.grip[:2]
    F.scripted_goto(w, [w.grip[0], w.grip[1], 1.16], gripper=1.0)
    F.scripted_goto(w, np.r_[np.asarray(F.TASK2_BURNER_XY) - off, 1.16], gripper=1.0, max_steps=300)


def test_held_lowering_stalls_within_two_steps_and_the_gap_rises():
    w = world(images=False)
    _carry_over_burner(w)
    rows = []
    for k in range(60):
        z, g = w.grip[2], w.gap_mm
        w.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
        rows.append(((z - w.grip[2]) / 0.01, w.gap_mm - g, w.explain(["on", "moka", "burner_plate"])["raw"]))
    ratio = np.array([r[0] for r in rows])
    contact = next(i for i, e in enumerate(w.contact_log[-60:]) if e["kind"] == "held_support")
    k = next(i for i, r in enumerate(rows) if r[2])                                    # On turns true ...
    first_low = next(i for i in range(1, 60) if ratio[i - 1:i + 1].mean() < 0.6 * 0.26)   # ... stall fires after
    assert k <= first_low and first_low - k <= 2
    rises = [r[1] for r in rows[k + 1:k + 8]]
    assert all(0.06 - 1e-9 <= r <= 0.12 + 1e-9 for r in rises)
    assert abs(w._bottom_points("moka")[:, 2].min() - 0.9305) < 5e-4
    assert contact >= 0


# ------------------------------------------------------------------------------------------------ 5. hinge wedge model
def _rates(log, start_gap):
    gaps = np.r_[start_gap, [r["gap_mm"] for r in log]]
    return np.diff(gaps)


def _fresh_knob(free_sense=-1, lever=True, **scene_kw):
    scene = F.rotary_part_scene(free_sense=free_sense, lever=lever, **scene_kw)
    w = world(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False, gap_noise_mm=0.0)
    F.scripted_grasp_rotary(w)
    F.scripted_gripper(w, 1.0, 8)                                                       # static closed hold (SP)
    return w


def test_wedge_blocked_free_and_reversal_shapes_match_sp():
    # contract sense +1 (about R2 ~ -z) is the blocked sense with free_sense -1 (task 2)
    w = _fresh_knob()
    g0 = w.gap_mm
    blocked = F.scripted_twist(w, +1, 12)
    r = _rates(blocked, g0)
    # sustained 0.34-0.51 (SP); the fake's first rates sit ~0.05 below SP (0.29-0.33) and rise monotonically to
    # ~0.46 (documented deviation, module header; review BF-R11), the K1w blocked verdict holds regardless
    assert 0.30 <= r[3:].min() and r[3:].max() <= 0.51 and r[4:].mean() >= 0.34
    assert abs(w.joint("knob") - (-0.005)) < 1e-9
    reverse = F.scripted_twist(w, -1, 30)
    rr = _rates(reverse, blocked[-1]["gap_mm"])
    falls = int(np.argmax(rr[1:] >= 0)) + 1                                             # unwedge length
    blocked_rise = blocked[-1]["gap_mm"] - g0
    assert 2 <= falls <= 16 and abs(falls - blocked_rise / 0.33) <= 3.0               # SP: 13 after +4.2 mm
    backlash = rr[falls:falls + 10]
    assert backlash.max() <= 0.62 and backlash[0] > 0
    assert np.abs(rr[falls + 12:]).max() <= 0.02                                         # then flat
    # free sense from a fresh close: the rate decays 0.29 -> ~0 over 12 steps
    w2 = _fresh_knob()
    g0 = w2.gap_mm
    free = F.scripted_twist(w2, -1, 12)
    rf = _rates(free, g0)
    assert 0.2 <= rf[1:3].max() <= 0.33 and rf[-2:].max() <= 0.02
    assert 0.10 < w2.joint("knob") < 0.17                                               # SP: q 0.134 in 12 steps


def test_end_stop_after_free_motion_jumps_the_rate():
    w = _fresh_knob()
    g0 = w.gap_mm
    F.scripted_twist(w, -1, 12)                       # free, q > 0
    log = F.scripted_twist(w, +1, 32)                 # reversal, free toward the lower limit, then the stop
    q = np.array([r["q"] for r in log])
    k_limit = int(np.argmax(q <= -0.0049))
    rates = _rates(log, log[0]["gap_mm"])
    after = rates[k_limit:k_limit + 3]
    assert after.max() >= 0.18 and after[1:].min() >= 0.3                                # SP: 0.18, 0.49, 0.44
    assert g0 > 0


def test_hand_yaw_rates_held_and_free_air():
    w = _fresh_knob()
    log = F.scripted_twist(w, -1, 12)
    rates = [abs(r["dyaw_deg"]) for r in log[3:]]
    assert 1.1 <= min(rates) and max(rates) <= 1.25
    air, still = world(images=False, gap_noise_mm=0.0), world(images=False, gap_noise_mm=0.0)
    F.scripted_gripper(air, 1.0, 25)
    F.scripted_gripper(still, 1.0, 25)
    log = F.scripted_twist(air, +1, 10)
    F.scripted_gripper(still, 1.0, 10)
    assert all(abs(abs(r["dyaw_deg"]) - 1.313) < 0.02 for r in log[4:])        # SP: 1.31 deg/step closed, empty
    assert abs(air.gap_mm - still.gap_mm) < 0.01                                 # the empty hand does not wedge


def test_colour_region_changes_about_twenty_percent_when_turned_on():
    import cv2
    scene = F.task2_like_scene()
    w = world(scene)
    F.scripted_grasp_rotary(w)
    region = w.truth_mask("external", "burner_plate")
    lab = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    base = lab(w.external_upright())
    for _ in range(80):
        if w.joint("knob") >= 0.5:
            break
        F.scripted_twist(w, -1, 1, close_first=False)
        assert not w.colour_region_changed("burner_plate") or w.joint("knob") >= 0.5
    onset = w.effect_onset("burner_plate")
    assert onset["on_step"] == onset["q_cross_step"] + 1                               # one-step LIBERO lag
    F.scripted_twist(w, -1, 2, close_first=False)
    assert w.colour_region_changed("burner_plate")
    now = lab(w.external_upright())
    hand = w.truth_mask("external", "robot")
    valid = region & ~hand
    chroma = np.hypot(*(now - base)[..., 1:].transpose(2, 0, 1)) > 12
    share = chroma[valid].mean()
    assert 0.12 <= share <= 0.28
    lum_only = (np.abs((now - base)[..., 0]) > 12) & ~chroma
    assert lum_only[valid].mean() >= 0.0


def test_arm_reflection_is_luminance_noise_near_the_region():
    """While the hand moves near the effect region its metal top changes luminance (|dL| > 12 on many pixels) but
    almost no chroma (the M0 probe's reason for the chroma statistic); a static hand changes nothing."""
    import cv2
    w = world()
    region = w.truth_mask("external", "burner_plate")
    lab = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    F.scripted_goto(w, (-0.08, 0.17, 1.00), render=False)
    w.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    ref = lab(w.external_upright())
    w.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    static = lab(w.external_upright())
    valid = region & ~w.truth_mask("external", "robot")
    assert (np.abs(static - ref).max(axis=2)[valid] > 12).mean() == 0.0
    lum, chroma = [], []
    for _ in range(12):
        w.advance([0.2, 0.25, 0.0, 1e-9, 0.0, 0.2, -1.0])
        now = lab(w.external_upright())
        valid = region & ~cv2.dilate(w.truth_mask("external", "robot").astype(np.uint8), np.ones((7, 7))).astype(bool)
        d = now - ref
        lum.append((np.abs(d[..., 0])[valid] > 12).mean())
        chroma.append((np.hypot(d[..., 1], d[..., 2])[valid] > 12).mean())
    assert max(lum) >= 0.10 and max(chroma) < 0.01


def test_round_part_slips_without_wedge_and_does_not_rotate():
    w = _fresh_knob(lever=False)
    g0 = w.gap_mm
    assert abs(g0 - (25.0 + 1.013)) < 0.05
    log = F.scripted_twist(w, +1, 12)                                                    # blocked sense
    assert np.abs(_rates(log, g0)).max() < 0.01                                          # no wedge
    assert abs(w.joint("knob") - (-0.005)) < 1e-9                                         # the part does not turn
    yaw = sum(r["dyaw_deg"] for r in log)
    assert abs(yaw) > 10.0                                                                # while the hand yaws


def test_dark_rotary_part_is_uniform_with_features_on_occluding_edges():
    scene = F.dark_rotary_part_scene()
    w = world(scene, eef=scene.info["hover_eef"]["knob"], start_tilt_deg=0.0)
    gray = w.wrist_upright().astype(float) @ [0.299, 0.587, 0.114]
    disk = w.truth_mask("wrist", "knob_disk")
    lever = w.truth_mask("wrist", "knob_lever")
    import cv2
    inner = cv2.erode(disk.astype(np.uint8), np.ones((15, 15))).astype(bool)
    assert gray[inner].std() <= 3.0 and gray[cv2.erode(lever.astype(np.uint8), np.ones((9, 9))).astype(bool)].std() <= 3.0
    part = (disk | lever).astype(np.uint8)
    edge = part.astype(bool) & ~cv2.erode(part, np.ones((3, 3))).astype(bool)          # the outer occluding edge
    grad = np.hypot(*np.gradient(gray))
    assert np.median(grad[edge]) > 10.0 > np.median(grad[inner])


def test_cp_knob_twist_envelope_non_circular():
    """Replay the commanded twists of CP probe_minus (world -z = contract +1, blocked), probe_plus (world +z, the
    unwedge) and the start of twist_cap0.1 on the SP-fitted fake (CP was not used for the fit): every gap rate falls
    inside the CP envelope of its phase, widened for +-0.02 mm noise on each CP gap reading (+-0.04 mm/step); also
    with +-0.02 mm uniform gap noise added to the fake (10 seeds, the envelope widened by that noise too). The
    transition steps between the phases are checked step by step with explicit tolerances, and the unwedge is one
    step longer than CP's (a documented deviation: tests on the fake must not assume SP's k0)."""
    cp = F.load_probe("cp_knob_twist_probe")
    w = _fresh_knob()
    g0 = w.gap_mm
    minus = F.scripted_twist(w, -1, 10, world_z=True)                # CP sign -1 (world z) == contract +1
    plus = F.scripted_twist(w, +1, 10, world_z=True)
    cont = F.scripted_twist(w, +1, 20, world_z=True, close_first=False)
    gaps = np.r_[g0, [r["gap_mm"] for r in minus + plus + cont]]
    logs = cp["logs"]
    cp_gaps = np.r_[cp["post_close"]["gap_mm"], [r["gap_mm"] for r in logs["probe_minus"] + logs["probe_plus"]
                                                  + logs["twist_cap0.1"][:20]]]
    real = np.diff(cp_gaps)
    phases = {"blocked": slice(2, 10), "unwedge": slice(11, 19), "backlash": slice(21, 26), "flat": slice(32, 40)}
    rng = np.random.default_rng(0)
    for noise in [0.0] + [0.02] * 10:
        fake = np.diff(gaps + (rng.uniform(-noise, noise, gaps.size) if noise else 0.0))
        tol = 0.04 + 2 * noise                                      # +-0.02 mm per CP gap reading
        for name, sl in phases.items():
            lo, hi = real[sl].min() - tol, real[sl].max() + tol
            assert lo <= fake[sl].min() and fake[sl].max() <= hi, (name, noise, fake[sl], real[sl])
    fake = np.diff(gaps)
    # the transition steps the phases leave out (review BF-R9), each with an explicit tolerance: the twist onset (0, 1),
    # the reversal step (10: the fake's yaw lag still rises +0.07 where CP falls -0.06), the last unwedge step (19),
    # the backlash onset (21) and the decay into flat (26-31). Step 20 is the fake's documented one-step-late unwedge
    # (module header): CP has ended its fall (-0.04) while the fake still falls (-0.26).
    transitions = {0: 0.15, 1: 0.10, 10: 0.20, 19: 0.10, 21: 0.10, 26: 0.05, 27: 0.05, 28: 0.05, 29: 0.05, 30: 0.05,
                   31: 0.05}
    for i, tol in transitions.items():
        assert abs(fake[i] - real[i]) <= tol, (i, fake[i], real[i])
    assert fake[20] < -0.05 <= real[20]
    unwedge_fake = int(np.argmax(fake[11:] >= -0.05)) + 1
    unwedge_cp = int(np.argmax(real[11:] >= -0.05)) + 1
    assert unwedge_fake - unwedge_cp == 1                         # one step late (documented; SP: 14 against 13)


def test_spring_back_option_and_no_spring_by_default():
    w = _fresh_knob()
    F.scripted_twist(w, -1, 20)
    F.scripted_gripper(w, -1.0, 1)                     # released on the first open step
    q = w.joint("knob")
    F.scripted_gripper(w, -1.0, 15)
    assert w.held() is None and abs(w.joint("knob") - q) < 1e-9                          # no spring (task 2)
    s = _fresh_knob(spring=0.2)
    F.scripted_twist(s, -1, 20)
    F.scripted_gripper(s, -1.0, 1)
    q = s.joint("knob")
    F.scripted_gripper(s, -1.0, 15)
    assert s.joint("knob") < 0.5 * q


def test_injections_slip_tug_hold_lost():
    w = _fresh_knob()
    F.scripted_twist(w, -1, 14)
    w.inject("slip", steps=3)
    log = F.scripted_twist(w, -1, 3, close_first=False)
    rates = _rates(log, w.contact_log[-4]["gap_mm"])
    assert rates.min() < -0.1                                                             # K7 gap fall
    w.inject("tug", vector_m=(0.002, 0.0), steps=4)
    hinge = w.scene.groups["knob"]["hinge"]
    F.scripted_twist(w, -1, 4, close_first=False)
    st = w._held
    c = np.asarray(hinge.centre_xy)
    orbit = c + (F._rot_axis(hinge.axis, hinge.q - st["q_close"]) @ np.r_[st["p_close"][:2] - c, 0.0])[:2]
    assert np.linalg.norm(w.grip[:2] - orbit) > 0.004
    w.inject("hold_lost")
    w.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])
    assert w.held() is None and w.gap_mm < 25.0


def test_orbit_follow_by_compliance_off_axis():
    scene = F.rotary_part_scene(grip_offset_m=0.012)
    w = world(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False)
    F.scripted_grasp_rotary(w)
    start = w.grip.copy()
    F.scripted_twist(w, -1, 60, hold_xyz=start)
    hinge = w.scene.groups["knob"]["hinge"]
    st = w._held
    c = np.asarray(hinge.centre_xy)
    orbit = c + (F._rot_axis(hinge.axis, hinge.q - st["q_close"]) @ np.r_[st["p_close"][:2] - c, 0.0])[:2]
    assert hinge.q > 0.9 and np.linalg.norm(w.grip[:2] - orbit) < 0.003                  # CP: no tug at 1.2 rad
    assert np.linalg.norm(w.grip[:2] - start[:2]) > 0.006                                 # the hand orbited


# ------------------------------------------------------------------------------------------------ 7. second order
def test_second_order_plant_response_settle_and_stall():
    w = world(F.planar_region_scene(), dynamics="second_order", images=False)
    z0 = w.grip[0]
    ratios = []
    for _ in range(6):
        x = w.grip[0]
        w.advance([0.2, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
        ratios.append((w.grip[0] - x) / 0.01)
    np.testing.assert_allclose(ratios[:3], np.array([0.115, 0.209, 0.234]) * 0.26 / 0.256, atol=0.02)
    assert abs(ratios[-1] - 0.26) < 0.02
    hold = HoldPrimitive(StepView(w.observation()), steps=14, gripper="open", until_quiet=True)
    out = run_primitive(w, hold)
    assert out["evidence"]["quiet"] and out["evidence"]["steps"] <= 14
    move = AnchoredMove(StepView(w.observation()), axis=(0, 0, -1), travel_m=0.4, gripper="open",
                        stop_on_contact=True)
    out = run_primitive(w, move)
    assert out["reason"] in ("contact_stall", "contact_touch")
    assert abs(w.grip[2] - (0.9305 + F.FINGERTIP_REACH_M)) < 0.003 or abs(w.grip[2] - (0.900 + F.FINGERTIP_REACH_M)) < 0.003
    assert z0 < w.grip[0]


# ------------------------------------------------------------------------------------------------ 8. world sim + oracle
def test_oracle_answers_from_truth():
    w = world()
    oracle = F.FakeOracleAnswers(w)
    plan = oracle.answer("plan", {"kind": "plan", "quarter_turns": 2})
    cells = {p["id"]: (p["w"], p["x"]) for p in plan["parts"]}
    assert all(x > 0 for _, x in cells.values()) and all(w_ > 0 for w_, _ in cells.values())
    assert cells["A"] == (31, 30) and cells["B"] == (47, 39)          # = the spec §3.9 example's cells for A and B
    from gpt_field.mark_prompting import coarse_cell_center, unrotate_pixel
    for pid in "ABCDE":
        truth = oracle.truth_uv(pid, "wrist")
        centre = np.asarray(unrotate_pixel(coarse_cell_center(cells[pid][0], 512), 2, 512))
        if 0 <= truth[0] < 512 and 0 <= truth[1] < 512:
            assert np.max(np.abs(centre - truth)) <= 32.0 + 1e-6
        else:                                          # cut by the border (the pot): a cell where the part is SEEN
            body = oracle.points[pid][0]
            mask = w.truth_mask("wrist", body)
            u, v = int(centre[0]), int(centre[1])
            assert mask[max(0, v - 32):v + 32, max(0, u - 32):u + 32].any()
    truth = oracle.truth_uv("A", "wrist")
    meta = {"kind": "point_fine", "pid": "A", "camera": "wrist", "quarter_turns": 2, "crop_px": 128,
            "center_uv": list(truth + 20.0)}
    pid = oracle.answer("point_fine", meta)["point_id"]
    assert 1 <= pid <= 64
    from gpt_field.mark_prompting import rotate_pixel
    shown = np.asarray(rotate_pixel(truth, 2, 512))
    c = np.asarray(rotate_pixel(truth + 20.0, 2, 512))
    origin = np.clip(np.round(c - 64), 0, 512 - 128)
    row, col = divmod(pid - 1, 8)
    answered = origin + [(col + 0.5) * 16, (row + 0.5) * 16]
    assert np.max(np.abs(answered - shown)) <= 8.0 + 1e-6                                 # nearest crop point
    far = dict(meta, center_uv=list(truth + 200.0))
    assert oracle.answer("point_fine", far)["point_id"] == 0                               # outside the crop
    assert oracle.answer("repair", {"kind": "repair"})["action"] == "retry"
    assert oracle.answer("review", {"kind": "review"})["action"] == "hold_still"
    assert len(oracle.report()["answers"]) == 5


def _oracle_chain(sim):
    """Truth-driven task-2 chain on the world: turn the knob past q_on, grasp the loop, carry, lower until success."""
    w = sim.world
    F.scripted_grasp_rotary(w)
    for _ in range(80):
        if w.joint("knob") >= 0.62:
            break
        F.scripted_twist(w, -1, 1, close_first=False)
    F.scripted_gripper(w, -1.0, 15)
    F.scripted_goto(w, w.grip + [0.0, 0.0, 0.05])
    F.scripted_grasp_thin_part(w)
    off = w.body_pose("moka_body")["xyz"][:2] - w.grip[:2]
    F.scripted_goto(w, [w.grip[0], w.grip[1], 1.16], gripper=1.0)
    F.scripted_goto(w, np.r_[np.asarray(F.TASK2_BURNER_XY) - off, 1.16], gripper=1.0, max_steps=300)
    for k in range(80):
        obs, success = sim.step([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
        if success:
            contact = [e for e in w.contact_log if e["kind"] == "held_support"]
            assert len(contact) == 1 and contact[0]["step"] == w.steps - 1          # On at the touching step,
            return k                                                                  # before any stall verdict
    return None


def test_world_sim_success_is_the_goal_predicate_and_truth_defers_to_the_settle():
    from gpt_field.agent_speedup_v2_truth import physical_summary
    w = world(images=False)
    sim = F.FakeV2WorldSim(w, None)
    sim.reset(2, 0)
    k = _oracle_chain(sim)
    assert k is not None and sim.status["success"] and sim.status["terminated"] and sim.status["libero_success"]
    assert w.held() is not None                                                           # success before release
    terminal = sim.truth()
    settled = sim.settle_truth()
    failing = {a["atom"][0]: a["why_not"] for a in terminal["atoms"]}
    assert set(failing["on"]) <= {"not_held", "at_rest"} and "not_held" in failing["on"]
    assert settled["strict_conjunction"] and settled["check_success"]
    summary = physical_summary(terminal, settled, raw_libero_success=True)
    assert summary["verdict"] == "physical_success" and summary["v2_success"] is True
    assert sim.answer_key()["goal_atoms"] == [["turnon", "knob"], ["on", "moka", "burner_plate"]]
    assert w.held() is not None                                                           # the settle ran on a clone


# ------------------------------------------------------------------------------- reference K1w / K4w on the fake
def _k1w(rates, omega_rad, *, k0=1, sigma_g=0.002, decide=12):
    """Contract §C9 K1w rule (reference copy for the fake's acceptance; the observer itself is BO's)."""
    lw = robot.PAD_LENGTH_R0_MM * float(np.mean(omega_rad))
    k = k0 + decide - 1
    early = float(np.mean(rates[k0:k0 + 4]))
    late = lambda end: float(np.mean(rates[end - 3:end + 1]))
    if late(k) <= max(0.1 * lw, 4 * sigma_g, 0.25 * early):
        return "free"
    if min(late(k), late(k - 1)) >= max(0.5 * lw, 0.75 * early):
        return "blocked"
    return "pending"


def test_reference_k1w_k4w_decisions_on_the_fake():
    """The fake supports the contract's T1 expectations: free at step 12, blocked at step 12, after a reversal
    (blocked_steps 12) the unwedge ends within ceil(1.5 x 12) and the free decision follows 12 steps later, and an end
    stop driven after a free classification fires K4w within 5 steps of the limit."""
    w = _fresh_knob()
    g0 = w.gap_mm
    free = F.scripted_twist(w, -1, 12)
    rates = _rates(free, g0)
    omega = [abs(math.radians(r["dyaw_deg"])) for r in free]
    assert _k1w(rates, omega) == "free"
    b = _fresh_knob()
    g0 = b.gap_mm
    blocked = F.scripted_twist(b, +1, 12)
    assert _k1w(_rates(blocked, g0), [abs(math.radians(r["dyaw_deg"])) for r in blocked]) == "blocked"
    rev = F.scripted_twist(b, -1, 30)
    rr = _rates(rev, blocked[-1]["gap_mm"])
    k0 = int(np.argmax(rr[1:] >= 0)) + 1
    assert k0 <= math.ceil(1.5 * 12)
    omega = [abs(math.radians(r["dyaw_deg"])) for r in rev]
    assert _k1w(rr, omega, k0=k0) == "free"
    # K4w: HingeTwist from the free classification (reversal toward the lower limit) into the end stop
    e = _fresh_knob()
    F.scripted_twist(e, -1, 12)
    log = F.scripted_twist(e, +1, 34)
    q = np.array([r["q"] for r in log])
    limit = int(np.argmax(q <= -0.0049))
    r = _rates(log, log[0]["gap_mm"])
    start = 15                                                      # end-stop mode armed after the free decision
    lw = robot.PAD_LENGTH_R0_MM * float(np.mean([abs(math.radians(x["dyaw_deg"])) for x in log[start:]]))
    fired = None
    for k in range(start + 4, len(r)):
        windows = [float(np.mean(r[j - 3:j + 1])) for j in (k - 1, k)]
        threshold = max(0.5 * lw, 5.0 * float(np.median(np.abs(r[start:k + 1]))))
        if min(windows) >= threshold:
            fired = k
            break
    assert fired is not None and 0 <= fired - limit <= 5


# ------------------------------------------------------------------- review BF-R1..R13: contact, release, coverage
def _rel_offset(w, r0_m, r1_m):
    """World xy of a point offset (R0, R1) from the grip in the hand's horizontal plane."""
    R = w.R
    return w.grip[:2] + r0_m * R[:2, 0] + r1_m * R[:2, 1]


@pytest.mark.parametrize("dynamics", ["kinematic", "second_order"])
def test_open_fingers_land_on_bare_support(dynamics):
    """BF-R1: the support plane is a collider. Open fingers over bare support stop at support_z + t_f in an empty scene
    and beside the task-2 bodies, as heights() reports the support there."""
    for scene, xy in ((F.FakeScene([]), (-0.20, 0.0)), (F.task2_like_scene(), (-0.12, 0.0))):
        w = world(scene, eef=(*xy, 1.0), start_tilt_deg=0.0, images=False, dynamics=dynamics)
        assert abs(w.heights([xy])[0] - 0.900) < 1e-12
        _descend(w, steps=40, command=-0.3)
        assert abs(w.grip[2] - (0.900 + F.FINGERTIP_REACH_M)) < 1e-4
        landing = [e for e in w.contact_log if e["kind"] == "finger_landing"]
        assert landing and landing[0]["bodies"] == ["support"]
    # beside the planar region: the outer finger lands on the support, never below it
    w = world(F.planar_region_scene(), eef=(-0.20, -0.06, 1.0), start_tilt_deg=0.0, images=False, dynamics=dynamics)
    _descend(w, steps=60, command=-0.3)
    tips = [w.grip + w.R @ [0.0, sgn * (abs(q) + 0.007), F.FINGERTIP_REACH_M] for sgn, q in ((1, w.q1), (-1, w.q2))]
    assert min(t[2] for t in tips) >= 0.900 - 1e-6


def test_closed_fingers_still_land_first_on_bare_support():
    """BF-R1 (the reviewer's 'palm at support + 31 mm with closed fingers' is not physical: closed fingers still reach
    t_f = 9.25 mm beyond the grip, robot fact R5, so they meet the support first and the palm never does)."""
    w = world(F.FakeScene([]), eef=(-0.20, 0.0, 1.0), start_tilt_deg=0.0, images=False)
    F.scripted_gripper(w, 1.0, 30)
    _descend(w, steps=40, command=-0.3, gripper=1.0)
    assert abs(w.grip[2] - (0.900 + F.FINGERTIP_REACH_M)) < 1e-4
    assert not any(e["kind"] == "palm" for e in w.contact_log)


def test_held_object_stalls_on_bare_support():
    """BF-R1: a held object lowered over bare support stalls on it (held_support), its lowest point at the support."""
    w = world(F.free_object_with_thin_part_scene(destination=False), images=False)
    F.scripted_grasp_thin_part(w)
    F.scripted_goto(w, [w.grip[0] - 0.12, w.grip[1] + 0.05, w.grip[2] + 0.02], gripper=1.0)
    for _ in range(40):
        w.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
    stall = [e for e in w.contact_log if e["kind"] == "held_support"]
    assert stall and stall[0]["bodies"] == ["support"]
    low = w._bottom_points("moka")[:, 2].min()
    assert abs(low - 0.900) < 5e-4 and low >= 0.900 - 1e-6
    xy = w.body_pose("moka_body")["xyz"][:2]
    assert abs(w.heights([xy], exclude_groups=("moka",))[0] - 0.900) < 1e-12


def test_palm_centred_on_a_free_body_blocks_without_tilt():
    """BF-R2: a palm pressing a free body's top INSIDE its footprint is blocked like a fixed one (palm at top + 31 mm,
    no tilt); only an overhang tips (test_palm_on_a_free_overhang_tips_it), and never beyond TIP_MAX_DEG."""
    crate = F.FakeBody("crate", "aabb", (-0.20, 0.08, 0.900), (0.025, 0.025, 0.08), role="free")
    w = world(F.FakeScene([crate]), eef=(-0.20, 0.08, 1.10), start_tilt_deg=0.0, images=False)
    _descend(w, steps=80)
    assert abs(w.grip[2] - (0.98 - robot.PALM_UNDERSIDE_MM / 1000.0)) < 2e-4
    assert w.body_pose("crate")["tilt_deg"] < 1e-9
    assert any(e["kind"] == "palm" for e in w.contact_log) and not any(e["kind"] == "palm_tip" for e in w.contact_log)
    # pressing on the loop overhang: the pot tips, the hand only follows what the pot yields, the tilt saturates
    v = world(images=False)
    q = F._matrix_quat(F._rot_axis((0, 0, 1), math.pi / 2) @ v.R)
    F.scripted_goto(v, (0.030, -0.0865, 1.10), quat_xyzw=q)
    z = []
    for _ in range(80):
        v.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, -1.0])
        z.append(float(v.grip[2]))
    tilt = v.body_pose("moka_body")["tilt_deg"]
    assert F.TIP_MAX_DEG - 0.5 <= tilt <= F.TIP_MAX_DEG + 1e-6
    assert abs(z[-1] - z[-5]) < 1e-6                                  # blocked once the tilt is used up
    tips = [e for e in v.contact_log if e["kind"] == "palm_tip"]
    assert tips and all(e["lever_m"] > 0 for e in tips)


def test_release_needs_the_pads_to_clear_the_part():
    """BF-R3: the hold ends only once the pads have opened past the clamped cross-section. With the gripper unable to
    open (open_limit_mm at the clamp width) the object stays held and moves with the hand: the release_incomplete
    world. A free gripper releases on its first open step."""
    w = world(images=False)
    F.scripted_grasp_thin_part(w, lift_m=0.0)
    clamp = w.gap_mm
    w.inject("open_limit_mm", mm=clamp)
    F.scripted_gripper(w, -1.0, 10)
    assert w.held() is not None and w.held()["group"] == "moka" and abs(w.gap_mm - clamp) < 1e-6
    pot0, grip0 = w.body_pose("moka_body")["xyz"].copy(), w.grip.copy()
    F.scripted_goto(w, w.grip + [0.0, 0.0, 0.03], gripper=-1.0, tol_m=0.001)
    np.testing.assert_allclose(w.body_pose("moka_body")["xyz"] - pot0, w.grip - grip0, atol=1e-6)
    assert not any(e["kind"] == "release" for e in w.contact_log)
    free = world(images=False)
    F.scripted_grasp_thin_part(free, lift_m=0.0)
    clamp = free.gap_mm
    F.scripted_gripper(free, -1.0, 1)
    assert free.held() is None and free.gap_mm > clamp + F.RELEASE_CLEAR_MM
    assert [e for e in free.contact_log if e["kind"] == "release"][-1]["reason"] == "opened"


def test_thin_posts_anywhere_under_the_palm_stop_the_hand():
    """BF-R4: robot samples <= 4 mm apart: a 6 x 6 mm post anywhere under the palm footprint (outside the finger
    columns) stops the hand with the palm on the post top (palm profile 31.04 mm for |xi| <= 92 mm)."""
    rng = np.random.default_rng(7)
    probe = world(F.FakeScene([]), eef=(-0.20, 0.08, 1.10), start_tilt_deg=0.0, images=False)
    tried = 0
    while tried < 10:
        r0, r1 = rng.uniform(-28.0, 28.0), rng.uniform(-88.0, 88.0)
        if abs(r0) < 6.14 + 4.0 and 34.0 <= abs(r1) <= 70.0:
            continue                                                    # under a finger: the finger lands first
        tried += 1
        xy = _rel_offset(probe, r0 / 1000.0, r1 / 1000.0)
        post = F.FakeBody("post", "aabb", (xy[0], xy[1], 0.900), (0.006, 0.006, 0.10))
        w = world(F.FakeScene([post]), eef=(-0.20, 0.08, 1.10), start_tilt_deg=0.0, images=False)
        _descend(w, steps=60)
        assert abs(w.grip[2] - (1.0 - 0.03104)) < 3e-4, (r0, r1, w.grip[2])
        assert any(e["kind"] == "palm" for e in w.contact_log)


def test_thin_ledge_meets_the_finger_side():
    """BF-R4: finger side walls sampled <= 4 mm apart in z: a 6 mm-thick ledge between the fingertip and the palm,
    which the old 7.85 mm rings straddled, stops a lateral move on the finger side."""
    ledge = F.FakeBody("ledge", "aabb", (-0.10, 0.02, 1.0085), (0.10, 0.20, 0.006))
    w = world(F.FakeScene([ledge]), eef=(-0.20, 0.02, 1.0), start_tilt_deg=0.0, images=False)
    for _ in range(40):
        w.advance([0.3, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    sides = [e for e in w.contact_log if e["kind"] == "finger_side"]
    assert sides and sides[0]["bodies"] == ["ledge"]
    assert abs(w.grip[0] + F.FINGER1_R0_MM[1] / 1000.0 - (-0.15)) < 1e-3   # the finger's +R0 face at the ledge edge


def test_two_sims_on_one_scene_are_independent_and_reset_undoes_injections():
    """BF-R5: hinge state and effect injections are sim-owned; reset() restores every injection."""
    scene = F.rotary_part_scene()
    a = world(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False)
    F.scripted_grasp_rotary(a)
    F.scripted_twist(a, -1, 12)
    q = a.joint("knob")
    assert q > 0.1
    b = world(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False)
    assert a.joint("knob") == q and b.joint("knob") == 0.0
    assert scene.groups["knob"]["hinge"].q == 0.0                       # the caller's scene is untouched
    a.inject("effect_never")
    a.inject("open_limit_mm", mm=20.0)
    assert a.scene.body("burner_plate").effect.never and not b.scene.body("burner_plate").effect.never
    a.reset()
    assert not a.scene.body("burner_plate").effect.never and a.open_limit_mm is None and a.joint("knob") == 0.0
    c = world(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False)
    assert not c.scene.body("burner_plate").effect.never


def test_neighbour_grasp_clamps_the_neighbour_with_no_co_motion():
    """BF-R7: a top-down grasp across the neighbour handle at oracle point N (jaw along x) clears the free object: it
    clamps 'other', and lifting moves the neighbour but never the pot (the grip_not_on_obj world)."""
    scene = F.free_object_with_thin_part_scene(neighbour=True)
    w = world(scene, images=False)
    n = np.asarray(scene.info["oracle_points"]["N"][1])
    pot0 = w.body_pose("moka_body")["xyz"].copy()
    q = F._matrix_quat(F._rot_axis((0, 0, 1), math.pi / 2) @ w.R)
    F.scripted_goto(w, scene.info["hover_eef"]["other_handle"], quat_xyzw=q, max_steps=300)
    F.scripted_goto(w, scene.info["grasp_eef"]["other_handle"], quat_xyzw=q)
    assert not any(e.get("bodies") and set(e["bodies"]) & {"moka_body", "moka_handle", "moka_spout"}
                   for e in w.contact_log)
    F.scripted_gripper(w, 1.0, 25)
    held = w.held()
    assert held is not None and held["group"] == "other" and abs(w.gap_mm - (20.0 + 1.013)) < 0.05
    other0 = w.body_pose("other_handle")["xyz"].copy()
    F.scripted_goto(w, w.grip + [0.0, 0.0, 0.03], gripper=1.0, quat_xyzw=q, tol_m=0.001)
    assert w.body_pose("other_handle")["xyz"][2] - other0[2] > 0.02
    np.testing.assert_allclose(w.body_pose("moka_body")["xyz"], pot0, atol=1e-9)
    assert np.linalg.norm(n[:2] - other0[:2]) < 1e-9


def test_held_press_creep_is_contact_checked_and_drops_off_the_part():
    """BF-R13: after the support stall the hand creeps down through the pinch, but the creep is contact-checked (a
    fingertip over an obstacle stops it) and the hold ends once the pads leave the held part."""
    post = F.FakeBody("post", "aabb", (0.40, 0.40, 0.900), (0.006, 0.006, 0.135), colour=(90, 90, 90))
    base = F.task2_like_scene()
    w = world(F.FakeScene(base.bodies + [post], info=base.info, goal=base.goal), images=False)
    _carry_over_burner(w)
    for _ in range(40):
        w.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
        if any(e["kind"] == "held_support" for e in w.contact_log):
            break
    assert w.held() is not None
    tip = _rel_offset(w, 0.0003, w.q1 + 0.007)
    top = w.grip[2] - F.FINGERTIP_REACH_M - 0.005                         # the post top 5 mm under the fingertip
    w.inject("move_group", group="post", vector_m=(tip[0] - 0.40, tip[1] - 0.40, top - 1.035))
    for _ in range(10):
        w.advance([0.0, 0.0, -1.0, 1e-9, 0.0, 0.0, 1.0])
    blocked = [e for e in w.contact_log if e["kind"] == "creep_blocked"]
    assert blocked and w.held() is not None
    over_post = (w.grip + w.R @ [0.0003, w.q1 + 0.007, F.FINGERTIP_REACH_M])[2]
    assert top - 1e-4 <= over_post <= top + 0.003                         # the fingertip rests on the post top
    # without the obstacle the creep goes on until the pads slide off the part, and then the hold is gone
    v = world(images=False)
    _carry_over_burner(v)
    released = None
    for k in range(200):
        v.advance([0.0, 0.0, -1.0, 1e-9, 0.0, 0.0, 1.0])
        assert v.grip[2] - F.FINGERTIP_REACH_M >= 0.9305 - 1e-6
        if v.held() is None:
            released = k
            break
    assert released is not None
    assert [e for e in v.contact_log if e["kind"] == "release"][-1]["reason"] == "pads_off_part"


def test_raised_obstacle_perches_the_object():
    """raised_obstacle_on_destination_scene: lowered over the destination centre the object rests on the obstacle
    top; after release On is false and supported_by_target fails (the perched / blocked_above world)."""
    scene = F.raised_obstacle_on_destination_scene()
    w = world(scene, images=False)
    F.scripted_grasp_thin_part(w)
    off = w.body_pose("moka_body")["xyz"][:2] - w.grip[:2]
    F.scripted_goto(w, [w.grip[0], w.grip[1], 1.20], gripper=1.0)
    F.scripted_goto(w, np.r_[np.asarray(F.TASK2_BURNER_XY) - off, 1.20], gripper=1.0, max_steps=300)
    for _ in range(80):
        w.advance([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, 1.0])
    assert [e["bodies"] for e in w.contact_log if e["kind"] == "held_support"][0] == ["obstacle"]
    F.scripted_gripper(w, -1.0, 10)
    assert w.held() is None
    top = scene.info["truth"]["obstacle_top_z"]
    assert abs(w._bottom_points("moka")[:, 2].min() - top) < 1e-3
    on = w.explain(["on", "moka", "burner_plate"])
    assert on["raw"] is False and "supported_by_target" in on["why_not"] and on["checks"]["com_over_region"]


def test_textureless_dark_box_and_uniform_patch_fail_d15():
    """D7a negative scene: the dark box top and the uniform patch stay below the D15 floor (std < 4 grey over 21 px),
    while the textured plane beside them passes."""
    w = world(F.textureless_dark_aabb_scene(), eef=(-0.20, 0.01, 1.15), start_tilt_deg=0.0)
    gray = w.wrist_upright().astype(float) @ [0.299, 0.587, 0.114]

    def std_at(xyz):
        uv, _ = w.project("wrist", xyz)
        u, v = int(round(uv[0])), int(round(uv[1]))
        return float(gray[v - 10:v + 11, u - 10:u + 11].std())
    assert std_at([-0.15, 0.04, 0.93]) < 4.0 and std_at([-0.24, 0.0, 0.90]) < 4.0
    assert std_at([-0.20, 0.12, 0.90]) >= 4.0 and std_at([-0.12, -0.06, 0.90]) >= 4.0
    assert w.heights([[-0.15, 0.04]])[0] == pytest.approx(0.93) and w.heights([[-0.24, 0.0]])[0] == 0.900


def test_injections_close_offset_drift_lateral_block_tilt_joint_group_effect():
    w = world(images=False)
    w.inject("close_offset_mm", mm=0.8)
    F.scripted_gripper(w, 1.0, 35)
    assert 1.7 <= w.gap_mm <= 1.95                                      # the empty close ends 0.8 mm off
    d = world(images=False)
    start = d.grip.copy()
    d.inject("drift", vector_m=(0.001, 0.0, 0.0), steps=5)
    for _ in range(8):
        d.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(d.grip - start, (0.005, 0.0, 0.0), atol=1e-9)
    b = world(images=False)
    start = b.grip.copy()
    b.inject("lateral_block", steps=4)
    for _ in range(4):
        b.advance([0.3, 0.3, 0.2, 1e-9, 0.0, 0.0, -1.0])
    assert np.allclose(b.grip[:2], start[:2]) and b.grip[2] > start[2]
    t = world(images=False)
    F.scripted_grasp_thin_part(t)
    t.inject("tilt", deg=15.0)
    assert abs(t.held()["tilt_deg"] - 15.0) < 0.5
    j = world(images=False)
    before = j.body_pose("knob_lever")["yaw_rad"]
    j.inject("set_joint", group="knob", q=0.3)
    assert j.joint("knob") == 0.3 and abs(abs(F._wrap(j.body_pose("knob_lever")["yaw_rad"] - before)) - 0.3) < 1e-9
    g = world(images=False)
    p0 = g.body_pose("frypan_pan")["xyz"].copy()
    g.inject("move_group", group="frypan", vector_m=(0.01, -0.02, 0.0))
    np.testing.assert_allclose(g.body_pose("frypan_pan")["xyz"] - p0, (0.01, -0.02, 0.0), atol=1e-12)
    e = world(images=False)
    e.inject("effect_never", name="burner_plate")
    e.inject("set_joint", group="knob", q=0.7)
    for _ in range(3):
        e.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    assert e.explain(["turnon", "knob"])["raw"] and not e.colour_region_changed("burner_plate")
    with pytest.raises(ValueError):
        e.inject("teleport")


def test_free_sense_plus_one_mirrors_minus_one_and_unwedged_stop():
    """free_sense=+1 mirrors -1 (contract +1 free, -1 blocked, same gap rates); wedge=False: an elongated part whose
    stop does not wedge gives a flat gap with the part stopped at its limit."""
    logs = {}
    for fs in (-1, +1):
        w = _fresh_knob(free_sense=fs)
        g0 = w.gap_mm
        free = F.scripted_twist(w, fs, 12)
        b = _fresh_knob(free_sense=fs)
        blocked = F.scripted_twist(b, -fs, 12)
        logs[fs] = (_rates(free, g0), w.joint("knob"), _rates(blocked, b.gap_mm - (blocked[-1]["gap_mm"] - g0)),
                    b.joint("knob"))
    np.testing.assert_allclose(logs[1][0], logs[-1][0], atol=0.01)
    assert abs(logs[1][1] - logs[-1][1]) < 0.005 and logs[1][1] > 0.1
    assert logs[1][3] == logs[-1][3] == -0.005
    flat = _fresh_knob(wedge=False)
    g0 = flat.gap_mm
    log = F.scripted_twist(flat, +1, 12)                                 # the blocked sense
    assert np.abs(_rates(log, g0)).max() < 0.01 and flat.joint("knob") == -0.005


def test_wrist_focal_option_scales_the_projection():
    a = world(F.planar_region_scene(), images=False)
    b = world(F.planar_region_scene(), images=False, f_wrist=300.0)
    p = [-0.18, 0.05, 0.9305]
    c = (F.IMAGE_PX - 1) / 2.0
    np.testing.assert_allclose(np.asarray(b.project("wrist", p)[0]) - c,
                               (np.asarray(a.project("wrist", p)[0]) - c) * 300.0 / robot.FOCAL_PX_AT_512, atol=1e-9)
    mask = b.truth_mask("wrist", "region")
    assert mask.sum() < a.truth_mask("wrist", "region").sum()           # a wider view shows the region smaller


def test_value_errors():
    w = world(images=False)
    for bad in ([0.0] * 6, [0.0] * 6 + [float("nan")], [2.0] + [0.0] * 6):
        with pytest.raises(ValueError):
            w.advance(bad)
    with pytest.raises(ValueError):
        F.FakeHinge(centre_xy=(0.0, 0.0), free_sense=0)
    with pytest.raises(ValueError):
        F.FakeTwoViewSim(F.FakeScene([]), dynamics="first_order")
    with pytest.raises(ValueError):
        F.FakeBody("x", "sphere", (0.0, 0.0, 0.9), (0.01, 0.01, 0.01))
    with pytest.raises(ValueError):
        F.FakeBody("x", "aabb", (0.0, 0.0, 0.9), (0.01, 0.01, 0.01), role="hinged")
    with pytest.raises(ValueError):
        F.FakeScene([F.FakeBody("x", "aabb", (0.0, 0.0, 0.9), (0.01, 0.01, 0.01), role="rotary")])
    with pytest.raises(ValueError):
        F.FakeScene([F.FakeBody("x", "aabb", (0.0, 0.0, 0.9), (0.01, 0.01, 0.01))] * 2)
    with pytest.raises(ValueError):
        F.FakeOracleAnswers(w).answer("guess", {"kind": "guess"})
    with pytest.raises(ValueError):
        w.explain(["inside", "moka", "burner_plate"])


# --------------------------------------------------------------------------- oracle parity with BH's stub (BF-R8)
def _bh_oracle():
    import importlib.util
    path = F.DATA_DIR.parents[2] / "scripts" / "agent_speedup_v2_oracle.py"
    spec = importlib.util.spec_from_file_location("bh_oracle_for_fake_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_rules_match_bh_stub():
    bh = _bh_oracle()
    assert (F.ORACLE_GRID, F.ORACLE_FINE_CROP_PX) == (bh.GRID, bh.FINE_CROP_PX)
    assert F.ORACLE_PLAN_MIN_VISIBLE_FRACTION == bh.PLAN_MIN_VISIBLE_FRACTION
    assert F.ORACLE_PLAN_BORDER_REACH_PX == bh.PLAN_BORDER_REACH_PX
    assert F.oracle_fine_point_mapping()[1] == bh.fine_point_mapping()[1]
    mapping = F.oracle_fine_point_mapping()[0]
    rng = np.random.default_rng(11)
    for uv in rng.uniform(-3.0, 515.0, (400, 2)):
        assert F.oracle_in_image(uv) == bh.in_image(uv)
        assert F.oracle_coarse_cell_of(uv) == bh.coarse_cell_of(uv)
        centre = rng.uniform(0, 512, 2)
        origin = F.oracle_crop_origin(centre)
        assert origin == bh.crop_origin(centre)
        assert F.oracle_in_crop(uv, origin, 128) == bh.in_crop(uv, origin, 128)
        if bh.in_crop(uv, origin, 128):
            assert (F.oracle_nearest_fine_point(uv, origin, 128, mapping)
                    == bh.nearest_fine_point(uv, origin, 128, bh.fine_point_mapping()[0]))


def test_oracle_border_reach_whole_parts_perturb_and_wrong_item():
    w = world()
    far = {"C": ("moka_handle", (0.030, -0.60, 1.03)), "D": ("moka_body", (0.030, -0.60, 1.03))}
    oracle = F.FakeOracleAnswers(w, points=far, whole=["D"])
    assert oracle.plan_cell("C", "wrist", 2) == (0, "truth_outside_image")      # visible, but beyond one cell
    cell, rule = oracle.plan_cell("D", "wrist", 2)
    assert cell > 0 and rule == "border_cut_visible"                             # a whole part: any visible pixel
    default = F.FakeOracleAnswers(w)
    assert default.plan_cell("C", "wrist", 2)[1] == "border_cut_visible"        # the start-view loop: within reach
    base = F.FakeOracleAnswers(w).answer("plan", {"kind": "plan", "quarter_turns": 2})
    shifted = F.FakeOracleAnswers(w, perturb_px=40.0, seed=1)
    assert shifted.answer("plan", {"kind": "plan", "quarter_turns": 2}) == base     # plan cells never perturbed
    meta = {"kind": "point_fine", "pid": "A", "camera": "wrist", "quarter_turns": 2,
            "center_uv": list(default.truth_uv("A", "wrist"))}
    shifted.answer("point_fine", meta)
    exact = default.shown_uv("A", "wrist", 2)
    assert abs(np.linalg.norm(np.asarray(shifted.log[-1]["truth_shown_uv"]) - exact) - 40.0) < 1e-9
    wrong = F.FakeOracleAnswers(w, wrong_item={"C": "D"})
    np.testing.assert_allclose(wrong.truth_xyz("C"), default.truth_xyz("D"))
    cyclic = F.FakeOracleAnswers(w, wrong_item="C")
    np.testing.assert_allclose(cyclic.truth_xyz("C"), default.truth_xyz("D"))   # BH's form: the next part
    rules = default.answer("plan", {"kind": "plan", "quarter_turns": 2}) and default.log[-1]["rules"]
    assert rules["A"]["w_rule"] == "truth_in_image"


def test_probe_matched_effect_ring_share():
    """BF-R10: the contract's ~20 % ring is the default; EFFECT_RING_PROBE_M reproduces the M0 probe's measured ~6.9 %
    red-shift share of the plate (a contract-vs-probe discrepancy reported to the lead)."""
    import cv2
    w = world(F.task2_like_scene(effect_ring_m=F.EFFECT_RING_PROBE_M), images=False)
    region = w.truth_mask("external", "burner_plate")
    lab = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    base = lab(w.external_upright())
    w.inject("set_joint", group="knob", q=0.7)
    for _ in range(3):
        w.advance([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
    assert w.colour_region_changed("burner_plate")
    now = lab(w.external_upright())
    valid = region & ~w.truth_mask("external", "robot")
    share = (np.hypot(*(now - base)[..., 1:].transpose(2, 0, 1)) > 12)[valid].mean()
    measured = F.load_probe("effect_probe_t2_records")["summary"]["agentview"]["plate"]["red_shift_fraction"]
    measured = measured["post_onset"]["mean"]                                     # 0.06877
    print("probe-matched ring share %.4f (probe %.4f)" % (share, measured))
    assert abs(share - measured) <= 0.01
