"""agent_SpeedUp_v1 wrist-image Align on the pinhole FakeWristSim; no simulator or model I/O.

The fake camera follows the ROBOT mount conventions exactly, so on it the image
Jacobian is -kappa P with kappa = FOCAL_PX / depth. FOCAL_PX, mark positions and
projections are ORACLE values used only to check results and to answer
pointing requests, never passed to the code under test.
"""

import copy
import json
import math

import numpy as np
import pytest

from agent_speedup_fakes import CENTER, FOCAL_PX, FakeWristSim
from gpt_field.action import _quaternion_product, _rotvec_quaternion
from gpt_field.contact_skills import AnchoredMove, StepView, quaternion_angle_rad
from gpt_field.observation import quaternion_to_axisangle
from gpt_field.wrist_servo import (
    GRIP_AXIS_OFFSET_M, P_MATRIX, WRIST_SERVO_VERSION, Align, ImageJacobian, NCCTrack, WristPatchMonitor,
    inplane_displacement, principal_point, setpoint)

MARK = (0.10, 0.0, 0.90)


def view_of(sim):
    return StepView(sim.observation())


def true_kappa(sim):
    cam, R = sim.camera()
    return FOCAL_PX / float((sim.marks[0] - cam) @ R[:, 2])


def oracle_uv(sim):
    uv, _ = sim.project(sim.marks[0])
    return uv


def cell_of(uv):
    return int(uv[1] // 64) * 8 + int(uv[0] // 64) + 1


def oracle_pointing(sim):
    def answer(request, view):
        uv = oracle_uv(sim)
        return cell_of(uv) if request["stage"] == "coarse" else uv
    return answer


def run(primitive, sim, view, *, pointing=None, limit=200, on_step=None):
    """Drive a primitive to completion, answering pointing requests (default: ORACLE projection)."""
    feedback, requests = [], []
    answer = pointing or oracle_pointing(sim)
    while not primitive.done:
        request = primitive.pointing_request
        if request is not None:
            requests.append(dict(request))
            with pytest.raises(RuntimeError, match="pointing"):
                primitive.next_action(view)
            if request["stage"] == "coarse":
                primitive.provide_coarse(answer(request, view))
            else:
                primitive.provide_point(answer(request, view), view)
            continue
        assert len(feedback) < limit
        action = primitive.next_action(view)
        assert len(action) == 7 and all(type(v) is float and abs(v) <= 1 for v in action)
        assert any(action[3:6]), "exactly-zero rotation would keep a stale OSC goal"
        after = StepView(sim.step(action))
        feedback.append(primitive.observe_transition(view, action, after))
        view = after
        if on_step is not None:
            on_step(primitive, sim, view)
    json.dumps(feedback, allow_nan=False)
    return view, feedback, requests


def outcome_of(primitive):
    outcome = primitive.outcome()
    json.dumps(outcome, allow_nan=False)
    assert set(outcome) == {"primitive", "status", "reason", "evidence", "actions"}
    assert outcome["actions"] == primitive.actions_executed
    return outcome


def tracked(sim, view=None, **kwargs):
    view = view or view_of(sim)
    return NCCTrack(view.wrist_gray_upright, oracle_uv(sim), **kwargs)


def hover_sim(offset_xy=(0.0, 0.02), height=0.10, **kwargs):
    """Grip site ``height`` above MARK, displaced laterally by ``offset_xy`` (world x, y)."""
    return FakeWristSim(eef=(MARK[0] + offset_xy[0], MARK[1] + offset_xy[1], MARK[2] + height),
                        marks=(MARK,), **kwargs)


def aligned_hover(**kwargs):
    sim = hover_sim(**kwargs)
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view))
    view, _, _ = run(align, sim, view)
    assert align.outcome()["reason"] == "aligned"
    return sim, view, align


# ------------------------------------------------------------ conventions
def test_constants_principal_point_and_setpoint_formula():
    assert WRIST_SERVO_VERSION == "agent-speedup-v1-wrist-servo-2"
    assert GRIP_AXIS_OFFSET_M == (0.0, -0.05)
    assert P_MATRIX.tolist() == [[1.0, 0.0], [0.0, -1.0]]
    assert principal_point(512, 512).tolist() == [255.5, 255.5]
    assert principal_point(640, 480).tolist() == [319.5, 239.5]
    assert setpoint(2000.0, GRIP_AXIS_OFFSET_M, 512, 512).tolist() == pytest.approx([255.5, 355.5])
    assert setpoint(2000.0, (0.01, 0.0), 512, 512).tolist() == pytest.approx([275.5, 255.5])
    assert setpoint(1000.0, (0.0, 0.02), 512, 512).tolist() == pytest.approx([255.5, 235.5])
    assert principal_point(512.0, 512.0).tolist() == [255.5, 255.5]
    for bad in ((0.0, GRIP_AXIS_OFFSET_M, 512, 512), (1.0, (0.0,), 512, 512), (1.0, GRIP_AXIS_OFFSET_M, 0, 512),
                (1.0, GRIP_AXIS_OFFSET_M, True, 512), (math.nan, GRIP_AXIS_OFFSET_M, 512, 512)):
        with pytest.raises(ValueError):
            setpoint(*bad)


@pytest.mark.parametrize("height", [0.05, 0.10, 0.15])
def test_depth_free_setpoint_is_where_a_mark_on_the_grip_axis_appears(height):
    sim = hover_sim(offset_xy=(0.0, 0.0), height=height)
    assert sim.grip_axis_lateral_error(MARK) == pytest.approx(0.0, abs=1e-12)
    assert oracle_uv(sim) == pytest.approx(setpoint(true_kappa(sim), GRIP_AXIS_OFFSET_M, 512, 512), abs=1e-9)
    # a 10 mm jaw-axis (R1) shift of the mark moves its image by +kappa*10 mm along u only
    sim.marks = [np.asarray(MARK) + 0.01 * sim.R[:, 1]]
    assert oracle_uv(sim) == pytest.approx(setpoint(true_kappa(sim), (0.01, -0.05), 512, 512), abs=1e-9)


def test_inplane_displacement_uses_the_frozen_frame():
    sim = hover_sim()
    before = view_of(sim)
    R = before.R.copy()
    after = StepView(sim.step([0.2, -0.4, 0.1, 1e-9, 0, 0, -1]))
    delta = after.position - before.position
    measured = inplane_displacement(before, after, R)
    assert measured.tolist() == pytest.approx([delta @ R[:, 1], delta @ R[:, 0]])
    assert measured.tolist() == pytest.approx([0.26 * 0.05 * 0.4, 0.26 * 0.05 * 0.2])  # R1 = -y, R0 = +x


# ---------------------------------------------------------------- NCC track
def smooth_image(shift=(0.0, 0.0), size=160):
    """Continuous sinusoidal texture sampled after an exact subpixel translation."""
    rng = np.random.default_rng(7)
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    xs, ys = xs - shift[0], ys - shift[1]
    image = np.full((size, size), 120.0)
    for _ in range(12):
        fx, fy = rng.uniform(-0.35, 0.35, 2)
        image += 12 * np.cos(fx * xs + fy * ys + rng.uniform(0, 2 * math.pi))
    return image


def test_ncc_track_subpixel_on_a_continuous_texture():
    start = np.array([70.3, 81.6])
    track = NCCTrack(smooth_image(), start)
    assert track.uv.tolist() == pytest.approx(start.tolist())
    position = start.copy()
    for shift in [(0.4, -0.3), (1.7, 0.6), (3.2, 2.9), (5.5, 4.1), (-2.3, 6.8)]:
        position = start + np.asarray(shift)
        uv, peak = track.match(smooth_image(shift))
        assert peak > 0.97
        assert float(np.linalg.norm(uv - position)) < 0.08
        assert track.uv.tolist() == uv.tolist()
    # a jump past the search window is a miss, not a wrong match
    uv, peak = track.match(smooth_image(position - start + (track.search + 2, 0.0)))
    assert uv is None and peak > 0.6  # best in-window score sits on the window edge
    uv, peak = track.match(np.full((160, 160), 120.0))
    assert uv is None and peak == -1.0
    assert float(np.linalg.norm(track.uv - position)) < 0.08


def test_ncc_track_follows_the_fake_mark_and_reports_loss():
    sim = hover_sim()
    view = view_of(sim)
    track = tracked(sim, view)
    for action in ([0.3, 0, 0, 1e-9, 0, 0, -1], [0, -0.4, 0, 1e-9, 0, 0, -1], [0.2, 0.2, 0, 1e-9, 0, 0, -1]):
        view = StepView(sim.step(action))
        uv, peak = track.match(view.wrist_gray_upright)
        assert peak >= track.min_peak == 0.6
        # the fake renders marks at rounded pixels and the template keeps the fractional pointed offset
        assert float(np.abs(uv - oracle_uv(sim)).max()) <= 1.0
    last = track.uv.copy()
    sim.occlude_rows = 0
    uv, peak = track.match(view_of(sim).wrist_gray_upright)
    assert uv is None and peak < 0.6 and track.uv.tolist() == last.tolist()
    assert track.misses == 1
    sim.occlude_rows = None
    uv, peak = track.match(view_of(sim).wrist_gray_upright)
    assert uv is not None and peak > 0.9


def test_ncc_track_rejects_bad_templates_and_keeps_border_points():
    sim = hover_sim()
    gray = view_of(sim).wrist_gray_upright
    with pytest.raises(ValueError, match="textureless_template"):
        NCCTrack(gray, (60.0, 60.0))
    with pytest.raises(ValueError, match="template_out_of_view"):
        NCCTrack(gray, (-3.0, 100.0))
    with pytest.raises(ValueError, match="template_out_of_view"):
        NCCTrack(gray, (100.0, 512.0))
    track = tracked(sim)
    template = track.template.copy()
    assert track.retemplate(gray, (60.0, 60.0)) is False
    assert track.template.tolist() == template.tolist() and track.uv.tolist() == pytest.approx(oracle_uv(sim).tolist())
    assert track.retemplate(gray) is True and track.retemplates == 1
    texture = smooth_image(size=200)
    edge = NCCTrack(texture, (3.25, 120.5))
    assert edge.uv.tolist() == [3.25, 120.5]
    uv, peak = edge.match(smooth_image((2.0, 1.0), size=200))
    assert peak > 0.95 and uv.tolist() == pytest.approx([5.25, 121.5], abs=0.15)
    json.dumps(edge.to_dict(), allow_nan=False)


def test_ncc_track_reacquires_a_rescaled_mark_after_a_descent():
    sim = hover_sim(offset_xy=(0.0, 0.0))
    track = tracked(sim)
    lowered = hover_sim(offset_xy=(0.0, 0.0), height=0.06)  # 40 mm lower: rendered mark radius 10 -> 12 px
    gray = view_of(lowered).wrist_gray_upright
    plain = copy.deepcopy(track)
    assert plain.match(gray)[0] is None
    uv, peak = track.match(gray, scales=NCCTrack.REACQUIRE_SCALES)
    assert peak >= 0.6 and track.last_scale > 1.0 and track.retemplates == 1
    assert float(np.linalg.norm(uv - oracle_uv(lowered))) < 1.5
    assert track.match(gray)[1] > 0.95  # re-extracted at the new scale


# ----------------------------------------------------------- image Jacobian
def test_image_jacobian_probes_solve_broyden_and_similarity():
    J_true = np.array([[-1500.0, 40.0], [25.0, 1480.0]])
    jacobian = ImageJacobian()
    assert not jacobian.ready and jacobian.broyden([1, 1], [0.002, 0.0]) is False
    with pytest.raises(RuntimeError):
        jacobian.kappa
    with pytest.raises(ValueError):
        jacobian.set_from_probes([(J_true @ [0.002, 0.0], [0.002, 0.0])])
    with pytest.raises(ValueError, match="ill_conditioned"):
        jacobian.set_from_probes([(J_true @ [0.002, 0.0], [0.002, 0.0]), (J_true @ [0.004, 1e-7], [0.004, 1e-7])])
    probes = [np.array(v) for v in ([0.002, 0.0], [0.003, 0.0], [0.003, 0.002], [0.003, 0.003])]
    jacobian.set_from_probes([(J_true @ dp, dp) for dp in probes])
    assert jacobian.ready and jacobian.J.ravel().tolist() == pytest.approx(J_true.ravel().tolist())
    assert jacobian.kappa == pytest.approx(float(np.mean(np.linalg.svd(J_true, compute_uv=False))))
    error = np.array([30.0, -12.0])
    dp = jacobian.solve(error)
    assert (jacobian.J @ dp).tolist() == pytest.approx((-error).tolist())

    model = ImageJacobian()
    model.set_from_probes([(-1400 * P_MATRIX @ dp, dp) for dp in probes])
    assert model.model_deviation() == pytest.approx(0.0, abs=1e-9)
    step = np.array([0.004, -0.001])
    assert model.broyden(J_true @ step, step) is True
    assert (model.J @ step).tolist() == pytest.approx((J_true @ step).tolist())
    before = model.J.copy()
    assert model.broyden(J_true @ [0.0003, 0.0], [0.0003, 0.0]) is False  # below min_step_m
    assert model.broyden([400.0, 0.0], [0.004, 0.0]) is False  # innovation: a tracking jump
    assert model.J.tolist() == before.tolist() and model.broyden_updates == 1 and model.broyden_rejected == 1

    samples = [(-1612.0 * P_MATRIX @ dp, dp) for dp in probes]
    assert ImageJacobian.similarity_kappa(samples) == pytest.approx(1612.0)
    assert ImageJacobian.similarity_kappa(samples, reflection=True) == pytest.approx(1612.0)
    rotation = np.array([[0.0, -900.0], [900.0, 0.0]])
    assert ImageJacobian.similarity_kappa([(rotation @ dp, dp) for dp in probes]) == pytest.approx(900.0)
    model.rescale(2000.0)
    assert model.kappa == pytest.approx(2000.0)
    record = model.to_dict()
    json.dumps(record, allow_nan=False)
    assert record["ready"] is True and record["broyden_updates"] == 1


def test_probe_jacobian_matches_minus_kappa_p_on_the_fake_camera():
    sim = hover_sim()
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view))
    assert align.phase == "probe_settle" and align.pointing_request is None
    captured = {}

    def capture(primitive, sim, view):
        if primitive.phase == "servo" and "J" not in captured:
            captured["J"], captured["kappa"] = primitive.jacobian.J.copy(), true_kappa(sim)

    z0 = view.position[2]
    run(align, sim, view, on_step=capture)
    kappa = captured["kappa"]
    error = np.linalg.norm(captured["J"] + kappa * P_MATRIX) / np.linalg.norm(kappa * P_MATRIX)
    assert error < 0.10
    outcome = outcome_of(align)
    record = outcome["evidence"]["probe_record"]
    assert [r["axis"] for r in record] == ["R1", "R0"]
    assert all(r["moved_m"] >= 0.8 * align.probe_m for r in record)
    assert outcome["evidence"]["probe_steps"] == sum(r["steps"] for r in record) <= 16
    assert outcome["evidence"]["kappa"] == pytest.approx(true_kappa(sim), rel=0.03)
    assert abs(sim.state[2] - z0) < 1e-6  # approach coordinate held


# ------------------------------------------------------------- closed loop
@pytest.mark.parametrize("offset_xy,height", [
    ((0.0, 0.02), 0.10), ((0.012, -0.012), 0.10), ((-0.015, 0.0), 0.08), ((0.01, 0.022), 0.12)])
def test_align_converges_from_a_lateral_offset(offset_xy, height):
    sim = hover_sim(offset_xy=offset_xy, height=height)
    assert 0.015 <= sim.grip_axis_lateral_error(MARK) <= 0.025
    view = view_of(sim)
    start_quaternion, z0 = view.quaternion_xyzw.copy(), view.position[2]
    align = Align(view, track=tracked(sim, view))
    view, feedback, _ = run(align, sim, view)
    outcome = outcome_of(align)
    assert outcome["status"] == "succeeded" and outcome["reason"] == "aligned"
    evidence = outcome["evidence"]
    assert evidence["residual_px"] < 6.0
    assert float(np.linalg.norm(oracle_uv(sim) - setpoint(true_kappa(sim), GRIP_AXIS_OFFSET_M, 512, 512))) < 6.0
    assert sim.grip_axis_lateral_error(MARK) < 0.003
    assert evidence["kappa"] == pytest.approx(true_kappa(sim), rel=0.04)
    assert evidence["residual_mm_est"] == pytest.approx(evidence["residual_px"] / evidence["kappa"] * 1000)
    assert evidence["offset_m"] == [0.0, -0.05] and evidence["repoints"] == 0
    assert evidence["servo_steps"] <= 30 and len(np.asarray(evidence["J"])) == 2
    assert abs(view.position[2] - z0) < 1e-6
    assert quaternion_angle_rad(view.quaternion_xyzw, start_quaternion) < 1e-6
    assert evidence["realization_measured"] == pytest.approx(0.26, abs=0.01)
    assert {f["phase"] for f in feedback} == {"probe_settle", "probe", "servo"}
    assert max(np.linalg.norm(np.asarray(a[:3])) * 0.05 for a in sim.actions) <= 0.010 + 0.003 + 1e-9


def test_align_starts_from_coarse_then_fine_pointing():
    sim = hover_sim(offset_xy=(0.01, 0.015))
    view = view_of(sim)
    align = Align(view, allow_pointing=True)
    assert align.phase == "point_coarse" and align.pointing_request == {"stage": "coarse"}
    with pytest.raises(RuntimeError):
        align.provide_point((100.0, 100.0), view)
    cell = cell_of(oracle_uv(sim))
    align.provide_coarse(cell)
    assert align.pointing_request == {"stage": "fine", "cell_id": cell}
    view, _, requests = run(align, sim, view)
    assert requests == [{"stage": "fine", "cell_id": cell}]
    outcome = outcome_of(align)
    assert outcome["reason"] == "aligned" and outcome["evidence"]["cell_id"] == cell
    assert outcome["evidence"]["pointed_uv"][0] == pytest.approx(list(oracle_uv(hover_sim(offset_xy=(0.01, 0.015)))))
    assert sim.grip_axis_lateral_error(MARK) < 0.003
    assert align.pointing_request is None and align.last_uv is not None and align.last_setpoint is not None


def test_mark_not_visible_and_textureless_mark():
    sim = hover_sim()
    view = view_of(sim)
    coarse_null = Align(view)
    coarse_null.provide_coarse(None)
    outcome = outcome_of(coarse_null)
    assert (outcome["status"], outcome["reason"], outcome["actions"]) == ("failed", "mark_not_visible", 0)
    with pytest.raises(RuntimeError):
        coarse_null.next_action(view)
    fine_null = Align(view)
    for bad in (0, 65, True, 3.0):
        with pytest.raises(ValueError):
            fine_null.provide_coarse(bad)
    fine_null.provide_coarse(12)
    fine_null.provide_point(None, view)
    assert outcome_of(fine_null)["reason"] == "mark_not_visible"
    flat = Align(view)
    flat.provide_coarse(1)
    flat.provide_point((40.0, 40.0), view)
    assert outcome_of(flat)["reason"] == "textureless_mark"
    no_track = Align(view, allow_pointing=False)
    assert outcome_of(no_track)["reason"] == "mark_lost"


def test_align_repoints_when_the_mark_is_lost_then_converges():
    sim = hover_sim(offset_xy=(0.0, 0.022))
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view))

    def hook(primitive, sim, view):
        if primitive.phase == "servo" and primitive.servo_steps == 2 and not primitive.repoints:
            sim._textures = [~texture for texture in sim._textures]  # still visible, no longer the template

    _, _, requests = run(align, sim, view, on_step=hook)
    assert [r["stage"] for r in requests] == ["coarse", "fine"]
    outcome = outcome_of(align)
    assert outcome["reason"] == "aligned" and outcome["evidence"]["repoints"] == 1
    assert outcome["evidence"]["lost_frames_total"] == 2
    assert sim.grip_axis_lateral_error(MARK) < 0.003


@pytest.mark.parametrize("kwargs", [{"allow_pointing": False}, {"max_repoints": 0}])
def test_align_reports_mark_lost_without_repoint_budget(kwargs):
    sim = hover_sim(offset_xy=(0.0, 0.022))
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view), **kwargs)

    def hook(primitive, sim, view):
        if primitive.phase == "servo" and primitive.servo_steps == 2:
            sim.occlude_rows = 0

    _, feedback, requests = run(align, sim, view, on_step=hook)
    outcome = outcome_of(align)
    assert (outcome["status"], outcome["reason"]) == ("failed", "mark_lost") and requests == []
    assert [f["peak"] < 0.6 for f in feedback[-2:]] == [True, True]


def test_setpoint_cutoff_at_construction_and_after_probes():
    sim, view, align = aligned_hover()
    s_v = align.last_setpoint[1]
    track, jacobian = align.track, align.jacobian
    cut = Align(view, track=track, jacobian=jacobian, reprobe=True, allow_pointing=False,
                setpoint_row_limit=s_v - 20)
    outcome = outcome_of(cut)
    assert (outcome["status"], outcome["reason"], outcome["actions"]) == ("succeeded", "setpoint_cutoff", 0)
    assert outcome["evidence"]["setpoint_uv"][1] > s_v - 20
    fresh_sim = hover_sim()
    fresh_view = view_of(fresh_sim)
    late = Align(fresh_view, track=tracked(fresh_sim, fresh_view), setpoint_row_limit=s_v - 20)
    run(late, fresh_sim, fresh_view)
    outcome = outcome_of(late)
    assert outcome["reason"] == "setpoint_cutoff" and outcome["evidence"]["probe_steps"] > 0
    assert outcome["actions"] == outcome["evidence"]["probe_steps"] + outcome["evidence"]["settle_steps"]
    keep = Align(view, track=track, jacobian=jacobian, reprobe=False, setpoint_row_limit=s_v + 50)
    assert not keep.done and keep.phase == "servo"


def test_align_no_convergence_and_probe_failure():
    sim = hover_sim()
    view = view_of(sim)
    short = Align(view, track=tracked(sim, view), max_servo_steps=2)
    run(short, sim, view)
    outcome = outcome_of(short)
    assert (outcome["status"], outcome["reason"], outcome["evidence"]["servo_steps"]) == (
        "failed", "align_no_convergence", 2)
    stuck = hover_sim(translation_ratio=0.0)
    stuck_view = view_of(stuck)
    probe = Align(stuck_view, track=tracked(stuck, stuck_view))
    run(probe, stuck, stuck_view)
    outcome = outcome_of(probe)
    assert (outcome["reason"], outcome["evidence"]["probe_steps"]) == ("probe_failed", 2 * Align.PROBE_MAX_STEPS)
    assert outcome["evidence"]["J"] is None


class JitterSim(FakeWristSim):
    """Adds a 0.4 deg world-x rotation disturbance on every step (no step is 'clean')."""

    def step(self, action):
        observation = super().step(action)
        q = _quaternion_product(_rotvec_quaternion(np.array([0.007 * (-1) ** self.steps, 0.0, 0.0])),
                                _rotvec_quaternion(self.state[3:6]))
        self.state[3:6] = quaternion_to_axisangle(q / np.linalg.norm(q))
        return self.observation()


def test_broyden_updates_only_on_clean_steps():
    sim, _, align = aligned_hover()
    assert align.jacobian.broyden_updates >= 3
    jitter = JitterSim(eef=(MARK[0], MARK[1] + 0.02, MARK[2] + 0.10), marks=(MARK,))
    view = view_of(jitter)
    noisy = Align(view, track=tracked(jitter, view), jacobian=align.jacobian, reprobe=False, max_servo_steps=8)
    updates = noisy.jacobian.broyden_updates
    run(noisy, jitter, view)
    assert noisy.jacobian.broyden_updates == updates
    json.dumps(outcome_of(noisy), allow_nan=False)


# ------------------------------------------------------------------- reuse
def test_retry_reuses_track_and_jacobian_with_a_shifted_setpoint_without_mutating_inputs():
    sim, view, align = aligned_hover(offset_xy=(0.0, 0.018))
    track, jacobian = copy.deepcopy(align.track), align.jacobian
    track_uv, J = track.uv.copy(), jacobian.J.copy()
    shifted = Align(view, offset_m=(0.010, -0.05), track=track, jacobian=jacobian, reprobe=False)
    assert shifted.phase == "servo"
    run(shifted, sim, view)
    outcome = outcome_of(shifted)
    assert outcome["reason"] == "aligned" and outcome["evidence"]["probe_steps"] == 0
    assert outcome["evidence"]["kappa_transfer"]["applied"] is True
    d = MARK - sim.state[:3]
    assert float(d @ sim.R[:, 1]) == pytest.approx(0.010, abs=0.003)
    assert abs(float(d @ sim.R[:, 0])) < 0.003
    assert track.uv.tolist() == track_uv.tolist() and jacobian.J.tolist() == J.tolist()
    assert shifted.track is not track and shifted.jacobian is not jacobian


def test_failed_reprobe_keeps_the_incoming_kappa_anchor():
    sim, view, align = aligned_hover()
    anchor = align.jacobian.anchor
    assert anchor["token"] == align.track.token and anchor["kappa"] == pytest.approx(align.jacobian.kappa)
    sim.translation_ratio = 0.0
    stuck = Align(view, track=align.track, jacobian=align.jacobian, reprobe=True, kappa_transfer=False)
    run(stuck, sim, view)
    assert outcome_of(stuck)["reason"] == "probe_failed"
    assert stuck.jacobian.anchor["xyz"].tolist() == anchor["xyz"].tolist()
    assert stuck.jacobian.anchor["kappa"] == anchor["kappa"]


def test_ladder_refresh_transfers_kappa_to_the_new_depth():
    sim, view, align = aligned_hover(offset_xy=(0.0, 0.02))
    track, jacobian = align.track, align.jacobian
    hover_kappa = jacobian.kappa
    descent = AnchoredMove(view, axis=view.R[:, 2], travel_m=0.02, gripper="open")
    view, _, _ = run(descent, sim, view)
    refresh = Align(view, track=track, jacobian=jacobian, reprobe=True, allow_pointing=False,
                    setpoint_row_limit=0.74 * 512, name="align_refresh")
    transfer = refresh.kappa_transfer
    # the stale hover kappa is >8% low at the new depth; the transfer inherits only the hover estimate's error
    assert hover_kappa < 0.92 * true_kappa(sim)
    assert transfer["applied"] is True and transfer["kappa"] == pytest.approx(true_kappa(sim), rel=0.05)
    assert transfer["approach_change_m"] == pytest.approx(0.02, abs=0.001)
    run(refresh, sim, view)
    outcome = outcome_of(refresh)
    assert outcome["reason"] == "aligned" and outcome["evidence"]["probe_steps"] > 0
    assert outcome["evidence"]["kappa"] == pytest.approx(true_kappa(sim), rel=0.05)
    assert sim.grip_axis_lateral_error(MARK) < 0.003


# ---------------------------------------------------------------- invariants
def test_pending_action_invariants():
    sim = hover_sim()
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view))
    action = align.next_action(view)
    with pytest.raises(RuntimeError, match="observe_transition"):
        align.next_action(view)
    after = StepView(sim.step(action))
    wrong = list(action)
    wrong[0] += 0.01
    with pytest.raises(ValueError, match="exact pending"):
        align.observe_transition(view, wrong, after)
    feedback = align.observe_transition(view, action, after)
    json.dumps(feedback, allow_nan=False)
    assert feedback["phase"] == "probe_settle" and feedback["primitive"] == "align"
    with pytest.raises(RuntimeError, match="no pending"):
        align.observe_transition(view, action, after)
    with pytest.raises(ValueError):
        Align(view, track="not a track")
    with pytest.raises(ValueError):
        Align(view, gripper="hold")


# ------------------------------------------------------------------ monitor
def test_wrist_patch_monitor_default_patch_ok_while_carried_and_fails_when_left_behind():
    # ORACLE placement: the mark sits at the monitor's default patch (cu, 0.70 H) in the wrist view
    a, b = 0.3 * 0.197 / FOCAL_PX, -(0.70 * 512 - CENTER) * 0.197 / FOCAL_PX  # 0.3 px off a rounding tie
    eef = (MARK[0] - 0.05 - b, MARK[1] + a, MARK[2] + 0.10)
    sim = FakeWristSim(eef=eef, marks=(MARK,))
    assert oracle_uv(sim) == pytest.approx([CENTER + 0.3, 0.70 * 512], abs=1e-6)
    view = view_of(sim)
    carried = WristPatchMonitor(view)
    # default patch: the most textured 21x21 window in rows 0.55-0.72 H, columns cu +/- 60
    assert carried.available and abs(carried.uv0[0] - CENTER) <= 61 and 0.55 * 512 - 1 <= carried.uv0[1] <= 0.72 * 512 + 1
    left = copy.deepcopy(carried)
    sim.attached = True
    for _ in range(6):
        view = StepView(sim.step([0, 0, 0.6, 1e-9, 0, 0, 0]))
        result = carried.observe(view)
        assert set(result) == {"available", "ok", "peak", "displacement_px"}
        assert result["available"] and result["ok"] and result["displacement_px"] < 1.0
    still = FakeWristSim(eef=eef, marks=(MARK,))
    results = []
    for _ in range(10):
        results.append(left.observe(StepView(still.step([0, 0, 0.6, 1e-9, 0, 0, 0]))))
    json.dumps(results, allow_nan=False)
    assert all(r["available"] is True and r["ok"] is False for r in results)  # receding patch (moves and shrinks)
    blank = WristPatchMonitor(view_of(hover_sim()), uv=(CENTER, 0.70 * 512))   # explicit textureless patch
    assert not blank.available
    assert blank.observe(view) == {"available": False, "ok": None, "peak": None, "displacement_px": None}


# ------------------------------------------------- live run 001: residual swing before the probe
def test_align_settles_a_residual_swing_before_probing(monkeypatch):
    """Live 001 steps 122/148: Align started while the arm still rotated after a reach; every probe frame
    exceeded CLEAN_ROTATION_RAD and the probe failed. The settle phase holds still and re-bases first."""
    from agent_speedup_fakes import DynWristSim
    sim = DynWristSim(eef=(MARK[0], MARK[1] + 0.02, MARK[2] + 0.10), marks=(MARK,))
    # residual swing left by the previous move (0.42 rad/s: peak ~0.6 deg; live 0.83). In this overdamped fake
    # larger swings die out within the old two probe restarts, so this value is the discriminating one.
    sim.omega = np.array([0.42, 0.0, 0.0])
    view = view_of(sim)
    align = Align(view, track=tracked(sim, view))
    assert align.phase == "probe_settle"
    view, feedback, _ = run(align, sim, view)
    outcome = outcome_of(align)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "aligned"), outcome["evidence"].get("detail")
    record = outcome["evidence"]["settle_record"]
    assert record and record[0]["steps"] > Align.SETTLE_QUIET_STEPS and record[0]["settled"]
    assert sim.grip_axis_lateral_error(MARK) < 0.004
    # without the settle phase the same swing fails the probe (the pre-fix behaviour)
    unsettled = DynWristSim(eef=(MARK[0], MARK[1] + 0.02, MARK[2] + 0.10), marks=(MARK,))
    unsettled.omega = np.array([0.42, 0.0, 0.0])
    view = view_of(unsettled)
    monkeypatch.setattr(Align, "_start_probe", Align._begin_probe)
    old = Align(view, track=tracked(unsettled, view))
    run(old, unsettled, view)
    assert old.outcome()["reason"] == "probe_failed"


# ------------------------------------------------- drawer live run 001: uniform handle bar (aperture problem)
def frontal_bar_sim(eef):
    from agent_speedup_fakes import FakeWristSim
    from gpt_field.contact_skills import axes_to_quaternion_xyzw
    rotvec = tuple(quaternion_to_axisangle(np.asarray(axes_to_quaternion_xyzw((0., 0., 1.), (0., -1., 0.)))))
    # horizontal bar along world x (= R0, image up/down for this frontal hand); uniform, so NCC slides along it
    return FakeWristSim(eef=eef, rotvec=rotvec, marks=(), bars=(((-0.03, -0.13, 1.09), (0.09, -0.13, 1.09), 0.003),))


BAR_CENTER = np.array([0.03, -0.13, 1.09])


def bar_pointing(sim):
    def answer(request, view):
        uv, _ = sim.project(BAR_CENTER)
        return cell_of(uv) if request["stage"] == "coarse" else uv
    return answer


@pytest.mark.parametrize("eef", [(0.05, -0.05, 1.07), (0.0, -0.05, 1.105), (0.07, -0.06, 1.08)])
def test_across_bar_align_pinches_a_uniform_bar_that_defeats_the_2d_probe(eef):
    old_sim = frontal_bar_sim(eef)
    view = view_of(old_sim)
    old = Align(view)
    run(old, old_sim, view, pointing=bar_pointing(old_sim), limit=300)
    assert old.outcome()["reason"] == "probe_failed"          # the live failure (jacobian_inconsistent_with_mount)
    sim = frontal_bar_sim(eef)
    view = view_of(sim)
    align = Align(view, axes="across_bar")
    view, feedback, _ = run(align, sim, view, pointing=bar_pointing(sim), limit=300)
    outcome = outcome_of(align)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "aligned"), outcome["evidence"].get("detail")
    assert abs(sim.state[2] - BAR_CENTER[2]) < 0.003, "grip axis across the bar"
    assert abs(sim.state[0] - BAR_CENTER[0]) < 0.006, "grip axis at the pointed point along the bar"
    evidence = outcome["evidence"]
    assert evidence["along_record"]["source"] == "pointed" and abs(evidence["along_error_m"]) <= 0.005
    assert evidence["kappa"] == pytest.approx(true_kappa_of(sim, BAR_CENTER), rel=0.08)
    assert evidence["jacobian"]["source"] in ("probe_across_bar", "servo_across_bar")


def true_kappa_of(sim, point):
    cam, R = sim.camera()
    return FOCAL_PX / float((np.asarray(point) - cam) @ R[:, 2])


def test_across_bar_refresh_reprobes_and_holds_the_along_position():
    sim = frontal_bar_sim((0.05, -0.05, 1.07))
    view = view_of(sim)
    hover = Align(view, axes="across_bar")
    view, _, _ = run(hover, sim, view, pointing=bar_pointing(sim))
    x_after_hover = sim.state[0]
    for _ in range(8):     # advance 2 cm along the approach, as the descent ladder does
        view = StepView(sim.step([0.0, -0.2, 0.0, 2e-9, 0.0, 0.0, -1.0]))
    refresh = Align(view, track=hover.track, jacobian=hover.jacobian, reprobe=True, allow_pointing=False,
                    axes="across_bar")
    assert refresh.kappa_transfer is None or refresh.kappa_transfer.get("skipped") == "across_bar_mode"
    view, _, requests = run(refresh, sim, view)
    outcome = outcome_of(refresh)
    assert outcome["reason"] in ("aligned", "setpoint_cutoff") and not requests
    assert outcome["evidence"]["along_record"]["source"] == "hold"
    assert abs(sim.state[0] - x_after_hover) < 0.004 and abs(sim.state[2] - BAR_CENTER[2]) < 0.003
    assert outcome["evidence"]["kappa"] > hover.jacobian.kappa     # closer: kappa re-measured, not transferred


def test_across_bar_guards_off_bar_hover_and_blocked_servo():
    far = frontal_bar_sim((0.03, -0.05, 1.035))           # bar 5.5 cm across from the grip axis
    view = view_of(far)
    align = Align(view, axes="across_bar")
    run(align, far, view, pointing=bar_pointing(far))
    assert align.outcome()["reason"] == "hover_off_bar_across"
    sim = frontal_bar_sim((0.05, -0.05, 1.07))

    def block(primitive, sim, view):
        if primitive.phase == "servo":
            sim.translation_ratio = 0.0                       # e.g. the hand rests on the plate (live 424/613)

    view = view_of(sim)
    blocked = Align(view, axes="across_bar")
    run(blocked, sim, view, pointing=bar_pointing(sim), on_step=block)
    outcome = outcome_of(blocked)
    assert outcome["reason"] == "align_blocked" and outcome["actions"] < 40
    assert outcome["evidence"]["blocked"]["realized_m"] <= 0.05 * outcome["evidence"]["blocked"]["commanded_m"]
