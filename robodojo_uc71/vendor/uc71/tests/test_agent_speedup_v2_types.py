"""Kernel acceptance tests (contract §C4.10): records, camera and robot math, HeightField, polygons, threshold helpers,
registries and the kernel's own lint. Test files may import v1 freely (contract §C1.5 item 5)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_robot as robot
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as k
from gpt_field.action import _quaternion_product, _rotvec_quaternion
from gpt_field.contact_skills import GripperCalibration, matrix_to_quaternion_xyzw, rotation_matrix
import gpt_field.wrist_servo as wrist_servo

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
# Task-2 episode-0 start state8 (survey dump; robot state, used here only as a test pose).
T2_START = (-0.2014, 0.0187, 1.1893, 3.1397, 0.0067, -0.0861, 0.0387, -0.0387)


def strict_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def vocabulary():
    if not Path(LIBERO, "libero/libero/bddl_files").is_dir():
        pytest.fail("LIBERO checkout required for the T0 lint vocabulary")
    return th.lint_vocabulary(LIBERO)["vocabulary"]


def world_rotated(R, rotvec):
    """World-frame rotation (the OSC action convention): R_after = Rot(rotvec) @ R."""
    return rotation_matrix(np.asarray(rotvec, float)) @ np.asarray(R, float)


def state_from(R, grip=(0.0, 0.0, 1.0), gap_m=0.08):
    q = matrix_to_quaternion_xyzw(R)
    angle = 2 * math.atan2(np.linalg.norm(q[:3]), q[3])
    rotvec = q[:3] / np.linalg.norm(q[:3]) * angle if angle > 1e-12 else np.zeros(3)
    return (*grip, *rotvec, gap_m / 2, -gap_m / 2)


# ------------------------------------------------------------------------------------------------ 1. records
def test_jsonable_equals_v1():
    from gpt_field.agent_speedup_v1 import jsonable as v1_jsonable

    class Thing:
        def __str__(self):
            return "thing"
    samples = [np.float32(1.5), np.int64(3), np.bool_(True), np.array([[1.0, np.nan], [np.inf, 2]]),
               {"a": {"b": (1, 2.5, float("nan"))}, 3: [np.float64(-np.inf), None, "s"]}, Thing(), float("inf"),
               (np.int8(-2), [np.array([True, False])]), {"nested": {"obj": Thing(), "x": np.float16(0.25)}}]
    for sample in samples:
        assert k.jsonable(sample) == v1_jsonable(sample)
        strict_json(k.jsonable(sample))


def test_meas_validation_and_dict():
    m = k.Meas(np.float64(0.93), 0.002, "kappa")
    assert m.to_dict() == {"value": 0.93, "sigma": 0.002, "source": "kappa"}
    strict_json(m.to_dict())
    for bad in (dict(value=float("nan")), dict(value=float("inf")), dict(sigma=-1e-3), dict(sigma=float("nan")),
                dict(source="tuned"), dict(value="x")):
        kwargs = {"value": 1.0, "sigma": 0.001, "source": "touch", **bad}
        with pytest.raises(ValueError):
            k.Meas(**kwargs)


def test_seed_validation():
    seed = k.Seed("wrist", (np.float32(10.0), 20.0), 3, "plan_cell", 32.0, T2_START)
    assert seed.uv == (10.0, 20.0) and len(seed.pose) == 8
    strict_json(seed.to_dict())
    good = dict(camera="external", uv=(1.0, 2.0), step=0, source="fine", quantization_px=8.0, pose=T2_START)
    for bad in (dict(camera="side"), dict(uv=(1.0, float("nan"))), dict(uv=(1.0,)), dict(step=-1), dict(step=1.5),
                dict(source="guess"), dict(quantization_px=0.0), dict(quantization_px=float("inf")),
                dict(pose=T2_START[:7]), dict(pose=(*T2_START[:7], float("nan")))):
        with pytest.raises(ValueError):
            k.Seed(**{**good, **bad})


def test_need_validation():
    k.PlanNeed("plan", 2, "top")
    k.PlanNeed("replan", 0, "bottom", status_block={"parts": []}, reask_error="x", holding="C")
    for bad in (dict(kind="other"), dict(quarter_turns=4), dict(quarter_turns=-1), dict(quarter_turns=True),
                dict(fingertip_edge="middle"), dict(holding="Z"), dict(status_block=[1])):
        with pytest.raises(ValueError):
            k.PlanNeed(**{"kind": "plan", "quarter_turns": 1, "fingertip_edge": "left", **bad})
    fine = dict(kind="point_fine", pid="A", text="part text", camera="wrist", center_uv=(256.0, 300.0),
                quarter_turns=2, purpose="grasp_point")
    need = k.PointNeed(**fine)
    assert need.center_uv == (256.0, 300.0) and need.previous_rejected is None and need.marker_uv is None
    k.PointNeed(**{**fine, "kind": "point_coarse", "center_uv": None, "purpose": "region_seed"})
    k.PointNeed(**{**fine, "camera": "external", "quarter_turns": 0, "previous_rejected": "x" * 120})
    for bad in (dict(center_uv=None), dict(center_uv=(1.0, float("nan"))), dict(camera="external"),
                dict(purpose="anything"), dict(pid="Q"), dict(kind="point"), dict(previous_rejected="x" * 121),
                dict(quarter_turns=5)):
        with pytest.raises(ValueError):
            k.PointNeed(**{**fine, **bad})
    last = {"reason": "hold_lost", "evidence_text": "gap 1.2 mm below the hold floor", "thresholds": [["G6_x", 1.0]]}
    k.RepairNeed("repair", 2, {"steps": []}, holding=None, last_outcome=last)
    k.RepairNeed("review", 0, {"steps": []})
    for bad in ({**last, "evidence": {}}, {"reason": "hold_lost", "evidence_text": "x"},
                {**last, "reason": "not_a_reason"}, {**last, "evidence_text": "x" * 201},
                {**last, "thresholds": [["a", float("nan")]]}, {**last, "thresholds": [["a", 1.0, 2.0]]}):
        with pytest.raises(ValueError):
            k.RepairNeed("repair", 2, {}, last_outcome=bad)
    with pytest.raises(ValueError):
        k.RepairNeed("fix", 0, {})
    with pytest.raises(ValueError):
        k.CallRecord("chat", 0.1, None, None, True)
    assert strict_json(k.CallRecord("plan", 0.25, "key", None, True).to_dict())["kind"] == "plan"
    reply = k.PointReply("fine", "wrist", 17, (100.0, 200.0), 8.0, (10, 20), 128, 4, 2)
    strict_json(reply.to_dict())
    with pytest.raises(ValueError):
        k.PointReply("point_fine", "wrist", 1, None, 8.0, None, None, None, 0)


def test_plan_records_and_wire_order():
    steps = (k.PlanStep("turn", "A", (), None, "any", (), (k.Until("effect", "B", "changes"), k.Until("stop", None,
                                                                                                         None))),
             k.PlanStep("place_on", "D", ("C",), "B", "any", (k.Keep("upright", "D"), k.Keep("clear_of", "E")), ()))
    parts = tuple(k.PlanPart(pid, f"part {pid}", move, 3, 4) for pid, move in
                  zip("ABCDE", ("hinged", "fixed", "free", "free", "free")))
    plan = k.Plan("scene text", parts, (k.HoldTrue("on", "D", None),), steps, "plan")
    wire = plan.to_json()
    assert list(wire) == ["scene", "parts", "hold_true", "steps"]
    assert list(wire["parts"][0]) == ["id", "text", "move", "w", "x"]
    assert list(wire["steps"][0]) == ["do", "obj", "grip", "ref", "sense", "keep", "until"]
    assert list(wire["steps"][0]["until"][0]) == ["o", "p", "txt"] and list(wire["hold_true"][0]) == ["rel", "a", "b"]
    strict_json(wire)
    assert plan.part("C").move == "free" and steps[0].grip_parts == ("A",) and steps[1].grip_parts == ("C",)
    with pytest.raises(KeyError):
        plan.part("H")
    for bad in (lambda: k.PlanPart("A", "t", "rolling", 0, 0), lambda: k.PlanPart("A", "t", "free", -1, 0),
                lambda: k.Keep("near", "A"), lambda: k.Until("forever", None, None),
                lambda: k.HoldTrue("under", "A", None), lambda: k.PlanStep("fly", "A", (), None, "any", (), ()),
                lambda: k.PlanStep("turn", "A", ("Z",), None, "any", (), ()),
                lambda: k.PlanStep("turn", "A", (), None, "sideways", (), ()),
                lambda: k.Plan("s", (), (), (), "guess"), lambda: k.RepairAnswer("w", "teleport", None, None)):
        with pytest.raises(ValueError):
            bad()
    answer = k.RepairAnswer("why", "insert_step", None, steps[0])
    assert answer.to_json()["step"]["do"] == "turn"
    context = k.StepContext.from_step(1, steps[1], attempt=0)
    assert context.grip == ("C",) and context.ref == "B" and context.resume is None
    assert k.StepContext.from_step(0, steps[0], attempt=2, grip=("B",)).grip == ("B",)
    state = k.StepState.pending(0, steps[0])
    assert state.status == "pending" and strict_json(state.compact())["do"] == "turn"
    with pytest.raises(ValueError):
        k.StepState(0, steps[0], "dropped", None, {}, {}, 0, 0)


# -------------------------------------------------------------------------------------------- 2. WristCamera
def random_pose(rng):
    axis = rng.normal(size=3)
    rotvec = axis / np.linalg.norm(axis) * rng.uniform(0.0, math.pi)
    grip = rng.uniform([-0.3, -0.3, 0.95], [0.3, 0.3, 1.3])
    gap = rng.uniform(0.0, 0.08)
    return (*grip, *rotvec, gap / 2, -gap / 2)


def test_wrist_camera_round_trip_and_ray():
    rng = np.random.default_rng(0)
    camera = k.WristCamera.from_robot()
    for _ in range(50):
        pose = k.HandPose.from_state8(random_pose(rng))
        uv = rng.uniform(0, 511, size=2)
        z = rng.uniform(0.05, 0.6)
        point = camera.backproject(pose, uv, z)
        back, depth = camera.project(pose, point)
        assert depth == pytest.approx(z, abs=1e-12)
        assert np.allclose(back, uv, atol=1e-9)
        again = camera.backproject(pose, back, depth)
        assert np.linalg.norm(again - point) < 1e-9
        origin, direction = camera.ray(pose, uv)
        assert np.allclose(origin, pose.camera) and abs(np.linalg.norm(direction) - 1) < 1e-12
        distance = np.linalg.norm(np.cross(point - origin, direction))
        assert distance < 1e-9
    pose = k.HandPose.from_state8(T2_START)
    behind = pose.camera - 0.1 * pose.approach
    assert camera.project(pose, behind)[0] is None and camera.project(pose, behind)[1] < 0
    assert camera.kappa_at(0.25) == pytest.approx(camera.f_px / 0.25)
    with pytest.raises(ValueError):
        camera.kappa_at(0.0)
    with pytest.raises(ValueError):
        k.WristCamera(f_px=-1.0)


def test_wrist_camera_matches_v1_fake_projection():
    from agent_speedup_fakes import FOCAL_PX, FakeWristSim
    rng = np.random.default_rng(1)
    camera = k.WristCamera(f_px=FOCAL_PX)
    sim = FakeWristSim()
    for _ in range(50):
        state = np.array(random_pose(rng))
        sim.state = state.copy()
        pose = k.HandPose.from_state8(state)
        z = rng.uniform(0.1, 0.5)
        a, b = rng.uniform(-0.6 * z, 0.6 * z, size=2)              # within the 37.5 deg half field of view
        target = pose.camera + z * pose.approach + a * pose.jaw + b * pose.up
        ours, z_ours = camera.project(pose, target)
        theirs, z_theirs = sim.project(target)
        assert z_ours == pytest.approx(z_theirs, abs=1e-12) == pytest.approx(z, abs=1e-12)
        assert theirs is not None and ours is not None
        assert np.max(np.abs(ours - theirs)) < 1e-6
        assert np.all((ours >= 0) & (ours <= 511))


def test_default_focal_is_r3():
    assert k.WristCamera.from_robot().f_px == robot.focal_length_px(512) == pytest.approx(333.6)
    assert k.WristCamera.from_robot().size_px == k.WRIST_IMAGE_PX == 512
    assert k.WristCamera.from_robot(256).f_px == pytest.approx(333.6 / 2)


def test_hand_pose_facts():
    pose = k.HandPose.from_state8(T2_START)
    R = rotation_matrix(T2_START[3:6])
    assert np.allclose(pose.R, R) and np.allclose(pose.grip, T2_START[:3])
    assert np.allclose(pose.hand, pose.grip - 0.097 * R[:, 2]) and np.allclose(pose.camera, pose.hand + 0.05 * R[:, 0])
    assert pose.gap_mm == pytest.approx(77.4) and np.allclose(pose.jaw, R[:, 1]) and np.allclose(pose.up, R[:, 0])
    assert np.allclose(pose.approach, (-0.0548, -0.0008, -0.9985), atol=1e-3)
    assert np.allclose(pose.quaternion_xyzw, _rotvec_quaternion(np.array(T2_START[3:6])))
    strict_json(pose.to_dict())
    with pytest.raises(ValueError):
        pose.grip[0] = 1.0


# --------------------------------------------------------------------------------------------- 3. RobotModel
def test_pad_band_and_separation():
    model = k.RobotModel()
    low, high = model.pad_band_z(1.0)
    assert low == pytest.approx(0.9956, abs=1e-12) and high == pytest.approx(1.0116, abs=1e-12)
    assert model.pad_separation_mm(25.39) == pytest.approx(24.39)
    assert model.t_f_m == pytest.approx(0.0093) and model.p_min_m == pytest.approx(0.031)
    assert model.pad_span_m == pytest.approx((-0.0116, 0.0044)) and model.pad_length_m == pytest.approx(0.016)
    assert model.grip_axis_offset_m == wrist_servo.GRIP_AXIS_OFFSET_M
    assert [t.name for t in model.thresholds()] == [t.name for t in robot.robot_thresholds()]
    strict_json(model.to_dict())


def bounds(poly):
    poly = np.asarray(poly)
    return poly.min(axis=0), poly.max(axis=0)


@pytest.mark.parametrize("q_mm", [40.0, 0.0])
def test_finger_columns_match_asset(q_mm):
    model = k.RobotModel()
    asset = robot.finger_landing_footprints()[str(int(q_mm))]
    f1_r1, f1_r0 = np.array(asset["finger1"]["R1_mm"]) / 1000, np.array(asset["finger1"]["R0_mm"]) / 1000
    f2_r1, f2_r0 = np.array(asset["finger2"]["R1_mm"]) / 1000, np.array(asset["finger2"]["R0_mm"]) / 1000
    grip = np.array([0.1, -0.2])
    # jaw yaw 0: jaw = +x, R0 = +y
    one, two = model.finger_columns_xy(grip, (1.0, 0.0), finger_q_mm=q_mm)
    (lo1, hi1), (lo2, hi2) = bounds(one), bounds(two)
    assert np.allclose([lo1[0], hi1[0]], grip[0] + f1_r1) and np.allclose([lo1[1], hi1[1]], grip[1] + f1_r0)
    assert np.allclose([lo2[0], hi2[0]], grip[0] + f2_r1) and np.allclose([lo2[1], hi2[1]], grip[1] + f2_r0)
    # jaw yaw 90 deg: jaw = +y, R0 = -x
    one, _ = model.finger_columns_xy(grip, (0.0, 1.0), finger_q_mm=q_mm)
    lo, hi = bounds(one)
    assert np.allclose([lo[1], hi[1]], grip[1] + f1_r1) and np.allclose([lo[0], hi[0]], grip[0] - f1_r0[::-1])
    if q_mm == 40.0:
        assert np.allclose(f1_r1, (0.0399, 0.05429)) and np.allclose(f1_r0, (-0.00548, 0.00614))


def test_finger_columns_interpolate_and_inflate():
    model = k.RobotModel()
    one, two = model.finger_columns_xy((0.0, 0.0), (1.0, 0.0), finger_q_mm=39.25)      # g_open 78.5 mm
    lo, hi = bounds(one)
    assert lo[0] == pytest.approx(0.03915, abs=1e-9) and hi[0] == pytest.approx(0.05354, abs=1e-9)
    lo2, hi2 = bounds(two)
    assert lo2[0] == pytest.approx(-0.05354, abs=1e-9) and hi2[0] == pytest.approx(-0.03915, abs=1e-9)
    big, _ = model.finger_columns_xy((0.0, 0.0), (1.0, 0.0), finger_q_mm=39.25, inflate_m=0.002)
    blo, bhi = bounds(big)
    assert np.allclose(blo, lo - 0.002) and np.allclose(bhi, hi + 0.002)
    clamped, _ = model.finger_columns_xy((0.0, 0.0), (1.0, 0.0), finger_q_mm=45.0)
    assert np.allclose(bounds(clamped)[0], bounds(model.finger_columns_xy((0, 0), (1, 0), finger_q_mm=40.0)[0])[0])
    for poly in (one, two, big):
        assert polygon_area(poly) > 0     # CCW
    with pytest.raises(ValueError):
        model.finger_columns_xy((0, 0), (0.0, 0.0), finger_q_mm=40.0)


def test_footprint_slot_and_palm():
    model = k.RobotModel()
    lo, hi = bounds(model.hand_footprint_xy((0.0, 0.0), (1.0, 0.0)))
    assert np.allclose(lo, (-0.10399, -0.03164)) and np.allclose(hi, (0.10043, 0.03162))
    lo, hi = bounds(model.finger_slot_xy((0.0, 0.0), (1.0, 0.0), finger_q_mm=40.0))
    assert np.allclose(lo, (-0.0399, -0.008)) and np.allclose(hi, (0.0399, 0.008))
    assert model.palm_underside_z(1.0) == pytest.approx(1.0 + robot.palm_underside_at(0.0) / 1000)
    assert model.palm_underside_z(1.0) == pytest.approx(1.03104, abs=1e-6)
    assert model.palm_underside_z(1.0, 200.0) is None


def test_yaw_about_approach_sign_and_task2_world_labels():
    model = k.RobotModel()
    rng = np.random.default_rng(2)
    for _ in range(20):
        R = rotation_matrix(rng.normal(size=3))
        delta = rng.uniform(0.01, 0.3)
        after = world_rotated(R, delta * R[:, 2])
        q0, q1 = matrix_to_quaternion_xyzw(R), matrix_to_quaternion_xyzw(after)
        assert model.yaw_about_approach(q0, q1, R[:, 2]) == pytest.approx(delta, abs=1e-9)
        assert model.yaw_about_approach(q0, matrix_to_quaternion_xyzw(world_rotated(R, -delta * R[:, 2])),
                                        R[:, 2]) == pytest.approx(-delta, abs=1e-9)
        # the quaternion product convention agrees with the matrix convention
        composed = _quaternion_product(_rotvec_quaternion(delta * R[:, 2]), q0)
        assert model.yaw_about_approach(q0, composed, R[:, 2]) == pytest.approx(delta, abs=1e-9)
    R = rotation_matrix(T2_START[3:6])
    q0 = matrix_to_quaternion_xyzw(R)
    for delta in (0.05, 0.1, 0.3):
        world_z = matrix_to_quaternion_xyzw(world_rotated(R, (0.0, 0.0, delta)))      # SP / CP "+1" (world +z)
        assert model.yaw_about_approach(q0, world_z, R[:, 2]) == pytest.approx(-delta, rel=2e-3)


def test_jaw_yaw_and_budget():
    model = k.RobotModel()
    R = rotation_matrix(T2_START[3:6])
    yaw0 = model.jaw_yaw(R)
    assert abs(yaw0 - math.pi / 2) < 0.01 and 0.0 <= yaw0 < math.pi
    for delta in (0.05, 0.2):
        twisted = world_rotated(R, delta * R[:, 2])                                   # contract sense +1
        change = (model.jaw_yaw(twisted) - yaw0 + math.pi / 2) % math.pi - math.pi / 2
        assert change == pytest.approx(-delta, rel=1e-2)
    q0 = matrix_to_quaternion_xyzw(R)
    assert model.yaw_budget_left(q0, q0, +1) == pytest.approx(robot.yaw_budget_rad())
    assert model.yaw_budget_left(q0, q0, -1) == pytest.approx(1.9619, abs=1e-3)
    q1 = matrix_to_quaternion_xyzw(world_rotated(R, 0.4 * R[:, 2]))
    assert model.net_yaw(q0, q1) == pytest.approx(0.4, abs=1e-9)
    assert model.yaw_budget_left(q0, q1, +1) == pytest.approx(robot.yaw_budget_rad() - 0.4)
    assert model.yaw_budget_left(q0, q1, -1) == pytest.approx(robot.yaw_budget_rad() + 0.4)
    with pytest.raises(ValueError):
        model.yaw_budget_left(q0, q1, 0)
    with pytest.raises(ValueError):
        model.yaw_budget_left(q0, q1, True)
    quaternion = model.quaternion_for(R[:, 1], R[:, 2])
    assert k.quaternion_angle_rad(quaternion, q0) < 1e-6
    assert np.allclose(k.quaternion_approach_axis(q0), R[:, 2])


# --------------------------------------------------------------------------------------------- polygons
def polygon_area(poly):
    p = np.asarray(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def winding_inside(poly, point):
    """Independent point-in-polygon (winding number)."""
    p = np.asarray(poly)
    wn = 0
    for (x0, y0), (x1, y1) in zip(p, np.roll(p, -1, axis=0)):
        cross = (x1 - x0) * (point[1] - y0) - (point[0] - x0) * (y1 - y0)
        if y0 <= point[1] < y1 and cross > 0:
            wn += 1
        elif y1 <= point[1] < y0 and cross < 0:
            wn -= 1
    return wn != 0


def test_polygon_helpers():
    square = k.rect_polygon((0.0, 0.0), (1.0, 0.0), 0.05, 0.02)
    assert polygon_area(square) == pytest.approx(4 * 0.05 * 0.02)
    tilted = k.rect_polygon((0.1, 0.2), (1.0, 1.0), 0.03, 0.01)
    assert polygon_area(tilted) == pytest.approx(4 * 0.03 * 0.01) and np.allclose(tilted.mean(axis=0), (0.1, 0.2))
    inside = k.poly_contains(square, [(0.0, 0.0), (0.049, 0.019), (0.051, 0.0), (0.0, -0.021)])
    assert inside.tolist() == [True, True, False, False]
    ell = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], float)
    assert k.poly_contains(ell, [(0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]).tolist() == [True, False, True]
    grown = k.poly_inflate(square, 0.01)
    assert polygon_area(grown) == pytest.approx((0.1 + 0.02) * (0.04 + 0.02))
    assert k.poly_contains(grown, [(0.0595, 0.0295)]).all() and not k.poly_contains(grown, [(0.0605, 0.0)]).any()
    assert polygon_area(k.poly_inflate(square, 0.0)) == pytest.approx(polygon_area(square))
    with pytest.raises(ValueError):
        k.poly_inflate(square, -0.01)
    assert k.poly_iou(square, square) == pytest.approx(1.0)
    assert k.poly_iou(square, square + [1.0, 0.0]) == 0.0
    half = k.rect_polygon((0.05, 0.0), (1.0, 0.0), 0.05, 0.02)
    assert k.poly_iou(square, half) == pytest.approx(1 / 3, abs=0.03)
    turned = k.rotate_xy(square, (0.0, 0.0), math.pi / 2)
    assert np.allclose(bounds(turned)[0], (-0.02, -0.05)) and np.allclose(bounds(turned)[1], (0.02, 0.05))
    assert np.allclose(k.rotate_xy([(1.0, 0.0)], (0.0, 0.0), math.pi / 2), [(0.0, 1.0)])


def distance_to_polygon(poly, point):
    """Euclidean distance from a point to the boundary-or-interior of a convex polygon."""
    p = np.asarray(poly, float)
    if k.poly_contains(p, [point])[0]:
        return 0.0
    best = math.inf
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        t = np.clip(np.dot(point - a, b - a) / max(np.dot(b - a, b - a), 1e-30), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (a + t * (b - a)))))
    return best


@pytest.mark.parametrize("yaw_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 137.0, -22.5])
def test_poly_inflate_is_rotation_invariant(yaw_deg):
    """Every side of a rotated rectangle grows by exactly the margin (B0-R3): the offset never depends on the world
    yaw, so keep-outs and clearances do not depend on the scene orientation."""
    margin, ha, hc = 0.02, 0.05, 0.02
    axis = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    across = np.array([-axis[1], axis[0]])
    centre = np.array([0.3, -0.1])
    rect = k.rect_polygon(centre, axis, ha, hc)
    grown = k.poly_inflate(rect, margin)
    rel = grown - centre
    for direction, half in ((axis, ha), (-axis, ha), (across, hc), (-across, hc)):
        assert float(np.max(rel @ direction)) == pytest.approx(half + margin, abs=1e-9)
    assert polygon_area(grown) == pytest.approx((2 * ha + 2 * margin) * (2 * hc + 2 * margin), rel=1e-9)
    assert polygon_area(grown) > 0                                          # CCW
    # 2.8 cm off the long side along its normal is outside (the axis square reached up to 2.83 cm at 45 deg)
    assert not k.poly_contains(grown, [centre + (hc + 0.028) * across])[0]
    assert k.poly_contains(grown, [centre + (hc + 0.0199) * across])[0]


def test_poly_inflate_contains_the_disk_offset_and_stays_close():
    rng = np.random.default_rng(11)
    margin = 0.01
    shapes = [k.rect_polygon((0, 0), (1, 2), 0.04, 0.01),
              np.array([(0.0, 0.0), (0.1, 0.0), (0.0, 0.009)]),                 # a 5 deg sharp corner
              rng.uniform(-0.05, 0.05, (12, 2)),                                  # a random hull
              np.array([(0.0, 0.0), (0.05, 0.05)]),                               # a segment
              np.array([(0.01, 0.02)])]                                           # a point
    for shape in shapes:
        hull = k.convex_hull(shape)
        grown = k.poly_inflate(shape, margin)
        assert polygon_area(grown) > 0
        for angle in np.linspace(0.0, 2 * math.pi, 73):                      # Minkowski sum with the disk inside
            for vertex in hull:
                point = vertex + margin * (1 - 1e-9) * np.array([math.cos(angle), math.sin(angle)])
                assert k.poly_contains(grown, [point])[0], (shape.tolist(), point)
        for vertex in grown:                                                   # never beyond sqrt 2 x the margin
            assert distance_to_polygon(hull, vertex) <= math.sqrt(2) * margin + 1e-12
    segment = k.poly_inflate([(0.0, 0.0), (0.05, 0.0)], margin)
    assert polygon_area(segment) == pytest.approx((0.05 + 2 * margin) * 2 * margin)
    triangle = [(0.0, 0.0), (0.05, 0.0), (0.02, 0.03)]
    near = k.poly_inflate([*triangle, (0.05 + 1e-13, 1e-13)], margin)        # a hull vertex 1.4e-13 m from another
    assert polygon_area(near) == pytest.approx(polygon_area(k.poly_inflate(triangle, margin)), rel=1e-9)


def test_poly_iou_resolves_small_polygons():
    """B0-R8: identical polygons below one HeightField cell still give 1.0; centimetre polygons are no longer
    quantized at 5 mm."""
    for side in (0.001, 0.002, 0.004, 0.006):
        p = k.rect_polygon((0.0123, 0.0456), (1, 0), side / 2, side / 2)
        assert k.poly_iou(p, p) == pytest.approx(1.0), side
    a = k.rect_polygon((0, 0), (1, 0), 0.01, 0.01)
    b = k.rect_polygon((0.004, 0), (1, 0), 0.01, 0.01)
    assert k.poly_iou(a, b) == pytest.approx(0.016 / 0.024, abs=0.02)
    big = k.rect_polygon((0, 0), (1, 1), 0.2, 0.15)
    assert k.poly_iou(big, k.rotate_xy(big, (0, 0), 0.05)) == pytest.approx(
        k.poly_iou(k.rotate_xy(big, (0, 0), 0.05), big))
    assert k.poly_iou(np.array([(0, 0), (1, 0), (2, 0)]), a) == 0.0             # zero area
    assert k.poly_iou(np.zeros((3, 2)), np.zeros((3, 2))) == 0.0              # zero extent


# --------------------------------------------------------------------------------------------- 4. HeightField
def fresh():
    return k.HeightField.empty((0.0, 0.0))


def test_heightfield_geometry_and_indexing():
    hf = fresh()
    assert hf.shape == (201, 201) and hf.cell_m == k.HF_CELL_M
    assert np.allclose(hf.cell_xy((100, 100)), (0.0, 0.0)) and np.allclose(hf.origin_xy, (-0.5, -0.5))
    rng = np.random.default_rng(3)
    for iy, ix in rng.integers(0, 201, size=(200, 2)):
        assert hf.index(hf.cell_xy((iy, ix))) == (iy, ix)
        assert hf.index(hf.cell_xy((iy, ix)) + rng.uniform(-0.0024, 0.0024, 2)) == (iy, ix)
    assert hf.cell_xy((3, 7))[0] > hf.cell_xy((3, 6))[0] and hf.cell_xy((4, 7))[1] > hf.cell_xy((3, 7))[1]
    assert hf.index((0.6, 0.0)) is None and hf.index((float("nan"), 0.0)) is None
    small = k.HeightField.empty((1.0, 2.0), half_extent_m=0.02, cell_m=0.01)
    assert small.shape == (5, 5) and np.allclose(small.cell_xy((2, 2)), (1.0, 2.0))


def test_heightfield_precedence():
    ij = (100, 100)
    hf = fresh()
    assert hf.stamp_ij(ij, 0.93, 0.003, "kappa", 1)
    assert hf.stamp_ij(ij, 0.95, 0.001, "touch", 2) and k.HF_SOURCES[hf.source[ij]] == "touch"   # touch over kappa
    assert not hf.stamp_ij(ij, 0.90, 0.001, "kappa", 3) and hf.z[ij] == 0.95                    # kappa never over touch
    for measurement in ("kappa", "loom", "dense"):
        for fill in ("plane", "obstacle"):
            hf = fresh()
            assert hf.stamp_ij(ij, 1.0, 0.01, fill, 1)
            assert hf.known[ij] and not hf.measured[ij]
            assert hf.stamp_ij(ij, 0.95, 0.004, measurement, 2)                  # any measurement replaces a fill
            assert k.HF_SOURCES[hf.source[ij]] == measurement and hf.z[ij] == 0.95 and hf.measured[ij]
            assert not hf.stamp_ij(ij, 1.2, 0.001, fill, 3) and hf.z[ij] == 0.95  # a fill never replaces it
    hf = fresh()
    assert hf.stamp_ij(ij, 0.95, 0.004, "kappa", 1)
    assert not hf.stamp_ij(ij, 0.97, 0.001, "dense", 2)                          # dense ranks below kappa
    hf = fresh()
    assert hf.stamp_ij(ij, 0.97, 0.002, "dense", 1) and hf.stamp_ij(ij, 0.95, 0.004, "kappa", 2)
    assert hf.z[ij] == 0.95
    hf = fresh()
    assert hf.stamp_ij(ij, 0.91, 0.002, "plane", 1) and hf.stamp_ij(ij, 0.99, 0.01, "obstacle", 2)
    assert k.HF_SOURCES[hf.source[ij]] == "obstacle" and hf.z[ij] == 0.99                # newer fill replaces
    assert hf.stamp_ij(ij, 0.91, 0.002, "plane", 3) and hf.z[ij] == 0.91 and hf.stamp[ij] == 3
    for bad in (lambda: hf.stamp_ij(ij, 0.9, 0.001, "none", 1), lambda: hf.stamp_ij(ij, 0.9, 0.001, "tuned", 1),
                lambda: hf.stamp_ij(ij, float("nan"), 0.001, "kappa", 1),
                lambda: hf.stamp_ij(ij, 0.9, -0.001, "kappa", 1), lambda: hf.stamp_ij(ij, 0.9, 0.001, "kappa", -1)):
        with pytest.raises(ValueError):
            bad()
    assert not hf.stamp_ij((500, 3), 0.9, 0.001, "kappa", 1) and not hf.stamp_xy((9.0, 9.0), 0.9, 0.0, "touch", 1)


def test_heightfield_fusion_and_disagreement():
    ij = (10, 20)
    hf = fresh()
    hf.stamp_ij(ij, 1.000, 0.004, "kappa", 1)
    assert hf.stamp_ij(ij, 1.004, 0.004, "loom", 2)                              # equal rank: inverse variance
    assert hf.z[ij] == pytest.approx(1.002) and hf.sigma[ij] == pytest.approx(0.004 / math.sqrt(2))
    hf = fresh()
    hf.stamp_ij(ij, 1.000, 0.001, "kappa", 1)
    hf.stamp_ij(ij, 1.003, 0.003, "kappa", 2)
    w0, w1 = 1 / 0.001 ** 2, 1 / 0.003 ** 2
    assert hf.z[ij] == pytest.approx((w0 * 1.000 + w1 * 1.003) / (w0 + w1))
    hf = fresh()
    hf.stamp_ij(ij, 1.000, 0.001, "kappa", 1)
    assert hf.stamp_ij(ij, 1.010, 0.001, "kappa", 2)                             # 10 mm > 3 sqrt(2) mm: replace
    assert hf.z[ij] == 1.010 and hf.sigma[ij] == 0.001
    hf = fresh()
    hf.stamp_ij(ij, 0.95, 0.0, "touch", 1)
    hf.stamp_ij(ij, 0.95, 0.002, "touch", 2)
    assert hf.z[ij] == 0.95 and hf.sigma[ij] == 0.0


def test_heightfield_samples_vacate_and_summary():
    hf = fresh()
    samples = [k.DepthSample((0.01 * i, 0.0, 0.9 + 0.001 * i), 0.002, 0.001, "kappa", (1.0, 2.0), i, 5, 900.0)
               for i in range(5)]
    assert hf.stamp_samples(samples, 5) == 5
    z, sigma, known = hf.sample((0.02, 0.0))
    assert known and z == pytest.approx(0.902) and sigma == pytest.approx(0.002)
    assert hf.sample((0.3, 0.3)) == (None, None, False)
    cells = np.array([hf.index((0.01 * i, 0.0)) for i in range(3)])
    assert hf.vacate(cells, 9) == 3 and hf.vacate(cells, 10) == 0
    assert not hf.known[tuple(cells[0])] and hf.source[tuple(cells[0])] == 0 and hf.stamp[tuple(cells[0])] == 10
    summary = strict_json(hf.summary())
    assert summary["counts"]["kappa"] == 2 and summary["known"] == 2 and summary["measured"] == 2
    assert summary["z_range"] == pytest.approx([0.903, 0.904])
    for bad in (dict(source="plane"), dict(sigma_z=-1.0), dict(xyz=(0.0, float("nan"), 1.0))):
        with pytest.raises(ValueError):
            k.DepthSample(**{"xyz": (0, 0, 1), "sigma_z": 0.001, "sigma_xy": 0.001, "source": "kappa", "uv": None,
                             "feature_id": None, "step": 0, "kappa": None, **bad})


def brute_force(hf, poly, exclude=()):
    skip = {tuple(c) for c in np.asarray(exclude, int).reshape(-1, 2)}
    cells = []
    ny, nx = hf.shape
    for iy in range(ny):
        for ix in range(nx):
            if (iy, ix) not in skip and winding_inside(poly, hf.cell_xy((iy, ix))):
                cells.append((iy, ix))
    if not cells:
        return None, None, 0.0, 0.0, cells
    known = [hf.known[c] for c in cells]
    measured = [hf.known[c] and k.HF_SOURCES[hf.source[c]] not in k.HF_FILLS for c in cells]
    kf, mf = float(np.mean(known)), float(np.mean(measured))
    ks = [c for c in cells if hf.known[c]]
    if not ks:
        return None, None, kf, mf, cells
    best = max(ks, key=lambda c: hf.z[c])
    return float(hf.z[best]), float(hf.sigma[best]), kf, mf, cells


def test_max_in_polygon_against_brute_force():
    rng = np.random.default_rng(4)
    hf = k.HeightField.empty((0.0, 0.0), half_extent_m=0.1)
    for iy in range(hf.shape[0]):
        for ix in range(hf.shape[1]):
            draw = rng.uniform()
            if draw < 0.3:
                continue
            source = ("plane", "obstacle", "dense", "kappa", "loom", "touch")[int(rng.integers(0, 6))]
            hf.stamp_ij((iy, ix), rng.uniform(0.9, 1.1), rng.uniform(0.0005, 0.005), source, 1)
    for trial in range(25):
        centre = rng.uniform(-0.08, 0.08, 2)
        angle = rng.uniform(0, math.pi)
        poly = k.rect_polygon(centre, (math.cos(angle), math.sin(angle)), rng.uniform(0.005, 0.05),
                              rng.uniform(0.005, 0.03))
        if trial % 3 == 0:
            poly = k.poly_inflate(k.rotate_xy(poly, centre, 0.3), 0.004)
        z, sigma, kf, mf, cells = brute_force(hf, poly)
        result = hf.max_in_polygon(poly)
        assert result[2] == pytest.approx(kf) and result[3] == pytest.approx(mf)
        assert (result[0] is None) == (z is None)
        if z is not None:
            assert result[0] == pytest.approx(z) and result[1] == pytest.approx(sigma)
        assert sorted(map(tuple, hf.cells_in_polygon(poly).tolist())) == sorted(cells)
        if len(cells) > 2:
            exclude = np.array(cells[: len(cells) // 2])
            z, sigma, kf, mf, _ = brute_force(hf, poly, exclude)
            result = hf.max_in_polygon(poly, exclude=exclude)
            assert result[2] == pytest.approx(kf) and result[3] == pytest.approx(mf)
            assert (result[0] is None) == (z is None) and (z is None or result[0] == pytest.approx(z))
    assert hf.max_in_polygon(k.rect_polygon((5.0, 5.0), (1, 0), 0.01, 0.01)) == (None, None, 0.0, 0.0)


# ------------------------------------------------------------------------------------- scene records
class Summarized:
    def to_dict(self):
        return {"radius": np.int64(10), "peak": np.float32(0.9)}


def test_part_record_queries_and_dict():
    widths = tuple((math.radians(15.0 * i), k.Meas(0.020 + 0.001 * i, 0.001, "kappa")) for i in range(12))
    part = k.PartRecord("A", "part text", "hinged", frozenset({"grip"}), width_by_yaw=widths,
                        elongation=(np.array([1.0, 0.0]), 0.8), xyz=np.array([0.1, 0.2, 0.95]), cov=np.eye(3) * 1e-6,
                        track=Summarized(), region_xy=np.zeros((4, 2)), region_cells=np.zeros((7, 2), int),
                        circle=(np.zeros(2), 0.047, 0.8), rigid_patch={"track": Summarized(), "dist_m": 0.02},
                        external={"uv": (1.0, 2.0), "template": object()}, top_z=k.Meas(0.955, 0.002, "kappa"),
                        sigma_h_m=0.002)
    assert part.width_at(math.pi - 0.01).value == pytest.approx(0.020)            # nearest yaw mod pi is 0
    assert part.width_at(math.radians(31.0)).value == pytest.approx(0.022)
    assert part.width_at(-math.radians(15.0)).value == pytest.approx(0.031)       # -15 deg == 165 deg
    assert part.is_elongated() and not k.PartRecord("B", "t", "fixed", frozenset()).is_elongated()
    assert k.PartRecord("B", "t", "fixed", frozenset()).width_at(0.0) is None
    data = strict_json(part.to_dict())
    assert data["track"] == {"radius": 10, "peak": pytest.approx(0.9)} and data["region_cells"] == 7
    assert "template" not in data["external"] and data["status"] == "seeded" and data["sigma_h_m"] == 0.002
    assert data["elongation"] == {"axis": [1.0, 0.0], "coherence": 0.8}
    assert strict_json(k.PartRecord("B", "t", "fixed", frozenset()).to_dict())["sigma_h_m"] is None   # nothing measured
    for bad in (dict(pid="Z"), dict(move_prior="rolling"), dict(roles=frozenset({"driver"})),
                dict(status="gone")):
        with pytest.raises(ValueError):
            k.PartRecord(**{"pid": "A", "text": "t", "move_prior": "free", "roles": frozenset(), **bad})


def test_part_record_planning_invariants():
    """B0-R10 / B0-R4: a record with a 3-D point carries sigma_h_m (the planner reads it), a registered or
    provisional one also its registration step; a stored elongation is a unit axis with a coherence in [0, 1]."""
    base = {"pid": "A", "text": "t", "move_prior": "free", "roles": frozenset({"grip"})}
    point = {"xyz": np.array([0.1, 0.2, 0.95]), "cov": np.eye(3) * 1e-6}
    good = k.PartRecord(**base, **point, sigma_h_m=0.001, provisional=True, registration_step=3)
    assert good.sigma_h_m == 0.001 and good.registration_step == 3 and not good.is_elongated()
    registered = k.PartRecord(**base, **point, sigma_h_m=np.float64(0.002), status="registered", registration_step=7,
                              elongation=((0.6, 0.8), np.float32(0.7)))
    assert isinstance(registered.sigma_h_m, float) and registered.is_elongated()
    assert isinstance(registered.elongation[1], float) and registered.elongation[0].shape == (2,)
    for bad in (dict(**point),                                                    # xyz without sigma_h_m
                dict(**point, sigma_h_m=float("nan")), dict(**point, sigma_h_m=-0.001),
                dict(**point, sigma_h_m=0.001, provisional=True),                 # no registration step
                dict(status="registered", sigma_h_m=0.001, registration_step=1),  # registered without xyz
                dict(**point, sigma_h_m=0.001, status="registered", registration_step=-1),
                dict(elongation=((1.0, 1.0), 0.8)),                                # not a unit axis
                dict(elongation=((1.0, 0.0), 1.5)), dict(elongation=((1.0, 0.0), float("nan"))),
                dict(elongation=((1.0, 0.0, 0.0), 0.8)), dict(elongation=((1.0, 0.0),))):
        with pytest.raises(ValueError):
            k.PartRecord(**base, **bad)
    import dataclasses
    with pytest.raises(ValueError):                                               # replace re-runs the checks
        dataclasses.replace(k.PartRecord(**base), status="registered")
    failure = k.RegistrationFailure("A", "part_not_found", {"tried": 2}, ())
    strict_json(failure.to_dict())
    with pytest.raises(ValueError):
        k.RegistrationFailure("A", "no_such_reason", {}, ())
    keepout = k.KeepOut("E", "clear_of", k.rect_polygon((0, 0), (1, 0), 0.02, 0.02), 0.9, 0.95, ())
    strict_json(keepout.to_dict())
    with pytest.raises(ValueError):
        k.KeepOut("E", "clear_of", np.zeros((4, 2)), 1.0, 0.9, ())


def test_part_record_review1_fields():
    """Design review 1 decision §2.1 K-1: top_bound / top_upper_z (rulings R2), top_band (S-1..S-3) and axis_check
    (S-4'), all defaulting to None; validated at construction and by dataclasses.replace; JSON-safe in to_dict."""
    import dataclasses
    base = {"pid": "C", "text": "t", "move_prior": "free", "roles": frozenset({"grip"}),
            "xyz": np.array([0.1, 0.2, 1.08]), "cov": np.eye(3) * 1e-6, "status": "registered",
            "registration_step": 4, "top_z": k.Meas(1.08, 0.003, "kappa")}
    plain = k.PartRecord(**base, sigma_h_m=0.003)
    assert (plain.top_bound, plain.top_upper_z, plain.top_band, plain.axis_check) == (None,) * 4
    assert not plain.top_is_lower_bound()
    data = strict_json(plain.to_dict())
    assert {key: data[key] for key in ("top_bound", "top_upper_z", "top_band", "axis_check")} == dict.fromkeys(
        ("top_bound", "top_upper_z", "top_band", "axis_check"))
    band = {"n": np.int64(12), "stat_m": np.float32(0.001), "model_m": 0.003, "members": [1.079, 1.081]}
    check = {"image_axis": np.array([0.0, 2.0]), "region_axis": (0.05, 1.0), "agree_rad": np.float64(0.05),
             "accepted": np.bool_(True), "source": "image", "region_cells": 9}
    lower = k.PartRecord(**base, sigma_h_m=math.hypot(0.001, 0.003), top_bound="lower", top_upper_z=np.float64(1.2),
                         top_band=band, axis_check=check, elongation=((0.0, 1.0), 0.8))
    assert lower.top_is_lower_bound() and isinstance(lower.top_upper_z, float) and lower.is_elongated()
    assert lower.top_band == {"n": 12, "stat_m": pytest.approx(0.001), "model_m": 0.003, "members": [1.079, 1.081]}
    assert isinstance(lower.top_band["n"], int) and lower.axis_check["accepted"] is True
    assert np.allclose(lower.axis_check["image_axis"], (0.0, 1.0))                  # stored as unit vectors
    assert np.isclose(np.linalg.norm(lower.axis_check["region_axis"]), 1.0) and lower.axis_check["region_cells"] == 9
    data = strict_json(lower.to_dict())
    assert data["top_bound"] == "lower" and data["top_upper_z"] == 1.2 and data["top_band"]["n"] == 12
    assert data["axis_check"]["image_axis"] == [0.0, 1.0] and data["axis_check"]["source"] == "image"
    refused = {"image_axis": (1.0, 0.0), "region_axis": (0.0, 1.0), "agree_rad": math.pi / 2, "accepted": False,
               "source": k.AXIS_INCONSISTENT}
    assert not k.PartRecord(**base, sigma_h_m=0.003, axis_check=refused).is_elongated()
    unmeasured = {"image_axis": None, "region_axis": None, "agree_rad": None, "accepted": False, "source": "x"}
    assert k.PartRecord(**base, sigma_h_m=0.003, axis_check=unmeasured).axis_check["region_axis"] is None
    good_check = dict(check, accepted=True)
    for bad in (dict(top_bound="upper", top_upper_z=1.2),                         # TOP_BOUNDS
                dict(top_bound="lower"), dict(top_upper_z=1.2),                    # set and cleared together
                dict(top_bound="lower", top_upper_z=1.07),                         # upper below the lower bound
                dict(top_bound="lower", top_upper_z=float("inf")),
                dict(top_band={"n": 3, "stat_m": 0.001}),                          # missing key
                dict(top_band={"n": 2.5, "stat_m": 0.001, "model_m": 0.003}),
                dict(top_band={"n": 3, "stat_m": -0.001, "model_m": 0.003}),
                dict(top_band={"n": 3, "stat_m": 0.003, "model_m": 0.005}),        # sigma_h below hypot = 5.8 mm
                dict(top_band=[3, 0.001, 0.003]),
                dict(axis_check={key: good_check[key] for key in ("image_axis", "region_axis", "agree_rad",
                                                                  "accepted")}),   # no source
                dict(axis_check=dict(good_check, accepted=1)), dict(axis_check=dict(good_check, source="")),
                dict(axis_check=dict(good_check, image_axis=(0.0, 0.0))),
                dict(axis_check=dict(good_check, region_axis=(1.0, float("nan")))),
                dict(axis_check=dict(good_check, agree_rad=-0.1)),
                dict(axis_check=dict(good_check, agree_rad=2.0, accepted=False)),   # axis angles lie in [0, pi/2]
                dict(axis_check=dict(good_check, agree_rad=k.AXIS_AGREE_MAX_RAD + 1e-3)),   # accepted beyond 7.5 deg
                dict(axis_check=dict(good_check, region_axis=None)),               # accepted without both axes
                dict(axis_check=dict(good_check, source=k.AXIS_INCONSISTENT)),     # a refusal cannot be accepted
                dict(axis_check=refused, elongation=((1.0, 0.0), 0.8))):           # refused axis, elongation kept
        with pytest.raises(ValueError):
            k.PartRecord(**{**base, "sigma_h_m": 0.004, **bad})
    with pytest.raises(ValueError):                                                # top_band needs top_z
        k.PartRecord(**{**base, "top_z": None}, sigma_h_m=0.004, top_band={"n": 3, "stat_m": 0.001, "model_m": 0.003})
    assert k.PartRecord("D", "t", "fixed", frozenset(), top_bound="lower", top_upper_z=1.1).top_is_lower_bound()
    with pytest.raises(ValueError):                                                # replace re-runs the checks
        dataclasses.replace(lower, top_bound=None)
    cleared = dataclasses.replace(lower, top_bound=None, top_upper_z=None)         # e.g. scene release_held
    assert not cleared.top_is_lower_bound() and cleared.top_band == lower.top_band
    assert k.AXIS_AGREE_MAX_RAD == pytest.approx(math.radians(7.5)) == pytest.approx(k.YAW_GRID_RAD / 2)
    assert k.TOP_BOUNDS == ("lower",) and k.TOP_BAND_KEYS == ("n", "stat_m", "model_m")
    assert k.AXIS_CHECK_KEYS == ("image_axis", "region_axis", "agree_rad", "accepted", "source")


def test_review1_registry_additions():
    assert k.TYPES_VERSION.startswith("agent-speedup-v2-types-")                  # pinned in the review-2 tests
    assert "grip_unstable" in k.RELATION_REASONS["held"]                          # O-1 tilt during the lift
    assert "transit_contact" in k.RELATION_REASONS["held"] and "transit_contact" in k.RELATION_REASONS["registered"]
    assert {"axis_inconsistent", "misaligned", "lower_bound_top"} <= set(k.EVIDENCE_FLAGS)
    assert k.AXIS_INCONSISTENT == "axis_inconsistent"
    for reason in ("grip_unstable", "transit_contact"):                            # failure reasons, never success
        assert not any(reason in codes for codes in k.SUCCESS_REASONS.values())
        k.RelationOutcome(relation="held", status="failed", reason=reason, pids={}, evidence={}, thresholds=[],
                          evidence_text="", local_retries=0, steps=1)
    dense = k.FeatureRecord(7, k.DENSE_FEATURE_KEY_PREFIX + "3", (1.0, 2.0), None, 4, None, None, None, None, None,
                            False, None)
    assert dense.is_dense_lattice() and dense.key == "dense:3"
    for key in (None, "C", k.RELIEF_FEATURE_KEY_PREFIX + "C:0"):
        assert not k.FeatureRecord(8, key, (1.0, 2.0), None, 4, None, None, None, None, None, False,
                                   None).is_dense_lattice()


# ------------------------------------------------------------------- design review 2 (decision §2.1 K-4, K-5)
def unit_deg(deg):
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def registered_c(**extra):
    base = {"pid": "C", "text": "t", "move_prior": "free", "roles": frozenset({"grip"}),
            "xyz": np.array([0.1, 0.2, 1.03]), "cov": np.eye(3) * 1e-6, "status": "registered",
            "registration_step": 6, "top_z": k.Meas(1.03, 0.003, "kappa"), "sigma_h_m": 0.003}
    return k.PartRecord(**{**base, **extra})


def oe_check(**extra):
    """A D10-OE (method outline_edges) check accepting an outline axis at 2 deg against a hull edge axis at 0 deg."""
    check = {"image_axis": unit_deg(2.0), "region_axis": (2.0, 0.0), "agree_rad": math.radians(2.0),
             "accepted": True, "source": "outline_edges", "method": "outline_edges", "region_collapsed": False,
             "views": {"axes": [], "disagree_rad": None}, "reason": None}
    return {**check, **extra}


def test_review2_registry_additions():
    assert k.TYPES_VERSION == "agent-speedup-v2-types-4"          # types-4: decision J §3 + gap round §2.1
    assert k.AXIS_CHECK_KEYS == ("image_axis", "region_axis", "agree_rad", "accepted", "source")   # kept (K-4)
    assert k.AXIS_CHECK_EXTRA_KEYS == ("method", "region_collapsed", "views", "reason")
    assert k.AXIS_CHECK_METHODS == ("outline_edges",) and k.NOT_ELONGATED == "not_elongated"
    assert k.AXIS_CHECK_REASONS == ("outline_gate", "no_edge_agreement", "views_disagree", "kept_earlier")
    assert k.AXIS_REFUSAL_SOURCES == {"outline_gate": "not_elongated", "no_edge_agreement": k.AXIS_INCONSISTENT,
                                      "views_disagree": k.AXIS_INCONSISTENT}
    assert k.AXIS_KEPT_EARLIER == "kept_earlier" and k.AXIS_VIEWS_KEYS == ("axes", "disagree_rad")
    assert "region_collapsed" in k.EVIDENCE_FLAGS and not set(k.EVIDENCE_FLAGS) & set(k.REASONS)
    vocab, known = set(vocabulary()), set(th.LINT_ALLOWLIST)
    for word in (*k.AXIS_CHECK_EXTRA_KEYS, *k.AXIS_CHECK_METHODS, *k.AXIS_CHECK_REASONS, *k.AXIS_VIEWS_KEYS,
                 k.NOT_ELONGATED):
        assert not set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(word))), word
    # axis angle mod pi in [0, pi/2], sign- and length-free
    assert k.axis_angle_rad((1.0, 0.0), (0.0, -3.0)) == pytest.approx(math.pi / 2)
    assert k.axis_angle_rad(unit_deg(10.0), (-2.0, 0.0)) == pytest.approx(math.radians(10.0))
    assert k.axis_angle_rad(unit_deg(179.0), unit_deg(1.0)) == pytest.approx(math.radians(2.0))
    assert k.axis_angle_rad(unit_deg(30.0), unit_deg(30.0)) == pytest.approx(0.0, abs=1e-7)
    for bad in (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), None), ((1.0, 0.0, 0.0), (1.0, 0.0))):
        with pytest.raises(ValueError):
            k.axis_angle_rad(*bad)


# ------------------------------------------- grasp-refusal decision J §3 + gap round judge decision §2.1 (types-4)
J_REASONS = ("early_finger_contact", "span_inconsistent", "pose_out_of_envelope")
GAP_FLAGS = ("external_template_on_robot", "template_on_robot", "seed_band_fallback", "kept_whole", "tight_fit",
             "lateral_unverified", "post_release_unverified", "marginal_refuted")


def failed_held(reason, **extra):
    return k.RelationOutcome(**{**dict(relation="held", status="failed", reason=reason, pids={}, evidence={},
                                       thresholds=[], evidence_text="", local_retries=1, steps=3), **extra})


def test_decision_j_reasons_are_registered_held_failures():
    for reason in J_REASONS:
        assert reason in k.REASONS and k.REASONS[reason].strip(), reason
        assert reason in k.RELATION_REASONS["held"], reason
        assert not any(reason in codes for codes in k.SUCCESS_REASONS.values()), reason
        assert failed_held(reason).reason == reason                               # positive: a held failure
        with pytest.raises(ValueError):                                           # negative: never a success
            failed_held(reason, status="succeeded")
    # GC-1's two abort reasons are primitive reasons; PC-1 refuses before any close primitive runs
    assert {"early_finger_contact", "span_inconsistent"} <= set(k.PRIMITIVE_REASONS)
    assert "pose_out_of_envelope" not in k.PRIMITIVE_REASONS
    # the relation layer's outcome code mapping keeps them (decision §2.1: no longer no_feasible_candidate)
    from gpt_field import agent_speedup_v2_relations as rel
    for reason in J_REASONS:
        assert rel._reason_code(reason, "no_feasible_candidate") == reason
    assert rel._reason_code("not_a_reason", "no_feasible_candidate") == "no_feasible_candidate"


def test_gap_round_registries_list_every_entry():
    """Every gap-round registry entry, listed (judge decision §2.1): evidence flags, the observer cause, the
    part_not_found sub-reasons, the repair effect codes and the recovery_exhausted event."""
    assert k.TYPES_VERSION == "agent-speedup-v2-types-4"
    assert set(GAP_FLAGS) <= set(k.EVIDENCE_FLAGS) and len(set(k.EVIDENCE_FLAGS)) == len(k.EVIDENCE_FLAGS)
    assert k.OBSERVER_CAUSES == ("external_unverified",)
    assert k.PART_NOT_FOUND_WHY == ("band_on_support", "region_of_other_object", "far_from_object",
                                    "larger_than_object")
    assert k.REPAIR_EFFECTS == ("no_change", "grip_order_changed", "reregistered", "replanned", "regrasp", "skipped")
    assert "recovery_exhausted" in k.EVENTS and "recovery_exhausted" not in k.NOMINAL_EVENTS
    assert "skipped" in k.STEP_STATUSES and "stays" in k.STEP_STATUSES        # a forced skip is skipped, never stays
    # negatives: evidence vocabulary is never an outcome reason, a primitive reason or an event
    vocabulary_codes = set(GAP_FLAGS) | set(k.OBSERVER_CAUSES) | set(k.PART_NOT_FOUND_WHY[1:]) | {"recovery_exhausted"}
    assert not vocabulary_codes & set(k.REASONS) and not vocabulary_codes & set(k.PRIMITIVE_REASONS)
    assert not (set(GAP_FLAGS) | set(k.OBSERVER_CAUSES)) & set(k.EVENTS)
    assert not set(k.REPAIR_EFFECTS) & set(k.REASONS) and "part_not_found" in k.REASONS
    for registry in (k.OBSERVER_CAUSES, k.PART_NOT_FOUND_WHY, k.REPAIR_EFFECTS):
        assert len(set(registry)) == len(registry)
    vocab, known = set(vocabulary()), set(th.LINT_ALLOWLIST)
    for word in (*GAP_FLAGS, *k.OBSERVER_CAUSES, *k.PART_NOT_FOUND_WHY, *k.REPAIR_EFFECTS, "recovery_exhausted",
                 *J_REASONS, *(k.REASONS[r] for r in J_REASONS)):
        assert not set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(word))), word


def test_kernel_object_heads_and_rim_bound():
    """Moved to the kernel in the gap round (the scene's part-object rules and annulus model read them)."""
    assert k.object_words("the toy car: front wheel") == ("toy", "car")
    assert k.object_words("A Toy  car:wheel") == ("toy", "car") and k.ARTICLES == ("a", "an", "the")
    assert k.object_words("no colon here") is None and k.object_words("the: x") is None      # negatives
    assert k.same_object(("toy", "car"), ("small", "toy", "car")) and k.same_object(None, ("x",))
    assert not k.same_object(("toy", "car"), ("music", "box"))
    assert k.RIM_WALL_RADIUS_FRACTION == 0.5
    assert th.CONSTANT_LABELS["agent_speedup_v2_types.RIM_WALL_RADIUS_FRACTION"]["label"] == "DD"
    from gpt_field import agent_speedup_v2_plan as plan
    assert plan.object_words is k.object_words and plan.same_object is k.same_object


def test_axis_check_d10oe_schema():
    """K-4: the required keys keep their names with D10-OE semantics (image_axis = outline axis, region_axis = the
    agreeing edge axis, agree_rad = their angle); the documented extra keys method, region_collapsed, views and
    reason are validated and kept (unknown extra keys too); a refusal reason, "views_disagree" included, is never
    accepted; accepted still needs agree_rad <= AXIS_AGREE_MAX_RAD."""
    import dataclasses
    rec = registered_c(axis_check=oe_check(views={"axes": [(3.0, 0.0)], "disagree_rad": np.float64(0.03)},
                                           region_collapsed=np.bool_(False), outline_share=0.41,
                                           edges={"hull": {"share": 0.3}}),
                       elongation=(unit_deg(2.0), 0.41))
    check = rec.axis_check
    assert rec.is_elongated() and check["accepted"] is True and check["method"] == "outline_edges"
    assert check["region_collapsed"] is False and check["reason"] is None
    assert check["outline_share"] == 0.41 and check["edges"] == {"hull": {"share": 0.3}}          # extra keys kept
    assert np.allclose(check["views"]["axes"][0], (1.0, 0.0)) and check["views"]["disagree_rad"] == 0.03
    assert np.isclose(np.linalg.norm(check["region_axis"]), 1.0)
    data = strict_json(rec.to_dict())["axis_check"]
    assert data["views"] == {"axes": [[1.0, 0.0]], "disagree_rad": 0.03} and data["method"] == "outline_edges"
    assert data["region_collapsed"] is False and data["reason"] is None
    # refusals: the outline gate failed (not elongated; P-C's harmonic may run), no agreeing edge, views disagree
    gate = registered_c(axis_check=oe_check(accepted=False, source=k.NOT_ELONGATED, reason="outline_gate",
                                            region_axis=None, agree_rad=None))
    edge = registered_c(axis_check=oe_check(accepted=False, source=k.AXIS_INCONSISTENT, reason="no_edge_agreement",
                                            region_axis=unit_deg(20.0), agree_rad=math.radians(18.0)))
    nine = {"axes": [unit_deg(11.0)], "disagree_rad": math.radians(9.0)}           # an earlier view 9 deg away
    views = registered_c(axis_check=oe_check(accepted=False, source=k.AXIS_INCONSISTENT, reason="views_disagree",
                                             views=nine))
    for refused in (gate, edge, views):
        assert not refused.is_elongated() and refused.axis_check["accepted"] is False
    assert views.axis_check["agree_rad"] == pytest.approx(math.radians(2.0))   # this view's own evidence stays
    with pytest.raises(ValueError):                              # "views_disagree" is never accepted
        registered_c(axis_check=oe_check(source="outline_edges", reason="views_disagree", views=nine))
    with pytest.raises(ValueError):                              # nor carried as an elongation
        registered_c(axis_check=oe_check(accepted=False, source=k.AXIS_INCONSISTENT, reason="views_disagree",
                                         views=nine), elongation=(unit_deg(2.0), 0.4))
    with pytest.raises(ValueError):                              # S-12: an acceptance against a 9 deg memory
        registered_c(axis_check=oe_check(views=nine))
    bad = [dict(reason="outline_gate", accepted=False, source=k.AXIS_INCONSISTENT),      # labels per reason
           dict(reason="no_edge_agreement", accepted=False, source=k.NOT_ELONGATED),
           dict(reason="views_disagree", accepted=False, source=k.NOT_ELONGATED, views=nine),
           dict(reason="no_edge_agreement"),                                            # a refusal reason, accepted
           dict(reason="outline_gate"),
           dict(reason="views_disagree", accepted=False, source=k.AXIS_INCONSISTENT),   # no remembered axis
           dict(reason="views_disagree", accepted=False, source=k.AXIS_INCONSISTENT, views=None),
           dict(reason="axis_disagree", accepted=False, source=k.AXIS_INCONSISTENT),    # S-4' reason under D10-OE
           dict(reason=""), dict(reason=3),
           dict(method="s4_prime"), dict(method=""),
           dict(agree_rad=k.AXIS_AGREE_MAX_RAD + 1e-3),                                 # accepted beyond 7.5 deg
           dict(region_axis=None), dict(agree_rad=None),
           dict(source=k.NOT_ELONGATED), dict(source=k.AXIS_INCONSISTENT),             # accepted, refusal source
           dict(region_collapsed=True),                                                 # S-13: a collapsed axis
           dict(region_collapsed=1), dict(region_collapsed="no"),
           dict(views={"axes": [(1.0, 0.0)]}), dict(views={"disagree_rad": None}),      # views schema
           dict(views={"axes": [(0.0, 0.0)], "disagree_rad": None}),
           dict(views={"axes": [(1.0, float("nan"))], "disagree_rad": None}),
           dict(views={"axes": [None], "disagree_rad": None}),
           dict(views={"axes": "xy", "disagree_rad": None}), dict(views={"axes": 3, "disagree_rad": None}),
           dict(views={"axes": [], "disagree_rad": 2.0}), dict(views={"axes": [], "disagree_rad": -0.1}),
           dict(views=[(1.0, 0.0)])]
    for extra in bad:
        with pytest.raises(ValueError):
            registered_c(axis_check=oe_check(**extra))
    with pytest.raises(ValueError):                              # replace re-runs the checks
        dataclasses.replace(views, axis_check=dict(views.axis_check, accepted=True, source="outline_edges"))
    # optional keys: absent or None means not evaluated / no memory
    plain = registered_c(axis_check={key: oe_check()[key] for key in k.AXIS_CHECK_KEYS + ("method",)},
                         elongation=(unit_deg(2.0), 0.4))
    assert plain.is_elongated() and "views" not in plain.axis_check
    assert registered_c(axis_check=oe_check(views=None, region_collapsed=None)).axis_check["views"] is None
    # a pre-D10-OE (S-4') check keeps its own reasons: method absent
    legacy = registered_c(axis_check={"image_axis": (1.0, 0.0), "region_axis": (0.0, 1.0), "agree_rad": math.pi / 2,
                                      "accepted": False, "source": k.AXIS_INCONSISTENT, "reason": "axis_disagree",
                                      "cells": 9})
    assert legacy.axis_check["reason"] == "axis_disagree" and legacy.axis_check["cells"] == 9


def test_axis_check_kept_earlier_and_collapse():
    """K-4 with S-12 / S-13: "kept_earlier" is an accepted check carrying the remembered axis (its own view's
    evidence is not gated: the outline gate or the edges refused it, or its region collapsed); it needs a remembered
    axis, and the stored elongation agrees with every remembered axis within AXIS_AGREE_MAX_RAD."""
    memory = {"axes": [unit_deg(31.0), unit_deg(33.0)], "disagree_rad": math.radians(14.0), "conflicted": False}
    kept = registered_c(axis_check=oe_check(image_axis=unit_deg(45.0), region_axis=unit_deg(70.0),
                                            agree_rad=math.radians(25.0), source=k.AXIS_KEPT_EARLIER,
                                            reason="kept_earlier", views=memory),
                        elongation=(unit_deg(33.0), 0.35))
    assert kept.is_elongated() and kept.axis_check["accepted"] and kept.axis_check["views"]["conflicted"] is False
    assert np.allclose(kept.elongation[0], unit_deg(33.0))
    collapsed_kept = registered_c(axis_check=oe_check(region_collapsed=True, reason="kept_earlier", views=memory,
                                                      image_axis=None, region_axis=None, agree_rad=None),
                                  elongation=(unit_deg(32.0), 0.35), width_by_yaw=())
    assert collapsed_kept.is_elongated() and collapsed_kept.width_at(0.0) is None
    for extra, elongation in ((dict(reason="kept_earlier", accepted=False, source=k.AXIS_INCONSISTENT, views=memory),
                               None),                                        # kept_earlier is an accepted check
                              (dict(reason="kept_earlier", views={"axes": [], "disagree_rad": None}),
                               (unit_deg(2.0), 0.4)),                        # nothing remembered to carry
                              (dict(reason="kept_earlier", views=None), (unit_deg(2.0), 0.4)),
                              (dict(reason="kept_earlier", views=memory), (unit_deg(42.0), 0.4)),   # 9 deg off
                              (dict(reason="kept_earlier", source=k.AXIS_INCONSISTENT, views=memory),
                               (unit_deg(32.0), 0.4))):
        with pytest.raises(ValueError):
            registered_c(axis_check=oe_check(**extra), elongation=elongation)
    # a collapsed region with no memory: the axis is refused, the width unmeasured (K-5)
    collapsed = registered_c(axis_check=oe_check(accepted=False, source=k.AXIS_INCONSISTENT, region_collapsed=True,
                                                 reason="no_edge_agreement"), width_by_yaw=())
    assert not collapsed.is_elongated() and collapsed.axis_check["region_collapsed"] is True
    reasonless = registered_c(axis_check=oe_check(accepted=False, source="collapsed", region_collapsed=True))
    assert reasonless.axis_check["reason"] is None and not reasonless.is_elongated()


def test_registered_record_may_carry_unmeasured_width():
    """K-5: a registered, non-provisional PartRecord whose region collapsed carries width_by_yaw = () ("width
    unmeasured"): it constructs (also through dataclasses.replace of a measured record), width_at answers None at
    every yaw (so the planner's P2 answers geometry_unmeasured), and to_dict keeps [] apart from a seeded record's
    None."""
    import dataclasses
    widths = tuple((math.radians(15.0 * i), k.Meas(0.016, 0.001, "kappa")) for i in range(12))
    measured = registered_c(width_by_yaw=widths)
    assert measured.width_at(0.3) is not None
    collapsed = dataclasses.replace(measured, width_by_yaw=(), axis_check=oe_check(
        accepted=False, source=k.AXIS_INCONSISTENT, region_collapsed=True, reason="no_edge_agreement"))
    assert collapsed.status == "registered" and not collapsed.provisional and collapsed.registration_step == 6
    assert collapsed.width_by_yaw == () and collapsed.xyz is not None and collapsed.top_z is not None
    assert all(collapsed.width_at(math.radians(deg)) is None for deg in range(-180, 181, 15))
    assert not collapsed.is_elongated()
    data = strict_json(collapsed.to_dict())
    assert data["width_by_yaw"] == [] and data["axis_check"]["region_collapsed"] is True
    seeded = k.PartRecord("C", "t", "free", frozenset({"grip"}))
    assert seeded.width_by_yaw is None and strict_json(seeded.to_dict())["width_by_yaw"] is None
    assert registered_c(width_by_yaw=()).width_at(1.0) is None                       # without an axis_check too


def test_held_frame_geometry_and_dict():
    state = T2_START
    pose = k.HandPose.from_state8(state)
    held = k.HeldFrame("D", "C", tuple(state), np.array([0.08, 0.0, 0.0]), np.zeros((5, 2), int), 0.004,
                       k.Meas(0.12, 0.002, "touch"), 0.05, None, None, True, np.ones((8, 8), bool), Summarized(),
                       14.0, None, k.Meas(0.9, 0.001, "plane"), 1.02)
    assert np.allclose(held.object_ref_point(pose), pose.grip + pose.R @ np.array([0.08, 0.0, 0.0]))
    assert np.allclose(held.object_ref_point(pose)[:2] - pose.grip[:2], 0.08 * pose.up[:2])
    assert held.lowest_point_z(pose) == pytest.approx(state[2] - 0.12)
    data = strict_json(held.to_dict())
    assert data["silhouette_px"] == 64 and data["vacated_cells"] == 5 and data["touchdown"] is None


def test_calibration_and_survey_summaries():
    feature = k.FeatureRecord(1, "A", (1.0, 2.0), (3.0, 4.0), 5, 900.0, 9.0, 0.37, 0.004, (0.0, 0.0, 0.9), True,
                              None)
    rejected = k.FeatureRecord(2, None, (5.0, 6.0), None, 2, None, None, None, None, None, False, "lost")
    with pytest.raises(ValueError):
        k.FeatureRecord(3, None, (0, 0), None, 0, None, None, None, None, None, False, "bad")
    survey = k.SurveyResult(tuple(T2_START), tuple(T2_START), {"probe": 12}, [feature, rejected],
                            [k.DepthSample((0, 0, 0.9), 0.002, 0.001, "kappa", None, 1, 3, 900.0)],
                            {"f_hat": k.Meas(333.0, 3.0, "loom")}, [{"stage": "x"}], {"open": np.zeros((4, 4))})
    summary = strict_json(survey.summary())
    assert summary["rejects_by_reason"] == {"lost": 1} and summary["valid_features"] == 1
    assert summary["f_hat"]["value"] == 333.0 and summary["frames"] == ["open"]
    record = k.CalibrationRecord(GripperCalibration.default(), ((77.0, 50.0),), (2.5, 2.4), {"commanded_rad": 0.1},
                                 {"gaps_mm": [77.4]}, ({"stage": "x"},), (), (), 190)
    strict_json(record.to_dict())
    cal = k.V2Calibration(GripperCalibration.default(), 0.05, 0.002, 0.0229, Summarized(), None, None, None,
                          np.zeros((512, 512), bool), {"external": np.full((4, 4), 0.001)}, {"wrist": 1.5}, 1.2, 2,
                          tuple(T2_START), tuple(_rotvec_quaternion(np.array(T2_START[3:6]))), survey,
                          robot.mount_decision(T2_START).to_dict(),
                          [th.Threshold("G5_object_present_gap_mm", 0.5, "DD", "max(...)")])
    data = strict_json(cal.summary())
    assert data["camera"]["radius"] == 10 and data["external"] is None and data["quarter_turns"] == 2
    assert data["thresholds"][0]["label"] == "DD" and data["fingertip_band_px"] == 0


# --------------------------------------------------------------------------------------- 5. threshold helpers
def test_th_max_and_th_value():
    t = k.th_max("G5_object_present_gap_mm", k.Term("DD", 0.5, "G5_FLOOR_MM"), k.Term("ONLINE", 0.12, "4*sigma_close"),
                 unit="mm")
    assert t.value == 0.5 and t.label == "DD" and t.name == "G5_object_present_gap_mm"
    assert t.derivation == "max(G5_FLOOR_MM=0.5, 4*sigma_close=0.12) = 0.5 mm; winner G5_FLOOR_MM [DD]"
    t = k.th_max("G5_object_present_gap_mm", k.Term("DD", 0.5, "G5_FLOOR_MM"), k.Term("ONLINE", 0.8, "4*sigma_close"),
                 unit="mm")
    assert t.value == 0.8 and t.label == "ONLINE" and "G5_FLOOR_MM=0.5" in t.derivation and "0.8" in t.derivation
    low = k.th_min("G12_stall_ratio", k.Term("DD", 0.4, "G12_MAX"), k.Term("ONLINE", 0.33, "0.6*free"), unit="ratio")
    assert low.value == 0.33 and low.label == "ONLINE"
    v = k.th_value("M8_tol_h_m", 0.012, "ONLINE", "max(M8_FLOOR_M, 3*sqrt(...))", sigma_dest=0.002, sigma_b=0.003)
    assert v.derivation == "max(M8_FLOOR_M, 3*sqrt(...)) with sigma_dest=0.002, sigma_b=0.003"
    assert k.th_value("X_a_m", 1.0, "DD", "formula").derivation == "formula"
    for bad in (lambda: k.th_max("x", unit="mm"), lambda: k.Term("TUNED", 1.0, "x"),
                lambda: k.Term("DD", float("nan"), "x"), lambda: k.Term("DD", 1.0, " "),
                lambda: k.th_value("x", 1.0, "DD", "")):
        with pytest.raises(ValueError):
            bad()
    dicts = k.thresholds_dicts([t, {"name": "n", "value": 1.0, "label": "DD", "derivation": "d"}])
    assert dicts[0]["label"] == "ONLINE" and dicts[1]["name"] == "n"


def test_relation_outcome_record():
    thresholds = [k.th_max("G6_hold_floor_mm", k.Term("DD", 0.5, "G6_FLOOR_MM"), unit="mm")]
    outcome = k.RelationOutcome("held", "succeeded", "hold_pass", {"obj": "D", "grip": "C", "ref": None},
                                {"gap_mm": np.float32(14.1), "grip": np.array([0.1, float("nan"), 1.0])}, thresholds,
                                "part C held, gap 1.41 cm", 0, 120)
    record = outcome.record(4, 300, 1)
    assert set(record) == {"n", "step", "skill", "relation", "status", "reason", "evidence", "thresholds",
                           "evidence_text", "pids", "plan_step", "local_retries", "steps", "calls", "model_claims"}
    assert record["skill"] == record["relation"] == "held" and record["thresholds"][0]["name"] == "G6_hold_floor_mm"
    assert record["evidence"]["grip"][1] is None
    strict_json(record)
    assert outcome.repair_payload() == {"reason": "hold_pass", "evidence_text": "part C held, gap 1.41 cm",
                                        "thresholds": [["G6_hold_floor_mm", 0.5]]}
    k.RepairNeed("repair", 2, {}, last_outcome=outcome.repair_payload())
    good = dict(relation="turned", status="failed", reason="turn_slip", pids={}, evidence={}, thresholds=[],
                evidence_text="", local_retries=2, steps=40)
    for bad in (dict(status="ok"), dict(reason="slipped"), dict(evidence_text="x" * 201), dict(thresholds=[{"a": 1}]),
                dict(local_retries=-1), dict(steps=1.5), dict(relation=""),
                dict(relation="anything"), dict(relation="grasp"), dict(relation="turn"),      # B0-R9
                dict(status="succeeded", reason="invalid_plan"),                  # a failure reason cannot succeed
                dict(relation="held", status="succeeded", reason="placed"),       # another relation's success
                dict(status="failed", reason="hold_pass"), dict(status="rejected", reason="turned_to_stop"),
                dict(relation="slide", status="succeeded", reason="relation_not_implemented"),
                dict(relation="plan", status="succeeded", reason="invalid_plan")):
        with pytest.raises(ValueError):
            k.RelationOutcome(**{**good, **bad})
    for fine in (dict(relation="held", reason="geometry_unmeasured"),             # a planner Infeasible reason
                 dict(relation="turned", reason="hold_lost"),                     # a nested grasp failure
                 dict(relation="supported_on", reason="keepout_contact"),         # a keeps reason
                 dict(relation="slide", reason="relation_not_implemented"),
                 dict(relation="place_in (reserved)", reason="relation_not_implemented"),
                 dict(relation="plan", status="rejected", reason="invalid_plan"),
                 dict(relation="keeps", reason="kept_relation_lost"),
                 dict(relation="calibrated", status="succeeded", reason="calibration_degraded"),
                 dict(relation="calibrated", status="failed", reason="calibration_degraded"),
                 dict(relation="turned", status="succeeded", reason="turned_by_effect")):
        assert k.RelationOutcome(**{**good, **fine}).reason == fine["reason"]
    assert set(k.SUCCESS_REASONS) == set(k.RELATIONS)
    for relation, codes in k.SUCCESS_REASONS.items():
        assert set(codes) <= set(k.RELATION_REASONS[relation]), relation
    assert set(k.RELATIONS) <= set(k.OUTCOME_RELATIONS) and set(k.RESERVED_DO_KINDS) == {"slide", "place_in",
                                                                                          "push_to"}
    report = k.ObsReport("wedge", "met", np.float64(0.5), "free_sense", {"rates": np.array([0.1, 0.2])},
                         thresholds, 12)
    strict_json(report.to_dict())
    with pytest.raises(ValueError):
        k.ObsReport("wedge", "unknown", None, None, {}, (), 1)


# ------------------------------------------------------------------------------------------ 6. registries
def test_registries_pass_vocabulary_and_are_unique():
    vocab = set(vocabulary())
    known = set(th.LINT_ALLOWLIST)
    words = [*k.REASONS, *k.EVENTS, *k.EVIDENCE_FLAGS, *k.PRIMITIVE_REASONS, *k.RELATIONS, *k.ROLES, *k.DO_KINDS,
             *k.HF_SOURCES, *k.MEAS_SOURCES, *k.SEED_SOURCES, *k.POINT_PURPOSES, *k.REPAIR_ACTIONS]
    for word in words:
        hits = set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(word)))
        assert not hits, (word, hits)
    for meaning in k.REASONS.values():
        assert not set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(meaning))), meaning
    for registry in (k.EVENTS, k.NOMINAL_EVENTS, k.EVIDENCE_FLAGS, k.PRIMITIVE_REASONS, k.PIDS, k.HF_SOURCES):
        assert len(set(registry)) == len(registry)
    source = Path(k.__file__).read_text()
    tree = ast.parse(source)
    reasons_node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(getattr(t, "id", None) == "REASONS" for t in node.targets))
    literal_keys = [key.value for key in reasons_node.keys]
    assert len(literal_keys) == len(set(literal_keys)) == len(k.REASONS)      # no silently repeated key
    for relation, codes in k.RELATION_REASONS.items():
        assert len(set(codes)) == len(codes) and set(codes) <= set(k.REASONS), relation
    assert set().union(*map(set, k.RELATION_REASONS.values())) == set(k.REASONS)
    assert set(k.NOMINAL_EVENTS) <= set(k.EVENTS)
    assert not set(k.EVIDENCE_FLAGS) & set(k.REASONS)                          # flags are never outcome reasons
    assert k.HF_SOURCES[0] == "none" and set(k.HF_RANK) == set(k.HF_SOURCES)
    assert all(k.HF_RANK[fill] < min(k.HF_RANK[m] for m in k.DEPTH_SOURCES) for fill in k.HF_FILLS)
    assert k.YAW_GRID_RAD == pytest.approx(math.radians(15)) and k.NEED_KINDS == (
        "plan", "replan", "point_fine", "point_coarse", "repair", "review")
    from gpt_field.mark_prompting import FINGERTIP_EDGE
    assert set(k.FINGERTIP_EDGES) == set(FINGERTIP_EDGE.values())


def test_protocols_are_structural():
    class Obs:
        name = "x"

        def start(self, view):
            return None

        def update(self, before, action, after, context=None):
            return None

    class Sink:
        def emit(self, event, **payload):
            return None
    assert isinstance(Obs(), k.Observer) and isinstance(Sink(), k.TraceSink) and not isinstance(Sink(), k.Observer)


# ------------------------------------------------------------------------------------------- 7. kernel lint
def test_kernel_passes_lint_and_literal_audit():
    path = Path(k.__file__)
    assert th.lint_module(path, vocabulary()) == []
    assert th.lint_module(path, vocabulary(), labels=th.CONSTANT_LABELS) == []
    assert th.literal_audit(path) == []
    assert "agent_speedup_v2_types.py" in th.CONTROL_MODULES
    labels = th.module_labels(path)
    assert labels and all(key.startswith("agent_speedup_v2_types.") for key in labels)
    assert all(th.CONSTANT_LABELS[key] == entry for key, entry in labels.items())


def imported_modules(tree):
    """Module stems an AST imports (relative and package imports resolved inside gpt_field)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {alias.name.split(".")[-1] if alias.name.startswith("gpt_field.") else alias.name
                    for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.startswith("gpt_field") or not module:
                stem = module.split(".")[-1] if module and module != "gpt_field" else ""
                if stem:
                    out.add(stem)
                else:
                    out |= {alias.name for alias in node.names}
            else:
                out.add(module)
    return out


def test_kernel_imports_no_v2_module():
    source = Path(k.__file__).read_text()
    imported = imported_modules(ast.parse(source))
    v2 = {name for name in imported if name.startswith("agent_speedup_v2")}
    assert v2 == {"agent_speedup_v2_robot", "agent_speedup_v2_thresholds"}, v2
    assert not any(name.startswith("agent_speedup_v1") for name in imported)
    assert {"contact_skills", "action", "wrist_servo"} >= {name for name in imported
                                                           if name not in v2 and name != "re"   # re: object_words
                                                           and not name.startswith(("numpy", "math",
                                                           "collections", "dataclasses", "typing", "__future__"))}
    assert "TYPE_CHECKING" not in source
