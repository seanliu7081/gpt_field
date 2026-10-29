"""BH (contract §C14.2, §C14.5): SIM-EXT oracle op, the ORACLE stub and the relation-smoke harness, without a
simulator or a model. The real-simulator proof is ``--scenario oracle_selfcheck`` (scripts/agent_speedup_v2_
relation_smoke.py); its report is checked here when present."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_robot as robot
from gpt_field import agent_speedup_v2_sim as v2sim
from gpt_field import mark_prompting
from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config, resolve_v2_task
from gpt_field.contact_skills import rotation_matrix
from gpt_field.policy import Decision

from agent_speedup_v2_fakes import FakeV2Sim

ROOT = Path(__file__).resolve().parents[1]
T2 = resolve_v2_task("libero_10_t2")
T2_START_STATE8 = (-0.2014, 0.0187, 1.1893, 3.1397, 0.0067, -0.0861, 0.0387, -0.0387)


def _load(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load("agent_speedup_v2_oracle", "scripts/agent_speedup_v2_oracle.py")
smoke = _load("agent_speedup_v2_relation_smoke", "scripts/agent_speedup_v2_relation_smoke.py")


# =========================================================================================================
# SIM-EXT: options, refusal, dispatch, metadata
def test_monitor_options_optional_oracle_key():
    assert v2sim.DEFAULT_MONITOR_OPTIONS == {"goal_aware": True, "truth_window": True}
    assert v2sim.validate_monitor_options(None) == {"goal_aware": True, "truth_window": True}
    assert "oracle" not in v2sim.validate_monitor_options({"goal_aware": False})
    assert v2sim.validate_monitor_options({"oracle": True})["oracle"] is True
    assert v2sim.validate_monitor_options({"oracle": False})["oracle"] is False
    for bad in ({"oracle": 1}, {"oracle": "yes"}, {"oracle": None}, {"unknown": True}):
        with pytest.raises(ValueError):
            v2sim.validate_monitor_options(bad)
    assert v2sim.EVALUATOR_OPS == ("truth", "settle_truth", "answer_key", "geometry", "oracle")
    config, monitor = build_v2_sim_config("libero_10_t2")
    command = v2sim.worker_command(config, {**monitor, "oracle": True})
    assert json.loads(command[command.index("--monitor-json") + 1])["oracle"] is True
    assert "oracle" not in json.loads(v2sim.worker_command(config, monitor)[-1])


def test_worker_refuses_oracle_by_default():
    worker = object.__new__(v2sim.V2LiberoWorker)
    with pytest.raises(PermissionError, match="oracle op disabled"):
        worker.oracle({"kind": "joints", "names": ["x"]})
    worker.oracle_enabled = False
    with pytest.raises(PermissionError, match="oracle op disabled"):
        worker.oracle({"kind": "joints", "names": ["x"]})


def test_worker_metadata_records_oracle_flag(monkeypatch):
    def fake_base_init(self, config):
        self.config, self.metadata, self.env, self.status = config, {}, None, {}
    monkeypatch.setattr(v2sim._LiberoWorker, "__init__", fake_base_init)
    config, monitor = build_v2_sim_config("libero_10_t2")
    plain = v2sim.V2LiberoWorker(config, monitor)
    assert plain.metadata["oracle_enabled"] is False and plain.oracle_enabled is False
    assert plain.metadata["monitor_options"] == monitor and "oracle" in plain.metadata["evaluator_ops"]
    enabled = v2sim.V2LiberoWorker(config, {**monitor, "oracle": True})
    assert enabled.metadata["oracle_enabled"] is True and enabled.metadata["monitor_options"]["oracle"] is True
    seen = {}
    monkeypatch.setattr(v2sim, "oracle_reply", lambda scene, request, **kw: seen.update(request=request, **kw) or {"ok": 1})
    enabled.scene = object()
    enabled.env = object()
    assert enabled.oracle({"kind": "joints", "names": ["j"]}) == {"ok": 1}
    assert seen["render_px"] == seen["image_px"] == 512


def test_process_client_refuses_without_option(monkeypatch):
    config, monitor = build_v2_sim_config("libero_10_t2")
    sent = []
    monkeypatch.setattr(v2sim.V2SimProcess, "_request", lambda self, request: sent.append(request) or {"r": 1})
    with pytest.raises(PermissionError, match="oracle op disabled"):
        v2sim.V2SimProcess(config, monitor_options=monitor).oracle({"kind": "joints", "names": ["a"]})
    assert sent == []
    process = v2sim.V2SimProcess(config, monitor_options={**monitor, "oracle": True})
    assert process.oracle({"kind": "joints", "names": ["a"]}) == {"r": 1}
    assert sent == [{"op": "oracle", "request": {"kind": "joints", "names": ["a"]}}]


_DISPATCH = r"""
import sys
import gpt_field.agent_speedup_v2_sim as m
class Stub(m.V2LiberoWorker):
    def __init__(self, config, monitor_options=None):
        self.monitor_options = m.validate_monitor_options(monitor_options)
        self.oracle_enabled = self.monitor_options.get("oracle", False) is True
        self.metadata = {"oracle_enabled": self.oracle_enabled}
        self.config = type("C", (), {"render_size": 512, "image_size": 512})()
    def _require_scene(self):
        return "scene"
    def close(self):
        pass
m.V2LiberoWorker = Stub
m.oracle_reply = lambda scene, request, **kw: {"echo": request, "scene": scene, **kw}
sys.exit(m._worker_main(None, {"oracle": sys.argv[1] == "on"}))
"""


@pytest.mark.parametrize("flag", ["on", "off"])
def test_worker_main_oracle_branch(flag):
    lines = "\n".join(json.dumps(r) for r in ({"op": "metadata"}, {"op": "oracle", "request": {"kind": "points"}},
                                               {"op": "nope"})) + "\n"
    proc = subprocess.run([sys.executable, "-c", _DISPATCH, flag], input=lines, capture_output=True, text=True,
                          timeout=120, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert replies[0] == {"ok": True, "result": {"oracle_enabled": flag == "on"}}
    if flag == "on":
        assert replies[1] == {"ok": True, "result": {"echo": {"kind": "points"}, "scene": "scene",
                                                     "render_px": 512, "image_px": 512}}
    else:
        assert replies[1] == {"ok": False, "error": "PermissionError: oracle op disabled"}
    assert replies[2]["ok"] is False and "unknown operation" in replies[2]["error"]


# =========================================================================================================
# SIM-EXT: projection and heights math
def _camera_from_hand(state8):
    """The MuJoCo eye-in-hand camera as the asset facts place it: hand + 0.05 R0; axes x = R1, y = R0, z = -R2."""
    R = rotation_matrix(np.asarray(state8[3:6], float))
    grip = np.asarray(state8[:3], float)
    pos = grip - robot.GRIP_SITE_ALONG_R2_M * R[:, 2] + robot.WRIST_CAMERA_ALONG_R0_M * R[:, 0]
    axes = np.column_stack([R[:, 1], R[:, 0], -R[:, 2]])
    return pos, axes.reshape(-1)


def test_project_world_points_pinhole_convention():
    xmat = np.eye(3).reshape(-1)               # camera looks along -z, x right, y up
    f = 256.0 / math.tan(math.radians(37.5))
    uv, depth, inside = v2sim.project_world_points([[0.1, 0.05, -1.0], [0.0, 0.0, 1.0], [3.0, 0.0, -1.0]],
                                                   [0, 0, 0], xmat, 75.0, 512, 512)
    assert uv[0] == pytest.approx([255.5 + f * 0.1, 255.5 - f * 0.05])
    assert depth[0] == pytest.approx(1.0) and inside[0] is True
    assert uv[1] is None and inside[1] is False and depth[1] == pytest.approx(-1.0)
    assert uv[2] is not None and inside[2] is False
    # a 256 render shown at 512 is the same picture: pixel edges scale, centres move by +0.5 px
    small, _, _ = v2sim.project_world_points([[0.1, 0.05, -1.0]], [0, 0, 0], xmat, 75.0, 256, 512)
    assert small[0] == pytest.approx(uv[0], abs=1e-9)


@pytest.mark.parametrize("pose", range(5))
def test_wrist_projection_matches_r2_model(pose):
    """The oracle's MuJoCo-camera projection (asset camera pose, f = 333.626) agrees with the R-2 model the
    policy uses (f = 333.6) within 1 px (contract §C14.2; the real-sim proof is the oracle_selfcheck)."""
    rng = np.random.default_rng(pose)
    rotvec = np.array(T2_START_STATE8[3:6]) + rng.normal(0, 0.25, 3)
    state8 = [*(np.array(T2_START_STATE8[:3]) + rng.normal(0, 0.05, 3)), *rotvec, 0.04, -0.04]
    pos, xmat = _camera_from_hand(state8)
    points = smoke.synthetic_wrist_points(state8)
    uv, _, inside = v2sim.project_world_points(points, pos, xmat, 75.0, 512, 512)
    model = smoke.r2_project(state8, points)
    errors = [math.dist(a, b) for a, b, ok in zip(uv, model, inside) if ok]
    assert len(errors) == 27 and max(errors) < 0.1


class _FakeMj:
    class mjtGeom:
        mjGEOM_MESH, mjGEOM_HFIELD, mjGEOM_BOX = 7, 1, 6

    @staticmethod
    def mju_rayGeom(pos, mat, size, pnt, vec, kind):     # axis-aligned boxes, vertical rays only
        assert kind == 6 and vec[2] == -1
        if abs(pnt[0] - pos[0]) <= size[0] and abs(pnt[1] - pos[1]) <= size[1] and pnt[2] >= pos[2] + size[2]:
            return float(pnt[2] - (pos[2] + size[2]))
        return -1.0


class _FakeScene:
    """table (body 1) + an object (body 2) + a robot hand box (body 3) right above the queried xy."""

    def __init__(self):
        boxes = [((0.0, 0.0, 0.875), (0.5, 0.5, 0.025), 1), ((0.1, 0.0, 0.93), (0.02, 0.02, 0.03), 2),
                 ((0.0, 0.0, 1.10), (0.03, 0.08, 0.04), 3)]
        self.mj = _FakeMj
        self.M = SimpleNamespace(ngeom=len(boxes), geom_rbound=[math.hypot(*s) for _, s, _ in boxes],
                                 geom_type=[6] * len(boxes), geom_size=[np.array(s) for _, s, _ in boxes])
        self.D = SimpleNamespace(geom_xpos=[np.array(p) for p, _, _ in boxes], geom_xmat=[np.eye(3).reshape(-1)] * 3)
        self.coll = [True] * len(boxes)
        self.geom_body = [b for _, _, b in boxes]
        self.robot_bodies = {3}

    def gname(self, g):
        return f"g{g}"

    def bname(self, b):
        return {1: "table", 2: "thing_main", 3: "robot0_right_hand"}[b]

    def owner(self, b):
        return {1: "table", 2: "thing", 3: "robot"}[b]


def test_heights_skip_every_robot_body():
    scene = _FakeScene()
    reply = v2sim.oracle_reply(scene, {"kind": "heights", "xy": [[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]]},
                               render_px=512, image_px=512)
    assert reply["z"][0] == pytest.approx(0.90) and reply["hit"][0]["owner"] == "table"   # hand above: skipped
    assert reply["z"][1] == pytest.approx(0.96) and reply["hit"][1]["geom"] == "g1"
    assert reply["z"][2] is None and reply["hit"][2] is None
    robot_on = v2sim.oracle_reply(scene, {"kind": "heights", "xy": [[0.0, 0.0]], "include_robot": True},
                                  render_px=512, image_px=512)
    assert robot_on["z"][0] == pytest.approx(1.14) and robot_on["hit"][0]["robot"] is True
    with pytest.raises(ValueError):
        v2sim.oracle_reply(scene, {"kind": "heights", "xy": [[0, 0]], "include_robot": 1}, render_px=512, image_px=512)


class _NamedMj:
    """mujoco stand-in: object ids by (type, name), box/mesh geom types."""

    class mjtObj:
        mjOBJ_BODY, mjOBJ_JOINT, mjOBJ_GEOM, mjOBJ_SITE, mjOBJ_CAMERA = 1, 3, 5, 6, 7

    class mjtGeom:
        mjGEOM_HFIELD, mjGEOM_BOX, mjGEOM_MESH = 1, 6, 7

    NAMES = {(1, "thing_main"): 1, (1, "thing_lid"): 2, (1, "ghost"): 3, (5, "thing_g0"): 0, (5, "thing_g1"): 1,
             (5, "thing_vis"): 2, (5, "ghost_vis"): 3, (6, "thing_site"): 0, (3, "thing_hinge"): 0,
             (7, "agentview"): 0, (7, "robot0_eye_in_hand"): 1}

    @classmethod
    def mj_name2id(cls, M, objtype, name):
        return cls.NAMES.get((int(objtype), name), -1)


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class _PointsScene:
    """Bodies 1 thing_main (root) -> 2 thing_lid, and 3 ghost (visual geom only). Geoms: 0 thing_g0 (box on body 1,
    rotated 90 deg about z), 1 thing_g1 (box on body 2), 2 thing_vis (visual mesh on body 1), 3 ghost_vis (visual box
    on body 3). One site and one hinge joint; two cameras."""

    def __init__(self):
        self.mj = _NamedMj
        sizes = [np.array([0.02, 0.03, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.05, 0.05, 0.05]),
                 np.array([0.01, 0.02, 0.03])]
        self.M = SimpleNamespace(ngeom=4, geom_type=[6, 6, 7, 6], geom_size=sizes, site_size=[np.array([0.01, 0.02, 0.005])],
                                 cam_fovy=[45.0, 75.0])
        quat_wxyz = np.array([0.5, 0.5, -0.5, 0.5])                    # 120 deg about (1, -1, 1) / sqrt(3)
        self.D = SimpleNamespace(
            geom_xpos=[np.array([0.1, 0.2, 0.95]), np.array([0.1, 0.2, 0.97]), np.array([0.1, 0.2, 0.96]),
                       np.array([-0.3, 0.0, 1.0])],
            geom_xmat=[_rz(90).reshape(-1), np.eye(3).reshape(-1), np.eye(3).reshape(-1), np.eye(3).reshape(-1)],
            xpos=[np.zeros(3), np.array([0.1, 0.2, 0.9]), np.array([0.1, 0.2, 0.97]), np.array([-0.3, 0.0, 1.0])],
            subtree_com=[np.zeros(3), np.array([0.1, 0.21, 0.93]), np.array([0.1, 0.2, 0.97]), np.array([-0.3, 0, 1])],
            xquat=[np.array([1.0, 0, 0, 0]), quat_wxyz, np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0])],
            site_xpos=[np.array([0.0, -0.1, 0.9])], site_xmat=[_rz(90).reshape(-1)],
            cam_xpos=[np.array([1.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.2])],
            cam_xmat=[np.eye(3).reshape(-1), np.eye(3).reshape(-1)])
        self.geom_body = [1, 2, 1, 3]
        self.coll = [True, True, False, False]

    def subtree(self, b):
        return {1: {1, 2}, 2: {2}, 3: {3}}[int(b)]

    def geoms_of(self, bodies):
        return sorted(g for g in range(self.M.ngeom) if self.coll[g] and self.geom_body[g] in bodies)

    def geom_pts(self, g):
        R = np.asarray(self.D.geom_xmat[g], float).reshape(3, 3)
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * self.M.geom_size[g]
        return corners @ R.T + self.D.geom_xpos[g]

    def aabb(self, g):
        p = self.geom_pts(g)
        return p.min(0), p.max(0)

    def joint_info(self, name):
        return {"joint": name, "q": 0.3, "qd": 0.0, "range": [-0.005, 2.1], "type": 3, "body": 2}


def _points(scene, *items, corners=False):
    request = {"kind": "points", "items": list(items), **({"corners": True} if corners else {})}
    return v2sim.oracle_reply(scene, request, render_px=512, image_px=512)


def test_points_op_where_offset_geom_list_body_and_site():
    """Review BH-4: every ``points`` selector and ``where`` on a fake scene (no simulator)."""
    scene = _PointsScene()
    # a rotated box: its aabb swaps the x/y half sizes (0.03 x 0.02); the top is centre z + 0.01
    reply = _points(scene, {"name": "top", "geom": "thing_g0"},
                    {"name": "mid", "geom": "thing_g0", "where": "aabb_centre"},
                    {"name": "com", "geom": "thing_g0", "where": "com"},
                    {"name": "up", "geom": "thing_g0", "where": "origin", "offset": [0.0, 0.0, 0.01]})
    assert reply["points"]["top"] == pytest.approx([0.1, 0.2, 0.96])
    assert reply["points"]["mid"] == pytest.approx([0.1, 0.2, 0.95])
    assert reply["points"]["com"] == pytest.approx([0.1, 0.2, 0.95])
    assert reply["points"]["up"] == pytest.approx([0.1, 0.2, 0.96])
    assert reply["aabb"]["top"][0] == pytest.approx([0.07, 0.18, 0.94])
    assert reply["aabb"]["top"][1] == pytest.approx([0.13, 0.22, 0.96])
    assert "corners" not in reply
    # a geom list: the union aabb; com/origin = the mean geom centre
    both = _points(scene, {"name": "u", "geom": ["thing_g0", "thing_g1"]},
                   {"name": "c", "geom": ["thing_g0", "thing_g1"], "where": "com"})
    assert both["points"]["u"] == pytest.approx([0.1, 0.2, 0.98]) and both["points"]["c"] == pytest.approx([0.1, 0.2, 0.96])
    # a body: its subtree's COLLISION geoms (the visual mesh is ignored); origin = xpos, com = subtree_com
    body = _points(scene, {"name": "t", "body": "thing_main"}, {"name": "o", "body": "thing_main", "where": "origin"},
                   {"name": "m", "body": "thing_main", "where": "com"}, {"name": "l", "body": "thing_lid"},
                   {"name": "g", "body": "ghost"})
    assert body["points"]["t"] == pytest.approx([0.1, 0.2, 0.98])
    assert body["aabb"]["t"][0] == pytest.approx([0.07, 0.18, 0.94])
    assert body["points"]["o"] == pytest.approx([0.1, 0.2, 0.9]) and body["points"]["m"] == pytest.approx([0.1, 0.21, 0.93])
    assert body["points"]["l"] == pytest.approx([0.1, 0.2, 0.98])
    assert body["points"]["g"] == pytest.approx([-0.3, 0.0, 1.03])  # no collision geom: all its geoms
    # a site: its rotated box (half sizes 0.02 x 0.01 x 0.005 after the 90 deg turn)
    site = _points(scene, {"name": "s", "site": "thing_site"}, {"name": "so", "site": "thing_site", "where": "origin"})
    assert site["points"]["s"] == pytest.approx([0.0, -0.1, 0.905]) and site["points"]["so"] == pytest.approx([0.0, -0.1, 0.9])
    assert site["aabb"]["s"][0] == pytest.approx([-0.02, -0.11, 0.895])
    # corners: oriented box corners per geom (8 each), aabb corners for a site
    corners = _points(scene, {"name": "b", "body": "thing_main"}, {"name": "s", "site": "thing_site"},
                      {"name": "g0", "geom": "thing_g0"}, corners=True)["corners"]
    assert [len(c) for c in corners["b"]] == [8, 8] and len(corners["s"]) == 1 and len(corners["s"][0]) == 8
    assert np.allclose(sorted(map(tuple, corners["g0"][0])), sorted(map(tuple, scene.geom_pts(0))))
    for bad in ({"name": "x", "geom": "thing_g0", "body": "thing_main"}, {"name": "x"},
                {"name": "x", "geom": "thing_g0", "where": "top"}, {"name": "x", "geom": "nope"},
                {"name": "x", "geom": []}, {"name": "x", "site": "thing_site", "offset": [0, 0]}, {"geom": "thing_g0"}):
        with pytest.raises(ValueError):
            _points(scene, bad)
    with pytest.raises(ValueError):
        v2sim.oracle_reply(scene, {"kind": "points", "items": [], "corners": 1}, render_px=512, image_px=512)


def _quat_xyzw_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_bodies_op_reorders_mujoco_wxyz_to_xyzw():
    scene = _PointsScene()
    reply = v2sim.oracle_reply(scene, {"kind": "bodies", "names": ["thing_main", "thing_lid"]}, render_px=512,
                               image_px=512)
    pose = reply["poses"]["thing_main"]
    assert pose["xyz"] == pytest.approx([0.1, 0.2, 0.9])
    assert pose["quat_xyzw"] == pytest.approx([0.5, -0.5, 0.5, 0.5])       # MuJoCo (w, x, y, z) = (0.5, 0.5, -0.5, 0.5)
    axis = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)                       # Rodrigues: 120 deg about the axis
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = math.radians(120.0)
    expected = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    assert np.allclose(_quat_xyzw_matrix(pose["quat_xyzw"]), expected)
    assert reply["poses"]["thing_lid"]["quat_xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    for bad in (["nope"], [], "", None):
        with pytest.raises(ValueError):
            v2sim.oracle_reply(scene, {"kind": "bodies", "names": bad}, render_px=512, image_px=512)


def test_joints_op_success_and_refusals():
    scene = _PointsScene()
    reply = v2sim.oracle_reply(scene, {"kind": "joints", "names": ["thing_hinge"]}, render_px=512, image_px=512)
    assert reply == {"q": {"thing_hinge": 0.3}, "range": {"thing_hinge": [-0.005, 2.1]}}
    assert v2sim.oracle_reply(scene, {"kind": "joints", "names": "thing_hinge"}, render_px=512,
                              image_px=512)["q"] == {"thing_hinge": 0.3}
    for bad in (["thing_hinge", "nope"], [], [""], None, [3]):
        with pytest.raises(ValueError):
            v2sim.oracle_reply(scene, {"kind": "joints", "names": bad}, render_px=512, image_px=512)


def test_project_op_returns_the_camera_pose():
    scene = _PointsScene()
    reply = v2sim.oracle_reply(scene, {"kind": "project", "camera": "wrist", "xyz": [[0.0, 0.0, 1.0]]},
                               render_px=512, image_px=512)
    assert reply["camera"] == "robot0_eye_in_hand" and reply["cam_xyz"] == pytest.approx([0.0, 0.0, 1.2])
    assert reply["cam_xmat"] == pytest.approx(np.eye(3).reshape(-1).tolist())
    assert reply["uv"][0] == pytest.approx([255.5, 255.5]) and reply["depth_m"][0] == pytest.approx(0.2)
    external = v2sim.oracle_reply(scene, {"kind": "project", "camera": "agentview", "xyz": []}, render_px=512,
                                  image_px=512)
    assert external["uv"] == [] and external["cam_xyz"] == pytest.approx([1.0, 0.0, 1.5])


def test_oracle_reply_rejects_malformed_requests():
    scene = _FakeScene()
    for bad in (None, {}, {"kind": "teleport"}, {"kind": "heights", "xy": [[0.0]]},
                {"kind": "heights", "xy": "0,0"}, {"kind": "project", "camera": "ceiling", "xyz": []}):
        with pytest.raises(ValueError):
            v2sim.oracle_reply(scene, bad, render_px=512, image_px=512)


# =========================================================================================================
# ORACLE stub on a fake evaluator
def _png(mask):
    return v2sim._oracle_png(np.asarray(mask, bool))


class FakeOracleSim:
    """Answers the oracle op from fixed part points and two top-down pinhole cameras (wrist at 1.25 m, external at
    2.0 m). Masks are discs around the projected part points; collision surfaces sit 1 mm below the points; the
    rendered (visual) surface of a part sits ``visual_dz[pid]`` above its point; a part's region geoms are one
    cube of half size ``region_half_m`` around its point; ``occluders[camera][pid] = (owner, front_m)`` puts that
    owner ("robot" = a robot body) ``front_m`` in front of the part at its pixel."""

    XYZ = {"A": [-0.10, 0.10, 0.955], "B": [0.05, 0.12, 0.93], "C": [0.05, -0.08, 1.03],
           "D": [0.03, -0.01, 1.05], "E": [-0.07, -0.08, 0.95]}

    def __init__(self, *, oracle=True, xyz=None, mask_radius_px=12, region_half_m=0.006, visual_dz=None,
                 occluders=None):
        self.monitor_options = {"goal_aware": True, "truth_window": True, **({"oracle": True} if oracle else {})}
        self.metadata = {"oracle_enabled": oracle}
        self.xyz = {k: list(v) for k, v in (xyz or self.XYZ).items()}
        self.mask_radius_px = mask_radius_px
        self.region_half_m = region_half_m
        self.visual_dz = dict(visual_dz or {})
        self.occluders = {camera: dict(v) for camera, v in (occluders or {}).items()}
        self.cameras = {"wrist": ([0.0, 0.0, 1.25], np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), 75.0),
                        "external": ([0.0, 0.0, 2.0], np.column_stack([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 45.0)}
        self.requests = []

    def project(self, camera, xyz):
        pos, axes, fovy = self.cameras[camera]
        return v2sim.project_world_points(xyz, pos, axes.reshape(-1), fovy, 512, 512)

    def _pixel(self, camera, uv):
        """What renders at ``uv``: the part whose projection is nearest (within 3 px), else the floor 5 m away."""
        names = list(self.xyz)
        projected, _, _ = self.project(camera, [self.xyz[n] for n in names])
        best = min((math.dist(p, uv), n) for p, n in zip(projected, names) if p is not None)
        cam_z = self.cameras[camera][0][2]
        if best[0] > 3.0:
            return {"px": [int(round(uv[0])), int(round(uv[1]))], "geom": None, "body": "floor", "owner": "floor",
                    "robot": False, "depth_m": 5.0}
        pid = best[1]
        depth = cam_z - (self.xyz[pid][2] + self.visual_dz.get(pid, 0.0))     # top-down: depth = height above
        owner = oracle.ORACLE_PARTS["libero_10_t2"][pid]["owner"]
        occluder = self.occluders.get(camera, {}).get(pid)
        if occluder is not None:
            owner, depth = occluder[0], depth - occluder[1]
        return {"px": [int(round(uv[0])), int(round(uv[1]))], "geom": f"{pid}_geom", "body": f"{pid}_body",
                "owner": owner, "robot": owner == "robot", "depth_m": depth}

    def oracle(self, request):
        self.requests.append(request)
        kind = request["kind"]
        if kind == "points":
            reply = {"points": {i["name"]: list(self.xyz[i["name"]]) for i in request["items"]}}
            if request.get("corners"):
                h = self.region_half_m
                reply["corners"] = {i["name"]: [[[x + a * h, y + b * h, z + c * h] for a in (-1, 1) for b in (-1, 1)
                                                 for c in (-1, 1)]] for i in request["items"]
                                    for x, y, z in [self.xyz[i["name"]]]}
            return reply
        if kind == "heights":
            z, hits = [], []
            for xy in request["xy"]:
                pid = next(p for p, v in self.xyz.items() if np.allclose(v[:2], xy))
                z.append(self.xyz[pid][2] - 0.001)
                hits.append({"geom": "g", "body": "b", "owner": oracle.ORACLE_PARTS["libero_10_t2"][pid]["owner"],
                             "robot": False})
            return {"z": z, "hit": hits}
        if kind == "project":
            uv, depth, inside = self.project(request["camera"], request["xyz"])
            pos, axes, _ = self.cameras[request["camera"]]
            return {"uv": uv, "depth_m": depth, "in_image": inside, "cam_xyz": list(pos),
                    "cam_xmat": axes.reshape(-1).tolist()}
        if kind == "segment":
            masks = {}
            vs, us = np.mgrid[0:512, 0:512]
            for item in request.get("masks", []):
                (uv,), _, _ = self.project(request["camera"], [self.xyz[item["name"]]])
                mask = (us - uv[0]) ** 2 + (vs - uv[1]) ** 2 <= self.mask_radius_px ** 2
                masks[item["name"]] = {"pixels": int(mask.sum()), "png": _png(mask) if mask.any() else None}
            at = [self._pixel(request["camera"], uv) for uv in request.get("uv", [])]
            return {"masks": masks, "at": at}
        raise AssertionError(kind)


def test_oracle_refused_without_the_simulator_option():
    with pytest.raises(PermissionError):
        oracle.OracleAnswers(FakeOracleSim(oracle=False), "libero_10_t2")
    with pytest.raises(PermissionError):
        oracle.OracleAnswers(SimpleNamespace(metadata={}), "libero_10_t2")
    with pytest.raises(ValueError):
        oracle.OracleAnswers(FakeOracleSim(), "bowl_plate")           # no ORACLE parts/plan for that key
    for bad in ({"perturb_px": -1.0}, {"perturb_px": float("nan")}, {"perturb_px": True}, {"wrong_item": "Z"}):
        with pytest.raises(ValueError):
            oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2", **bad)


def test_oracle_plan_is_the_spec_plan_with_live_cells():
    sim = FakeOracleSim()
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    template = oracle.ORACLE_PLANS["libero_10_t2"]
    assert len(template["scene"]) <= 80 and oracle.compact_chars(template) <= 834
    assert list(template) == ["scene", "parts", "hold_true", "steps"]
    for k in range(4):
        plan = answers.answer("plan", {"quarter_turns": k, "image_size": 512})
        assert [p["id"] for p in plan["parts"]] == ["A", "B", "C", "D", "E"]
        assert plan["steps"] == template["steps"] and plan["hold_true"] == []
        for part in plan["parts"]:
            surface = [*sim.xyz[part["id"]][:2], sim.xyz[part["id"]][2] - 0.001]
            (uvw,), _, (inw,) = sim.project("wrist", [surface])
            (uvx,), _, (inx,) = sim.project("external", [surface])
            assert inw and inx
            assert part["w"] == oracle.coarse_cell_of(mark_prompting.rotate_pixel(uvw, k, 512))
            assert part["x"] == oracle.coarse_cell_of(uvx)
            assert 1 <= part["w"] <= 64 and 1 <= part["x"] <= 64
    assert template["parts"][0]["w"] == 0          # the template itself is never mutated
    record = answers.report()["answers"][0]
    assert record["kind"] == "plan" and record["cells"][0]["w_rule"] == "truth_in_image"


def test_oracle_plan_validates_with_the_plan_module_when_present():
    plan_module = pytest.importorskip("gpt_field.agent_speedup_v2_plan")
    answers = oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2")
    plan = plan_module.parse_plan(answers.answer("plan", {"quarter_turns": 2}))
    assert plan_module.validate_plan(plan) == []


def test_oracle_plan_cells_outside_and_border_cut():
    xyz = dict(FakeOracleSim.XYZ, D=[0.155, -0.01, 1.05], C=[0.6, -0.08, 1.03])   # D just above the top edge
    sim = FakeOracleSim(xyz=xyz, mask_radius_px=30)
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    (uv_d,), _, (in_d,) = sim.project("wrist", [[0.155, -0.01, 1.049]])
    assert not in_d and -64 < uv_d[1] < 0
    plan = answers.answer("plan", {"quarter_turns": 0})
    parts = {p["id"]: p for p in plan["parts"]}
    assert parts["C"]["w"] == 0                       # far outside the image
    assert parts["D"]["w"] == oracle.coarse_cell_of([uv_d[0], 0.0])   # border-cut: the visible top cell
    cells = {c["pid"]: c for c in answers.report()["answers"][-1]["cells"]}
    assert cells["D"]["w_rule"] == "border_cut_visible" and cells["C"]["w_rule"] == "not_visible"
    small = oracle.OracleAnswers(FakeOracleSim(xyz=xyz, mask_radius_px=5), "libero_10_t2")
    assert {p["id"]: p["w"] for p in small.answer("plan", {"quarter_turns": 0})["parts"]}["D"] == 0
    # far beyond the border with a large visible region: only a whole-object part (D) may take it
    far = dict(FakeOracleSim.XYZ, D=[0.32, -0.01, 1.05], C=[0.32, -0.08, 1.03])
    wide = FakeOracleSim(xyz=far, mask_radius_px=330)
    (uv_far,), _, _ = wide.project("wrist", [[0.32, -0.01, 1.049]])
    assert uv_far[1] < -200
    for k in range(4):
        cells = {p["id"]: p["w"] for p in oracle.OracleAnswers(wide, "libero_10_t2").answer(
            "plan", {"quarter_turns": k})["parts"]}
        assert cells["C"] == 0
        assert cells["D"] == oracle.coarse_cell_of(mark_prompting.rotate_pixel([uv_far[0], 0.0], k, 512))


@pytest.mark.parametrize("k", range(4))
def test_point_fine_maps_back_within_quantization(k):
    sim = FakeOracleSim()
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    for pid in "ABCDE":
        (uv,), _, _ = sim.project("wrist", [[*sim.xyz[pid][:2], sim.xyz[pid][2] - 0.001]])
        centre = [uv[0] + 23.0, uv[1] - 9.0]
        origin = oracle.crop_origin(mark_prompting.rotate_pixel(centre, k, 512))
        reply = answers.answer("point_fine", {"pid": pid, "camera": "wrist", "quarter_turns": k, "origin": origin,
                                              "crop_px": 128, "image_size": 512})
        record = answers.records[-1]
        assert reply["point_id"] >= 1 and record["in_crop"] is True
        mapped = mark_prompting.unrotate_pixel(oracle.v1_fine_point_pixel_at(origin, reply["point_id"]), k, 512)
        assert math.dist(mapped, uv) <= math.sqrt(2) * 8 + 0.5
        assert record["answered_uv"] == pytest.approx(list(mapped)) and record["error_px"] == pytest.approx(
            math.dist(mapped, uv))
    # the truth outside the crop -> 0 [C-37]; a centre-only meta falls back to the clamped crop
    (uv,), _, _ = sim.project("wrist", [sim.xyz["A"]])
    far = [uv[0] + 150.0 if uv[0] < 256 else uv[0] - 150.0, uv[1]]
    assert answers.answer("point_fine", {"pid": "A", "camera": "wrist", "quarter_turns": k, "center_uv": far,
                                         "crop_px": 128})["point_id"] == 0
    assert answers.records[-1]["in_crop"] is False


def test_point_coarse_perturbation_and_wrong_item():
    sim = FakeOracleSim()
    answers = oracle.OracleAnswers(sim, "libero_10_t2", perturb_px=20.0, seed=3)
    for k in range(4):
        reply = answers.answer("point_coarse", {"pid": "B", "camera": "wrist", "quarter_turns": k})
        record = answers.records[-1]
        assert reply["cell_id"] >= 1
        assert math.dist(record["shown_uv_turned"], record["truth_uv_turned"]) == pytest.approx(20.0)
        assert record["error_px"] <= math.sqrt(2) * 32 + 20.5
    again = oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2", perturb_px=20.0, seed=3)
    again.answer("point_coarse", {"pid": "B", "camera": "wrist", "quarter_turns": 0})
    assert again.records[-1]["shown_uv_turned"] == answers.records[0]["shown_uv_turned"]   # seeded
    wrong = oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2", wrong_item="C")
    wrong.answer("point_coarse", {"pid": "C", "camera": "external", "quarter_turns": 0})
    assert wrong.records[-1]["source_pid"] == "D"           # the next part (plan order): the pot body
    plan = wrong.answer("plan", {"quarter_turns": 0})
    cells = {p["id"]: (p["w"], p["x"]) for p in plan["parts"]}
    assert cells["C"] == cells["D"]
    outside = oracle.OracleAnswers(FakeOracleSim(xyz=dict(FakeOracleSim.XYZ, B=[0.9, 0.9, 0.93])), "libero_10_t2")
    assert outside.answer("point_coarse", {"pid": "B", "camera": "wrist", "quarter_turns": 0}) == {"cell_id": 0}


def test_border_rule_sees_only_the_part_region():
    """Review BH-3: C's mask is the whole pot; its region (the loop's boxes) decides the border rule. C's truth sits
    20 px above the wrist image's top edge and the pot's visible pixels reach 40 px into the image: a small loop
    (all outside the image) gives 0, a loop reaching into the image gives the cell where the loop shows."""
    x = (255.5 + 20.0) * (1.25 - 1.029) / (256.0 / math.tan(math.radians(37.5)))   # v = 255.5 - f x / d = -20
    xyz = dict(FakeOracleSim.XYZ, C=[x, -0.08, 1.03])
    small = FakeOracleSim(xyz=xyz, mask_radius_px=60, region_half_m=0.004)
    (uv,), _, (inside,) = small.project("wrist", [[x, -0.08, 1.029]])
    assert not inside and -21 < uv[1] < -19
    answers = oracle.OracleAnswers(small, "libero_10_t2")
    cells = {p["id"]: p["w"] for p in answers.answer("plan", {"quarter_turns": 0})["parts"]}
    rules = {c["pid"]: c["w_rule"] for c in answers.report()["answers"][-1]["cells"]}
    assert cells["C"] == 0 and rules["C"] == "not_visible"          # the pot body alone never answers for the loop
    region = answers.region_mask("C", "wrist")
    assert region is not None and not region.any()                 # the loop's boxes project outside the image
    assert answers.region_mask("D", "wrist") is None and answers.region_mask("B", "wrist") is None
    big = FakeOracleSim(xyz=xyz, mask_radius_px=60, region_half_m=0.03)
    for k in range(4):
        plan = oracle.OracleAnswers(big, "libero_10_t2").answer("plan", {"quarter_turns": k})
        w = {p["id"]: p["w"] for p in plan["parts"]}["C"]
        assert w == oracle.coarse_cell_of(mark_prompting.rotate_pixel([uv[0], 0.0], k, 512))
    # an in-image truth keeps the truth cell whatever the region
    assert cells["A"] == oracle.coarse_cell_of(small.project("wrist", [[-0.10, 0.10, 0.954]])[0][0])


def test_region_mask_is_the_union_of_projected_box_hulls():
    sim = FakeOracleSim(region_half_m=0.01)
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    mask = answers.region_mask("C", "wrist")
    corners = sim.oracle({"kind": "points", "corners": True, "items": [{"name": "C", "geom": ["g"]}]})["corners"]["C"][0]
    uv = np.asarray(sim.project("wrist", corners)[0])
    lo, hi = uv.min(0), uv.max(0)
    vs, us = np.nonzero(mask)
    assert us.min() >= lo[0] - oracle.REGION_DILATE_PX - 1 and us.max() <= hi[0] + oracle.REGION_DILATE_PX + 1
    assert vs.min() >= lo[1] - oracle.REGION_DILATE_PX - 1 and vs.max() <= hi[1] + oracle.REGION_DILATE_PX + 1
    assert mask[int(round((lo[1] + hi[1]) / 2)), int(round((lo[0] + hi[0]) / 2))]
    hull = oracle.convex_hull_2d([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0)])
    assert sorted(hull) == [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0)]
    restricted = answers.restrict_to_region("C", "wrist", np.ones((512, 512), bool))
    assert np.array_equal(restricted, mask)
    assert answers.restrict_to_region("C", "wrist", np.zeros((512, 512), bool)) is None
    assert np.array_equal(answers.restrict_to_region("B", "wrist", np.ones((512, 512), bool)), np.ones((512, 512), bool))


def test_pointing_records_occlusion_and_optional_robot_zero():
    """Review BH-6: every pointing record and plan cell carries the occlusion flags; answers change only with the
    opt-in zero_when_robot_occluded, and only for a robot occluder."""
    sim = FakeOracleSim(occluders={"wrist": {"A": ("robot", 0.03), "B": ("moka_pot_1", 0.05)}})
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    (uv_a,), _, _ = sim.project("wrist", [sim.xyz["A"]])
    meta = {"pid": "A", "camera": "wrist", "quarter_turns": 0, "center_uv": uv_a, "crop_px": 128}
    assert answers.answer("point_fine", meta)["point_id"] >= 1
    record = answers.records[-1]
    assert record["occluded"] is True and record["occlusion"]["occluder"] == "robot"
    assert record["occlusion"]["front_m"] == pytest.approx(0.031, abs=1e-6)       # 3 cm + the 1 mm surface offset
    assert answers.answer("point_coarse", {"pid": "B", "camera": "wrist", "quarter_turns": 0})["cell_id"] >= 1
    assert answers.records[-1]["occluded"] is True and answers.records[-1]["occlusion"]["occluder"] == "moka_pot_1"
    assert answers.answer("point_coarse", {"pid": "E", "camera": "wrist", "quarter_turns": 0})["cell_id"] >= 1
    assert answers.records[-1]["occluded"] is False and answers.records[-1]["occlusion"]["occluder"] is None
    plan = answers.answer("plan", {"quarter_turns": 1})
    cells = {c["pid"]: c for c in answers.report()["answers"][-1]["cells"]}
    assert cells["A"]["occluded"] == {"w": True, "x": False} and cells["E"]["occluded"] == {"w": False, "x": False}
    assert all(p["w"] >= 1 for p in plan["parts"])                 # plan cells keep the contract rule
    assert answers.report()["occluded_answers"] == {"point_fine": 1, "point_coarse": 1, "plan": 1}
    strict = oracle.OracleAnswers(sim, "libero_10_t2", zero_when_robot_occluded=True)
    assert strict.answer("point_fine", meta) == {"point_id": 0}
    assert strict.records[-1]["answer_rule"] == "robot_occluded_zero"
    assert strict.answer("point_coarse", {"pid": "A", "camera": "wrist", "quarter_turns": 0}) == {"cell_id": 0}
    assert strict.answer("point_coarse", {"pid": "B", "camera": "wrist", "quarter_turns": 0})["cell_id"] >= 1
    with pytest.raises(ValueError):
        oracle.OracleAnswers(sim, "libero_10_t2", zero_when_robot_occluded=1)


def test_unknown_pid_raises_value_error():
    answers = oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2")
    for pid in (None, "Z", 3):
        for kind in ("point_fine", "point_coarse"):
            with pytest.raises(ValueError, match="unknown ORACLE part id"):
                answers.answer(kind, {"pid": pid, "camera": "wrist", "quarter_turns": 0})
    with pytest.raises(ValueError):
        answers.region_mask("Z", "wrist")


def test_surface_truth_reports_collision_and_visual_surfaces():
    """Review BH-1: G1 candidates at a measured xy. The fake's collision surface is 1 mm below each point and the
    rendered surface ``visual_dz`` above it; the back-projected visual z converges to the rendered surface."""
    sim = FakeOracleSim(visual_dz={"A": 0.004, "C": -0.002})
    answers = oracle.OracleAnswers(sim, "libero_10_t2")
    rows = answers.surface_truth([sim.xyz["A"][:2], sim.xyz["C"][:2]])
    a, c = rows
    assert a["surface_z"] == pytest.approx(0.954) and a["surface_hit"]["owner"] == "flat_stove_1"
    for camera in ("wrist", "external"):
        assert a["visual"][camera]["z"] == pytest.approx(0.959, abs=1e-6) and a["visual"][camera]["converged"]
        assert a["visual"][camera]["front_m_first"] == pytest.approx(0.005, abs=1e-6)    # rendered 5 mm nearer
        assert a["visual"][camera]["xy_offset_mm"] < 0.01 and a["visual"][camera]["passes"] <= 3
        assert c["visual"][camera]["z"] == pytest.approx(1.028, abs=1e-6)
    assert answers.surface_truth([]) == []
    far = FakeOracleSim(xyz=dict(FakeOracleSim.XYZ, B=[2.5, 0.12, 0.93]))
    row = oracle.OracleAnswers(far, "libero_10_t2").surface_truth([[2.5, 0.12]], cameras=("wrist",))[0]
    assert row["surface_z"] == pytest.approx(0.929) and row["visual"]["wrist"] is None     # out of view


def test_repair_review_and_report():
    answers = oracle.OracleAnswers(FakeOracleSim(), "libero_10_t2")
    assert answers.answer("repair", {}) == {"why": "oracle", "action": "retry", "part": None, "step": None}
    assert answers.answer("review", {"review": True})["action"] == "hold_still"
    answers.answer("point_fine", {"pid": "A", "camera": "external", "quarter_turns": 0,
                                  "center_uv": FakeOracleSim().project("external", [FakeOracleSim.XYZ["A"]])[0][0]})
    with pytest.raises(ValueError):
        answers.answer("teleport", {})
    report = answers.report()
    assert report["counts"] == {"repair": 1, "review": 1, "point_fine": 1}
    assert report["max_error_px"]["point_fine"] <= math.sqrt(2) * 8 + 0.5
    assert report["fine_mapping"] in ("v1_layout_restated", "agent_speedup_v2_client.fine_point_pixel_at")
    json.dumps(report, allow_nan=False)
    # answers are copies: editing one never changes the record or the template
    review = answers.answer("review", {})
    review["action"] = "retry"
    assert oracle.REVIEW_ANSWER["action"] == "hold_still"


def test_grid_helpers_restate_v1_marks():
    for cell in range(1, 65):
        assert oracle.coarse_cell_centre(cell) == mark_prompting.coarse_cell_center(cell, 512)
        assert oracle.coarse_cell_of(oracle.coarse_cell_centre(cell)) == cell
    assert oracle.coarse_cell_of([-0.6, 10]) == 0 and oracle.coarse_cell_of([511.4, 511.4]) == 64
    assert mark_prompting.fine_crop_size(512, mark_prompting.WIDE_MARGIN_CELLS) == oracle.FINE_CROP_PX
    for point in range(1, 65):
        assert oracle.v1_fine_point_pixel_at((40, 200), point) == mark_prompting.fine_point_pixel(
            (40, 200), point, 512, mark_prompting.WIDE_MARGIN_CELLS)
    assert oracle.crop_origin((10, 500)) == (0, 384)
    assert oracle.in_crop((0, 0), (0, 0), 128) and not oracle.in_crop((127.6, 0), (0, 0), 128)


def test_oracle_part_names_exist_in_the_task_dump():
    dump = Path("/tmp/claude-0/-workspace/2e071f26-a38c-4a1c-b40e-bd363cbcfe54/impl/v2/survey/tasks/t2_dump.json")
    if not dump.is_file():
        pytest.skip("SURVEY t2_dump.json not present")
    data = json.loads(dump.read_text())
    geoms = {g["geom"] for entry in data["objects"] + data["fixtures"] for g in entry["col_geoms"]}
    bodies = {b for entry in data["objects"] + data["fixtures"] for b in entry["per_body_col_aabb"]}
    owners = {entry["name"] for entry in data["objects"] + data["fixtures"]}
    for pid, spec in oracle.ORACLE_PARTS["libero_10_t2"].items():
        names = spec["item"]["geom"]
        assert set([names] if isinstance(names, str) else names) <= geoms, pid
        assert spec["body"] in bodies and spec["owner"] in owners


def test_no_package_module_imports_the_oracle_or_the_harness():
    """B0's firewall test owns the full import graph; this guards BH's two script modules (AST, function-level
    imports included) across every v2 package module."""
    import ast
    for path in (ROOT / "src" / "gpt_field").glob("agent_speedup_v2*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert "oracle" not in name and "relation_smoke" not in name and not name.startswith("scripts"), (
                    path.name, name)


# =========================================================================================================
# harness
def test_harness_cli_and_scenario_loader(tmp_path, monkeypatch):
    args = smoke.parse_args(["--task", "libero_10_t2", "--scenario", "oracle_selfcheck", "--first-sense", "+1"])
    assert args.first_sense_value == 1 and args.max_steps == smoke.DEFAULT_MAX_STEPS and args.episode_index == 0
    assert smoke.world_z_label(+1, T2_START_STATE8) == -1 and smoke.world_z_label(-1, T2_START_STATE8) == 1
    with pytest.raises(SystemExit):
        smoke.parse_args(["--task", "libero_10_t2", "--scenario", "x", "--first-sense", "1"])
    assert smoke.load_scenario("oracle_selfcheck").uses_policy is False
    monkeypatch.setattr(smoke, "SCENARIO_FILE", tmp_path / "absent.py")
    with pytest.raises(ValueError, match="BR"):
        smoke.load_scenario("turn_t2")
    module = tmp_path / "scenarios.py"
    module.write_text("class S:\n    name = 'turn_t2'\nSCENARIOS = {'turn_t2': S(), 'oracle_selfcheck': None}\n")
    monkeypatch.setattr(smoke, "SCENARIO_FILE", module)
    monkeypatch.delitem(sys.modules, "agent_speedup_v2_smoke_scenarios", raising=False)
    monkeypatch.setattr(smoke, "SCENARIOS", dict(smoke.SCENARIOS))
    assert smoke.load_scenario("turn_t2").name == "turn_t2"
    assert smoke.SCENARIOS["oracle_selfcheck"] is not None        # BR's file never replaces BH's self-check
    with pytest.raises(ValueError, match="unknown scenario"):
        smoke.load_scenario("nonexistent")
    sys.modules.pop("agent_speedup_v2_smoke_scenarios", None)


def test_synthetic_points_land_on_the_designed_grid():
    points = smoke.synthetic_wrist_points(T2_START_STATE8)
    uv = smoke.r2_project(T2_START_STATE8, points)
    f = robot.focal_length_px(512) * math.tan(math.radians(37.5))
    expected = {(round(255.5 + a * f, 6), round(255.5 - b * f, 6)) for a in (-0.8, 0, 0.8) for b in (-0.8, 0, 0.8)}
    assert {(round(u, 6), round(v, 6)) for u, v in uv} == expected


def test_selfcheck_landing_statuses():
    depth = np.full((512, 512), 0.50)
    mask = np.zeros((512, 512), bool)
    mask[100:110, 100:110] = True
    land = smoke.OracleSelfcheck._land
    assert land("A", [104.0, 104.0], 0.505, True, {}, mask, depth)["status"] == "landed"
    assert land("A", [110.5, 104.0], 0.50, True, {}, mask, depth)["status"] == "landed"      # 1.5 px off
    assert land("A", [114.0, 104.0], 0.50, True, {}, mask, depth)["status"] == "missed"
    assert land("A", [104.0, 104.0], 0.53, True, {}, mask, depth)["status"] == "occluded"    # rendered in front
    assert land("A", [104.0, 104.0], 0.47, True, {}, mask, depth)["status"] == "missed"      # point floats
    assert land("A", None, None, False, None, mask, depth)["status"] == "out_of_view"
    assert land("A", [600.0, 4.0], 0.5, False, None, mask, depth)["status"] == "out_of_view"


class _OracleFakeV2Sim(FakeV2Sim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor_options = {"goal_aware": True, "truth_window": True, "oracle": True}
        self.metadata = {**self.metadata, "oracle_enabled": True}

    def oracle(self, request):
        self.ops.append(("oracle", request["kind"]))
        return {"joints": {"q": {"flat_stove_1_button": 0.6}}}.get(request["kind"], {})


class _Transport:
    offline_mock = True

    def __init__(self, answers):
        self.answers, self.kinds = answers, ["plan", "point_fine"]


class _Client:
    offline_mock = True


class _Policy:
    def __init__(self, client, episode_factory, overrides):
        self.client, self.episode_factory, self.overrides = client, episode_factory, overrides
        self.acts, self.transitions, self.resets = 0, 0, 0

    def reset(self):
        self.resets += 1

    def act(self, observation):
        self.acts += 1
        diagnostics = {"macro": "turned(A)", "primitive": "hold", "relation": {"do": "turn"},
                       "v2_events": [{"event": "wedge"}] if self.acts == 2 else [],
                       "outcomes": [{"n": 1, "relation": "turned", "status": "succeeded"}] if self.acts == 3 else []}
        return Decision((0.0, 0.0, 0.0, 0.0, 0.0, 1e-4, -1.0), diagnostics)

    def observe_transition(self, before, decision, after):
        self.transitions += 1
        return {}

    def agent_stats(self):
        return {"acts": self.acts}


class _Scenario:
    name = "fake_turn"

    def __init__(self):
        self.reported = None

    def factory(self, args):
        return "episode-factory"

    def overrides(self, args):
        return {"first_sense": args.first_sense_value}

    def report(self, sim, policy, outcomes, args):
        self.reported = (policy.acts, list(outcomes))
        return {"passed": True, "truth": sim.oracle({"kind": "joints", "names": ["flat_stove_1_button"]})}


def test_run_policy_scenario_end_to_end_with_fakes(tmp_path, monkeypatch):
    scenario = _Scenario()
    monkeypatch.setitem(smoke.SCENARIOS, "fake_turn", scenario)
    sims = []

    def sim_factory(config, monitor):
        assert monitor.get("oracle") is True
        sims.append(_OracleFakeV2Sim(config, prompt=T2.prompt, task_id=T2.task_id, success_at=5))
        return sims[-1]

    made = {}

    def client_factory(answers, obs, out):
        made["answers"] = answers
        return _Client(), _Transport(answers)

    def policy_factory(client, episode_factory, overrides):
        made["policy"] = _Policy(client, episode_factory, overrides)
        return made["policy"]

    out = tmp_path / "run"
    args = smoke.parse_args(["--task", "libero_10_t2", "--scenario", "fake_turn", "--first-sense", "-1",
                             "--max-steps", "50", "--out", str(out)])
    result = smoke.run(args, sim_factory=sim_factory, client_factory=client_factory, policy_factory=policy_factory)
    policy = made["policy"]
    assert policy.resets == 1 and policy.acts == policy.transitions == 5      # LIBERO success at step 5 ends it
    assert policy.episode_factory == "episode-factory" and policy.overrides == {"first_sense": -1}
    harness = result["harness"]
    assert harness["steps"] == 5 and harness["terminated"] is True and harness["libero_success"] is True
    assert harness["calls"]["by_kind"] == {"plan": 1, "point_fine": 1} and harness["calls"]["real_model_calls"] == 0
    assert harness["events"] == {"wedge": 1} and harness["outcomes"][0]["relation"] == "turned"
    assert harness["first_sense"] == {"contract": -1, "world_z_label": 1}
    assert harness["episode"]["frozen_sha256"] == T2.frozen_reset_sha256[0]
    assert scenario.reported == (5, [{"n": 1, "relation": "turned", "status": "succeeded"}])
    assert sims[0].closed and ("oracle", "joints") in sims[0].ops
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] is True and report["harness"]["scenario"] == "fake_turn"
    assert len((out / "trace.jsonl").read_text().splitlines()) == 5
    assert json.loads((out / "manifest.json").read_text())["zero_model_calls"] is True
    with pytest.raises(FileExistsError):
        smoke.run(args, sim_factory=sim_factory, client_factory=client_factory, policy_factory=policy_factory)


def test_policy_scenarios_refuse_a_live_transport(tmp_path, monkeypatch):
    monkeypatch.setitem(smoke.SCENARIOS, "fake_turn", _Scenario())

    class LiveTransport(_Transport):
        offline_mock = False

    args = smoke.parse_args(["--task", "libero_10_t2", "--scenario", "fake_turn", "--out", str(tmp_path / "r")])
    with pytest.raises(RuntimeError, match="offline"):
        smoke.run(args, sim_factory=lambda c, m: _OracleFakeV2Sim(c, prompt=T2.prompt, task_id=T2.task_id),
                  client_factory=lambda answers, obs, out: (_Client(), LiveTransport(answers)),
                  policy_factory=lambda *a: _Policy(*a))
    assert json.loads((tmp_path / "r" / "report.json").read_text())["passed"] is False


def _selfcheck_run(runs_dir, *, task="libero_10_t2", episode_index=0):
    """The latest (manifest created_utc) oracle_selfcheck run of that task and episode under ``runs_dir``, picked by
    manifest, not by directory name (review BH-5: '..._ep1_002' sorts after '..._002')."""
    chosen = None
    for manifest_path in Path(runs_dir).glob("agent_speedup_v2_w0_bh_oracle_selfcheck_*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (manifest.get("scenario"), manifest.get("task"), manifest.get("episode_index")) != (
                "oracle_selfcheck", task, episode_index) or not (manifest_path.parent / "report.json").is_file():
            continue
        if chosen is None or str(manifest.get("created_utc")) > str(chosen[0].get("created_utc")):
            chosen = (manifest, manifest_path.parent)
    return None if chosen is None else chosen[1]


def test_selfcheck_run_is_picked_by_manifest(tmp_path):
    for name, episode, created in (("001", 0, "20261018T181508Z"), ("002", 0, "20261018T181820Z"),
                                   ("ep1_002", 1, "20261018T181849Z"), ("003", 0, "20261018T170000Z")):
        run = tmp_path / f"agent_speedup_v2_w0_bh_oracle_selfcheck_{name}"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"scenario": "oracle_selfcheck", "task": "libero_10_t2",
                                                       "episode_index": episode, "created_utc": created}))
        (run / "report.json").write_text("{}")
    assert _selfcheck_run(tmp_path).name.endswith("_002") and not _selfcheck_run(tmp_path).name.endswith("ep1_002")
    assert _selfcheck_run(tmp_path, episode_index=1).name.endswith("ep1_002")
    assert _selfcheck_run(tmp_path, episode_index=7) is None


def test_real_sim_selfcheck_report_when_present():
    """The Wave-0 real-sim proof (task 2 episode 0); skipped when no such run directory exists."""
    run = _selfcheck_run(ROOT / "runs")
    if run is None:
        pytest.skip("no task-2 episode-0 oracle_selfcheck run under runs/")
    report = json.loads((run / "report.json").read_text())
    assert report["harness"]["episode"]["episode_index"] == 0 and report["harness"]["task_key"] == "libero_10_t2"
    assert report["passed"] is True and report["failures"] == []
    assert report["wrist_vs_r2_max_err_px"] <= 1.0 and len(report["poses"]) >= 5
    assert report["non_perturbation"]["bitwise_identical"] is True
    assert all(report["landed_counts"][camera][pid] >= 1 for camera in ("wrist", "external") for pid in "ABC")
    if report.get("motion_lag"):                    # review BH-2 rows (harness v2 on): reported, never gated
        assert report["motion_lag"]["gated"] is False and report["motion_lag"]["settled"]["r2_vs_oracle_max_px"] <= 1.0
