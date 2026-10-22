"""Real-simulator, zero-model-call relation-smoke harness for agent_SpeedUp_v2 (TEST TOOL; contract §C14.5, T2).

A diagnosis tool, never a scored run. It starts ``V2SimProcess`` with the evaluator-only ORACLE op enabled, resets
one episode (episodes without a frozen reset hash are accepted; the measured sha256 is recorded), runs one
scenario and writes ``report.json`` (the scenario's diagnosis-only report plus the harness's own fields: episode
sha256, steps, calls, the ORACLE report) and ``manifest.json`` into a FRESH output directory.

usage (project venv; the worker runs in --sim-python; one _LiberoWorker per worker process):
  env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python scripts/agent_speedup_v2_relation_smoke.py \\
      --task libero_10_t2 --episode-index 0 --scenario oracle_selfcheck [--out DIR]
  ... --scenario turn_t2 --first-sense +1   (BR's scenarios, scripts/agent_speedup_v2_smoke_scenarios.py)

Scenario plug-in interface (BH defines, BR fills; ``Scenario`` below):
- ``factory(args)`` -> the episode generator factory ``Callable[[RelationContext], Generator]``;
- ``overrides(args)`` -> {first_sense, depth_rule, forced_yaw_rad} (only those; offline-mock clients only);
- ``report(sim, policy, outcomes, args)`` -> diagnosis-only truth read live through ``sim.oracle(...)``.
- optional ``uses_policy = False`` + ``drive(ctx)``: a policy-free scenario that drives the robot itself
  (``oracle_selfcheck``). Absent ``uses_policy`` means True.
``SCENARIOS`` holds ``oracle_selfcheck``; BR's file is imported only when another scenario is requested, so this
harness runs before BR's file exists.

Policy scenarios run the §C3.4 loop on the real simulator without video:
``AgentSpeedUpV2Policy(client=AgentSpeedUpV2Client(..., transport=ScriptedV2Transport(OracleAnswers(sim, key))),
episode_factory=scenario.factory(args), overrides=scenario.overrides(args))``.

``oracle_selfcheck`` (policy-free, robot-only scripted moves, no object contact) proves on the real task that:
1. the ORACLE's wrist projection agrees with the R-2 model (camera = hand + 0.05 R0, hand = grip - 0.097 R2,
   f = 333.6 px at 512) within 1 px at >= 5 poses (synthetic camera-frame points plus the part points);
2. the projected part points land on the parts in both cameras: within 2 px of a pixel whose segmentation is the
   part (body/owner mask) AND whose rendered depth is within 1 cm of the point's depth (the knob lever, the handle
   loop and the burner are required; the pot and the pan handle are reported). Status per (pose, camera, part):
   landed / occluded (something renders in front) / out_of_view / missed (a failure);
3. ``heights`` ignores the robot: at the grip xy (the hand directly above) the robot-skipped surface is a scene
   geom > 5 cm below the hand, while ``include_robot`` hits a robot body;
4. the oracle's upright render matches the policy's upright images (flip convention) and the part points lie on
   the collision surface (``points`` vs ``heights``);
5. ``OracleAnswers`` answers: plan cells overlap the part's visible mask (k = 0..3), fine points map back within
   the fine quantization (sqrt(2) x 8 px) and answer 0 outside the crop, coarse cells within sqrt(2) x 32 px;
6. non-perturbation: a second worker replays the identical action list WITHOUT oracle queries and every
   observation sha256 matches.
Reported, not gated: ``motion_lag`` (review BH-2: full-speed moves measured every step; the oracle reads the
post-refresh kinematics, one substep ahead of the observation, so compare against ORACLE truth on settled frames
only), each part's G1 truth candidates at its anchor xy (review BH-1: aabb z, collision surface, each camera's
visual surface), and region-limited part masks (review BH-3: C is the handle loop's boxes, not the whole pot).
Overlay PNGs (masks, projected points, R-2 points) per pose and camera go to ``<out>/overlays``.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from gpt_field import agent_speedup_v2_robot as robot  # noqa: E402
from gpt_field.agent_speedup_v2_sim import SIM_VERSION, V2SimProcess, initial_observation_sha256  # noqa: E402
from gpt_field.agent_speedup_v2_tasks import (  # noqa: E402
    DEFAULT_LIBERO_ROOT, DEFAULT_SIM_PYTHON, IMAGE_SIZE, V2_TASK_ALIASES, V2_TASK_KEYS, build_v2_sim_config,
    resolve_v2_task)
from gpt_field.contact_skills import rotation_matrix  # noqa: E402

HARNESS_VERSION = "agent-speedup-v2-relation-smoke-2"
ORACLE_FILE = ROOT / "scripts" / "agent_speedup_v2_oracle.py"
SCENARIO_FILE = ROOT / "scripts" / "agent_speedup_v2_smoke_scenarios.py"
FIRST_SENSES = {"+1": 1, "-1": -1}
DEPTH_RULES = ("auto", "touch_stop", "height_stop")
FORCED_YAWS = ("none", "jaw_y", "jaw_x")
DEFAULT_MAX_STEPS = 1300
HOLD_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------------------------------------
# plug-in interface
class Scenario(Protocol):
    name: str

    def factory(self, args) -> Callable[[Any], Any]: ...      # Callable[[RelationContext], Generator]

    def overrides(self, args) -> dict: ...                     # first_sense / depth_rule / forced_yaw_rad

    def report(self, sim, policy, outcomes: list, args) -> dict: ...


def _load_file_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def oracle_module():
    """scripts/agent_speedup_v2_oracle.py (test-only; never imported by a control module)."""
    return _load_file_module("agent_speedup_v2_oracle", ORACLE_FILE)


def load_scenario(name):
    """``SCENARIOS[name]``; BR's scenario file is imported only for a scenario other than oracle_selfcheck."""
    if name not in SCENARIOS:
        if not SCENARIO_FILE.is_file():
            raise ValueError(f"scenario {name!r} needs {SCENARIO_FILE.name} (BR, Wave 2), not present")
        module = _load_file_module("agent_speedup_v2_smoke_scenarios", SCENARIO_FILE)
        SCENARIOS.update({key: value for key, value in module.SCENARIOS.items() if key not in SCENARIOS})
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


# ---------------------------------------------------------------------------------------------------------
# small helpers
def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_out(scenario):
    return ROOT / "runs" / f"agent_speedup_v2_smoke_{scenario}_{utc_stamp()}"


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=1, allow_nan=False) + "\n")


def file_sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def observation_sha256(observation):
    return initial_observation_sha256(observation.to_dict())


def decode_png(url):
    from PIL import Image
    raw = base64.b64decode(url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


def world_z_label(sense, state8):
    """SP/CP's world-z sign of a contract twist sense (about R2) at this hand pose (§C1.3)."""
    r2_z = float(rotation_matrix(np.asarray(state8[3:6], float))[2, 2])
    return int(sense * (1 if r2_z >= 0 else -1))


# ---------------------------------------------------------------------------------------------------------
# simulator plumbing
def start_sim(task_key, *, oracle=True, sim_python=DEFAULT_SIM_PYTHON, libero_root=DEFAULT_LIBERO_ROOT,
              sim_factory=None):
    spec = resolve_v2_task(task_key)
    config, monitor = build_v2_sim_config(spec.key, sim_python=sim_python, libero_root=libero_root)
    monitor = {**monitor, "oracle": True} if oracle else dict(monitor)
    factory = sim_factory or (lambda cfg, options: V2SimProcess(cfg, monitor_options=options))
    sim = factory(config, monitor)
    sim.start()
    return sim, spec, config, monitor


def reset_episode(sim, spec, episode_index):
    tasks = sim.list_tasks()
    official = [t["prompt"] for t in tasks if t.get("task_id") == spec.task_id]
    if official != [spec.prompt]:
        raise RuntimeError(f"simulator task {spec.task_id} prompt {official} differs from the registry")
    obs = sim.reset(spec.task_id, episode_index)
    if obs.prompt != spec.prompt:
        raise RuntimeError("reset observation prompt differs from the registry")
    sha = observation_sha256(obs)
    frozen = spec.frozen_reset_sha256.get(episode_index)
    return obs, {"episode_index": episode_index, "sha256": sha, "frozen_sha256": frozen,
                 "matches_frozen": (sha == frozen) if frozen else None, "prompt": obs.prompt,
                 "initial_state8": list(obs.state)}


@dataclass
class HarnessContext:
    sim: Any
    spec: Any
    args: Any
    out: Path
    obs: Any
    episode: dict
    restart: Callable[..., Any]            # restart(oracle: bool) -> (sim, spec, config, monitor) of a NEW worker
    steps: int = 0
    actions: list = field(default_factory=list)
    observation_shas: list = field(default_factory=list)

    def step(self, action):
        self.obs, _success = self.sim.step(list(action))
        self.steps += 1
        self.actions.append(list(action))
        self.observation_shas.append(observation_sha256(self.obs))
        return self.obs


# ---------------------------------------------------------------------------------------------------------
# R-2 wrist model (robot facts R1-R3) and overlays
def r2_camera(state8):
    R = rotation_matrix(np.asarray(state8[3:6], float))
    grip = np.asarray(state8[:3], float)
    centre = grip - robot.GRIP_SITE_ALONG_R2_M * R[:, 2] + robot.WRIST_CAMERA_ALONG_R0_M * R[:, 0]
    return centre, R


def r2_project(state8, xyz, width=IMAGE_SIZE):
    centre, R = r2_camera(state8)
    f, c = robot.focal_length_px(width), (width - 1) / 2.0
    out = []
    for point in np.asarray(xyz, float).reshape(-1, 3):
        d = point - centre
        z = float(d @ R[:, 2])
        out.append(None if z <= 1e-9 else [c + f * float(d @ R[:, 1]) / z, c - f * float(d @ R[:, 0]) / z])
    return out


def synthetic_wrist_points(state8, depths=(0.10, 0.25, 0.60), spread=(-0.8, 0.0, 0.8)):
    """World points on a camera-frame grid of the R-2 camera (all in view by construction)."""
    centre, R = r2_camera(state8)
    t = math.tan(math.radians(float(robot.ROBOT_FACTS["asset"]["wrist_camera"]["fovy_deg"])) / 2.0)
    return [(centre + d * (R[:, 2] + a * t * R[:, 1] + b * t * R[:, 0])).tolist()
            for d in depths for a in spread for b in spread]


PART_COLOURS = {"A": (255, 140, 0), "B": (220, 30, 220), "C": (0, 200, 255), "D": (60, 220, 60),
                "E": (255, 230, 0)}


def draw_overlay(rgb, path, *, masks=(), marks=(), title=""):
    """masks: (bool HxW, rgb); marks: (label, uv, rgb, style in {"cross", "circle", "dot"})."""
    from PIL import Image, ImageDraw
    base = np.asarray(rgb, dtype=float).copy()
    for mask, colour in masks:
        if mask is None:
            continue
        m = np.asarray(mask, bool)
        base[m] = 0.6 * base[m] + 0.4 * np.asarray(colour, float)
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        base[edge] = colour
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for label, uv, colour, style in marks:
        if uv is None:
            continue
        u, v = float(uv[0]), float(uv[1])
        if style == "cross":
            draw.line((u - 7, v, u + 7, v), fill=(0, 0, 0), width=3)
            draw.line((u, v - 7, u, v + 7), fill=(0, 0, 0), width=3)
            draw.line((u - 6, v, u + 6, v), fill=colour, width=1)
            draw.line((u, v - 6, u, v + 6), fill=colour, width=1)
            if label:
                draw.text((u + 6, v - 14), label, fill=colour)
        elif style == "circle":
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=colour, width=1)
        else:
            draw.point((u, v), fill=colour)
    if title:
        draw.rectangle((0, 0, 8 + 6 * len(title), 13), fill=(0, 0, 0))
        draw.text((3, 1), title, fill=(255, 255, 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# ---------------------------------------------------------------------------------------------------------
# oracle_selfcheck (BH): policy-free, zero model calls, robot-only moves
SELFCHECK_WRIST_TOL_PX = 1.0        # contract §C14.2: within 1 px of the R-2 model
SELFCHECK_LAND_TOL_PX = 2.0         # a projected point lands on its part within 2 px
SELFCHECK_DEPTH_TOL_M = 0.01        # ... at a pixel whose rendered depth is within 1 cm of the point's depth
SELFCHECK_HAND_CLEAR_M = 0.05       # heights at the grip xy: robot-skipped surface >= 5 cm below the hand
SELFCHECK_SURFACE_TOL_M = 0.002     # part point z vs the collision surface at its xy
SELFCHECK_RENDER_TOL_GREY = 1.0     # mean |oracle render - policy upright image|
SELFCHECK_MIN_POSES = 5
SELFCHECK_REQUIRED_PARTS = ("A", "B", "C")
FINE_QUANT_TOL_PX = math.sqrt(2.0) * 8.0 + 0.5
COARSE_QUANT_TOL_PX = math.sqrt(2.0) * 32.0 + 0.5
# (name, steps, action, settle steps): small robot-only moves at ~1.1-1.2 m, away from every object
POSE_SCRIPT = (
    ("start", 0, HOLD_OPEN, 0),
    ("shift_x", 12, (0.4, -0.15, 0.0, 0.0, 0.0, 0.0, -1.0), 10),
    ("yaw", 10, (0.0, 0.0, 0.0, 0.0, 0.0, 0.35, -1.0), 10),
    ("tilt", 6, (0.0, 0.0, 0.0, 0.2, -0.15, 0.0, -1.0), 10),
    ("descend", 10, (-0.2, 0.15, -0.35, 0.0, 0.0, -0.2, -1.0), 10),
    ("back", 10, (-0.3, 0.0, 0.25, -0.2, 0.15, 0.0, -1.0), 10),
)
# Kinematic lag rows (review BH-2; information, not gated): full-scale moves from the "back" pose (grip ~1.17 m,
# >= 10 cm above every object top), measured at every step, then MOTION_SETTLE_STEPS of hold and one settled row.
MOTION_SCRIPT = (
    ("fast_x", 3, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
    ("fast_yaw", 3, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0)),
    ("fast_yaw_back", 3, (0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0)),
    ("fast_x_back", 3, (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
)
MOTION_SETTLE_STEPS = 15
MOTION_GRIP_SITE = "gripper0_grip_site"     # state8 xyz is this site's position (robot facts R1)


class OracleSelfcheck:
    name = "oracle_selfcheck"
    uses_policy = False

    def factory(self, args):
        return None

    def overrides(self, args):
        return {}

    def report(self, sim, policy, outcomes, args):
        return dict(getattr(self, "_result", None) or {"passed": False, "error": "drive() did not run"})

    # ------------------------------------------------------------------------------------------------
    def drive(self, ctx: HarnessContext) -> dict:
        oracle = oracle_module()
        answers = oracle.OracleAnswers(ctx.sim, ctx.spec.key)
        parts = oracle.ORACLE_PARTS[answers.task_key]
        poses = []
        for index, (name, steps, action, settle) in enumerate(POSE_SCRIPT):
            for _ in range(steps):
                ctx.step(action)
            for _ in range(settle):
                ctx.step(HOLD_OPEN)
            poses.append(self._measure(ctx, oracle, answers, parts, index, name))
        motion = self._motion_lag(ctx)
        replay = self._replay(ctx)
        self._result = self._summarize(poses, replay, answers, ctx, motion)
        return self._result

    # kinematic lag while moving (review BH-2): information, never a gate ------------------------------
    @staticmethod
    def _motion_row(ctx, label, moving):
        from gpt_field.contact_skills import StepView
        sim, obs = ctx.sim, ctx.obs
        state8 = list(obs.state)
        synth = synthetic_wrist_points(state8)
        probe = sim.oracle({"kind": "project", "camera": "wrist", "xyz": synth})
        model = r2_project(state8, synth)
        errors = [math.dist(a, b) for a, b, inside in zip(probe["uv"], model, probe["in_image"])
                  if a is not None and b is not None and inside]
        seg = sim.oracle({"kind": "segment", "camera": "wrist", "masks": [], "mask_png": False, "rgb": True})
        render = np.abs(decode_png(seg["rgb_png"]).astype(float)
                        - np.asarray(StepView(obs).wrist_rgb_upright).astype(float)).mean()
        grip = sim.oracle({"kind": "points", "items": [{"name": "grip", "site": MOTION_GRIP_SITE,
                                                         "where": "origin"}]})["points"]["grip"]
        return {"label": label, "step": ctx.steps, "moving": moving, "grip_xyz_state": state8[:3],
                "r2_vs_oracle_max_px": max(errors) if errors else None,
                "r2_vs_oracle_mean_px": float(np.mean(errors)) if errors else None,
                "render_vs_policy_mean_abs_grey": float(render),
                "grip_site_vs_state_mm": 1000.0 * math.dist(grip, state8[:3])}

    def _motion_lag(self, ctx):
        """Full-speed robot-only moves measured at EVERY step without settling, then once after settling: the
        oracle (post-refresh) vs the observation (pre-refresh) gap. Reported, not gated (review BH-2)."""
        rows = []
        for name, steps, action in MOTION_SCRIPT:
            for i in range(steps):
                ctx.step(action)
                rows.append(self._motion_row(ctx, f"{name}_{i}", True))
        for _ in range(MOTION_SETTLE_STEPS):
            ctx.step(HOLD_OPEN)
        rows.append(self._motion_row(ctx, "settled", False))

        def worst(key, moving):
            values = [row[key] for row in rows if row["moving"] is moving and row[key] is not None]
            return max(values) if values else None

        return {"gated": False, "rows": rows,
                "moving_max": {key: worst(key, True) for key in ("r2_vs_oracle_max_px", "render_vs_policy_mean_abs_grey",
                                                                 "grip_site_vs_state_mm")},
                "settled": {key: worst(key, False) for key in ("r2_vs_oracle_max_px",
                                                               "render_vs_policy_mean_abs_grey",
                                                               "grip_site_vs_state_mm")},
                "note": "the oracle reads post-refresh kinematics, one physics substep ahead of the observation; "
                        "compare observation-derived estimates with ORACLE truth on settled frames only, or budget "
                        "the moving_max lag"}

    # one pose ---------------------------------------------------------------------------------------
    def _measure(self, ctx, oracle, answers, parts, index, name):
        sim, obs = ctx.sim, ctx.obs
        state8 = list(obs.state)
        from gpt_field.contact_skills import StepView
        view = StepView(obs)
        upright = {"wrist": np.asarray(view.wrist_rgb_upright), "external": np.asarray(view.external_rgb_upright)}
        truth = answers.truth_points(detail=True)
        pids = list(parts)
        xyz = [truth[pid]["xyz"] for pid in pids]
        # 1. wrist: ORACLE (MuJoCo camera) vs the R-2 model
        synth = synthetic_wrist_points(state8)
        probe = sim.oracle({"kind": "project", "camera": "wrist", "xyz": synth + xyz})
        model = r2_project(state8, synth + xyz)
        errors = [math.dist(a, b) for a, b, inside in zip(probe["uv"], model, probe["in_image"])
                  if a is not None and b is not None and inside]
        wrist_check = {"points": len(errors), "max_err_px": max(errors) if errors else None,
                       "mean_err_px": float(np.mean(errors)) if errors else None,
                       "synthetic_points": len(synth)}
        # 2. part points in both cameras: segmentation + depth
        cameras = {}
        mask_items = [{"name": pid, **spec["mask"]} for pid, spec in parts.items()] + [{"name": "robot", "robot": True}]
        for camera in ("wrist", "external"):
            projected = sim.oracle({"kind": "project", "camera": camera, "xyz": xyz})
            seg = sim.oracle({"kind": "segment", "camera": camera, "uv": [uv for uv in projected["uv"] if uv],
                              "masks": mask_items, "depth_png": True, "rgb": index == 0})
            depth = decode_png(seg["depth_png"]).astype(float) * float(seg["depth_png_unit_m"])
            masks = {key: (decode_png(m["png"]) > 0 if m["png"] else None) for key, m in seg["masks"].items()}
            # review BH-3: a part with a ``region`` is checked against its own boxes (the handle loop, not the pot)
            masks.update({pid: answers.restrict_to_region(pid, camera, masks.get(pid)) for pid in pids})
            at = iter(seg["at"])
            rows = {}
            for pid, uv, d, inside in zip(pids, projected["uv"], projected["depth_m"], projected["in_image"]):
                pixel = next(at) if uv else None
                rows[pid] = self._land(pid, uv, d, inside, pixel, masks.get(pid), depth)
            render = None
            if seg.get("rgb_png"):
                render = float(np.abs(decode_png(seg["rgb_png"]).astype(float) - upright[camera].astype(float)).mean())
            cameras[camera] = {"parts": rows, "uv": dict(zip(pids, projected["uv"])),
                               "mask_pixels": {k: m["pixels"] for k, m in seg["masks"].items()},
                               "region_mask_pixels": {pid: (int(masks[pid].sum()) if masks.get(pid) is not None
                                                            else 0) for pid in pids},
                               "mask_bbox": {k: m["bbox"] for k, m in seg["masks"].items()},
                               "render_vs_policy_mean_abs_grey": render, "_masks": masks}
        # 3. heights ignore the robot (the hand is directly above the grip xy)
        grip_xy = state8[:2]
        skipped = sim.oracle({"kind": "heights", "xy": [grip_xy]})
        with_robot = sim.oracle({"kind": "heights", "xy": [grip_xy], "include_robot": True})
        z0, z1 = skipped["z"][0], with_robot["z"][0]
        heights_check = {
            "grip_z": state8[2], "robot_skipped_z": z0, "robot_skipped_hit": skipped["hit"][0],
            "include_robot_z": z1, "include_robot_hit": with_robot["hit"][0],
            "ok": bool(z0 is not None and z1 is not None and not skipped["hit"][0]["robot"]
                       and with_robot["hit"][0]["robot"] and z0 <= state8[2] - SELFCHECK_HAND_CLEAR_M
                       and z1 >= z0 + SELFCHECK_HAND_CLEAR_M)}
        # 4. points vs heights: the vertical ray at the part point's xy first meets the part's OWN geom(s), and the
        #    point sits on that surface (its aabb top floats above it when the box is tilted)
        # the three G1 truth candidates at each anchor xy (review BH-1, reported for the lead's ruling): aabb z,
        # the collision surface, and each camera's visual (rendered) surface
        candidates = dict(zip(pids, answers.surface_truth([truth[pid]["xyz"][:2] for pid in pids])))
        surface = {}
        for pid in pids:
            geoms = parts[pid]["item"]["geom"]
            geoms = [geoms] if isinstance(geoms, str) else list(geoms)
            hit = truth[pid]["surface_hit"] or {}
            surface[pid] = {"z_source": truth[pid]["z_source"], "aabb_z": truth[pid]["aabb_z"],
                            "surface_z": truth[pid]["surface_z"], "surface_geom": hit.get("geom"),
                            "aabb_above_surface_mm": None if truth[pid]["surface_z"] is None else
                            1000.0 * (truth[pid]["aabb_z"] - truth[pid]["surface_z"]),
                            "on_surface": hit.get("geom") in geoms and abs(
                                truth[pid]["xyz"][2] - truth[pid]["surface_z"]) <= SELFCHECK_SURFACE_TOL_M,
                            "visual_z": {camera: (row or {}).get("z") for camera, row in
                                         candidates[pid]["visual"].items()},
                            "visual_minus_surface_mm": {
                                camera: (None if not row or candidates[pid]["surface_z"] is None else
                                         1000.0 * (row["z"] - candidates[pid]["surface_z"]))
                                for camera, row in candidates[pid]["visual"].items()},
                            "visual_owner": {camera: (row or {}).get("owner") for camera, row in
                                             candidates[pid]["visual"].items()}}
        # 5. OracleAnswers at this pose
        answer_checks = self._answer_checks(oracle, answers, cameras, pids)
        self._overlays(ctx, index, name, upright, cameras, probe, model, len(synth), pids)
        for camera in cameras.values():
            camera.pop("_masks")
        return {"index": index, "name": name, "step": ctx.steps, "state8": state8,
                "truth_points": {pid: truth[pid]["xyz"] for pid in pids}, "wrist_vs_r2": wrist_check,
                "cameras": cameras, "heights_ignore_robot": heights_check, "point_surface": surface,
                "answers": answer_checks}

    @staticmethod
    def _land(pid, uv, point_depth, inside, pixel, mask, depth):
        row = {"uv": uv, "point_depth_m": point_depth, "in_image": bool(inside), "pixel": pixel}
        if uv is None or not inside or mask is None:
            row.update(status="out_of_view", land_px=None)
            return row
        size = depth.shape[0]
        r = int(math.ceil(SELFCHECK_LAND_TOL_PX))
        u0, v0 = float(uv[0]), float(uv[1])
        best = None
        for dv in range(-r - 1, r + 2):
            for du in range(-r - 1, r + 2):
                iu, iv = int(round(u0)) + du, int(round(v0)) + dv
                if not (0 <= iu < size and 0 <= iv < size) or not mask[iv, iu]:
                    continue
                if abs(depth[iv, iu] - point_depth) > SELFCHECK_DEPTH_TOL_M:
                    continue
                dist = math.hypot(iu - u0, iv - v0)
                best = dist if best is None else min(best, dist)
        iu, iv = int(round(u0)), int(round(v0))
        depth_err = float(depth[iv, iu] - point_depth)
        row.update(land_px=best, depth_err_at_pixel_m=depth_err, on_mask_at_pixel=bool(mask[iv, iu]))
        if best is not None and best <= SELFCHECK_LAND_TOL_PX:
            row["status"] = "landed"
        elif depth_err < -SELFCHECK_DEPTH_TOL_M:
            row["status"] = "occluded"          # something renders in front of the point
        else:
            row["status"] = "missed"
        return row

    @staticmethod
    def _answer_checks(oracle, answers, cameras, pids):
        width = IMAGE_SIZE
        out = {"plan": [], "point_fine": [], "point_coarse": [], "failures": []}
        cell = width // oracle.GRID
        for turns in range(4):
            plan = answers.answer("plan", {"kind": "plan", "quarter_turns": turns, "image_size": width})
            for part in plan["parts"]:
                pid = part["id"]
                for key, camera, k in (("w", "wrist", turns), ("x", "external", 0)):
                    value = part[key]
                    mask = cameras[camera]["_masks"].get(pid)
                    row = {"pid": pid, "grid": key, "quarter_turns": k, "cell": value}
                    if value and mask is not None:
                        r, c = divmod(value - 1, oracle.GRID)
                        turned = np.rot90(mask, k)
                        row["mask_px_in_cell"] = int(turned[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].sum())
                        if row["mask_px_in_cell"] == 0:
                            out["failures"].append(f"plan {key}={value} of {pid} (k={k}) shows none of the part")
                    out["plan"].append(row)
        for camera, k in (("wrist", 2), ("external", 0)):
            for pid in pids:
                uv = cameras[camera]["uv"].get(pid)
                if uv is None or cameras[camera]["parts"][pid]["status"] == "out_of_view":
                    continue
                near = [uv[0] + 17.0, uv[1] - 11.0]
                far = [uv[0] + 100.0 if uv[0] < width / 2 else uv[0] - 100.0, uv[1]]
                for label, centre in (("near", near), ("far", far)):
                    answer = answers.answer("point_fine", {"kind": "point_fine", "pid": pid, "camera": camera,
                                                           "quarter_turns": k, "center_uv": centre,
                                                           "crop_px": oracle.FINE_CROP_PX, "image_size": width})
                    record = answers.records[-1]
                    row = {"pid": pid, "camera": camera, "centre": label, "point_id": answer["point_id"],
                           "in_crop": record["in_crop"], "error_px": record["error_px"]}
                    if label == "near" and not (answer["point_id"] and record["error_px"] <= FINE_QUANT_TOL_PX):
                        out["failures"].append(f"point_fine near {pid}/{camera}: {row}")
                    if label == "far" and record["in_crop"] is False and answer["point_id"] != 0:
                        out["failures"].append(f"point_fine outside the crop must answer 0: {row}")
                    out["point_fine"].append(row)
                answer = answers.answer("point_coarse", {"kind": "point_coarse", "pid": pid, "camera": camera,
                                                         "quarter_turns": k, "image_size": width})
                record = answers.records[-1]
                row = {"pid": pid, "camera": camera, "cell_id": answer["cell_id"], "error_px": record["error_px"]}
                if not (answer["cell_id"] and record["error_px"] <= COARSE_QUANT_TOL_PX):
                    out["failures"].append(f"point_coarse {pid}/{camera}: {row}")
                out["point_coarse"].append(row)
        return out

    @staticmethod
    def _overlays(ctx, index, name, upright, cameras, probe, model, n_synth, pids):
        folder = ctx.out / "overlays"
        for camera in ("wrist", "external"):
            info = cameras[camera]
            masks = [(info["_masks"].get(pid), PART_COLOURS[pid]) for pid in pids if pid in ("A", "B", "C", "E")]
            marks = [(pid, info["uv"][pid], PART_COLOURS[pid], "cross") for pid in pids]
            if camera == "wrist":
                marks += [("", uv, (255, 255, 255), "circle") for uv in model[n_synth:]]
                marks += [("", uv, (255, 60, 60), "dot") for uv in probe["uv"][:n_synth]]
            status = " ".join(f"{pid}:{info['parts'][pid]['status'][:4]}" for pid in pids)
            draw_overlay(upright[camera], folder / f"pose{index}_{name}_{camera}.png", masks=masks, marks=marks,
                         title=f"pose {index} {name} {camera} | {status}")

    # non-perturbation -------------------------------------------------------------------------------
    @staticmethod
    def _replay(ctx):
        sim, spec, _config, _monitor = ctx.restart(oracle=False)
        try:
            obs, episode = reset_episode(sim, spec, ctx.episode["episode_index"])
            differing = []
            for i, action in enumerate(ctx.actions):
                obs, _ = sim.step(action)
                if observation_sha256(obs) != ctx.observation_shas[i]:
                    differing.append(i)
            refusals = {}
            try:                                   # the process client refuses before sending ...
                sim.oracle({"kind": "joints", "names": ["flat_stove_1_button"]})
                refusals["process_client"] = False
            except PermissionError:
                refusals["process_client"] = True
            try:                                   # ... and the worker refuses on its own (raw JSONL request)
                sim._request({"op": "oracle", "request": {"kind": "joints", "names": ["flat_stove_1_button"]}})
                refusals["worker"] = False
            except Exception as exc:               # SimulatorError("simulator worker: PermissionError: ...")
                refusals["worker"] = "PermissionError: oracle op disabled" in str(exc)
        finally:
            sim.close()
        return {"reset_sha256_equal": episode["sha256"] == ctx.episode["sha256"], "steps": len(ctx.actions),
                "differing_steps": differing[:20], "n_differing": len(differing),
                "bitwise_identical": episode["sha256"] == ctx.episode["sha256"] and not differing,
                "oracle_refusals_without_option": refusals,
                "oracle_refused_without_option": bool(refusals) and all(refusals.values())}

    # summary ----------------------------------------------------------------------------------------
    @staticmethod
    def _summarize(poses, replay, answers, ctx, motion=None):
        failures = []
        wrist = [p["wrist_vs_r2"]["max_err_px"] for p in poses if p["wrist_vs_r2"]["max_err_px"] is not None]
        if len(wrist) < SELFCHECK_MIN_POSES:
            failures.append(f"wrist R-2 comparison at {len(wrist)} poses (< {SELFCHECK_MIN_POSES})")
        if wrist and max(wrist) > SELFCHECK_WRIST_TOL_PX:
            failures.append(f"wrist projection differs from the R-2 model by {max(wrist):.3f} px")
        landed = {camera: {pid: 0 for pid in SELFCHECK_REQUIRED_PARTS} for camera in ("wrist", "external")}
        for pose in poses:
            for camera, info in pose["cameras"].items():
                for pid, row in info["parts"].items():
                    if row["status"] == "missed":
                        failures.append(f"pose {pose['index']} {camera}: part {pid} missed ({row})")
                    if row["status"] == "landed" and pid in landed[camera]:
                        landed[camera][pid] += 1
                render = info["render_vs_policy_mean_abs_grey"]
                if render is not None and render > SELFCHECK_RENDER_TOL_GREY:
                    failures.append(f"{camera} oracle render differs from the policy image by {render:.2f} grey")
            if not pose["heights_ignore_robot"]["ok"]:
                failures.append(f"pose {pose['index']}: heights did not ignore the robot")
            for pid in SELFCHECK_REQUIRED_PARTS:
                if not pose["point_surface"][pid]["on_surface"]:
                    failures.append(f"pose {pose['index']}: part {pid} point is not on its collision surface")
            failures.extend(f"pose {pose['index']}: {text}" for text in pose["answers"]["failures"])
        for camera, counts in landed.items():
            for pid, count in counts.items():
                if count == 0:
                    failures.append(f"part {pid} never landed in the {camera} camera")
        for pid in SELFCHECK_REQUIRED_PARTS:
            if poses[0]["cameras"]["external"]["parts"][pid]["status"] != "landed":
                failures.append(f"start pose external: part {pid} did not land")
        if not replay["bitwise_identical"]:
            failures.append(f"oracle queries perturbed the episode ({replay['n_differing']} differing steps)")
        if not replay["oracle_refused_without_option"]:
            failures.append("the oracle op was not refused without the monitor option")
        status_table = {f"pose{p['index']}_{p['name']}": {camera: {pid: row["status"] for pid, row in info["parts"].items()}
                                                         for camera, info in p["cameras"].items()} for p in poses}
        return {"scenario": "oracle_selfcheck", "passed": not failures, "failures": failures,
                "poses": poses, "status_table": status_table, "landed_counts": landed,
                "wrist_vs_r2_max_err_px": max(wrist) if wrist else None, "non_perturbation": replay,
                "motion_lag": motion,
                "tolerances": {"wrist_px": SELFCHECK_WRIST_TOL_PX, "land_px": SELFCHECK_LAND_TOL_PX,
                               "depth_m": SELFCHECK_DEPTH_TOL_M, "hand_clear_m": SELFCHECK_HAND_CLEAR_M,
                               "surface_m": SELFCHECK_SURFACE_TOL_M, "render_grey": SELFCHECK_RENDER_TOL_GREY,
                               "fine_px": FINE_QUANT_TOL_PX, "coarse_px": COARSE_QUANT_TOL_PX},
                "pose_script": [list(map(jsonable, row)) for row in POSE_SCRIPT],
                "oracle_report_counts": answers.report()["counts"]}


SCENARIOS: dict = {"oracle_selfcheck": OracleSelfcheck()}


# ---------------------------------------------------------------------------------------------------------
# policy scenarios (§C3.4 loop on the real simulator, no video)
def _m0_runner():
    return _load_file_module("_agent_speedup_v2_runner_readonly", ROOT / "scripts" / "evaluate_agent_speedup_v2.py")


def default_client_factory(answers, obs, out):
    """The v2 client with the ORACLE behind ScriptedV2Transport (BC, Wave 1); offline only."""
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, ScriptedV2Transport
    runner = _m0_runner()
    transport = ScriptedV2Transport(answers)
    client = AgentSpeedUpV2Client(runner.configuration(), task_prompt=obs.prompt,
                                  token_provider=runner._v1.offline_token_provider, transport=transport,
                                  log_dir=out / "client")
    return client, transport


def default_policy_factory(client, episode_factory, overrides):
    from gpt_field.agent_speedup_v2 import AgentSpeedUpV2Policy
    return AgentSpeedUpV2Policy(client, episode_factory=episode_factory, overrides=overrides)


def run_policy_scenario(ctx: HarnessContext, scenario, *, answers=None, client_factory=None, policy_factory=None):
    """Build client + policy, run act / step / observe_transition until termination or --max-steps, then call
    ``scenario.report`` while the simulator is still open. Returns (report, harness_fields)."""
    oracle = oracle_module()
    answers = answers or oracle.OracleAnswers(ctx.sim, ctx.spec.key)
    client, transport = (client_factory or default_client_factory)(answers, ctx.obs, ctx.out)
    if getattr(transport, "offline_mock", False) is not True or getattr(client, "offline_mock", True) is not True:
        raise RuntimeError("relation smokes run only with an offline scripted transport (zero model calls)")
    policy = (policy_factory or default_policy_factory)(client, scenario.factory(ctx.args),
                                                         dict(scenario.overrides(ctx.args) or {}))
    policy.reset()
    outcomes, events = [], {}
    trace_path = ctx.out / "trace.jsonl"
    obs = ctx.obs
    started = time.monotonic()
    with trace_path.open("w") as trace:
        for _ in range(ctx.args.max_steps):
            decision = policy.act(obs)
            diagnostics = decision.diagnostics or {}
            step_outcomes = list(diagnostics.get("outcomes") or [])
            outcomes.extend(step_outcomes)
            for row in diagnostics.get("v2_events") or []:
                events[row.get("event")] = events.get(row.get("event"), 0) + 1
            before = obs
            obs = ctx.step(decision.action)
            policy.observe_transition(before, decision, obs)
            trace.write(json.dumps(jsonable({
                "step": ctx.steps, "action": list(decision.action), "state8": list(obs.state),
                "macro": diagnostics.get("macro"), "primitive": diagnostics.get("primitive"),
                "relation": diagnostics.get("relation"), "api_calls_in_act": diagnostics.get("api_calls_in_act"),
                "v2_events": diagnostics.get("v2_events"), "outcomes": step_outcomes,
                "sim_status": {k: ctx.sim.status.get(k) for k in ("steps", "success", "terminated",
                                                                  "libero_success")}}), allow_nan=False) + "\n")
            if ctx.sim.status.get("terminated"):
                break
    report = scenario.report(ctx.sim, policy, outcomes, ctx.args)
    kinds = list(getattr(transport, "kinds", []) or [])
    fields = {"steps": ctx.steps, "terminated": bool(ctx.sim.status.get("terminated")),
              "libero_success": bool(ctx.sim.status.get("libero_success", ctx.sim.status.get("success"))),
              "calls": {"total": len(kinds), "by_kind": {k: kinds.count(k) for k in sorted(set(kinds))},
                        "offline_mock": True, "real_model_calls": 0},
              "events": events, "outcomes": outcomes, "control_s": round(time.monotonic() - started, 2),
              "oracle_report": answers.report()}
    if hasattr(policy, "agent_stats"):
        write_json(ctx.out / "agent_stats.json", policy.agent_stats())
    return report, fields


# ---------------------------------------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=(*V2_TASK_KEYS, *V2_TASK_ALIASES), required=True)
    parser.add_argument("--episode-index", type=int, choices=range(50), default=0)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--first-sense", choices=tuple(FIRST_SENSES), default=None,
                        help="CONTRACT twist sense about the approach R2 (+1 is the blocked sense on task 2); the "
                             "report also prints the world-z (SP/CP) label")
    parser.add_argument("--depth-rule", choices=DEPTH_RULES, default="auto")
    parser.add_argument("--forced-yaw", choices=FORCED_YAWS, default="none")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 5000:
        parser.error("--max-steps must be in [1, 5000]")
    args.first_sense_value = FIRST_SENSES.get(args.first_sense) if args.first_sense else None
    return args


def run(args, *, sim_factory=None, client_factory=None, policy_factory=None):
    scenario = load_scenario(args.scenario)
    out = Path(args.out) if args.out is not None else default_out(args.scenario)
    if out.exists():
        raise FileExistsError(f"output {out} exists; choose a fresh directory")
    out.mkdir(parents=True)
    started, began = utc_stamp(), time.monotonic()
    manifest = {"harness_version": HARNESS_VERSION, "scenario": args.scenario, "task": args.task,
                "episode_index": args.episode_index, "created_utc": started, "argv": sys.argv,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "zero_model_calls": True, "scored": False, "diagnosis_only": True,
                "environment": {k: os.environ.get(k) for k in ("MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES",
                                                               "MUJOCO_GL")},
                "sources": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
                    Path(__file__).resolve(), ORACLE_FILE, SCENARIO_FILE,
                    ROOT / "src/gpt_field/agent_speedup_v2_sim.py")},
                "versions": {"sim": SIM_VERSION, "oracle": oracle_module().ORACLE_VERSION}}
    write_json(out / "manifest.json", manifest)

    def restart(oracle=True):
        return start_sim(args.task, oracle=oracle, sim_python=args.sim_python, libero_root=args.libero_root,
                         sim_factory=sim_factory)

    sim, spec, _config, monitor = restart(oracle=True)
    report, fields, error = {}, {}, None
    try:
        obs, episode = reset_episode(sim, spec, args.episode_index)
        ctx = HarnessContext(sim=sim, spec=spec, args=args, out=out, obs=obs, episode=episode, restart=restart)
        if getattr(scenario, "uses_policy", True):
            report, fields = run_policy_scenario(ctx, scenario, client_factory=client_factory,
                                                 policy_factory=policy_factory)
        else:
            report = scenario.drive(ctx)
            fields = {"steps": ctx.steps, "calls": {"total": 0, "real_model_calls": 0, "offline_mock": True}}
        harness = {"harness_version": HARNESS_VERSION, "scenario": args.scenario, "task_key": spec.key,
                   "episode": episode, "sim_metadata": {k: getattr(sim, "metadata", {}).get(k) for k in (
                       "oracle_enabled", "monitor_options", "v2_sim_version", "evaluator_ops")},
                   "monitor_options": monitor, "wall_s": round(time.monotonic() - began, 1), **fields}
        if args.first_sense_value is not None:
            harness["first_sense"] = {"contract": args.first_sense_value,
                                      "world_z_label": world_z_label(args.first_sense_value, episode["initial_state8"])}
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        harness = {"harness_version": HARNESS_VERSION, "scenario": args.scenario, "error": error}
        raise
    finally:
        try:
            sim.close()
        finally:
            write_json(out / "report.json", {**report, "harness": harness} if not error
                       else {"passed": False, "harness": harness})
    return {**report, "harness": {**harness, "out": str(out)}}


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"scenario": args.scenario, "passed": result.get("passed"),
                      "failures": result.get("failures"), "steps": result["harness"].get("steps"),
                      "out": result["harness"]["out"]}, indent=1))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
