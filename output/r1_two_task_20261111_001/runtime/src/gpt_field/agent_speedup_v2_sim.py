"""agent_SpeedUp_v2 simulator worker and process client (evaluator side; spec §6.2-§6.5).

Promoted from the Phase A prototype ``survey/infra/v2_worker_prototype.py``. ``libero.py`` is not edited:

- ``V2LiberoWorker(_LiberoWorker)`` runs the base worker with ``collision_check_mode='none'`` (the only mode
  ``LiberoConfig`` accepts for libero_10) and attaches the name-free ``GenericContactMonitor`` itself. For the v1
  regression tasks it may instead run with ``task_manipulation_v1`` (record-only), in which case the v1 monitor
  stays attached and the generic monitor is chained on top, for comparability (spec §6.4). Either way the
  status carries ``collision`` = the generic record (mode ``generic_contact_v2``), ``libero_success`` (raw) and
  ``collision_free_success``; the chained v1 record is ``collision_v1_record_only``. ``libero_success`` is the
  worker's own ``env.check_success()`` evaluation, taken right after the base step (before the generic
  monitor's endpoint refresh, so at the same kinematics as the base worker's evaluation); the runner compares
  it with the base worker's ``success``. In chained mode it must also equal the base status's
  ``libero_success`` (checked here).
- Physics is not perturbed by the generic monitor: its endpoint and reset refreshes keep ``qacc_warmstart``
  (``agent_speedup_v2_contacts.forward_keeping_warmstart``); the reset refresh is skipped when the chained v1
  monitor already ran one. The evaluator ``Scene`` is built at every reset (it records the reset orientation of
  every object, which the truth's ``orientation_kept`` check compares against).
- Extra evaluator-only JSONL ops: ``truth`` (strict physical truth at the current state, windowed over the last
  control steps), ``settle_truth`` (release-and-settle counterfactual, then truth; mutates the terminal state
  and ends the episode), ``answer_key`` (BDDL goal atoms) and ``geometry`` (monitor role map). None of them
  travels in an observation, and none changes status, success or termination.
- SIM-EXT (BH, contract §C14.2): ``oracle`` (points / project / joints / bodies / heights / segment), read-only
  evaluator geometry for the test-only ORACLE stub and the smoke reports. It is refused (PermissionError) unless
  the worker is started with the optional monitor option ``oracle=True``; the runner never passes it in live mode.
  Kinematic timing: the oracle reads the geometry AFTER the generic monitor's endpoint refresh (``sim.forward``),
  whereas the step's observation (images and state8) is produced inside ``env.step`` before it, from the
  kinematics of the last physics substep's start. So the oracle is one substep AHEAD of what the policy saw: at a
  settled pose they agree (<= 0.1 px vs the R-2 model), while the hand moves the oracle's wrist projection and
  render differ from the observation's by up to ~4 px (full-speed yaw) / ~3 px (full-speed translation). Compare
  observation-derived quantities with oracle truth on settled frames only, or budget that lag.
- ``V2SimProcess(SimProcess)`` starts ``python -m gpt_field.agent_speedup_v2_sim --worker`` in its own session
  (a terminal Ctrl+C reaches only the runner) and drains an in-flight reply on close; this is the semantics of
  v1's ``_SessionSimProcess`` (``scripts/evaluate_agent_speedup_v1.py``), re-stated here because a package
  module cannot import a script. The JSONL protocol has no request ids, so ``_in_flight`` stays True when a
  request is interrupted (KeyboardInterrupt / SIGTERM) before its reply was read: the next request would read
  the stale reply. The runner skips its finalization truth ops in that case.

One ``_LiberoWorker`` per process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .agent_speedup_v2_contacts import CONTACTS_VERSION, MODE, GenericContactMonitor, forward_keeping_warmstart
from .libero import LiberoConfig, SimProcess, _LiberoWorker

SIM_VERSION = "agent-speedup-v2-sim-v2"
WORKER_MODULE = "gpt_field.agent_speedup_v2_sim"
SIM_CLOSE_DRAIN_S = 3.0
ALLOWED_BASE_MODES = ("none", "task_manipulation_v1")
DEFAULT_MONITOR_OPTIONS = {"goal_aware": True, "truth_window": True}
EVALUATOR_OPS = ("truth", "settle_truth", "answer_key", "geometry")
# SIM-EXT (BH, contract §C14.2): the evaluator-only ORACLE op, refused unless the worker was started with the
# optional monitor option ``oracle=True`` (smokes only). Optional options are omitted from the validated dict when
# absent, so validate_monitor_options(None) and DEFAULT_MONITOR_OPTIONS stay exactly M0's.
EVALUATOR_OPS += ("oracle",)
OPTIONAL_MONITOR_OPTIONS = ("oracle",)


def validate_monitor_options(options):
    options = {**DEFAULT_MONITOR_OPTIONS, **(options or {})}
    unknown = set(options) - set(DEFAULT_MONITOR_OPTIONS) - set(OPTIONAL_MONITOR_OPTIONS)
    if unknown:
        raise ValueError(f"unknown monitor options: {sorted(unknown)}")
    for key, value in options.items():
        if type(value) is not bool:
            raise ValueError(f"monitor option {key} must be a boolean")
    return options


def validate_v2_sim_config(config: LiberoConfig):
    if config.collision_check_mode not in ALLOWED_BASE_MODES:
        raise ValueError("the v2 worker owns contact monitoring; collision_check_mode must be 'none' "
                         "(or 'task_manipulation_v1' to chain the v1 monitor on its own tasks)")
    if config.stop_on_collision:
        raise ValueError("v2 collisions are record-only; stop_on_collision must be False")
    return config


class V2LiberoWorker(_LiberoWorker):
    """Base worker + generic record-only monitor + evaluator-only truth ops."""

    def __init__(self, config, monitor_options=None):
        validate_v2_sim_config(config)
        self.monitor_options = validate_monitor_options(monitor_options)
        super().__init__(config)
        chained = config.collision_check_mode == "task_manipulation_v1"
        self.metadata.update(
            v2_sim_version=SIM_VERSION, base_collision_check_mode=config.collision_check_mode,
            collision_check_mode=MODE, stop_on_collision=False, collision_policy="record_only",
            success_qualification="raw_libero_goal",
            collision_monitoring={"mode": MODE, "version": CONTACTS_VERSION, "rule_table": None,
                                  "goal_aware": self.monitor_options["goal_aware"],
                                  "sampling": "every actual physics substep and final control endpoint",
                                  "chained_v1_monitor": ("task_manipulation_v1 (record-only)" if chained else None)},
            evaluator_ops=list(EVALUATOR_OPS), monitor_options=dict(self.monitor_options))
        self.oracle_enabled = self.monitor_options.get("oracle", False) is True     # SIM-EXT (BH)
        self.metadata["oracle_enabled"] = self.oracle_enabled
        self.generic = None
        self.scene = None
        self.window = None
        self._window_targets = ([], [])
        self.counterfactual_applied = False

    # episode -------------------------------------------------------------------------------------------
    def reset(self, task_id, episode_index):
        response = super().reset(task_id, episode_index)
        self.counterfactual_applied = False
        raw = self._raw_success()
        self.generic = GenericContactMonitor(self.env, goal_aware=self.monitor_options["goal_aware"])
        if not self._chained():          # the chained v1 monitor already refreshed the contacts at reset
            forward_keeping_warmstart(self.env.sim)
        self.generic.inspect(phase="initial_state")
        self.generic.attach()
        from .agent_speedup_v2_truth import Scene, Window, window_targets
        self.scene = Scene(self.env)     # records the reset orientations (truth: orientation_kept)
        self.window = None
        if self.monitor_options["truth_window"]:
            self.window = Window()
            self._window_targets = window_targets(self.scene)
            self.window.update(self.scene, *self._window_targets)
        self._update_status(raw)
        response["status"] = self.status.copy()
        return response

    def _chained(self):
        return getattr(self, "_collision_monitor", None) is not None

    def _raw_success(self):
        """The worker's own raw LIBERO evaluation (a read-only predicate), independent of the base status."""
        raw = bool(self.env.check_success())
        if self._chained() and raw != self.status.get("libero_success"):
            raise RuntimeError("v2 raw LIBERO success differs from the chained base worker's libero_success")
        return raw

    def _update_status(self, raw):
        snapshot = self.generic.snapshot()
        v1 = self.status.pop("collision", None)
        self.status.update(libero_success=bool(raw), collision=snapshot,
                           collision_free_success=bool(raw) and not snapshot["detected"])
        if self._chained():
            self.status["collision_v1_record_only"] = v1

    def step(self, action):
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("reset is required before stepping")
        if self.counterfactual_applied:
            raise RuntimeError("the release-and-settle counterfactual ended this episode")
        self.generic.begin_action(self.status["steps"])
        response = super().step(action)
        raw = self._raw_success()        # same kinematics as the base worker's evaluation
        # A chained v1 monitor has already refreshed (sim.forward) and sampled the endpoint.
        self.generic.finish_action(refresh=not self._chained())
        if self.window is not None:
            self.window.update(self.scene, *self._window_targets)
        self._update_status(raw)
        response["status"] = self.status.copy()
        return response

    # evaluator-only ops ----------------------------------------------------------------------------------
    def _require_scene(self):
        if self.env is None:
            raise RuntimeError("reset is required before truth")
        if self.scene is None:           # not built at reset: no reset orientations (orientation_kept fails)
            from .agent_speedup_v2_truth import Scene
            self.scene = Scene(self.env, at_reset=False)
        return self.scene

    def truth(self):
        from .agent_speedup_v2_truth import evaluate_goal, jsonable
        scene = self._require_scene()
        result = evaluate_goal(scene, self.window)
        result.update(state="terminal" if not self.counterfactual_applied else "after_counterfactual",
                      steps=self.status.get("steps"), libero_success_status=self.status.get("libero_success"),
                      contact_record_detected=(self.status.get("collision") or {}).get("detected"))
        return jsonable(result)

    def settle_truth(self, protocol=None):
        from .agent_speedup_v2_truth import evaluate_goal, jsonable, release_and_settle
        if self.counterfactual_applied:
            raise RuntimeError("the release-and-settle counterfactual runs once per episode")
        scene = self._require_scene()
        monitors = [m for m in (getattr(self, "_collision_monitor", None), self.generic) if m is not None]
        self.counterfactual_applied = True
        from .agent_speedup_v2_truth import Window
        window = self.window if self.window is not None else Window()
        settle = release_and_settle(self.env, monitors, scene=scene, window=window, protocol=protocol)
        result = evaluate_goal(scene, window)
        result.update(state="after_release_settle", settle=settle,
                      contact_record_after_settle=self.generic.snapshot()["counts"])
        return jsonable(result)

    def answer_key(self):
        from .agent_speedup_v2_truth import answer_key
        return answer_key(self._require_scene())

    def geometry(self):
        if self.generic is None:
            raise RuntimeError("reset is required before geometry")
        return {"roles": self.generic.role_summary(), "geom_count": len(self.generic.geometry)}

    def oracle(self, request):
        """SIM-EXT (BH): evaluator geometry for the test-only ORACLE stub and the smoke reports (§C14.2).

        Refused (PermissionError) unless the worker was started with ``monitor_options["oracle"] is True``. It
        only reads the simulator (and renders for ``segment``): no forward, no step, no state change.

        Every reply is at the POST-REFRESH kinematics of the last step (after ``generic.finish_action``'s
        forward), one physics substep ahead of the step's observation, which ``super().step`` rendered before the
        refresh. They agree at a settled pose; while the hand moves the wrist projection / render lag is up to
        ~4 px at full speed (module docstring). Callers compare against observations on settled frames only."""
        if not getattr(self, "oracle_enabled", False):
            raise PermissionError("oracle op disabled")
        return oracle_reply(self._require_scene(), request, render_px=self.config.render_size,
                            image_px=self.config.image_size)


def _json_line(value):
    from .agent_speedup_v2_truth import jsonable
    return json.dumps(jsonable(value), allow_nan=False)


def _worker_main(config, monitor_options=None):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if worker is None:
                    worker = V2LiberoWorker(config, monitor_options)
                op = request["op"]
                if op == "metadata":
                    result = worker.metadata
                elif op == "tasks":
                    result = {"tasks": worker.tasks()}
                elif op == "reset":
                    result = worker.reset(request["task_id"], request["episode_index"])
                elif op == "step":
                    result = worker.step(request["action"])
                elif op == "truth":
                    result = worker.truth()
                elif op == "settle_truth":
                    result = worker.settle_truth(request.get("protocol"))
                elif op == "answer_key":
                    result = worker.answer_key()
                elif op == "geometry":
                    result = worker.geometry()
                elif op == "oracle":                       # SIM-EXT (BH); refused unless enabled
                    result = worker.oracle(request.get("request"))
                else:
                    raise ValueError(f"unknown operation: {op}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(_json_line(response) + "\n")
            protocol.flush()
    finally:
        if worker is not None:
            worker.close()
        protocol.close()
    return 0


def worker_command(config: LiberoConfig, monitor_options=None):
    return [config.python_executable or sys.executable, "-u", "-m", WORKER_MODULE, "--worker",
            "--config-json", json.dumps(asdict(config)),
            "--monitor-json", json.dumps(validate_monitor_options(monitor_options))]


class V2SimProcess(SimProcess):
    """JSONL client for the v2 worker: own session, drain on close, evaluator-only truth ops."""

    def __init__(self, config: LiberoConfig, *, monitor_options=None, command=None):
        validate_v2_sim_config(config)
        self.monitor_options = validate_monitor_options(monitor_options)
        self._in_flight = False
        super().__init__(config, command=command if command is not None else worker_command(config,
                                                                                               self.monitor_options))

    def _request(self, request):
        """SimProcess._request with an in-flight flag: it stays True when an interrupt propagates out of the
        request before its reply was read (the reply is then still in the pipe). An ordinary exception means the
        reply line was consumed (an error reply or invalid JSON) or the worker is gone, so nothing is queued."""
        self._in_flight = True
        try:
            result = super()._request(request)
        except Exception:
            self._in_flight = False
            raise
        self._in_flight = False
        return result

    def start(self):
        if self._process is not None:
            return self
        env = os.environ.copy()
        package_src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_src, env.get("PYTHONPATH")]))
        self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                                         bufsize=0, start_new_session=True)
        try:
            self.metadata = self._request({"op": "metadata"})
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        proc = self._process
        if proc is not None and proc.stdin is not None and proc.stdout is not None:
            try:
                proc.stdin.close()
                deadline = time.monotonic() + SIM_CLOSE_DRAIN_S
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while (remaining := deadline - time.monotonic()) > 0 and selector.select(remaining):
                        if not os.read(proc.stdout.fileno(), 1 << 20):
                            break
            except (OSError, ValueError):
                pass
        super().close()

    # evaluator-only (diagnosis; never shown to the policy)
    def truth(self):
        return self._request({"op": "truth"})

    def settle_truth(self, protocol=None):
        return self._request({"op": "settle_truth", **({"protocol": protocol} if protocol else {})})

    def answer_key(self):
        return self._request({"op": "answer_key"})

    def geometry(self):
        return self._request({"op": "geometry"})

    def oracle(self, request):
        """SIM-EXT (BH): the evaluator-only ORACLE op (§C14.2). Refused here too when this process was not
        started with ``monitor_options["oracle"] is True``; the worker refuses independently."""
        if self.monitor_options.get("oracle") is not True:
            raise PermissionError("oracle op disabled")
        return self._request({"op": "oracle", "request": request})


# ---------------------------------------------------------------------------------------------------------
# SIM-EXT (BH, contract §C14.2): ORACLE op implementation. Evaluator-only and test-only: the ORACLE pointing stub
# (scripts/agent_speedup_v2_oracle.py) and the smoke reports read it; no control module may import this module.
# Every request is read-only on the simulator state (``segment`` renders, which changes no physics). All replies are
# at the post-refresh kinematics, one physics substep ahead of the step's observation (see V2LiberoWorker.oracle).
ORACLE_KINDS = ("points", "project", "joints", "bodies", "heights", "segment")
ORACLE_CAMERAS = {"wrist": "robot0_eye_in_hand", "external": "agentview"}
ORACLE_WHERE = ("aabb_top_centre", "aabb_centre", "com", "origin")
ORACLE_RAY_TOP_Z_M = 2.5            # ``heights`` rays start here (above every LIBERO scene geom) and go straight down
ORACLE_DEPTH_PNG_UNIT_M = 1e-4      # ``segment`` depth_png: uint16 in 0.1 mm (saturates at 6.5535 m)
ORACLE_MAX_ITEMS = 4096


def _oracle_list(value, what, limit=ORACLE_MAX_ITEMS):
    if not isinstance(value, (list, tuple)) or len(value) > limit:
        raise ValueError(f"oracle {what} must be a list of at most {limit} items")
    return list(value)


def _oracle_vector(value, n, what):
    import numpy as np
    array = np.asarray(value, dtype=float)
    if array.shape != (n,) or not np.all(np.isfinite(array)):
        raise ValueError(f"oracle {what} must be {n} finite numbers")
    return array


def _oracle_names(value, what):
    names = [value] if isinstance(value, str) else (list(value) if isinstance(value, (list, tuple)) else None)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"oracle {what} must be a non-empty name or list of names")
    return names


def _oracle_id(scene, objtype, name, what):
    index = int(scene.mj.mj_name2id(scene.M, objtype, name))
    if index < 0:
        raise ValueError(f"oracle: unknown {what} {name!r}")
    return index


def oracle_camera_name(camera):
    """'wrist' / 'external' (the policy's names) or the MuJoCo camera name itself."""
    if camera in ORACLE_CAMERAS:
        return ORACLE_CAMERAS[camera]
    if camera in ORACLE_CAMERAS.values():
        return camera
    raise ValueError(f"oracle camera must be one of {sorted(ORACLE_CAMERAS)}")


def project_world_points(xyz, cam_pos, cam_xmat, fovy_deg, render_px, image_px):
    """Upright pixels (u right, v down, pixel centres at integers, principal point (W-1)/2) of world points
    through a MuJoCo camera (columns of ``cam_xmat`` = camera x right, y up, z backward; ``fovy_deg`` vertical),
    rendered at ``render_px`` and resized (square, no padding) to ``image_px``. Returns (uv, depth_m, in_image):
    uv None when the point is not in front of the camera; depth_m = distance along the optical axis."""
    import numpy as np
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    R = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    local = (points - np.asarray(cam_pos, dtype=float)) @ R
    depth = -local[:, 2]
    f = (render_px / 2.0) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    c = (render_px - 1) / 2.0
    scale = image_px / float(render_px)
    uv, depths, inside = [], [], []
    for (x, y, _), z in zip(local, depth):
        depths.append(float(z))
        if not z > 1e-9:
            uv.append(None)
            inside.append(False)
            continue
        u = (c + f * x / z + 0.5) * scale - 0.5
        v = (c - f * y / z + 0.5) * scale - 0.5
        uv.append([float(u), float(v)])
        inside.append(bool(-0.5 <= u < image_px - 0.5 and -0.5 <= v < image_px - 0.5))
    return uv, depths, inside


def _oracle_point(scene, item, *, corners=False):
    import numpy as np
    mj, M, D = scene.mj, scene.M, scene.D
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        raise ValueError("oracle points items need a 'name'")
    where = item.get("where", "aabb_top_centre")
    if where not in ORACLE_WHERE:
        raise ValueError(f"oracle 'where' must be one of {ORACLE_WHERE}")
    offset = _oracle_vector(item.get("offset", (0.0, 0.0, 0.0)), 3, "offset")
    selectors_given = [key for key in ("body", "geom", "site") if key in item]
    if len(selectors_given) != 1:
        raise ValueError("oracle points items need exactly one of 'body', 'geom', 'site'")
    geoms = None
    if "geom" in item:          # one geom, or a list (their union aabb); visual geoms allowed
        geoms = [_oracle_id(scene, mj.mjtObj.mjOBJ_GEOM, name, "geom") for name in _oracle_names(item["geom"], "geom")]
        boxes = [scene.aabb(g) for g in geoms]
        origin = com = np.mean([np.asarray(D.geom_xpos[g], float) for g in geoms], axis=0)
    elif "body" in item:        # the body's subtree collision geoms (all its geoms when it has no collision geom)
        bid = _oracle_id(scene, mj.mjtObj.mjOBJ_BODY, item["body"], "body")
        bodies = scene.subtree(bid)
        geoms = scene.geoms_of(bodies) or sorted(g for g in range(M.ngeom) if int(scene.geom_body[g]) in bodies)
        if not geoms:
            raise ValueError(f"oracle: body {item['body']!r} has no geoms")
        boxes = [scene.aabb(g) for g in geoms]
        origin = np.asarray(D.xpos[bid], float)
        com = np.asarray(D.subtree_com[bid], float)
    else:
        sid = _oracle_id(scene, mj.mjtObj.mjOBJ_SITE, item["site"], "site")
        origin = com = np.asarray(D.site_xpos[sid], float)
        half = np.abs(np.asarray(D.site_xmat[sid], float).reshape(3, 3)) @ np.asarray(M.site_size[sid], float)
        boxes = [(origin - half, origin + half)]
    lo = np.min([np.asarray(b[0], float) for b in boxes], axis=0)
    hi = np.max([np.asarray(b[1], float) for b in boxes], axis=0)
    centre = (lo + hi) / 2.0
    point = {"aabb_top_centre": np.array([centre[0], centre[1], hi[2]]), "aabb_centre": centre,
             "com": com, "origin": origin}[where] + offset
    return item["name"], point, lo, hi, (_oracle_corners(scene, geoms, boxes) if corners else None)


def _oracle_corners(scene, geoms, boxes):
    """Per-geom corner sets for the ``points`` op's optional ``corners``: the oriented box corners of a box geom
    (``Scene.geom_pts``), the 8 aabb corners of any other geom (a mesh would send every vertex), and the 8 corners
    of a site's aabb (``geoms`` None). Used by the ORACLE stub to draw a part's image region (BH-3)."""
    import numpy as np

    def aabb_corners(lo, hi):
        return [[float((lo, hi)[i][0]), float((lo, hi)[j][1]), float((lo, hi)[k][2])]
                for i in (0, 1) for j in (0, 1) for k in (0, 1)]

    if geoms is None:
        return [aabb_corners(*boxes[0])]
    box = int(scene.mj.mjtGeom.mjGEOM_BOX)
    return [np.asarray(scene.geom_pts(g), float).tolist() if int(scene.M.geom_type[g]) == box
            else aabb_corners(*scene.aabb(g)) for g in geoms]


def oracle_height(scene, x, y, *, skip_bodies=frozenset(), top_z=ORACLE_RAY_TOP_Z_M):
    """Highest COLLISION surface at (x, y): a downward ray from ``top_z`` against every collision geom (contype or
    conaffinity nonzero) whose body is not in ``skip_bodies``, by exact per-geom intersection (``mju_rayGeom`` /
    ``mj_rayMesh`` / ``mj_rayHfield``). ``mj_ray`` is not used: it ignores alpha-0 geoms, and LIBERO's collision
    geoms are often invisible (the flat stove's burner plate, top 0.9305, would be missed and its 0.925 base hit).
    Returns (z, geom id) or (None, None)."""
    import numpy as np
    mj, M, D = scene.mj, scene.M, scene.D
    pnt = np.array([float(x), float(y), float(top_z)])
    vec = np.array([0.0, 0.0, -1.0])
    best, best_geom = None, None
    for g in range(M.ngeom):
        if not scene.coll[g] or int(scene.geom_body[g]) in skip_bodies:
            continue
        rbound = float(M.geom_rbound[g])
        if rbound > 0 and math.hypot(pnt[0] - D.geom_xpos[g][0], pnt[1] - D.geom_xpos[g][1]) > rbound:
            continue
        kind = int(M.geom_type[g])
        if kind == int(mj.mjtGeom.mjGEOM_MESH):
            dist = mj.mj_rayMesh(M, D, g, pnt, vec)
        elif kind == int(mj.mjtGeom.mjGEOM_HFIELD):
            dist = mj.mj_rayHfield(M, D, g, pnt, vec)
        else:
            dist = mj.mju_rayGeom(np.asarray(D.geom_xpos[g], float), np.asarray(D.geom_xmat[g], float),
                                  np.asarray(M.geom_size[g], float), pnt, vec, kind)
        if dist >= 0 and (best is None or dist < best):
            best, best_geom = float(dist), g
    return (None, None) if best is None else (float(top_z) - best, best_geom)


def _oracle_camera(scene, camera):
    mj = scene.mj
    name = oracle_camera_name(camera)
    cid = _oracle_id(scene, mj.mjtObj.mjOBJ_CAMERA, name, "camera")
    return name, cid


def _oracle_png(mask_or_rgb):
    import base64
    import io
    import numpy as np
    from PIL import Image
    array = np.asarray(mask_or_rgb)
    image = Image.fromarray(np.ascontiguousarray(array.astype(np.uint8) * (255 if array.dtype == bool else 1)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _oracle_png16(values):
    import base64
    import io
    import numpy as np
    from PIL import Image
    image = Image.fromarray(np.ascontiguousarray(np.asarray(values).astype(np.uint16)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _oracle_segment(scene, request, render_px, image_px):
    """Upright segmentation, depth (and optionally RGB) of one camera at the current state; per-pixel lookups at
    ``uv`` and binary masks for ``masks`` items ({"name", one of "body" (subtree) / "owner" (BDDL object) /
    "geom" / "robot": true})."""
    import numpy as np
    if render_px != image_px:
        raise ValueError("oracle segment needs render_size == image_size")
    mj, M = scene.mj, scene.M
    name, _ = _oracle_camera(scene, request.get("camera"))
    seg, depth = scene.sim.render(width=render_px, height=render_px, camera_name=name, depth=True,
                                  segmentation=True)
    seg, depth = np.asarray(seg)[::-1], np.asarray(depth, float)[::-1]      # OpenGL rows -> upright
    extent = float(M.stat.extent)
    near, far = float(M.vis.map.znear) * extent, float(M.vis.map.zfar) * extent
    depth_m = near / (1.0 - depth * (1.0 - near / far))
    is_geom = seg[..., 0] == int(mj.mjtObj.mjOBJ_GEOM)
    geom_id = np.where(is_geom, seg[..., 1], -1)
    body_id = np.where(geom_id >= 0, np.asarray(scene.geom_body)[np.clip(geom_id, 0, None)], -1)
    present = [int(b) for b in np.unique(body_id) if b >= 0]
    owners = {b: scene.owner(b) for b in present}
    at = []
    for uv in _oracle_list(request.get("uv", []), "uv"):
        u, v = _oracle_vector(uv, 2, "uv")
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < image_px and 0 <= iv < image_px):
            at.append(None)
            continue
        g, b = int(geom_id[iv, iu]), int(body_id[iv, iu])
        at.append({"px": [iu, iv], "geom": scene.gname(g) if g >= 0 else None,
                   "body": scene.bname(b) if b >= 0 else None, "owner": owners.get(b),
                   "robot": b in scene.robot_bodies, "depth_m": float(depth_m[iv, iu])})
    masks = {}
    for item in _oracle_list(request.get("masks", []), "masks", limit=64):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("oracle segment masks need a 'name'")
        if "body" in item:
            members = scene.subtree(_oracle_id(scene, mj.mjtObj.mjOBJ_BODY, item["body"], "body"))
            mask = np.isin(body_id, sorted(members))
        elif "owner" in item:
            mask = np.isin(body_id, [b for b, o in owners.items() if o == item["owner"]])
        elif "geom" in item:
            mask = geom_id == _oracle_id(scene, mj.mjtObj.mjOBJ_GEOM, item["geom"], "geom")
        elif item.get("robot") is True:
            mask = np.isin(body_id, sorted(scene.robot_bodies))
        else:
            raise ValueError("oracle segment masks need one of 'body', 'owner', 'geom', 'robot'")
        vs, us = np.nonzero(mask)
        masks[item["name"]] = {"pixels": int(mask.sum()),
                               "bbox": [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())] if len(us) else None,
                               "centroid": [float(us.mean()), float(vs.mean())] if len(us) else None,
                               "png": _oracle_png(mask) if request.get("mask_png", True) else None}
    reply = {"camera": name, "size": [image_px, image_px], "at": at, "masks": masks,
             "visible_owners": sorted(set(owners.values()))}
    if request.get("depth_png") is True:
        reply["depth_png"] = _oracle_png16(np.clip(np.round(depth_m / ORACLE_DEPTH_PNG_UNIT_M), 0, 65535))
        reply["depth_png_unit_m"] = ORACLE_DEPTH_PNG_UNIT_M
    if request.get("rgb") is True:
        reply["rgb_png"] = _oracle_png(np.asarray(scene.sim.render(width=render_px, height=render_px,
                                                                   camera_name=name))[::-1])
    return reply


def oracle_reply(scene, request, *, render_px, image_px):
    """Dispatch one ORACLE request (§C14.2 table) against an evaluator ``Scene`` at the current state."""
    import numpy as np
    if not isinstance(request, dict) or request.get("kind") not in ORACLE_KINDS:
        raise ValueError(f"oracle request needs 'kind' in {ORACLE_KINDS}")
    kind = request["kind"]
    mj, M, D = scene.mj, scene.M, scene.D
    if kind == "points":
        want_corners = request.get("corners", False)
        if type(want_corners) is not bool:
            raise ValueError("oracle corners must be a boolean")
        points, boxes, corners = {}, {}, {}
        for item in _oracle_list(request.get("items"), "items"):
            name, point, lo, hi, item_corners = _oracle_point(scene, item, corners=want_corners)
            points[name], boxes[name] = point.tolist(), [lo.tolist(), hi.tolist()]
            corners[name] = item_corners
        reply = {"points": points, "aabb": boxes}
        if want_corners:
            reply["corners"] = corners
        return reply
    if kind == "project":
        name, cid = _oracle_camera(scene, request.get("camera"))
        xyz = [_oracle_vector(p, 3, "xyz").tolist() for p in _oracle_list(request.get("xyz"), "xyz")]
        uv, depth, inside = project_world_points(xyz, D.cam_xpos[cid], D.cam_xmat[cid], float(M.cam_fovy[cid]),
                                                 render_px, image_px) if xyz else ([], [], [])
        # the camera pose lets a caller turn a rendered depth back into a world point (G1 visual surface, BH-1)
        return {"camera": name, "size": [image_px, image_px], "uv": uv, "depth_m": depth, "in_image": inside,
                "cam_xyz": np.asarray(D.cam_xpos[cid], float).tolist(),
                "cam_xmat": np.asarray(D.cam_xmat[cid], float).reshape(-1).tolist()}
    if kind == "joints":
        names = _oracle_names(request.get("names"), "joint names")
        for name in names:                 # joint_info itself does not check the name (id -1 reads the last joint)
            _oracle_id(scene, mj.mjtObj.mjOBJ_JOINT, name, "joint")
        info = {name: scene.joint_info(name) for name in names}
        return {"q": {k: v["q"] for k, v in info.items()}, "range": {k: v["range"] for k, v in info.items()}}
    if kind == "bodies":
        poses = {}
        for name in _oracle_names(request.get("names"), "body names"):
            bid = _oracle_id(scene, mj.mjtObj.mjOBJ_BODY, name, "body")
            w, x, y, z = (float(q) for q in D.xquat[bid])
            poses[name] = {"xyz": np.asarray(D.xpos[bid], float).tolist(), "quat_xyzw": [x, y, z, w]}
        return {"poses": poses}
    if kind == "heights":
        include_robot = request.get("include_robot", False)
        if type(include_robot) is not bool:
            raise ValueError("oracle include_robot must be a boolean")
        skip = frozenset() if include_robot else frozenset(scene.robot_bodies)
        z, hits = [], []
        for xy in _oracle_list(request.get("xy"), "xy"):
            x, y = _oracle_vector(xy, 2, "xy")
            height, g = oracle_height(scene, x, y, skip_bodies=skip)
            z.append(height)
            body = None if g is None else int(scene.geom_body[g])
            hits.append(None if g is None else {"geom": scene.gname(g), "body": scene.bname(body),
                                                "owner": scene.owner(body), "robot": body in scene.robot_bodies})
        return {"z": z, "hit": hits, "include_robot": include_robot}
    return _oracle_segment(scene, request, render_px, image_px)


def initial_observation_sha256(observation_dict):
    """v1's frozen reset recipe (scripts/evaluate_agent_speedup_v1.py:1000)."""
    import hashlib
    return hashlib.sha256(json.dumps(observation_dict, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--monitor-json", default="{}")
    args = parser.parse_args(argv)
    return _worker_main(LiberoConfig(**json.loads(args.config_json)), json.loads(args.monitor_json))


if __name__ == "__main__":
    raise SystemExit(main())
