"""Live plan-only check for agent_SpeedUp_v2 (TEST TOOL; REAL model calls; ORACLE used for scoring only).

Per task (episode 0, frozen reset, collision mode none): reset the real simulator, run Calibrate+ (BM's
``CalibratePlus`` with a probe-tracker ``SurveyRecorder``, as BR's calibrate_plus is restated by the survey probe) to
obtain the camera model, the external model and the frame-alignment turns k exactly as the policy would, then:

1. PLAN: the policy's own path. ``StepRunner.obtain_plan("plan")`` (one re-ask carrying the validation issue text)
   driven by the policy's ``NeedFulfiller`` over ``AgentSpeedUpV2Client.request_plan`` (prompt, schema, the four
   images, reasoning effort as the client sets it), ``parse_plan`` and ``validate_plan``. Since prompt 2 the re-ask
   is the client's targeted correction (``previous_plan`` + issues; an answer that drops an unnamed step is a
   schema error), recorded per attempt as ``previous_plan_sent`` and ``reask_check``. ``--plan-only`` stops after
   the plan (no pointing calls).
2. POINTING: for every plan part with a grip, effect or ref (destination) role, the pointing round of BR's
   ``point_part`` (contract §C10, design review 1 decision §2.6 / rulings item 9): the seed is the plan cell
   (``merge_plan``'s rule: the w cell in the k-turned wrist grid -> unrotated upright pixel; the x cell when w = 0),
   sigma_uv = the seed quantization (32 px); coarse first when sigma_uv > 64 / sqrt(2 ln 2) ~ 54 px, else
   point_fine centred on the plan cell; a fine 0 -> point_coarse on the same image -> point_fine at the coarse cell
   centre; a 0 / 0 path -> part_not_found. Grip parts use purpose grasp_point, region parts region_seed. The
   pointing happens in the START view (no hover motion: BR's hover pointing is not built), so it measures grounding
   at the plan view.
3. SCORING (ORACLE ``segment``, never fed back): the owner / body / geom at every pointed pixel and the owner shares
   within 8 px; the owner shares inside each plan cell (w in the wrist image, x in the external image); visible
   owners per camera; the k the ORACLE's external projection implies (a check of the calibrated k).

A global call ledger (``<out>/ledger.json``) caps the model calls across all task processes (``--max-calls``,
default 80): each task may use what is left minus ``--reserve-per-task`` for every task still to come.

usage (project venv; the worker runs in the sim python; one _LiberoWorker per process, so one task per process):
  env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python scripts/agent_speedup_v2_plan_check.py \\
      --task libero_10_t2 --out runs/agent_speedup_v2_planck_001 --remaining-tasks 11
  .venv/bin/python scripts/agent_speedup_v2_plan_check.py --summarize runs/agent_speedup_v2_planck_001
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

PLAN_CHECK_VERSION = "agent-speedup-v2-plan-check-2"
HARNESS_FILE = ROOT / "scripts" / "agent_speedup_v2_relation_smoke.py"
IMAGE_PX = 512
CELL_PX = 64
SIGMA_UV_COARSE_FIRST_PX = 64.0 / math.sqrt(2.0 * math.log(2.0))    # rulings item 9: ~54.35 px
HIT_RADIUS_PX = 8.0                      # fine quantization (half the 16 px crop-point spacing)
CELL_SAMPLE_STEP_PX = 4                  # owner shares inside a plan cell: a 16 x 16 lattice
POINT_ROLES = ("grip", "effect", "ref")
DEFAULT_MAX_CALLS = 80
DEFAULT_RESERVE_PER_TASK = 2             # one plan call + one possible re-ask for every task still to come
TASK_ORDER = ("libero_10_t2", "libero_10_t0", "libero_10_t1", "libero_10_t3", "libero_10_t4", "libero_10_t5",
              "libero_10_t6", "libero_10_t7", "libero_10_t8", "libero_10_t9", "bowl_plate", "drawer_bowl")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harness():
    return _load("agent_speedup_v2_relation_smoke", HARNESS_FILE)


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path, value):
    Path(path).write_text(json.dumps(harness().jsonable(value), indent=1, allow_nan=False) + "\n")


# ----------------------------------------------------------------------------------------------- ledger
def read_ledger(out: Path) -> dict:
    path = out / "ledger.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"max_calls": None, "used": 0, "tasks": {}}


def write_ledger(out: Path, ledger: dict) -> None:
    write_json(out / "ledger.json", ledger)


# ----------------------------------------------------------------------------------------------- driver
class Driver:
    """Steps the real simulator; primitives see StepViews (the survey probe's Driver, without recorders)."""

    def __init__(self, sim, obs):
        from gpt_field.contact_skills import StepView
        self._StepView = StepView
        self.sim = sim
        self.view = StepView(obs)
        self.steps = 0

    def step(self, action):
        obs, _success = self.sim.step([float(v) for v in action])
        self.view = self._StepView(obs)
        self.steps += 1
        return self.view

    def run(self, primitive):
        while not primitive.done:
            action = primitive.next_action(self.view)
            before = self.view
            after = self.step(action)
            primitive.observe_transition(before, action, after)
        return primitive.outcome()


def calibrate(driver, robot_model):
    """BR's calibrate_plus restated to the parts the plan request needs (survey probe ``calibrate_plus``): BM
    CalibratePlus with a probe-tracker SurveyRecorder (grid at the reference frame) -> CameraModel.decide (R-2f
    granted) -> ExternalModel.fit -> frame_alignment_turns at the final view."""
    from gpt_field import agent_speedup_v2_geometry as geometry
    from gpt_field import agent_speedup_v2_primitives as primitives
    recorder = geometry.SurveyRecorder(driver.view, robot_model, camera=None, external=True, start_step=driver.steps,
                                       probe_tracker=True, grid_at_reference=True)
    primitive = primitives.CalibratePlus(driver.view, recorder=recorder, robot=robot_model)
    began = time.perf_counter()
    outcome = driver.run(primitive)
    loom = recorder.result(None)
    camera = geometry.CameraModel.decide(loom.loom["f_hat"], r2f_granted=True)
    result = recorder.result(camera)
    external = geometry.ExternalModel.fit(result.external_samples)
    turns = int(geometry.frame_alignment_turns(driver.view.R, external))
    f_hat = loom.loom.get("f_hat")
    return {"outcome_reason": outcome.get("reason") if isinstance(outcome, dict) else None,
            "external": external, "external_available": external is not None, "turns": turns,
            "f_used": float(camera.f_used), "f_hat": None if f_hat is None else float(getattr(f_hat, "value", f_hat)),
            "steps": driver.steps, "wall_s": round(time.perf_counter() - began, 2),
            "image_right_world_xy": None if external is None else external.image_right_world_xy()}


class _TruthRight:
    def __init__(self, right):
        self._right = right

    def image_right_world_xy(self):
        return self._right


def truth_turns(sim, view):
    """The k the ORACLE's external projection implies at this view (scoring/diagnosis only)."""
    from gpt_field import agent_speedup_v2_geometry as geometry
    grip = np.asarray(view.position, dtype=float)
    step = 0.01
    points = [grip] + [grip + step * e for e in np.eye(3)]
    uv = np.asarray(sim.oracle({"kind": "project", "camera": "external",
                                "xyz": [p.tolist() for p in points]})["uv"], dtype=float)
    J = (uv[1:] - uv[0]).T / step
    block = J[:, :2]
    if abs(float(np.linalg.det(block))) < 1e-9:
        return None, None
    right = np.linalg.solve(block, np.array([1.0, 0.0]))
    right = right / np.linalg.norm(right)
    return int(geometry.frame_alignment_turns(view.R, _TruthRight(right))), right.tolist()


# ----------------------------------------------------------------------------------------------- oracle scoring
def segment_at(sim, camera, uvs):
    if not uvs:
        return []
    reply = sim.oracle({"kind": "segment", "camera": camera, "masks": [], "mask_png": False,
                        "uv": [[float(u), float(v)] for u, v in uvs]})
    return reply["at"]


def visible_owners(sim, camera):
    reply = sim.oracle({"kind": "segment", "camera": camera, "masks": [], "mask_png": False, "uv": []})
    return reply["visible_owners"]


def _label_of(pixel):
    if pixel is None or pixel.get("geom") is None and pixel.get("body") is None:
        return "none"
    if pixel.get("robot"):
        return "robot"
    return pixel.get("owner") or f"body:{pixel.get('body')}"


def shares(pixels):
    counts = {}
    for pixel in pixels:
        label = _label_of(pixel)
        counts[label] = counts.get(label, 0) + 1
    total = max(1, len(pixels))
    return dict(sorted(((k, round(v / total, 3)) for k, v in counts.items()), key=lambda kv: -kv[1]))


def disc_uvs(uv, radius):
    u0, v0 = uv
    out = []
    r = int(math.ceil(radius))
    for dv in range(-r, r + 1, 2):
        for du in range(-r, r + 1, 2):
            if du * du + dv * dv <= radius * radius:
                u, v = u0 + du, v0 + dv
                if 0 <= u < IMAGE_PX and 0 <= v < IMAGE_PX:
                    out.append((u, v))
    return out


def cell_block_upright(cell, quarter_turns):
    """The upright-pixel block (u0, v0, u1, v1) of grid cell ``cell`` of the image turned by ``quarter_turns``
    (a quarter turn maps the 8 x 8 grid onto itself, so a cell is one 64 x 64 block in the upright image)."""
    from gpt_field.mark_prompting import coarse_cell_center, unrotate_pixel
    u, v = unrotate_pixel(coarse_cell_center(int(cell), IMAGE_PX), int(quarter_turns), IMAGE_PX)
    col, row = int(u // CELL_PX), int(v // CELL_PX)
    return (col * CELL_PX, row * CELL_PX, col * CELL_PX + CELL_PX, row * CELL_PX + CELL_PX)


def block_uvs(block):
    u0, v0, u1, v1 = block
    half = CELL_SAMPLE_STEP_PX / 2.0
    return [(u + half, v + half) for v in range(v0, v1, CELL_SAMPLE_STEP_PX) for u in range(u0, u1, CELL_SAMPLE_STEP_PX)]


def save_masks(sim, task_key, task_dir, owners_by_camera):
    """ORACLE owner masks of every visible owner (and the robot) per camera at the plan view, and, for a task with
    ORACLE_PARTS, each part's region-limited visible mask (scoring only): ``masks_<camera>.npz``."""
    for camera, owners in owners_by_camera.items():
        items = [{"name": f"owner:{o}", "owner": o} for o in owners if o not in ("robot",)]
        items.append({"name": "robot", "robot": True})
        reply = sim.oracle({"kind": "segment", "camera": camera, "masks": items})
        arrays = {}
        for name, info in reply["masks"].items():
            if info["pixels"] and info.get("png"):
                arrays[name] = _decode_mask(info["png"])
        oracle = harness().oracle_module()
        if task_key in oracle.ORACLE_PARTS:
            answers = oracle.OracleAnswers(sim, task_key)
            for pid in oracle.ORACLE_PARTS[task_key]:
                mask = answers.visible_mask(pid, camera)
                if mask is not None:
                    arrays[f"oracle_part:{pid}"] = mask
        np.savez_compressed(task_dir / f"masks_{camera}.npz", **arrays)


def _decode_mask(url):
    return harness().decode_png(url) > 0


# ----------------------------------------------------------------------------------------------- pointing
def point_round(fulfiller, view, calls, *, pid, text, camera, center_uv, sigma_uv, k, purpose):
    """BR point_part restated (module docstring step 2). Returns {"path", "final_uv", "reason", "attempts"}."""
    from gpt_field.agent_speedup_v2_types import PointNeed
    turns = k if camera == "wrist" else 0
    attempts = []

    def ask(kind, centre):
        need = PointNeed(kind, pid, text, camera, None if centre is None else tuple(map(float, centre)), turns,
                         purpose)
        answer = fulfiller.fulfil(need, view, calls)
        reply = answer.reply
        attempts.append({"kind": kind, "center_uv": None if centre is None else list(map(float, centre)),
                         "answer_id": None if reply is None else reply.answer_id,
                         "uv": None if reply is None or reply.uv_upright is None else list(reply.uv_upright),
                         "origin": None if reply is None or reply.origin is None else list(reply.origin),
                         "schema_error": bool(answer.schema_error), "latency_s": answer.call.latency_s,
                         "request_log": answer.call.request_log})
        return None if reply is None else reply.uv_upright

    coarse_first = sigma_uv > SIGMA_UV_COARSE_FIRST_PX
    if coarse_first:
        cell_uv = ask("point_coarse", None)
        if cell_uv is None:
            return {"path": "coarse_first", "final_uv": None, "reason": "part_not_found", "attempts": attempts}
        uv = ask("point_fine", cell_uv)
        return {"path": "coarse_first", "final_uv": uv, "reason": None if uv else "part_not_found",
                "fallback_uv": None if uv else list(cell_uv), "attempts": attempts}
    uv = ask("point_fine", center_uv)
    if uv is not None:
        return {"path": "fine", "final_uv": list(uv), "reason": None, "attempts": attempts}
    cell_uv = ask("point_coarse", None)
    if cell_uv is None:
        return {"path": "fine0_coarse0", "final_uv": None, "reason": "part_not_found", "attempts": attempts}
    uv = ask("point_fine", cell_uv)
    return {"path": "fine0_coarse_fine", "final_uv": None if uv is None else list(uv),
            "reason": None if uv is not None else "part_not_found",
            "fallback_uv": None if uv is not None else list(cell_uv), "attempts": attempts}


def point_and_score(sim, fulfiller, view, calls, *, part, camera, seed_uv, k, purpose, roles):
    """One part's pointing round (``point_round``) plus the ORACLE scoring of every answered pixel."""
    seed_uv = [float(seed_uv[0]), float(seed_uv[1])]
    sigma_uv = float(CELL_PX / 2.0)          # merge_plan's plan-cell seed quantization
    result = point_round(fulfiller, view, calls, pid=part.pid, text=part.text, camera=camera, center_uv=seed_uv,
                         sigma_uv=sigma_uv, k=k, purpose=purpose)
    for attempt in result["attempts"]:
        if attempt["uv"] is not None:
            pixel = segment_at(sim, camera, [attempt["uv"]])[0]
            attempt["hit_pixel"] = pixel
            attempt["hit_label"] = _label_of(pixel)
    final = result["final_uv"] or result.get("fallback_uv")
    if final is not None:
        pixel = segment_at(sim, camera, [final])[0]
        result["final_pixel"] = pixel
        result["final_label"] = _label_of(pixel)
        result["final_disc_shares"] = shares(segment_at(sim, camera, disc_uvs(final, HIT_RADIUS_PX)))
    result.update(camera=camera, seed_uv=seed_uv, sigma_uv_px=sigma_uv, purpose=purpose, roles=sorted(roles),
                  coarse_first=sigma_uv > SIGMA_UV_COARSE_FIRST_PX, seed_pixel=segment_at(sim, camera, [seed_uv])[0])
    return result


def part_camera_seed(part, k):
    """The pointing camera and seed of a plan part: the w cell (wrist, unrotated by k) first, else the x cell."""
    from gpt_field.mark_prompting import coarse_cell_center, unrotate_pixel
    if part.w_cell:
        return "wrist", unrotate_pixel(coarse_cell_center(part.w_cell, IMAGE_PX), k, IMAGE_PX)
    if part.x_cell:
        return "external", coarse_cell_center(part.x_cell, IMAGE_PX)
    return None, None


# ----------------------------------------------------------------------------------------------- overlays
def draw_overlay(rgb, path, items, title):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for i in range(1, 8):
        draw.line([(i * CELL_PX, 0), (i * CELL_PX, IMAGE_PX)], fill=(255, 255, 255, 50))
        draw.line([(0, i * CELL_PX), (IMAGE_PX, i * CELL_PX)], fill=(255, 255, 255, 50))
    palette = [(255, 60, 60), (60, 200, 255), (255, 220, 40), (120, 255, 120), (255, 120, 255), (255, 160, 60),
               (180, 180, 255), (255, 255, 255)]
    for index, item in enumerate(items):
        colour = palette[index % len(palette)]
        if item.get("cell_block"):
            draw.rectangle(item["cell_block"], outline=colour + (255,), width=2)
            draw.text((item["cell_block"][0] + 3, item["cell_block"][1] + 2), f"{item['pid']} cell", fill=colour)
        if item.get("crop"):
            draw.rectangle(item["crop"], outline=colour + (140,), width=1)
        for uv, mark in item.get("marks", []):
            u, v = uv
            draw.line([(u - 7, v), (u + 7, v)], fill=colour + (255,), width=2)
            draw.line([(u, v - 7), (u, v + 7)], fill=colour + (255,), width=2)
            draw.text((u + 6, v + 4), f"{item['pid']}{mark}", fill=colour)
    draw.text((4, IMAGE_PX - 14), title[:90], fill=(255, 255, 255))
    image.save(path)


# ----------------------------------------------------------------------------------------------- one task
def call_usage(log_dir: Path) -> list[dict]:
    rows = []
    for request in sorted(log_dir.glob("*.request.json")):
        record = json.loads(request.read_text())
        stem = request.name[:-len(".request.json")]
        response_path = log_dir / f"{stem}.response.json"
        result_path = log_dir / f"{stem}.result.json"
        usage = {}
        if response_path.is_file():
            usage = (json.loads(response_path.read_text()).get("usage") or {})
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        details_out = usage.get("output_tokens_details") or {}
        details_in = usage.get("input_tokens_details") or {}
        rows.append({"ordinal": record.get("request_ordinal"), "mode": record.get("mode"),
                     "status": result.get("status"), "error_type": result.get("error_type"),
                     "input_tokens": usage.get("input_tokens"), "cached_tokens": details_in.get("cached_tokens"),
                     "output_tokens": usage.get("output_tokens"),
                     "reasoning_tokens": details_out.get("reasoning_tokens"), "log": request.name})
    return rows


def run_task(args) -> int:
    import gpt_field.agent_speedup_v2 as policy_mod
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FINE_CROP_PX
    from gpt_field.agent_speedup_v2_plan import part_roles, render_bddl, compact_chars
    from gpt_field.agent_speedup_v2_sim import V2SimProcess
    from gpt_field.agent_speedup_v2_types import RobotModel
    from gpt_field.mark_prompting import coarse_cell_center, unrotate_pixel, rotate_quarter, coarse_overlay
    from gpt_field.mark_prompting import FINGERTIP_EDGE
    h = harness()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task_dir = out / args.task
    if task_dir.exists():
        raise SystemExit(f"{task_dir} exists; tasks are never re-run into the same directory")
    task_dir.mkdir()
    (task_dir / "calls").mkdir()
    ledger = read_ledger(out)
    ledger["max_calls"] = args.max_calls
    allowed = args.max_calls - int(ledger["used"]) - args.reserve_per_task * args.remaining_tasks
    record = {"version": PLAN_CHECK_VERSION, "task": args.task, "started_utc": utc_stamp(),
              "allowed_calls": allowed, "ledger_used_before": ledger["used"], "errors": []}
    if allowed < 1:
        record["errors"].append("no call budget left")
        write_json(task_dir / "task.json", record)
        return 2

    def factory(config, options):
        return V2SimProcess(dataclasses.replace(config, collision_check_mode="none"), monitor_options=options)

    sim, spec, config, monitor = h.start_sim(args.task, oracle=True, sim_factory=factory)
    client = None
    calls = []
    try:
        record["sim_config"] = {"suite": spec.suite, "task_id": spec.task_id, "collision_check_mode": "none",
                                "scene": spec.scene}
        obs, episode = h.reset_episode(sim, spec, 0)
        record["episode"] = episode
        if not episode.get("matches_frozen"):
            record["errors"].append("reset sha256 differs from the frozen hash")
        record["visible_owners_at_reset"] = {cam: visible_owners(sim, cam) for cam in ("wrist", "external")}
        robot_model = RobotModel()
        driver = Driver(sim, obs)
        cal = calibrate(driver, robot_model)
        view = driver.view
        k_truth, right_truth = truth_turns(sim, view)
        k = cal["turns"]
        k_source = "calibration"
        if not cal["external_available"]:
            k, k_source = k_truth, "oracle_fallback_no_external_model"
        record["calibration"] = {key: v for key, v in cal.items() if key != "external"}
        record["calibration"].update(k_truth=k_truth, image_right_truth=right_truth, k_used=k, k_source=k_source,
                                     plan_view_state8=list(view.state))
        record["visible_owners_at_plan"] = {cam: visible_owners(sim, cam) for cam in ("wrist", "external")}
        ext, wrist = view.external_rgb_upright, view.wrist_rgb_upright
        Image.fromarray(ext).save(task_dir / "external_upright.png")
        Image.fromarray(wrist).save(task_dir / "wrist_upright.png")
        Image.fromarray(rotate_quarter(wrist, k)).save(task_dir / "wrist_shown_turned.png")
        save_masks(sim, args.task, task_dir, record["visible_owners_at_plan"])
        if args.dry:
            record["dry"] = True
            return 0

        api = dataclasses.replace(h_config(), max_api_calls=int(allowed))
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
        client = AgentSpeedUpV2Client(api, task_prompt=obs.prompt, token_provider=token_provider,
                                      log_dir=task_dir / "calls")
        events = policy_mod.EventSink(lambda: driver.steps)
        fulfiller = policy_mod.NeedFulfiller(client, events)
        ctx = SimpleNamespace(view=view, cal=SimpleNamespace(external=cal["external"]), events=events, holding=None,
                              active_relation=None, monitor=SimpleNamespace(clear=lambda scope: None))
        runner = policy_mod.StepRunner(ctx, relations=SimpleNamespace())
        if k_source != "calibration":
            runner.turns = lambda _view: k
        began = time.monotonic()
        generator = runner.obtain_plan("plan")
        attempts = []
        try:
            need = next(generator)
            while True:
                answer = fulfiller.fulfil(need, view, calls)
                meta = client.last_request_meta or {}
                attempts.append({"raw": answer.raw, "parse_error": answer.parse_error,
                                 "schema_error": answer.schema_error, "latency_s": answer.call.latency_s,
                                 "reask_error_sent": need.reask_error, "quarter_turns": need.quarter_turns,
                                 "fingertip_edge": need.fingertip_edge,
                                 "previous_plan_sent": bool(meta.get("previous_plan")),
                                 "reask_check": client.last_reask_check if need.reask_error is not None else None,
                                 "parsed": None if answer.plan is None else answer.plan.to_json()})
                need = generator.send(answer)
        except StopIteration as stop:
            plan, last = stop.value
        record["plan_wall_s"] = round(time.monotonic() - began, 2)
        validation = [row for row in events.drain() if row["event"] in ("plan_validation", "relation_plan")]
        record["plan_attempts"] = attempts
        record["plan_events"] = validation
        record["plan_valid"] = plan is not None
        record["reasks"] = max(0, len(attempts) - 1)
        if plan is None:
            write_json(task_dir / "task.json", record)
            return 0
        roles = part_roles(plan)
        record["plan"] = plan.to_json()
        record["bddl"] = render_bddl(plan)
        record["compact_chars"] = compact_chars(plan.to_json())
        record["roles"] = {pid: sorted(r) for pid, r in roles.items()}
        (task_dir / "plan.bddl").write_text(record["bddl"] + "\n")
        if args.plan_only:
            record["plan_only"] = True
            return 0

        # ---- plan-cell grounding (ORACLE, no calls)
        cells = {}
        for part in plan.parts:
            row = {"text": part.text, "move": part.move, "w": part.w_cell, "x": part.x_cell}
            if part.w_cell:
                block = cell_block_upright(part.w_cell, k)
                row["w_block_upright"] = list(block)
                row["w_shares"] = shares(segment_at(sim, "wrist", block_uvs(block)))
            if part.x_cell:
                block = cell_block_upright(part.x_cell, 0)
                row["x_block_upright"] = list(block)
                row["x_shares"] = shares(segment_at(sim, "external", block_uvs(block)))
            cells[part.pid] = row
        record["plan_cells"] = cells

        # ---- pointing rounds
        pointing = {}
        overlay_items = {"wrist": [], "external": []}
        for part in plan.parts:
            part_roles_set = set(roles.get(part.pid, ()))
            wanted = part_roles_set & set(POINT_ROLES)
            if not wanted:
                continue
            purpose = "grasp_point" if "grip" in part_roles_set else "region_seed"
            if part.w_cell:
                camera = "wrist"
                seed_uv = unrotate_pixel(coarse_cell_center(part.w_cell, IMAGE_PX), k, IMAGE_PX)
            elif part.x_cell:
                camera = "external"
                seed_uv = coarse_cell_center(part.x_cell, IMAGE_PX)
            else:
                pointing[part.pid] = {"skipped": "w = x = 0 (not visible per the plan)", "roles": sorted(wanted)}
                continue
            remaining = allowed - len(calls)
            if remaining < 1:
                pointing[part.pid] = {"skipped": "call budget", "roles": sorted(wanted)}
                continue
            try:
                result = point_and_score(sim, fulfiller, view, calls, part=part, camera=camera, seed_uv=seed_uv,
                                         k=k, purpose=purpose, roles=wanted)
            except Exception as exc:          # budget or API error: keep what we have
                pointing[part.pid] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "roles": sorted(wanted)}
                record["errors"].append(f"pointing {part.pid}: {type(exc).__name__}")
                break
            pointing[part.pid] = result
            block = cell_block_upright(part.w_cell, k) if camera == "wrist" else cell_block_upright(part.x_cell, 0)
            marks = [(a["uv"], ":" + ("f" if a["kind"] == "point_fine" else "c")) for a in result["attempts"]
                     if a["uv"] is not None]
            crop = None
            fine = [a for a in result["attempts"] if a["kind"] == "point_fine"]
            if fine and fine[0]["center_uv"] is not None:
                cu, cv = fine[0]["center_uv"]
                half = FINE_CROP_PX / 2.0
                crop = (max(0, cu - half), max(0, cv - half), min(IMAGE_PX, cu + half), min(IMAGE_PX, cv + half))
            overlay_items[camera].append({"pid": part.pid, "cell_block": block, "crop": crop, "marks": marks})
        record["pointing"] = pointing
        for camera, rgb in (("wrist", wrist), ("external", ext)):
            draw_overlay(rgb, task_dir / f"overlay_{camera}.png", overlay_items[camera],
                         f"{args.task} {camera} k={k}")
        # all plan cells on both cameras
        cell_items = {"wrist": [], "external": []}
        for part in plan.parts:
            if part.w_cell:
                cell_items["wrist"].append({"pid": part.pid, "cell_block": cell_block_upright(part.w_cell, k)})
            if part.x_cell:
                cell_items["external"].append({"pid": part.pid, "cell_block": cell_block_upright(part.x_cell, 0)})
        for camera, rgb in (("wrist", wrist), ("external", ext)):
            draw_overlay(rgb, task_dir / f"cells_{camera}.png", cell_items[camera], f"{args.task} plan cells {camera}")
        return 0
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {str(exc)[:300]}")
        (task_dir / "traceback.txt").write_text(traceback.format_exc())
        return 1
    finally:
        record["calls"] = [c.to_dict() for c in calls]
        record["call_usage"] = call_usage(task_dir / "calls")
        if client is not None:
            stats = client.history_stats()
            record["client_stats"] = {key: stats.get(key) for key in (
                "calls_by_kind", "accepted_by_kind", "output_tokens_by_kind", "reasoning_tokens_by_kind",
                "visible_tokens_by_kind", "over_visible_budget_by_kind", "overload_retry_requests",
                "transient_retry_requests", "transient_retry_codes", "reask_requests",
                "reask_rejected_for_dropped_steps")}
            record["client_identity"] = {"prompt_version": client.prompt_version,
                                         "reasoning_effort": client.reasoning_effort, "model": client.config.model}
        record["model_calls"] = len(record["call_usage"])
        record["finished_utc"] = utc_stamp()
        write_json(task_dir / "task.json", record)
        ledger = read_ledger(out)
        ledger["max_calls"] = args.max_calls
        ledger["used"] = int(ledger["used"]) + record["model_calls"]
        ledger["tasks"][args.task] = record["model_calls"]
        write_ledger(out, ledger)
        try:
            sim.close()
        except Exception:
            pass


def body_map(sim, camera, step_px=4):
    """ORACLE body label of every ``step_px`` lattice pixel (scoring only): (names, labels[rows, cols]) with labels
    indexing names; "robot" for robot bodies, "none" for no geom."""
    centre = (step_px - 1) / 2.0
    uvs = [(u + centre, v + centre) for v in range(0, IMAGE_PX, step_px) for u in range(0, IMAGE_PX, step_px)]
    labels, names = [], {}
    chunk = 4096
    for i in range(0, len(uvs), chunk):
        for pixel in segment_at(sim, camera, uvs[i:i + chunk]):
            if pixel is None or pixel.get("body") is None:
                name = "none"
            elif pixel.get("robot"):
                name = "robot"
            else:
                name = pixel["body"]
            labels.append(names.setdefault(name, len(names)))
    side = IMAGE_PX // step_px
    return list(names), np.asarray(labels, dtype=np.int16).reshape(side, side)


def replay_task(args) -> int:
    """--replay: replay (reset + Calibrate+) of a finished task, checked bit-identical against the stored plan view
    (state8 and both upright images); writes ORACLE body maps (``bodymap_<camera>.npz``, scoring only) and, with
    --resume-pointing, runs the pointing round (real model calls, ledger-capped) for the stored plan's parts whose
    round did not complete in the first run (an API error), recorded as ``resumed``."""
    from gpt_field.agent_speedup_v2_plan import parse_plan, part_roles
    from gpt_field.agent_speedup_v2_sim import V2SimProcess
    from gpt_field.agent_speedup_v2_types import RobotModel
    h = harness()
    out = Path(args.out).resolve()
    task_dir = out / args.task
    rec = json.loads((task_dir / "task.json").read_text())

    def factory(config, options):
        return V2SimProcess(dataclasses.replace(config, collision_check_mode="none"), monitor_options=options)

    sim, spec, _config, _monitor = h.start_sim(args.task, oracle=True, sim_factory=factory)
    client = None
    calls = []
    replay = {"started_utc": utc_stamp(), "errors": []}
    try:
        obs, episode = h.reset_episode(sim, spec, 0)
        driver = Driver(sim, obs)
        cal = calibrate(driver, RobotModel())
        view = driver.view
        same_state = list(view.state) == list(rec["calibration"]["plan_view_state8"])
        same_images = all(np.array_equal(np.asarray(Image.open(task_dir / f"{name}_upright.png")), image)
                          for name, image in (("external", view.external_rgb_upright),
                                              ("wrist", view.wrist_rgb_upright)))
        replay.update(sha256=episode["sha256"], k=cal["turns"], same_state=same_state, same_images=same_images)
        if not (same_state and same_images and cal["turns"] == rec["calibration"]["k_used"]):
            replay["errors"].append("replay differs from the stored plan view; nothing written")
            return 1
        for camera in ("wrist", "external"):
            names, labels = body_map(sim, camera)
            np.savez_compressed(task_dir / f"bodymap_{camera}.npz", names=np.asarray(names), labels=labels,
                                step_px=np.asarray(4))
        if not args.resume_pointing:
            return 0
        import gpt_field.agent_speedup_v2 as policy_mod
        from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        ledger = read_ledger(out)
        allowed = args.max_calls - int(ledger["used"])
        todo = []
        plan = parse_plan(rec["plan"])
        roles = part_roles(plan)
        for part in plan.parts:
            wanted = set(roles.get(part.pid, ())) & set(POINT_ROLES)
            done = "attempts" in (rec.get("pointing") or {}).get(part.pid, {})
            if wanted and not done and part_camera_seed(part, rec["calibration"]["k_used"])[0] is not None:
                todo.append((part, wanted))
        replay["resume_parts"] = [part.pid for part, _ in todo]
        if not todo or allowed < 1:
            return 0
        (task_dir / "calls_resume").mkdir(exist_ok=True)
        api = dataclasses.replace(h_config(), max_api_calls=int(allowed))
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
        client = AgentSpeedUpV2Client(api, task_prompt=obs.prompt, token_provider=token_provider,
                                      log_dir=task_dir / "calls_resume")
        fulfiller = policy_mod.NeedFulfiller(client, policy_mod.EventSink(lambda: driver.steps))
        k = rec["calibration"]["k_used"]
        for part, wanted in todo:
            camera, seed_uv = part_camera_seed(part, k)
            purpose = "grasp_point" if "grip" in roles.get(part.pid, ()) else "region_seed"
            previous = rec["pointing"].get(part.pid)
            try:
                result = point_and_score(sim, fulfiller, view, calls, part=part, camera=camera, seed_uv=seed_uv,
                                         k=k, purpose=purpose, roles=wanted)
            except Exception as exc:
                replay["errors"].append(f"pointing {part.pid}: {type(exc).__name__}: {str(exc)[:160]}")
                break
            result["resumed"] = {"first_run": previous}
            rec["pointing"][part.pid] = result
        return 0
    except Exception as exc:
        replay["errors"].append(f"{type(exc).__name__}: {str(exc)[:300]}")
        (task_dir / "replay_traceback.txt").write_text(traceback.format_exc())
        return 1
    finally:
        replay["finished_utc"] = utc_stamp()
        if calls or client is not None:
            usage = call_usage(task_dir / "calls_resume")
            replay["calls"] = [c.to_dict() for c in calls]
            replay["call_usage"] = usage
            rec["model_calls"] = int(rec.get("model_calls") or 0) + len(usage)
            rec["call_usage"] = list(rec.get("call_usage") or []) + [{**u, "resumed": True} for u in usage]
            rec["calls"] = list(rec.get("calls") or []) + [{**c.to_dict(), "resumed": True} for c in calls]
            ledger = read_ledger(out)
            ledger["used"] = int(ledger["used"]) + len(usage)
            ledger["tasks"][args.task] = int(ledger["tasks"].get(args.task, 0)) + len(usage)
            write_ledger(out, ledger)
        rec.setdefault("replays", []).append(replay)
        write_json(task_dir / "task.json", rec)
        try:
            sim.close()
        except Exception:
            pass


def h_config():
    """The runner's API configuration (v1 protocol helper, read-only)."""
    spec = importlib.util.spec_from_file_location("_agent_speedup_v2_runner_readonly",
                                                  ROOT / "scripts" / "evaluate_agent_speedup_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configuration()


# ----------------------------------------------------------------------------------------------- summary
# Scoring-only reference (evaluator side, never sent to the model): what each official goal needs in the v2 relation
# vocabulary, from the BDDL goal AND init (an object already open, a goal atom the sentence does not state).
TASK_NEEDS = {
    "libero_10_t0": {"goal": "(In alphabet_soup basket) (In tomato_sauce basket)",
                     "needs": ["place_in(alphabet soup, basket)", "place_in(tomato sauce, basket)"],
                     "kinds": ["place_in", "place_in"]},
    "libero_10_t1": {"goal": "(In cream_cheese basket) (In butter basket)",
                     "needs": ["place_in(cream cheese, basket)", "place_in(butter, basket)"],
                     "kinds": ["place_in", "place_in"]},
    "libero_10_t2": {"goal": "(Turnon flat_stove) (On moka_pot flat_stove_cook_region)",
                     "needs": ["turn(knob) until effect(burner)", "place_on(moka pot, burner)"],
                     "kinds": ["turn", "place_on"]},
    "libero_10_t3": {"goal": "(Close cabinet_bottom) (In black_bowl cabinet_bottom); init: bottom drawer OPEN",
                     "needs": ["place_in(black bowl, bottom drawer)", "slide close(bottom drawer)"],
                     "kinds": ["place_in", "slide"]},
    "libero_10_t4": {"goal": "(On porcelain_mug plate_1 [left]) (On white_yellow_mug plate_2 [right])",
                     "needs": ["place_on(white mug, left plate)", "place_on(yellow-white mug, right plate)"],
                     "kinds": ["place_on", "place_on"]},
    "libero_10_t5": {"goal": "(In black_book desk_caddy_back_contain_region)",
                     "needs": ["place_in(book, caddy back compartment)"], "kinds": ["place_in"]},
    "libero_10_t6": {"goal": "(On porcelain_mug plate) (On chocolate_pudding table_plate_right_region)",
                     "needs": ["place_on(white mug, plate)",
                               "pudding to a table spot right of the plate (no 'beside' relation: push_to / "
                               "place_on the table)"],
                     "kinds": ["place_on", "push_to|place_on"]},
    "libero_10_t7": {"goal": "(In alphabet_soup basket) (In cream_cheese basket)",
                     "needs": ["place_in(alphabet soup, basket)", "place_in(cream cheese, basket)"],
                     "kinds": ["place_in", "place_in"]},
    "libero_10_t8": {"goal": "(On moka_pot_1 cook_region) (On moka_pot_2 cook_region) (Turnon flat_stove); init: "
                             "stove already ON (Turnon in init: it must stay on)",
                     "needs": ["place_on(moka pot 1, stove)", "place_on(moka pot 2, stove)"],
                     "kinds": ["place_on", "place_on"]},
    "libero_10_t9": {"goal": "(In white_yellow_mug microwave_heating_region) (Close microwave); init: door OPEN",
                     "needs": ["place_in(yellow-white mug, microwave)", "close the door (hinged: turn until stop)"],
                     "kinds": ["place_in", "turn"]},
    "bowl_plate": {"goal": "(On akita_black_bowl plate)", "needs": ["place_on(black bowl, plate)"],
                   "kinds": ["place_on"]},
    "drawer_bowl": {"goal": "(In akita_black_bowl cabinet_top_region); init: top drawer closed",
                    "needs": ["slide open(top drawer)", "place_in(bowl, top drawer)"], "kinds": ["slide", "place_in"]},
}
CONTROLLER_KINDS = ("hold", "turn", "place_on", "release")     # BR's CONTROLLERS (slide / place_in / push_to: not
#                                                                implemented in Wave 2)


def _kinds_match(plan_kinds, need_kinds):
    """(missing, extra) relation kinds: the plan's establishing steps (hold / release excluded) against the needs
    (an "a|b" need is met by either)."""
    have = [k for k in plan_kinds if k not in ("hold", "release")]
    missing = []
    for need in need_kinds:
        options = need.split("|")
        found = next((k for k in have if k in options), None)
        if found is None:
            missing.append(need)
        else:
            have.remove(found)
    return missing, have


def summarize(out: Path) -> dict:
    out = Path(out)
    intent_path = out / "intent.json"
    intent = json.loads(intent_path.read_text()) if intent_path.is_file() else {}
    rows = []
    for key in TASK_ORDER:
        path = out / key / "task.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text())
        usage = rec.get("call_usage", [])
        tokens = {name: sum(int(u.get(name) or 0) for u in usage)
                  for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")}
        by_kind = {}
        for c in rec.get("calls", []):
            item = by_kind.setdefault(c["kind"], {"n": 0, "latency_s": []})
            item["n"] += 1
            item["latency_s"].append(round(c["latency_s"], 1))
        plan = rec.get("plan") or {}
        steps = plan.get("steps", [])
        kinds = [s["do"] for s in steps]
        needs = TASK_NEEDS.get(key, {})
        missing, extra = _kinds_match(kinds, needs.get("kinds", []))
        task_intent = intent.get(key, {})
        points = []
        for pid, p in (rec.get("pointing") or {}).items():
            want = task_intent.get(pid, {})
            label = p.get("final_label")
            pixel = p.get("final_pixel") or {}
            hit_obj = None
            if want.get("owners") is not None and label is not None:
                hit_obj = label in want["owners"]
            dist = None
            uv = p.get("final_uv") or p.get("fallback_uv")
            mask_path = out / key / f"masks_{p.get('camera')}.npz"
            if uv is not None and want.get("owners") and mask_path.is_file():
                masks = np.load(mask_path)
                union = None
                for owner in want["owners"]:
                    name = f"owner:{owner}"
                    if name in masks.files:
                        union = masks[name] if union is None else (union | masks[name])
                if union is not None and union.any():
                    vs, us = np.nonzero(union)
                    dist = round(float(np.min(np.hypot(us - uv[0], vs - uv[1]))), 1)
            points.append({"pid": pid, "text": (plan_part(plan, pid) or {}).get("text"), "roles": p.get("roles"),
                           "dist_to_intended_px": dist, "resumed": bool(p.get("resumed")),
                           "camera": p.get("camera"), "path": p.get("path"), "calls": len(p.get("attempts", [])),
                           "found": p.get("final_uv") is not None, "uv": p.get("final_uv") or p.get("fallback_uv"),
                           "hit_owner": label, "hit_body": pixel.get("body"), "hit_geom": pixel.get("geom"),
                           "disc": p.get("final_disc_shares"), "intended": want.get("owners"),
                           "object_hit": hit_obj, "part_hit": want.get("part_hit"), "note": want.get("note"),
                           "skipped": p.get("skipped") or p.get("error")})
        attempts = rec.get("plan_attempts", [])
        clipped, odd = [], []
        from gpt_field.agent_speedup_v2_plan import SCENE_MAX_CHARS, TEXT_MAX_CHARS, TXT_MAX_CHARS
        for n, attempt in enumerate(attempts):
            raw = attempt.get("raw") or {}
            fields = [("scene", raw.get("scene"), SCENE_MAX_CHARS)]
            fields += [(f"part {q['id']}", q.get("text"), TEXT_MAX_CHARS) for q in raw.get("parts", [])]
            fields += [(f"step {i} until txt", u.get("txt"), TXT_MAX_CHARS) for i, st in enumerate(raw.get("steps", []))
                       for u in st.get("until", []) if u.get("txt") is not None]
            for name, text, limit in fields:
                if isinstance(text, str) and len(text) >= limit:
                    clipped.append(f"attempt {n}: {name}")
                if isinstance(text, str) and any(ord(ch) > 127 for ch in text):
                    odd.append(f"attempt {n}: {name}: {text[-6:]!r}")
        hold_before = sum(1 for a, b in zip(steps, steps[1:]) if a["do"] == "hold" and b["do"] in ("place_on", "place_in")
                          and a["obj"] == b["obj"])
        issues = [[i["code"] for i in e.get("issues", [])] for e in rec.get("plan_events", [])
                  if e.get("event") == "plan_validation"]
        rows.append({"task": key, "prompt": (rec.get("episode") or {}).get("prompt"),
                     "reset_matches_frozen": (rec.get("episode") or {}).get("matches_frozen"),
                     "k": (rec.get("calibration") or {}).get("k_used"),
                     "k_truth": (rec.get("calibration") or {}).get("k_truth"),
                     "valid": rec.get("plan_valid"), "reasks": rec.get("reasks"), "issues_per_attempt": issues,
                     "parse_or_schema_errors": [a.get("parse_error") or ("schema" if a.get("schema_error") else None)
                                                for a in attempts],
                     "scene": plan.get("scene"), "parts": [[p["id"], p["text"], p["move"], p["w"], p["x"]]
                                                           for p in plan.get("parts", [])],
                     "roles": rec.get("roles"), "bddl": rec.get("bddl"), "steps": kinds,
                     "needs": needs.get("needs"), "needs_goal": needs.get("goal"), "missing_kinds": missing,
                     "extra_kinds": extra,
                     "unimplemented_kinds": sorted({k for k in kinds if k not in CONTROLLER_KINDS}),
                     "first_attempt_issue_codes": issues[0] if issues else None,
                     "clipped_text_fields": clipped, "non_ascii_text": odd, "redundant_hold_steps": hold_before,
                     "previous_plan_sent": [a.get("previous_plan_sent") for a in attempts],
                     "reask_checks": [a.get("reask_check") for a in attempts if a.get("reask_check") is not None],
                     "sanitized_fields": sanitized_fields(attempts[-1]) if attempts else [],
                     "pointing": points, "model_calls": rec.get("model_calls"), "calls_by_kind": by_kind,
                     "tokens": tokens, "plan_wall_s": rec.get("plan_wall_s"), "errors": rec.get("errors"),
                     "intent_notes": task_intent.get("_notes")})
    summary = {"version": PLAN_CHECK_VERSION, "tasks": rows,
               "totals": {"tasks": len(rows), "valid_plans": sum(1 for r in rows if r["valid"]),
                          "first_plans_valid": sum(1 for r in rows if r["first_attempt_issue_codes"] == []),
                          "reasks": sum(int(r["reasks"] or 0) for r in rows),
                          "first_attempt_issue_codes": sorted({c for r in rows
                                                               for c in (r["first_attempt_issue_codes"] or [])}),
                          "redundant_hold_steps": sum(r["redundant_hold_steps"] for r in rows),
                          "clipped_text_fields": sum(len(r["clipped_text_fields"]) for r in rows),
                          "non_ascii_text": sum(len(r["non_ascii_text"]) for r in rows),
                          "missing_kinds": {r["task"]: r["missing_kinds"] for r in rows if r["missing_kinds"]},
                          "model_calls": sum(int(r["model_calls"] or 0) for r in rows),
                          "tokens": {name: sum(r["tokens"][name] for r in rows)
                                     for name in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                  "cached_tokens")}}}
    write_json(out / "summary.json", summary)
    return summary


def sanitized_fields(attempt):
    """The text fields of an attempt the plan parse changed (raw model text vs the parsed plan)."""
    raw, parsed = attempt.get("raw") or {}, attempt.get("parsed")
    if not parsed:
        return []
    out = []
    if raw.get("scene") != parsed.get("scene"):
        out.append("scene")
    for a, b in zip(raw.get("parts", []), parsed.get("parts", [])):
        if a.get("text") != b.get("text"):
            out.append(f"part {a.get('id')}")
    for i, (a, b) in enumerate(zip(raw.get("steps", []), parsed.get("steps", []))):
        for j, (u, v) in enumerate(zip(a.get("until", []), b.get("until", []))):
            if u.get("txt") != v.get("txt"):
                out.append(f"step {i} until {j} txt")
    return out


def plan_part(plan, pid):
    return next((p for p in plan.get("parts", []) if p["id"] == pid), None)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--reserve-per-task", type=int, default=DEFAULT_RESERVE_PER_TASK)
    parser.add_argument("--remaining-tasks", type=int, default=0, help="tasks still to run after this one")
    parser.add_argument("--store-dir")
    parser.add_argument("--profile")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry", action="store_true", help="sim, calibration and ORACLE only; no model call")
    parser.add_argument("--plan-only", action="store_true", help="stop after the plan: no pointing calls")
    parser.add_argument("--replay", action="store_true", help="replay a finished task: ORACLE body maps")
    parser.add_argument("--resume-pointing", action="store_true",
                        help="with --replay: point the parts whose round an API error cut short (model calls)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summarize:
        summary = summarize(Path(args.out))
        print(json.dumps(summary["totals"]))
        return 0
    if not args.task:
        raise SystemExit("--task is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise SystemExit("unset CUDA_VISIBLE_DEVICES (render device via MUJOCO_EGL_DEVICE_ID)")
    if args.replay:
        return replay_task(args)
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
