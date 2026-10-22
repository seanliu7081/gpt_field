"""Explain a TERMINAL direct-history run by replaying its executed actions.

Run with the source manifest's simulator Python, not the policy Python. This
standalone evaluator never imports an Astra client or continues the policy.
It does not edit the source run or the simulator bridge. Privileged object and
contact evidence is collected only in this isolated, unscored reconstruction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
TERMINAL = {"success", "budget_exceeded", "user_stop", "infrastructure_error"}
BOWL = "akita_black_bowl_1"
PLATE = "plate_1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def vector(values, size, label, *, normalized=False):
    if not isinstance(values, list) or len(values) != size or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or (normalized and abs(v) > 1) for v in values
    ):
        raise ValueError(f"Invalid {label}")
    return values


def load_terminal_source(source):
    """Freeze complete records only after both terminal markers agree."""
    source = Path(source).absolute()
    manifest, status, summary, original = (
        read(source / name) for name in
        ("manifest.json", "run_state.json", "summary.json", "simulator.json")
    )
    if status.get("status") not in TERMINAL or summary.get("status") != status["status"]:
        raise ValueError("Replay requires matching terminal run_state.json and summary.json")
    if status.get("stage") != "finished":
        raise ValueError("Source run has not finished finalizing")
    config = manifest["configuration"]
    if digest(config) != manifest.get("identity_hash"):
        raise ValueError("Source manifest identity mismatch")
    if (config.get("kind") != "fresh_direct_history_baseline"
            or config.get("mode") != "live_astra_direct_history"
            or config.get("task") != TASK or config.get("task_id") != 0
            or config.get("episode_index") != 0):
        raise ValueError("Source must be the real direct-history bowl task, initial state zero")
    sim = config["simulator"]
    if (sim.get("suite") != "libero_spatial" or sim.get("seed") != 7
            or sim.get("max_steps") is not None or sim.get("settle_steps") != 10
            or sim.get("image_size") != 224 or sim.get("render_size") != 256):
        raise ValueError("Source simulator configuration does not match the recorded protocol")
    root = Path(__file__).resolve().parents[1]
    for name, expected in config["software"]["source_sha256"].items():
        if sha(root / "src/gpt_field" / name) != expected:
            raise ValueError(f"Source package changed: {name}")
    runner = config["runner"]
    if sha(root / runner["path"]) != runner["sha256"]:
        raise ValueError("Original evaluator runner changed")
    raw = (source / "trace.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Terminal trace is not newline-complete")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or rows[0].get("event") != "reset":
        raise ValueError("Trace must begin with reset")
    reset = rows[0]
    if reset["observation"]["prompt"] != TASK:
        raise ValueError("Reset task mismatch")
    previous = vector(reset["observation"]["observation/state"], 8, "reset state")
    reset_status = reset["sim_status"]
    if (reset_status.get("task_id"), reset_status.get("episode_index"), reset_status.get("steps")) != (0, 0, 0):
        raise ValueError("Reset task/state identity mismatch")
    pairs, pending, feedback_steps = [], None, []
    for row in rows[1:]:
        event = row.get("event")
        if event == "action":
            if pending is not None or row.get("step") != len(pairs):
                raise ValueError("Action sequence is not contiguous")
            if vector(row["state"], 8, "action state") != previous:
                raise ValueError("Action state disagrees with prior observation")
            vector(row["action"], 7, "normalized action", normalized=True)
            pending = row
        elif event == "executed":
            if pending is None or row.get("step") != len(pairs):
                raise ValueError("Executed event has no contiguous matching action")
            if row["action"] != pending["action"]:
                raise ValueError("Issued and executed actions disagree")
            after = vector(row["next_state"], 8, "executed state")
            if (row["observation"]["observation/state"] != after
                    or row["observation"]["prompt"] != TASK
                    or row["sim_status"]["steps"] != len(pairs) + 1
                    or row["sim_status"]["success"] != row["success"]):
                raise ValueError("Executed observation/status mismatch")
            pairs.append((pending, row))
            previous, pending = after, None
        elif event == "feedback":
            if not pairs or pending is not None or row.get("step") != len(pairs) - 1:
                raise ValueError("Feedback has no matching executed transition")
            if row["step"] in feedback_steps:
                raise ValueError("Duplicate feedback")
            before, after = pairs[-1]
            f = row["feedback"]
            if (f["before_state"] != before["state"] or f["after_state"] != after["next_state"]
                    or f["normalized_action"] != after["action"]):
                raise ValueError("Feedback contradicts actual transition")
            feedback_steps.append(row["step"])
        else:
            raise ValueError(f"Unexpected trace event: {event}")
    if len(pairs) != summary.get("steps") or len(pairs) != status.get("step"):
        raise ValueError("Terminal action count does not match complete trace transitions")
    if len(feedback_steps) != summary.get("measured_feedback_count"):
        raise ValueError("Terminal feedback count does not match trace")
    filenames = ("manifest.json", "run_state.json", "summary.json", "simulator.json", "trace.jsonl")
    source_hashes = {name: sha(source / name) for name in filenames}
    return {"source": source, "manifest": manifest, "summary": summary,
            "original": original, "reset": reset, "pairs": pairs,
            "source_hashes": source_hashes, "trailing_unexecuted_action": pending is not None}


def on_components(bowl_position, plate_position, contact):
    distance = math.dist(bowl_position[:2], plate_position[:2])
    above = bool(bowl_position[2] >= plate_position[2])
    centered = bool(distance < 0.03)
    return {"bowl_body_origin_at_or_above_plate": above,
            "bowl_plate_collision_contact": bool(contact),
            "xy_body_origin_distance_m": distance,
            "xy_distance_strictly_below_0_03_m": centered,
            "on_components_conjunction": above and bool(contact) and centered}


def evaluator_evidence(worker, step, action):
    """Read-only physics queries; never sent to an Astra client or policy."""
    env = worker.env.env
    positions = {}
    for name in (BOWL, PLATE, "akita_black_bowl_2"):
        geometry = env.object_states_dict[name].get_geom_state()
        positions[name] = {"body_origin_xyz_m": geometry["pos"].copy().tolist(),
                           "body_quaternion_wxyz": geometry["quat"].copy().tolist()}
    bowl, plate = env.object_states_dict[BOWL], env.object_states_dict[PLATE]
    components = on_components(positions[BOWL]["body_origin_xyz_m"],
                               positions[PLATE]["body_origin_xyz_m"], plate.check_contact(bowl))
    predicate = bool(plate.check_ontop(bowl))
    success = bool(worker.env.check_success())
    if components["on_components_conjunction"] != predicate or predicate != success:
        raise ValueError("Independent component calculation disagrees with actual LIBERO predicate")
    bowl_geoms = set(env.get_object(BOWL).contact_geoms)
    plate_geoms = set(env.get_object(PLATE).contact_geoms)
    contacts = []
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        first = env.sim.model.geom_id2name(contact.geom1)
        second = env.sim.model.geom_id2name(contact.geom2)
        if (first in bowl_geoms and second in plate_geoms) or (second in bowl_geoms and first in plate_geoms):
            contacts.append({"geom_1": first, "geom_2": second, "contact_distance_m": float(contact.dist)})
    gripper_contact = bool(env.check_contact(env.robots[0].gripper, env.get_object(BOWL)))
    return {"step": step, "normalized_action": action, "objects": positions,
            "goal_components": components, "libero_on_predicate": predicate,
            "libero_success": success, "bowl_plate_collision_pairs": contacts,
            "any_gripper_bowl_1_contact": gripper_contact}


def replay(source, output, *, state_atol=1e-10):
    if not math.isfinite(state_atol) or state_atol < 0:
        raise ValueError("state_atol must be finite and nonnegative")
    loaded = load_terminal_source(source)
    source, output = loaded["source"], Path(output).absolute()
    if output.parent != source.parent or output == source:
        raise ValueError("Diagnostic output must be a separate sibling of the source run")
    if output.exists():
        raise FileExistsError("Diagnostic output already exists; never overwrite evidence")
    manifest = loaded["manifest"]
    expected_python = manifest["configuration"]["simulator"]["python_executable"]
    if os.path.abspath(sys.executable) != os.path.abspath(expected_python):
        raise ValueError(f"Run this helper with the recorded simulator interpreter: {expected_python}")
    # Remove credentials before any simulator import; no API library is used.
    os.environ.pop("OPENAI_API_KEY", None)
    import numpy as np
    from gpt_field.libero import LiberoConfig, _LiberoWorker
    from gpt_field.observation import decode_image

    config = LiberoConfig(**manifest["configuration"]["simulator"])
    output.mkdir()
    result = {"kind": "post_terminal_evaluator_only_diagnostic_replay", "scored_evaluation": False,
              "astra_requests": 0, "policy_continuation": False, "source_run": str(source),
              "source_status": loaded["summary"]["status"], "source_identity_hash": manifest["identity_hash"],
              "source_hashes": loaded["source_hashes"], "helper_sha256": sha(__file__),
              "created_utc": datetime.now(timezone.utc).isoformat(), "simulator": asdict(config),
              "state_atol": state_atol, "status": "validating", "evidence_validated": False,
              "requested_actions": len(loaded["pairs"]), "replayed_actions": 0,
              "state_checks": 0, "image_pairs_checked": 0, "max_state_error": 0.0,
              "max_pixel_error": 0, "trailing_unexecuted_action_ignored": loaded["trailing_unexecuted_action"],
              "simulator_closed": False,
              "render_environment": {name: os.environ.get(name) for name in (
                  "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "scope": "Recorded-action reconstruction only, not a second scored episode or a new policy. "
                       "Robot states and camera pixels are checked; original hidden physics states were not saved."}
    worker, evidence, snapshots = None, [], []
    started = time.monotonic()

    def compare_observation(actual, expected, label):
        if actual["prompt"] != expected["prompt"]:
            raise ValueError(f"Task mismatch at {label}")
        error = np.abs(np.asarray(actual["observation/state"]) - np.asarray(expected["observation/state"]))
        if error.shape != (8,) or not np.isfinite(error).all():
            raise ValueError(f"Invalid robot state at {label}")
        result["state_checks"] += 1
        result["max_state_error"] = max(result["max_state_error"], float(error.max()))
        if float(error.max()) > state_atol:
            raise ValueError(f"Robot state mismatch at {label}: {float(error.max())}")
        for key in ("observation/image", "observation/wrist_image"):
            actual_rgb, expected_rgb = np.asarray(decode_image(actual[key])), np.asarray(decode_image(expected[key]))
            if actual_rgb.shape != expected_rgb.shape:
                raise ValueError(f"Camera shape mismatch at {label}")
            pixel_error = int(np.abs(actual_rgb.astype(np.int16) - expected_rgb.astype(np.int16)).max())
            result["max_pixel_error"] = max(result["max_pixel_error"], pixel_error)
            if pixel_error:
                raise ValueError(f"Camera pixel mismatch at {label}, {key}: {pixel_error}")
        result["image_pairs_checked"] += 1

    try:
        worker = _LiberoWorker(config)
        if {"metadata": worker.metadata, "tasks": worker.tasks()} != loaded["original"]:
            raise ValueError("Simulator metadata, controller, source or task provenance changed")
        result["simulator_metadata_match"] = True
        initial = worker.reset(0, 0)
        if initial["status"] != loaded["reset"]["sim_status"]:
            raise ValueError("Initial state/BDDL/status identity mismatch")
        compare_observation(initial["observation"], loaded["reset"]["observation"], "reset")
        if worker.env.env.parsed_problem["goal_state"] != [["on", BOWL, PLATE]]:
            raise ValueError("Unexpected parsed task goal")
        initial_evidence = evaluator_evidence(worker, -1, None)
        evidence.append(initial_evidence)
        snapshots.append({"reason": "reset", "evidence": initial_evidence})
        initial_height = initial_evidence["objects"][BOWL]["body_origin_xyz_m"][2]
        previous_gripper, previous_dz, crossed_lift = -1, 0.0, False
        for before, after in loaded["pairs"]:
            step, action = before["step"], before["action"]
            measured = worker.step(action)
            result["replayed_actions"] += 1
            if measured["status"] != after["sim_status"] or bool(measured["status"]["success"]) != after["success"]:
                raise ValueError(f"Evaluator status mismatch after action {step}")
            compare_observation(measured["observation"], after["observation"], f"after action {step}")
            current = evaluator_evidence(worker, step, action)
            evidence.append(current)
            grip = 1 if action[6] > 0 else -1 if action[6] < 0 else 0
            reasons = []
            if grip > 0 and previous_gripper <= 0:
                reasons.append("close_command_started")
            if grip < 0 and previous_gripper >= 0:
                reasons.append("open_command_started")
            if action[2] > 0 and previous_dz <= 0:
                reasons.append("upward_translation_command_started")
            height = current["objects"][BOWL]["body_origin_xyz_m"][2]
            if not crossed_lift and height - initial_height >= 0.01:
                crossed_lift = True
                reasons.append("target_bowl_first_1cm_height_increase")
            for reason in reasons:
                snapshots.append({"reason": reason, "evidence": current})
            previous_gripper, previous_dz = grip, action[2]
            if result["replayed_actions"] % 50 == 0:
                print(json.dumps({"event": "diagnostic_replay_progress", "replayed_actions": result["replayed_actions"],
                                  "max_state_error": result["max_state_error"], "max_pixel_error": result["max_pixel_error"]}), flush=True)
        for name, expected in loaded["source_hashes"].items():
            if sha(source / name) != expected:
                raise ValueError(f"Source run changed during replay: {name}")
        peak = max(evidence, key=lambda row: row["objects"][BOWL]["body_origin_xyz_m"][2])
        closest = min(evidence, key=lambda row: row["goal_components"]["xy_body_origin_distance_m"])
        snapshots.extend([{"reason": "target_bowl_peak_height", "evidence": peak},
                          {"reason": "closest_xy_body_origin_alignment", "evidence": closest},
                          {"reason": "final", "evidence": evidence[-1]}])
        result.update(status="validated", evidence_validated=True,
                      final_goal_components=evidence[-1]["goal_components"],
                      final_objects=evidence[-1]["objects"],
                      final_libero_success=evidence[-1]["libero_success"],
                      target_bowl_max_height_increase_m=peak["objects"][BOWL]["body_origin_xyz_m"][2] - initial_height,
                      target_bowl_max_height_step=peak["step"],
                      minimum_xy_body_origin_distance_m=closest["goal_components"]["xy_body_origin_distance_m"],
                      minimum_xy_body_origin_distance_step=closest["step"],
                      goal_component_true_counts=dict(Counter(key for row in evidence[1:] for key in (
                          "bowl_body_origin_at_or_above_plate", "bowl_plate_collision_contact", "xy_distance_strictly_below_0_03_m")
                          if row["goal_components"][key])),
                      snapshots=snapshots,
                      predicate_trace="predicate_trace.jsonl",
                      interpretation="On requires target bowl body-origin z >= plate z, actual collision contact, "
                                     "and XY body-origin distance strictly < 0.03 m. The goal does not require release, "
                                     "upright orientation, prior lifting, or a dwell period. Contact with the gripper "
                                     "alone does not establish a secure grasp.")
    except Exception as exc:
        result.update(status="validation_failed", evidence_validated=False,
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        if worker is not None:
            try:
                worker.close()
                result["simulator_closed"] = True
            except Exception as exc:
                result.update(status="validation_failed", evidence_validated=False,
                              cleanup_error=f"{type(exc).__name__}: {exc}")
        result["elapsed_s"] = time.monotonic() - started
    if result["evidence_validated"]:
        with (output / "predicate_trace.jsonl").open("x") as stream:
            for row in evidence:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    else:
        # Never publish reconstructed object/contact evidence as trusted when
        # original observations, metadata or source provenance did not match.
        for key in ("snapshots", "final_goal_components", "final_objects", "predicate_trace",
                    "final_libero_success", "target_bowl_max_height_increase_m", "target_bowl_max_height_step",
                    "minimum_xy_body_origin_distance_m", "minimum_xy_body_origin_distance_step",
                    "goal_component_true_counts", "interpretation"):
            result.pop(key, None)
    with (output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"event": "diagnostic_replay_finished", **{key: result.get(key) for key in (
        "status", "evidence_validated", "replayed_actions", "state_checks", "image_pairs_checked",
        "max_state_error", "max_pixel_error", "error")}, "output": str(output)}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-atol", type=float, default=1e-10)
    args = parser.parse_args(argv)
    result = replay(args.source, args.output, state_atol=args.state_atol)
    return 0 if result["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
