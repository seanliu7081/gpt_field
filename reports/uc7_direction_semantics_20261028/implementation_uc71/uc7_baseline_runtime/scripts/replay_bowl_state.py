"""Validate deterministic recovery of a recorded bowl-episode prefix, without inference.

This diagnostic replays already executed actions in a fresh, isolated simulator.
It is not another scored evaluation and never continues beyond the saved prefix.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zlib

import numpy as np

import gpt_field.libero as libero_module
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import decode_image


TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
TRACE_NAME = "libero_spatial_t00_i00_field_attempt00.jsonl.gz"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def snapshot_records(path):
    """Freeze bytes first; an absent live gzip footer is allowed, corruption is not."""
    path = Path(path)
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Trace shrank while taking its read-only snapshot")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        plain = decoder.decompress(raw) + decoder.flush()
    except zlib.error as exc:
        raise ValueError("Trace gzip is corrupt") from exc
    if decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Trace must contain one gzip member without trailing data")
    lines = plain.split(b"\n")
    partial = lines.pop()
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid complete JSON record at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Trace record {number} is not an object")
        rows.append(row)
    if decoder.eof and partial:
        raise ValueError("Closed trace ends with a non-newline-terminated record")
    return rows, {"trace": path.name, "snapshot_bytes": size, "snapshot_sha256": sha256(raw),
                  "gzip_complete": decoder.eof, "partial_final_line_ignored": bool(partial),
                  "complete_records": len(rows)}


def vector(values, length, label, *, normalized=False):
    if not isinstance(values, list) or len(values) != length or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or (normalized and abs(v) > 1) for v in values
    ):
        raise ValueError(f"{label} must contain {length} finite {'normalized ' if normalized else ''}numbers")
    return values


def select_prefix(rows, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not rows or rows[0].get("event") != "reset":
        raise ValueError("Trace must begin with exactly one reset")
    reset = rows[0]
    previous = vector(reset["observation"]["observation/state"], 8, "reset state")
    if reset["observation"].get("prompt") != TASK:
        raise ValueError("Reset task prompt mismatch")
    status = reset.get("sim_status", {})
    if status.get("task_id") != 0 or status.get("episode_index") != 0 or status.get("steps") != 0:
        raise ValueError("Reset task/initial-state identity mismatch")
    pairs, pending = [], None
    for row in rows[1:]:
        index = len(pairs)
        if type(row.get("step")) is not int or row["step"] != index:
            raise ValueError("Trace steps are not contiguous and ordered")
        if row.get("event") == "executed":
            if pending is None:
                raise ValueError("Executed event has no matching decision")
            after = vector(row.get("next_state"), 8, "executed state")
            if row.get("observation", {}).get("observation/state") != after:
                raise ValueError("Executed state and observation/state disagree")
            if row["observation"].get("prompt") != TASK:
                raise ValueError("Executed task prompt mismatch")
            if row.get("sim_status", {}).get("steps") != index + 1:
                raise ValueError("Executed simulator step counter mismatch")
            pairs.append((pending, row))
            previous, pending = after, None
        elif "event" not in row and "action" in row:
            if pending is not None:
                raise ValueError("Decision lacks its matching executed event")
            if status.get("terminated"):
                raise ValueError("Recorded action occurs after simulator termination")
            before = vector(row.get("state"), 8, "decision state")
            if before != previous:
                raise ValueError("Decision state disagrees with preceding measured state")
            vector(row.get("action"), 7, "action", normalized=True)
            pending = row
        else:
            raise ValueError("Unexpected trace event")
        if row.get("event") == "executed":
            status = row["sim_status"]
    if len(pairs) < steps:
        raise ValueError(f"Requested {steps} executed actions but only {len(pairs)} complete decision/executed pairs exist")
    return reset, pairs[:steps], {"available_completed_actions": len(pairs),
                                  "trailing_unexecuted_decision_ignored": pending is not None}


def load_identity(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    original = json.loads((run_dir / "simulator.json").read_text())
    identity = manifest["configuration"]
    if canonical_hash(identity) != manifest.get("identity_hash"):
        raise ValueError("Run manifest identity hash mismatch")
    if (identity.get("task_ids") != [0] or identity.get("episode_indices") != [0]
            or identity.get("reference", {}).get("task") != TASK):
        raise ValueError("Run must identify the exact requested bowl task and initial state 0")
    config = LiberoConfig(**identity["simulator"])
    if config.suite != "libero_spatial" or config.seed != 7 or config.max_steps is not None:
        raise ValueError("Replay requires the original spatial/seed7/unbounded simulator protocol")
    if not config.python_executable or not Path(config.python_executable).is_absolute():
        raise ValueError("Original simulator must identify an absolute virtualenv interpreter")
    # Keep the recorded interpreter string intact; resolving its symlink changes environments.
    sources = identity["software"]["source_sha256"]
    package = Path(libero_module.__file__).parent
    for name in ("__init__.py", "libero.py", "observation.py"):
        if sha256((package / name).read_bytes()) != sources.get(name):
            raise ValueError(f"Original simulator source mismatch: {name}")
    if original["tasks"][0].get("prompt") != TASK or original["metadata"].get("task_order_index") != 0:
        raise ValueError("Original simulator task identity mismatch")
    return config, original, manifest["identity_hash"]


def image_comparison(expected, actual):
    result = {}
    for key in ("observation/image", "observation/wrist_image"):
        reference_url, replay_url = expected.get(key), actual.get(key)
        if reference_url is None or replay_url is None:
            result[key] = {"available": False, "match": None}
            continue
        reference = np.asarray(decode_image(reference_url))
        replay = np.asarray(decode_image(replay_url))
        same_shape = reference.shape == replay.shape
        difference = np.abs(reference.astype(np.int16) - replay.astype(np.int16)) if same_shape else None
        result[key] = {
            "available": True,
            "reference_png_sha256": sha256(base64.b64decode(reference_url.split(",", 1)[1], validate=True)),
            "replay_png_sha256": sha256(base64.b64decode(replay_url.split(",", 1)[1], validate=True)),
            "reference_rgb_sha256": sha256(reference.tobytes()), "replay_rgb_sha256": sha256(replay.tobytes()),
            "match": bool(same_shape and np.array_equal(reference, replay)),
            "max_pixel_error": int(difference.max()) if difference is not None else None,
            "mean_pixel_error": float(difference.mean()) if difference is not None else None,
        }
    return result


def replay(run_dir, *, steps=500, state_atol=1e-10, simulator_factory=SimProcess):
    if isinstance(state_atol, bool) or not math.isfinite(state_atol) or state_atol < 0:
        raise ValueError("state_atol must be finite and nonnegative")
    run_dir = Path(run_dir)
    config, original, identity_hash = load_identity(run_dir)
    rows, snapshot = snapshot_records(run_dir / TRACE_NAME)
    reset, pairs, selection = select_prefix(rows, steps)
    result = {"diagnostic": "deterministic_action_replay", "scored_evaluation": False,
              "astra_requests": 0, "continuation_performed": False,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_run": str(run_dir), "source_identity_hash": identity_hash,
              "source_snapshot": {**snapshot, **selection}, "task": TASK,
              "task_id": 0, "episode_index": 0, "seed": config.seed,
              "python_executable": config.python_executable, "requested_actions": steps,
              "completed_actions": 0, "state_atol": state_atol, "max_state_error": 0.0,
              "max_state_error_by_component": [0.0] * 8, "state_checks": 0,
              "per_action_max_state_error": [], "status": "validating",
              "simulator_closed": False, "helper_sha256": sha256(Path(__file__).read_bytes()),
              "render_environment": {name: os.environ.get(name) for name in (
                  "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "scope": "Replay of a previously executed prefix, not another evaluation or policy continuation. Agreement validates only this recorded prefix."}

    def check_state(actual, expected, label):
        expected = np.asarray(vector(expected, 8, label), dtype=float)
        error = np.abs(np.asarray(actual, dtype=float) - expected)
        if error.shape != (8,) or not np.isfinite(error).all():
            raise ValueError(f"Invalid measured state at {label}")
        maximum = float(error.max())
        result["state_checks"] += 1
        result["max_state_error"] = max(result["max_state_error"], maximum)
        result["max_state_error_by_component"] = np.maximum(result["max_state_error_by_component"], error).tolist()
        if maximum > state_atol:
            raise ValueError(f"State mismatch at {label}: {maximum} > {state_atol}")
        return maximum

    started = time.monotonic()
    simulator = simulator_factory(config)
    try:
        with simulator as sim:
            if {"metadata": sim.metadata, "tasks": sim.list_tasks()} != original:
                raise ValueError("Simulator metadata/source/controller/task provenance changed")
            result["simulator_metadata_match"] = True
            obs = sim.reset(0, 0)
            if obs.prompt != TASK or sim.status != reset["sim_status"]:
                raise ValueError("Reset task/state-file/BDDL/status identity mismatch")
            result["reset_state_error"] = check_state(obs.state, reset["observation"]["observation/state"], "reset")
            result["reset_images"] = image_comparison(reset["observation"], obs.to_dict())
            for decision, executed in pairs:
                step = decision["step"]
                before_error = check_state(obs.state, decision["state"], f"before action {step}")
                obs, success = sim.step(decision["action"])
                result["completed_actions"] += 1
                after_error = check_state(obs.state, executed["next_state"], f"after action {step}")
                result["per_action_max_state_error"].append(max(before_error, after_error))
                if obs.prompt != TASK or sim.status != executed["sim_status"] or success != executed["success"]:
                    raise ValueError(f"Simulator task/status outcome mismatch after action {step}")
                if result["completed_actions"] % 100 == 0:
                    print(json.dumps({"event": "replay_progress", "completed_actions": result["completed_actions"],
                                      "max_state_error": result["max_state_error"]}), flush=True)
            result["final_images"] = image_comparison(pairs[-1][1]["observation"], obs.to_dict())
            available = [item["match"] for item in result["final_images"].values() if item["available"]]
            result["final_images_match"] = all(available) if len(available) == 2 else None
            result["status"] = "validated" if result["final_images_match"] is True else "image_mismatch_or_unavailable"
    except Exception as exc:
        result.update(status="validation_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        simulator.close()
        result["simulator_closed"] = True
        result["elapsed_s"] = time.monotonic() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-atol", type=float, default=1e-10)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Replay evidence already exists; choose a fresh evidence filename")
    result = replay(args.run, steps=args.steps, state_atol=args.state_atol)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"event": "replay_finished", **{key: result.get(key) for key in (
        "status", "completed_actions", "max_state_error", "final_images_match", "simulator_closed", "error")},
        "output": str(output)}), flush=True)
    return 0 if result["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
