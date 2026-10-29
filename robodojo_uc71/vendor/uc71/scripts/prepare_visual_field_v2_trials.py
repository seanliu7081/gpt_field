"""Prepare four fresh zero-shot Hybrid v2 trials without model or simulator calls."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "visual-field-v2-zero-shot-four-scenes-v1"
SCENES = ("bowl_original", "bowl_relayout", "bowl_obstacles", "drawer_original")
DIRECT_REFERENCES = {
    "bowl_original": "historical_zero_shot_missing_001/original_bowl_zero_direct",
    "bowl_relayout": "historical_zero_shot_missing_001/relayout_bowl_zero_direct",
    "bowl_obstacles": "bowl_obstacle_direct_medium_zero_shot_001",
    "drawer_original": "historical_zero_shot_missing_001/original_drawer_zero_direct",
}
OBSERVATION_KEYS = {"observation/image", "observation/wrist_image", "observation/state",
                    "observation/joint_position", "observation/joint_velocity", "prompt"}


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_bytes(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Input hash mismatch: {path}")
    return data


def reconstruct_initial(scene, validation):
    """Combine an archived allowed reset with validated q/dq; require exact hash."""
    validation = Path(validation)
    records = [read_json(validation / f"{scene['id']}_reset_{i}.json") for i in (1, 2)]
    with (Path(scene["source_run"]) / "trace.jsonl").open() as stream:
        reset = json.loads(next(stream))
    if reset.get("event") != "reset":
        raise ValueError("Archived first record must be a reset")
    obs = dict(reset["observation"])
    for record in records:
        status, config = record["initial_status"], record["configuration"]
        if (record["scene"] != scene["id"] or status["task_id"] != scene["task_id"]
                or status["episode_index"] != scene["episode_index"]
                or status["bddl_sha256"] != scene["actual_bddl_sha256"]
                or status["initial_state_sha256"] != scene["official_initial_state"]["sha256"]
                or config["seed"] != scene["seed"] or not config["include_arm_joints"]
                or config["collision_check_mode"] != "task_manipulation_v1"
                or status["collision"]["detected"]):
            raise ValueError("Reset validation does not match the declared scene")
        if scene.get("initial_state") and status.get("initial_state_override_sha256") != scene["initial_state_sha256"]:
            raise ValueError("Custom initial-state validation differs")
        obs["observation/joint_position"] = record["joint_position"]
        obs["observation/joint_velocity"] = record["joint_velocity"]
        if (set(obs) != OBSERVATION_KEYS or len(obs["observation/state"]) != 8
                or len(obs["observation/joint_position"]) != 7
                or len(obs["observation/joint_velocity"]) != 7
                or obs["prompt"] != scene["task_prompt"]):
            raise ValueError("Invalid allowed initial observation")
        if hashlib.sha256(canonical(obs)).hexdigest() != record["observation_sha256"]:
            raise ValueError("Exact validated initial observation hash differs")
        for key, view in (("observation/image", "external"), ("observation/wrist_image", "wrist")):
            rgb = record["rgb"][view]
            png = checked_bytes(validation / rgb["artifact"], rgb["sha256"])
            if base64.b64decode(obs[key].split(",", 1)[1], validate=True) != png:
                raise ValueError("Validated RGB differs from the archived observation")
    if records[0]["observation_sha256"] != records[1]["observation_sha256"]:
        raise ValueError("Repeated validation resets differ")
    return obs, {"observation_sha256": records[0]["observation_sha256"],
                 "archived_reset_sha256": hashlib.sha256(canonical(reset)).hexdigest(),
                 "source_trace": str(Path(scene["source_run"]) / "trace.jsonl"),
                 "validation_records": [{"path": str(validation / f"{scene['id']}_reset_{i}.json"),
                                         "sha256": sha256(validation / f"{scene['id']}_reset_{i}.json")}
                                        for i in (1, 2)]}


def runner_arguments(trial):
    args = ["scripts/evaluate_visual_field_subscription.py", "--output", trial["output"],
            "--task", trial["task"], "--controller", "hybrid", "--method-version", "v2",
            "--episode-index", str(trial["episode_index"]), "--seed", str(trial["seed"]),
            "--expected-initial", trial["expected_initial"]]
    for key, flag in (("initial_state", "--initial-state"), ("bddl", "--bddl")):
        if trial.get(key):
            args.extend((flag, trial[key]))
    return args


def prepare(project=ROOT, output=None, source_matrix=None):
    project = Path(project).resolve()
    output = Path(output or project / "runs/visual_field_v2_zero_shot_001").resolve()
    source_matrix = Path(source_matrix or project / "runs/visual_field_matrix_v1/matrix.json").resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh preparation directory: {output}")
    matrix = read_json(source_matrix)
    if matrix.get("schema_version") != "visual-field-matrix-v1":
        raise ValueError("Expected the frozen v1 scene matrix")
    by_id = {s["id"]: s for s in matrix["scenes"]}
    if set(by_id) != set(SCENES) or len(matrix["scenes"]) != 4:
        raise ValueError("Exactly the four declared scenes are required")
    listed = {str(Path(a["path"]).resolve()): a for a in matrix["frozen_input_artifacts"]}
    planned, scenes, trials = {}, [], []

    def schedule(source, target, expected):
        data = checked_bytes(source, expected)
        planned[Path(target)] = data
        return {"path": str(target), "sha256": expected, "byte_count": len(data)}

    for name in SCENES:
        old = by_id[name]
        task = "drawer_bowl" if name == "drawer_original" else "bowl_plate"
        if old["task"] != task or old["seed"] != 7 or old["episode_index"] != 0:
            raise ValueError("Historical scene task, seed or episode changed")
        scene = {k: old[k] for k in ("id", "task", "suite", "task_id", "task_prompt", "seed", "episode_index", "category", "actual_bddl_sha256")}
        for key, filename in (("official_bddl", "original_task.bddl"), ("official_initial_state", "official.pruned_init")):
            checked_bytes(old[key]["path"], old[key]["sha256"])
            frozen_old = source_matrix.parent / "inputs" / name / filename
            if listed.get(str(frozen_old), {}).get("sha256") != old[key]["sha256"]:
                raise ValueError("Official input is not the declared frozen matrix artifact")
            scene[key] = dict(old[key])
            scene[key + "_snapshot"] = schedule(frozen_old, output / "inputs" / name / filename, old[key]["sha256"])
        for key, filename, hash_key in (("initial_state", "initial_state.npy", "initial_state_sha256"),
                                        ("bddl", "task_override.bddl", "actual_bddl_sha256")):
            scene[key] = None
            if old.get(key):
                source = Path(old[key]).resolve()
                if listed.get(str(source), {}).get("sha256") != old[hash_key]:
                    raise ValueError("Custom scene is not the declared frozen matrix artifact")
                scene[key] = schedule(source, output / "inputs" / name / filename, old[hash_key])["path"]
            scene[key + "_sha256"] = old.get(hash_key)
        obs, evidence = reconstruct_initial(old, source_matrix.parent / "validation")
        expected = output / "inputs" / name / "expected_initial.json"
        planned[expected] = canonical(obs) + b"\n"
        scene["initial_observation_evidence"] = evidence
        scene["expected_initial"] = str(expected)
        scene["expected_initial_sha256"] = hashlib.sha256(planned[expected]).hexdigest()
        reference_run = project / "runs" / DIRECT_REFERENCES[name]
        reference_path = reference_run / "summary.json"
        reference = {"run": str(reference_run), "summary": str(reference_path),
                     "summary_sha256": sha256(reference_path),
                     "snapshot": str(output / "references" / f"{name}_direct_summary.json"),
                     "comparison_type": "nonconcurrent_nonmatched_reference", "speed_ratio": None,
                     "limitation": "Older Direct has different observation/prompt and possibly contact criteria; no fair matched speed or success comparison."}
        schedule(reference_path, reference["snapshot"], reference["summary_sha256"])
        trial = {"id": f"{name}_zero_hybrid_v2", "scene": name, "task": task,
                 "controller": "hybrid", "method_version": "v2", "shot": "zero", "demo": None,
                 "seed": old["seed"], "episode_index": old["episode_index"],
                 "initial_state": scene["initial_state"], "bddl": scene["bddl"],
                 "expected_initial": str(expected), "expected_initial_sha256": scene["expected_initial_sha256"],
                 "output": str(output / "trials" / f"{name}_zero_hybrid_v2"), "direct_reference": reference}
        trial["runner_arguments"] = runner_arguments(trial)
        scenes.append(scene)
        trials.append(trial)
    plan = {"schema_version": VERSION, "status": "prepared_inputs_only", "project": str(project),
            "source_matrix": {"path": str(source_matrix), "sha256": sha256(source_matrix)},
            "trial_count": 4, "new_direct_trials": 0, "model_calls_during_preparation": 0,
            "settings": {"model": "gpt-6-astra", "reasoning_effort": "medium", "billing": "chatgpt_subscription",
                         "method_version": "v2", "controller": "hybrid", "shots": 0,
                         "api_key_fallback": False, "include_arm_joints": True,
                         "collision_check_mode": "task_manipulation_v1", "model_completion_stopping": False,
                         "max_steps": None, "max_wall_time_s": None, "max_model_calls": None,
                         "automatic_stall_termination": False},
            "queue_rules": {"continue_after": ["success", "collision_failure"],
                            "pause_after": ["user_stop", "subscription_error", "infrastructure_error"],
                            "implicit_retries": False, "signal_behavior": "Forward SIGINT to active child and await cleanup; never launch another after a queue stop."},
            "reporting": {"time": "control_elapsed_s - timing_components.video_render_s",
                          "vision_planning_and_calibration_time_included": True,
                          "incomplete_times": "Attempt durations only, not completion times.",
                          "attempt_counts": "Observable command/recovery proxies, not certified physical grasp failures.",
                          "speed_ratios": "Always null: archived Direct conditions are not matched to v2."},
            "scenes": scenes, "trials": trials,
            "frozen_input_artifacts": [{"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}
                                       for path, data in sorted(planned.items())]}
    output.mkdir(parents=True)
    for path, data in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    write_json(output / "matrix.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-matrix", type=Path)
    args = parser.parse_args()
    plan = prepare(args.project, args.output, args.source_matrix)
    print(json.dumps({"trial_count": len(plan["trials"]), "model_calls": 0, "new_direct_trials": 0}))


if __name__ == "__main__":
    main()
