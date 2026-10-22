"""Freeze inputs for 16 matched RGB+joint-feedback trials; never launch a rollout.

Evaluator-only demonstration/layout comparisons are written separately from the
observable demonstration bundles. No model, simulator, network or auth is used.
Run with PYTHONPATH=src and the local LIBERO Python (h5py/torch are needed only
when reading the already-downloaded official initialization files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "visual-field-matrix-v1"
SCENE_SOURCES = (
    ("bowl_original", "bowl_plate", "bowl_distance_medium_one_demo_001", "official_distribution_control"),
    ("bowl_relayout", "bowl_plate", "bowl_relayout_distance_medium_one_demo_002", "custom_layout_shift"),
    ("bowl_obstacles", "bowl_plate", "bowl_obstacle_distance_medium_one_demo_001", "custom_layout_and_distractor_asset_shift"),
    ("drawer_original", "drawer_bowl", "drawer_distance_medium_one_demo_001", "official_distribution_control"),
)
DEMO_SOURCES = {"bowl_plate": "bowl_distance_medium_one_demo_001", "drawer_bowl": "drawer_distance_medium_one_demo_001"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze_file(source, target, expected_sha256):
    """Refuse altered input or replacement of an existing evidence artifact."""
    source, target = Path(source), Path(target)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Input hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(content)
    return {"path": str(target.resolve()), "sha256": expected_sha256, "byte_count": len(content)}


def joint_layout(model_xml):
    """Read recorded model joint ordering, without constructing a simulator."""
    root = ET.fromstring(model_xml)
    world = root.find("worldbody")
    if world is None or root.find("include") is not None:
        raise ValueError("An expanded recorded model XML is required")
    positions, nq, nv = {}, 0, 0
    for joint in world.iter():
        if joint.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
        if kind not in ("free", "ball", "hinge", "slide"):
            raise ValueError("Unknown joint type")
        name = joint.get("name")
        if not name or name in positions:
            raise ValueError("Recorded joints require unique explicit names")
        width, velocity_width = {"free": (7, 6), "ball": (4, 3)}.get(kind, (1, 1))
        positions[name] = {"qpos_address": nq, "width": width, "type": kind}
        nq += width
        nv += velocity_width
    return positions, nq, nv


def compare_target_positions(demo_initial, evaluation_initial, demo_layout,
                             target_names, *, evaluation_addresses=None,
                             minimum_xy_difference_m=0.001):
    """Keep only distances in output; state/pose vectors remain evaluator-only.

    MuJoCo's recorded flat state is time,qpos,qvel, with no activation state in
    these tasks. Validate this shape for the demonstration. Custom model target
    qpos addresses come from the previously verified name-based mapping.
    """
    joints, nq, nv = demo_layout
    demo_initial = np.asarray(demo_initial, dtype=float)
    evaluation_initial = np.asarray(evaluation_initial, dtype=float)
    if demo_initial.shape != (1 + nq + nv,) or not np.isfinite(demo_initial).all():
        raise ValueError("Unsupported demo flat-state layout")
    if evaluation_initial.ndim != 1 or not np.isfinite(evaluation_initial).all():
        raise ValueError("Invalid evaluation flat state")
    if evaluation_addresses is None and evaluation_initial.shape != demo_initial.shape:
        raise ValueError("A changed model requires explicit verified joint mapping")
    differences = {}
    for name in target_names:
        joint = joints[name]
        if joint["type"] != "free":
            raise ValueError("Position comparison requires a free target joint")
        start = 1 + joint["qpos_address"]
        other = 1 + (joint["qpos_address"] if evaluation_addresses is None else evaluation_addresses[name])
        if other < 1 or other + 3 > len(evaluation_initial):
            raise ValueError("Invalid mapped target joint address")
        delta = evaluation_initial[other:other+3] - demo_initial[start:start+3]
        xy = float(np.linalg.norm(delta[:2]))
        differences[name] = {"xy_difference_m": xy, "xyz_difference_m": float(np.linalg.norm(delta)),
                             "different_xy": xy > minimum_xy_difference_m}
    if not all(v["different_xy"] for v in differences.values()):
        raise ValueError("Frozen demonstration does not have distinct target positions; select a separate reference explicitly")
    return {"target_positions_differ": True, "minimum_xy_difference_m": minimum_xy_difference_m,
            "targets": differences,
            "interpretation": "Distinct initial XY positions; not proof of unseen target assets or a distribution shift."}


def build_matrix(scenes, demonstrations, output):
    """Pure plan builder; shared scene/reference identity is explicit per pair."""
    output = Path(output).resolve()
    if {s["id"] for s in scenes} != {s[0] for s in SCENE_SOURCES} or len(scenes) != 4:
        raise ValueError("Expected the four declared scenes exactly once")
    trials, pairs = [], []
    for scene in scenes:
        for shot in ("zero", "one"):
            pair_id = f"{scene['id']}_{shot}"
            pair_index = len(pairs)
            # Balanced deterministic order, declared before any outcomes.
            order = ("hybrid", "direct") if pair_index % 4 in (0, 3) else ("direct", "hybrid")
            pair_trials = []
            for controller in order:
                trial_id = f"{pair_id}_{controller}"
                demo = demonstrations[scene["task"]] if shot == "one" else None
                trial = {"id": trial_id, "pair_id": pair_id, "scene": scene["id"],
                         "controller": controller, "task": scene["task"], "shot": shot,
                         "demo": demo["path"] if demo else None,
                         "demo_sha256": demo["sha256"] if demo else None,
                         "initial_state": scene.get("initial_state"),
                         "initial_state_sha256": scene.get("initial_state_sha256"),
                         "bddl": scene.get("bddl"), "bddl_sha256": scene["actual_bddl_sha256"],
                         "seed": 7, "episode_index": 0,
                         "output": str(output / "trials" / trial_id)}
                args = ["scripts/evaluate_visual_field_subscription.py", "--output", trial["output"],
                        "--task", trial["task"], "--controller", controller,
                        "--episode-index", "0", "--seed", "7"]
                for key, flag in (("demo", "--demo"), ("initial_state", "--initial-state"), ("bddl", "--bddl")):
                    if trial[key] is not None:
                        args.extend((flag, trial[key]))
                trial["runner_arguments"] = args
                pair_trials.append(trial_id)
                trials.append(trial)
            pairs.append({"id": pair_id, "scene": scene["id"], "shot": shot,
                          "execution_order": pair_trials, "status": "not_started"})
    return {
        "schema_version": VERSION, "status": "prepared_inputs_only", "trial_count": 16,
        "pair_count": 8, "task_family_count": 2, "replicates_per_condition": 1,
        "live_model_calls": 0, "simulator_rollouts": 0,
        "settings": {"model": "gpt-6-astra", "reasoning_effort": "medium",
                     "billing": "chatgpt_subscription", "api_key_calls_allowed": 0,
                     "include_arm_joints": True, "collision_check_mode": "task_manipulation_v1",
                     "direct_action_steps": 5, "max_steps": None, "max_wall_time_s": None,
                     "max_model_calls": None, "model_completion_stopping": False,
                     "policy_inputs": ["two_RGB_views", "eef_pose", "gripper_qpos", "arm_joint_positions",
                                       "task_language", "observable_history", "one_external_demo_if_enabled"],
                     "forbidden_policy_inputs": ["depth", "supplied_camera_calibration", "object_poses",
                                                 "contact_identities", "goal_status", "privileged_scene_state"],
                     "primary_time_metric": "control_elapsed_s - timing_components.video_render_s",
                     "local_vision_inference_included_in_time": True},
        "comparison": {"label": "Visual Field Hybrid versus grounded Direct with shared visual feedback",
                       "shared_local_visual_feedback": True, "pure_direct_baseline_included": False,
                       "limitation": "This pair does not measure the end-to-end gain over a separate pure Direct without shared local visual processing."},
        "outcome_rules": {"success": "evaluator goal plus no forbidden contact under the shared task contact contract",
                          "collision": "terminal scored failure; preserve raw LIBERO goal separately",
                          "user_stop": "incomplete, not a completion time or automatic timeout failure",
                          "infrastructure_error": "separate incomplete attempt; preserve retries as additional attempts",
                          "speedup_eligibility": "both arms successful with matching audited inputs and criteria",
                          "ineligible_times": "attempt durations only; no speed ratio",
                          "video_accounting": "retrospective measured subtraction, not a recording-disabled rollout"},
        "scope": {"target_object_OOD": False, "object_asset_overlap": "target bowls and destinations remain the demonstrated assets",
                  "original_scenes": "official-distribution controls with different demonstration initial target positions",
                  "shift_claim": "relative to the supplied demonstration, not to Astra pretraining or prior experiment exposure",
                  "novel_distractors": "obstacle scene replaces distractors with ketchup and milk absent from the bowl demonstration",
                  "statistical_limit": "one trial per arm/scene/shot; no reliability or generalized speed claim"},
        "runtime_prerequisites": [
            "Freeze and validate the new runner, clients, RGB+q interface and common guidance before live calls.",
            "Verify task_manipulation_v1 contacts allow intended drawer/handle/support interactions and stop forbidden contacts identically.",
            "Check official initialization and task-file hashes still match prepared records; compare actual reset RGB/state/q within each pair.",
            "Audit actual requests for medium subscription routing, exact demo or no demo, and absence of hidden state/calibration/depth.",
            "Include local vision processing, calibration-from-motion, planner and all request waiting in control time.",
            "Keep preparation/demo-position evidence evaluator-only; only demo bundle and allowed live observations enter model input.",
            "A no-cap run may stall indefinitely; only an explicit user stop or the declared terminal conditions end it.",
        ], "demonstrations": demonstrations, "scenes": scenes, "pairs": pairs, "trials": trials,
    }


def prepare(project, output):
    """Read archived evidence, verify independent references, then freeze inputs."""
    from gpt_field.demonstration import load_demonstration
    import h5py
    import torch

    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preparation directory: {output}")
    demos, internals, planned_copies = {}, {}, []
    for task, source_run in DEMO_SOURCES.items():
        source = project / "runs" / source_run / "demonstration.json"
        demo = load_demonstration(source, task)
        metadata, bundle = demo.metadata(), read_json(source)
        hdf5 = project / "data/libero_demonstrations" / bundle["source"]["suite"] / bundle["source"]["file_name"]
        if sha256(hdf5) != bundle["source"]["file_sha256"]:
            raise ValueError("Downloaded demonstration HDF5 hash mismatch")
        with h5py.File(hdf5, "r") as dataset:
            recorded = dataset["data"][bundle["source"]["demo_key"]]
            xml = recorded.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode("utf-8")
            internals[task] = (np.asarray(recorded.attrs["init_state"], dtype=float), joint_layout(xml))
            xml_sha = hashlib.sha256(xml.encode()).hexdigest()
        frozen = output / "inputs" / f"{task}_demonstration.json"
        demos[task] = {"path": str(frozen), "sha256": metadata["bundle_sha256"],
                       "source": str(source), "source_hdf5": str(hdf5),
                       "source_hdf5_sha256": metadata["source_file_sha256"],
                       "source_model_xml_sha256": xml_sha, "demo_key": metadata["demo_key"],
                       "action_count": metadata["action_count"], "image_count": metadata["image_count"],
                       "demo_state_format": "8D eef/gripper after-state; live arm joints are an additional permitted input",
                       "selection": "Retain the existing independently selected demo_0; verify target XY differs before reuse."}
        planned_copies.append((source, frozen, metadata["bundle_sha256"]))
    scenes, evidence = [], {}
    for scene_id, task, source_run, category in SCENE_SOURCES:
        run = project / "runs" / source_run
        manifest = read_json(run / "manifest.json")
        config = manifest["configuration"]
        with (run / "trace.jsonl").open() as stream:
            reset = json.loads(next(stream))
        status = reset["sim_status"]
        initial = Path(status["initial_state_file"])
        if sha256(initial) != status["initial_state_sha256"]:
            raise ValueError("Official initial-state source changed")
        prompt_file = config["task"].replace(" ", "_") + ".bddl"
        official_bddl = Path(config["simulator"]["libero_root"]) / "libero/libero/bddl_files" / config["simulator"]["suite"] / prompt_file
        original_hash = status.get("original_bddl_sha256", status["bddl_sha256"])
        if sha256(official_bddl) != original_hash:
            raise ValueError("Official task source changed")
        scene = {"id": scene_id, "task": task, "suite": config["simulator"]["suite"],
                 "task_id": config["task_id"], "task_prompt": config["task"],
                 "episode_index": 0, "seed": 7, "category": category,
                 "source_run": str(run), "source_manifest_sha256": sha256(run / "manifest.json"),
                 "official_initial_state": {"path": str(initial), "sha256": status["initial_state_sha256"]},
                 "official_bddl": {"path": str(official_bddl), "sha256": original_hash},
                 "initial_state": None, "initial_state_sha256": None, "bddl": None,
                 "actual_bddl_sha256": status["bddl_sha256"],
                 "prior_reset_observation_sha256": hashlib.sha256(json.dumps(reset["observation"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 "prior_reset_joint_feedback_available": False}
        planned_copies.extend([(initial, output / "inputs" / scene_id / "official.pruned_init", status["initial_state_sha256"]),
                               (official_bddl, output / "inputs" / scene_id / "original_task.bddl", original_hash)])
        if config.get("initial_state_override"):
            artifact = run / config["initial_state_override"]["artifact"]
            evaluation = np.load(artifact, allow_pickle=False)
            frozen = output / "inputs" / scene_id / "initial_state.npy"
            scene["initial_state"], scene["initial_state_sha256"] = str(frozen), config["initial_state_override"]["sha256"]
            planned_copies.append((artifact, frozen, scene["initial_state_sha256"]))
        else:
            evaluation = np.asarray(torch.load(initial, map_location="cpu", weights_only=False)[0])
        addresses = None
        if config.get("task_bddl_override"):
            artifact = run / config["task_bddl_override"]["artifact"]
            frozen = output / "inputs" / scene_id / "task_override.bddl"
            scene["bddl"] = str(frozen)
            planned_copies.append((artifact, frozen, scene["actual_bddl_sha256"]))
            layout = read_json(project / "runs/bowl_obstacle_validation_001/layout.json")
            addresses = {j["joint"]: j["target_qpos_address"][0] for j in layout["state_mapping"]["mapped_joints"]}
            scene["joint_mapping_evidence"] = {"path": str(project / "runs/bowl_obstacle_validation_001/layout.json"),
                                              "sha256": sha256(project / "runs/bowl_obstacle_validation_001/layout.json")}
        demo_initial, layout = internals[task]
        target_names = ["akita_black_bowl_1_joint0"] + (["plate_1_joint0"] if task == "bowl_plate" else [])
        evidence[scene_id] = compare_target_positions(demo_initial, evaluation, layout, target_names, evaluation_addresses=addresses)
        evidence[scene_id]["target_assets_overlap_demo"] = True
        evidence[scene_id]["drawer_fixture_position_compared"] = False if task == "drawer_bowl" else None
        scenes.append(scene)
    # Validate every source before creating any output; copying then verifies bytes again.
    for source, target, expected in planned_copies:
        if sha256(source) != expected:
            raise ValueError(f"Preparation input changed: {source}")
    matrix = build_matrix(scenes, demos, output)
    output.mkdir(parents=True)
    artifacts = [freeze_file(*item) for item in planned_copies]
    write_json(output / "demo_position_independence.json", {
        "schema_version": "visual-field-demo-position-independence-v1",
        "scope": "Offline evaluator evidence only; never append to a model request or visual tracker input.",
        "method": "Compare recorded demo init_state free-joint XY with official/custom test state, using verified named-joint qpos addresses.",
        "limitation": "Initial states precede evaluation settling; no fresh simulation, fixture-position comparison or full-scene OOD certification.",
        "scenes": evidence})
    matrix["frozen_input_artifacts"] = artifacts
    matrix["preparation_script_sha256"] = sha256(__file__)
    matrix["demo_position_evidence"] = {"path": str(output / "demo_position_independence.json"),
                                        "sha256": sha256(output / "demo_position_independence.json")}
    write_json(output / "matrix.json", matrix)
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    matrix = prepare(args.project, args.output)
    print(json.dumps({"matrix": str(args.output.resolve() / "matrix.json"),
                      "trials": matrix["trial_count"], "pairs": matrix["pair_count"],
                      "status": matrix["status"], "live_model_calls": 0}))


if __name__ == "__main__":
    main()
