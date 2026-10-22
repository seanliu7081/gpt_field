"""POSTHOC contact-versus-predicted-geometry diagnosis of sealed v2 results.

Truth is accessed only after both completed manifests have been rehashed. This
does not change maps, planner settings, paths or original evaluation records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = HERE.parent / "field_planning"
sys.path.insert(0, str(OLD))
from robot_geometry import RobotGeometry
from native_distance_evaluation import NativeDistanceGroundTruthEvaluator
from run_planning import sha256, write_json
from planning_fields import AstraEnvironmentField, flatten_union_components


def verify_run(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["source_unchanged"] or not manifest["inputs_unchanged"]:
        raise ValueError("requires complete sealed run")
    checked = {str(directory / "manifest.json"): sha256(directory / "manifest.json")}
    for path, expected in manifest["consumed_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("consumed input changed: " + path)
        checked[path] = expected
    for path, row in manifest["sources"].items():
        copied = directory / row["copy"]
        if sha256(path) != row["sha256"] or sha256(copied) != row["copy_sha256"]:
            raise ValueError("frozen runtime source changed: " + path)
        checked[path] = row["sha256"]
        checked[str(copied)] = row["copy_sha256"]
    for relative, expected in manifest["artifacts_sha256"].items():
        path = directory / relative
        if sha256(path) != expected:
            raise ValueError("result artifact changed: " + str(path))
        checked[str(path)] = expected
    return manifest, checked


def component_spec(spec, root):
    reachable = set()
    def visit(index):
        if index in reachable:
            return
        reachable.add(index)
        node = spec["nodes"][index]
        if node["left"] is not None:
            visit(node["left"])
            visit(node["right"])
    visit(root)
    order = sorted(reachable)
    mapping = {old: new for new, old in enumerate(order)}
    result = {key: copy.deepcopy(value) for key, value in spec.items() if key not in ("nodes", "root", "_validation")}
    result["nodes"] = []
    for index in order:
        node = copy.deepcopy(spec["nodes"][index])
        if node["left"] is not None:
            node["left"], node["right"] = mapping[node["left"]], mapping[node["right"]]
        result["nodes"].append(node)
    result["root"] = mapping[root]
    return result


def run(plans, evaluation, output):
    plans, evaluation, output = map(lambda p: Path(p).resolve(), (plans, evaluation, output))
    if output.exists():
        raise FileExistsError(output)
    plan_manifest, consumed = verify_run(plans)
    _, evaluation_inputs = verify_run(evaluation)
    consumed.update(evaluation_inputs)
    consumed[str(Path(__file__).resolve())] = sha256(__file__)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    records = metrics["records"]
    index = {(r["snapshot"], r["case_id"], r["seed"], r["method"]): r for r in records}
    if len(records) != 120 or len(index) != 120:
        raise ValueError("expected120unique method/query records")
    keys = {method: {(r["snapshot"], r["case_id"], r["seed"]) for r in records if r["method"] == method} for method in ("astra", "depthpro", "oracle")}
    if any(len(k) != 40 or k != keys["astra"] for k in keys.values()):
        raise ValueError("query pairing mismatch")
    for key in keys["astra"]:
        rows = [index[(*key, method)] for method in keys]
        if any(r["start_q"] != rows[0]["start_q"] or r["goal_q"] != rows[0]["goal_q"] for r in rows):
            raise ValueError("endpoint pairing mismatch")
    statuses = {method: dict(Counter(r["status"] for r in records if r["method"] == method)) for method in keys}
    failures = [r for r in records if r["method"] == "astra" and r["status"] == "planned" and not r["truth"]["valid_at_all_samples"]]
    if statuses["astra"] != {"planned": 40} or statuses["depthpro"] != {"invalid_goal": 40} or len(failures) != 8:
        raise ValueError("unexpected declared final counts")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for snapshot in metrics["settings"]["snapshots"]:
        relevant = [r for r in failures if r["snapshot"] == snapshot]
        if not relevant:
            continue
        robot = RobotGeometry(Path(plan_manifest["preparation"]) / snapshot)
        field = AstraEnvironmentField(snapshot, project_root=PROJECT)
        truth = NativeDistanceGroundTruthEvaluator(HERE.parent / "rgb_scene/references_004/snapshots" / snapshot / "evaluator")
        consumed.update(truth.provenance()["consumed_sha256"])
        components = [(field.environment["nodes"][i]["name"], component_spec(field.environment, i)) for i in flatten_union_components(field.environment)]
        for record in relevant:
            first = record["truth"]["first_invalid_sample"]
            q = np.asarray(first["q"], float)
            truth_check = truth.check(q)
            if truth_check["valid"] or not truth_check["environment_collision"]:
                raise ValueError("reported collision did not reproduce with original native contacts")
            contacts = []
            for contact in truth.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if (a in truth.robot_ids) == (b in truth.robot_ids) or float(contact.dist) >= -1e-7:
                    continue
                world = np.asarray(contact.pos).copy()
                camera = world @ robot.camera_from_world[:3, :3].T + robot.camera_from_world[:3, 3]
                query = field.query(camera[None])
                component_values = [(name, float(field.core.evaluate_field(spec, camera[None])[0])) for name, spec in components]
                closest = min(component_values, key=lambda item: item[1])
                phi = float(query.raw_phi_m[0])
                contacts.append({
                    "geom1": mujoco.mj_id2name(truth.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                    "geom2": mujoco.mj_id2name(truth.model, mujoco.mjtObj.mjOBJ_GEOM, b),
                    "native_contact_distance_m": float(contact.dist),
                    "native_contact_position_world_m": world.tolist(),
                    "native_contact_position_camera_m": camera.tolist(),
                    "predicted_environment_state_at_contact": int(query.state[0]),
                    "predicted_environment_phi_m": phi if np.isfinite(phi) else None,
                    "predicted_environment_domain_supported": bool(query.support[0]),
                    "predicted_free_at_actual_contact_position": bool(query.state[0] == 1),
                    "minimum_implicit_component": closest[0], "minimum_component_phi_m": closest[1],
                    "component_selection_semantics": "minimum implicit scalar among retained union operands; not claimed exact nearest Euclidean surface",
                    "phi_exceeds_absolute_contact_depth": bool(np.isfinite(phi) and phi > abs(float(contact.dist))),
                })
            if not contacts:
                raise ValueError("missing reproduced environment contact")
            points, _ = robot.samples(q)
            sampled = field.query(points)
            own = robot.check(q)
            row = {"snapshot": snapshot, "case_id": record["case_id"], "seed": record["seed"],
                   "first_invalid_sample_index": first["sample_index"], "q7": q.tolist(),
                   "original_native_truth": truth_check, "contact_queries": contacts,
                   "predicted_robot_surface_at_same_dense_truth_pose": sampled.summary(),
                   "robot_self_check_at_same_pose": own,
                   "diagnostic_interpretation": "Predicted environment is free at actual native contact positions: representation/geometry mismatch contributes, not merely the planner skipping an already-represented obstacle." if all(c["predicted_free_at_actual_contact_position"] for c in contacts) else "At least one actual contact is not predicted free; spatial/temporal sampling may also contribute."}
            rows.append(row)
        consumed.update(field.provenance(verify=True)["consumed_sha256"])
    intact = {path: sha256(path) == expected for path, expected in consumed.items()}
    if not all(intact.values()):
        raise RuntimeError("input/source changed during posthoc diagnosis")
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "posthoc failure diagnosis only; no planner feedback or retuning",
              "input_manifests_verified_before_truth_access": True,
              "matched_query_audit": {"records": 120, "queries_per_method": 40, "statuses": statuses,
                                      "astra_truth_valid": 32, "astra_truth_invalid": 8,
                                      "astra_rrt_paths": [{"snapshot": r["snapshot"], "case_id": r["case_id"], "seed": r["seed"], "truth_valid": r["truth"]["valid_at_all_samples"]} for r in records if r["method"] == "astra" and r["reason"] == "rrt_connect"]},
              "failure_rows": rows,
              "all_failed_path_contact_positions_predicted_free": all(all(c["predicted_free_at_actual_contact_position"] for c in row["contact_queries"]) for row in rows),
              "all_failed_path_dense_pose_robot_samples_predicted_free": all(row["predicted_robot_surface_at_same_dense_truth_pose"]["free"] == row["predicted_robot_surface_at_same_dense_truth_pose"]["points"] for row in rows),
              "consumed_sha256": consumed, "inputs_unchanged": intact,
              "limitations": ["Contact positions are MuJoCo's native reported contact locations, not manually chosen image landmarks.",
                              "Positive CSG values are model-space clearance lower bounds, not measured physical clearance.",
                              "This diagnosis demonstrates scene-geometry mismatch at collision locations; it does not establish exact shape, pose or scale attribution.",
                              "Finite surface/edge sampling remains a limitation even when these particular failures show positive field values at true contacts."]}
    write_json(output / "report.json", report)
    print(json.dumps({"file": str(output / "report.json"), "failures": len(rows),
                      "all_contact_positions_predicted_free": report["all_failed_path_contact_positions_predicted_free"],
                      "all_dense_pose_robot_samples_predicted_free": report["all_failed_path_dense_pose_robot_samples_predicted_free"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=HERE / "runs/planning_001")
    parser.add_argument("--evaluation", type=Path, default=HERE / "runs/evaluation_001")
    parser.add_argument("--output", type=Path, default=HERE / "runs/failure_diagnostics_001")
    args = parser.parse_args()
    run(args.plans, args.evaluation, args.output)
