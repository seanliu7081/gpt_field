"""Distance-only correction for legacy MuJoCo MPR query inaccuracies.

Reuses frozen planning_001 and the unchanged original collision/contact engine.
Only mj_geomDistance calls temporarily enable the installed native GJK backend.
The original evaluation_001 is preserved; corrected results use a NEW folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

import robot_geometry
import run_planning

OriginalGroundTruthEvaluator = robot_geometry.GroundTruthEvaluator
BACKEND = {
    "name": "MuJoCo native GJK geomDistance, scoped nativeccd enable flag",
    "engine_version": mujoco.__version__,
    "enable_flag": int(mujoco.mjtEnableBit.mjENBL_NATIVECCD),
    "scope": "distance queries only; restore original enableflags in finally",
    "contact_classification": "unchanged original MuJoCo collision/contact pipeline",
    "planner_and_robot_geometry": "unchanged frozen planning_001 sources and native geoms",
    "reason": "legacy MPR returned negative distances for independently separated native convex geometries",
    "positive_distance_cap_m": .05,
    "beyond_cap": "retain declared cap, including nativeCCD 1e10 out-of-range sentinel",
}


class NativeDistanceGroundTruthEvaluator(OriginalGroundTruthEvaluator):
    def _minimum_distance(self, pairs):
        original_flags = int(self.model.opt.enableflags)
        self.model.opt.enableflags = original_flags | int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
        try:
            return super()._minimum_distance(pairs)
        finally:
            self.model.opt.enableflags = original_flags

    def check(self, q, penetration_tolerance_m=1e-7, distances=True):
        result = super().check(q, penetration_tolerance_m, distances)
        if distances:
            result["distance_semantics"] = "native eligible geom pairs using scoped native GJK; positive distances capped; original contacts unchanged"
            result["distance_backend"] = BACKEND["name"]
        return result

    def provenance(self):
        result = super().provenance()
        result["distance_backend"] = dict(BACKEND)
        source = Path(__file__).resolve()
        result["consumed_sha256"][str(source)] = run_planning.sha256(source)
        return result


def evaluate(plans, output, references):
    plans, output = Path(plans).resolve(), Path(output).resolve()
    original_manifest = json.loads((plans / "manifest.json").read_text())
    if not run_planning.verify_sources(original_manifest["sources"]):
        raise RuntimeError("original planning runtime changed")
    original_manifest_hash = run_planning.sha256(plans / "manifest.json")
    original_class = robot_geometry.GroundTruthEvaluator
    robot_geometry.GroundTruthEvaluator = NativeDistanceGroundTruthEvaluator
    try:
        manifest = run_planning.evaluate(plans, output, references)
    finally:
        robot_geometry.GroundTruthEvaluator = original_class
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["settings"]["truth_distance_backend"] = dict(BACKEND)
    metrics["settings"]["reuses_unchanged_predicted_plans"] = True
    contradiction = [
        {"snapshot": row["snapshot"], "case_id": row["case_id"], "seed": row["seed"],
         "distance_m": row["truth"]["environment_min_distance_m"]}
        for row in metrics["records"] if row["truth"] is not None and row["truth"]["valid_at_all_samples"]
        and row["truth"]["environment_min_distance_m"] < -1e-7
    ]
    if contradiction:
        run_planning.write_json(output / "unresolved_distance_contact_conflict.json", contradiction)
        manifest["status"] = "distance_contact_conflict"
        run_planning.write_json(output / "manifest.json", manifest)
        raise RuntimeError("native distance and original contact classification still conflict")
    run_planning.write_json(metrics_path, metrics)
    run_planning.write_json(output / "distance_backend.json", BACKEND)
    manifest["distance_backend"] = dict(BACKEND)
    manifest["distance_contact_contradictions"] = contradiction
    manifest["original_plan_manifest_sha256"] = original_manifest_hash
    manifest["original_planning_sources_unchanged"] = run_planning.verify_sources(original_manifest["sources"])
    manifest["artifacts_sha256"] = run_planning.artifact_hashes(output)
    manifest["source_unchanged"] = run_planning.verify_sources(manifest["sources"])
    manifest["inputs_unchanged"] = all(run_planning.sha256(p) == digest for p, digest in manifest["consumed_sha256"].items())
    if not all(manifest[key] for key in ("original_planning_sources_unchanged", "source_unchanged", "inputs_unchanged")):
        raise RuntimeError("source/input changed during distance-only correction")
    run_planning.write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=run_planning.PROJECT / "scene_test/rgb_scene/references_004")
    args = parser.parse_args()
    evaluate(args.plans, args.output, args.references)
