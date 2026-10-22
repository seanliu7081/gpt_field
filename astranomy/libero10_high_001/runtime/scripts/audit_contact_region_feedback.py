"""Saved-RGB contact-region audit; no robot, simulator, model, or policy rollout.

Manual boxes are deliberately offline diagnostic fixtures. Their desired offset
preserves the starting image relation and is NOT a validated grasp relation.
Replaying an unchanged historical trajectory cannot measure correction benefit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import numpy as np

from gpt_field.observation import PolicyObservation

VERSION = "contact-region-offline-audit-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "runs/contact_routing_diagnostic_001"
DEFAULT_OUTPUT = ROOT / "runs/contact_region_preparation_001"
CHECKPOINTS = ("handle_a", "handle_b", "bowl_a", "bowl_b")
ARMS = ("current", "direct5")
WINDOW_ACTIONS = 30

# Chosen by inspection of only the checkpoint's RGB, before feedback evaluation.
# Boxes enclose visible material/edges. A hidden contact patch is never invented.
PIXEL_FIXTURES = {
    "handle_a": {
        "external": None,
        "wrist": {"tool": [394, 400, 442, 429], "target": [177, 365, 306, 390]},
        "notes": ["External top handle is occluded by the hand; external reference unavailable.",
                  "Wrist right finger edge and visible top handle surface; no hidden contact point."],
    },
    "handle_b": {
        "external": {"tool": [310, 231, 336, 251], "target": [350, 236, 375, 257]},
        "wrist": {"tool": [394, 400, 442, 429], "target": [94, 25, 333, 74]},
        "notes": ["External upper finger edge and upper handle; wrist topmost visible handle.",
                  "Large vertical image separation is preserved for tracking diagnostics, not a grasp instruction."],
    },
    "bowl_a": {
        "external": {"tool": [151, 275, 169, 296], "target": [143, 302, 163, 318]},
        "wrist": {"tool": [394, 400, 442, 429], "target": [397, 320, 446, 388]},
        "notes": ["Right finger edge and visible right bowl rim; contact surface behind finger is not annotated.",
                  "Small external patch may be rejected; keep that failure rather than enlarging after results."],
    },
    "bowl_b": {
        "external": {"tool": [111, 279, 127, 300], "target": [125, 300, 144, 320]},
        "wrist": {"tool": [356, 399, 402, 429], "target": [306, 347, 344, 389]},
        "notes": ["Right finger edge and visible patterned rim/body patch adjacent to it.",
                  "Same-body and correspondence assumptions are manual hypotheses, not geometric truth."],
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_window(path, *, maximum_actions=WINDOW_ACTIONS):
    """Whitelist observations; reject malformed ordering instead of joining scenes."""
    if type(maximum_actions) is not int or maximum_actions < 0:
        raise ValueError("maximum_actions must be a nonnegative integer")
    initial = None
    endpoints = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"malformed trace line {line_number}") from exc
            if row.get("event") == "reset":
                if initial is not None:
                    raise ValueError("multiple scenes/resets in one replay window")
                initial = PolicyObservation.from_dict(row["observation"])
                if maximum_actions == 0:
                    break
            elif row.get("event") == "executed":
                if initial is None or row.get("step") != len(endpoints):
                    raise ValueError("noncontiguous executed observations")
                observation = PolicyObservation.from_dict(row["observation"])
                if observation.prompt != initial.prompt:
                    raise ValueError("task changed within replay window")
                action = np.asarray(row["action"], dtype=float)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("executed action must have seven finite values")
                endpoints.append((observation, action.tolist()))
                if len(endpoints) == maximum_actions:
                    break
    if initial is None:
        raise ValueError("trace has no reset observation")
    return initial, endpoints


def reference_fixture(checkpoint_id):
    fixture = PIXEL_FIXTURES[checkpoint_id]
    reference = {"reference_id": f"offline-manual-{checkpoint_id}",
                 "target_id": "drawer_handle" if checkpoint_id.startswith("handle") else "bowl"}
    for view in ("external", "wrist"):
        boxes = fixture[view]
        if boxes is None:
            reference[view] = None
            continue
        tool = np.asarray(boxes["tool"], dtype=float) / 512
        target = np.asarray(boxes["target"], dtype=float) / 512
        offset = (tool[:2] + tool[2:] - target[:2] - target[2:]) / 2
        reference[view] = {"tool_region_uv": tool.tolist(), "target_region_uv": target.tolist(),
                           "desired_offset_uv": offset.tolist()}
    return reference


def observation_digest(observation):
    return {"observation_sha256": data_sha(observation.to_dict()),
            "external_data_url_sha256": hashlib.sha256(observation.image.encode()).hexdigest(),
            "wrist_data_url_sha256": hashlib.sha256(observation.wrist_image.encode()).hexdigest(),
            "state8_sha256": data_sha(list(observation.state))}


def freeze(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    """Freeze source bytes, observations, fixture semantics and fixed windows first."""
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = output / "offline_fixture_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("offline fixtures already frozen; preserve prior selection")
    selected_path = source / "preparation/selected.json"
    selected = json.loads(selected_path.read_text())
    available = {item["id"]: item for item in selected["checkpoints"]}
    if any(name not in available for name in CHECKPOINTS):
        raise ValueError("expected four diagnostic checkpoints")
    rows = []
    for checkpoint_id in CHECKPOINTS:
        for arm in ARMS:
            trace = source / "trials" / f"{checkpoint_id}__{arm}" / "trace.jsonl"
            initial, endpoints = load_window(trace)
            checkpoint = PolicyObservation.from_dict(json.loads(
                Path(available[checkpoint_id]["observation_path"]).read_text()))
            if initial.to_dict() != checkpoint.to_dict():
                raise ValueError("trace reset differs from frozen checkpoint")
            rows.append({"id": f"{checkpoint_id}__{arm}", "checkpoint_id": checkpoint_id,
                         "historical_arm": arm, "trace_path": str(trace), "trace_sha256": sha(trace),
                         "maximum_actions": WINDOW_ACTIONS, "actual_window_actions": len(endpoints),
                         "initial": observation_digest(initial),
                         "endpoint_digests": [observation_digest(obs) for obs, _ in endpoints],
                         "actions_sha256": data_sha([action for _, action in endpoints]),
                         "reference": reference_fixture(checkpoint_id),
                         "fixture_notes": PIXEL_FIXTURES[checkpoint_id]["notes"]})
    manifest = {"version": VERSION, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_path": str(selected_path), "selection_sha256": sha(selected_path),
                "annotation_origin": "manual_rgb_only_offline_diagnostic_fixture",
                "desired_relation": "preserve initial region-center image offset; NOT validated grasp target",
                "desired_offset_semantics": "tool_region_center_minus_target_region_center",
                "shadow_nominal_goal": "immutable checkpoint XYZ; never a future observation",
                "policy_input_eligible": False, "demonstration": False,
                "window_rule": "first 30 recorded executed actions after each frozen checkpoint; no reseeding",
                "fixture_selection_observed_feedback_results": False,
                "conditions": ["manual_reference_shadow", "null_reference_shadow"],
                "windows": rows}
    write_json(manifest_path, manifest)
    write_json(output / "offline_fixture_freeze.json", {"manifest_sha256": sha(manifest_path),
               "frozen_at_utc": manifest["frozen_at_utc"], "feedback_runs_before_freeze": 0})
    return manifest


def validated_windows(manifest_path):
    manifest_path = Path(manifest_path)
    frozen = json.loads((manifest_path.parent / "offline_fixture_freeze.json").read_text())
    if sha(manifest_path) != frozen["manifest_sha256"]:
        raise ValueError("frozen fixture manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("annotation_origin") != "manual_rgb_only_offline_diagnostic_fixture":
        raise ValueError("manual offline provenance required")
    if sha(manifest["selection_path"]) != manifest["selection_sha256"]:
        raise ValueError("source checkpoint selection changed")
    rows = []
    for item in manifest["windows"]:
        if sha(item["trace_path"]) != item["trace_sha256"]:
            raise ValueError("archived trace changed after freeze")
        initial, endpoints = load_window(item["trace_path"], maximum_actions=item["maximum_actions"])
        if (observation_digest(initial) != item["initial"] or
                [observation_digest(obs) for obs, _ in endpoints] != item["endpoint_digests"] or
                data_sha([action for _, action in endpoints]) != item["actions_sha256"]):
            raise ValueError("frozen replay observation/action mismatch")
        rows.append((item, initial, endpoints))
    return manifest, rows


def replay_window(initial, endpoints, reference, *, feedback_factory, target_id):
    """Only shadow proposals: never notify a correction as applied or alter RGB/state."""
    feedback = feedback_factory()
    begin = time.perf_counter()
    initialized = feedback.configure(initial, reference, gripper_open=True, target_id=target_id)
    feedback.begin_nominal_goal(initial.state[:3], intent_key="offline-fixed-nominal")
    initialization_s = time.perf_counter() - begin
    samples, latencies, proposals = [], [], 0
    nominal = tuple(initial.state[:3])
    for index, (observation, action) in enumerate(endpoints):
        begin = time.perf_counter()
        if action[6] >= 0:
            feedback.invalidate("offline_recorded_gripper_close_or_neutral")
            feedback.observe(observation)
            goal = None
            diagnostics = feedback.diagnostics()
        else:
            # Immutable causal hold target, never the next observation/future state.
            goal, diagnostics = feedback.corrected_goal(observation, nominal)
        elapsed = time.perf_counter() - begin
        latencies.append(elapsed)
        if goal is not None:
            proposals += 1
        samples.append({"endpoint": index + 1, "source_observation": observation_digest(observation),
                        "recorded_action": action, "shadow_nominal_goal": list(nominal),
                        "shadow_goal": None if goal is None else list(map(float, goal)),
                        "correction_applied": False, "elapsed_s": elapsed,
                        "diagnostics": diagnostics})
    statuses = Counter(str(row["diagnostics"].get("status", "unspecified")) for row in samples)
    active_latencies = [row["elapsed_s"] for row in samples if row["diagnostics"].get("tracking_valid")]
    per_view = {}
    for view in ("external", "wrist"):
        view_rows = [row["diagnostics"].get("views", {}).get(view, {}) for row in samples]
        per_view[view] = {
            "initialized": bool(initialized.get("views", {}).get(view, {}).get("tracking_valid")),
            "tracked_observations": sum(bool(row.get("tracking_valid")) for row in view_rows),
            "validated_observations": sum(bool(row.get("response_validated")) for row in view_rows),
            "maximum_validated_rank": max((int(row.get("validated_rank", 0)) for row in view_rows), default=0),
            "maximum_held_out_samples": max((len(row.get("held_out_sample_ids", [])) for row in view_rows), default=0),
            "response_mismatch_observed": any(row.get("response_rejected") for row in view_rows),
            "statuses": dict(Counter(str(row.get("status", "unavailable")) for row in view_rows)),
        }
    return {"initialization": initialized, "initialization_s": initialization_s,
            "samples": samples, "summary": {"observations": len(endpoints),
            "initialization_tracking_valid": bool(initialized.get("tracking_valid")),
            "tracked_observations": sum(bool(row["diagnostics"].get("tracking_valid")) for row in samples),
            "validated_observations": sum(bool(row["diagnostics"].get("response_validated")) for row in samples),
            "views": per_view,
            "shadow_proposals": proposals, "applied_corrections": 0,
            "statuses": dict(statuses), "total_feedback_s": initialization_s + sum(latencies),
            "median_feedback_s": float(np.median(latencies)) if latencies else None,
            "median_feedback_while_tracking_valid_s": float(np.median(active_latencies)) if active_latencies else None,
            "p95_feedback_s": float(np.percentile(latencies, 95)) if latencies else None,
            "reference_condition": "unavailable" if reference is None else "manual_offline_fixture"}}


def run(output=DEFAULT_OUTPUT, *, revision="region_feedback_initial"):
    # Import only the RGB/state component. No runtime client/simulator dependency.
    from gpt_field.contact_region_feedback import ContactRegionFeedback
    output = Path(output).resolve()
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision label must be nonempty")
    manifest, windows = validated_windows(output / "offline_fixture_manifest.json")
    source_paths = [ROOT / "src/gpt_field" / name for name in
                    ("contact_region_feedback.py", "operation_visual.py", "speed_first_pregrasp.py", "observation.py")]
    source_paths.append(Path(__file__).resolve())
    source_hashes_before = {str(path): sha(path) for path in source_paths}
    # Every repeat keeps earlier evidence, including incomplete source-change runs.
    previous = [output / name for name in ("offline_audit.json", "offline_replays") if (output / name).exists()]
    if previous:
        history = output / "offline_audit_history"
        index = 1
        while (history / f"pass_{index:03d}").exists():
            index += 1
        archived = history / f"pass_{index:03d}"
        archived.mkdir(parents=True)
        for path in previous:
            shutil.move(str(path), archived / path.name)
    results = []
    for item, initial, endpoints in windows:
        for condition in manifest["conditions"]:
            reference = item["reference"] if condition == "manual_reference_shadow" else None
            result = replay_window(initial, endpoints, reference,
                                   feedback_factory=ContactRegionFeedback,
                                   target_id=item["reference"]["target_id"])
            result.update(window_id=item["id"], condition=condition,
                          development_revision=revision,
                          annotation_origin=manifest["annotation_origin"],
                          source_trace_sha256=item["trace_sha256"])
            name = f"{item['id']}__{condition}.json"
            write_json(output / "offline_replays" / name, result)
            results.append({"window_id": item["id"], "condition": condition,
                            "artifact": str(output / "offline_replays" / name),
                            "initialization_s": result["initialization_s"],
                            "initialization": result["initialization"], **result["summary"]})
    validated_windows(output / "offline_fixture_manifest.json")
    source_hashes_after = {str(path): sha(path) for path in source_paths}
    summary = {"version": VERSION, "manifest_sha256": sha(output / "offline_fixture_manifest.json"),
               "development_revision": revision,
               "evaluation_context": "development_replay_not_held_out_grasp_performance",
               "core_source_sha256": sha(ROOT / "src/gpt_field/contact_region_feedback.py"),
               "dependency_sources_sha256": {name: sha(ROOT / "src/gpt_field" / name) for name in
                   ("operation_visual.py", "speed_first_pregrasp.py", "observation.py")},
               "audit_source_sha256": sha(Path(__file__)), "model_calls": 0, "simulator_calls": 0,
               "source_hashes_before": source_hashes_before, "source_hashes_after": source_hashes_after,
               "source_integrity_passed": source_hashes_before == source_hashes_after,
               "policy_rollouts": 0, "applied_corrections": 0,
               "physical_benefit_measured": False, "gpt_grounding_measured": False,
               "limitations": ["Manual boxes are never policy inputs, demonstrations, or evidence GPT can ground them.",
                   "The desired offset preserves a visible initial relation, not a certified 3D grasp pose.",
                   "Historical actions/RGB/state remain unchanged; proposal effects are counterfactual and unmeasured.",
                   "A shadow proposal or declining pixel error does not demonstrate improved contact or grasp.",
                   "Natural motion may not excite enough independent directions to validate a response model.",
                   "No simulator truth, object poses, depth or camera calibration enters feedback."],
               "condition_totals": {condition: {
                   "windows": sum(r["condition"] == condition for r in results),
                   "initialization_successes": sum(r["initialization_tracking_valid"] for r in results if r["condition"] == condition),
                   **{key: sum(r[key] for r in results if r["condition"] == condition) for key in
                      ("observations", "tracked_observations", "validated_observations", "shadow_proposals",
                       "applied_corrections", "total_feedback_s", "initialization_s")}}
                   for condition in manifest["conditions"]},
               "results": results}
    write_json(output / "offline_audit.json", summary)
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source changed during replay; results preserved but rerun unchanged frozen inputs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="region_feedback_initial")
    args = parser.parse_args()
    result = freeze(args.source, args.output) if args.stage == "freeze" else run(args.output, revision=args.revision)
    print(json.dumps({"stage": args.stage, "output": str(args.output),
                      "windows": len(result.get("windows", result.get("results", [])))}))


if __name__ == "__main__":
    main()
