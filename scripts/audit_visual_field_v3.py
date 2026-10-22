"""Read-only V2 record audit and conservative V3 preflight counterfactual.

No simulator, model, transport or action executor is constructed. Legacy masks
are never removed, clipped or granted new contact permissions. This diagnoses
recorded proposals; it does not estimate a counterfactual rollout or speedup.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VERSION = "visual-field-v3-legacy-preflight-audit-v1"
DEFAULT_RUN = ROOT / "runs/visual_field_v2_zero_shot_001/trials/bowl_original_zero_hybrid_v2"
VIEWS = ("external", "wrist")
OBS_KEYS = {"observation/image", "observation/wrist_image", "observation/state", "prompt",
            "observation/joint_position", "observation/joint_velocity"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_observation(obs):
    if set(obs) != OBS_KEYS:
        raise ValueError("Recorded observation includes missing or unpermitted input keys")
    for key, n in (("observation/state", 8), ("observation/joint_position", 7), ("observation/joint_velocity", 7)):
        if len(obs[key]) != n or not all(type(x) in (int, float) and math.isfinite(x) for x in obs[key]):
            raise ValueError("Malformed measured robot observation")
    if not isinstance(obs["prompt"], str):
        raise ValueError("Missing task language")


def observation_shapes(obs):
    validate_observation(obs)
    shapes = {}
    for view, key in (("external", "observation/image"), ("wrist", "observation/wrist_image")):
        url = obs[key]
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError("Only embedded recorded RGB observations are supported")
        data = base64.b64decode(url.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            shapes[view] = (image.height, image.width)
    return shapes


def legacy_reference_to_v3(reference):
    """Conservative conversion: no inferred task roles, mask deletion or clipping."""
    if reference is None:
        return None
    old = copy.deepcopy(reference)
    required = {"moving_id", "target_id", "moving_kind", "tool_point_uv", "moving_point_uv",
                "target_point_uv", "target_anchor_uv", "moving_radius_uv", "blocked_boxes_uv", "unknown_boxes_uv"}
    if set(old) != required:
        raise ValueError("Expected the exact legacy V2 visual-reference fields")
    radius = old.pop("moving_radius_uv")
    if type(radius) not in (int, float) or not math.isfinite(radius) or not 0 <= radius <= .3:
        raise ValueError("Invalid legacy moving radius")
    if radius == 0:
        raise ValueError("A zero legacy radius supplies no observed V3 footprint; do not invent extent")
    moving = old["moving_point_uv"]
    if len(moving) != 2 or not all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in moving):
        raise ValueError("Invalid legacy moving point")
    old["moving_footprint_uv"] = {"lower_uv": [x - radius for x in moving],
                                  "upper_uv": [x + radius for x in moving]}
    for group, field, value in (("blocked_boxes_uv", "role", "obstacle"), ("unknown_boxes_uv", "source", "occluded")):
        for region in old[group]:
            if set(region) != {"object_id", "lower_uv", "upper_uv"}:
                raise ValueError("Legacy region unexpectedly contains semantic permissions")
            region[field] = value
    return old


def legacy_intersections(reference, *, grid_size=40, padding=True):
    """Independent continuous-UV contradictions, using actual V2 grid padding.

    These are image-envelope conflicts, not evidence of a physical collision.
    A radius-only pass is also reported as a discretization-independent lower bound.
    """
    if reference is None:
        return []
    if type(grid_size) is not int or not 8 <= grid_size <= 96:
        raise ValueError("Unsupported grid size")
    margin = reference["moving_radius_uv"] + (1 / (grid_size - 1) if padding else 0.)
    issues = []
    for feature, key in (("start", "moving_point_uv"), ("goal", "target_point_uv")):
        point = reference[key]
        if any(x < margin or x > 1 - margin for x in point):
            issues.append({"code": "envelope_out_of_view", "feature_id": feature,
                           "point_uv": point, "margin_uv": margin})
        for group in ("blocked_boxes_uv", "unknown_boxes_uv"):
            for region in reference[group]:
                if all(lo - margin <= x <= hi + margin for x, lo, hi in zip(point, region["lower_uv"], region["upper_uv"])):
                    issues.append({"code": "envelope_intersects_excluded_region", "feature_id": feature,
                                   "category": group, "region_id": region["object_id"],
                                   "point_uv": point, "margin_uv": margin})
    return issues


def response_prediction(response):
    texts = [part["text"] for item in response.get("output", []) if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    if len(texts) != 1:
        raise ValueError("Accepted model response must contain one structured output")
    return json.loads(texts[0])


def request_records(run):
    records, files, statuses = {}, [], Counter()
    for path in sorted((run / "api").glob("*.request.json")):
        request = read_json(path)
        key = request["request_key"]
        if key in records:
            raise ValueError("Repeated request identity needs explicit review")
        payload = request["payload"]
        if (payload.get("model") != "gpt-6-astra" or payload.get("reasoning", {}).get("effort") != "medium"
                or request.get("billing") != "chatgpt_subscription"):
            raise ValueError("Recorded request does not match medium subscription protocol")
        result_path = path.with_name(path.name.replace(".request.json", ".result.json"))
        response_path = path.with_name(path.name.replace(".request.json", ".response.json"))
        result = read_json(result_path) if result_path.exists() else {}
        statuses[result.get("status", "no_terminal_result")] += 1
        if result.get("budget", {}).get("api_key_calls", 0) != 0:
            raise ValueError("An API-key call appears in the recorded budget")
        record = {"request_file": str(path), "request_sha256": sha256(path),
                  "request_ordinal": request["request_ordinal"], "result_status": result.get("status")}
        for file in (path, result_path, response_path):
            if file.exists():
                files.append({"path": str(file), "sha256": sha256(file)})
        if result.get("status") == "valid":
            response = read_json(response_path)
            if response.get("model") != "gpt-6-astra" or response.get("reasoning", {}).get("effort") != "medium":
                raise ValueError("Reported model/effort differs from the historical protocol")
            record["prediction"] = response_prediction(response)
        records[key] = record
    return records, files, dict(statuses)


def source_identity():
    paths = sorted((ROOT / "src/gpt_field").glob("*.py")) + [Path(__file__).resolve()]
    return {str(path): sha256(path) for path in paths}


def audit_run(run, output, *, preflight=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output == run or output.is_relative_to(run):
        raise FileExistsError("Choose a fresh audit directory outside the historical run")
    if preflight is None:
        from gpt_field.visual_field_v3 import preflight_reference
        preflight = preflight_reference
    source_before = source_identity()
    summary, manifest = read_json(run / "summary.json"), read_json(run / "manifest.json")
    if manifest.get("method_version") != "v2" or manifest.get("shots") != 0 or manifest.get("controller") != "hybrid":
        raise ValueError("This audit accepts only archived zero-shot Hybrid V2 records")
    requests, api_files, request_statuses = request_records(run)
    blocks, current_obs, pending = [], None, {}
    executed_steps, predictions_seen, controller_counts, selection_counts = set(), set(), Counter(), Counter()
    tracked_views, model_reliable = Counter(), Counter()
    last_status = None
    with (run / "trace.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("event")
            if kind == "reset":
                if current_obs is not None:
                    raise ValueError("Multiple resets are not supported in this single-trial audit")
                current_obs = event["observation"]
                validate_observation(current_obs)
                last_status = event.get("sim_status")
            elif kind == "action":
                step, diagnostics = event["step"], event["diagnostics"]
                if step in pending or step in executed_steps or current_obs is None:
                    raise ValueError("Action sequence is not a unique ordered decision sequence")
                pending[step] = event
                prediction = diagnostics.get("prediction")
                if prediction is None:
                    continue
                key = diagnostics["request_key"]
                if key in predictions_seen or key not in requests or requests[key].get("prediction") != prediction:
                    raise ValueError("Recorded prediction differs from its accepted raw model response")
                predictions_seen.add(key)
                shapes = observation_shapes(current_obs)
                views = {}
                for view in VIEWS:
                    legacy = prediction["visual_references"][view]
                    converted = legacy_reference_to_v3(legacy)
                    before_preflight = canonical_hash(converted)
                    result = preflight(converted, shapes[view], grid_size=40)
                    if canonical_hash(converted) != before_preflight:
                        raise ValueError("Preflight mutated the preserved reference or masks")
                    if not isinstance(result, dict) or not isinstance(result.get("blockers"), list):
                        raise ValueError("Unsupported V3 preflight result contract")
                    exact_issues = legacy_intersections(legacy, grid_size=40)
                    radius_issues = legacy_intersections(legacy, grid_size=40, padding=False)
                    # V3 deliberately removes V2's extra border-grid padding, but
                    # retains actual footprint coverage and inflated region masks.
                    must_block = any(x["code"] == "envelope_intersects_excluded_region" for x in exact_issues)
                    must_block |= any(x["code"] == "envelope_out_of_view" for x in radius_issues)
                    if must_block and not result["blockers"]:
                        raise ValueError("V3 preflight silently accepted a conservatively retained legacy conflict")
                    views[view] = {"legacy_reference_sha256": canonical_hash(legacy),
                                   "converted_reference": converted, "legacy_grid40_conflicts": exact_issues,
                                   "radius_only_conflicts": radius_issues, "v3_preflight": result,
                                   "retained_blocked_region_count": len(legacy["blocked_boxes_uv"]) if legacy else 0,
                                   "retained_unknown_region_count": len(legacy["unknown_boxes_uv"]) if legacy else 0}
                blocks.append({"step": step, "chunk_index": diagnostics.get("chunk_index"), "request_key": key,
                               "request_ordinal": requests[key]["request_ordinal"], "controller": prediction["controller"],
                               "phase": prediction["phase"], "prediction_sha256": canonical_hash(prediction),
                               "observation_sha256": canonical_hash(current_obs), "views": views,
                               "v3_has_pre_execution_blocker": any(v["v3_preflight"]["blockers"] for v in views.values()),
                               "recorded_selection": diagnostics.get("selection"),
                               "counterfactual_action_executed": False})
            elif kind == "executed":
                step = event["step"]
                if step not in pending or step in executed_steps or step != len(executed_steps):
                    raise ValueError("Executed action sequence is incomplete or duplicated")
                decision = pending.pop(step)
                if event["action"] != decision["action"]:
                    raise ValueError("Recorded executed command differs from its decision")
                executed_steps.add(step)
                diagnostics = decision["diagnostics"]
                controller_counts[diagnostics["controller"]] += 1
                if diagnostics.get("selection"):
                    selection_counts[diagnostics["selection"]["status"]] += 1
                current_obs = event["observation"]
                validate_observation(current_obs)
                last_status = event.get("sim_status")
            elif kind == "feedback":
                visual = event.get("feedback", {}).get("visual", {})
                for name, view in visual.get("views", {}).items():
                    tracked_views[name] += int(bool(view.get("reliable")))
                for name in ("action_to_joint", "action_to_eef", "joint_to_image", "eef_to_image"):
                    model_reliable[name] += int(bool(visual.get(name, {}).get("reliable")))
    if (len(executed_steps) != summary["steps"] or controller_counts["visual_field"] != summary["field_actions"]
            or controller_counts["direct"] != summary["direct_actions"]):
        raise ValueError("Summary action accounting differs from the actual trace")
    if len(predictions_seen) != sum(record.get("prediction") is not None for record in requests.values()):
        raise ValueError("Accepted responses without a recorded decision need explicit review")
    if summary.get("task_success") and (summary.get("status") != "success" or not (last_status or {}).get("success")
                                        or not (last_status or {}).get("libero_success")
                                        or (last_status or {}).get("collision", {}).get("detected")):
        raise ValueError("Recorded summary claims completion without a qualified final LIBERO signal")
    observed_time = summary["control_elapsed_s"] - summary["timing_components"]["video_render_s"]
    if summary.get("control_without_video_s") is not None and not math.isclose(observed_time, summary["control_without_video_s"], abs_tol=1e-6):
        raise ValueError("Historical timing components disagree")
    sources = [{"path": str(run / name), "sha256": sha256(run / name)} for name in ("manifest.json", "summary.json", "trace.jsonl")]
    if source_identity() != source_before:
        raise ValueError("Current audit/core code changed while auditing; repeat after source freeze")
    report = {
        "schema_version": VERSION, "audit_status": "pass", "audit_kind": "offline_record_and_preflight_counterfactual",
        "source_run": str(run), "sources": sources, "api_sources": api_files, "audit_source_sha256": source_before,
        "observed": {"status": summary["status"], "task_success": summary.get("task_success"),
                     "libero_success": summary.get("libero_success"), "last_recorded_simulator_success": (last_status or {}).get("success"),
                     "executed_actions": len(executed_steps), "controller_counts": dict(controller_counts),
                     "field_selection_status_counts": dict(selection_counts), "local_servo_actions": summary.get("local_servo_actions"),
                     "requests_started": len(requests), "accepted_prediction_blocks": len(blocks),
                     "request_terminal_status_counts": request_statuses, "pending_decisions_without_execution": sorted(pending),
                     "control_without_video_s": observed_time,
                     "time_kind": "completion" if summary.get("task_success") else "attempt",
                     "tracked_view_feedback_counts": dict(tracked_views), "model_reliable_feedback_counts": dict(model_reliable)},
        "counterfactual": {"legacy_controller_grid_size": 40, "padding_uv": 1 / 39,
                           "blocks_with_legacy_grid40_conflicts": sum(any(v["legacy_grid40_conflicts"] for v in b["views"].values()) for b in blocks),
                           "blocks_with_radius_only_conflicts": sum(any(v["radius_only_conflicts"] for v in b["views"].values()) for b in blocks),
                           "blocks_with_v3_pre_execution_blockers": sum(b["v3_has_pre_execution_blocker"] for b in blocks),
                           "conversion": {"footprint": "Unclipped square centered on the recorded moving point, +/- legacy radius; an approximation, not newly observed extent.",
                                          "blocked_regions": "All coordinates and IDs retained with role=obstacle; no new contact-target permissions.",
                                          "unknown_regions": "All coordinates and IDs retained with source=occluded as an explicit conservative legacy label; unknown is never treated as free.",
                                          "semantic_limit": "V2 did not record typed roles/visibility causes; names are not used to invent them."},
                           "new_model_calls": 0, "new_simulator_steps": 0, "counterfactual_actions_executed": 0,
                           "rescued_success_claim": False, "estimated_speedup": None,
                           "per_view_model_persistence_replayed": False,
                           "local_recovery_execution_replayed": False,
                           "references_missing": sum(v["converted_reference"] is None for b in blocks for v in b["views"].values()),
                           "border_rule_difference": "V2 also pads image borders by one grid cell; V3 requires actual footprint coverage and applies grid padding only around regions. Border-only padding differences are reported, not treated as dropped masks."},
        "blocks": blocks,
        "limitations": ["This checks whether V3 identifies contradictions in unchanged historical proposals; a newly prompted V3 model may produce different proposals.",
                        "Direct actions historically executed despite these field-reference conflicts. Diagnosing them does not establish the outcome of a gated or corrected rollout.",
                        "Image-space overlap is not physical collision or 3D clearance evidence.",
                        "Available historical model/tracking evidence is reported separately; this preflight-only audit does not prove new model persistence or recovery efficacy.",
                        "Recorded evaluator status is evidence from the archived experiment, not independently replayed physical state."]}
    output.mkdir(parents=True)
    with (output / "audit.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run, args.output)
    print(json.dumps({"audit_status": report["audit_status"], "recorded_outcome": report["observed"]["status"],
                      "recorded_blocks": report["observed"]["accepted_prediction_blocks"],
                      "v3_precheck_blocked": report["counterfactual"]["blocks_with_v3_pre_execution_blockers"],
                      "new_model_calls": 0, "new_simulator_steps": 0}))


if __name__ == "__main__":
    main()
