"""Offline V4 feedback/field audit. No live model, simulator or task result.

The archived replay checks visual evidence only: V3 image masks have no justified
metric lift and are NOT converted to metric free space. The separate synthetic
fixture checks call amortization for identical measured transitions, not robot
completion time. Its intent/geometry is authored test data, not a GPT prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.visual_field_policy import images
from gpt_field.visual_field_policy_v4 import VisualFieldPolicyV4
from gpt_field.visual_field_subscription_v4 import VisualFieldSubscriptionClientV4
from gpt_field.visual_monitor_v4 import RoleAwareVisualMonitor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/visual_field_v3_zero_shot_001/trials/bowl_original_zero_hybrid_v3"
VERSION = "visual-field-v4-offline-evidence-audit-v1"
EXTERNAL_KEYS = {"moving_id", "target_id", "moving_kind", "moving_point_uv",
                 "target_point_uv", "target_anchor_uv", "tool_point_uv"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_identity():
    files = sorted((ROOT / "src/gpt_field").glob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def role_references(legacy):
    """Keep archived feature identities, without treating image masks as 3D."""
    external, wrist = legacy["external"], legacy["wrist"]
    return {
        "external": None if external is None else {key: copy.deepcopy(external[key]) for key in EXTERNAL_KEYS},
        "wrist": None if wrist is None else {
            "target_id": wrist["target_id"], "target_anchor_uv": copy.deepcopy(wrist["target_anchor_uv"]),
            "desired_target_uv": copy.deepcopy(wrist["target_point_uv"]),
        },
    }


def replay_visual_evidence(trace_path):
    """Feed only policy-allowed observations, references and executed actions.

    Evaluator status, contact identities and hidden simulator state are ignored.
    An eligible visual gate is not an accepted metric path or an executed action.
    """
    monitor = RoleAwareVisualMonitor()
    current = pending = None
    rows, blocks = [], []
    started = time.perf_counter()
    with Path(trace_path).open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "reset":
                if current is not None:
                    raise ValueError("Only one archived episode is supported")
                current = PolicyObservation.from_dict(event["observation"])
            elif event["event"] == "action":
                if current is None or pending is not None:
                    raise ValueError("Action sequence is not acknowledged in order")
                if event["step"] != len(rows):
                    raise ValueError("Archived actions must be contiguous from zero")
                pending = {"step": event["step"], "action": event["action"]}
                prediction = event["diagnostics"].get("prediction")
                if prediction is not None:
                    refs = role_references(prediction["visual_references"])
                    monitor.configure(images(current), refs, current.state,
                                      intent_id=len(blocks) + 1)
                    legacy = prediction["visual_references"]
                    blocks.append({"first_step": event["step"], "source_controller": prediction["controller"],
                        "wrist_reference_present": refs["wrist"] is not None,
                        "external_moving_id": None if refs["external"] is None else refs["external"]["moving_id"],
                        "retained_external_point_uv": None if refs["external"] is None else refs["external"]["moving_point_uv"],
                        "legacy_image_regions_not_metric_geometry": {
                            view: {group: len((legacy[view] or {}).get(group, []))
                                   for group in ("blocked_boxes_uv", "unknown_boxes_uv")}
                            for view in ("external", "wrist")}})
                cap, reason = monitor.control_limit()
                pending.update(before_visual_gate_reason=reason, before_visual_command_cap_m=cap)
            elif event["event"] == "executed":
                if pending is None or pending["step"] != event["step"] or pending["action"] != event["action"]:
                    raise ValueError("Archived execution does not match its decision")
                following = PolicyObservation.from_dict(event["observation"])
                observed = monitor.observe(images(following), following.state, event["action"])
                cap, reason = monitor.control_limit()
                rows.append({**pending, "measured_translation_m": float(np.linalg.norm(
                        np.subtract(following.state[:3], current.state[:3]))),
                    "after_visual_gate_reason": reason, "after_visual_command_cap_m": cap,
                    "monitor": observed})
                current, pending = following, None
    if pending is not None:
        raise ValueError("Audit requires a fully acknowledged archived trace")
    if not rows:
        raise ValueError("No archived measured transitions")

    def window(low, high):
        selected = [row for row in rows if low <= row["step"] <= high]
        return {"first_step": low, "last_step": high, "transitions": len(selected),
            "measured_path_m": sum(row["measured_translation_m"] for row in selected),
            "statuses": dict(Counter(row["monitor"]["external_status"] for row in selected)),
            "visual_gate_eligible_before_actions": sum(row["before_visual_gate_reason"] is None for row in selected),
            "validated_response_transitions": sum(int(row["monitor"]["response_validated"]) for row in selected),
            "missing_wrist_but_visual_gate_eligible": sum(not row["monitor"]["wrist_target_reliable"]
                and row["after_visual_gate_reason"] is None for row in selected)}

    return {"kind": "archived_v3_observation_replay_only", "executed_v4_actions": 0,
        "metric_field_geometry_supplied": False, "metric_collision_clearance_established": False,
        "task_success_established": False,
        "limitations": ["Archived trajectories were executed by V3 Direct, not V4.",
            "Visual gate eligibility does not establish a feasible metric route or collision clearance.",
            "The desired wrist bearing is copied from the archived virtual target point for diagnostics only.",
            "V3 image masks have no justified 3D conversion; this replay constructs no metric intent."],
        "local_replay_wall_s": time.perf_counter()-started,
        "transitions": len(rows), "reference_blocks": blocks,
        "windows": {"initial_reference_0_34": window(0, 34), "corrected_reference_45_69": window(45, 69)},
        "status_counts": dict(Counter(row["monitor"]["external_status"] for row in rows)),
        "visual_gate_reasons": dict(Counter(row["after_visual_gate_reason"] or "eligible" for row in rows)),
        "rows": rows}


class FixtureTransport:
    """No network. Exact fixture outputs through the real subscription contract."""
    def __init__(self, predictions):
        self.predictions = copy.deepcopy(predictions)
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(copy.deepcopy(payload))
        if not self.predictions:
            raise AssertionError("Unexpected additional model request in synthetic fixture")
        return {"id": f"offline_v4_{len(self.calls)}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"}, "usage": {"input_tokens": 0, "output_tokens": 0},
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text",
                "text": json.dumps(self.predictions.pop(0))}]}]}


def fixture_observation(state):
    """Controlled robot feature moves with measured x; target/background stay fixed."""
    pixels = np.full((128, 128, 3), 100, dtype=np.uint8)
    x = 30 + int(round(state[0]*800))
    moving = np.random.default_rng(410).integers(0, 256, (11, 11, 3), dtype=np.uint8)
    target = np.random.default_rng(411).integers(0, 256, (11, 11, 3), dtype=np.uint8)
    pixels[45:56, x-5:x+6] = moving
    pixels[75:86, 100:111] = target
    image = image_to_data_url(pixels, size=128, rotate=False)
    wrist = image_to_data_url(np.full_like(pixels, 90), size=128, rotate=False)
    return PolicyObservation(image, wrist, tuple(state), resolve_distance_task("bowl_plate").prompt,
                             (0., -.5, 0., -1., 0., 1., 0.), (0.,)*7)


def fixture_prediction(state, *, actions=None):
    moving = [(30+int(round(state[0]*800)))/127, 50/127]
    return {"controller": "visual_field" if actions is None else "direct", "phase": "approach",
        "target_id": "fixture_target", "hypothesis_id": "offline_fixture_1",
        "reason": "Authored synthetic response; no actual GPT inference or obstacle measurement.",
        "visual_references": {"external": {"moving_id": "fixture_tool", "moving_kind": "tool",
            "moving_point_uv": moving, "target_id": "fixture_target", "target_anchor_uv": [105/127, 80/127],
            "target_point_uv": [86/127, 50/127], "tool_point_uv": None}, "wrist": None},
        "field_intent": None if actions is not None else {"goal_offset_m": [.07, 0., 0.],
            "orientation_target_xyzw": [0., 0., 0., 1.], "gripper_command": "hold", "goal_tolerance_m": .003,
            "max_actions": 30, "max_travel_m": .12, "command_limit_m": .01, "tool_radius_m": .005,
            "uncertainty_m": .001, "obstacles": [], "unknown_regions": []},
        "direct_actions": copy.deepcopy(actions)}


def synthetic_call_amortization(steps=20):
    """Same measured action trajectory, with mocked Field1 versus Direct5 outputs."""
    if type(steps) is not int or not 10 <= steps <= 25 or steps % 5:
        raise ValueError("Use 10, 15, 20 or 25 actions, divisible into Direct5 chunks")
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    transport = FixtureTransport([fixture_prediction(state)])
    client = VisualFieldSubscriptionClientV4(token_provider=lambda: "offline-not-a-credential", transport=transport)
    policy = VisualFieldPolicyV4(client)
    before = fixture_observation(state)
    history, observations, decisions = [], [before], []
    started = time.perf_counter()
    for index in range(steps):
        decision = policy.act(before)
        selected = decision.diagnostics.get("selection") or {}
        if not selected.get("accepted") or np.linalg.norm(decision.action[:3]) <= 0:
            raise AssertionError(f"Synthetic Field did not execute at action {index}: {selected.get('status')}")
        state = np.asarray(before.state).copy()
        state[:3] += np.asarray(decision.action[:3])*.05*.25
        after = fixture_observation(state)
        history.append(policy.observe_transition(before, decision, after))
        observations.append(after); decisions.append(decision)
        before = after
    policy.finish()
    hybrid_wall = time.perf_counter()-started

    direct_predictions = [fixture_prediction(observations[index].state,
        actions=[list(value.action) for value in decisions[index:index+5]]) for index in range(0, steps, 5)]
    direct_transport = FixtureTransport(direct_predictions)
    direct_client = VisualFieldSubscriptionClientV4(token_provider=lambda: "offline-not-a-credential", transport=direct_transport)
    direct = VisualFieldPolicyV4(direct_client)
    started = time.perf_counter()
    for index in range(steps):
        decision = direct.act(observations[index])
        if decision.action != decisions[index].action:
            raise AssertionError("Direct fixture must execute precisely the matched trajectory")
        direct.observe_transition(observations[index], decision, observations[index+1])
    direct.finish()
    direct_wall = time.perf_counter()-started
    measured_path = sum(value["measured_translation_m"] for value in history)
    return {"kind": "authored_synthetic_rgb_and_measured_state_fixture", "real_model_calls": 0,
        "real_robot_or_simulator_steps": 0, "actual_task_success": None, "live_speedup_measured": False,
        "fixture": {"robot_gain": .25, "image_motion_pixels_per_meter": 800, "missing_wrist_reference": True,
            "intent_source": "authored fixture, not a V4 GPT prediction", "hypothesized_obstacles": [],
            "metric_scene_claim": "synthetic known-empty example only", "measured_distance_m": measured_path,
            "matching": "Exactly identical commands, RGB and measured transitions in both arms"},
        "hybrid": {"moving_actions": steps, "fake_model_requests": len(transport.calls),
            "field": policy.local.summary(), "monitor": policy.monitor.summary(),
            "diagnostic_counters": policy.diagnostic_counters, "local_policy_processing_s": policy.local_total_s,
            "offline_fixture_wall_s": hybrid_wall},
        "direct5": {"moving_actions": steps, "fake_model_requests": len(direct_transport.calls),
            "measured_distance_m": measured_path, "local_policy_processing_s": direct.local_total_s,
            "offline_fixture_wall_s": direct_wall},
        "request_reduction_fraction": 1-len(transport.calls)/len(direct_transport.calls),
        "limitations": ["No inference latency is simulated or estimated.",
            "Local timing includes Python tracking/planning; synthetic observation construction is in fixture wall time.",
            "Direct commands are authored from the matched Field trajectory, not GPT-generated baseline performance.",
            "This verifies request amortization, not success rate, fewer recoveries or completed-task speedup."]}


def audit_run(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output == run or output.is_relative_to(run):
        raise FileExistsError("Choose a fresh output directory outside the immutable historical run")
    trace = run / "trace.jsonl"
    source_before, trace_before = source_identity(), sha256(trace)
    replay = replay_visual_evidence(trace)
    synthetic = synthetic_call_amortization()
    if source_before != source_identity() or trace_before != sha256(trace):
        raise RuntimeError("Inputs or source changed during the audit; rerun on a stable revision")
    report = {"version": VERSION, "source_sha256": source_before,
        "input": {"trace": str(trace), "trace_sha256": trace_before},
        "archived_replay": replay, "synthetic_call_amortization": synthetic,
        "live_trials_started": 0, "api_key_calls": 0, "subscription_inference_calls": 0}
    output.mkdir(parents=True)
    path = output / "audit.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (output / "artifact.json").write_text(json.dumps({"audit_sha256": sha256(path),
        "input_trace_sha256": trace_before, "source_sha256": source_before}, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run, args.output)
    print(json.dumps({"artifact": str(args.output / "audit.json"),
        "replay_windows": result["archived_replay"]["windows"],
        "synthetic": result["synthetic_call_amortization"]}, indent=2))
