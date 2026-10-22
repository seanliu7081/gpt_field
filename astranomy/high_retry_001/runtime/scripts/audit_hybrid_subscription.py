#!/usr/bin/env python3
"""Independent, offline audit of recorded subscription hybrid robot rollouts.

Reconstructs prompts/history and validates actions using the frozen field
planner/adapter or exact direct action slices. It never authenticates, requests
inference, imports a simulator worker, or replays simulator actions. Opaque model
output is compared as data and is never printed or included in audit reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.action import ActionAdapter
from gpt_field.astra import AstraClient, ENDPOINT, _canonical, output_schema, validate_prediction
from gpt_field.block_feedback_history import BLOCK_FEEDBACK_PROTOCOL, BLOCK_SUMMARY_TYPE
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.compact_field import COMPACT_FIELD_VERSION, expand_compact_field
from gpt_field.compact_subscription import compact_instructions, compact_output_schema, compact_prompt_version
from gpt_field.direct_history import HISTORY_PROTOCOL, TASK
from gpt_field.experiment import digest
from gpt_field.field import GridField
from gpt_field.field_grounding import (
    FIELD_EVENT_REASONS, GROUNDED_FIELD_VERSION, VisualGoalConsistencyChecker, effective_compact_field,
)
from gpt_field.hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_GRID,
    hybrid_output_schema, hybrid_prompt_version,
)
from gpt_field.observation import PolicyObservation
from gpt_field.planner import LocalPlanner, PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec
from gpt_field.tracking import PathTracker

_spec = importlib.util.spec_from_file_location("_close_audit_helpers", ROOT / "scripts/audit_subscription_close.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
read, rows, sha, near = _base.read, _base.rows, _base.sha, _base.near


class Audit(_base.Audit):
    def __init__(self):
        super().__init__()
        self.notes = []


def _wrapper(value, grid, audit, prefix, chunk_size):
    """Independent routing/action checks and the shared field validator."""
    expected = {"controller", "target_region", "near_target", "routing_reason", "field", "direct"}
    audit.check(isinstance(value, dict) and set(value) == expected, prefix + "exact hybrid wrapper keys")
    if not isinstance(value, dict) or set(value) != expected:
        return False
    contact = value["controller"] == "contact_direct"
    audit.check(value["controller"] in ("coarse_field", "contact_direct"), prefix + "known selected controller")
    audit.check(value["target_region"] in ("black_bowl", "plate"), prefix + "active target region allowlist")
    audit.check(type(value["near_target"]) is bool and value["near_target"] is contact,
                prefix + "near-target label agrees with selected branch")
    reason = value["routing_reason"]
    audit.check(isinstance(reason, str) and bool(reason.strip()) and len(reason) <= 500,
                prefix + "short nonempty routing explanation")
    active, unused = ("direct", "field") if contact else ("field", "direct")
    audit.check(value[unused] is None, prefix + "unused branch is null")
    try:
        if contact:
            direct = value[active]
            if not isinstance(direct, dict) or set(direct) != {"actions"}:
                raise ValueError("Direct branch must contain only actions")
            actions = direct["actions"]
            if not isinstance(actions, list) or len(actions) != chunk_size:
                raise ValueError("Direct chunk length differs from the recorded configuration")
            for action in actions:
                if (not isinstance(action, list) or len(action) != 7
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or abs(v) > 1 for v in action)):
                    raise ValueError("Direct action must contain seven finite normalized numbers")
        else:
            validate_prediction(value[active], grid, active)
        valid = True
    except Exception:
        valid = False
    audit.check(valid, prefix + "selected branch matches the recorded field/action contract")
    return valid


def _completion_contract(raw, audit, prefix, controller_mode="hybrid"):
    """Check the opt-in terminal signal independently of the online validator."""
    expected = ({"actions"} if controller_mode == "direct" else
                {"controller", "target_region", "near_target", "routing_reason", "field", "direct"})
    expected |= {"task_complete", "completion_reason"}
    valid = isinstance(raw, dict) and set(raw) == expected
    audit.check(valid, prefix + "exact completion-aware compact wrapper keys")
    if not valid:
        return False
    complete, reason = raw["task_complete"], raw["completion_reason"]
    audit.check(type(complete) is bool, prefix + "model completion flag is a boolean")
    audit.check(isinstance(reason, str) and len(reason) <= 500
                and (complete is not True or bool(reason.strip())),
                prefix + "model completion reason is bounded and nonempty when complete")
    if complete is True:
        if controller_mode == "direct":
            audit.check(raw.get("actions") == [[0, 0, 0, 0, 0, 0, -1]] * 5,
                        prefix + "forced Direct completion payload contains released stationary commands")
            return True
        field, direct = raw.get("field"), raw.get("direct")
        released_hold = (
            raw.get("controller") == "coarse_field" and isinstance(field, dict)
            and field.get("motion_mode") == "hold" and field.get("gripper_command") == "open"
        ) or (
            raw.get("controller") == "contact_direct" and isinstance(direct, dict)
            and direct.get("actions") == [[0, 0, 0, 0, 0, 0, -1]] * 5
        )
        audit.check(raw.get("target_region") == "plate" and released_hold,
                    prefix + "completion payload declares placement with released stationary commands")
    return complete is True


def _source_audit(config, audit, frozen=None):
    """Verify immutable run snapshots without claiming current code was run."""
    archive = frozen if frozen is not None and frozen.is_dir() else None
    # These modules implement the offline reconstruction itself. A historical
    # simulator/runner snapshot can differ from current code without affecting
    # reconstruction, but the algorithms and prompts used by this audit cannot.
    reconstruction_modules = {"action.py", "astra.py", "block_feedback_history.py", "chatgpt_plan.py", "compact_field.py",
        "compact_subscription.py", "direct_history.py", "experiment.py", "field.py",
        "hybrid_subscription.py", "observation.py", "planner.py", "schema.py", "tracking.py",
        "field_grounding.py"}
    count, frozen_count, changed = 0, 0, []
    for filename, expected_hash in config.get("software", {}).get("source_sha256", {}).items():
        current = ROOT / "src/gpt_field" / filename
        source = archive / filename if archive is not None else current
        audit.check(source.is_file() and sha(source) == expected_hash,
                    ("archived source matches run manifest: " if archive else "recorded source unchanged: ") + filename)
        count += 1
        if archive is not None:
            frozen_count += 1
            unchanged = current.is_file() and sha(current) == expected_hash
            if not unchanged:
                changed.append("src/gpt_field/" + filename)
            if filename in reconstruction_modules:
                audit.check(unchanged, "current audit reconstruction source matches recorded version: " + filename)
    runner = config.get("runner", {})
    if runner.get("path"):
        current = ROOT / runner["path"]
        source = archive / Path(runner["path"]).name if archive is not None else current
        audit.check(source.is_file() and sha(source) == runner.get("sha256"),
                    "archived runner matches run manifest" if archive else "recorded runner source unchanged")
        if archive is not None:
            frozen_count += 1
            if not current.is_file() or sha(current) != runner.get("sha256"):
                changed.append(runner["path"])
    if frozen is not None and archive is None:
        for filename, expected_hash in read(frozen).items():
            source = ROOT / filename
            audit.check(source.is_file() and sha(source) == expected_hash, "frozen source unchanged: " + filename)
            frozen_count += 1
    if archive is not None:
        audit.notes.append("Recorded source provenance was checked against the explicit frozen source archive. "
                           "Current audit reconstruction modules also match that recorded version.")
        if changed:
            audit.notes.append("Current workspace files differ from the recorded archive: " + ", ".join(changed))
    return count, frozen_count, changed


def _stop_override_audit(run, summary, executed, completion, audit):
    """Keep a later user stop override separate from the immutable run result."""
    path = run / "stop_rule_override.json"
    if not path.is_file():
        return None
    record = read(path)
    prefix = "live stopping override: "
    audit.check(record.get("user_requested_stop") == "first of LIBERO success or Astra task_complete=true"
                and record.get("new_model_calls") == 0,
                prefix + "journal declares the user-selected OR rule without new model calls")
    events_path = run / (summary.get("video") or "episode.mp4")
    events_path = events_path.with_suffix(".events.jsonl")
    events = rows(events_path) if events_path.is_file() else []
    triggers = [event for event in events if event.get("event") == "model_completion"
                or (event.get("event") in ("observation", "executed") and event.get("success") is True)]
    trigger = triggers[0] if triggers else None
    status = record.get("status")
    if status in ("interrupt_sent_on_libero_success", "astra_completed_natively"):
        audit.check(trigger is not None, prefix + "claimed completion has a recorded video event")
        if trigger is not None:
            audit.check(record.get("completion_step") == trigger.get("step")
                        and near(record.get("completion_event_wall_s"), trigger.get("wall_elapsed_s")),
                        prefix + "journal uses the first recorded success or model-completion event")
            model_first = trigger.get("event") == "model_completion"
            audit.check((status == "astra_completed_natively") is model_first,
                        prefix + "journal identifies the actual first completion condition")
            expected_steps = trigger["step"] + int(trigger.get("event") == "executed")
            audit.check(len(executed) == expected_steps,
                        prefix + "no acknowledged robot action after first completion condition")
        if status == "interrupt_sent_on_libero_success":
            audit.check(summary.get("status") == "user_stop" and record.get("signal") == "SIGINT"
                        and record.get("acknowledged_actions") == len(executed),
                        prefix + "external success stop preserves the original user-stop summary")
            audit.notes.append("A later user override stopped this already-running model-only episode on recorded LIBERO success. "
                               "The original user_stop status and scoring are preserved; the override is reported separately.")
        else:
            audit.check(summary.get("status") == "model_declared_complete" and completion is not None,
                        prefix + "native model completion remains the original terminal outcome")
    else:
        audit.check(status == "runner_exited_without_monitor_interrupt" and trigger is None,
                    prefix + "monitor ended without a completion condition rather than claiming success")
    return record


def _observation_message(observation, index):
    """Construct the allowlisted user message independently of the hybrid client."""
    obs = PolicyObservation.from_dict(observation)
    return {"role": "user", "content": [
        {"type": "input_text", "text": _canonical({"type": "observation", "observation_index": index,
            "prompt": obs.prompt, "observation/state": list(obs.state)})},
        {"type": "input_text", "text": "External camera:"},
        {"type": "input_image", "image_url": obs.image, "detail": "high"},
        {"type": "input_text", "text": "Wrist camera:"},
        {"type": "input_image", "image_url": obs.wrist_image, "detail": "high"},
    ]}


def _retained_input(history, observation, index):
    messages = copy.deepcopy(history) + [_observation_message(observation, index)]
    image_messages = [message for message in messages if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(part.get("type") == "input_image" for part in message["content"])]
    for message in image_messages[:-2]:
        message["content"] = [{"type": "input_text", "text": "[Older camera image omitted by fixed image-retention rule.]"}
            if part.get("type") == "input_image" else part for part in message["content"]]
    return messages


def _history_stats(history, observations, feedback_count, feedback_history="full"):
    result = {"protocol": HISTORY_PROTOCOL, "observations": observations, "feedback_records": feedback_count,
        "messages": len(history), "retained_image_pairs": sum(message.get("role") == "user"
            and isinstance(message.get("content"), list)
            and any(part.get("type") == "input_image" for part in message["content"]) for message in history),
        "image_history_limit": 2, "text_history": "all_observations_validated_responses_output_and_state_feedback",
        "reasoning_replay": "opaque_encrypted_output_items"}
    if feedback_history == "block_summary":
        result.update(protocol=BLOCK_FEEDBACK_PROTOCOL, feedback_history=feedback_history,
                      text_history="all_observations_validated_outputs_completed_block_summaries_and_incomplete_feedback")
        result["feedback_block_summaries"] = sum(
            any(part.get("type") == "input_text" and f'"type":"{BLOCK_SUMMARY_TYPE}"' in part.get("text", "")
                for part in message.get("content", []) if isinstance(part, dict))
            for message in history if message.get("role") == "user")
    return result


def _feedback_block_summary(records):
    """Recompute the allowlisted summary directly from acknowledged feedback."""
    first, last = records[0], records[-1]
    gripper_runs = []
    for record in records:
        command = record["gripper_command"]
        if gripper_runs and gripper_runs[-1]["command"] == command:
            gripper_runs[-1]["actions"] += 1
            gripper_runs[-1]["last_step"] = record["step"]
        else:
            gripper_runs.append({"command": command, "actions": 1,
                                 "first_step": record["step"], "last_step": record["step"]})
    return {"type": BLOCK_SUMMARY_TYPE, "protocol": BLOCK_FEEDBACK_PROTOCOL,
        "chunk_index": first["chunk_index"], "first_step": first["step"], "last_step": last["step"],
        "executed_actions": len(records), "before_state": first["before_state"], "after_state": last["after_state"],
        "commanded_translation_path_m": math.fsum(item["commanded_translation_norm_m"] for item in records),
        "measured_translation_path_m": math.fsum(item["measured_translation_norm_m"] for item in records),
        "commanded_translation_sum_m": [math.fsum(item["desired_translation_delta_m"][axis] for item in records)
                                         for axis in range(3)],
        "measured_translation_sum_m": [math.fsum(item["measured_translation_delta_m"][axis] for item in records)
                                        for axis in range(3)],
        "measured_net_translation_m": math.dist(first["before_state"][:3], last["after_state"][:3]),
        "translation_tracking_error_mean_m": math.fsum(item["translation_tracking_error_m"] for item in records) / len(records),
        "translation_tracking_error_max_m": max(item["translation_tracking_error_m"] for item in records),
        "gripper_command_runs": gripper_runs,
        "gripper_close_steps": sum(item["gripper_command"] > 0 for item in records),
        "last_rolling_progress": last["rolling_progress"]}


def _field_events(trace, enabled, audit):
    """Read event evidence; numeric and visual causes are replayed later."""
    events = {}
    required = {"type", "step", "chunk_index", "reason"}
    for row in trace:
        if row.get("event") != "field_control_event":
            continue
        event = row.get("field_control_event")
        valid = (enabled and isinstance(event, dict)
                 and set(event) in (required, required | {"visual_check"})
                 and event.get("type") == "field_control_event"
                 and event.get("reason") in FIELD_EVENT_REASONS
                 and type(event.get("step")) is int and event["step"] > 0
                 and type(event.get("chunk_index")) is int and event["chunk_index"] >= 0
                 and row.get("step") == event["step"])
        audit.check(valid, "field control event has an enabled, allowlisted measured-transition contract")
        if not valid:
            continue
        audit.check(event["step"] not in events, "at most one field event per measured transition")
        visual = event.get("visual_check")
        if visual is not None:
            audit.check(isinstance(visual, dict) and set(visual) <= {
                "status", "current_distance_px", "baseline_distance_px"},
                "field control event visual evidence excludes hidden state")
        events[event["step"]] = event
    return events


def _next_refresh_step(start, controller, field_steps, direct_steps, events):
    end = start + (field_steps if controller == "coarse_field" else direct_steps)
    return min([end, *(step for step in events if start < step <= end)]) if controller == "coarse_field" else end


def _expected_field_event(after, compact, field, plan, tracker, visual, feedback,
                          motion_window, step, chunk_index, config, tracking_status):
    """Recompute refresh causes from allowed observations, never task success."""
    if compact["motion_mode"] == "hold":
        return None
    reason = "visual_divergence" if visual.get("refresh_requested") else None
    goal = [field.center[i] + compact["goal_offset_m"][i] for i in range(3)]
    if reason is None and (field.query(after.state[:3]) is None or any(
            not config.workspace_min[i] <= after.state[i] <= config.workspace_max[i] for i in range(3))):
        reason = "invalid_direction"
    angle = math.sqrt(sum(v * v for v in after.state[3:6]))
    scale = math.sin(angle / 2) / angle if angle else .5
    current_quaternion = [v * scale for v in after.state[3:6]] + [math.cos(angle / 2)]
    target = compact["orientation_target_xyzw"]
    dot = sum(a * b for a, b in zip(current_quaternion, target)) / math.sqrt(sum(v * v for v in target))
    orientation_error = 2 * math.acos(min(1., abs(dot)))
    if reason is None and math.dist(after.state[:3], goal) <= .005 and orientation_error <= .05:
        reason = "goal_reached"
    if (reason is None and (plan.to_dict().get("search") or {}).get("boundary_clipped")
            and tracker.endpoint_reached(after.state[:3])):
        reason = "field_boundary_reached"
    if reason is None and (tracking_status in ("invalid_measured_start", "no_valid_reference")
                          or (plan.selected_index is None and plan.status != "goal_reached")):
        reason = "invalid_direction"
    motion_window.append((feedback["commanded_translation_norm_m"], feedback["measured_translation_norm_m"]))
    del motion_window[:-5]
    if (reason is None and len(motion_window) == 5
            and sum(item[0] for item in motion_window) >= .005
            and sum(item[1] for item in motion_window) <= .0005):
        reason = "no_progress"
    if reason is None:
        return None
    event = {"type": "field_control_event", "step": step, "chunk_index": chunk_index, "reason": reason}
    if reason == "visual_divergence":
        event["visual_check"] = {key: visual[key] for key in
            ("status", "current_distance_px", "baseline_distance_px") if key in visual}
    return event


def _tracking_geometry(field, position, delta, details, plan, config, previous_progress, audit, prefix):
    """Check tracked-command bounds separately from shared follower replay."""
    norm = math.sqrt(math.fsum(value * value for value in delta))
    target = [position[axis] + delta[axis] for axis in range(3)]
    audit.check(norm <= details["max_command_m"] + 1e-12 <= .05 + 2e-12,
                prefix + "tracked command respects the normalized translation limit")
    audit.check(near(details.get("command_norm_m"), norm)
                and near(details.get("command_target_xyz"), target),
                prefix + "tracked command geometry matches the logged target")
    progress = details["progress_m"]
    audit.check(previous_progress - 1e-12 <= progress <= details["path_length_m"] + 1e-12,
                prefix + "tracking progress is monotonic and remains on the selected path")
    reference = details.get("reference_xyz")
    if reference is not None:
        audit.check(plan.selected_index is not None, prefix + "tracking reference requires a selected trajectory")
        arc = details.get("reference_progress_m")
        audit.check(progress - 1e-12 <= arc <= min(details["path_length_m"], progress + details["lookahead_m"]) + 1e-12,
                    prefix + "tracking reference lies within the configured lookahead")
        if plan.selected_index is not None:
            positions = plan.candidates[plan.selected_index].positions
            traversed, expected_reference = 0., positions[-1]
            for left, right in zip(positions[:-1], positions[1:]):
                length = math.dist(left, right)
                if arc <= traversed + length:
                    fraction = max(0., min(1., (arc - traversed) / length))
                    expected_reference = [left[i] + fraction * (right[i] - left[i]) for i in range(3)]
                    break
                traversed += length
            audit.check(near(reference, expected_reference), prefix + "reference interpolates the selected geometric path")
        audit.check(all(config.workspace_min[i] <= target[i] <= config.workspace_max[i] for i in range(3))
                    and all(config.workspace_min[i] <= reference[i] <= config.workspace_max[i] for i in range(3)),
                    prefix + "tracked target and reference remain inside workspace")
        audit.check(field.segment_valid(position, reference) and field.segment_valid(position, target),
                    prefix + "tracked chords pass full predicted-mask segment checks")
    else:
        audit.check(norm == 0., prefix + "no valid tracking reference yields zero translation")


def _execution_evidence(summary, state, actions, executed, audit):
    """Distinguish a proposal, an execution-intent marker, and an actual reply."""
    # The runner logs proposals before rendering decision video and setting the
    # execution marker. The marker itself precedes sim.step, so even it does not
    # prove delivery. Only the simulator's returned execution record proves that.
    unacknowledged = len(executed) if len(actions) > len(executed) else None
    pending = summary.get("pending_action_index")
    state_pending = state.get("pending_action_index")
    audit.check(pending == state_pending and type(pending) is type(state_pending),
                "summary and state execution markers agree")
    audit.check(pending is None or (type(pending) is int and pending == unacknowledged),
                "execution marker refers only to a proposed unacknowledged action")
    error_stage = summary.get("error", {}).get("stage") if isinstance(summary.get("error"), dict) else None
    evidence = {"all_proposals_acknowledged": unacknowledged is None,
        "unacknowledged_action_index": unacknowledged, "execution_marker_recorded": pending is not None,
        "dispatch_state": "no_unacknowledged_proposal" if unacknowledged is None else "indeterminate",
        "terminal_error_stage": error_stage}
    if unacknowledged is not None:
        if pending is None and error_stage == "deciding":
            evidence["proposal_stage"] = "before_execution_marker"
            audit.notes.append("Final action was proposed while deciding, without a durable execution marker; "
                               "dispatch or execution is not established by the proposal record.")
        elif pending is None:
            evidence["proposal_stage"] = "unmarked"
            audit.notes.append("Final action has no acknowledgement or execution marker; dispatch is indeterminate.")
        else:
            evidence["proposal_stage"] = "execution_marked"
            audit.notes.append("Final action has an execution-intent marker but no simulator acknowledgement; "
                               "delivery and physical execution remain indeterminate.")
    return evidence


def _video_sequence_complete_or_interrupted(expected, actual, summary, actions, executed, audit):
    if actual == expected:
        return True
    # A decision is logged before its video frames and before the execution
    # marker. SIGINT can therefore leave only a prefix of the final unexecuted
    # decision's video events, followed by the recorder's terminal frame.
    last = max(actions, default=-1)
    error = summary.get("error") or {}
    if (summary.get("status") not in ("user_stop", "infrastructure_error")
            or error.get("stage") != "deciding" or summary.get("pending_action_index") is not None
            or last in executed or not actual or actual[-1] != expected[-1]
            or actual[:-1] != expected[:len(actual) - 1]):
        return False
    omitted = expected[len(actual) - 1:-1]
    decision_events = {"routing_decision", "controller_switch", "field_refresh", "action_chunk",
                       "local_replan", "field_action", "direct_action"}
    if not omitted or any(step != last or kind not in decision_events for kind, step in omitted):
        return False
    audit.notes.append("The interrupted final action proposal has no execution acknowledgement or intent marker. "
                       "Its decision video was interrupted before all frames were recorded; all preceding executed "
                       "actions and the terminal video event remain complete and ordered.")
    return True


def _video_audit(run, summary, actions, executed, feedback, audit, completion=None):
    video_path = run / (summary.get("video") or "episode.mp4")
    event_path = video_path.with_suffix(".events.jsonl")
    if not video_path.is_file() or not event_path.is_file():
        audit.check(False, "finalized video and event timeline exist")
        return None, 0
    events = rows(event_path)
    expected = [("observation", 0)]
    previous = None
    for step, action in actions.items():
        diag = action["diagnostics"]
        control = diag["controller"]
        if diag["refreshed"]:
            expected.append(("routing_decision", step))
            if previous is not None and previous != control:
                expected.append(("controller_switch", step))
            expected.append(("field_refresh" if control == "coarse_field" else "action_chunk", step))
        if control == "coarse_field":
            expected.append(("local_replan" if diag.get("plan_refreshed", True) else "field_action", step))
        else:
            expected.append(("direct_action", step))
        if step in executed and step in feedback:
            expected.append(("executed", step))
        previous = control
    if completion is not None:
        expected.append(("model_completion", completion["step"]))
    expected.append(("terminal", completion["step"] if completion is not None else max(actions, default=0)))
    actual = [(event["event"], event["step"]) for event in events]
    audit.check(_video_sequence_complete_or_interrupted(expected, actual, summary, actions, executed, audit),
                "video shows every routing, switch, field/chunk, plan/action, execution in order")
    cursor = 0
    offline = summary["billing"] == "offline_mock"
    for event in events:
        prefix = f"video {event.get('event')} step {event.get('step')}: "
        audit.check(event.get("frame_start") == cursor and type(event.get("frame_count")) is int
                    and event["frame_count"] > 0, prefix + "contiguous frame interval")
        audit.check(near(event.get("video_time_s"), cursor / 20), prefix + "20 Hz frame timestamp")
        cursor += event.get("frame_count", 0)
        audit.check(event.get("offline_mock") is offline and event.get("offline_smoke") is offline,
                    prefix + "live or scripted provenance")
        audit.check(event.get("policy") == "hybrid_field_direct", prefix + "hybrid video policy")
        step, kind = event.get("step"), event.get("event")
        if kind == "model_completion" and completion is not None:
            diag = completion["diagnostics"]
            audit.check(event.get("action") == [] and event.get("selected_index") is None,
                        prefix + "model completion produces no action or selected local plan")
            audit.check(event.get("controller") == diag.get("controller")
                        and event.get("routing") == diag.get("routing"),
                        prefix + "model completion controller and routing displayed")
            audit.check(event.get("compact_prediction") == diag.get("compact_prediction"),
                        prefix + "accepted compact terminal wrapper displayed")
            audit.check(event.get("task_complete") is True
                        and event.get("completion_reason") == diag.get("completion_reason"),
                        prefix + "explicit model completion evidence displayed")
        if step not in actions or kind in ("observation", "terminal"):
            continue
        action, diag = actions[step], actions[step]["diagnostics"]
        audit.check(event.get("controller") == diag["controller"] and event.get("routing") == diag["routing"],
                    prefix + "actual selected controller and routing displayed")
        if diag["controller"] == "contact_direct":
            audit.check(event.get("field_center") is None and event.get("field_step") is None
                        and event.get("selected_index") is None and event.get("candidate_count") is None,
                        prefix + "direct mode displays no stale field or local plan")
        if kind in ("local_replan", "field_action", "direct_action", "executed"):
            audit.check(event.get("action") == action["action"], prefix + "exact actual action displayed")
            expected_state = executed[step]["next_state"] if kind == "executed" and step in executed else action["state"]
            audit.check(event.get("state") == expected_state, prefix + "actual robot state displayed")
            if diag["controller"] == "coarse_field" and "plan_step" in diag:
                for key in ("plan_refreshed", "plan_start_step", "plan_step", "plan_execution_steps", "planned_delta_xyz"):
                    audit.check(near(event.get(key), diag[key]), prefix + "actual cached-plan metadata: " + key)
        if kind == "executed" and step in feedback:
            audit.check(event.get("feedback") == feedback[step]["feedback"], prefix + "exact measured feedback displayed")
        if kind in ("routing_decision", "controller_switch", "field_refresh", "action_chunk"):
            audit.check(event.get("action") == [] and event.get("selected_index") is None,
                        prefix + "generation shown before action and plan selection")
        if kind == "routing_decision":
            audit.check(event.get("hybrid_prediction") == diag.get("hybrid_prediction"), prefix + "complete accepted wrapper displayed")
        if kind == "field_refresh":
            audit.check(event.get("field_prediction") == diag.get("field_prediction"), prefix + "actual generated field displayed")
        if kind == "action_chunk":
            audit.check(event.get("actions") == diag["hybrid_prediction"]["direct"]["actions"], prefix + "actual direct chunk displayed")
    audit.check(bool(events) and events[-1].get("status") == summary["status"], "video terminal outcome matches summary")
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(video_path)],
            check=True, capture_output=True, text=True)
        video = json.loads(probe.stdout)["streams"][0]
        audit.check((video["width"], video["height"]) == (1600, 960), "video dimensions preserve all panels")
        audit.check(video["codec_name"] == "h264" and video["r_frame_rate"] == "20/1", "video codec and frame rate")
        audit.check(int(video["nb_read_frames"]) == cursor, "decoded frame count matches event ranges")
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path), "-f", "null", "-"],
                                 capture_output=True, text=True)
        audit.check(decoded.returncode == 0 and not decoded.stderr.strip(), "complete original MP4 decodes without errors")
        return video, cursor
    except Exception:
        audit.check(False, "ffprobe and full video decode completed")
        return None, cursor


def audit_run(run: Path, output: Path, frozen: Path | None = None, *, allow_offline_smoke=False):
    run, output = run.resolve(), output.resolve()
    manifest = read(run / "manifest.json")
    if manifest.get("configuration", {}).get("policy", {}).get("distance_field_control") is True:
        spec = importlib.util.spec_from_file_location(
            "_distance_control_audit", Path(__file__).with_name("audit_distance_control.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.audit_run(run, output, frozen, allow_offline_smoke=allow_offline_smoke)
    if output == run or run in output.parents:
        raise ValueError("Use a fresh separate audit directory outside the original run")
    if output.exists():
        raise FileExistsError("Audit output already exists")
    summary, state = read(run / "summary.json"), read(run / "run_state.json")
    if state.get("stage") != "finished" or summary.get("status") == "running":
        raise ValueError("Audit only a terminal, finalized run")
    offline = summary.get("billing") == "offline_mock"
    if offline and not allow_offline_smoke:
        raise ValueError("Scripted smoke requires explicit --allow-offline-smoke")
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    config = manifest["configuration"]
    billing = "offline_mock" if offline else "chatgpt_subscription"
    audit.check(manifest.get("identity_hash") == digest(config), "manifest identity digest")
    audit.check(config.get("kind") == "subscription_hybrid_full_task", "hybrid full-task experiment identity")
    audit.check(config.get("task") == summary.get("task") == TASK, "exact requested black bowl task")
    audit.check(config.get("task_id") == config.get("episode_index") == summary.get("task_id") == summary.get("episode_index") == 0,
                "task zero and initial state zero")
    simconfig = config.get("simulator", {})
    audit.check(simconfig.get("suite") == "libero_spatial" and simconfig.get("seed") == 7,
                "LIBERO-Spatial seed-seven setup")
    audit.check(simconfig.get("max_steps") is None and config.get("max_policy_steps") is None,
                "no robot action horizon")
    for key in ("max_api_calls", "max_wall_time_s", "max_cost_usd", "input_price_per_million",
                "cached_input_price_per_million", "cache_write_price_per_million", "output_price_per_million"):
        audit.check(config.get("api", {}).get(key) is None, "no API-key pricing or episode cap: " + key)
    audit.check(config.get("api_key_fallback") is False and config.get("model_fallback") is False,
                "no API-key or model fallback")
    policy_config = config.get("policy", {})
    reasoning_effort = policy_config.get("reasoning_effort")
    stop_on_model_complete = config.get("stop_on_model_complete", policy_config.get("stop_on_model_complete", False))
    audit.check(type(stop_on_model_complete) is bool, "model completion stop is an explicit boolean setting")
    if "stop_on_model_complete" in policy_config:
        audit.check(policy_config["stop_on_model_complete"] is stop_on_model_complete,
                    "policy and experiment completion settings agree")
    stop_on_success = simconfig.get("stop_on_success", True)
    audit.check(type(stop_on_success) is bool, "simulator success stopping is a boolean setting")
    if "stop_on_success" in simconfig:
        rule_has_success = "LIBERO success" in config.get("stopping_rule", [])
        audit.check(stop_on_success is rule_has_success and (stop_on_success or stop_on_model_complete),
                    "simulator stopping agrees with the recorded success/model completion rule")
    for key in ("success_feedback_to_policy", "reward_feedback_to_policy", "object_state_feedback_to_policy"):
        audit.check(policy_config.get(key) is False, "no oracle policy input: " + key)
    grid = GridSpec(**policy_config["grid"])
    field_update_steps = policy_config["update_steps"]
    chunk_size = policy_config["direct_action_chunks"]
    if (grid not in (GridSpec(), HYBRID_GRID)
            or any(type(value) is not int or value < 1 for value in (field_update_steps, chunk_size))):
        raise ValueError("Unsupported recorded hybrid grid or action chunk length")
    planner_config = PlannerConfig(**policy_config["planner"])
    field_execution_steps = policy_config.get("field_execution_steps", 1)
    if type(field_execution_steps) is not int or not 1 <= field_execution_steps <= planner_config.horizon:
        raise ValueError("Field execution length must fit the recorded planning horizon")
    representation = policy_config.get("field_representation", "dense")
    if representation not in ("dense", "dense-grid-v1", COMPACT_FIELD_VERSION, GROUNDED_FIELD_VERSION):
        raise ValueError("Unsupported recorded field representation")
    compact = representation in (COMPACT_FIELD_VERSION, GROUNDED_FIELD_VERSION)
    controller_mode = policy_config.get("controller_mode", "hybrid")
    execution_mode = policy_config.get("execution_mode", "fixed")
    feedback_history = policy_config.get("feedback_history", "full")
    centered_placement_guidance = policy_config.get("centered_placement_guidance", False)
    grasp_settle_guidance = policy_config.get("grasp_settle_guidance", False)
    field_improvements = policy_config.get("field_improvements", False)
    audit.check(type(field_improvements) is bool, "field improvements are an explicit boolean setting")
    audit.check((representation == GROUNDED_FIELD_VERSION) is field_improvements,
                "grounded representation requires its explicit improved field protocol")
    audit.check(not field_improvements or (compact and controller_mode == "hybrid"
                and execution_mode == "tracking" and reasoning_effort == "medium"
                and not stop_on_model_complete and stop_on_success),
                "improved fields require compact Hybrid tracking, medium effort and LIBERO-only stopping")
    if field_improvements:
        for name, expected in (("field_refresh_policy", "events_or_30_actions"),
                               ("local_path_strategy", "analytic_pure_goal_else_lattice"),
                               ("goal_tolerance_m", .005), ("goal_orientation_tolerance_rad", .05)):
            audit.check(policy_config.get(name) == expected, "improved field setting: " + name)
    audit.check(type(grasp_settle_guidance) is bool,
                "grasp settling guidance is an explicit boolean setting")
    audit.check(not grasp_settle_guidance or centered_placement_guidance,
                "grasp settling guidance requires the centered placement protocol")
    audit.check(type(centered_placement_guidance) is bool,
                "centered placement guidance is an explicit boolean setting")
    audit.check(not centered_placement_guidance or
                (compact and controller_mode in ("hybrid", "direct") and not stop_on_model_complete),
                "centered placement guidance requires compact Hybrid or Direct with evaluator-only success stopping")
    audit.check(controller_mode in ("hybrid", "direct"), "recorded full-task controller mode is supported")
    audit.check(execution_mode in ("fixed", "tracking"), "recorded field execution mode is supported")
    audit.check(feedback_history in ("full", "block_summary"), "recorded feedback history mode is supported")
    for setting, expected in (("controller_mode", controller_mode), ("execution_mode", execution_mode),
                             ("feedback_history", feedback_history),
                             ("centered_placement_guidance", centered_placement_guidance),
                             ("grasp_settle_guidance", grasp_settle_guidance),
                             ("field_improvements", field_improvements),
                             ("stop_on_model_complete", stop_on_model_complete)):
        if setting in summary:
            audit.check(summary[setting] == expected and type(summary[setting]) is type(expected),
                        "summary protocol setting agrees with manifest: " + setting)
    model_routing = controller_mode == "hybrid"
    routing_source = "model_prediction" if model_routing else "forced_direct"
    history_protocol = BLOCK_FEEDBACK_PROTOCOL if feedback_history == "block_summary" else HISTORY_PROTOCOL
    audit.check(not stop_on_model_complete or compact, "model completion protocol requires compact representation")
    if compact:
        audit.check(controller_mode in ("hybrid", "direct"),
                    "full-task compact audit distinguishes hybrid routing and forced Direct")
        audit.check((grid, chunk_size, field_update_steps, planner_config.horizon,
                     field_execution_steps) == (HYBRID_GRID, 5, 30, 10, 10)
                    and reasoning_effort in ("high", "medium"),
                    "compact protocol uses the fixed current control configuration")
    prompt_version = (compact_prompt_version(controller_mode, stop_on_model_complete=stop_on_model_complete,
                                            execution_mode=execution_mode, feedback_history=feedback_history,
                                            centered_placement_guidance=centered_placement_guidance,
                                            grasp_settle_guidance=grasp_settle_guidance,
                                            field_improvements=field_improvements) if compact else
                      hybrid_prompt_version(grid, chunk_size, field_update_steps,
                                            planner_config.horizon, field_execution_steps))
    if "prompt_version" in policy_config:
        audit.check(policy_config["prompt_version"] == prompt_version,
                    "manifest prompt version agrees with the recorded protocol settings")
    schema_suffix = "completion_v2" if stop_on_model_complete else "v1"
    if execution_mode != "fixed" or feedback_history != "full":
        schema_suffix = f"{execution_mode}_{feedback_history}" + ("_completion_v3" if stop_on_model_complete else "_v3")
    if centered_placement_guidance:
        schema_suffix = f"{execution_mode}_{feedback_history}_" + (
            "centered_v4" if controller_mode == "hybrid" else "centered_placement_v4")
    if grasp_settle_guidance:
        schema_suffix = f"{execution_mode}_{feedback_history}_grasp_settle_v5"
    if field_improvements:
        schema_suffix = f"{feedback_history}_grounded_v6"
    schema_name = (f"astra_compact_{controller_mode}_{schema_suffix}" if compact else
                   "astra_hybrid_history_" + prompt_version.rsplit("-", 1)[-1])
    expected_schema = (compact_output_schema(controller_mode, stop_on_model_complete=stop_on_model_complete,
                                             field_improvements=field_improvements)
                       if compact else hybrid_output_schema(grid, chunk_size))
    for name, expected in (("field_update_steps", field_update_steps), ("direct_action_steps", chunk_size)):
        if name in policy_config:
            audit.check(policy_config[name] == expected, "recorded horizon aliases agree: " + name)
    audit.check(grid.half_width_m == .08, "recorded field retains 16 cm coverage")
    audit.check(policy_config.get("planner") == json.loads(json.dumps(asdict(
        replace(PlannerConfig(), horizon=planner_config.horizon)))),
                "recorded planner horizon with fixed one-centimeter primitives and workspace")
    audit.check(policy_config.get("adapter_max_rotation_rad") == .15, "original field rotation adapter cap")
    audit.check(config.get("billing") == summary.get("billing") == billing, "billing provenance")
    audit.check(summary.get("api_key_calls") == 0 and summary.get("api_key_fallback") is False, "no API-key calls reported")

    trace = rows(run / "trace.jsonl") if (run / "trace.jsonl").is_file() else []
    field_events = _field_events(trace, field_improvements, audit)
    resets = [row for row in trace if row.get("event") == "reset"]
    audit.check(len(resets) == 1 and trace[0].get("event") == "reset", "one reset before all actions")
    actions_list = [row for row in trace if row.get("event") == "action"]
    executed_list = [row for row in trace if row.get("event") == "executed"]
    feedback_list = [row for row in trace if row.get("event") == "feedback"]
    completion_rows = [row for row in trace if row.get("event") == "model_completion"]
    declared_complete = summary.get("status") == "model_declared_complete"
    audit.check(len(completion_rows) == int(declared_complete),
                "one terminal model completion event exactly when model declares completion")
    completion = completion_rows[0] if completion_rows else None
    audit.check(not declared_complete or stop_on_model_complete,
                "model-declared completion requires explicit stop setting")
    groups = {}
    for label, items in (("action", actions_list), ("executed", executed_list), ("feedback", feedback_list)):
        audit.check([row.get("step") for row in items] == list(range(len(items))), label + " indexes contiguous from zero")
        groups[label] = {row["step"]: row for row in items}
    actions, executed, feedback = groups["action"], groups["executed"], groups["feedback"]
    clean = len(actions) == len(executed) == len(feedback)
    expected_events = ["reset"]
    route_rows = {row["step"]: row for row in trace if row.get("event") == "routing_decision"}
    switch_rows = {row["step"]: row for row in trace if row.get("event") == "controller_switch"}
    trace_controller = None
    next_trace_refresh = 0
    for step in actions:
        control = actions[step]["diagnostics"]["controller"]
        if step == next_trace_refresh:
            expected_events.append("routing_decision")
            if trace_controller is not None and control != trace_controller:
                expected_events.append("controller_switch")
            next_trace_refresh = _next_refresh_step(step, control, field_update_steps, chunk_size, field_events)
        expected_events.append("action")
        if step in executed:
            expected_events.append("executed")
        if step in feedback:
            expected_events.append("feedback")
            if step + 1 in field_events:
                expected_events.append("field_control_event")
        trace_controller = control
    if completion is not None:
        expected_events.append("model_completion")
    audit.check([row["event"] for row in trace] == expected_events, "trace action/execution/feedback ordering")
    audit.check(len(executed) <= len(actions) <= len(executed) + 1, "at most one terminal unacknowledged action")
    audit.check(len(feedback) <= len(executed) <= len(feedback) + 1, "at most one terminal missing-feedback transition")
    execution_evidence = _execution_evidence(summary, state, actions, executed, audit)
    if not clean:
        audit.check(summary.get("status") not in ("success", "diagnostic_complete", "model_declared_complete"),
                    "partial terminal trace cannot claim completion")
        audit.notes.append("Terminal failure leaves an unexecuted action or missing final feedback; only recorded transitions are verified.")
    audit.check(summary.get("steps") == state.get("step") == len(executed), "acknowledged action count")
    audit.check(summary.get("measured_feedback_count") == len(feedback), "measured feedback count")
    audit.check(state.get("status") == summary.get("status"), "state and summary terminal outcome")
    initial = resets[0]["observation"] if resets else None
    observations = {0: initial} if initial else {}
    for step, result in executed.items():
        observations[step + 1] = result["observation"]

    request_paths = sorted((run / "requests").glob("*.request.json"))
    ledger = summary.get("budget", {})
    audit.check(ledger.get("billing") == billing and ledger.get("api_key_calls") == 0, "subscription-only shared ledger")
    audit.check(ledger.get("mock_response_calls" if offline else "api_calls") == len(request_paths), "single ledger request count")
    audit.check(summary.get("mock_response_calls" if offline else "api_calls") == len(request_paths), "summary request count")
    audit.check(ledger.get("subscription_cost_usd") is None and ledger.get("cost_usd_upper_bound") is None,
                "no invented subscription dollar price")
    if offline:
        audit.check(summary.get("api_calls") == ledger.get("api_calls") == ledger.get("subscription_inference_calls") == 0,
                    "scripted fixture has zero actual inference calls")
    history, history_feedback, accepted = [], 0, []
    usage = Counter()
    next_request_step = 0
    expected_instructions = ChatGPTPlanHybridClient._instructions(
        grid, "hybrid", chunk_size, field_update_steps, planner_config.horizon, field_execution_steps) + (
        " Retain only the most recent 2 observation image pairs; all earlier textual observations, accepted complete "
        "hybrid output items (including opaque reasoning), and execution feedback remain in the shared conversation.")
    if compact:
        expected_instructions = compact_instructions(controller_mode, 2, stop_on_model_complete=stop_on_model_complete,
                                                      execution_mode=execution_mode, feedback_history=feedback_history,
                                                      centered_placement_guidance=centered_placement_guidance,
                                                      grasp_settle_guidance=grasp_settle_guidance,
                                                      field_improvements=field_improvements)
    for index, path in enumerate(request_paths):
        record = read(path)
        prefix = f"request {index + 1}: "
        step = next_request_step
        audit.check(step in observations, prefix + "model request has actual preceding observation")
        while history_feedback < step and history_feedback in feedback:
            history.append({"role": "user", "content": [{"type": "input_text",
                "text": _canonical(feedback[history_feedback]["feedback"])}]})
            history_feedback += 1
            if history_feedback in field_events:
                history.append({"role": "user", "content": [{"type": "input_text",
                    "text": _canonical(field_events[history_feedback])}]})
        expected_input = _retained_input(history, observations[step], len(accepted)) if step in observations else None
        if feedback_history == "block_summary" and accepted and expected_input is not None:
            previous = accepted[-1]
            block_length = field_update_steps if previous["prediction"]["controller"] == "coarse_field" else chunk_size
            records = [feedback[i]["feedback"] for i in range(previous["step"], step) if i in feedback]
            complete_block = (len(records) == block_length and step - previous["step"] == block_length
                              and [item.get("chunk_step") for item in records] == list(range(block_length)))
            closing_event = field_events.get(step)
            early_closed = (field_improvements and previous["prediction"]["controller"] == "coarse_field"
                            and closing_event is not None and 0 < len(records) <= block_length
                            and step - previous["step"] == len(records)
                            and closing_event.get("chunk_index") == records[0].get("chunk_index")
                            and [item.get("chunk_step") for item in records] == list(range(len(records))))
            complete_block = complete_block or early_closed
            audit.check(complete_block, prefix + "summary covers one complete acknowledged controller block")
            if complete_block:
                tail = 2 if closing_event is not None else 1
                expected_input[-len(records) - tail:-tail] = [{"role": "user", "content": [{"type": "input_text",
                    "text": _canonical(_feedback_block_summary(records))}]}]
        payload = record.get("payload", {})
        audit.check(record.get("request_ordinal") == index + 1 and record.get("attempt") == 1,
                    prefix + "one ordered attempt per routing request")
        audit.check(index == len(accepted), prefix + "no continuation or retry after rejected request")
        audit.check(not any(item.get("task_complete") for item in accepted),
                    prefix + "no model request after accepted completion declaration")
        audit.check(record.get("prompt_version") == prompt_version and record.get("mode") == "hybrid",
                    prefix + "recorded hybrid protocol provenance")
        audit.check(record.get("grid") == asdict(grid), prefix + "request grid matches manifest")
        audit.check(record.get("billing") == billing and record.get("configuration") == config.get("api"),
                    prefix + "shared subscription configuration")
        audit.check(record.get("configuration", {}).get("endpoint") == ENDPOINT, prefix + "public Responses endpoint")
        audit.check(payload.get("model") == "gpt-6-astra" and payload.get("stream") is True and payload.get("store") is False,
                    prefix + "exact Astra streaming subscription request")
        expected_reasoning = {"effort": reasoning_effort} if reasoning_effort is not None else None
        audit.check(payload.get("reasoning") == expected_reasoning,
                    prefix + "requested reasoning effort matches the recorded configuration")
        audit.check(not UNSUPPORTED_FIELDS.intersection(payload), prefix + "unsupported subscription fields omitted")
        audit.check(payload.get("instructions") == expected_instructions, prefix + "declared hybrid prompt exact match")
        audit.check(payload.get("text", {}).get("format") == {"type": "json_schema", "name": schema_name,
            "strict": True, "schema": expected_schema},
            prefix + "strict hybrid schema matches recorded grid and action count")
        audit.check(payload.get("include") == ["reasoning.encrypted_content"], prefix + "opaque reasoning replay enabled")
        audit.check(payload.get("input") == expected_input,
                    prefix + "entire shared history, opaque output, observations, feedback and image retention exact")
        audit.check(record.get("history_stats_before_request") == _history_stats(history, len(accepted), history_feedback, feedback_history),
                    prefix + "history statistics agree with independent transcript")
        material = {"payload": payload, "grid": asdict(grid), "mode": "hybrid", "prompt_version": prompt_version,
            "endpoint": ENDPOINT, "history_protocol": history_protocol, "image_history": 2}
        if compact:
            material.update(controller_mode=controller_mode, field_representation=representation)
        expected_key = hashlib.sha256(_canonical(material).encode()).hexdigest()
        audit.check(record.get("request_key") == expected_key, prefix + "request fingerprint recomputed")
        stem = path.name.removesuffix(".request.json")
        response_path, result_path = path.with_name(stem + ".response.json"), path.with_name(stem + ".result.json")
        result = read(result_path) if result_path.exists() else {}
        audit.check(result.get("status") in ("valid", "error"), prefix + "request has terminal validity result")
        if not response_path.exists():
            usage["unknown_usage_calls"] += 1
            audit.check(result.get("status") == "error", prefix + "missing response never accepted")
            continue
        response = read(response_path)
        counters = response.get("usage", {})
        if (isinstance(counters, dict) and type(counters.get("input_tokens")) is int
                and type(counters.get("output_tokens")) is int and counters["input_tokens"] >= 0 and counters["output_tokens"] >= 0):
            usage["input_tokens"] += counters["input_tokens"]
            usage["output_tokens"] += counters["output_tokens"]
            for section, name, total in (("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                                         ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
                value = counters.get(section, {}).get(name) if isinstance(counters.get(section), dict) else None
                if type(value) is int and value >= 0:
                    usage[total] += value
        else:
            usage["unknown_usage_calls"] += 1
        audit.check(response.get("billing") == billing, prefix + "response billing provenance")
        if result.get("status") != "valid":
            continue
        audit.check(response.get("status") == "completed" and not response.get("error"), prefix + "response completed before acceptance")
        audit.check(response.get("model") == "gpt-6-astra", prefix + "response model remains exactly Astra")
        reported_reasoning = response.get("reasoning")
        reported_effort = reported_reasoning.get("effort") if isinstance(reported_reasoning, dict) else None
        if reasoning_effort is not None and reported_effort is not None:
            audit.check(reported_effort == reasoning_effort, prefix + "reported reasoning effort matches request")
        try:
            raw_value = AstraClient._extract(response)
            value = copy.deepcopy(raw_value)
            task_complete = False
            effective, grounding = None, None
            if stop_on_model_complete:
                task_complete = _completion_contract(raw_value, audit, prefix, controller_mode)
                value.pop("task_complete", None)
                value.pop("completion_reason", None)
            if compact and controller_mode == "direct":
                value = {"controller": "contact_direct", "target_region": "plate" if task_complete else "black_bowl",
                    "near_target": True,
                    "routing_reason": "Fixed direct controller for timing comparison; routing was not predicted.",
                    "field": None, "direct": value}
            if compact and value.get("controller") == "coarse_field":
                effective = value["field"]
                if field_improvements:
                    effective, grounding = effective_compact_field(effective, observations[step])
                value["field"] = expand_compact_field(effective, grid)
        except Exception:
            audit.check(False, prefix + "one complete structured output can be extracted")
            continue
        _wrapper(value, grid, audit, prefix, chunk_size)
        accepted.append({"step": step, "prediction": value, "raw_prediction": raw_value, "request_key": expected_key,
                         "request_path": path, "output": response["output"], "task_complete": task_complete,
                         "effective_compact_field": effective, "goal_grounding": grounding})
        next_request_step = _next_refresh_step(step, value["controller"], field_update_steps, chunk_size, field_events)
        history = expected_input + copy.deepcopy(response["output"]) if expected_input is not None else history
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "unknown_usage_calls"):
        audit.check(ledger.get(key) == usage[key], "subscription usage reconciliation: " + key)

    selected = {record["step"]: record for record in accepted}
    planner, adapter = LocalPlanner(planner_config), ActionAdapter()
    tracker = PathTracker(lookahead_m=policy_config.get("tracking_lookahead_m", .03),
                          max_command_m=policy_config.get("tracking_max_command_m", .05),
                          endpoint_tolerance_m=.005 if field_improvements else 1e-9) if execution_mode == "tracking" else None
    if tracker is not None:
        audit.notes.append("Tracking actions were reconstructed with the shared PathTracker algorithm; "
                           "selected-path geometry, progress, command norm, workspace bounds and full predicted-mask chords "
                           "were additionally checked separately. This does not independently prove physical collision safety.")
    visual_checker = VisualGoalConsistencyChecker() if field_improvements else None
    last_visual_check, active_compact, motion_window = None, None, []
    if field_improvements:
        audit.notes.append("Improved fields replay the shared compact goal-grounding and visual patch-tracking algorithms "
                           "using only recorded allowed RGB/state observations. Numeric refresh causes and event ordering "
                           "are checked separately; visual matches are heuristics, not independently verified object poses.")
    active_field, active_wrapper, previous_controller, last_feedback = None, None, None, None
    progress, counts, switch_steps = [], Counter(), []
    next_refresh, chunk_start, chunk_index = 0, 0, -1
    active_plan, plan_start, plan_length = None, 0, 0
    for step, record in actions.items():
        prefix = f"action {step}: "
        diag = record["diagnostics"]
        observation = PolicyObservation.from_dict(observations[step])
        audit.check(record.get("state") == list(observation.state), prefix + "decision uses preceding measured state")
        refreshed = step == next_refresh
        if refreshed:
            chunk_start = step
            chunk_index += 1
        chunk_step = step - chunk_start
        audit.check(diag.get("refreshed") is refreshed and diag.get("routing_decision_step") == chunk_start,
                    prefix + "configured action-block routing commitment")
        if refreshed:
            counts["hybrid"] += 1
            audit.check(step in selected, prefix + "routing decision linked to an accepted model wrapper")
            if step not in selected:
                continue
            selection = selected[step]
            audit.check(selection.get("task_complete") is False,
                        prefix + "no action proposed from a model completion response")
            active_wrapper = selection["prediction"]
            active_plan = None
            next_refresh = _next_refresh_step(step, active_wrapper["controller"], field_update_steps, chunk_size, field_events)
            audit.check(diag.get("hybrid_prediction") == active_wrapper, prefix + "complete accepted wrapper unchanged")
            if compact:
                audit.check(diag.get("compact_prediction") == selection["raw_prediction"],
                            prefix + "raw compact model output preserved separately from local expansion")
                audit.check(diag.get("field_representation") == representation,
                            prefix + "compact representation recorded")
                duration = diag.get("field_expansion_latency_s")
                audit.check(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                            prefix + "local expansion latency recorded")
            if field_improvements:
                active_compact = selection["effective_compact_field"]
                motion_window = []
                audit.check(near(diag.get("effective_compact_field"), active_compact)
                            and near(diag.get("goal_grounding"), selection["goal_grounding"]),
                            prefix + "effective goal and rotation compensation recomputed from raw model output and measured pose")
            audit.check(diag.get("api_request_key") == selection["request_key"], prefix + "correct model request fingerprint")
            audit.check(Path(diag.get("api_request_log", "")).name == selection["request_path"].name,
                        prefix + "correct model request artifact")
            if active_wrapper["controller"] == "coarse_field":
                counts["field"] += 1
                active_field = GridField(grid, observation.state[:3], FieldPrediction.from_dict(active_wrapper["field"], grid))
                if visual_checker is not None:
                    last_visual_check = visual_checker.configure(
                        observation, selection["raw_prediction"]["field"]["goal_reference"])
            else:
                counts["direct_chunk"] += 1
                active_field = None
        if active_wrapper is None:
            continue
        controller = active_wrapper["controller"]
        routing = {key: active_wrapper[key] for key in ("controller", "target_region", "near_target", "routing_reason")}
        audit.check(diag.get("controller") == controller,
                    prefix + "selected controller held through chunk")
        audit.check(diag.get("routing") == routing and diag.get("routing_is_model_prediction") is model_routing,
                    prefix + "routing provenance agrees with model-selected or forced controller")
        audit.check(model_routing or controller == "contact_direct", prefix + "forced Direct uses only direct actions")
        changed = previous_controller is not None and previous_controller != controller
        audit.check(diag.get("mode_changed") is changed and diag.get("previous_controller") == previous_controller,
                    prefix + "controller switch metadata")
        if changed:
            switch_steps.append(step)
            audit.check(refreshed, prefix + "controller only switches at boundary")
            switch = switch_rows.get(step, {})
            audit.check(switch.get("from_controller") == previous_controller
                        and switch.get("to_controller") == controller and switch.get("routing") == routing
                        and switch.get("source") == routing_source, prefix + "controller switch trace matches routing")
        if refreshed:
            route = route_rows.get(step, {})
            audit.check(route.get("controller") == controller and route.get("routing") == routing
                        and route.get("previous_controller") == previous_controller
                        and route.get("source") == routing_source, prefix + "explicit routing trace matches accepted wrapper")
        audit.check(diag.get("field_refreshed") is (refreshed and controller == "coarse_field")
                    and diag.get("direct_chunk_refreshed") is (refreshed and controller == "contact_direct"),
                    prefix + "branch refresh flags")
        audit.check(diag.get("chunk_index") == chunk_index and diag.get("chunk_step") == chunk_step,
                    prefix + "chronological chunks and mode-relative action indexes")
        if "update_steps" in diag:
            audit.check(diag["update_steps"] == (field_update_steps if controller == "coarse_field" else chunk_size),
                        prefix + "active mode action horizon")
        audit.check(diag.get("feedback") == last_feedback, prefix + "latest actual feedback before decision")
        audit.check(diag.get("history_stats", {}).get("feedback_records") == step,
                    prefix + "shared history includes every previous action")
        if field_improvements:
            audit.check(diag.get("field_improvements") is True
                        and near(diag.get("field_refresh_event"), field_events.get(step))
                        and near(diag.get("visual_goal_check"), last_visual_check),
                        prefix + "improved field event and visual evidence replay exactly")
        if controller == "coarse_field":
            plan_refreshed = (active_plan is None or step - plan_start == plan_length
                              or (field_improvements and tracker.endpoint_reached(observation.state[:3])))
            if plan_refreshed:
                counts["plan"] += 1
                if field_improvements:
                    goal = [active_field.center[i] + active_compact["goal_offset_m"][i] for i in range(3)]
                    active_plan = planner.plan_analytic(active_field, observation.state[:3], goal_xyz=goal,
                                                       compact_prediction=active_compact, goal_tolerance_m=.005)
                else:
                    active_plan = planner.plan(active_field, observation.state[:3])
                plan_start = step
                # An early event is observed in the future; it must not shorten
                # a plan's action allowance before that transition occurs.
                plan_length = min(field_execution_steps, chunk_start + field_update_steps - step)
                if tracker is not None:
                    tracker.reset(None if active_plan.selected_index is None else
                                  active_plan.candidates[active_plan.selected_index].positions)
            plan = active_plan
            plan_step = step - plan_start
            delta = plan.delta_xyz
            if field_execution_steps > 1 and plan.selected_index is not None:
                positions = plan.candidates[plan.selected_index].positions
                delta = (tuple(positions[plan_step + 1][axis] - positions[plan_step][axis] for axis in range(3))
                         if plan_step + 1 < len(positions) else (0., 0., 0.))
            if field_execution_steps > 1:
                audit.check(diag.get("plan_refreshed") is plan_refreshed
                            and diag.get("plan_start_step") == plan_start
                            and diag.get("plan_step") == plan_step
                            and diag.get("plan_execution_steps") == plan_length,
                            prefix + "committed local-plan execution cadence")
                audit.check(near(diag.get("planned_delta_xyz"), list(delta)),
                            prefix + "current fixed primitive from selected trajectory")
                audit.check(("plan_latency_s" in diag) is plan_refreshed,
                            prefix + "planning latency recorded only for actual replans")
            if tracker is not None:
                previous_progress = tracker.progress_m
                delta, tracking = tracker.command(observation.state[:3], active_field,
                                                  planner_config.workspace_min, planner_config.workspace_max)
                audit.check(diag.get("execution_mode") == "tracking"
                            and near(diag.get("commanded_delta_xyz"), list(delta))
                            and near(diag.get("tracking"), tracking),
                            prefix + "recorded tracking action and diagnostics match shared follower replay")
                _tracking_geometry(active_field, observation.state[:3], delta, tracking, plan,
                                   planner_config, previous_progress, audit, prefix)
            else:
                audit.check(not any(key in diag for key in ("tracking", "commanded_delta_xyz")),
                            prefix + "fixed execution has no tracking commands")
            expected_action = adapter.adapt(delta, active_wrapper["field"]["orientation_target_xyzw"],
                active_wrapper["field"]["gripper_command"], observation.state)
            audit.check(near(diag.get("plan"), plan.to_dict()), prefix + "recorded local plan independently recomputed and held through execution")
            audit.check(diag.get("status") == plan.status, prefix + "recorded local planner status")
            audit.check(diag.get("field_center") == list(active_field.center), prefix + "field retains its own fresh observation center")
            audit.check(near(record["action"], list(expected_action)), prefix + "original adapter and cross-controller grip intent")
            if refreshed:
                audit.check(diag.get("field_prediction") == active_wrapper["field"],
                            prefix + ("field matches local compact expansion" if compact else
                                      "field contains exact model values and masks"))
        else:
            audit.check(record["action"] == active_wrapper["direct"]["actions"][chunk_step], prefix + "exact direct slice without planner or clipping")
            audit.check(not any(key in diag for key in ("plan", "field_center", "field_prediction")),
                        prefix + "direct mode has no field plan or stale field")
        if step in executed:
            result = executed[step]
            counts[controller] += 1
            audit.check(result.get("action") == record["action"] and result.get("controller") == controller,
                        prefix + "simulator acknowledged exactly the selected action")
            audit.check(result.get("next_state") == result["observation"]["observation/state"],
                        prefix + "returned observation matches measured state")
            audit.check(result.get("success") == result.get("sim_status", {}).get("success"),
                        prefix + "success remains evaluator metadata")
            expected = _base.expected_feedback(record["state"], record["action"], result["next_state"], step, progress)
            expected.update(chunk_index=chunk_index, chunk_step=chunk_step)
            if step in feedback:
                audit.check(near(feedback[step]["feedback"], expected), prefix + "all feedback values independently recomputed")
                last_feedback = feedback[step]["feedback"]
                if field_improvements and controller == "coarse_field":
                    after = PolicyObservation.from_dict(result["observation"])
                    last_visual_check = visual_checker.observe(after)
                    expected_event = _expected_field_event(after, active_compact, active_field, plan, tracker,
                        last_visual_check, expected, motion_window, step + 1, chunk_index,
                        planner_config, tracking.get("status"))
                    audit.check(near(field_events.get(step + 1), expected_event),
                                prefix + "field refresh cause independently recomputed after measured transition")
                elif field_improvements:
                    audit.check(step + 1 not in field_events,
                                prefix + "Direct transitions cannot emit field refresh events")
            if record["action"][6] != 0:
                adapter._gripper = 1.0 if record["action"][6] > 0 else -1.0
        previous_controller = controller
    terminal_accepted = [record for record in accepted if record.get("task_complete")]
    audit.check(len(terminal_accepted) == int(declared_complete),
                "accepted completion declaration agrees with terminal outcome")
    if completion is not None:
        prefix = "model completion: "
        step, diag = completion.get("step"), completion.get("diagnostics", {})
        audit.check(type(step) is int and step == len(executed) == len(actions) == len(feedback)
                    and step == next_refresh,
                    prefix + "terminal decision follows all acknowledgements at a model refresh boundary")
        audit.check(summary.get("pending_action_index") is None,
                    prefix + "no terminal action dispatched or pending")
        audit.check(completion.get("task_complete") is True and diag.get("task_complete") is True,
                    prefix + "explicit terminal declaration recorded")
        audit.check(completion.get("action_proposed") is False and completion.get("action_executed") is False,
                    prefix + "terminal response records no action proposal or execution")
        audit.check(diag.get("refreshed") is True and diag.get("step") == step,
                    prefix + "terminal model call is a refreshed decision at current measured step")
        audit.check(not any(key in diag for key in ("plan", "plan_latency_s", "planned_delta_xyz", "action")),
                    prefix + "terminal response invokes no local plan or action adapter")
        audit.check(diag.get("history_stats", {}).get("feedback_records") == step,
                    prefix + "terminal model history includes all executed feedback")
        selection = selected.get(step)
        audit.check(selection is not None and selection.get("task_complete") is True,
                    prefix + "terminal event links to accepted completion response")
        if selection is not None:
            wrapper, raw = selection["prediction"], selection["raw_prediction"]
            controller = wrapper["controller"]
            routing = {key: wrapper[key] for key in ("controller", "target_region", "near_target", "routing_reason")}
            audit.check(diag.get("hybrid_prediction") == completion.get("prediction") == wrapper,
                        prefix + "terminal normalized prediction matches validated response")
            audit.check(diag.get("compact_prediction") == raw,
                        prefix + "raw compact terminal output preserved")
            audit.check(diag.get("controller") == completion.get("controller") == controller
                        and diag.get("routing") == routing and diag.get("routing_is_model_prediction") is model_routing,
                        prefix + "terminal routing retains model-selected or forced provenance")
            audit.check(diag.get("previous_controller") == completion.get("previous_controller") == previous_controller,
                        prefix + "terminal previous controller matches final executed block")
            audit.check(diag.get("completion_reason") == completion.get("completion_reason") == raw.get("completion_reason"),
                        prefix + "completion evidence matches model output")
            audit.check(diag.get("api_request_key") == completion.get("api_request_key") == selection["request_key"],
                        prefix + "terminal request fingerprint matches response")
            audit.check(Path(diag.get("api_request_log", "")).name == selection["request_path"].name
                        and diag.get("api_request_log") == completion.get("api_request_log"),
                        prefix + "terminal request artifact matches response")
            audit.check(summary.get("final_controller") == controller,
                        prefix + "final controller reports terminal prediction")
            counts["hybrid"] += 1
            counts["field" if controller == "coarse_field" else "direct_chunk"] += 1
        metadata = {key: value for key, value in completion.items()
                    if key not in ("event", "diagnostics", "prediction", "mode")}
        audit.check(summary.get("model_completion") == state.get("model_completion") == metadata,
                    prefix + "terminal summary and state preserve exact completion metadata")
        audit.check(summary.get("model_declared_complete") is True and summary.get("benchmark_completed") is True
                    and summary.get("stop_on_model_complete") is True,
                    prefix + "model-based timing endpoint explicitly labeled")
        audit.check(completion.get("reasoning_effort") == summary.get("reasoning_effort") == reasoning_effort,
                    prefix + "terminal effort matches recorded policy")
        audit.notes.append("Run stopped on the model's explicit completion declaration before planning or executing another action; "
                           "this is a timing endpoint, not independent proof of LIBERO success.")
    for name, expected in (("validated_hybrid_refreshes", counts["hybrid"]), ("validated_field_refreshes", counts["field"]),
        ("validated_direct_chunks", counts["direct_chunk"]), ("local_plan_count", counts["plan"]),
        ("coarse_field_actions", counts["coarse_field"]), ("contact_direct_actions", counts["contact_direct"]),
        ("controller_switches", len(switch_steps))):
        audit.check(summary.get(name) == expected, "summary count: " + name)
    unused_valid = len(accepted) - counts["hybrid"]
    audit.check(unused_valid == 0 or (unused_valid == 1 and summary.get("status") in ("infrastructure_error", "user_stop")),
                "every accepted wrapper used once, except documented terminal pre-action failure")
    if unused_valid:
        audit.notes.append("Final accepted model wrapper was not used in a logged action before termination.")
    for name, predicate in (("gripper_close_actions", lambda grip: grip > 0),
                            ("gripper_open_actions", lambda grip: grip < 0), ("gripper_neutral_actions", lambda grip: grip == 0)):
        audit.check(summary.get(name) == sum(predicate(row["action"][6]) for row in executed.values()), "summary count: " + name)
    path_length = sum(_base.norm([result["next_state"][axis] - actions[step]["state"][axis] for axis in range(3)])
                      for step, result in executed.items())
    audit.check(near(summary.get("path_length_m"), path_length), "measured end-effector path length")
    success_steps = [step for step, row in executed.items() if row.get("success")]
    reset_success = bool(resets and resets[0].get("sim_status", {}).get("success"))
    observed_success = bool(success_steps or reset_success)
    audit.check(summary.get("libero_success_observed") is observed_success, "summary success observation agrees with simulator replies")
    if success_steps and stop_on_success:
        audit.check(success_steps == [max(executed)], "no action executed after evaluator success")
    completed = not offline and (summary.get("status") == "success" or (declared_complete and observed_success))
    audit.check(not completed or observed_success, "successful evaluation requires actual LIBERO success")
    audit.check(summary.get("evaluation_completed") is completed and summary.get("scored_episodes") == int(completed)
                and summary.get("success_rate") == (1.0 if completed else None)
                and summary.get("libero_success") == (True if completed else None), "completion and score semantics")
    first_close = next((step for step, row in executed.items() if row["action"][6] > 0), None)
    close = summary.get("first_close_event")
    audit.check((close is not None) == (first_close is not None), "first close metadata presence")
    if first_close is not None:
        audit.check(close.get("action_index") == first_close and close.get("action_ordinal") == first_close + 1
                    and close.get("controller") == executed[first_close]["controller"], "first close exact action identity")
        audit.check(close.get("full_closure_verified") is False and close.get("grasp_verified") is False,
                    "close command not falsely declared a grasp")
    audit.check(summary.get("status") != "first_gripper_close_executed", "hybrid full task is not stopped merely at first close")

    same_initial = None
    prior = ROOT / "runs/astra_field_history_001/trace.jsonl"
    if prior.is_file() and initial is not None:
        with prior.open() as handle:
            prior_reset = json.loads(next(handle))
        same_initial = initial == prior_reset["observation"]
        audit.check(same_initial, "initial state and both RGB views exactly match prior same-task field run")
    source_count, frozen_count, changed_source = _source_audit(config, audit, frozen)
    video, frames = _video_audit(run, summary, actions, executed, feedback, audit, completion)
    stopping_override = _stop_override_audit(run, summary, executed, completion, audit)
    artifact_names = ["summary.json", "manifest.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl",
                      "stop_rule_override.json"]
    report = {"passed": not audit.issues, "checks": audit.checks, "issues": audit.issues, "notes": audit.notes,
        "source_run": str(run), "status": summary["status"], "billing": billing,
        "requested_reasoning_effort": reasoning_effort,
        "controller_mode": controller_mode, "execution_mode": execution_mode, "feedback_history": feedback_history,
        "centered_placement_guidance": centered_placement_guidance,
        "grasp_settle_guidance": grasp_settle_guidance,
        "field_improvements": field_improvements,
        "field_control_events": len(field_events),
        "field_event_reasons": dict(Counter(event["reason"] for event in field_events.values())),
        "grounding_and_visual_checks_use_shared_reconstruction_algorithms": field_improvements,
        "tracking_uses_shared_reconstruction_algorithm": tracker is not None,
        "stop_on_model_complete": stop_on_model_complete,
        "stop_on_libero_success": stop_on_success,
        "live_stopping_override": stopping_override,
        "model_declared_complete": declared_complete,
        "terminal_model_completion_responses": len(terminal_accepted),
        "field_update_steps": field_update_steps, "direct_action_steps": chunk_size,
        "planner_horizon": planner_config.horizon,
        "field_execution_steps": field_execution_steps,
        "executed_actions": len(executed), "feedback_records": len(feedback), "trace_complete": clean,
        "execution_evidence": execution_evidence,
        "validated_hybrid_responses": len(accepted), "coarse_fields": counts["field"], "direct_chunks": counts["direct_chunk"],
        "independently_recomputed_local_plans": counts["plan"], "controller_switch_steps": switch_steps,
        "subscription_requests": 0 if offline else len(request_paths), "mock_requests": len(request_paths) if offline else 0,
        "usage_totals": dict(usage), "libero_success_observed": observed_success, "evaluation_completed": completed,
        "first_close_action_index": first_close, "same_initial_observation_as_prior_field_run": same_initial,
        "video": video, "event_frames": frames, "recorded_source_files_checked": source_count,
        "frozen_source_files_checked": frozen_count,
        "source_archive": str(frozen.resolve()) if frozen is not None and frozen.is_dir() else None,
        "current_workspace_sources_differing_from_archive": changed_source,
        "opaque_model_output_printed": False, "live_inference_calls_by_audit": 0, "simulator_actions_by_audit": 0,
        "limitations": "Checks recorded simulator success and data/control consistency; does not independently reconstruct physical object state or prove grasp.",
        "artifact_sha256": {name: sha(run / name) for name in artifact_names if (run / name).is_file()}}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "AUDIT.md").write_text(
        f"Hybrid subscription audit: {'PASS' if report['passed'] else 'FAIL'}\n\n"
        f"{audit.checks} checks; {len(audit.issues)} issues. {len(executed)} acknowledged actions, "
        f"{len(accepted)} validated hybrid responses, {counts['plan']} independently recomputed local plans.\n\n"
        f"Routing switched at zero-based actions {switch_steps}. Terminal status: `{summary['status']}`. "
        f"LIBERO success observed: {observed_success}; scored evaluation completed: {completed}.\n\n"
        "Full shared histories, opaque output equality, direct action slices, original field planning, feedback, "
        "subscription token accounting and the complete MP4 were checked offline. No model or simulator calls were made.\n\n"
        + ("Issues:\n\n" + "\n".join("- " + issue for issue in audit.issues) + "\n" if audit.issues else "")
        + ("Notes:\n\n" + "\n".join("- " + note for note in audit.notes) + "\n" if audit.notes else ""))
    print(json.dumps({key: report[key] for key in ("passed", "checks", "issues", "status", "executed_actions",
        "validated_hybrid_responses", "independently_recomputed_local_plans", "controller_switch_steps", "video")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-sources", type=Path,
                        help="Optional original-source hash JSON, or a directory of modules and runner copied at run start")
    parser.add_argument("--allow-offline-smoke", action="store_true")
    args = parser.parse_args(argv)
    report = audit_run(args.run, args.output, args.frozen_sources, allow_offline_smoke=args.allow_offline_smoke)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
