"""Bounded, deterministic memory of accepted interaction attempts.

This module is presentation only: it never selects an action, certifies a grasp,
reads simulator state, decodes images, or asks for another model call. The caller
supplies its complete *accepted* transcript; observation/step indices are the
available temporal references (wall-clock timestamps are not invented).
"""
from __future__ import annotations

import copy
import json
import math

ATTEMPT_MEMORY_VERSION = "attempt-memory-v1"
_OUTCOMES = ("not_engaged", "possibly_engaged", "lost_engagement", "uncertain")
_PREREQUISITE_STATES = ("supported", "contradicted", "uncertain")
_MAX_ATTEMPTS = 3
MAX_MEMORY_CHARS = 6000


def _string_schema(maximum):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def attempt_review_schema():
    """Schema for a short review produced in the existing action response."""
    previous = {"type": "object", "additionalProperties": False,
                "required": ["attempt_id", "outcome", "evidence"], "properties": {
                    "attempt_id": _string_schema(80),
                    "outcome": {"type": "string", "enum": list(_OUTCOMES)},
                    "evidence": _string_schema(240)}}
    prerequisite = {"type": "object", "additionalProperties": False,
                    "required": ["key", "state", "evidence"], "properties": {
                        "key": _string_schema(48),
                        "state": {"type": "string", "enum": list(_PREREQUISITE_STATES)},
                        "evidence": _string_schema(160)}}
    return {"type": "object", "additionalProperties": False,
            "required": ["previous_attempt", "next_change", "prerequisites"], "properties": {
                "previous_attempt": {"anyOf": [previous, {"type": "null"}]},
                "next_change": _string_schema(240),
                "prerequisites": {"type": "array", "maxItems": 3, "items": prerequisite}}}


def _keys(value, required, label):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{label} has missing or unexpected fields")


def _text(value, maximum, label):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be nonempty text of at most {maximum} characters")
    return value


def validate_attempt_review(value):
    """Validate without interpreting a model judgement as measured truth."""
    _keys(value, ("previous_attempt", "next_change", "prerequisites"), "interaction_review")
    _text(value["next_change"], 240, "next_change")
    previous = value["previous_attempt"]
    if previous is not None:
        _keys(previous, ("attempt_id", "outcome", "evidence"), "previous_attempt")
        _text(previous["attempt_id"], 80, "attempt_id")
        _text(previous["evidence"], 240, "previous_attempt.evidence")
        if previous["outcome"] not in _OUTCOMES:
            raise ValueError("Invalid previous_attempt outcome")
    prerequisites = value["prerequisites"]
    if not isinstance(prerequisites, list) or len(prerequisites) > 3:
        raise ValueError("prerequisites must be a list of at most three items")
    keys = set()
    for item in prerequisites:
        _keys(item, ("key", "state", "evidence"), "prerequisite")
        key = _text(item["key"], 48, "prerequisite.key")
        _text(item["evidence"], 160, "prerequisite.evidence")
        if item["state"] not in _PREREQUISITE_STATES or key in keys:
            raise ValueError("Invalid or duplicate prerequisite state/key")
        keys.add(key)
    return copy.deepcopy(value)


def attempt_memory_instructions():
    return (
        f"Interaction memory {ATTEMPT_MEMORY_VERSION}. attempt_memory summarizes accepted "
        "past commands and measured robot feedback into recent attempts. attempt IDs are "
        "local bookkeeping, not tracked object identities. A target label, phase, engagement "
        "outcome or prerequisite is a dated MODEL CLAIM, not verified object state. Step and "
        "observation indices identify the source; older evidence need not remain true now. "
        "Measured state8 and finger settling describe the robot, not object attachment. "
        "Post-close end-effector displacement does not prove that an object moved with it. "
        "In this SAME action response include interaction_review with previous_attempt=null "
        "when no listed attempt can be judged, otherwise its listed attempt_id, outcome "
        "(not_engaged, possibly_engaged, lost_engagement, uncertain), and brief visual/state "
        "evidence. Use uncertain when occluded or ambiguous; never convert settled fingers "
        "into engagement proof. next_change briefly states what this action changes after a "
        "failed attempt (alignment, approach depth, orientation, or another observed cause), "
        "or why continuing the current strategy is supported. Consecutive close operations "
        "can be one attempt while fingers are still settling; do not shorten necessary "
        "settling merely to reduce the operation count. Prefer a purposeful change over "
        "repeating an unchanged unsuccessful strategy. Include up to three task-relevant "
        "prerequisites as {key,state:supported|contradicted|uncertain,evidence}; reuse a stable "
        "key to update a previous claim. Inspect the CURRENT views before relying on an "
        "earlier prerequisite, such as an accessible destination; preserve task-required "
        "interactions with other objects. Opening ends the local grasp record but does not "
        "certify release, placement, or task success. This memory adds no mandatory probe, "
        "extra verification call, action veto or recovery controller: choose the next normal "
        "Field/Direct/gripper action in this response using current evidence."
    )


def _objects(message):
    for part in message.get("content", []):
        if not isinstance(part, dict) or part.get("type") not in ("input_text", "output_text"):
            continue
        try:
            value = json.loads(part.get("text", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            yield value


def _vector(value, length):
    if (isinstance(value, (list, tuple)) and len(value) == length
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)):
        return [round(float(v), 8) for v in value]
    return None


def _index(value):
    return type(value) is int and value >= 0


def _bounded_text(value, limit):
    return value[:limit] if isinstance(value, str) else None


def _dated(value, index, current):
    return {**copy.deepcopy(value), "source": "model", "observation_index": index,
            "age_observations": current - index, "verified": False}


def _command(intent):
    operation = intent.get("gripper_operation")
    if isinstance(operation, dict) and operation.get("command") in ("open", "close"):
        return operation["command"]
    return None


def _direct_close_intent(intent):
    actions = (intent.get("direct") or {}).get("actions", [])
    return (isinstance(actions, list) and bool(actions)
            and all(_vector(action, 7) is not None for action in actions)
            and any(action[6] > 0 for action in actions))


def _new_attempt(number, target, phase, observation_index, state):
    return {"attempt_id": f"attempt-{number:04d}", "target_region_model_label": target,
            "start_observation_index": observation_index, "last_observation_index": observation_index,
            "status": "active", "end_reason": None,
            "start_state8": copy.deepcopy(state), "latest_measured_state8": copy.deepcopy(state),
            "latest_model_phase": phase, "close_operation_requests": 0,
            "close_operation_actions": 0, "closed_command_actions": 0,
            "executed_actions": 0, "first_feedback_step": None, "last_feedback_step": None,
            "first_close_state8": None, "latest_close_state8": None,
            "post_close_measured_translation_m": [0., 0., 0.],
            "latest_gripper_result": None, "latest_verification_motion": None, "latest_outcome_claim": None,
            "latest_next_change_claim": None, "_closed": False,
            "_nonclose_intent_since_close": False, "_block_motion": None,
            "_block_signs": [], "_first_close_observation_index": None,
            "mixed_gripper_block_seen": False, "post_close_displacement_ambiguous": False}


def _feedback_commands(value):
    if value.get("type") == "execution_feedback":
        values = [value.get("gripper_command")]
    else:
        runs = value.get("gripper_command_runs", [])
        values = [run.get("command") for run in runs if isinstance(run, dict)]
    return [number for number in values if type(number) in (int, float) and math.isfinite(number)]


def _opens_after_close(value, was_closed):
    for command in _feedback_commands(value):
        if command > 0:
            was_closed = True
        elif command < 0 and was_closed:
            return True
    return False


def _add_feedback(attempt, value, command):
    kind = value.get("type")
    if kind == "execution_feedback":
        first = last = value.get("step")
        count = 1
        closed = int(type(value.get("gripper_command")) in (int, float)
                     and value["gripper_command"] > 0)
    else:
        first, last, count = value.get("first_step"), value.get("last_step"), value.get("executed_actions")
        closed = value.get("gripper_close_steps", 0)
    before, after = _vector(value.get("before_state"), 8), _vector(value.get("after_state"), 8)
    if (not _index(first) or not _index(last) or not _index(count) or count < 1
            or last - first + 1 != count or before is None or after is None
            or (attempt["last_feedback_step"] is not None and first <= attempt["last_feedback_step"])):
        return False
    if not _index(closed) or closed > count:
        return False
    if attempt["first_feedback_step"] is None:
        attempt["first_feedback_step"] = first
    attempt["last_feedback_step"] = last
    attempt["latest_measured_state8"] = after
    attempt["executed_actions"] += count
    attempt["closed_command_actions"] += closed
    if command == "close":
        attempt["close_operation_actions"] += count
    if closed and attempt["first_close_state8"] is None:
        attempt["first_close_state8"] = before
        attempt["_first_close_observation_index"] = attempt["last_observation_index"]
    # Ordinary Direct can also close the hand. Preserve the first measured close
    # without claiming that a held object exists, or that every +1 is a new grasp.
    if closed:
        attempt["_closed"] = True
    for number in _feedback_commands(value):
        sign = 1 if number > 0 else -1 if number < 0 else 0
        if sign not in attempt["_block_signs"]:
            attempt["_block_signs"].append(sign)
    if 1 in attempt["_block_signs"] and -1 in attempt["_block_signs"]:
        attempt["mixed_gripper_block_seen"] = True
        if attempt["_first_close_observation_index"] == attempt["last_observation_index"]:
            # Summaries expose only block endpoints. Do not invent the pose at
            # an internal close/open boundary, even when raw feedback existed.
            attempt["first_close_state8"] = None
            attempt["post_close_displacement_ambiguous"] = True
            attempt["post_close_measured_translation_m"] = None
            attempt["latest_verification_motion"] = None
    if command == "close":
        attempt["latest_close_state8"] = after
    elif attempt["_closed"] and command != "open":
        if (attempt["latest_model_phase"] in ("verify", "grasp", "handle_grasp", "drawer_open")
                and not (attempt["post_close_displacement_ambiguous"]
                         and attempt["_first_close_observation_index"] == attempt["last_observation_index"])):
            if attempt["_block_motion"] is None:
                attempt["_block_motion"] = {
                    "source": "measured_robot_feedback", "model_phase": attempt["latest_model_phase"],
                    "observation_index": attempt["last_observation_index"], "first_step": first,
                    "before_xyz_m": before[:3], "executed_actions": 0}
            motion = attempt["_block_motion"]
            motion.update(last_step=last, after_xyz_m=after[:3])
            motion["executed_actions"] += count
            motion["measured_translation_m"] = [round(after[i] - motion["before_xyz_m"][i], 8) for i in range(3)]
            attempt["latest_verification_motion"] = copy.deepcopy(motion)
        if attempt["post_close_measured_translation_m"] is not None:
            for axis in range(3):
                attempt["post_close_measured_translation_m"][axis] += after[axis] - before[axis]
    return True


def build_attempt_memory(messages, current_observation_index, current_state=None):
    """Build at most three recent records from full committed local history.

    Only accepted assistant statements before the current observation are read.
    Pending/current/future responses and explicit incomplete/rejected messages
    cannot contribute. No regex/LLM infers success from free-form explanations.
    """
    if not _index(current_observation_index):
        raise ValueError("current_observation_index must be a nonnegative integer")
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be an accepted transcript sequence")
    state = None if current_state is None else _vector(current_state, 8)
    if current_state is not None and state is None:
        raise ValueError("current_state must be eight finite measured values")
    attempts, active, prerequisites = [], None, {}
    current_index, observed_state, accepted_intent = None, None, None
    feedback_attempt = None
    latest_review, unmatched_claim = None, None

    def end(reason):
        nonlocal active
        if active is not None:
            active.update(status="ended", end_reason=reason)
            active = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        for value in _objects(message):
            if role == "user" and value.get("type") == "observation":
                accepted_intent = None
                feedback_attempt = None
                candidate = value.get("observation_index")
                current_index = candidate if _index(candidate) else None
                observed_state = _vector(value.get("observation/state"), 8)
                if current_index == current_observation_index and state is None:
                    state = copy.deepcopy(observed_state)
                continue
            if current_index is None or current_index >= current_observation_index:
                continue
            if role == "assistant":
                accepted_intent = None
                feedback_attempt = None
                if (message.get("accepted") is False or value.get("accepted") is False
                        or message.get("status", "completed") != "completed"
                        or value.get("controller") not in ("coarse_field", "contact_direct")):
                    continue
                accepted_intent = value
                review = value.get("interaction_review")
                if review is not None:
                    try:
                        review = validate_attempt_review(review)
                    except ValueError:
                        review = None
                if review is not None:
                    latest_review = _dated({"next_change": review["next_change"]}, current_index, current_observation_index)
                    previous = review["previous_attempt"]
                    if previous is not None:
                        record = next((a for a in attempts if a["attempt_id"] == previous["attempt_id"]), None)
                        claim = _dated(previous, current_index, current_observation_index)
                        if record is not None:
                            record["latest_outcome_claim"] = claim
                        else:
                            unmatched_claim = {**claim, "reference_status": "unmatched_not_applied"}
                    for claim in review["prerequisites"]:
                        prerequisites.pop(claim["key"], None)
                        prerequisites[claim["key"]] = _dated(claim, current_index, current_observation_index)
                phase = _bounded_text((value.get("control_context") or {}).get("phase"), 48)
                target = _bounded_text(value.get("target_region"), 80)
                command = _command(value)
                # A destination label in transport/align/release belongs to the
                # existing object's attempt, not a new attempt at the receptacle.
                new_target = (active is not None and target != active["target_region_model_label"]
                              and (phase in ("grasp", "handle_grasp") or command == "close"
                                   or (phase == "approach" and not active["_closed"])))
                retry = (active is not None
                         and (command == "close" or (phase in ("grasp", "handle_grasp")
                                                     and _direct_close_intent(value)))
                         and active["_nonclose_intent_since_close"]
                         and (active["latest_outcome_claim"] or {}).get("outcome")
                         in ("not_engaged", "lost_engagement"))
                if new_target or retry:
                    end("new_interaction_target" if new_target else "model_reported_retry")
                if active is None and (command == "close" or phase in ("approach", "grasp", "handle_grasp", "recover")):
                    active = _new_attempt(len(attempts) + 1, target, phase, current_index, observed_state)
                    attempts.append(active)
                if active is not None:
                    active["_block_motion"] = None
                    active["_block_signs"] = []
                    active["last_observation_index"] = current_index
                    active["latest_model_phase"] = phase
                    if review is not None:
                        active["latest_next_change_claim"] = _dated(
                            {"text": review["next_change"]}, current_index, current_observation_index)
                    if command == "close":
                        active["close_operation_requests"] += 1
                        active["_nonclose_intent_since_close"] = False
                    elif active["_closed"] and command != "open":
                        active["_nonclose_intent_since_close"] = True
                feedback_attempt = active
                continue
            if role != "user" or feedback_attempt is None or accepted_intent is None:
                continue
            command = _command(accepted_intent)
            kind = value.get("type")
            if kind in ("execution_feedback", "execution_block_summary"):
                opens = _opens_after_close(value, feedback_attempt["_closed"])
                if _add_feedback(feedback_attempt, value, command):
                    # A real opening after closure marks a bookkeeping boundary.
                    # Keep the entire accepted block's feedback on its old record.
                    if opens or (command == "open" and feedback_attempt["_closed"]):
                        end("executed_open_command")
            elif kind == "local_gripper_result":
                last_step = feedback_attempt["last_feedback_step"]
                if (last_step is None or value.get("step") != last_step + 1
                        or value.get("command") != command or command not in ("open", "close")
                        or not _index(value.get("actions_executed"))):
                    continue
                result = {"source": "measured_robot_feedback", "observation_index": current_index,
                          "step": value["step"], "command": command,
                          "actions_executed": value["actions_executed"]}
                for key in ("finger_motion_observed", "fingers_stable"):
                    if type(value.get(key)) is bool:
                        result[key] = value[key]
                if isinstance(value.get("stop_reason"), str):
                    result["stop_reason"] = value["stop_reason"][:80]
                error = value.get("pose_error_m")
                if type(error) in (int, float) and math.isfinite(error):
                    result["pose_error_m"] = round(error, 8)
                feedback_attempt["latest_gripper_result"] = result

    selected = attempts[-_MAX_ATTEMPTS:]
    records = []
    for attempt in selected:
        record = {k: copy.deepcopy(v) for k, v in attempt.items() if not k.startswith("_")}
        record["age_observations"] = current_observation_index - attempt["last_observation_index"]
        if record["post_close_measured_translation_m"] is not None:
            record["post_close_measured_translation_m"] = [round(x, 8) for x in record["post_close_measured_translation_m"]]
        record["continuation_close_requests"] = max(0, record["close_operation_requests"] - 1)
        records.append(record)
    result = {"type": "attempt_memory", "version": ATTEMPT_MEMORY_VERSION,
            "source": "accepted_model_statements_and_measured_robot_feedback_only",
            "historical_claims_not_current_facts": True,
            "measurement_note": "State8/displacement describe the robot only; no object attachment or task success is certified.",
            "current_observation_index": current_observation_index,
            "current_measured_state8": state,
            "active_attempt_id": active["attempt_id"] if active is not None else None,
            "attempts_seen": len(attempts), "omitted_older_attempts": len(attempts) - len(records),
            "attempts": records, "latest_review_claim": latest_review,
            "unmatched_attempt_claim": unmatched_claim,
            "prerequisites": list(prerequisites.values())[-3:]}

    # Bound the actual wire representation, including adversarially long valid
    # model strings. Keep the current attempt; remove older records first.
    def size():
        return len(json.dumps(result, separators=(",", ":"), ensure_ascii=True))
    while size() > MAX_MEMORY_CHARS and len(result["attempts"]) > 1:
        result["attempts"].pop(0)
        result["omitted_older_attempts"] += 1
    if size() > MAX_MEMORY_CHARS:
        result["unmatched_attempt_claim"] = None
    while size() > MAX_MEMORY_CHARS and result["prerequisites"]:
        result["prerequisites"].pop(0)
    if size() > MAX_MEMORY_CHARS:
        result["latest_review_claim"] = None
        for record in result["attempts"]:
            record["latest_next_change_claim"] = None
    if size() > MAX_MEMORY_CHARS:
        # ASCII escaping can expand user/model unicode strings sixfold. Evidence
        # is supplementary; preserve provenance, attempt IDs, states and counts.
        for record in result["attempts"]:
            claim = record["latest_outcome_claim"]
            if claim is not None:
                claim["evidence"] = claim["evidence"][:80]
    if size() > MAX_MEMORY_CHARS:
        raise ValueError("Attempt memory exceeded its bounded presentation contract")
    return result
