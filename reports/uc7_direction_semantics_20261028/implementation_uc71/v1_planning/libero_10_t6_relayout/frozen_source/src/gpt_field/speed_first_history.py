"""Bound the model-facing transcript without discarding local evidence.

Only accepted observable history is read. Older model explanations remain
explicitly attributed model statements; neither recovery nor settled fingers is
treated as proof of a failed grasp, a successful grasp, or task completion.
"""

from __future__ import annotations

import copy
import json
import math

from .astra import _canonical
from .block_feedback_history import _text_object, summarize_feedback_block


BOUNDED_HISTORY_PROTOCOL = "speed-first-bounded-presentation-v1"
BOUNDED_HISTORY_INSTRUCTIONS = (
    " Model-input history is bounded: the latest four observation blocks are retained, "
    "including the latest two live RGB pairs, followed by their accepted intent and measured "
    "feedback. A deterministic older_history_digest retains selected older recovery reasons, "
    "intent changes and measured outcomes. Full raw evidence remains in local logs. Digest "
    "model_routing_reason is a historical model statement, not independently verified truth. "
    "Historical goal offsets belong to their stated observation, not the current state. "
    "Use the final observation as current; do not replay historical commands. Omitted older "
    "records and opaque reasoning are not new evidence. Gripper settling does not certify a grasp."
)


def _observation(message):
    if message.get("role") != "user":
        return None
    for part in message.get("content", []):
        if part.get("type") != "input_text":
            continue
        try:
            value = json.loads(part.get("text", ""))
        except (TypeError, ValueError):
            continue
        if (isinstance(value, dict) and value.get("type") == "observation"
                and type(value.get("observation_index")) is int):
            return value
    return None


def _message(value):
    return {"role": "user", "content": [{"type": "input_text", "text": _canonical(value)}]}


def _short(value):
    """Bound copied scalars/lists; never copy arbitrary nested source fields."""
    if isinstance(value, str):
        return value[:500]
    if value is None or type(value) is bool:
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(value, 6)
    if isinstance(value, (list, tuple)) and len(value) <= 8:
        return [_short(item) for item in value if not isinstance(item, (dict, list, tuple))]
    return None


def _select(value, keys):
    return {key: _short(value[key]) for key in keys if key in value}


def _compact_closed_gripper_operations(messages):
    """New explicit operations can exceed Direct5; summarize only closed blocks."""
    result, index, operation = [], 0, False
    while index < len(messages):
        item = messages[index]
        value = _text_object(item)
        if item.get("role") == "assistant" and value is not None:
            operation = isinstance(value.get("gripper_operation"), dict)
        if not (operation and item.get("role") == "user" and value is not None
                and value.get("type") == "execution_feedback"):
            result.append(copy.deepcopy(item))
            index += 1
            continue
        end, records = index, []
        while end < len(messages):
            record = _text_object(messages[end])
            if (messages[end].get("role") != "user" or record is None
                    or record.get("type") != "execution_feedback"
                    or record.get("chunk_index") != value.get("chunk_index")):
                break
            records.append(record)
            end += 1
        closing = _text_object(messages[end]) if end < len(messages) else None
        completed = (_text_object(messages[end + 1]) if end + 1 < len(messages) else None)
        count = len(records)
        complete = (isinstance(closing, dict) and isinstance(completed, dict)
                    and messages[end].get("role") == messages[end + 1].get("role") == "user"
                    and closing.get("type") == "distance_control_event"
                    and completed.get("type") == "local_gripper_result"
                    and type(value.get("step")) is int and type(value.get("chunk_index")) is int
                    and closing.get("chunk_index") == value["chunk_index"]
                    and closing.get("step") == value["step"] + count
                    and completed.get("step") == closing["step"]
                    and completed.get("chunk_index") == value["chunk_index"]
                    and completed.get("actions_executed") == count
                    and [r.get("step") for r in records] == list(range(value["step"], value["step"] + count))
                    and [r.get("chunk_step") for r in records] == list(range(count)))
        if complete:
            try:
                result.append(_message(summarize_feedback_block(records)))
            except (KeyError, TypeError, ValueError, OverflowError):
                complete = False
        if not complete:
            result.extend(copy.deepcopy(messages[index:end]))
        index = end
    return result


def _turn_digest(block):
    observed = _observation(block[0])
    row = {"observation_index": observed["observation_index"]}
    for item in block[1:]:
        value = _text_object(item)
        if value is None:
            continue
        if item.get("role") == "assistant" and value.get("controller") in ("coarse_field", "contact_direct"):
            row.update(_select(value, ("controller", "target_region")))
            row["model_routing_reason"] = _short(value.get("routing_reason"))
            context = value.get("control_context") or {}
            row["model_phase"] = _short(context.get("phase"))
            branch = value.get("field")
            if isinstance(branch, dict):
                row["historical_field_intent"] = _select(branch, (
                    "goal_offset_m", "orientation_target_xyzw", "gripper_command", "motion_mode"))
            operation = value.get("gripper_operation")
            if isinstance(operation, dict):
                row["gripper_operation"] = _select(operation, ("command",))
            direct = value.get("direct")
            if isinstance(direct, dict):
                actions = direct.get("actions", [])
                if (actions and all(isinstance(a, (list, tuple)) and len(a) == 7
                                    and all(type(x) in (int, float) and math.isfinite(x) for x in a)
                                    for a in actions)):
                    row["historical_direct_intent"] = {
                        "action_count": len(actions),
                        "normalized_translation_sum": [round(math.fsum(a[j] for a in actions), 6) for j in range(3)],
                        "gripper_commands": [a[6] for a in actions[:5]],
                    }
        elif item.get("role") == "user" and value.get("type") == "execution_block_summary":
            row["measured_block"] = _select(value, (
                "first_step", "last_step", "executed_actions", "before_state", "after_state",
                "measured_translation_sum_m", "measured_translation_path_m", "gripper_close_steps"))
        elif item.get("role") == "user" and value.get("type") == "distance_control_event":
            row["local_control_event"] = _select(value, ("step", "reason"))
        elif item.get("role") == "user" and value.get("type") == "local_gripper_result":
            row["local_gripper_result"] = _select(value, (
                "step", "chunk_index", "command", "actions_executed", "done", "stop_reason",
                "finger_motion_observed", "fingers_stable", "pose_error_m",
                "grasp_success_certified", "task_success_certified"))
    return row


def _strategy(row):
    return (row.get("controller"), row.get("model_phase"), row.get("target_region"),
            (row.get("gripper_operation") or {}).get("command"))


def bounded_history_input(messages, *, recent_observations=4, digest_entries=12,
                          max_digest_chars=16000, max_reasoning_chars=16000):
    """Return a deterministic presentation plus accounting, without input mutation.

    The current block and preceding blocks are copied verbatim, except completed
    explicit gripper feedback may be summarized and opaque reasoning is retained
    only up to a bounded character allowance. No encrypted reasoning is decoded.
    """
    if type(recent_observations) is not int or recent_observations < 2:
        raise ValueError("Retain at least two observations for both live image pairs")
    if type(digest_entries) is not int or digest_entries < 1:
        raise ValueError("digest_entries must be a positive integer")
    if type(max_digest_chars) is not int or max_digest_chars < 2000:
        raise ValueError("max_digest_chars must be at least 2000")
    if type(max_reasoning_chars) is not int or max_reasoning_chars < 0:
        raise ValueError("max_reasoning_chars must be a nonnegative integer")
    source = _compact_closed_gripper_operations(messages)
    starts = [i for i, item in enumerate(source) if _observation(item) is not None]
    if not starts or starts[0] != 0:
        raise ValueError("Bounded zero-shot history must start with a live observation")
    blocks = [source[start:end] for start, end in zip(starts, starts[1:] + [len(source)])]
    old, recent = blocks[:-recent_observations], blocks[-recent_observations:]
    rows = [_turn_digest(block) for block in old]
    for i, row in enumerate(rows):
        row["intent_switch"] = i > 0 and _strategy(row) != _strategy(rows[i - 1])
    # Preserve recent details plus older explicit recovery and changed intent.
    recovery = [i for i, row in enumerate(rows) if row.get("model_phase") == "recover"
                or (row.get("local_control_event") or {}).get("reason") in ("no_progress", "invalid_direction")]
    switches = [i for i, row in enumerate(rows) if row["intent_switch"]]
    grippers = [i for i, row in enumerate(rows) if "local_gripper_result" in row]
    latest_gripper = grippers[-1] if grippers else None
    priorities = [*grippers[-1:], *reversed(range(max(0, len(rows) - 4), len(rows))),
                  *reversed(recovery[-4:]), *reversed(switches[-4:]), *reversed(range(len(rows)))]
    chosen = []
    for i in priorities:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == digest_entries:
            break
    selected = [rows[i] for i in sorted(chosen)]
    digest = {
        "type": "older_history_digest", "protocol": BOUNDED_HISTORY_PROTOCOL,
        "source": "accepted_model_statements_and_measured_robot_feedback_only",
        "historical_not_current": True, "older_observations": len(old),
        "omitted_older_observations": len(old) - len(selected), "records": selected,
    }
    while len(_canonical(digest)) > max_digest_chars and digest["records"]:
        # Retain the latest measured gripper outcome even if it is much older
        # than the current navigation/recovery block. It remains dated history.
        protected = rows[latest_gripper] if latest_gripper is not None else None
        removable = next((i for i, row in enumerate(digest["records"]) if row is not protected), None)
        if removable is None:
            break
        digest["records"].pop(removable)
        digest["omitted_older_observations"] += 1
    tail = [copy.deepcopy(item) for block in recent for item in block]
    reasoning_chars, dropped_reasoning, keep_reasoning = 0, 0, set()
    for i in reversed(range(len(tail))):
        if tail[i].get("type") == "reasoning":
            size = len(_canonical(tail[i]))
            if size + reasoning_chars <= max_reasoning_chars:
                keep_reasoning.add(i)
                reasoning_chars += size
            else:
                dropped_reasoning += 1
    tail = [item for i, item in enumerate(tail) if item.get("type") != "reasoning" or i in keep_reasoning]
    result = ([_message(digest)] if old else []) + tail
    stats = {
        "protocol": BOUNDED_HISTORY_PROTOCOL, "retained_recent_observations": len(recent),
        "older_observations": len(old), "digest_records": len(digest["records"]),
        "digest_chars": len(_canonical(digest)) if old else 0,
        "opaque_reasoning_chars": reasoning_chars, "recent_reasoning_items_omitted": dropped_reasoning,
        "source_messages": len(messages), "presented_messages": len(result),
    }
    return result, stats


class SpeedFirstHistoryMixin:
    """Place before ChatGPTPlanSpeedFirstClient in the new opt-in client's MRO."""

    def reset_history(self):
        super().reset_history()
        self._bounded_history_pending = None
        self.last_history_presentation = None

    def build_payload(self, observation, grid=None, mode="hybrid"):
        with self._history_lock:
            payload = super().build_payload(observation, grid, mode)
            complete_input = payload["input"]
            payload["input"], self.last_history_presentation = bounded_history_input(complete_input)
            self._bounded_history_pending = (payload, complete_input)
            payload["instructions"] += BOUNDED_HISTORY_INSTRUCTIONS
            return payload

    def _history_input(self, payload):
        pending = self._bounded_history_pending
        if pending is None or payload is not pending[0]:
            raise RuntimeError("Missing matching complete history for bounded request commit")
        # Accepted response commits the complete textual transcript, not the
        # presentation digest. Failed requests never call this hook.
        restored = {**payload, "input": pending[1]}
        return super()._history_input(restored)

    def history_stats(self):
        stats = super().history_stats()
        stats.update(model_history_protocol=BOUNDED_HISTORY_PROTOCOL,
                     model_history_presentation=copy.deepcopy(self.last_history_presentation),
                     text_history="complete_local_text_bounded_model_presentation")
        return stats
