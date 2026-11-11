"""Deterministic summaries of completed, measured execution blocks.

The complete per-action evidence remains in the runner trace. This transformation
does not call a model, inspect opaque reasoning, or use evaluator information.
Only feedback following an accepted controller response can be summarized; an
incomplete or malformed block is left untouched.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from .astra import _canonical


BLOCK_FEEDBACK_PROTOCOL = "responses_output_block_feedback_v1"
BLOCK_SUMMARY_TYPE = "execution_block_summary"


def _text_object(message: dict) -> dict | None:
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return None
    part = content[0]
    if not isinstance(part, dict) or part.get("type") not in ("input_text", "output_text"):
        return None
    try:
        value = json.loads(part["text"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _block_size(message: dict, field_steps: int, direct_steps: int) -> int | None:
    if message.get("role") != "assistant":
        return None
    value = _text_object(message)
    if value is None:
        return None
    if value.get("controller") == "coarse_field" or "goal_offset_m" in value:
        return field_steps
    if value.get("controller") == "contact_direct" or "actions" in value:
        return direct_steps
    return None


def summarize_feedback_block(records: list[dict]) -> dict[str, Any]:
    """Summarize only allowlisted measured robot feedback, retaining its order."""
    first, last = records[0], records[-1]
    commands = []
    for record in records:
        command = record["gripper_command"]
        if commands and commands[-1]["command"] == command:
            commands[-1]["actions"] += 1
            commands[-1]["last_step"] = record["step"]
        else:
            commands.append({"command": command, "actions": 1,
                             "first_step": record["step"], "last_step": record["step"]})
    error = [record["translation_tracking_error_m"] for record in records]
    return {
        "type": BLOCK_SUMMARY_TYPE,
        "protocol": BLOCK_FEEDBACK_PROTOCOL,
        "chunk_index": first["chunk_index"],
        "first_step": first["step"], "last_step": last["step"],
        "executed_actions": len(records),
        "before_state": copy.deepcopy(first["before_state"]),
        "after_state": copy.deepcopy(last["after_state"]),
        "commanded_translation_path_m": math.fsum(record["commanded_translation_norm_m"] for record in records),
        "measured_translation_path_m": math.fsum(record["measured_translation_norm_m"] for record in records),
        "commanded_translation_sum_m": [math.fsum(record["desired_translation_delta_m"][axis] for record in records)
                                         for axis in range(3)],
        "measured_translation_sum_m": [math.fsum(record["measured_translation_delta_m"][axis] for record in records)
                                        for axis in range(3)],
        "measured_net_translation_m": math.dist(first["before_state"][:3], last["after_state"][:3]),
        "translation_tracking_error_mean_m": math.fsum(error) / len(records),
        "translation_tracking_error_max_m": max(error),
        "gripper_command_runs": commands,
        "gripper_close_steps": sum(record["gripper_command"] > 0 for record in records),
        "last_rolling_progress": copy.deepcopy(last["rolling_progress"]),
    }


def compact_feedback_history(messages: list[dict], *, field_steps: int = 30,
                             direct_steps: int = 5, allow_field_events: bool = False) -> list[dict]:
    """Replace complete contiguous feedback blocks, preserving every other item."""
    result = []
    expected = None
    index = 0
    while index < len(messages):
        message = messages[index]
        predicted_steps = _block_size(message, field_steps, direct_steps)
        if predicted_steps is not None:
            expected = predicted_steps
        value = _text_object(message) if message.get("role") == "user" else None
        if value is None or value.get("type") != "execution_feedback":
            result.append(copy.deepcopy(message))
            index += 1
            continue
        end = index
        records = []
        while end < len(messages):
            item = messages[end]
            record = _text_object(item) if item.get("role") == "user" else None
            if (record is None or record.get("type") != "execution_feedback"
                    or record.get("chunk_index") != value.get("chunk_index")):
                break
            records.append(record)
            end += 1
        complete = (expected is not None and len(records) == expected
                    and type(records[0].get("step")) is int
                    and [record.get("chunk_step") for record in records] == list(range(expected))
                    and [record.get("step") for record in records]
                    == list(range(records[0]["step"], records[0]["step"] + expected)))
        if allow_field_events and not complete and expected == field_steps and end < len(messages):
            closing = _text_object(messages[end]) if messages[end].get("role") == "user" else None
            complete = (isinstance(closing, dict) and closing.get("type") == "field_control_event"
                        and closing.get("reason") in ("visual_divergence", "goal_reached",
                            "field_boundary_reached", "invalid_direction", "no_progress")
                        and 0 < len(records) <= field_steps
                        and closing.get("chunk_index") == value.get("chunk_index")
                        and type(records[0].get("step")) is int
                        and closing.get("step") == records[0]["step"] + len(records)
                        and [r.get("chunk_step") for r in records] == list(range(len(records)))
                        and [r.get("step") for r in records]
                            == list(range(records[0]["step"], records[0]["step"] + len(records))))
        if complete:
            try:
                summary = _canonical(summarize_feedback_block(records))
            except (KeyError, TypeError, ValueError, OverflowError):
                complete = False
            else:
                result.append({"role": "user", "content": [{"type": "input_text", "text": summary}]})
        if not complete:
            result.extend(copy.deepcopy(messages[index:end]))
        index = end
    return result
