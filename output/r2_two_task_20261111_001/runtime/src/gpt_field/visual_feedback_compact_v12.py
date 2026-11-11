"""Lossless deduplication of equal recent feedback, after V11 presentation.

Only duplicated execution-block feedback is replaced with an explicit pointer
to a full observation record in the SAME request. Raw history and outputs,
images, robot observations and distinct feedback are never rewritten.
"""
from __future__ import annotations

import json

from .astra import _canonical
from .visual_feedback_compact_v11 import compact_payload_feedback as compact_v11

VERSION = "observable-feedback-shared-record-v12-v1"


def compact_payload_feedback(payload):
    result = compact_v11(payload)
    records = []
    for message in result.get("input", []):
        if message.get("role") != "user":
            continue
        for part in message.get("content", []):
            if part.get("type") != "input_text":
                continue
            try:
                record = json.loads(part["text"])
            except (ValueError, TypeError, KeyError):
                continue
            if isinstance(record, dict):
                records.append((part, record))
    for index, (part, record) in enumerate(records):
        if record.get("type") != "visual_execution_block":
            continue
        feedback = record.get("end_visual_feedback")
        if not isinstance(feedback, dict):
            continue
        if "feedback_ref" in feedback:
            # V11 recognizes the field name and may add its representation
            # marker to our reference on a second pass. The reference already
            # describes its meaning; remove only that redundant marker.
            if set(feedback) <= {"feedback_ref", "feedback_representation"}:
                feedback.pop("feedback_representation", None)
                part["text"] = _canonical(record)
            continue
        # Use only the next observation: no stale, speculative or cross-request
        # reference, and never compare rounded/selected subsets of evidence.
        following = next((item for _, item in records[index+1:]
                          if item.get("type") == "observation"), None)
        if not isinstance(following, dict) or type(following.get("observation_index")) is not int:
            continue
        current = following.get("observable_context", {}).get("visual_feedback")
        # Canonical JSON equality preserves types (True must not equal 1).
        if _canonical(feedback) != _canonical(current):
            continue
        record["end_visual_feedback"] = {"feedback_ref": {
            "observation_index": following["observation_index"],
            "path": "observable_context.visual_feedback",
            "meaning": "exactly identical full feedback in this request",
        }}
        part["text"] = _canonical(record)
    return result
