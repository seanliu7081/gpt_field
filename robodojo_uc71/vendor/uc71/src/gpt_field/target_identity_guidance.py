"""Optional visual target-description prompt, with no observation or control access.

Descriptions are supplied by the user from RGB appearance. They are neither
object poses nor demonstrations, and do not alter state, schemas or execution.
"""
from __future__ import annotations

import hashlib
import json


TARGET_IDENTITY_VERSION = "target-identity-v1"
MAX_TARGET_DESCRIPTION_CHARS = 1500


def normalize_target_description(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("target_description must be a nonempty string or None")
    description = value.strip()
    if not description or len(description) > MAX_TARGET_DESCRIPTION_CHARS:
        raise ValueError("target_description must contain 1 to 1500 characters after stripping whitespace")
    return description


def target_identity_instructions(description: str | None) -> str:
    description = normalize_target_description(description)
    if description is None:
        return ""
    return (
        f"Visual target identity clarification ({TARGET_IDENTITY_VERSION}). "
        "The following description is user-provided visual appearance guidance from the initial "
        "RGB observation, not a live position estimate, an object pose, a demonstration, or a "
        "task-success signal. Target appearance description: "
        + json.dumps(description, ensure_ascii=False)
        + ". Apply this description when interacting with the described object; retain the "
        "task-required interactions with other objects. Identify the intended task object using "
        "its shape, rim or opening when present, "
        "surface pattern and other visible structure together; do not rely on a single color. "
        "Distinguish separate objects from similarly colored background or appliance components. "
        "Maintain the same object instance across task stages and temporary occlusion using the "
        "available observations and history; do not silently substitute a different object. "
        "If identity is uncertain, state that uncertainty in the existing routing explanation "
        "and choose the next action from the current views. Following an unsuccessful grasp, "
        "reconsider both object identity and gripper alignment within the existing model calls, "
        "rather than assuming only the motion magnitude was wrong. This guidance adds no "
        "mandatory check, extra observation call, crop, local tracker, or execution gate. "
        "Infer the current position from current RGB and measured robot feedback; the description "
        "provides no fixed coordinates, camera calibration, depth, contact or completion evidence."
    )


def target_identity_metadata(description: str) -> dict:
    description = normalize_target_description(description)
    if description is None:
        raise ValueError("Target identity metadata requires an enabled description")
    return {
        "version": TARGET_IDENTITY_VERSION,
        "description": description,
        "text_sha256": hashlib.sha256(target_identity_instructions(description).encode()).hexdigest(),
        "source": "human_RGB_description",
    }
