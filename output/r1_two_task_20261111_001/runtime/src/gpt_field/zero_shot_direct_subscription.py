"""Zero-shot, collision-free task guidance with five model Direct actions.

This opt-in client shares observable task and collision guidance with the new
Hybrid arm. It never attaches a demonstration, expands a field or alters the
accepted Direct actions. Subscription transport and feedback history are reused.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .astra import SchemaError, output_schema
from .block_feedback_history import compact_feedback_history
from .compact_subscription import ChatGPTPlanCompactClient
from .distance_subscription import (
    control_context_schema, shared_zero_shot_task_guidance, validate_control_context,
)
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_GRID, validate_hybrid_prediction
from .schema import GridSpec


ZERO_SHOT_DIRECT_PROMPT_VERSION = "astra-direct-bowl-plate-zero-shot-collision-free-v1"
ZERO_SHOT_DIRECT_SCHEMA_NAME = "astra_direct_bowl_plate_zero_shot_collision_free_v1"
ZERO_SHOT_DIRECT_REPRESENTATION = "direct-actions-bowl-plate-zero-shot-collision-free-v1"
ZERO_SHOT_DIRECT_KEYS = frozenset({"actions", "target_region", "routing_reason", "control_context"})


def zero_shot_direct_instructions(image_history: int = 2, *, task_key: str = "bowl_plate") -> str:
    common = shared_zero_shot_task_guidance(image_history, task_key=task_key)
    return (
        f"Interface {ZERO_SHOT_DIRECT_PROMPT_VERSION}; action representation {ZERO_SHOT_DIRECT_REPRESENTATION}. "
        + common["task_and_zero_shot"] + common["allowed_observation"]
        + "Use pure Direct control for every task phase. Return strict JSON with exactly actions, "
        "target_region, routing_reason and control_context. actions contains exactly five consecutive "
        "actions, each seven finite normalized numbers in [-1,1]. They execute unchanged from "
        "successive actual measured poses. After five actions, reconsider using current observations "
        "and measured feedback. No distance field, local motion planner, local visual servo, automatic "
        "recovery or controller switching modifies these actions. A close command is not proof of "
        "grasp. A zero gripper action does not change remembered nonzero gripper intent. "
        "target_region is black_bowl or plate. routing_reason is a short observable explanation of "
        "the active target and action choice, not private reasoning. control_context contains phase "
        "and visual_reference. phase is approach, grasp, transport, align, release, verify or recover. "
        "Use grasp for black_bowl and release for plate. Every phase uses Direct. "
        "visual_reference may be null. If supplied, it follows the fixed external-camera reference "
        "contract: kind=end_effector/held_object, moving_point_uv, target_point_uv, alignment_assumption="
        "same_rigid_point_desired_projection. Both UV vectors are normalized in [0,1], origin "
        "top-left, u right and v down. The target is the desired projection of the SAME rigid "
        "point, accounting for orientation, height and grasp offset. An unavailable or ambiguous "
        "reference remains null; no local calibration or correction is performed. "
        "Keep the latest two live observation image pairs, all earlier textual observations, raw "
        "accepted output items including opaque reasoning, and execution_block_summary records "
        "for complete five-action feedback blocks. Preserve incomplete feedback unchanged. Full "
        "per-action observations and feedback remain in local logs. "
        + common["scene_clearance"] + common["grasp_settling"] + common["placement"]
        + common["reobservation"] + common["post_release_verification"]
        + common["recovery_and_stopping"] + common["collision_free"]
        + "Choose the next five Direct actions from the CURRENT observation and measured feedback. "
    )


def zero_shot_direct_output_schema(*, task_key: str = "bowl_plate") -> dict:
    shared_zero_shot_task_guidance(task_key=task_key)
    task = resolve_distance_task(task_key)
    schema = output_schema(HYBRID_GRID, "direct")
    schema["properties"].update({
        "target_region": {"type": "string", "enum": list(task.target_regions)},
        "routing_reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "control_context": control_context_schema(task_key=task.key),
    })
    schema["required"] += ["target_region", "routing_reason", "control_context"]
    return schema


class ChatGPTPlanZeroShotDirectClient(ChatGPTPlanCompactClient):
    """Matched medium subscription Direct control with no reference input."""

    field_representation = ZERO_SHOT_DIRECT_REPRESENTATION

    def __init__(self, config=None, *, task_key: str = "bowl_plate", demonstration=None,
                 zero_shot_collision_free: bool = True, controller_mode: str = "direct",
                 reasoning_effort: str = "medium", field_improvements: bool = False,
                 execution_mode: str = "fixed", distance_field_control: bool = False,
                 feedback_history: str = "block_summary", image_history: int = 2,
                 centered_placement_guidance: bool = True, grasp_settle_guidance: bool = True,
                 stop_on_model_complete: bool = False, **kwargs):
        self.task_spec = resolve_distance_task(task_key)
        if demonstration is not None:
            raise ValueError("Zero-shot Direct cannot receive a demonstration")
        shared_zero_shot_task_guidance(image_history, task_key=task_key)
        required = {
            "zero_shot_collision_free": (zero_shot_collision_free, True),
            "controller_mode": (controller_mode, "direct"), "reasoning_effort": (reasoning_effort, "medium"),
            "field_improvements": (field_improvements, False), "execution_mode": (execution_mode, "fixed"),
            "distance_field_control": (distance_field_control, False),
            "feedback_history": (feedback_history, "block_summary"), "image_history": (image_history, 2),
            "centered_placement_guidance": (centered_placement_guidance, True),
            "grasp_settle_guidance": (grasp_settle_guidance, True),
            "stop_on_model_complete": (stop_on_model_complete, False),
        }
        for name, (value, expected) in required.items():
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"Matched zero-shot Direct requires {name}={expected!r}")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("Matched zero-shot Direct requires model='gpt-6-astra'")
        if "task_prompt" in kwargs and kwargs["task_prompt"] != self.task_spec.prompt:
            raise ValueError("task_prompt must match the selected task")
        kwargs["task_prompt"] = self.task_spec.prompt
        self.demonstration = None
        self.target_regions = self.task_spec.target_regions
        self.distance_field_control = False
        self.demo_direct_control = False
        self.zero_shot_direct_control = True
        self.zero_shot_collision_free = True
        self.last_control_context = None
        super().__init__(config, controller_mode=controller_mode, reasoning_effort=reasoning_effort,
            field_improvements=field_improvements, execution_mode=execution_mode,
            feedback_history=feedback_history, image_history=image_history,
            centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
            stop_on_model_complete=stop_on_model_complete, **kwargs)

    @property
    def prompt_version(self) -> str:
        return ZERO_SHOT_DIRECT_PROMPT_VERSION

    def _reset_prediction_diagnostics(self) -> None:
        super()._reset_prediction_diagnostics()
        self.last_control_context = None

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict:
        if self.demonstration is not None:
            raise ValueError("Zero-shot Direct cannot receive a demonstration")
        payload = ChatGPTPlanHybridClient.build_payload(self, observation, grid, mode)
        payload["instructions"] = zero_shot_direct_instructions(self.image_history, task_key=self.task_spec.key)
        payload["input"] = compact_feedback_history(payload["input"], field_steps=self.field_update_steps,
                                                    direct_steps=self.chunk_size, allow_field_events=False)
        payload["text"]["format"] = {"type": "json_schema", "name": ZERO_SHOT_DIRECT_SCHEMA_NAME,
                                      "strict": True, "schema": zero_shot_direct_output_schema(task_key=self.task_spec.key)}
        return payload

    def validate_response(self, data: Any, grid: GridSpec) -> dict:
        self._reset_prediction_diagnostics()
        if not isinstance(data, dict) or set(data) != ZERO_SHOT_DIRECT_KEYS:
            raise SchemaError("Zero-shot Direct output has missing or unexpected fields")
        context = copy.deepcopy(validate_control_context(data["control_context"], task_key=self.task_spec.key))
        phase, target = context["phase"], data["target_region"]
        if phase == "grasp" and target != "black_bowl":
            raise SchemaError("Bowl grasp requires target_region=black_bowl")
        if phase == "release" and target != self.task_spec.placement_target:
            raise SchemaError(f"Bowl release requires target_region={self.task_spec.placement_target}")
        normalized = {
            "controller": "contact_direct", "target_region": target, "near_target": True,
            "routing_reason": data["routing_reason"], "field": None,
            "direct": {"actions": copy.deepcopy(data["actions"])},
        }
        validate_hybrid_prediction(normalized, grid, self.chunk_size, target_regions=self.target_regions)
        self.last_control_context = context
        self.last_compact_prediction = copy.deepcopy(data)
        return normalized
