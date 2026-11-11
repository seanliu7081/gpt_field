"""Opt-in Direct5 baseline for the historical no-demo distance protocols.

Task guidance is extracted verbatim from the original no-demo distance prompt.
This client adds neither the later collision-free prompt nor the joint/RGB
visual-field protocol. Existing clients and prompts are left unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from .block_feedback_history import compact_feedback_history
from .compact_subscription import ChatGPTPlanCompactClient
from .demo_direct_subscription import ChatGPTPlanDemoDirectClient, demo_direct_output_schema
from .distance_subscription import distance_instructions
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient
from .schema import GridSpec


HISTORICAL_ZERO_SHOT_DIRECT_KEYS = frozenset({"actions", "target_region", "routing_reason", "control_context"})


def _task_label(task_key):
    return "drawer" if resolve_distance_task(task_key).key == "drawer_bowl" else "bowl-plate"


def historical_zero_shot_direct_prompt_version(task_key="bowl_plate"):
    return f"astra-direct-{_task_label(task_key)}-historical-zero-shot-v1"


def historical_zero_shot_direct_schema_name(task_key="bowl_plate"):
    return f"astra_direct_{_task_label(task_key).replace('-', '_')}_historical_zero_shot_v1"


def historical_zero_shot_direct_representation(task_key="bowl_plate"):
    return f"direct-actions-{_task_label(task_key)}-historical-zero-shot-v1"


def shared_historical_zero_shot_task_guidance(image_history=2, *, task_key="bowl_plate"):
    """Contiguous original task spans; no semantic or collision-policy additions."""
    if type(image_history) is not int or image_history != 2:
        raise ValueError("Historical zero-shot Direct requires the latest two live image pairs")
    task = resolve_distance_task(task_key)
    source = distance_instructions(image_history, task_key=task.key,
                                   demonstration=False, zero_shot_collision_free=False)

    def span(start, end=None):
        if source.count(start) != 1 or (end is not None and source.count(end) != 1):
            raise ValueError("Historical shared guidance anchors are no longer unique")
        first, last = source.index(start), len(source) if end is None else source.index(end)
        if first >= last:
            raise ValueError("Historical shared guidance anchors are out of order")
        return source[first:last]

    if task.key == "bowl_plate":
        return {
            "allowed_observation": span("Control the bowl task using only ", "Keep the latest "),
            "scene_clearance": span("These are visual estimates, ", "The follower tracks a local path "),
            "grasp_settling": span("Actions execute at 20 Hz, so five actions advance only "),
            "placement": span("Place the bowl centered within the plate rim. ",
                              "Actions execute at 20 Hz, so five actions advance only "),
            "reobservation": span("Missing, ambiguous or lost visual evidence means ", "After release, verify "),
            "post_release_verification": span("After release, verify ", "A verify/hold interval still requests "),
            "recovery_and_stopping": span("Do not declare the task complete from a phase name, ",
                                           "Place the bowl centered within the plate rim. "),
        }
    return {
        "allowed_observation": span("Task: open the top drawer and put the bowl inside. ", "Keep the latest "),
        "articulated_handle": span("A drawer handle is attached to ", "control_context always includes "),
        "handle_and_task_progression": span("Infer which handle belongs ", "For coarse_field, output stage, "),
        "scene_clearance": span("Geometry is visually estimated, ", "The follower tracks from each "),
        "grasp_settling": span("Actions execute at 20 Hz, ", "Before bowl release, "),
        "placement": span("Before bowl release, ", "A hold/verify interval requests "),
        "recovery_and_stopping": span("Automatic upward reobservation is disabled "),
    }


def historical_zero_shot_direct_instructions(image_history=2, *, task_key="bowl_plate"):
    task = resolve_distance_task(task_key)
    common = shared_historical_zero_shot_task_guidance(image_history, task_key=task.key)
    targets = ("target_region is drawer_handle, black_bowl or drawer_interior. " if task.key == "drawer_bowl"
               else "target_region is black_bowl or plate. ")
    phases = (
        "align, release, verify, recover, handle_grasp, drawer_open or handle_release. Use "
        "handle_grasp, drawer_open and handle_release only for target_region=drawer_handle. "
        "Use grasp for black_bowl and release for drawer_interior. Every phase uses Direct. "
        if task.key == "drawer_bowl" else
        "align, release, verify or recover. Use grasp for black_bowl and release for plate. "
        "Every phase uses Direct. "
    )
    return (
        f"Interface {historical_zero_shot_direct_prompt_version(task.key)}; action representation "
        f"{historical_zero_shot_direct_representation(task.key)}. "
        + (f"Task: {task.prompt}. " if task.key == "bowl_plate" else "")
        + common["allowed_observation"]
        + "Use pure Direct control for every task phase. Return strict JSON with exactly actions, "
        "target_region, routing_reason and control_context. actions contains exactly five consecutive "
        "actions, each seven finite normalized numbers in [-1,1]. They execute unchanged from "
        "successive actual measured poses. After five actions, reconsider using current observations "
        "and measured feedback. No distance field, local motion planner, local visual servo, automatic "
        "recovery or controller switching modifies these actions. A close command is not proof of "
        "grasp. A zero gripper action does not change remembered nonzero gripper intent. "
        + targets + "routing_reason is a short observable explanation of the active target and "
        "action choice, not private reasoning. control_context contains phase and visual_reference. "
        "phase is approach, grasp, transport, " + phases
        + "visual_reference may be null. If supplied, it follows the same fixed external-camera "
        "reference contract: kind=end_effector/held_object, moving_point_uv, target_point_uv, "
        "alignment_assumption=same_rigid_point_desired_projection. Both UV vectors are normalized "
        "in [0,1], origin top-left, u right and v down. The target is the desired projection of "
        "the SAME rigid point, accounting for orientation, height and grasp offset. An unavailable "
        "or ambiguous reference remains null; no local calibration or correction is performed. "
        "Keep the latest two live observation image pairs, all earlier textual observations, raw "
        "accepted output items including opaque reasoning, and execution_block_summary records "
        "for complete five-action feedback blocks. Preserve incomplete feedback unchanged. Full "
        "per-action observations and feedback remain in local logs. "
        + common.get("articulated_handle", "") + common.get("handle_and_task_progression", "")
        + common["scene_clearance"] + common["grasp_settling"] + common["placement"]
        + common.get("reobservation", "") + common.get("post_release_verification", "")
        + common["recovery_and_stopping"]
        + "Choose the next five Direct actions from the CURRENT observation and measured feedback. "
    )


def historical_zero_shot_direct_output_schema(*, task_key="bowl_plate"):
    # The output schema and task/phase validation contain no demonstration input.
    return demo_direct_output_schema(task_key=task_key)


class ChatGPTPlanHistoricalZeroShotDirectClient(ChatGPTPlanDemoDirectClient):
    """Original task guidance, no demonstration, unchanged Direct5 execution.

    Reuse the task-specific Direct response validator, but bypass the demo-only
    initializer and prefix handling. Transport remains the same OAuth client.
    """
    def __init__(self, config=None, *, task_key="bowl_plate", demonstration=None,
                 historical_zero_shot_direct_control=True, zero_shot_collision_free=False,
                 controller_mode="direct", reasoning_effort="medium", field_improvements=False,
                 execution_mode="fixed", distance_field_control=False, feedback_history="block_summary",
                 image_history=2, centered_placement_guidance=True, grasp_settle_guidance=True,
                 stop_on_model_complete=False, **kwargs):
        self.task_spec = resolve_distance_task(task_key)
        if demonstration is not None:
            raise ValueError("Historical zero-shot Direct cannot receive a demonstration")
        required = {
            "historical_zero_shot_direct_control": (historical_zero_shot_direct_control, True),
            "zero_shot_collision_free": (zero_shot_collision_free, False),
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
                raise ValueError(f"Historical zero-shot Direct requires {name}={expected!r}")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("Historical zero-shot Direct requires model='gpt-6-astra'")
        if "task_prompt" in kwargs and kwargs["task_prompt"] != self.task_spec.prompt:
            raise ValueError("task_prompt must match the selected task")
        kwargs["task_prompt"] = self.task_spec.prompt
        self.demonstration = None
        self.target_regions = self.task_spec.target_regions
        self.field_representation = historical_zero_shot_direct_representation(task_key)
        self.distance_field_control = self.demo_direct_control = self.zero_shot_collision_free = False
        self.historical_zero_shot_direct_control = True
        self.last_control_context = None
        ChatGPTPlanCompactClient.__init__(self, config, controller_mode=controller_mode,
            reasoning_effort=reasoning_effort, field_improvements=field_improvements,
            execution_mode=execution_mode, feedback_history=feedback_history, image_history=image_history,
            centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
            stop_on_model_complete=stop_on_model_complete, **kwargs)

    @property
    def prompt_version(self):
        return historical_zero_shot_direct_prompt_version(self.task_spec.key)

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None, mode="hybrid"):
        if self.demonstration is not None:
            raise ValueError("Historical zero-shot Direct cannot receive a demonstration")
        payload = ChatGPTPlanHybridClient.build_payload(self, observation, grid, mode)
        payload["instructions"] = historical_zero_shot_direct_instructions(self.image_history, task_key=self.task_spec.key)
        payload["input"] = compact_feedback_history(payload["input"], field_steps=self.field_update_steps,
            direct_steps=self.chunk_size, allow_field_events=False)
        payload["text"]["format"] = {"type": "json_schema",
            "name": historical_zero_shot_direct_schema_name(self.task_spec.key), "strict": True,
            "schema": historical_zero_shot_direct_output_schema(task_key=self.task_spec.key)}
        return payload

    def _history_input(self, payload):
        return payload["input"]
