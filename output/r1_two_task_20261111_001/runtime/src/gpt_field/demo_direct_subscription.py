"""Matched one-demo task baselines with model-generated Direct actions only.

Transport, observable inputs, demonstration bytes, task guidance, and measured
feedback summaries match the distance-controller experiment. The output
contract always supplies five Direct actions; no field is generated or planned.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .astra import SchemaError, output_schema
from .block_feedback_history import compact_feedback_history
from .compact_subscription import ChatGPTPlanCompactClient
from .demonstration import DemoReference
from .distance_subscription import control_context_schema, distance_instructions, validate_control_context
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_GRID, validate_hybrid_prediction
from .schema import GridSpec


DEMO_DIRECT_PROMPT_VERSION = "astra-direct-drawer-one-demo-v1"
DEMO_DIRECT_SCHEMA_NAME = "astra_direct_drawer_one_demo_v1"
DEMO_DIRECT_REPRESENTATION = "direct-actions-drawer-one-demo-v1"
DEMO_DIRECT_KEYS = frozenset({"actions", "target_region", "routing_reason", "control_context"})


def demo_direct_prompt_version(task_key: str = "drawer_bowl") -> str:
    return (DEMO_DIRECT_PROMPT_VERSION if resolve_distance_task(task_key).key == "drawer_bowl"
            else "astra-direct-bowl-plate-one-demo-v1")


def demo_direct_schema_name(task_key: str = "drawer_bowl") -> str:
    return (DEMO_DIRECT_SCHEMA_NAME if resolve_distance_task(task_key).key == "drawer_bowl"
            else "astra_direct_bowl_plate_one_demo_v1")


def demo_direct_representation(task_key: str = "drawer_bowl") -> str:
    return (DEMO_DIRECT_REPRESENTATION if resolve_distance_task(task_key).key == "drawer_bowl"
            else "direct-actions-bowl-plate-one-demo-v1")


def shared_demo_task_guidance(image_history: int = 2, *, task_key: str = "drawer_bowl") -> dict[str, str]:
    """Exact task-guidance spans shared with the existing Hybrid one-demo arm.

    Keep historical prompts unchanged. Extract named, contiguous guidance spans
    and fail loudly if the source contract changes; audits can verify every span
    against each arm's actual recorded requests. Method-specific field/schema
    instructions are deliberately outside these spans.
    """
    task = resolve_distance_task(task_key)
    source = distance_instructions(image_history, task_key=task.key, demonstration=True)

    def span(start: str, end: str) -> str:
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError("Shared task guidance anchors are no longer unique")
        first, last = source.index(start), source.index(end)
        if first >= last:
            raise ValueError("Shared task guidance anchors are out of order")
        return source[first:last]

    if task.key == "bowl_plate":
        return {
            "allowed_observation": span(f"Task: {task.prompt}. ", "Keep the latest "),
            "scene_clearance": span("These are visual estimates, ", "The follower tracks a local path "),
            "grasp_settling": span("Actions execute at 20 Hz, so five actions advance only ",
                                     " The first input message is exactly "),
            "placement": span("Place the bowl centered within the plate rim. ",
                              "Actions execute at 20 Hz, so five actions advance only "),
            "reobservation": span("Missing, ambiguous or lost visual evidence means ", "After release, verify "),
            "post_release_verification": span("After release, verify ", "A verify/hold interval still requests "),
            "recovery_and_stopping": span("Do not declare the task complete from a phase name, ",
                                           "Place the bowl centered within the plate rim. "),
            "demonstration_reference": span("The first input message is exactly ",
                                             "Continue to choose the normal Field/Direct schema "),
            "no_demo_autoplay": "No reference action is executed automatically.",
        }
    return {
        "allowed_observation": span("Task: open the top drawer and put the bowl inside. ", "Keep the latest "),
        "articulated_handle": span("A drawer handle is attached to ", "control_context always includes "),
        "handle_and_task_progression": span("Infer which handle belongs ", "For coarse_field, output stage, "),
        "scene_clearance": span("Geometry is visually estimated, ", "The follower tracks from each "),
        "grasp_settling": span("Actions execute at 20 Hz, ", "Before bowl release, "),
        "placement": span("Before bowl release, ", "A hold/verify interval requests "),
        "recovery_and_stopping": span("Automatic upward reobservation is disabled ", " The first input message is "),
        "demonstration_reference": span("The first input message is exactly ",
                                         "Continue to choose the normal Field/Direct schema "),
        "no_demo_autoplay": "No reference action is executed automatically.",
    }


def demo_direct_instructions(image_history: int = 2, *, task_key: str = "drawer_bowl") -> str:
    if type(image_history) is not int or image_history != 2:
        raise ValueError("Matched one-demo Direct requires the latest two live image pairs")
    task = resolve_distance_task(task_key)
    common = shared_demo_task_guidance(image_history, task_key=task.key)
    target_contract = ("target_region is drawer_handle, black_bowl or drawer_interior. " if task.key == "drawer_bowl"
                       else "target_region is black_bowl or plate. ")
    phase_contract = (
        "align, release, verify, recover, handle_grasp, drawer_open or handle_release. Use "
        "handle_grasp, drawer_open and handle_release only for target_region=drawer_handle. "
        "Use grasp for black_bowl and release for drawer_interior. Every phase uses Direct. "
        if task.key == "drawer_bowl" else
        "align, release, verify or recover. Use grasp for black_bowl and release for plate. "
        "Every phase uses Direct. "
    )
    return (
        f"Interface {demo_direct_prompt_version(task.key)}; action representation {demo_direct_representation(task.key)}. "
        + common["allowed_observation"]
        + "Use pure Direct control for every task phase. Return strict JSON with exactly actions, "
        "target_region, routing_reason and control_context. actions contains exactly five consecutive "
        "actions, each seven finite normalized numbers in [-1,1]. They execute unchanged from "
        "successive actual measured poses. After five actions, reconsider using current observations "
        "and measured feedback. No distance field, local motion planner, local visual servo, automatic "
        "recovery or controller switching modifies these actions. A close command is not proof of "
        "grasp. A zero gripper action does not change remembered nonzero gripper intent. "
        + target_contract + "routing_reason is a short "
        "observable explanation of the active target and action choice, not private reasoning. "
        "control_context contains phase and visual_reference. phase is approach, grasp, transport, "
        + phase_contract +
        "visual_reference may be null. If supplied, it follows the same fixed external-camera "
        "reference contract: kind=end_effector/held_object, moving_point_uv, target_point_uv, "
        "alignment_assumption=same_rigid_point_desired_projection. Both UV vectors are normalized "
        "in [0,1], origin top-left, u right and v down. The target is the desired projection of "
        "the SAME rigid point, accounting for orientation, height and grasp offset. An unavailable "
        "or ambiguous reference remains null; no local calibration or correction is performed. "
        "Keep the latest two live observation image pairs, all earlier textual observations, raw "
        "accepted output items including opaque reasoning, and execution_block_summary records "
        "for complete five-action feedback blocks. Preserve incomplete feedback unchanged. Full "
        "per-action observations and feedback remain in local logs. The immutable external "
        "demonstration stays separate from live history and is present on every request. "
        + common.get("articulated_handle", "") + common.get("handle_and_task_progression", "")
        + common["scene_clearance"] + common["grasp_settling"] + common["placement"]
        + common.get("reobservation", "") + common.get("post_release_verification", "")
        + common["recovery_and_stopping"] + " " + common["demonstration_reference"]
        + "Choose the next five Direct actions from the CURRENT observation and measured feedback. "
        + common["no_demo_autoplay"]
    )


def demo_direct_output_schema(*, task_key: str = "drawer_bowl") -> dict:
    task = resolve_distance_task(task_key)
    schema = output_schema(HYBRID_GRID, "direct")
    schema["properties"].update({
        "target_region": {"type": "string", "enum": list(task.target_regions)},
        "routing_reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "control_context": control_context_schema(task_key=task.key),
    })
    schema["required"] += ["target_region", "routing_reason", "control_context"]
    return schema


class ChatGPTPlanDemoDirectClient(ChatGPTPlanCompactClient):
    """Astra medium, subscription-only, immutable-demo five-action baseline."""

    field_representation = DEMO_DIRECT_REPRESENTATION

    def __init__(self, config=None, *, task_key: str = "drawer_bowl",
                 demonstration: DemoReference | None = None, controller_mode: str = "direct",
                 reasoning_effort: str = "medium", field_improvements: bool = False,
                 execution_mode: str = "fixed", distance_field_control: bool = False,
                 feedback_history: str = "block_summary", image_history: int = 2,
                 centered_placement_guidance: bool = True, grasp_settle_guidance: bool = True,
                 stop_on_model_complete: bool = False, **kwargs):
        self.task_spec = resolve_distance_task(task_key)
        if not isinstance(demonstration, DemoReference):
            raise TypeError("Matched one-demo Direct requires a validated DemoReference")
        if demonstration.metadata()["task_key"] != self.task_spec.key:
            raise ValueError("Demonstration must match the selected task")
        required = {
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
                raise ValueError(f"Matched one-demo Direct requires {name}={expected!r}")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("Matched one-demo Direct requires model='gpt-6-astra'")
        if "task_prompt" in kwargs and kwargs["task_prompt"] != self.task_spec.prompt:
            raise ValueError("task_prompt must match the selected task")
        kwargs["task_prompt"] = self.task_spec.prompt
        self.demonstration = demonstration
        self.target_regions = self.task_spec.target_regions
        self.field_representation = demo_direct_representation(self.task_spec.key)
        self.distance_field_control = False
        self.demo_direct_control = True
        self.last_control_context = None
        super().__init__(config, controller_mode=controller_mode, reasoning_effort=reasoning_effort,
            field_improvements=field_improvements, execution_mode=execution_mode,
            feedback_history=feedback_history, image_history=image_history,
            centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
            stop_on_model_complete=stop_on_model_complete, **kwargs)

    @property
    def prompt_version(self) -> str:
        return demo_direct_prompt_version(self.task_spec.key)

    def _reset_prediction_diagnostics(self) -> None:
        super()._reset_prediction_diagnostics()
        self.last_control_context = None

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict:
        payload = ChatGPTPlanHybridClient.build_payload(self, observation, grid, mode)
        payload["instructions"] = demo_direct_instructions(self.image_history, task_key=self.task_spec.key)
        live = compact_feedback_history(payload["input"], field_steps=self.field_update_steps,
                                        direct_steps=self.chunk_size, allow_field_events=False)
        payload["input"] = [self.demonstration.message(), *live]
        payload["text"]["format"] = {"type": "json_schema", "name": demo_direct_schema_name(self.task_spec.key),
                                      "strict": True, "schema": demo_direct_output_schema(task_key=self.task_spec.key)}
        return payload

    def _history_input(self, payload: Mapping[str, Any]) -> list[dict]:
        if not payload["input"] or payload["input"][0] != self.demonstration.message():
            raise ValueError("The immutable demonstration prefix is missing or changed")
        return payload["input"][1:]

    def validate_response(self, data: Any, grid: GridSpec) -> dict:
        self._reset_prediction_diagnostics()
        if not isinstance(data, dict) or set(data) != DEMO_DIRECT_KEYS:
            raise SchemaError("Demo Direct output has missing or unexpected fields")
        context = copy.deepcopy(validate_control_context(data["control_context"], task_key=self.task_spec.key))
        phase, target = context["phase"], data["target_region"]
        if phase in ("handle_grasp", "drawer_open", "handle_release") and target != "drawer_handle":
            raise SchemaError("Handle interaction phases require target_region=drawer_handle")
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
