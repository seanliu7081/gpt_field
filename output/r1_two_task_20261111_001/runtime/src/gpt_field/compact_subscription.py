"""Subscription control with a compact analytic field and an unchanged local planner.

Only the raw compact response is retained in conversation history. Expansion into
the dense local grid happens after transport completion, solely for the policy.
Forced field/direct modes are for timing comparisons and do not claim model routing.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
import time
from typing import Any, Mapping

from .astra import SchemaError, _canonical, output_schema, validate_prediction
from .block_feedback_history import BLOCK_FEEDBACK_PROTOCOL, BLOCK_SUMMARY_TYPE, compact_feedback_history
from .compact_field import (
    COMPACT_FIELD_VERSION, compact_field_schema, expand_compact_field, validate_compact_field,
)
from .direct_history import HISTORY_PROTOCOL
from .field_grounding import (
    FIELD_EVENT_REASONS, GROUNDED_FIELD_VERSION, effective_compact_field,
    grounded_compact_field_schema,
)
from .hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_CHUNK_SIZE, HYBRID_FIELD_EXECUTION_STEPS,
    HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID, HYBRID_KEYS, HYBRID_PLANNER_HORIZON, HYBRID_TARGET_REGIONS,
    hybrid_output_schema, validate_hybrid_prediction,
)
from .schema import GridSpec
from .observation import PolicyObservation


COMPACT_PROMPT_VERSION = "astra-compact-field-direct-history-v1"
CONTROLLER_MODES = ("hybrid", "field", "direct")
COMPLETION_KEYS = {"task_complete", "completion_reason"}
EXECUTION_MODES = ("fixed", "tracking")
FEEDBACK_HISTORY_MODES = ("full", "block_summary")


def compact_prompt_version(controller_mode: str = "hybrid", *,
                           stop_on_model_complete: bool = False,
                           execution_mode: str = "fixed", feedback_history: str = "full",
                           centered_placement_guidance: bool = False,
                           grasp_settle_guidance: bool = False,
                           field_improvements: bool = False) -> str:
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError("controller_mode must be hybrid, field, or direct")
    if type(stop_on_model_complete) is not bool:
        raise ValueError("stop_on_model_complete must be a boolean")
    if type(centered_placement_guidance) is not bool:
        raise ValueError("centered_placement_guidance must be a boolean")
    if type(grasp_settle_guidance) is not bool:
        raise ValueError("grasp_settle_guidance must be a boolean")
    if type(field_improvements) is not bool:
        raise ValueError("field_improvements must be a boolean")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("execution_mode must be fixed or tracking")
    if feedback_history not in FEEDBACK_HISTORY_MODES:
        raise ValueError("feedback_history must be full or block_summary")
    if stop_on_model_complete and controller_mode == "field":
        raise ValueError("Model completion is available only for hybrid or direct control")
    if field_improvements:
        if controller_mode != "hybrid" or execution_mode != "tracking" or stop_on_model_complete:
            raise ValueError("Field improvements require hybrid tracking without model-completion stopping")
        if grasp_settle_guidance and not centered_placement_guidance:
            raise ValueError("Grasp settling guidance requires centered placement guidance")
        return COMPACT_PROMPT_VERSION.replace(
            "-v1", f"-hybrid-tracking-{feedback_history}-grounded-v6")
    if grasp_settle_guidance:
        if not centered_placement_guidance or controller_mode == "field" or stop_on_model_complete:
            raise ValueError("Grasp settling guidance requires centered hybrid or direct control without model-completion stopping")
        return COMPACT_PROMPT_VERSION.replace(
            "-v1", f"-{controller_mode}-{execution_mode}-{feedback_history}-centered-placement-grasp-settle-v5")
    if centered_placement_guidance:
        if controller_mode == "field" or stop_on_model_complete:
            raise ValueError("Centered placement guidance requires hybrid or direct control without model-completion stopping")
        return COMPACT_PROMPT_VERSION.replace(
            "-v1", f"-{controller_mode}-{execution_mode}-{feedback_history}-centered-placement-v4")
    if execution_mode != "fixed" or feedback_history != "full":
        completion = "-completion" if stop_on_model_complete else ""
        return COMPACT_PROMPT_VERSION.replace(
            "-v1", f"-{controller_mode}-{execution_mode}-{feedback_history}{completion}-v3")
    if stop_on_model_complete:
        return COMPACT_PROMPT_VERSION.replace("-v1", f"-{controller_mode}-completion-v2")
    return COMPACT_PROMPT_VERSION.replace("-v1", f"-{controller_mode}-v1")


def compact_output_schema(controller_mode: str = "hybrid", *,
                          stop_on_model_complete: bool = False,
                          field_improvements: bool = False,
                          target_regions=HYBRID_TARGET_REGIONS) -> dict[str, Any]:
    compact_prompt_version(controller_mode, stop_on_model_complete=stop_on_model_complete)
    if type(field_improvements) is not bool:
        raise ValueError("field_improvements must be a boolean")
    if field_improvements and (controller_mode != "hybrid" or stop_on_model_complete):
        raise ValueError("Grounded field schema requires hybrid control without model-completion stopping")
    if controller_mode == "field":
        return compact_field_schema()
    if controller_mode == "direct":
        schema = output_schema(HYBRID_GRID, "direct")
    else:
        schema = hybrid_output_schema(HYBRID_GRID, HYBRID_CHUNK_SIZE, target_regions=target_regions)
        field_schema = grounded_compact_field_schema() if field_improvements else compact_field_schema()
        schema["properties"]["field"]["anyOf"] = [field_schema, {"type": "null"}]
    if stop_on_model_complete:
        schema["properties"]["task_complete"] = {"type": "boolean"}
        schema["properties"]["completion_reason"] = {"type": "string", "maxLength": 500}
        schema["required"] += ["task_complete", "completion_reason"]
    return schema


def compact_operational_guidance(*, centered_placement_guidance: bool = False,
                                 grasp_settle_guidance: bool = False) -> str:
    """Identical observable grasp/placement advice for Hybrid and Direct arms."""
    if type(centered_placement_guidance) is not bool or type(grasp_settle_guidance) is not bool:
        raise ValueError("Operational guidance flags must be booleans")
    if grasp_settle_guidance and not centered_placement_guidance:
        raise ValueError("Grasp settling guidance requires centered placement guidance")
    guidance = ""
    if centered_placement_guidance:
        guidance += (
            "Place the bowl centered within the plate rim. Visual overlap between the bowl and "
            "plate is insufficient for centered placement. Before releasing the bowl, compare its "
            "alignment with the plate in both RGB camera views and correct any visible offset. "
            "After releasing the bowl and retracting the open gripper, compare the bowl and plate "
            "alignment again in both views; correct any remaining visible offset using the current "
            "visual evidence. "
        )
    if grasp_settle_guidance:
        guidance += (
            "Actions execute at 20 Hz, so five actions advance only 0.25 seconds of simulation; "
            "waiting for a model response does not advance the physics. Gripper closure may require "
            "multiple five-action chunks. Hold the grasp pose while commanding closure until the "
            "measured finger joint positions settle; do not start lifting merely because a close "
            "command was issued. Then perform a short test lift and check both RGB camera views "
            "to determine whether the bowl follows the gripper. After a missed grasp, adjust the "
            "visibly observed alignment before another attempt. "
        )
    return guidance


def compact_instructions(controller_mode: str = "hybrid", image_history: int = 2, *,
                         stop_on_model_complete: bool = False,
                         execution_mode: str = "fixed", feedback_history: str = "full",
                         centered_placement_guidance: bool = False,
                         grasp_settle_guidance: bool = False,
                         field_improvements: bool = False) -> str:
    version = compact_prompt_version(controller_mode, stop_on_model_complete=stop_on_model_complete,
                                     execution_mode=execution_mode, feedback_history=feedback_history,
                                     centered_placement_guidance=centered_placement_guidance,
                                     grasp_settle_guidance=grasp_settle_guidance,
                                     field_improvements=field_improvements)
    guidance = compact_operational_guidance(centered_placement_guidance=centered_placement_guidance,
                                            grasp_settle_guidance=grasp_settle_guidance)
    shared = (
        f"Interface {version}; field representation {COMPACT_FIELD_VERSION}. Complete the task using "
        "only task text, two RGB camera images, robot state, and retained observation/output/execution "
        "feedback. No object poses, depth, contact sensors, reward, success labels, demonstrations, or "
        "future images are available. State is [end-effector xyz meters, orientation axis-angle xyz "
        "radians, two gripper joint positions], not robot joint angles. Coordinates use the world frame. "
        "Normalized action feedback is [dx,dy,dz,rx,ry,rz,gripper]: translation 1 means 0.05 meters; "
        "rotation 1 means 0.5 radians in world axis-angle. Gripper +1 closes, -1 opens, 0 requests no "
        "increment. Commands are relative to each actual measured pose and do not guarantee realized "
        "motion. Compare measured tracking and visual object motion; gripper closure alone proves "
        "neither grasp nor success. "
        f"Retain the latest {image_history} image pairs and all earlier textual observations, raw accepted "
        "output items (including opaque reasoning), and per-action feedback. Historical offsets are "
        "relative to their own preceding observation, never the current one. chunk_index identifies "
        "chronological controller intervals; chunk_step starts at zero in each interval. "
        "Return only strict JSON matching the schema. "
    )
    routing = (
        "Return one wrapper with controller, target_region, near_target, routing_reason, field, direct. "
        "Choose coarse_field for free-space approach/transport (30 actions before reconsidering), or "
        "contact_direct for fine alignment/grasp/lift clearance/placement/release (5 actions). Routing "
        "is reversible at these boundaries; there is no automatic stage machine or forced grasp. "
        "Infer proximity visually, never by thresholding potential or parsing stage. target_region is "
        "black_bowl for approach/grasp/lift clearance or plate for placement. near_target is true exactly "
        "for contact_direct. routing_reason briefly states observable evidence, not private reasoning. "
        "Set the inactive branch to null. Holding the bowl alone does not preclude free-space field transport. "
    )
    field = (
        "Output the compact field object: stage, unit orientation_target_xyzw quaternion, gripper_command "
        "open/close/hold, motion_mode move/hold, goal_offset_m, attraction_weights, repulsors, blocked_boxes, "
        "unknown_boxes. Offsets are world xyz meters from the current end-effector center. The local "
        "33x33x33 grid has 0.005-meter spacing and fixed +/-0.08-meter coverage. Return parameters, never "
        "dense grid values. Local expansion evaluates sqrt(sum_i attraction_weights[i] * "
        "((offset[i]-goal_offset_m[i])/0.08)^2)) plus each positive Gaussian repulsor "
        "weight*exp(-0.5*||offset-center_offset_m||^2/radius_m^2). Lower is preferred; this is an "
        "uncalibrated preference potential, not metric distance or a shortest-path solution. Goal may "
        "lie beyond the cube. Use at most four repulsors (center_offset_m, radius_m, weight), and at "
        "most four blocked and four unknown axis-aligned boxes (lower_offset_m, upper_offset_m). "
        "Boxes include boundaries; blocked takes precedence over unknown. Unknown is insufficient "
        "visual evidence, not free space. Masks describe end-effector positions, not certified whole-arm "
        "collision avoidance. Hold the inferred scene/stage and auxiliary commands fixed while evaluating "
        "hypothetical positions. Reuse the expanded fixed-center field for 30 executed actions. Exact "
        "lattice planning selects ten 1-cm primitives, all inside the cube, then executes all ten ordered "
        "fixed world translation deltas before replanning from actual measured pose. Three local plans "
        "occur at interval steps 0, 10, 20; the 10-cm path may turn to remain within the 16-cm cube. "
        "Remaining translations are neither regenerated nor corrected toward old waypoints during the "
        "ten-action window. No complete valid plan, or motion_mode=hold, yields zero translation for "
        "that window. Orientation still adapts to every measured state with a 0.15-radian cap; every "
        "action records feedback. Gripper hold repeats the last executed nonzero intent across both "
        "controllers, initially open; direct gripper zero leaves that remembered intent unchanged. "
    )
    direct = (
        "Output an object containing only actions: exactly five consecutive actions, each seven finite "
        "normalized values in [-1,1] in the order described above. Execute the five actions as supplied, "
        "with no local field planner, correction, or forced closure; reconsider after five measured "
        "actions. Zero translation/rotation requests no change from the actual measured pose. "
    )
    if feedback_history == "block_summary":
        shared = shared.replace(
            "output items (including opaque reasoning), and per-action feedback.",
            "output items (including opaque reasoning), and deterministic execution_block_summary "
            "records for completed controller blocks. Full per-action feedback is stored locally; "
            "a summary contains the block's first/last measured robot states, executed-action count, "
            "commanded and measured path lengths and displacement sums, tracking-error statistics, "
            "ordered gripper command runs, and latest rolling progress. These are robot-motion "
            "measurements, not evidence of object contact or task success. Incomplete blocks retain "
            "their original per-action feedback.")
    if execution_mode == "tracking":
        field = field.replace(
            "lattice planning selects ten 1-cm primitives, all inside the cube, then executes all ten ordered "
            "fixed world translation deltas before replanning from actual measured pose. Three local plans "
            "occur at interval steps 0, 10, 20; the 10-cm path may turn to remain within the 16-cm cube. "
            "Remaining translations are neither regenerated nor corrected toward old waypoints during the "
            "ten-action window.",
            "lattice planning selects a geometric path of ten 1-cm segments inside the cube. A local "
            "feedback follower re-targets from each actual measured pose toward a point up to 3 cm "
            "ahead along that path, with translation command norm capped at 5 cm. It checks the "
            "command chord against the field masks and cube, using a shorter valid target if needed. "
            "The ten geometric segments are not ten fixed replayed action deltas: actual progress "
            "depends on measured tracking. Three local plans occur at interval steps 0, 10, 20; "
            "each plan guides ten measured control actions before replanning. The 10-cm geometric "
            "path may turn to remain within the 16-cm cube. The same fixed-center field is still "
            "reconsidered by the model only after 30 actions.")
    if field_improvements:
        shared = shared.replace(COMPACT_FIELD_VERSION, GROUNDED_FIELD_VERSION)
        routing = routing.replace("(30 actions before reconsidering)", "(at most 30 actions before reconsidering)")
        # Replace the complete execution contract to avoid contradicting early refresh.
        start = field.index("Reuse the expanded fixed-center field")
        end = field.index("Orientation still adapts")
        field = field[:start] + (
            "Reuse the expanded fixed-center field for at most 30 executed actions. For a pure goal "
            "attraction field, analytic straight-line planning uses at most ten segments of at most "
            "1 cm each, stops at the goal or valid local cube boundary, and never extends beyond the "
            "known cube. Other fields retain constrained lattice planning. A local feedback follower "
            "uses the actual measured pose, up to 3 cm path lookahead and a 5 cm translation norm cap. "
            "Each plan guides at most ten measured control actions; arrival can end it earlier. "
            "Goal arrival requires measured position within 5 mm and orientation within 0.05 radians; "
            "continue local rotation correction if only the position has arrived. "
            "Goal arrival, field-boundary arrival, invalid direction, sustained measured lack of "
            "progress, or reliable visual contradiction requests fresh model routing before more "
            "actions; these events do not stop the task or certify success. A field_control_event "
            "record explains early reconsideration. No valid motion, or motion_mode=hold, yields "
            "zero translation. "
        ) + field[end:]
        field += (
            "Also include goal_reference, either null or an object with kind, object_center_offset_m, "
            "desired_object_center_offset_m, visual_anchor. For kind=end_effector use the raw "
            "goal_offset_m and set both object offsets to null. For kind=held_object, estimate the "
            "held bowl center and desired bowl-center position from BOTH current RGB views; both "
            "offsets are world xyz meters relative to the CURRENT end-effector center, never relative "
            "to the camera or to each other. Supply both three-vectors within +/-0.5 m. Local code "
            "rotates the estimated current gripper-to-object offset by R_target R_current.T and "
            "sets the effective end-effector goal to desired_object_center_offset_m minus that "
            "rotated offset. This assumes the visible grasp stays rigid; it does not prove a grasp "
            "or measure an object pose. Preserve raw goal_offset_m as your uncorrected end-effector "
            "intent for audit; the local derived goal replaces it only for held_object. Use null "
            "goal_reference when neither reference can be estimated reliably. visual_anchor is null "
            "unless a textured point rigidly attached to the moving gripper or held bowl can be "
            "identified in the FIXED EXTERNAL observation/image. If available, supply moving_point_uv "
            "and target_point_uv, each normalized [u,v] in [0,1], origin top-left, u right and v down. "
            "The target is the desired image location of that same moving point, accounting for "
            "grasp offset and requested orientation, not automatically the plate center. Never use "
            "wrist-camera coordinates for this anchor. A local immutable-template checker ignores "
            "textureless, ambiguous or lost matches, and requests reconsideration only after two "
            "reliable checks show separation worsening beyond pixel noise while the measured robot "
            "actually moves. A null anchor means visual consistency checking is unavailable; it "
            "does not imply the goal is correct. "
        )
        guidance += (
            "In both free-space transport and contact Direct alignment, account for the observed "
            "bowl-center offset from a rim grasp: align the bowl center over the desired plate "
            "position, not merely the gripper center. Verify the alignment in both RGB views and "
            "re-estimate it after visible slip, regrasp or orientation change. This additional "
            "guidance does not alter the five-action Direct interface. "
        )
    if controller_mode == "hybrid":
        instructions = shared + routing + "For coarse_field: " + field + "For contact_direct: " + direct
        if stop_on_model_complete:
            instructions += (
                "Also include task_complete (boolean) and completion_reason (string, at most 500 "
                "characters) at the top level of every wrapper. Set task_complete=true only when "
                "the entire requested task is already complete in the CURRENT visual evidence: the "
                "correct bowl was grasped and transferred, is now resting on the plate after release, "
                "and the open gripper is clear of it. A planned placement, closed gripper, temporary "
                "hold, or approach is not completion. Keep task_complete=false until all requested "
                "work is visibly finished. completion_reason briefly states observable evidence, "
                "not private reasoning, and must be nonempty when task_complete=true; it may be "
                "empty when false. This is your visual judgment, not a benchmark success label. "
                "A true response stops execution immediately, before any actions from that response; "
                "do not declare completion in anticipation of its proposed actions. Still supply a "
                "valid controller payload with target_region=plate: either coarse_field with "
                "motion_mode=hold and gripper_command=open, or contact_direct with exactly five "
                "[0,0,0,0,0,0,-1] actions. Keep the inactive branch null. "
            )
        return instructions + guidance
    if controller_mode == "field":
        return shared + "Field control is fixed for this comparison; output no routing wrapper. " + field
    instructions = shared + "Direct control is fixed for this comparison; output no routing wrapper. " + direct
    instructions += guidance
    if stop_on_model_complete:
        instructions = instructions.replace("Output an object containing only actions:",
                                            "Output an object containing actions, task_complete, and completion_reason:")
        instructions += (
            "Include task_complete (boolean) and completion_reason (string, at most 500 characters) "
            "on every response. Set task_complete=true only when the entire requested task is already "
            "complete in the CURRENT visual evidence: the correct bowl was grasped and transferred, "
            "is now resting on the plate after release, and the open gripper is clear of it. A planned "
            "placement, closed gripper, temporary hold, or approach is not completion. Keep "
            "task_complete=false until all requested work is visibly finished. completion_reason "
            "briefly states observable evidence, not private reasoning, and must be nonempty when "
            "task_complete=true; it may be empty when false. This is your visual judgment, not a "
            "benchmark success label. A true response stops execution immediately, before any actions "
            "from that response; do not declare completion in anticipation of its proposed actions. "
            "When true, supply exactly five [0,0,0,0,0,0,-1] actions. "
        )
    return instructions


class ChatGPTPlanCompactClient(ChatGPTPlanHybridClient):
    """Current hybrid settings, with small model outputs and local grid expansion."""

    field_representation = COMPACT_FIELD_VERSION

    def __init__(self, config=None, *, controller_mode: str = "hybrid",
                 stop_on_model_complete: bool = False, execution_mode: str = "fixed",
                 feedback_history: str = "full", centered_placement_guidance: bool = False,
                 grasp_settle_guidance: bool = False, field_improvements: bool = False, **kwargs):
        compact_prompt_version(controller_mode, stop_on_model_complete=stop_on_model_complete,
                               execution_mode=execution_mode, feedback_history=feedback_history,
                               centered_placement_guidance=centered_placement_guidance,
                               grasp_settle_guidance=grasp_settle_guidance,
                               field_improvements=field_improvements)
        if field_improvements and kwargs.get("reasoning_effort", "high") != "medium":
            raise ValueError("Field improvements require explicit medium reasoning effort")
        self.controller_mode = controller_mode
        self.stop_on_model_complete = stop_on_model_complete
        self.execution_mode = execution_mode
        self.feedback_history = feedback_history
        self.centered_placement_guidance = centered_placement_guidance
        self.grasp_settle_guidance = grasp_settle_guidance
        self.field_improvements = field_improvements
        if field_improvements:
            self.field_representation = GROUNDED_FIELD_VERSION
        self._prediction_observation = None
        self.history_protocol = BLOCK_FEEDBACK_PROTOCOL if feedback_history == "block_summary" else HISTORY_PROTOCOL
        self.routing_is_model_prediction = controller_mode == "hybrid"
        self._reset_prediction_diagnostics()
        super().__init__(config, **kwargs)
        actual = (self.grid, self.chunk_size, self.field_update_steps, self.planner_horizon,
                  self.field_execution_steps)
        expected = (HYBRID_GRID, HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_PLANNER_HORIZON,
                    HYBRID_FIELD_EXECUTION_STEPS)
        if actual != expected or self.reasoning_effort not in ("medium", "high"):
            raise ValueError("Compact control requires grid33/5mm, direct5, field30, horizon10, execute10, medium or high effort")

    def _reset_prediction_diagnostics(self) -> None:
        self.last_field_expansion_s = 0.0
        self.last_compact_prediction = None
        self.last_effective_compact_field = None
        self.last_goal_grounding = None
        self.last_task_complete = False
        self.last_completion_reason = ""

    def reset_history(self) -> None:
        with self._history_lock:
            super().reset_history()
            self._reset_prediction_diagnostics()

    @property
    def prompt_version(self) -> str:
        return compact_prompt_version(self.controller_mode, stop_on_model_complete=self.stop_on_model_complete,
                                      execution_mode=self.execution_mode, feedback_history=self.feedback_history,
                                      centered_placement_guidance=self.centered_placement_guidance,
                                      grasp_settle_guidance=self.grasp_settle_guidance,
                                      field_improvements=self.field_improvements)

    def history_stats(self) -> dict:
        with self._history_lock:
            stats = super().history_stats()
            if self.feedback_history == "block_summary":
                stats.update(protocol=self.history_protocol, feedback_history=self.feedback_history,
                             text_history="all_observations_validated_outputs_completed_block_summaries_and_incomplete_feedback")
                stats["feedback_block_summaries"] = sum(
                    any(part.get("type") == "input_text" and f'"type":"{BLOCK_SUMMARY_TYPE}"' in part.get("text", "")
                        for part in message.get("content", []) if isinstance(part, dict))
                    for message in self._history if message.get("role") == "user")
            return stats

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict[str, Any]:
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = compact_instructions(
            self.controller_mode, self.image_history, stop_on_model_complete=self.stop_on_model_complete,
            execution_mode=self.execution_mode, feedback_history=self.feedback_history,
            centered_placement_guidance=self.centered_placement_guidance,
            grasp_settle_guidance=self.grasp_settle_guidance,
            field_improvements=self.field_improvements)
        if self.feedback_history == "block_summary":
            payload["input"] = compact_feedback_history(payload["input"], field_steps=self.field_update_steps,
                                                       direct_steps=self.chunk_size,
                                                       allow_field_events=self.field_improvements)
        suffix = "completion_v2" if self.stop_on_model_complete else "v1"
        if self.execution_mode != "fixed" or self.feedback_history != "full":
            suffix = f"{self.execution_mode}_{self.feedback_history}"
            suffix += "_completion_v3" if self.stop_on_model_complete else "_v3"
        if self.centered_placement_guidance:
            suffix = f"{self.execution_mode}_{self.feedback_history}_centered_placement_v4"
            if self.controller_mode == "hybrid":
                suffix = f"{self.execution_mode}_{self.feedback_history}_centered_v4"
        if self.grasp_settle_guidance:
            suffix = f"{self.execution_mode}_{self.feedback_history}_grasp_settle_v5"
        if self.field_improvements:
            suffix = f"{self.feedback_history}_grounded_v6"
        payload["text"]["format"] = {
            "type": "json_schema", "name": f"astra_compact_{self.controller_mode}_{suffix}", "strict": True,
            "schema": compact_output_schema(
                self.controller_mode, stop_on_model_complete=self.stop_on_model_complete,
                field_improvements=self.field_improvements,
                target_regions=getattr(self, "target_regions", HYBRID_TARGET_REGIONS)),
        }
        return payload

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
                    "prompt_version": self.prompt_version, "endpoint": self.config.endpoint,
                    "controller_mode": self.controller_mode, "field_representation": self.field_representation,
                    "history_protocol": self.history_protocol, "image_history": self.image_history}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def record_field_event(self, event: Mapping[str, Any]) -> None:
        """Retain why a field interval ended early; never count this as an action."""
        required = {"type", "step", "chunk_index", "reason"}
        if not self.field_improvements:
            raise ValueError("Field events require the improved field protocol")
        if not isinstance(event, Mapping) or set(event) not in (required, required | {"visual_check"}):
            raise ValueError("field event has missing or unexpected fields")
        if event["type"] != "field_control_event" or event["reason"] not in FIELD_EVENT_REASONS:
            raise ValueError("invalid field event type or reason")
        if any(type(event[key]) is not int or event[key] < 0 for key in ("step", "chunk_index")):
            raise ValueError("field event step and chunk_index must be nonnegative integers")
        record = {key: event[key] for key in required}
        visual = event.get("visual_check")
        if visual is not None:
            if not isinstance(visual, Mapping):
                raise ValueError("visual_check must be an object or null")
            safe = {}
            if "status" in visual:
                if not isinstance(visual["status"], str) or len(visual["status"]) > 80:
                    raise ValueError("visual_check.status must be a short string")
                safe["status"] = visual["status"]
            for key in ("current_distance_px", "baseline_distance_px"):
                if key in visual:
                    value = visual[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"visual_check.{key} must be finite and nonnegative")
                    safe[key] = float(value)
            record["visual_check"] = safe
        with self._history_lock:
            self._history.append({"role": "user", "content": [{"type": "input_text", "text": _canonical(record)}]})

    def validate_response(self, data: Any, grid: GridSpec) -> dict[str, Any]:
        self._reset_prediction_diagnostics()
        task_complete, completion_reason = False, ""
        data_without_completion = copy.deepcopy(data)
        if self.stop_on_model_complete:
            expected_keys = (HYBRID_KEYS if self.controller_mode == "hybrid" else {"actions"}) | COMPLETION_KEYS
            if not isinstance(data, dict) or set(data) != expected_keys:
                raise SchemaError("Compact completion output has missing or unexpected fields")
            task_complete = data_without_completion.pop("task_complete")
            completion_reason = data_without_completion.pop("completion_reason")
            if type(task_complete) is not bool:
                raise SchemaError("task_complete must be a boolean")
            if not isinstance(completion_reason, str) or len(completion_reason) > 500:
                raise SchemaError("completion_reason must be a string of at most 500 characters")
            if task_complete and not completion_reason.strip():
                raise SchemaError("A completed task requires a nonempty completion_reason")
        if self.controller_mode == "hybrid":
            if not isinstance(data_without_completion, dict) or set(data_without_completion) != HYBRID_KEYS:
                raise SchemaError("Compact hybrid output has missing or unexpected fields")
            wrapper = data_without_completion
        else:
            contact = self.controller_mode == "direct"
            if contact:
                validate_prediction(data_without_completion, grid, "direct")
            wrapper = {
                "controller": "contact_direct" if contact else "coarse_field",
                "target_region": "plate" if task_complete else "black_bowl", "near_target": contact,
                "routing_reason": f"Fixed {self.controller_mode} controller for timing comparison; routing was not predicted.",
                "field": None if contact else data_without_completion,
                "direct": data_without_completion if contact else None,
            }
        expansion_s = 0.0
        if wrapper["controller"] == "coarse_field":
            if self.field_improvements:
                if self._prediction_observation is None:
                    raise SchemaError("Grounded field validation requires its measured observation")
                compact, grounding = effective_compact_field(wrapper["field"], self._prediction_observation)
                self.last_goal_grounding = grounding
            else:
                compact = wrapper["field"]
                validate_compact_field(compact)
            self.last_effective_compact_field = copy.deepcopy(compact)
            started = time.perf_counter()
            wrapper["field"] = expand_compact_field(compact, grid)
            expansion_s = time.perf_counter() - started
        validate_hybrid_prediction(wrapper, grid, self.chunk_size,
            target_regions=getattr(self, "target_regions", HYBRID_TARGET_REGIONS))
        if task_complete:
            if wrapper["target_region"] != "plate":
                raise SchemaError("A completed bowl placement must target the plate")
            if wrapper["controller"] == "coarse_field":
                if (wrapper["field"]["motion_mode"] != "hold"
                        or wrapper["field"]["gripper_command"] != "open"):
                    raise SchemaError("A completed task requires a hold/open terminal field")
            elif any(action != [0, 0, 0, 0, 0, 0, -1] for action in wrapper["direct"]["actions"]):
                raise SchemaError("A completed task requires zero-motion/open terminal direct actions")
        self.last_compact_prediction = copy.deepcopy(data)
        self.last_field_expansion_s = expansion_s
        self.last_task_complete = task_complete
        self.last_completion_reason = completion_reason
        return wrapper

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "hybrid") -> dict[str, Any]:
        with self._history_lock:
            self._reset_prediction_diagnostics()
            try:
                if self.field_improvements:
                    self._prediction_observation = PolicyObservation.from_dict(observation)
                return super().predict(observation, grid, mode)
            except BaseException:
                self._reset_prediction_diagnostics()
                raise
            finally:
                self._prediction_observation = None
