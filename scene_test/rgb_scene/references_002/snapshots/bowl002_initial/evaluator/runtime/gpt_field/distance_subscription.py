"""Versioned subscription contract for locally updated distance-field control.

The client only validates model intent and records observable boundary feedback.
Distance solving, visual calibration, field reuse, and execution live in the
policy. No API-key path, simulator state, or completion oracle is introduced.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .astra import SchemaError, _canonical, _numbers
from .block_feedback_history import _text_object, summarize_feedback_block
from .compact_subscription import (
    ChatGPTPlanCompactClient, compact_operational_guidance, compact_output_schema,
)
from .hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_KEYS
from .distance_tasks import resolve_distance_task
from .demonstration import DemoReference
from .schema import GridSpec


DISTANCE_FIELD_VERSION = "goal-distance-feedback-v1"
DISTANCE_PROMPT_VERSION = "astra-distance-field-local-feedback-v7"
CONTROL_PHASES = ("approach", "grasp", "transport", "align", "release", "verify", "recover")
DISTANCE_EVENT_REASONS = frozenset({
    "goal_reached", "field_boundary_reached", "invalid_direction", "no_progress",
    "visual_unavailable", "visual_divergence", "placement_unverified", "placement_misaligned",
    "placement_observation_consistent", "field_budget_exhausted", "local_goal_outside_field",
})
ALIGNMENT_ASSUMPTION = "same_rigid_point_desired_projection"
MAX_FIELD_REUSE_ACTIONS = 120


ZERO_SHOT_DISTANCE_PROMPT_VERSION = "astra-distance-field-bowl-plate-zero-shot-collision-free-v1"
ZERO_SHOT_DISTANCE_SCHEMA_NAME = "astra_distance_field_bowl_plate_zero_shot_collision_free_v1"


def _validate_zero_shot_mode(task_key: str, zero_shot_collision_free: bool,
                            demonstration: bool = False) -> None:
    if type(zero_shot_collision_free) is not bool:
        raise ValueError("zero_shot_collision_free must be a boolean")
    if zero_shot_collision_free:
        if demonstration:
            raise ValueError("Zero-shot collision-free control cannot receive a demonstration")
        if resolve_distance_task(task_key).key != "bowl_plate":
            raise ValueError("Zero-shot collision-free control requires task_key='bowl_plate'")


def distance_prompt_version(task_key: str = "bowl_plate", *, demonstration: bool = False,
                            zero_shot_collision_free: bool = False) -> str:
    if type(demonstration) is not bool:
        raise ValueError("demonstration must be a boolean")
    _validate_zero_shot_mode(task_key, zero_shot_collision_free, demonstration)
    if zero_shot_collision_free:
        return ZERO_SHOT_DISTANCE_PROMPT_VERSION
    if demonstration:
        suffix = "drawer" if resolve_distance_task(task_key).key == "drawer_bowl" else "bowl-plate"
        return f"astra-distance-field-local-feedback-{suffix}-one-demo-v1"
    return (DISTANCE_PROMPT_VERSION if resolve_distance_task(task_key).key == "bowl_plate"
            else "astra-distance-field-local-feedback-drawer-v1")


def distance_field_version(task_key: str = "bowl_plate") -> str:
    return (DISTANCE_FIELD_VERSION if resolve_distance_task(task_key).key == "bowl_plate"
            else "goal-distance-feedback-drawer-v1")


def distance_schema_name(task_key: str = "bowl_plate", *, zero_shot_collision_free: bool = False) -> str:
    _validate_zero_shot_mode(task_key, zero_shot_collision_free)
    if zero_shot_collision_free:
        return ZERO_SHOT_DISTANCE_SCHEMA_NAME
    return ("astra_distance_field_feedback_v7" if resolve_distance_task(task_key).key == "bowl_plate"
            else "astra_distance_field_feedback_drawer_v1")


def control_context_schema(*, task_key: str = "bowl_plate") -> dict[str, Any]:
    task = resolve_distance_task(task_key)
    uv = {"type": "array", "minItems": 2, "maxItems": 2,
          "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    reference = {"type": "object", "additionalProperties": False,
        "required": ["kind", "moving_point_uv", "target_point_uv", "alignment_assumption"],
        "properties": {
            "kind": {"type": "string", "enum": ["end_effector", "held_object"]},
            "moving_point_uv": copy.deepcopy(uv), "target_point_uv": copy.deepcopy(uv),
            "alignment_assumption": {"type": "string", "enum": [ALIGNMENT_ASSUMPTION]},
        }}
    return {"type": "object", "additionalProperties": False,
        "required": ["phase", "visual_reference"],
        "properties": {"phase": {"type": "string", "enum": list(task.control_phases)},
                       "visual_reference": {"anyOf": [reference, {"type": "null"}]}}}


def validate_control_context(value: Any, *, task_key: str = "bowl_plate") -> dict:
    task = resolve_distance_task(task_key)
    if not isinstance(value, dict) or set(value) != {"phase", "visual_reference"}:
        raise SchemaError("control_context has missing or unexpected fields")
    if value["phase"] not in task.control_phases:
        raise SchemaError("control_context.phase is not a supported explicit phase")
    reference = value["visual_reference"]
    if reference is None:
        return value
    keys = {"kind", "moving_point_uv", "target_point_uv", "alignment_assumption"}
    if not isinstance(reference, dict) or set(reference) != keys:
        raise SchemaError("visual_reference has missing or unexpected fields")
    if reference["kind"] not in ("end_effector", "held_object"):
        raise SchemaError("visual_reference.kind must identify the end effector or held object")
    if reference["alignment_assumption"] != ALIGNMENT_ASSUMPTION:
        raise SchemaError("visual_reference requires the same rigid-point projection assumption")
    for key in ("moving_point_uv", "target_point_uv"):
        uv = reference[key]
        if (not isinstance(uv, list) or len(uv) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0. <= v <= 1. for v in uv)):
            raise SchemaError(f"visual_reference.{key} must contain two finite normalized image coordinates")
    return value


def distance_output_schema(*, task_key: str = "bowl_plate") -> dict[str, Any]:
    task = resolve_distance_task(task_key)
    schema = compact_output_schema("hybrid", field_improvements=True, target_regions=task.target_regions)
    schema["properties"]["control_context"] = control_context_schema(task_key=task_key)
    schema["required"].append("control_context")
    return schema


def distance_instructions(image_history: int = 2, *, task_key: str = "bowl_plate",
                          demonstration: bool = False, zero_shot_collision_free: bool = False) -> str:
    task = resolve_distance_task(task_key)
    version = distance_prompt_version(task_key, demonstration=demonstration,
                                      zero_shot_collision_free=zero_shot_collision_free)
    if type(image_history) is not int or image_history < 1:
        raise ValueError("image_history must be a positive integer")
    if zero_shot_collision_free:
        common = shared_zero_shot_task_guidance(image_history, task_key=task.key)
        instructions = _bowl_distance_instructions(image_history).replace(DISTANCE_PROMPT_VERSION, version, 1)
        return (instructions + " " + common["task_and_zero_shot"] + common["collision_free"]
                + hybrid_collision_field_guidance())
    if task.key == "bowl_plate":
        if not demonstration:
            return _bowl_distance_instructions(image_history)
        instructions = _bowl_distance_instructions(image_history).replace(DISTANCE_PROMPT_VERSION, version, 1)
        instructions = instructions.replace(
            "Control the bowl task using only task text, two current RGB camera images, measured robot "
            "state and retained observable feedback.",
            f"Task: {task.prompt}. Control this task using the single external reference demonstration, "
            "task text, two current RGB camera images, measured robot state and retained observable feedback.", 1)
        instructions = instructions.replace("demonstrations or future images are supplied.",
            "or future images of the current evaluation are supplied.", 1)
        return instructions + (
            " The first input message is exactly one independent LIBERO reference demonstration, "
            "explicitly labeled external_demonstration. It supplies the complete recorded normalized "
            "action and measured robot-state sequence, plus a sampled chronological set of paired RGB "
            "keyframes. Each after_state and keyframe is AFTER its labeled action_index; it is not "
            "the observation preceding that action. The demonstration is a separate episode, never "
            "an executed action, observation, success signal or future state of this evaluation. "
            "Study its approach orientation, finger alignment, bowl grasp and test lift, free-space "
            "transport, centered placement on the plate, release and open-gripper retreat. Transfer "
            "these relationships to the CURRENT scene and measured pose; do not blindly replay "
            "absolute demonstration coordinates or its action timing. The reference stays fixed "
            "and available on every request, separately from the latest two live observation image "
            "pairs. It cannot certify that the current task is complete. Continue to choose the "
            "normal Field/Direct schema from the current observation and measured execution feedback. "
            "No reference action is executed automatically."
        )
    instructions = _drawer_distance_instructions(image_history)
    if not demonstration:
        return instructions
    instructions = instructions.replace(distance_prompt_version(task_key), version, 1)
    instructions = instructions.replace(
        "Control this task using only task text, two current RGB camera images, measured robot state "
        "and retained observable feedback.",
        "Control this task using the single external reference demonstration, task text, two current "
        "RGB camera images, measured robot state and retained observable feedback.", 1)
    instructions = instructions.replace("demonstrations or future images are supplied.",
                                        "or future images of the current evaluation are supplied.", 1)
    instructions = instructions.replace("After grasping the handle, use small deliberate pulls",
        "After establishing visually supported handle engagement, use small deliberate pulls", 1)
    return instructions + (
        " The first input message is exactly one independent LIBERO reference demonstration, "
        "explicitly labeled external_demonstration. It supplies the complete recorded normalized "
        "action and measured robot-state sequence, plus a sampled chronological set of paired RGB "
        "keyframes. Each after_state and keyframe is AFTER its labeled action_index; it is not "
        "the observation preceding that action. The demonstration is a separate episode, never "
        "an executed action, observation, success signal or future state of this evaluation. "
        "Study its approach orientation, finger alignment, handle grasp and pull, handle release, "
        "bowl grasp and insertion. Do not impose a closed-finger handle grasp if the reference "
        "shows a different contact strategy; infer engagement from the reference and current RGB "
        "response. Transfer these relationships to the CURRENT scene and measured "
        "pose; do not blindly replay absolute demonstration coordinates or its action timing. "
        "The reference stays fixed and available on every request, separately from the latest "
        "two live observation image pairs. It cannot certify that the current task is complete. "
        "Continue to choose the normal Field/Direct schema from the current observation and "
        "measured execution feedback. No reference action is executed automatically."
    )


def collision_free_guidance() -> str:
    """Exact observable collision objective shared by the two zero-shot arms."""
    return (
        "The full robot, including the arm and gripper fingers, and the held bowl must avoid all "
        "unintended scene contacts throughout the task. This includes the ketchup bottle, milk "
        "carton, ramekin, cabinet, stove, table and any other visible obstacle. Allow only necessary "
        "finger-bowl grasp contact and bowl-table or bowl-plate support contact; these exceptions "
        "do not permit the arm, fingers or carried bowl to strike other objects. Do not push, tip "
        "or shift obstacles, and do not use unintended contact to explore or force a passage. "
        "Plan clearance for the swept volume of the arm, fingers and held bowl, including grasp "
        "offset, payload extent and changes in orientation, not only the end-effector point. "
        "Estimate clearance conservatively from the current RGB views and observed motion; no "
        "ground-truth object poses, collision geometry, contact oracle or hidden scene state is "
        "available. Occluded or uncertain space is not free space. If the visible evidence does "
        "not support a clear motion, hold or choose a visibly clear reobservation, then replan "
        "before advancing; do not force through or assume that a short action is safe. Visible "
        "obstacle tilt or displacement is an unintended disturbance, even if the bowl reaches "
        "the plate. Phase labels and model confidence cannot certify collision-free execution. "
    )


def hybrid_collision_field_guidance() -> str:
    return (
        "For each coarse_field response, inspect the intended local route and represent relevant "
        "visible blockers conservatively in blocked_boxes; represent occluded or uncertain "
        "regions in unknown_boxes. Add repulsors when useful for extra clearance, but a soft "
        "repulsor does not replace a blocked solid obstacle. Boxes describe forbidden or unknown "
        "end-effector positions after accounting for the visually estimated held-bowl and finger "
        "extent, grasp offset and rotational swept volume. Inflate the obstacle estimate for "
        "these extents and uncertainty within the current local grid. Do not return all-empty "
        "blocked_boxes, unknown_boxes and repulsors when a relevant obstacle or its required "
        "clearance margin intersects the proposed route. Empty constraints are justified only "
        "when the current views support clearance over the proposed local swept volume. Do "
        "not erase a blocker to obtain a path. If conservative constraints leave no supported "
        "path, request a hold, a nearer safe goal or reobservation/replanning. The local grid "
        "and mask checks are not full-arm collision certification, so also avoid visually "
        "threatened arm configurations. No hidden obstacle coordinates may be assumed. "
    )


def shared_zero_shot_task_guidance(image_history: int = 2, *, task_key: str = "bowl_plate") -> dict[str, str]:
    """Keep task, grasp and placement spans identical without any demo advice."""
    _validate_zero_shot_mode(task_key, True)
    if type(image_history) is not int or image_history != 2:
        raise ValueError("Matched zero-shot control requires the latest two live image pairs")
    source = _bowl_distance_instructions(image_history)

    def span(start: str, end: str | None) -> str:
        if source.count(start) != 1 or (end is not None and source.count(end) != 1):
            raise ValueError("Shared zero-shot guidance anchors are no longer unique")
        first, last = source.index(start), source.index(end) if end is not None else len(source)
        if first >= last:
            raise ValueError("Shared zero-shot guidance anchors are out of order")
        return source[first:last]

    task = resolve_distance_task(task_key)
    return {
        "task_and_zero_shot": (
            f"Task: {task.prompt}. This is zero-shot control: no demonstration, reference trajectory "
            "or prerecorded action sequence is supplied. Choose actions using only the task text, "
            "current paired RGB, measured robot state and retained live observable feedback/history. "),
        "allowed_observation": span("Control the bowl task using only ", "Keep the latest "),
        "scene_clearance": span("These are visual estimates, ", "The follower tracks a local path "),
        "grasp_settling": span("Actions execute at 20 Hz, so five actions advance only ", None),
        "placement": span("Place the bowl centered within the plate rim. ", "Actions execute at 20 Hz, so five actions advance only "),
        "reobservation": span("Missing, ambiguous or lost visual evidence means ", "After release, verify "),
        "post_release_verification": span("After release, verify ", "A verify/hold interval still requests "),
        "recovery_and_stopping": span("Do not declare the task complete from a phase name, ", "Place the bowl centered within the plate rim. "),
        "collision_free": collision_free_guidance(),
    }


def _bowl_distance_instructions(image_history: int = 2) -> str:
    """State this protocol directly, without obsolete fixed-cadence claims."""
    if type(image_history) is not int or image_history < 1:
        raise ValueError("image_history must be a positive integer")
    return (
        f"Interface {DISTANCE_PROMPT_VERSION}; field representation {DISTANCE_FIELD_VERSION}. "
        "Control the bowl task using only task text, two current RGB camera images, measured robot "
        "state and retained observable feedback. No object poses, depth, contact sensors, reward, "
        "benchmark success labels, demonstrations or future images are supplied. State is "
        "[end-effector xyz meters, orientation axis-angle xyz radians, two gripper joint positions], "
        "not robot joint angles. All spatial offsets and rotation increments use the world frame. "
        "Normalized actions are [dx,dy,dz,rx,ry,rz,gripper]: translation 1 means 0.05 m, rotation 1 "
        "means 0.5 radians, +1 gripper closes and -1 opens. Commands start from each actual measured "
        "pose and do not guarantee realized motion. A gripper command or low robot-motion error "
        "does not prove object grasp, contact, placement or task completion. "
        f"Keep the latest {image_history} observation image pairs and earlier textual observations, "
        "raw accepted output items including opaque reasoning, execution_block_summary records "
        "and distance_control_event records. Full per-action observations and feedback remain in "
        "local logs; summaries describe measured robot motion rather than hidden object state. "
        "Historical offsets remain relative to their own generation observation. "
        "Return strict JSON with controller, target_region, near_target, routing_reason, field, "
        "direct and control_context. Select coarse_field for free-space motion or contact_direct "
        "for contact alignment, grasp, release and recovery. near_target is true exactly for "
        "contact_direct. target_region is black_bowl or plate. Set the inactive branch to null; "
        "routing_reason briefly cites current observable evidence, not private reasoning. "
        "For coarse_field, output stage, a unit orientation_target_xyzw quaternion, gripper_command "
        "open/close/hold, motion_mode move/hold, goal_offset_m, attraction_weights, repulsors, "
        "blocked_boxes, unknown_boxes and goal_reference. Return compact parameters, never dense "
        "values. The fixed-center 33x33x33 grid uses 0.005 m spacing and +/-0.08 m coverage per axis. "
        "Prefer an effective goal inside the known cube, normally within +/-0.075 m per axis. "
        "A goal outside the known cube requires a new observation or nearer model goal; it is not "
        "silently clipped to a pseudo-goal or solved beyond the observed domain. "
        "The local planner computes shortest accumulated motion cost on a 26-neighbor grid using "
        "Dijkstra. Positive Gaussian repulsors define traversal density 1 plus the repulsor sum; "
        "edge cost is Euclidean edge length times the average endpoint density. attraction_weights "
        "remain in the compact schema for compatibility but do not change this distance cost. "
        "Blocked and unknown regions cannot be traversed. This cost-to-go is a discretized local "
        "distance, not an SDF or a measured distance to object surfaces. Provide at most four "
        "repulsors with center_offset_m, radius_m and weight, and at most four blocked and four "
        "unknown axis-aligned boxes with lower_offset_m and upper_offset_m. These are visual "
        "estimates, not certified full-arm collision geometry. Unknown does not mean free space. "
        "The follower tracks a local path from each actual measured pose, using up to 3 cm "
        "lookahead, a 5 cm translation norm bound and predicted-mask chord checks. Geometric "
        "waypoints and measured control actions are distinct. A local path is reconsidered at "
        "most every ten executed actions and may end sooner on measured arrival. Position "
        "arrival uses a 5 mm tolerance and orientation arrival uses 0.05 radians; local orientation "
        "correction has a 0.15-radian command cap. The initial Field interval is 30 actions. It may "
        "reuse the same field beyond 30 only while reliable local visual tracking, measured "
        "progress and field validity support reuse, up to 120 total actions for that model "
        "response. This is a model-refresh bound, not an episode timeout or task-completion rule. "
        "Boundary arrival, invalid motion, loss of reliable feedback, lack of progress or "
        "placement-verification concerns can request a fresh model decision earlier. "
        "goal_reference is null or has kind, object_center_offset_m, desired_object_center_offset_m "
        "and visual_anchor. For kind=end_effector, both object offsets are null and goal_offset_m "
        "is the end-effector target. For kind=held_object, estimate the current held bowl-center "
        "offset and desired bowl-center offset from both RGB views, each in world meters relative "
        "to the current end effector. Local code rotates the current offset by R_target R_current.T "
        "then subtracts it from the desired object-center offset to obtain an effective end-effector "
        "goal. This assumes a rigid grasp and cannot verify one. Preserve raw goal_offset_m for "
        "audit. visual_anchor, when supplied, uses the fixed external camera and must be consistent "
        "with control_context.visual_reference. Do not invent an anchor if a suitable point is not "
        "visible. Gripper hold repeats the last nonzero intent across both controllers, initially "
        "open. A move/hold field never certifies task completion. "
        "For contact_direct, output exactly five consecutive actions with seven finite normalized "
        "values in [-1,1]. All five actions execute unchanged: no local field correction, clipping "
        "or forced closure. Local visual feedback may observe these actions, but does not truncate "
        "the five-action chunk. Direct gripper zero does not change remembered nonzero intent. "
        "control_context always includes phase and visual_reference. phase is one of approach, "
        "grasp, transport, align, release, verify or recover; use this explicit label instead of "
        "encoding a controller phase in free text. visual_reference is null or includes kind "
        "end_effector/held_object, moving_point_uv, target_point_uv and alignment_assumption="
        "same_rigid_point_desired_projection. Both [u,v] vectors are normalized to [0,1] in the "
        "FIXED EXTERNAL observation/image, origin top-left, u right and v down. The moving point "
        "must be visibly textured and rigidly attached to the chosen end effector or held object. "
        "The target is the desired projection of that SAME rigid point, accounting for grasp "
        "offset, chosen height and orientation; it is not automatically the plate center. Never "
        "substitute wrist-camera coordinates or unobservable object geometry. Local bounded XY "
        "goal correction is allowed only for planar align with this assumption and a valid "
        "calibration measured from executed motions; it does not infer depth or change Direct "
        "actions. Missing, ambiguous or lost visual evidence means reobserve/recover with a "
        "Direct five-action decision when needed, not fabricate anchors or assume alignment. "
        "After release, verify centering, apparent support and bowl/gripper separation from both "
        "views. Local placement feedback is only 2D observational evidence; it cannot certify "
        "LIBERO success, support height or containment. A verify/hold interval still requests "
        "local feedback and recovery decisions when evidence is insufficient or misaligned. "
        "After five Field hold actions in verify, or in a released plate-target hold, local "
        "verification may override the effective hold once with a bounded +Z reobservation move "
        "of at most 2 cm inside the same predicted-valid cube, with the gripper open. This move "
        "lasts at most ten control actions before a placement_unverified or placement_misaligned "
        "boundary decision. It may occur only once per release/recovery epoch, never repeated "
        "climbing on successive hold responses, and never overrides a Direct action. Raw model "
        "hold intent and the distinct local reobservation intent remain separately logged. "
        "Do not declare the task complete from a phase name, hold command or feedback status. "
        "There is no task_complete output in this protocol; only the external evaluator can stop "
        "on actual LIBERO success, with no episode action or time cap. "
        + compact_operational_guidance(centered_placement_guidance=True, grasp_settle_guidance=True)
    )


def _drawer_distance_instructions(image_history: int) -> str:
    return (
        f"Interface {distance_prompt_version('drawer_bowl')}; field representation "
        f"{distance_field_version('drawer_bowl')}. Task: open the top drawer and put the bowl inside. "
        "Control this task using only task text, two current RGB camera images, measured robot state "
        "and retained observable feedback. No object poses, depth, contact sensors, drawer joint "
        "positions, known handle coordinates, camera calibration, rewards, benchmark success labels, "
        "demonstrations or future images are supplied. State is [end-effector xyz meters, orientation "
        "axis-angle xyz radians, two gripper joint positions], not robot joint angles. Spatial "
        "offsets and rotation increments use the world frame. Normalized actions are "
        "[dx,dy,dz,rx,ry,rz,gripper]: translation 1 means 0.05 m, rotation 1 means 0.5 radians, "
        "+1 gripper closes and -1 opens. Commands start from each actual measured pose; they do "
        "not prove realized movement, a successful grasp, drawer opening, placement or completion. "
        f"Keep the latest {image_history} observation image pairs and earlier textual observations, "
        "raw accepted output items including opaque reasoning, execution_block_summary records "
        "and distance_control_event records. Full per-action observations and feedback remain in "
        "local logs. Historical offsets remain relative to their own generation observation. "
        "Return strict JSON with controller, target_region, near_target, routing_reason, field, "
        "direct and control_context. target_region is drawer_handle, black_bowl or drawer_interior. "
        "near_target is true exactly for contact_direct. Set the inactive branch to null. "
        "routing_reason briefly cites observable evidence, not private reasoning. Use coarse_field "
        "for free-space approach to the top drawer handle, free-space bowl approach, and transport "
        "toward an already visibly open drawer. Use contact_direct for interaction with the handle, "
        "bowl grasp, constrained insertion, release and recovery. A drawer handle is attached to "
        "an articulated drawer; never treat it as a freely carried object. "
        "control_context always includes phase and visual_reference. phase is approach, grasp, "
        "transport, align, release, verify, recover, handle_grasp, drawer_open or handle_release. "
        "Use handle_grasp, drawer_open and handle_release only with target_region=drawer_handle "
        "and controller=contact_direct. Handle approach may use a free-space Field; actual handle "
        "grasp, pull and release always use exactly five Direct actions. Infer which handle belongs "
        "to the top drawer and the pull direction from the two RGB views and observed response. "
        "Do not assume a fixed pull axis, handle location, joint target, precomputed opening "
        "trajectory or drawer state. After grasping the handle, use small deliberate pulls and "
        "reobserve drawer-front motion; release and adjust from visual evidence if the handle "
        "slips or the drawer fails to move. A handle grasp does NOT call for a vertical test lift. "
        "Once the top drawer is visibly open enough for the bowl, release the handle and move "
        "clear before approaching the bowl. Use phase=grasp with target_region=black_bowl for "
        "the bowl grasp. Use phase=release with target_region=drawer_interior for placing the bowl. "
        "Opening the drawer does not complete the task: the bowl must be placed inside its "
        "visible interior. Closing the drawer is not required. Do not add a drawer-closing stage. "
        "For coarse_field, output stage, unit orientation_target_xyzw, gripper_command open/close/hold, "
        "motion_mode move/hold, goal_offset_m, attraction_weights, repulsors, blocked_boxes, "
        "unknown_boxes and goal_reference. Output compact parameters, never dense values. The "
        "fixed-center 33x33x33 grid has 0.005 m spacing and +/-0.08 m coverage per axis. Prefer an "
        "effective goal inside the known cube, normally within +/-0.075 m per axis. An out-of-cube "
        "goal requires a new observation or nearer model goal; it is not clipped to a pseudo-goal. "
        "The planner computes shortest accumulated motion cost on a 26-neighbor grid using Dijkstra. "
        "Positive Gaussian repulsors define traversal density 1 plus the repulsor sum; edge cost "
        "is Euclidean edge length times average endpoint density. attraction_weights remain for "
        "schema compatibility but do not change this distance cost. Blocked and unknown regions "
        "cannot be traversed. This is a discretized local distance, not an SDF or measured distance "
        "to object surfaces. Supply at most four repulsors (center_offset_m, radius_m, weight), "
        "four blocked boxes and four unknown boxes (lower_offset_m, upper_offset_m). Geometry is "
        "visually estimated, not certified full-arm collision geometry. Unknown does not mean "
        "free space. Mark drawer front, side walls, upper surfaces and uncertain interior "
        "clearance conservatively from RGB; do not infer free clearance from an occluded view. "
        "The follower tracks from each actual measured pose, with up to 3 cm lookahead, a 5 cm "
        "translation norm bound and predicted-mask chord checks. It reconsiders the path at most "
        "every ten executed actions and may end earlier on arrival. Geometric waypoints and "
        "control actions differ. Arrival uses 5 mm position and 0.05 radians orientation tolerances; "
        "the orientation command cap is 0.15 radians. The initial Field interval is 30 actions. "
        "It can extend to at most 120 total actions per model response only with reliable local "
        "visual tracking, measured progress and valid local geometry. Boundary arrival, invalid "
        "motion, lost feedback, lack of progress or verification concerns request a model "
        "decision earlier. This bounds model refresh, not the episode or success criterion. "
        "goal_reference is null or contains kind, object_center_offset_m, "
        "desired_object_center_offset_m and visual_anchor. For end_effector, object offsets are "
        "null and goal_offset_m is the end-effector target. held_object refers only to the grasped "
        "bowl, not the drawer handle: estimate its current and desired center offsets relative to "
        "the current end effector from RGB. Local code rotates the current offset by "
        "R_target R_current.T and subtracts it from the desired offset. This assumes a rigid "
        "grasp without verifying one. Preserve raw goal_offset_m for audit. A supplied visual_anchor "
        "uses the fixed external view and must agree with control_context.visual_reference. "
        "visual_reference is null or contains kind=end_effector/held_object, moving_point_uv, "
        "target_point_uv and alignment_assumption=same_rigid_point_desired_projection. UV values "
        "are normalized in [0,1] in FIXED EXTERNAL observation/image, origin top-left, u right, "
        "v down. The moving patch must be visibly textured and rigidly attached to the chosen "
        "point. The target is the desired projection of that SAME point at the chosen height "
        "and orientation, accounting for grasp offset; it is not automatically an interior "
        "center or a handle center. Both patches must be visible and distinguishable. Null, "
        "textureless, overlapping or ambiguous anchors are unavailable, not aligned. Local XY "
        "goal correction is allowed only in phase=align with a valid calibration from measured "
        "motions and unchanged working height/orientation; it cannot infer depth or change Direct "
        "actions. Do not fabricate image anchors or metric geometry. "
        "For contact_direct output exactly five consecutive actions, each seven finite normalized "
        "values in [-1,1]. All five execute unchanged: no local correction, clipping, forced "
        "closure or early truncation. Gripper hold repeats the last executed nonzero intent, "
        "initially open; Direct gripper zero does not change that remembered intent. Actions "
        "execute at 20 Hz, so five actions advance 0.25 seconds; waiting for the model does not "
        "advance physics. For either grasp, hold pose while closing until observed finger joints "
        "settle; a close command alone is not a grasp. For the BOWL grasp only, perform a short "
        "test lift and check both RGB views for the bowl following the gripper. For the HANDLE "
        "grasp, check response to a small visually inferred drawer-opening pull instead. "
        "Before bowl release, check in both views that the bowl is inside the visible top drawer "
        "opening and has plausible bottom support, with clearance from edges. Use Direct for "
        "constrained lowering/insertion and release. After release, reobserve bowl stability, "
        "apparent support and separation from the gripper. A hold/verify interval requests another "
        "decision after five Field actions when placement remains unverified or misaligned. "
        "Automatic upward reobservation is disabled for this drawer task: no local +Z recovery "
        "move overrides a hold, since drawer surfaces can obstruct vertical motion. Choose any "
        "reobservation or recovery movement explicitly with a five-action Direct response based "
        "on visible clearance. No local check certifies 3D containment, contact or LIBERO success. "
        "There is no task_complete output. Only the external evaluator can stop on actual LIBERO "
        "success, with no episode action or time cap."
    )


def validate_control_event(event: Mapping[str, Any]) -> dict:
    required = {"type", "step", "chunk_index", "reason"}
    if not isinstance(event, Mapping) or set(event) not in (required, required | {"feedback"}):
        raise ValueError("distance control event has missing or unexpected fields")
    if (event["type"] != "distance_control_event" or not isinstance(event["reason"], str)
            or event["reason"] not in DISTANCE_EVENT_REASONS):
        raise ValueError("distance control event has an unsupported type or reason")
    if any(type(event[key]) is not int or event[key] < 0 for key in ("step", "chunk_index")):
        raise ValueError("distance control event indexes must be nonnegative integers")
    result = {key: event[key] for key in required}
    if "feedback" in event:
        feedback = event["feedback"]
        allowed = {"status", "reliable", "calibration_valid", "pixel_error_norm", "local_goal_updates",
                   "task_success_certified"}
        if not isinstance(feedback, Mapping) or set(feedback) - allowed:
            raise ValueError("distance feedback has non-allowlisted fields")
        safe = {}
        for key, value in feedback.items():
            if key == "status" and (not isinstance(value, str) or len(value) > 80):
                raise ValueError("distance feedback status must be a short string")
            if key in ("reliable", "calibration_valid") and type(value) is not bool:
                raise ValueError(f"distance feedback {key} must be a boolean")
            if key == "task_success_certified" and value is not False:
                raise ValueError("local feedback cannot certify task success")
            if key == "local_goal_updates" and (type(value) is not int or value < 0):
                raise ValueError("local_goal_updates must be a nonnegative integer")
            if key == "pixel_error_norm" and value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("pixel_error_norm must be finite and nonnegative or null")
                value = float(value)
            safe[key] = value
        result["feedback"] = safe
    return result


def compact_distance_feedback_history(messages: list[dict]) -> list[dict]:
    """Summarize only contiguous blocks closed by an explicit control event."""
    result, controller, index = [], None, 0
    while index < len(messages):
        message = messages[index]
        value = _text_object(message)
        if message.get("role") == "assistant" and value is not None:
            if value.get("controller") in ("coarse_field", "contact_direct"):
                controller = value["controller"]
        if message.get("role") != "user" or value is None or value.get("type") != "execution_feedback":
            result.append(copy.deepcopy(message))
            index += 1
            continue
        end, records = index, []
        while end < len(messages):
            item = messages[end]
            record = _text_object(item) if item.get("role") == "user" else None
            if (record is None or record.get("type") != "execution_feedback"
                    or record.get("chunk_index") != value.get("chunk_index")):
                break
            records.append(record)
            end += 1
        closing = (_text_object(messages[end]) if end < len(messages)
                   and messages[end].get("role") == "user" else None)
        try:
            closing = validate_control_event(closing)
        except (TypeError, ValueError):
            closing = None
        count = len(records)
        complete = (closing is not None and type(records[0].get("step")) is int
                    and type(records[0].get("chunk_index")) is int
                    and closing["chunk_index"] == records[0]["chunk_index"]
                    and closing["step"] == records[0]["step"] + count
                    and [r.get("step") for r in records] == list(range(records[0]["step"], closing["step"]))
                    and [r.get("chunk_step") for r in records] == list(range(count))
                    and ((controller == "coarse_field" and 0 < count <= MAX_FIELD_REUSE_ACTIONS)
                         or (controller == "contact_direct" and count == 5)))
        if complete:
            try:
                summary = summarize_feedback_block(records)
                replacement = {"role": "user", "content": [{"type": "input_text", "text": _canonical(summary)}]}
            except (KeyError, TypeError, ValueError, OverflowError):
                complete = False
        if complete:
            result.append(replacement)
        else:
            result.extend(copy.deepcopy(messages[index:end]))
        index = end
    return result


class ChatGPTPlanDistanceClient(ChatGPTPlanCompactClient):
    """Opt-in Astra medium subscription client; all execution stays local."""

    field_representation = DISTANCE_FIELD_VERSION

    def __init__(self, config=None, *, distance_field_control: bool = True,
                 controller_mode: str = "hybrid", reasoning_effort: str = "medium",
                 field_improvements: bool = True, execution_mode: str = "tracking",
                 feedback_history: str = "block_summary", centered_placement_guidance: bool = True,
                 grasp_settle_guidance: bool = True, stop_on_model_complete: bool = False,
                 task_key: str = "bowl_plate", demonstration: DemoReference | None = None,
                 zero_shot_collision_free: bool = False, **kwargs):
        self.task_spec = resolve_distance_task(task_key)
        _validate_zero_shot_mode(task_key, zero_shot_collision_free, demonstration is not None)
        self.zero_shot_collision_free = zero_shot_collision_free
        if zero_shot_collision_free:
            shared_zero_shot_task_guidance(kwargs.get("image_history", 2), task_key=task_key)
            if config is not None and config.model != "gpt-6-astra":
                raise ValueError("Zero-shot collision-free control requires model='gpt-6-astra'")
            if type(kwargs.get("chunk_size", 5)) is not int or kwargs.get("chunk_size", 5) != 5:
                raise ValueError("Zero-shot collision-free control requires chunk_size=5")
        if demonstration is not None:
            if not isinstance(demonstration, DemoReference):
                raise TypeError("demonstration must be a validated DemoReference")
            if demonstration.metadata()["task_key"] != self.task_spec.key:
                raise ValueError("Demonstration must match the selected task")
        self.demonstration = demonstration
        self.target_regions = self.task_spec.target_regions
        if "task_prompt" in kwargs and kwargs["task_prompt"] != self.task_spec.prompt:
            raise ValueError("Distance task_prompt must match the selected task contract")
        kwargs["task_prompt"] = self.task_spec.prompt
        required = {"distance_field_control": (distance_field_control, True),
            "controller_mode": (controller_mode, "hybrid"), "reasoning_effort": (reasoning_effort, "medium"),
            "field_improvements": (field_improvements, True), "execution_mode": (execution_mode, "tracking"),
            "feedback_history": (feedback_history, "block_summary"),
            "centered_placement_guidance": (centered_placement_guidance, True),
            "grasp_settle_guidance": (grasp_settle_guidance, True),
            "stop_on_model_complete": (stop_on_model_complete, False)}
        for name, (value, expected) in required.items():
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"Distance-field control requires {name}={expected!r}")
        self.distance_field_control = True
        self.last_control_context = None
        super().__init__(config, controller_mode=controller_mode, reasoning_effort=reasoning_effort,
            field_improvements=field_improvements, execution_mode=execution_mode, feedback_history=feedback_history,
            centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
            stop_on_model_complete=stop_on_model_complete, **kwargs)
        self.field_representation = distance_field_version(self.task_spec.key)

    @property
    def prompt_version(self) -> str:
        return distance_prompt_version(self.task_spec.key, demonstration=self.demonstration is not None,
                                       zero_shot_collision_free=self.zero_shot_collision_free)

    def _reset_prediction_diagnostics(self) -> None:
        super()._reset_prediction_diagnostics()
        self.last_control_context = None

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict[str, Any]:
        # Start with uncompressed permitted history, preserving subscription
        # controls; the compact parent's fixed thirty-action compactor cannot
        # represent extended or early-closed distance-control intervals.
        payload = ChatGPTPlanHybridClient.build_payload(self, observation, grid, mode)
        payload["instructions"] = distance_instructions(self.image_history, task_key=self.task_spec.key,
                                                         demonstration=self.demonstration is not None,
                                                         zero_shot_collision_free=self.zero_shot_collision_free)
        payload["input"] = compact_distance_feedback_history(payload["input"])
        if self.demonstration is not None:
            payload["input"] = [self.demonstration.message(), *payload["input"]]
        payload["text"]["format"] = {"type": "json_schema", "name": distance_schema_name(self.task_spec.key,
                                      zero_shot_collision_free=self.zero_shot_collision_free),
                                      "strict": True, "schema": distance_output_schema(task_key=self.task_spec.key)}
        return payload

    def _history_input(self, payload: Mapping[str, Any]) -> list[dict]:
        if self.demonstration is None:
            return super()._history_input(payload)
        if not payload["input"] or payload["input"][0] != self.demonstration.message():
            raise ValueError("The immutable demonstration prefix is missing or changed")
        # Static reference images must neither duplicate nor displace live images
        # in the episode history. Reattach the exact prefix in build_payload.
        return payload["input"][1:]

    def validate_response(self, data: Any, grid: GridSpec) -> dict[str, Any]:
        self._reset_prediction_diagnostics()
        if not isinstance(data, dict) or set(data) != HYBRID_KEYS | {"control_context"}:
            raise SchemaError("Distance-control wrapper has missing or unexpected fields")
        context = copy.deepcopy(validate_control_context(data["control_context"], task_key=self.task_spec.key))
        if self.task_spec.key == "drawer_bowl":
            phase, target = context["phase"], data["target_region"]
            if phase in ("handle_grasp", "drawer_open", "handle_release"):
                if data["controller"] != "contact_direct" or target != "drawer_handle":
                    raise SchemaError("Drawer handle interaction phases require contact_direct targeting drawer_handle")
            if phase == "grasp" and target != "black_bowl":
                raise SchemaError("Drawer bowl-grasp phase requires target_region=black_bowl")
            if phase == "release" and target != "drawer_interior":
                raise SchemaError("Drawer bowl-release phase requires target_region=drawer_interior")
        normalized = super().validate_response({key: copy.deepcopy(value) for key, value in data.items()
                                                if key != "control_context"}, grid)
        self.last_control_context = context
        self.last_compact_prediction = copy.deepcopy(data)
        return normalized

    def record_control_event(self, event: Mapping[str, Any]) -> None:
        record = validate_control_event(event)
        with self._history_lock:
            if record["step"] != self._feedback_count:
                raise ValueError("distance control event must follow its last acknowledged feedback")
            previous = _text_object(self._history[-1]) if self._history else None
            if previous is not None and previous.get("type") == "distance_control_event":
                raise ValueError("distance control event already recorded at this boundary")
            if (previous is None or previous.get("type") != "execution_feedback"
                    or previous.get("step") != record["step"] - 1
                    or previous.get("chunk_index") != record["chunk_index"]):
                raise ValueError("distance control event must close its actual measured action block")
            self._history.append({"role": "user", "content": [{"type": "input_text", "text": _canonical(record)}]})

    def record_transition(self, feedback: Mapping[str, Any]) -> None:
        """Keep the exact measured-feedback allowlist with mode-specific bounds.

        The inherited validator limits feedback to the nominal thirty actions.
        Do not mutate that configuration to permit locally extended intervals.
        """
        expected = {"type", "step", "chunk_index", "chunk_step", "before_state", "after_state",
            "normalized_action", "desired_translation_delta_m", "desired_world_rotation_delta_rad",
            "measured_translation_delta_m", "measured_translation_norm_m", "commanded_translation_norm_m",
            "translation_tracking_error_m", "gripper_joint_delta", "gripper_command", "rolling_progress"}
        progress_keys = {"window_size", "translation_command_steps", "low_motion_translation_steps",
            "commanded_translation_path_m", "measured_translation_path_m", "measured_net_translation_m",
            "gripper_close_steps"}
        if not isinstance(feedback, Mapping) or set(feedback) != expected or feedback["type"] != "execution_feedback":
            raise ValueError("Feedback contains missing or non-allowlisted fields")
        progress = feedback["rolling_progress"]
        if not isinstance(progress, dict) or set(progress) != progress_keys:
            raise ValueError("Progress feedback contains non-allowlisted fields")
        for key in ("step", "chunk_index", "chunk_step"):
            if type(feedback[key]) is not int or feedback[key] < 0:
                raise ValueError(f"Feedback {key} must be a nonnegative integer")
        for key, length in (("before_state", 8), ("after_state", 8), ("normalized_action", 7),
                            ("desired_translation_delta_m", 3), ("desired_world_rotation_delta_rad", 3),
                            ("measured_translation_delta_m", 3), ("gripper_joint_delta", 2)):
            _numbers(feedback[key], length, key)
        scalar_keys = ("measured_translation_norm_m", "commanded_translation_norm_m",
                       "translation_tracking_error_m", "gripper_command")
        for value in [*(feedback[key] for key in scalar_keys), *progress.values()]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Feedback statistics must contain only finite numbers")
        with self._history_lock:
            if self.last_compact_prediction is None:
                raise ValueError("Feedback requires an accepted controller response")
            controller = self.last_compact_prediction["controller"]
            limit = self._feedback_action_limit(controller)
            if feedback["chunk_step"] >= limit:
                raise ValueError(f"Feedback chunk_step exceeds {limit}-action {controller} interval")
            if feedback["step"] != self._feedback_count:
                raise ValueError("Feedback step must follow the preceding acknowledged action")
            self._history.append({"role": "user", "content": [{"type": "input_text",
                "text": _canonical(copy.deepcopy(dict(feedback)))}]})
            self._feedback_count += 1

    def _feedback_action_limit(self, controller):
        """Explicit operation profiles may override their own interval bound."""
        return MAX_FIELD_REUSE_ACTIONS if controller == "coarse_field" else 5
