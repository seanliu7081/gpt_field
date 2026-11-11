"""Explicit operations and a bounded, evidence-attributed local ledger.

This module has no simulator, model, image decoding, or action execution path.
The model owns semantic intent; measured robot movement is not object movement.
Legacy phase names exist only in the adapter and are absent from the wire prompt.
"""
from __future__ import annotations

import copy
import json
import math

from .attempt_memory import attempt_memory_instructions
from .distance_tasks import resolve_distance_task
from .robot_geometry_guidance import robot_geometry_instructions
from .speed_first_history import BOUNDED_HISTORY_INSTRUCTIONS
from .speed_first_pregrasp import pregrasp_reference_schema, validate_pregrasp_reference
from .speed_first_visual import visual_reference_schema, validate_visual_reference
from .target_identity_guidance import normalize_target_description, target_identity_instructions


OPERATION_VERSION = "operation-field-hybrid-v4"
OPERATION_KINDS = (
    "move", "pregrasp", "grasp_close", "verify_grasp", "transport", "place_align",
    "release", "articulate", "recover", "direct",
)
MAX_LEDGER_CHARS = 7000
_OPERATION_KEYS = {
    "id", "kind", "target_id", "intent", "expected_result", "local_action_budget",
    "allow_local_continuation", "prerequisites", "verify_motion",
}
_VISUAL_KEYS = {
    "mode", "pregrasp_reference", "placement_reference", "tracking_reference",
    "unavailable_reasons",
}
_VIEWS = ("external", "wrist")


def _text_schema(maximum):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _nullable(schema):
    return {"anyOf": [{"type": "null"}, schema]}


def operation_schema():
    """Strict wire schema; cross-field conditions are checked atomically below."""
    prerequisite = {
        "type": "object", "additionalProperties": False,
        "required": ["key", "state", "evidence", "source"],
        "properties": {
            "key": _text_schema(48),
            "state": {"type": "string", "enum": ["supported", "contradicted", "uncertain"]},
            "evidence": _text_schema(160),
            "source": {"type": "string", "enum": ["current_rgb_and_robot_state"]},
        },
    }
    verify = {
        "type": "object", "additionalProperties": False,
        "required": ["offset_xyz_m", "keep_orientation"],
        "properties": {
            "offset_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "number", "minimum": -.03, "maximum": .03}},
            "keep_orientation": {"type": "boolean", "enum": [True]},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(_OPERATION_KEYS),
        "properties": {
            "id": _text_schema(64), "kind": {"type": "string", "enum": list(OPERATION_KINDS)},
            "target_id": _text_schema(64), "intent": _text_schema(240),
            "expected_result": _text_schema(240),
            "local_action_budget": {"type": "integer", "minimum": 1, "maximum": 60},
            "allow_local_continuation": {"type": "boolean"},
            "prerequisites": {"type": "array", "maxItems": 3, "items": prerequisite},
            "verify_motion": _nullable(verify),
        },
    }


def visual_intent_schema():
    from .operation_visual import tracking_reference_schema

    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(_VISUAL_KEYS),
        "properties": {
            "mode": {"type": "string", "enum": ["none", "pregrasp", "held_placement"]},
            "pregrasp_reference": pregrasp_reference_schema(),
            "placement_reference": visual_reference_schema(),
            "tracking_reference": tracking_reference_schema(),
            "unavailable_reasons": {
                "type": "object", "additionalProperties": False,
                "required": list(_VIEWS), "properties": {
                    view: _nullable(_text_schema(160)) for view in _VIEWS
                },
            },
        },
    }


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} has missing or unexpected fields")


def _text(value, limit, name):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} requires nonempty text of at most {limit} characters")
    return value


def _vector(value, count, name):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} requires {count} finite numbers")
    return [float(x) for x in value]


def _index(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_operation(value):
    """Validate semantic intent, without interpreting it as observed truth."""
    _keys(value, _OPERATION_KEYS, "operation")
    for key, limit in (("id", 64), ("target_id", 64), ("intent", 240), ("expected_result", 240)):
        _text(value[key], limit, f"operation.{key}")
    if value["kind"] not in OPERATION_KINDS:
        raise ValueError("Invalid operation.kind")
    budget = value["local_action_budget"]
    if type(budget) is not int or not 1 <= budget <= 60:
        raise ValueError("local_action_budget must be an integer in [1,60]")
    if type(value["allow_local_continuation"]) is not bool:
        raise ValueError("allow_local_continuation must be boolean")
    if budget > 30 and not value["allow_local_continuation"]:
        raise ValueError("A budget above 30 requires explicit continuation permission")
    prerequisites = value["prerequisites"]
    if not isinstance(prerequisites, list) or len(prerequisites) > 3:
        raise ValueError("prerequisites must be a list of at most three claims")
    seen = set()
    for item in prerequisites:
        _keys(item, ("key", "state", "evidence", "source"), "prerequisite")
        key = _text(item["key"], 48, "prerequisite.key")
        _text(item["evidence"], 160, "prerequisite.evidence")
        if key in seen or item["state"] not in ("supported", "contradicted", "uncertain"):
            raise ValueError("Duplicate prerequisite key or invalid state")
        if item["source"] != "current_rgb_and_robot_state":
            raise ValueError("Prerequisites require attributed current observation evidence")
        seen.add(key)
    verify = value["verify_motion"]
    if verify is not None:
        _keys(verify, ("offset_xyz_m", "keep_orientation"), "verify_motion")
        offset = _vector(verify["offset_xyz_m"], 3, "verify_motion.offset_xyz_m")
        norm = math.sqrt(sum(x*x for x in offset))
        if verify["keep_orientation"] is not True or not 0. < norm <= .03 + 1e-12:
            raise ValueError("Verification requires nonzero <=3cm translation at fixed orientation")
        if value["kind"] != "grasp_close":
            raise ValueError("verify_motion is only an explicit grasp_close continuation")
        if not value["allow_local_continuation"] or budget != 60:
            raise ValueError("Close-then-verify requires continuation permission and budget 60")
    return copy.deepcopy(value)


def validate_visual_intent(value, operation):
    from .operation_visual import validate_tracking_reference

    checked_operation = validate_operation(operation)
    _keys(value, _VISUAL_KEYS, "visual_intent")
    expected_mode = {"pregrasp": "pregrasp", "place_align": "held_placement"}.get(
        checked_operation["kind"], "none")
    if value["mode"] != expected_mode:
        raise ValueError("visual_intent.mode must explicitly match operation applicability")
    pregrasp = validate_pregrasp_reference(value["pregrasp_reference"])
    placement = validate_visual_reference(value["placement_reference"])
    tracking = validate_tracking_reference(value["tracking_reference"])
    if pregrasp is not None and expected_mode != "pregrasp":
        raise ValueError("Pregrasp reference is only valid for pregrasp operation")
    if placement is not None and expected_mode != "held_placement":
        raise ValueError("Placement reference is only valid for place_align operation")
    if pregrasp is not None and pregrasp["target_id"] != checked_operation["target_id"]:
        raise ValueError("Pregrasp reference and operation target_id must agree")
    if pregrasp is not None:
        for view in _VIEWS:
            if pregrasp[view] is not None and "tool_reference_mode" not in pregrasp[view]:
                raise ValueError("V4 pregrasp views require explicit tool_reference_mode")
    reasons = value["unavailable_reasons"]
    _keys(reasons, _VIEWS, "unavailable_reasons")
    for view in _VIEWS:
        available = ((pregrasp is not None and pregrasp[view] is not None)
                     or (tracking is not None and tracking[view] is not None)
                     or (placement is not None and view == "external"))
        reason = reasons[view]
        if available and reason is not None:
            raise ValueError(f"{view} has a reference: unavailable reason must be null")
        if not available:
            _text(reason, 160, f"unavailable_reasons.{view}")
    return {**copy.deepcopy(value), "pregrasp_reference": pregrasp,
            "placement_reference": placement, "tracking_reference": tracking}


def legacy_context(operation, visual, task_key="bowl_plate"):
    """Execution-only adapter; the model never sees these old phase labels."""
    operation = validate_operation(operation)
    visual = validate_visual_intent(visual, operation)
    phase = {
        "move": "approach", "pregrasp": "align", "grasp_close": "grasp",
        "verify_grasp": "verify", "transport": "transport", "place_align": "align",
        "release": "release", "articulate": "drawer_open", "recover": "recover",
        "direct": "approach",
    }[operation["kind"]]
    if phase not in resolve_distance_task(task_key).control_phases:
        phase = "grasp"  # Generic articulated interaction compatibility only.
    return {"phase": phase, "visual_reference": copy.deepcopy(visual["placement_reference"]),
            "pregrasp_reference": copy.deepcopy(visual["pregrasp_reference"])}


def operation_instructions(task_key="bowl_plate", image_history=2, *,
                           robot_geometry_guidance=False, target_description=None,
                           attempt_memory=True):
    """A standalone operation prompt: never concatenate legacy align instructions."""
    if type(image_history) is not int or image_history != 2:
        raise ValueError("Operation control requires two recent live image pairs")
    if type(robot_geometry_guidance) is not bool or type(attempt_memory) is not bool:
        raise ValueError("Instruction options must be boolean")
    task = resolve_distance_task(task_key)
    target_description = normalize_target_description(target_description)
    text = (
        f"{OPERATION_VERSION}. Task: {task.prompt}. This is ZERO-SHOT. No demonstrations, "
        "recorded solution, depth, camera calibration or object poses are supplied. Use both "
        "current native512 external and wrist RGB views and measured robot state8: [end-effector "
        "xyz meters, axis-angle xyz radians, two finger joint positions]. Actions are WORLD-frame "
        "[dx,dy,dz,rx,ry,rz,gripper] in [-1,1]. One translation unit means .05m, one rotation unit "
        "means .5rad; gripper +1 closes, -1 opens, and ordinary Direct zero remains zero. Physics "
        "advances with executed actions at 20 Hz. Inference waiting never settles fingers. Optimize "
        "time to completing the whole task, preserving conditions needed by later operations. "
        "Collisions are recorded separately and do not themselves terminate this simulation "
        "protocol. Repeated attempts are permitted. Only raw evaluator LIBERO success completes "
        "the episode; that signal and hidden evaluator information are not policy inputs. "
        "Your completion judgement cannot stop the experiment.\n\n"
        "Return controller,target_region,near_target,routing_reason,field,direct,gripper_operation,"
        "operation,visual_intent and interaction_review. There is no control_context or phase "
        "in this response. operation describes what to achieve, while controller selects how "
        "this response executes. Choose kind=move for free-space approach/reposition, pregrasp "
        "for OPEN-hand tool-to-target alignment before contact, grasp_close for an explicit "
        "stationary closure, verify_grasp for a model-selected check of engagement, transport "
        "for moving a believed-held object, place_align for held-object/destination alignment, "
        "release for an explicit stationary opening, articulate for interaction with a moving "
        "mechanism, recover for purposeful recovery, or direct for another short contact action. "
        "Pregrasp and place_align are DIFFERENT objectives; physical bowl/handle alignment before "
        "closing is pregrasp even when it is part of approach. Preserve operation.id while "
        "pursuing the same object and intended relation; use a new id after changing operation "
        "kind or target. target_id must equal this response's target_region; it is a policy "
        "task-region label, never a simulator pose or hidden object identifier. Use intent and "
        "current RGB appearance to distinguish physical objects within that label. intent states "
        "the desired tool/object relationship; expected_result states "
        "what future RGB/robot evidence would support useful progress. State uncertainty instead "
        "of claiming success from a commanded action.\n\n"
        "operation.prerequisites contains at most three {key,state,evidence,source} records. "
        "state is supported/contradicted/uncertain and source is current_rgb_and_robot_state. "
        "These are your dated MODEL CLAIMS, not measured object truth. Reuse keys to update "
        "conditions such as current engagement or destination accessibility. Inspect current "
        "views before relying on old conditions; preserve an already opened path or accessible "
        "destination while moving to another subgoal. operation_feedback dates these claims, "
        "shows robot motion separately, and marks old claims stale. Neither finger settling nor "
        "end-effector displacement proves object attachment. If a previous action did not produce "
        "the expected result, choose a relevant alignment/depth/orientation/strategy change "
        "rather than repeating an unchanged unsuccessful command.\n\n"
        f"target_region is one of {', '.join(task.target_regions)}. near_target is true exactly "
        "when controller=contact_direct. Inactive branches are null. Use coarse_field for useful "
        "free-space motion including short pregrasp corrections; use contact_direct for uncertain "
        "contact, insertion or articulated motion. Coarse Field fields are stage, unit "
        "orientation_target_xyzw, gripper_command=open/close/hold, motion_mode=move/hold, "
        "goal_offset_m, attraction_weights, repulsors,blocked_boxes,unknown_boxes,goal_reference=null. "
        "goal_offset_m is a grip-site waypoint relative to CURRENT measured XYZ. Account for the "
        "held object's offset when selecting transport/place goals. Stay inside the16cm cube, "
        "preferably +/- .075m on each axis; a distant target requires later model-selected "
        "waypoints. The161-cubed grid is evaluated lazily at1mm spacing. A clear uniform density "
        "uses its exact straight path; otherwise lazy26-neighbor A* minimizes edge length times "
        "average endpoint density. Density is1 plus nonnegative Gaussian repulsors; this is an "
        "estimated task cost, not measured obstacle clearance. attraction_weights remains a "
        "compatibility field. Supply at most four repulsors {center_offset_m,radius_m,weight} "
        "and at most four boxes of each kind {lower_offset_m,upper_offset_m}. Prefer soft costs "
        "for uncertain obstacles; hard boxes exclude their regions. Empty lists are allowed; "
        "never invent geometry. Numerical1mm resolution does not imply1mm perception accuracy. "
        "The measured-pose follower uses bounded actions and3cm lookahead; local plans have at "
        "most ten1cm segments and are replanned at most every ten executed actions.\n\n"
        "operation.local_action_budget is an explicit maximum, not a required action count. "
        "Ordinary Direct uses exactly5 and allow_local_continuation=false: provide exactly five "
        "normalized seven-number actions with gripper_operation=null. They execute unchanged. "
        "A Field motion normally uses budget30 and allow_local_continuation=false. You may "
        "explicitly permit useful continued execution of the SAME waypoint with budget31..60 "
        "and allow_local_continuation=true. This never invents a new waypoint or repeats Direct "
        "actions. Arrival, invalid geometry, or measured stall can return earlier. A Field hold "
        "returns after five actions. Do not extend motion merely to wait for visual validation. "
        "Gripper hold repeats the most recent nonzero command, initially open.\n\n"
        "For an explicit grasp_close/release choose contact_direct,near_target=true,field=null,"
        "direct=null and gripper_operation={command:close/open}. The local controller holds the "
        "CURRENT measured pose, closes for at least15 actions or opens for at least10, and ends "
        "after measured settling or at30actions. Use budget30,continuation=false normally. Five "
        "actions advance only.25s; repeated close chunks can be necessary until fingers settle. "
        "Settling is robot feedback, not grasp proof. Do not unnecessarily repeat a completed "
        "open/close operation. Default verify_motion=null. For grasp_close ONLY you may explicitly "
        "request a close-then-check operation with verify_motion={offset_xyz_m:[dx,dy,dz],"
        "keep_orientation:true}, budget60 and continuation=true. Choose a nonzero WORLD offset "
        "with Euclidean norm<=.03m appropriate to this object's mechanism. The local controller "
        "may execute that bounded translation after a settled close at unchanged orientation; "
        "the offset is relative to the settled-close pose. No direction is inferred. A freely "
        "movable object may support a short lift; a constrained handle needs movement compatible "
        "with its mechanism. The next views, not completed motion, determine engagement.\n\n"
        "visual_intent.mode MUST be pregrasp for kind=pregrasp, held_placement for kind=place_align, "
        "and none for other kinds. This declares applicability even when no reference can be "
        "observed. It does NOT claim tracking success. Set unavailable_reasons.external/wrist "
        "to a brief actual reason for every view without any usable declared reference; use null "
        "when that view has a reference. Missing visual support never blocks normal execution. "
        "Use useful ordinary motions instead of measurement-only probing or waiting.\n\n"
        "Optional tracking_reference provides simple visible-body regions for local progress "
        "feedback: {reference_id,external,wrist}. Each available view is {object_roi_uv,tool_roi_uv,"
        "target_roi_uv}; each ROI is null or [u_min,v_min,u_max,v_max] normalized to[0,1] with "
        "top-left origin. At least one ROI per nonnull view and at least one view are required. "
        "object is the manipulated object's visible material, tool is a visible rigid hand, "
        "palm or finger feature, and target is a destination or stationary contextual feature. "
        "A tool monitoring ROI does not need to be the tiny contact pad. Use tightly visible "
        "same-body texture/edges, excluding unrelated background/fingers from object ROIs. "
        "Roles are your semantic hypotheses. Reuse the reference_id only for the same regions; "
        "reseed with a new ID when selecting new image regions. Local code chooses texture "
        "within these regions and reports tracking and co-motion evidence. Co-motion is only "
        "a cue; wrist-camera motion, occlusion and rigid scene motion can be ambiguous. This "
        "ROI tracker does not invent precise contact points, estimate3D pose, move the robot, "
        "or certify grasp/task success. If visible material regions are available, provide "
        "them during engagement checks and transport; otherwise keep tracking_reference=null.\n\n"
        "Optional pregrasp_reference is ONLY for kind=pregrasp with an open gripper and fixed "
        "orientation. Use Field gripper_command=open or negative/open in all five Direct "
        "actions. Its shape is {reference_id,target_id,external,wrist}; each available view "
        "contains tool_contact_uv,target_contact_uv,tool_roi_uv,target_roi_uv,desired_offset_uv,"
        "tool_reference_mode. Contact points must initially be VISIBLE MATERIAL, not an invisible "
        "pinch midpoint, empty gap or guessed hidden contact. Each ROI supports the same rigid "
        "body; nearby auxiliary texture may be up to24pixels from its visible contact point. "
        "desired_offset_uv is desired TOOL-MINUS-TARGET image relation, not automatically zero. "
        "Use tracked_material in external; wrist may use tracked_material or fixed_wrist for an "
        "initially visible hand reference. Fixed wrist reference expires after12unchecked frames "
        "and finger/orientation changes invalidate it. Both views work independently. Retain "
        "reference_id and desired relation while tracking the same objective; provide current "
        "transported points/ROIs. Natural measured translations learn a local image response; "
        "only held-out-validated observed directions may revise the Field goal by <=6mm TOTAL "
        "from its immutable nominal waypoint, inside a4cm local domain. Useful pregrasp motion "
        "should normally be short, within3cm. Choose orientation first. Loss of tracking or "
        "validation simply restores ordinary goal following. Do not fabricate points to satisfy "
        "this optional feature. Null references with honest unavailability reasons are valid.\n\n"
        "Optional placement_reference is ONLY for kind=place_align with a believed rigidly held "
        "object and closed gripper. It is {reference_id,moving_point_uv,target_point_uv,"
        "desired_offset_uv,planar_alignment:true,rigid_grasp:true} in the FIXED EXTERNAL view. "
        "Choose a moving object material patch and a stationary destination material patch. "
        "The desired difference accounts for which physical patches were selected and height; "
        "it is not automatically zero. Keep height and orientation fixed. Existing localXY "
        "response may refine the waypoint; image agreement alone does not establish3D support "
        "or justify releasing. When inappropriate use placement_reference=null and normal "
        "control. Pregrasp and placement references are mutually exclusive. Inspect actual "
        "object/destination relations before release and support/separation after release. "
        "operation_feedback is one current bounded presentation, not additional model input "
        "history to repeat. Distinguish measured motion, tracked image evidence and semantic "
        "claims when judging progress. "
    )
    text += "\n\n" + BOUNDED_HISTORY_INSTRUCTIONS
    if robot_geometry_guidance:
        text += "\n\n" + robot_geometry_instructions()
    if target_description is not None:
        text += "\n\n" + target_identity_instructions(target_description)
    if attempt_memory:
        text += "\n\n" + attempt_memory_instructions()
    return text


def _compact_visual(value):
    """Bound JSON diagnostics, excluding any possible images or raw coordinates."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("visual_diagnostics must be a dictionary or null")
    # A whitelist prevents accidental image/state-oracle propagation in feedback.
    allowed = {
        "status", "reliable", "reference_id", "mode", "applicable", "tracking_status",
        "correction_status", "source", "reason", "engagement_evidence", "co_motion",
        "object_motion", "target_motion", "measured_robot_motion_m", "views", "evidence",
        "motion_evidence", "task_success_certified", "grasp_certified", "object_attachment_verified",
        "verification_evidence", "verification_reason", "verification_active", "tracking_valid",
        "active_views", "measured_robot_displacement_m", "external_target_image_motion_observed",
        "observed_transitions", "tracking_errors", "operation_kind", "epoch", "limitations",
    }

    def small(item, depth=0):
        if depth > 3:
            return "details_omitted"
        if item is None or type(item) is bool:
            return item
        if type(item) in (int, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            return item[:160]
        if isinstance(item, list):
            return [small(x, depth+1) for x in item[:6]]
        if isinstance(item, dict):
            return {str(k)[:48]: small(v, depth+1) for k, v in list(item.items())[:10]}
        return None

    result = {key: small(value[key]) for key in sorted(allowed) if key in value}
    result.update(task_success_certified=False, grasp_certified=False,
                  object_attachment_verified=False)
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > 1800:
        result = {key: result[key] for key in (
            "status", "reliable", "source", "reason", "tracking_valid", "active_views",
            "verification_active", "verification_evidence", "verification_reason",
            "measured_robot_displacement_m", "external_target_image_motion_observed") if key in result}
        result.update(details_omitted=True, task_success_certified=False, grasp_certified=False,
                      object_attachment_verified=False)
    return result


class OperationEvidenceLedger:
    """Persistent, bounded feedback; never a task oracle or an execution gate.

    Call accept only after the entire response is accepted. Observation indices
    and completed-action steps supply causal dates without invented wall times.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._operations = []
        self._claims = {}
        self._last_observation = None
        self._last_step = None
        self._completed_actions = 0
        self._state = None
        self._visual = None

    def accept(self, operation, visual, observation_index, state8=None):
        operation = validate_operation(operation)
        visual = validate_visual_intent(visual, operation)
        index = _index(observation_index, "observation_index")
        state = None if state8 is None else _vector(state8, 8, "state8")
        if self._last_observation is not None and index <= self._last_observation:
            raise ValueError("Accepted observation indices must strictly increase")
        current = self._operations[-1] if self._operations else None
        if current is not None and current["id"] == operation["id"]:
            if current["kind"] != operation["kind"] or current["target_id"] != operation["target_id"]:
                raise ValueError("An operation id cannot change kind or target")
        else:
            current = {
                "id": operation["id"], "kind": operation["kind"], "target_id": operation["target_id"],
                "first_observation_index": index, "total_operation_actions": 0,
                "measured_path_m": 0., "measured_translation_m": [0., 0., 0.],
                "measured_orientation_change_rad": 0., "finger_change_m": [0., 0.],
                "feedback_chain_contiguous": True,
            }
            self._operations.append(current)
            self._operations = self._operations[-4:]
        current.update(
            intent=operation["intent"], expected_result=operation["expected_result"],
            last_observation_index=index, request_action_count=0,
            requested_action_budget=operation["local_action_budget"],
            allow_local_continuation=operation["allow_local_continuation"],
            visual_applicability=visual["mode"],
            visual_reference_supplied=any(visual[k] is not None for k in (
                "pregrasp_reference", "placement_reference", "tracking_reference")),
            visual_unavailability=copy.deepcopy(visual["unavailable_reasons"]),
            source="accepted_model_operation_and_measured_robot_feedback",
            last_request_state8=state,
        )
        for claim in operation["prerequisites"]:
            self._claims[claim["key"]] = {
                "key": claim["key"], "state": claim["state"], "evidence": claim["evidence"],
                "source": "model_semantic_claim", "claimed_observation_source": claim["source"],
                "observation_index": index, "completed_step_at_claim": self._last_step,
                "completed_action_count_at_claim": self._completed_actions,
                "independently_verified": False,
            }
        self._claims = dict(sorted(self._claims.items(),
            key=lambda item: item[1]["observation_index"])[-8:])
        self._last_observation, self._state = index, state
        self._visual = None
        return copy.deepcopy(current)

    def record_feedback(self, before_state, after_state, step, visual_diagnostics=None):
        before = _vector(before_state, 8, "before_state")
        after = _vector(after_state, 8, "after_state")
        step = _index(step, "step")
        visual = _compact_visual(visual_diagnostics)
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("Feedback steps must strictly increase")
        if visual is not None:
            visual["completed_action_step"] = step
            visual["evidence_type"] = "local_image_measurements_and_inference_not_verified_object_state"
        if self._operations:
            current = self._operations[-1]
            if ((self._last_step is not None and step != self._last_step+1)
                    or (self._state is not None and any(abs(before[i]-self._state[i]) > 1e-6 for i in range(8)))):
                current["feedback_chain_contiguous"] = False
            delta = [after[i]-before[i] for i in range(3)]
            current["total_operation_actions"] += 1
            current["request_action_count"] += 1
            current["measured_path_m"] += math.sqrt(sum(x*x for x in delta))
            current["measured_translation_m"] = [current["measured_translation_m"][i]+delta[i] for i in range(3)]
            # Rotation-vector subtraction is not a physical angle. Keep only a
            # representation-independent quaternion geodesic for the summary.
            from .action import _rotvec_quaternion
            import numpy as np
            qa = _rotvec_quaternion(np.asarray(before[3:6]))
            qb = _rotvec_quaternion(np.asarray(after[3:6]))
            angle = 2.*math.acos(min(1., abs(float(np.dot(qa, qb)))))
            current["measured_orientation_change_rad"] += angle
            current["finger_change_m"] = [current["finger_change_m"][i]+after[i+6]-before[i+6] for i in range(2)]
            current["latest_feedback_step"] = step
        self._completed_actions += 1
        self._last_step, self._state, self._visual = step, after, visual
        return True

    def presentation(self, current_observation_index, current_state=None):
        index = _index(current_observation_index, "current_observation_index")
        state = self._state if current_state is None else _vector(current_state, 8, "current_state")
        if self._last_observation is not None and index < self._last_observation:
            raise ValueError("Cannot present future operation evidence to an earlier observation")
        claims = []
        for claim in self._claims.values():
            value = copy.deepcopy(claim)
            age = index-claim["observation_index"]
            actions = self._completed_actions-claim["completed_action_count_at_claim"]
            value.update(age_observations=age, executed_actions_since_claim=max(0, actions),
                         stale=age > 0 or actions > 0, usable_as_verified_condition=False)
            claims.append(value)
        result = {
            "type": "operation_feedback", "version": OPERATION_VERSION,
            "current_observation_index": index, "last_completed_step": self._last_step,
            "current_measured_state8": copy.deepcopy(state),
            "recent_operations": copy.deepcopy(self._operations),
            "prerequisites": sorted(claims, key=lambda x: x["observation_index"], reverse=True),
            "latest_visual_evidence": copy.deepcopy(self._visual),
            "task_success_certified": False, "object_attachment_verified": False,
            "interpretation": "Robot motion is measured; visual relations and semantic claims are evidence, not contact or task proof.",
        }
        def size():
            return len(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        while size() > MAX_LEDGER_CHARS and len(result["recent_operations"]) > 1:
            result["recent_operations"].pop(0)
        while size() > MAX_LEDGER_CHARS and result["prerequisites"]:
            result["prerequisites"].pop()
        if size() > MAX_LEDGER_CHARS:
            result["latest_visual_evidence"] = {"status": "details_omitted_for_budget", "task_success_certified": False}
        if size() > MAX_LEDGER_CHARS:
            raise ValueError("Operation feedback exceeded bounded presentation budget")
        return result
