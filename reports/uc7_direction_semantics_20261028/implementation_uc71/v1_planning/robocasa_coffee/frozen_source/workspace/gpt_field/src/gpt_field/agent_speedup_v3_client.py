"""Reviewed v3 executive: one action and bounded geometry edits in one model call.

Transport, approval enforcement, synchronized public observations, six paired
images and the complete accepted transcript are inherited from user-check v7.
This module neither reconstructs a scene nor executes a robot action.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from .agent_speedup_v2_user_check_agent_client import (
    AGENT_INSTRUCTIONS as V7_AGENT_INSTRUCTIONS,
    AGENT_KIND,
    AgentFirstV2Client,
    agent_schema as v7_agent_schema,
    validate_agent_decision as validate_v7_agent_decision,
)
from .agent_speedup_v2_client import (
    TRANSIENT_MAX_RETRIES, TRANSIENT_BACKOFF_BASE_S, OVERLOAD_JITTER_S, OVERLOAD_MAX_WAIT_S,
)
from .agent_speedup_v2_user_check_client import UserCheckV2Client
from .agent_speedup_v3_fragments import adaptive_fragment_capabilities
from .agent_speedup_v3_geometry import scene_update_schema, validate_scene_update
from .agent_speedup_v3_program import program_schema, validate_program
from .astra import SchemaError, _canonical
from .ordered_transport import ordered_dumps
from .agent_speedup_v3_transport import V3BoundedSubscriptionTransport, transport_settings


CLIENT_VERSION = "agent-speedup-v3-client-11"
PROMPT_VERSION = "agent-speedup-v3-prompt-10"
HISTORY_PROTOCOL = "persistent_v3_raw_audit_compact_wire_six_visual_pairs_10"
MOTION_INTENTS = ("transit", "approach", "grasp", "place", "retreat", "manipulate", "observe")
MAX_CONTEXT_IDS = 12
MAX_ID_LENGTH = 80

COMPACT_OUTPUT_INSTRUCTIONS = """
OUTPUT SERIALIZATION. Emit exactly one compact JSON object conforming to the
schema. Do not indent or align fields, insert blank lines, or pad between tokens.
Use ordinary spaces inside text values where needed. Finish immediately after
the closing brace; do not append whitespace. Keep all required fields and the
complete requested plan or decision. This changes formatting, not task content.
""".strip()


SUBSCRIPTION_AVAILABILITY_RETRY_POLICY = "agent-speedup-v3-subscription-availability-retry-1"
SUBSCRIPTION_AVAILABILITY_CODES = (
    "subscription_sharing_user_unavailable", "subscription_sharing_usage_unavailable",
)


def subscription_recovery_settings() -> dict:
    """Auditable v3 addition to the inherited bounded transport recovery policy."""
    return {
        "version": SUBSCRIPTION_AVAILABILITY_RETRY_POLICY,
        "http_status": 503, "error_codes": list(SUBSCRIPTION_AVAILABILITY_CODES),
        "max_retries": TRANSIENT_MAX_RETRIES, "max_attempts": 1 + TRANSIENT_MAX_RETRIES,
        "backoff_base_s": TRANSIENT_BACKOFF_BASE_S, "jitter_max_s": OVERLOAD_JITTER_S,
        "retry_after_respected": True, "max_single_wait_s": OVERLOAD_MAX_WAIT_S,
        "budget": "shared inherited transient retry and wall-clock budgets",
        "eligibility": "no completed response envelope and no committed observation",
        "same_request": True, "credentials_preserved": True,
        "api_key_fallback": False, "model_fallback": False,
        "robot_actions_during_retry": 0,
        "transport_bounds": transport_settings(),
    }


class V3SubscriptionAvailabilityRetryMixin:
    """Recover only documented temporary subscription lookup failures.

    Reuse the existing same-request loop, accounting and Retry-After limits.
    This class neither handles authentication nor changes accepted history.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = V3BoundedSubscriptionTransport()
        super().__init__(*args, **kwargs)

    def preflight(self):
        if isinstance(self.transport, V3BoundedSubscriptionTransport):
            self.transport.preflight()
        return super().preflight()

    def _retry_cause(self, exc, initial_observation_count, retries):
        code = getattr(exc, "error_code", None)
        if code in SUBSCRIPTION_AVAILABILITY_CODES:
            if (getattr(exc, "http_status", None) != 503
                    or self.last_response is not None
                    or self._observation_count != initial_observation_count):
                return None
            return "transient" if retries["transient"] < TRANSIENT_MAX_RETRIES else None
        return super()._retry_cause(exc, initial_observation_count, retries)

    def history_stats(self):
        result = super().history_stats()
        result.update(
            subscription_availability_retry_policy=SUBSCRIPTION_AVAILABILITY_RETRY_POLICY,
            subscription_recovery_settings=subscription_recovery_settings(),
            subscription_availability_retry_requests=sum(
                self._transient_retry_codes.get(code, 0) for code in SUBSCRIPTION_AVAILABILITY_CODES),
        )
        return result

    def _write_log(self, path, data):
        if isinstance(data, dict):
            is_request = "payload" in data and "request_ordinal" in data
            diagnostics = getattr(self.transport, "last_diagnostics", None)
            if diagnostics is not None and not is_request:
                data = {**data, "v3_transport_diagnostics": copy.deepcopy(diagnostics)}
            data = {**data, "subscription_availability_retry_policy": SUBSCRIPTION_AVAILABILITY_RETRY_POLICY,
                    "subscription_recovery_settings": subscription_recovery_settings(),
                    "agent_speedup_v3_client_version": self.client_version}
        return super()._write_log(path, data)


class AgentSpeedUpV3PlanClient(V3SubscriptionAvailabilityRetryMixin, UserCheckV2Client):
    """The unchanged draft-plan protocol with v3 subscription recovery only."""

    client_version = "agent-speedup-v3-plan-client-11"
    prompt_version = PROMPT_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for kind in self._instruction_text:
            self._instruction_text[kind] += "\n\n" + COMPACT_OUTPUT_INSTRUCTIONS


def motion_context_schema() -> dict:
    identifier = {"type": "string", "minLength": 1, "maxLength": MAX_ID_LENGTH}
    nullable_id = {"anyOf": [identifier, {"type": "null"}]}
    inventory = {
        "reviewed": {"type": "boolean"},
        "support_ids": {"type": "array", "items": identifier, "maxItems": MAX_CONTEXT_IDS},
        "nearby_obstacle_ids": {"type": "array", "items": identifier, "maxItems": MAX_CONTEXT_IDS},
        "unresolved": {"type": "boolean"},
    }
    fields = {
        "intent": {"type": "string", "enum": list(MOTION_INTENTS)},
        "target_id": copy.deepcopy(nullable_id),
        "held_object_id": copy.deepcopy(nullable_id),
        "allowed_contact_ids": {"type": "array", "items": identifier, "maxItems": MAX_CONTEXT_IDS},
        "scene_inventory": {"type": "object", "properties": inventory,
                            "required": list(inventory), "additionalProperties": False},
    }
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def validate_motion_context(value) -> dict:
    """Intent identifies a proposed interaction, never measured contact or grasp."""
    required = set(motion_context_schema()["required"])
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("motion_context requires exactly intent, target_id, held_object_id, allowed_contact_ids, scene_inventory")
    if not isinstance(value["intent"], str) or value["intent"] not in MOTION_INTENTS:
        raise ValueError("motion_context.intent is unsupported")

    def identifier(item):
        return isinstance(item, str) and bool(item.strip()) and len(item) <= MAX_ID_LENGTH

    for key in ("target_id", "held_object_id"):
        if value[key] is not None and not identifier(value[key]):
            raise ValueError(f"motion_context.{key} must be null or a nonempty bounded ID")
    def identifiers(items, name):
        if (not isinstance(items, list) or len(items) > MAX_CONTEXT_IDS
                or any(not identifier(item) for item in items)):
            raise ValueError(f"{name} must be a bounded list of IDs")
        if len(set(items)) != len(items):
            raise ValueError(f"{name} must not contain duplicates")

    identifiers(value["allowed_contact_ids"], "motion_context.allowed_contact_ids")
    inventory = value["scene_inventory"]
    if (not isinstance(inventory, Mapping)
            or set(inventory) != {"reviewed", "support_ids", "nearby_obstacle_ids", "unresolved"}):
        raise ValueError("scene_inventory requires exactly reviewed, support_ids, nearby_obstacle_ids, unresolved")
    for key in ("reviewed", "unresolved"):
        if type(inventory[key]) is not bool:
            raise ValueError(f"scene_inventory.{key} must be a boolean")
    for key in ("support_ids", "nearby_obstacle_ids"):
        identifiers(inventory[key], f"scene_inventory.{key}")
    return copy.deepcopy(dict(value))


def v3_agent_schema() -> dict:
    schema = v7_agent_schema()
    schema["properties"]["type"]["enum"] += ["program", "geometry_patch", "continue_intent", "acquire_evidence", "escalate"]
    continuation = {"type": "object", "properties": {
        "intent_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "resume": {"type": "boolean"}}, "required": ["intent_id", "resume"],
        "additionalProperties": False}
    schema["properties"]["continuation"] = {"anyOf": [continuation, {"type": "null"}]}
    schema["properties"]["scene_update"] = scene_update_schema()
    schema["properties"]["motion_context"] = motion_context_schema()
    schema["properties"]["program"] = {"anyOf": [program_schema(), {"type": "null"}]}
    schema["required"] = list(schema["properties"])
    return schema


def validate_v3_agent_decision(value) -> dict:
    """Validate both additions before returning the unchanged single-action contract."""
    if not isinstance(value, Mapping) or set(value) != set(v3_agent_schema()["required"]):
        raise SchemaError("v3 agent output must contain the seven action fields plus scene_update, motion_context, continuation and program")
    base_action = {key: value[key] for key in v7_agent_schema()["required"]}
    route = value["type"]
    continuation = value["continuation"]
    if route == "program":
        if continuation is not None:
            raise SchemaError("A new program requires continuation=null")
        base_action["type"] = "observe"
    elif value["program"] is not None:
        raise SchemaError("program must be null for other decision types")
    if route in ("geometry_patch", "continue_intent", "acquire_evidence"):
        if (not isinstance(continuation, Mapping) or set(continuation) != {"intent_id", "resume"}
                or not isinstance(continuation["intent_id"], str)
                or not 1 <= len(continuation["intent_id"].strip()) <= 80
                or type(continuation["resume"]) is not bool
                or (route == "continue_intent" and not continuation["resume"])
                or (route == "acquire_evidence" and continuation["resume"])):
            raise SchemaError("Geometry continuation requires an exact pending intent_id and explicit resume boolean")
        # Reuse strict text/null-action validation without modifying shared v2.
        base_action["type"] = "observe"
    elif route == "escalate":
        if continuation is not None:
            raise SchemaError("escalate does not resume a pending intent")
        base_action["type"] = "observe"
    elif continuation is not None:
        raise SchemaError("continuation must be null for a new action")
    action = validate_v7_agent_decision(base_action)
    action["type"] = route
    action["continuation"] = copy.deepcopy(continuation)
    try:
        action["scene_update"] = validate_scene_update(value["scene_update"])
        action["motion_context"] = validate_motion_context(value["motion_context"])
        action["program"] = validate_program(value["program"]) if route == "program" else None
    except (ValueError, TypeError, KeyError) as exc:
        raise SchemaError(f"v3 agent output: {exc}") from None
    return action


LEAN_EXECUTIVE_INSTRUCTIONS = """
Complete the reviewed task with a Franka Panda using synchronized external/wrist
RGB, measured robot state, public robot/camera calibration and action history.
No simulator object poses, depth, segmentation, contact identities, rewards or
evaluator predicates are available. The six latest image pairs retain their
own timestamps/calibration. Accepted outputs and opaque reasoning are retained;
older request text omits repeated capabilities and controller diagnostics. Full
original observations remain in the local audit. Historical blockers describe
that time; current_decision is authoritative for the current blocker.

Preserve reviewed physical identities, final goals, hold_true and applicable
keep constraints. The initial decomposition is guidance: choose work order,
grips and routes. Prefer a persistent program for a sufficiently specified
operation; its controller handles progress, phase changes, verification and
bounded recovery. Normal phase transitions do not require another model call.
Use model decisions for changed intent, an actual execution failure or evidence
that local execution cannot acquire. An intermediate waypoint or early yield
is not completion. Only the independent evaluator determines task success.

Return one schema-valid JSON decision. Include scene_update and motion_context in this SAME response;
unused action payloads are null. Choose:
- program: program={operations:[...]} or {stages:[...]}, described below.
- execute: execution={segments:[...]}. One motion_context applies to every
  segment; use program stages when contact permissions change.
- motion: short world-frame action; relative XYZ applies once from measured
  entry, absolute XYZ is world position; quaternion_wxyz is absolute or null to
  retain orientation. steps=1..5 at 20 Hz; target <=0.05 m and 0.35 rad.
- macro: macro={step_index,budget_steps,continuation} for optional native reviewed
  hold/turn/place_on/release. budget_steps=1..60; continuation resumes only the
  same paused native session without intervening direct motion. Native release
  returns to its original pick support; direct open releases at the current pose.
- observe: hold briefly for fresh RGB; this does not change viewpoint.
- finish: request final observation; this does not assert success or terminate.
- geometry_patch/continue_intent/acquire_evidence/escalate: see recovery below.

execute has 1..6 segments with skill, xyz_m, quaternion_wxyz, gripper, profile,
max_steps, observation_after. move_to_pose uses absolute grip-site XYZ;
move_relative/approach applies a world displacement once at segment entry.
Arm-motion segments keep the gripper. set_gripper uses null pose and open/close;
wait uses null pose and keep. max_steps=1..120, total<=240; these are control
chunks, not guaranteed distances. Initial target limits: transit 0.30 m/1.2 rad,
contact 0.08 m/0.5 rad. Runtime bounds commands using observed response and nearby
geometry; contact-profile speed can increase away from obstacles. Explicit
observation_after pauses; program evidence checkpoints are handled locally.

World XYZ is metres, +Z up; wxyz is absolute hand orientation. XYZ already locates
the grip site between fingers, 0.097 m along hand-local +Z from hand origin; do
not add that offset. Hand-local +Z approaches, Y separates jaws; pad midpoint is
[0,0,-0.0036] relative to grip site. Use supplied robot geometry for dimensions.
For camera-local [X,Y,Z], Z<0: u=cx+fx*X/(-Z), v=cy-fy*Y/(-Z).
p_world=position_world_m+rotation_world_from_camera*p_camera. Images are upright,
uncropped; camera +X right, +Y up, -Z forward. Pixels give rays, not known depth.
""".strip()

LEAN_GEOMETRY_INSTRUCTIONS = """
LOCAL GEOMETRY IS AN ESTIMATE, NOT A CLEARANCE CERTIFICATE.
scene_update={entities:[],invalidate_ids:[],remove_ids:[]} is valid. Provide only
useful new/changed coarse estimates for the current goals. Keep unchanged
geometry; do not reconstruct the full scene on every call or invent free-region
boxes to authorize motion. Preserve cavities and finger openings with separate
walls/rims when needed. Parent IDs describe physical pose dependencies;
related_ids are semantic only. Occlusion is not removal or proof of movement.

All model poses, extents, attachment and free regions are model_visual_estimate
hypotheses with uncertainty_m, coverage and source. Do not refresh timestamps,
reduce uncertainty, remove inconvenient obstacles or enlarge free space merely
to pass a check. Correct visible contradictions selectively. Existing solids
and uncertainty constrain motions even when no exhaustive free-space map exists.
Unknown coverage remains unknown. execution_allowed=true can authorize bounded,
locally monitored motion while status=unknown and execution_status=admissible;
physical_clearance_verified remains false. Read blocking_reasons separately from
advisories. Missing map coverage alone need not trigger another geometry request;
actual predicted collision, unresolved payload bounds, tracking anomaly and
failed interaction verification still constrain execution. Never claim certainty
from admission. Runtime, not model wording, decides the permitted increment.

motion_context identifies intent, target_id, held_object_id, allowed_contact_ids,
and scene_inventory={reviewed,support_ids,nearby_obstacle_ids,unresolved}. IDs are
stable and refer to stored or newly supplied estimates. Record the visible local
support/neighbors honestly, leaving unresolved=true where appropriate. This
inventory is an estimate, not an independent completeness test. Allowed contacts
are phase-specific; they never authorize arbitrary palm/payload/neighbor impact.
A close command used for pushing is a hand posture, not a grasp: use manipulate
or approach and held_object_id=null. A genuine grasp uses grasp intent and an
explicit object. Model attachment claims never establish a physical payload.

anchors is null unless the same visible material point is matched in both
synchronized images. Then supply external_uv/wrist_uv, same_feature=true and a
defensible anchor_local_m (null means diagnostic only). Silhouette centers are
not material correspondences. Accepted calibrated matching may translate an
estimate while retaining shape uncertainty; one point proves neither orientation
nor free volume. local_pose_revision marks accepted local pose evidence.
""".strip()

LEAN_RECOVERY_INSTRUCTIONS = """
current_decision preserves the pending goal, original action, operation_state,
tracking, blocking_reasons, advisories and ALL missing_evidence_requests. Consider
these requests in context: resolve actual blockers from fresh public evidence;
an advisory is not a mandatory model reconstruction. whole_goal_assessment and
executable_prefix_assessment distinguish the original goal from a bounded prefix.
coverage_witnesses identify estimated boundaries, not verified free space.

operation_state binds the manipulated object across program/execute/motion.
Release verification refers to that object, never the destination. Physical
close/open history and accepted visual evidence persist across routes. Finger
gap and model belief do not prove hold/release. A visual predicate advances only
after geometry confirmation is accepted. Preserve unresolved prior payloads;
never start a new object operation before the previous accepted release.

geometry_patch uses continuation={intent_id,resume}; continue_intent requires
resume=true. Other action payloads are null. Preserve paused intent, original
absolute goal, held object and contact rules; only inventory may change with
new evidence. Resume retains consumed watchdog and recovery counts; route
switching does not reset them or reapply relative motion. resume=false only
patches. Continue ordinary admissible progress locally without requesting a new
model decision for each chunk. Do not repeat an unchanged rejected action.

If required evidence is not visible, acquire_evidence uses
continuation={intent_id,resume:false}, unchanged protected motion_context and
empty scene_update. Runtime checks a bounded informative viewing path, keeps the
gripper and pending goal, then obtains fresh RGB. Predicted visibility is not
measured evidence or automatic goal completion. If no useful admissible view or
bounded recovery exists, escalate with null action payloads and explain the
missing evidence. This yields evidence_acquisition_failed, never success.
""".strip()

LEAN_PROGRAM_INSTRUCTIONS = """
Prefer program={operations:[...]} with 1..3 pick_place operations, a response
batch bound rather than an episode cap. Each has id, kind, object_id,
destination_id, after, quaternion_wxyz, source_inventory, destination_inventory,
route_obstacle_ids, contact_ids and max_steps. after lists predecessor IDs; the
graph is acyclic and execution serial. The next operation requires predecessors'
accepted visual release. Source/destination inventories are separate; contact_ids
has grasp/carry/place/retreat arrays. Do not merge contacts across phases or omit
other relevant stored obstacles. quaternion_wxyz is absolute or null;
max_steps=1..120 is an initial chunk. No XYZ waypoints, offsets, travel distances,
task names or simulator coordinates belong in compact operation contracts.

Runtime derives approach/grasp, close, lift+visual_hold, transfer/place, open and
retreat+visual_release from current estimated geometry and observed progress.
It checks both stationary/carried hypotheses during a finite unconfirmed lift;
transfer/place need accepted visual hold. Released objects remain obstacles.
For other manipulation use an appropriate macro, explicit action or
program={stages:[...]} with up to24 stages. Each stage has motion_context,
goal={kind,reference_id,quaternion_wxyz}, gripper, profile, verification,max_steps.
Kinds: approach/grasp/lift_clear/transfer/place/hold/retreat. Verification:
arrival/visual_hold/visual_release; arrival means mechanical arrival only.

Batch all sufficiently specified work. Local pauses retain remaining operations
and budgets; patch/resume the cause rather than rebuilding the same plan.
A replacement program must preserve unfinished reviewed goals and bound payload.
The 300-second end-to-end objective includes preparation/artifact completion,
excludes human approval waiting and never permits skipped verification or goals.
""".strip()

AGENT_INSTRUCTIONS = "\n\n".join((LEAN_EXECUTIVE_INSTRUCTIONS, LEAN_GEOMETRY_INSTRUCTIONS,
    LEAN_RECOVERY_INSTRUCTIONS, LEAN_PROGRAM_INSTRUCTIONS, COMPACT_OUTPUT_INSTRUCTIONS))


def current_decision_core(context):
    """Keep the complete current causal requirements outside generic truncation."""
    native = context.get("native_evidence") or {}
    supplied = native.get("current_decision") if isinstance(native, Mapping) else None
    if supplied is not None:
        return copy.deepcopy(supplied)
    geometry = native.get("shared_geometry", {}) if isinstance(native, Mapping) else {}
    last_motion = geometry.get("last_motion", {}) if isinstance(geometry, Mapping) else {}
    requests, seen, reasons = [], set(), []

    def collect(value):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "missing_evidence_requests" and isinstance(child, list):
                    for requirement in child:
                        identity = _canonical(requirement)
                        if identity not in seen:
                            seen.add(identity)
                            requests.append(copy.deepcopy(requirement))
                elif key == "reason" and isinstance(child, str) and child not in reasons:
                    reasons.append(child)
                elif isinstance(child, (Mapping, list, tuple)):
                    collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    # The current geometry assessment is authoritative; completed historical
    # stages must not resurrect superseded requirements in a new decision.
    sources = [last_motion] if last_motion else [context.get("execution"), context.get("last_feedback")]
    for source in sources:
        collect(source)
    return {"version": "v3-current-decision-1", "step": context.get("step"),
            "pending_intent": None, "missing_evidence_requests": requests,
            "blocker_reasons": reasons, "requirements_truncated": False,
            "task_success": "not_available_to_policy"}



def v3_fragment_capabilities() -> dict:
    """Expose the actual runtime controller's metadata, not a parallel contract."""
    return adaptive_fragment_capabilities()


def _same_observation_references(context, core):
    """Replace exact repeated values only; the complete current core stays present."""
    references = {}

    def index(value, path):
        if isinstance(value, (dict, list)):
            key = _canonical(value)
            if len(key) >= 256:
                references.setdefault(key, path)
        if isinstance(value, dict):
            for name, child in value.items():
                index(child, path + "." + name)

    index(core, "current_decision")

    def project(value):
        if isinstance(value, (dict, list)):
            reference = references.get(_canonical(value))
            if reference:
                return {"same_observation_ref": reference}
        if isinstance(value, dict):
            return {key: project(child) for key, child in value.items()}
        if isinstance(value, list):
            return [project(child) for child in value]
        return copy.deepcopy(value)

    return project(context)


def _semantic_feedback(value):
    """Keep outcomes, causal evidence and interaction transitions, not sample arrays."""
    exact = {"missing_evidence_requests", "blocking_reasons", "advisories", "geometry_confirmation",
             "motion_context", "operation_state", "recent_events", "recovery_events", "goal"}
    scalars = {"status", "reason", "error", "error_type", "why", "action", "type", "kind",
               "semantic_result", "summary", "source", "phase", "intent_id", "operation_id",
               "object_id", "bound_object_id", "held_object_id", "target_id", "reference_id",
               "step", "actual_step", "observed_at_step", "started_step", "start_step", "end_step",
               "actual_command_step", "release_requested_step", "release_actual_command_step",
               "reference_step", "stage", "stage_index", "segment_index", "id", "skill",
               "actions", "executed_steps", "consumed_steps", "remaining_watchdog_steps",
               "remaining_chunk_steps", "stage_consumed_steps", "remaining_stage_count",
               "completed_stage_count", "completed_operation_count", "attempts", "recovery_count",
               "accepted", "done", "active", "paused", "verified", "gripper", "last_command",
               "commanded_gripper", "tracking_status", "execution_status", "execution_allowed",
               "progress", "position_progress_m", "orientation_progress_rad", "position_error_m",
               "orientation_error_rad", "hold_supported", "release_supported", "local_pose_revision",
               "conditional_probe_geometry_supported", "physical_clearance_verified", "task_success",
               "same_observation_ref", "last_gripper_command", "attached", "attachment_verified", "release_pending",
               "released", "held", "close_settled"}
    omitted = {"command_candidates", "sweep_refinement", "refinement", "sweep_samples",
               "coverage_witnesses", "channels", "normal_responses", "calibration",
               "before_state", "after_state", "start", "end", "trace", "samples"}
    if isinstance(value, dict):
        projected = {}
        for key, child in value.items():
            if key in exact:
                projected[key] = copy.deepcopy(child)
            elif key in omitted:
                continue
            elif isinstance(child, (dict, list)):
                kept = _semantic_feedback(child)
                if kept:
                    projected[key] = kept
            elif key in scalars:
                projected[key] = copy.deepcopy(child)
        return projected
    if isinstance(value, list):
        # Duplicate nested summaries refer to the same audit event; retain the
        # first occurrence, preserving order. No independent steps are merged.
        results, seen = [], set()
        for item in value:
            kept = _semantic_feedback(item) if isinstance(item, (dict, list)) else copy.deepcopy(item)
            identity = _canonical(kept)
            if kept is not None and kept != {} and kept != [] and identity not in seen:
                seen.add(identity)
                results.append(kept)
        return results
    return copy.deepcopy(value)


def _historical_decision(core):
    """The complete latest core remains canonical; old decisions retain their causes."""
    preserved = {key: copy.deepcopy(core[key]) for key in (
        "version", "step", "operation_state", "motion_context", "missing_evidence_requests",
        "blocking_reasons", "advisories",
        "requirements_truncated", "repeated_action_guard", "evidence_pause", "task_success") if key in core}
    pending = core.get("pending_intent")
    if isinstance(pending, dict):
        preserved["pending_intent"] = {key: copy.deepcopy(pending[key]) for key in (
            "intent_id", "kind", "motion_context", "current_target", "gripper_command", "phase",
            "stage_index", "current_operation_id", "remaining_watchdog_steps", "consumed_steps",
            "remaining_stage_count", "completed_stage_count", "completed_operations",
            "remaining_operation_ids", "missing_evidence_requests") if key in pending}
    for key in ("blocker", "whole_goal_assessment", "current_tracking", "evidence_acquisition"):
        if key in core:
            preserved[key] = _semantic_feedback(core[key])
    return preserved


def _feedback_events(value):
    """Deduplicate semantic snapshot copies, preserving distinct measured events."""
    events, requests, seen_events, seen_requests = [], [], set(), set()

    def visit(item):
        if isinstance(item, dict):
            event = {key: copy.deepcopy(child) for key, child in item.items()
                     if not isinstance(child, (dict, list))}
            for key in ("geometry_confirmation", "recent_events", "recovery_events", "goal"):
                if key in item:
                    event[key] = copy.deepcopy(item[key])
            if event:
                identity = _canonical(event)
                if identity not in seen_events:
                    seen_events.add(identity)
                    events.append(event)
            for key, child in item.items():
                if key in ("missing_evidence_requests", "blocking_reasons", "advisories"):
                    for request in child if isinstance(child, list) else [child]:
                        record = {"kind": key, "value": copy.deepcopy(request)}
                        identity = _canonical(record)
                        if identity not in seen_requests:
                            seen_requests.add(identity)
                            requests.append(record)
                elif isinstance(child, (dict, list)) and key not in (
                        "geometry_confirmation", "recent_events", "recovery_events", "goal"):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and item:
            event = {"note": item}
            identity = _canonical(event)
            if identity not in seen_events:
                seen_events.add(identity)
                events.append(event)

    visit(_semantic_feedback(value))
    return {"outcomes_and_transitions": events, "evidence_requirements": requests}


def project_agent_history(history):
    """Wire-only projection; never modify accepted outputs, sensor metadata or audits.

    Old capabilities/belief repeat later capabilities and accepted model
    outputs. Feedback retains semantic outcomes, interaction transitions and
    missing evidence; only the latest decision core remains unabridged.
    Non-protocol user text is never interpreted or rewritten.
    """
    projected = copy.deepcopy(history)
    user_indices = [index for index, item in enumerate(projected) if item.get("role") == "user"]
    latest = user_indices[-1] if user_indices else -1
    for index in user_indices:
        for part in projected[index].get("content", []):
            if part.get("type") != "input_text":
                continue
            try:
                fields = json.loads(part["text"])
            except (TypeError, ValueError):
                continue
            if not isinstance(fields, dict) or fields.get("request") != AGENT_KIND:
                continue
            context = fields.get("context")
            if not isinstance(context, dict):
                continue
            core = fields.get("current_decision")
            if index == latest and isinstance(core, dict):
                context = _same_observation_references(context, core)
            for key in ("execution", "last_feedback", "recent_feedback"):
                if key in context:
                    context[key] = _semantic_feedback(context[key])
            macro = context.get("macro")
            if (isinstance(macro, dict) and not macro.get("active")
                    and not macro.get("paused") and not macro.get("last_outcome")
                    and not macro.get("last_feedback") and not macro.get("result")):
                macro.pop("setup_progress", None)
            if index != latest:
                # The accepted assistant outputs remain byte-for-byte intact.
                # Do not discard previous local transitions hidden in feedback.
                for key in ("capabilities", "model_belief"):
                    context.pop(key, None)
                if isinstance(core, dict):
                    fields["current_decision"] = _historical_decision(core)
                # Pose/fingers/calibration remain beside the original image.
                # Historical controller snapshots become semantic events; the
                # current canonical state and original audit remain complete.
                measured = context.pop("measured_state", None)
                if isinstance(measured, dict):
                    context["last_gripper_command"] = measured.get("last_gripper_command")
                context = _feedback_events(context)
                fields["historical_observation"] = True
            fields["context"] = context
            fields["wire_projection"] = "semantic_feedback; latest_core_exact; original_in_audit"
            part["text"] = json.dumps(fields, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return projected


def prompt_size_snapshot(payload):
    """Character accounting, not a token/latency estimate; images are counted separately."""
    inputs = payload.get("input", [])
    parts = [part for item in inputs for part in item.get("content", [])
             if isinstance(part, dict)]
    input_text = sum(len(part.get("text", "")) for part in parts if part.get("type") == "input_text")
    output_text = sum(len(part.get("text", "")) for part in parts if part.get("type") == "output_text")
    instructions = len(payload.get("instructions", ""))
    schema = len(ordered_dumps(payload.get("text", {}).get("format", {}).get("schema", {})))
    return {"instructions_chars": instructions, "input_text_chars": input_text,
            "accepted_output_text_chars": output_text, "schema_chars": schema,
            "total_prompt_chars": instructions + input_text + output_text + schema,
            "image_count": sum(part.get("type") == "input_image" for part in parts),
            "opaque_reasoning_items": sum(item.get("type") == "reasoning" for item in inputs),
            "excludes": "image bytes, opaque reasoning bytes and transport formatting; not token counts"}


class AgentSpeedUpV3Client(V3SubscriptionAvailabilityRetryMixin, AgentFirstV2Client):
    """Reuse subscription/high execution and raw history with v3-only output fields."""

    client_version = CLIENT_VERSION
    prompt_version = PROMPT_VERSION
    history_protocol = HISTORY_PROTOCOL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instruction_text[AGENT_KIND] = AGENT_INSTRUCTIONS
        self._schemas[AGENT_KIND] = v3_agent_schema()
        self._unprojected_agent_input = None
        self._prompt_sizes = None

    def request_agent(self, view, context):
        # Capture the core before inherited execution-detail compaction. Keep
        # the shared client untouched and keep full raw audits/history intact.
        core = current_decision_core(context)
        clean = copy.deepcopy(context)
        native = clean.get("native_evidence")
        if isinstance(native, dict):
            native.pop("current_decision", None)
        self._current_decision_for_request = core
        self._original_context_for_audit = copy.deepcopy(context)
        try:
            return super().request_agent(view, clean)
        finally:
            self._current_decision_for_request = None
            self._original_context_for_audit = None

    def predict(self, request, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            return super().predict(request, grid, mode)
        # The inherited client commits its wire payload. Restore the exact raw
        # accepted input after that successful commit, under the same RLock.
        # Failed attempts never append observations or change accepted history.
        with self._history_lock:
            result = super().predict(request, grid, mode)
            self._history = copy.deepcopy(self._unprojected_agent_input) + copy.deepcopy(self.last_response["output"])
            self._agent_history = copy.deepcopy(self._history)
            return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_PROTOCOL
        result["client_version"] = CLIENT_VERSION
        result["geometry_updates"] = "bounded_edits_in_existing_executive_call"
        result["prompt_sizes"] = copy.deepcopy(self._prompt_sizes)
        result["history_projection"] = "wire_only; accepted_transcript_unprojected"
        return result

    def _write_log(self, path, data):
        if isinstance(data, dict):
            data = {**data, "agent_speedup_v3_client_version": CLIENT_VERSION,
                    "agent_speedup_v3_prompt_version": PROMPT_VERSION,
                    "agent_speedup_v3_history_protocol": HISTORY_PROTOCOL}
            if "payload" in data and "request_ordinal" in data:
                data["prompt_sizes"] = copy.deepcopy(self._prompt_sizes)
                # Save only this request's raw observation, not another copy of
                # the whole transcript. Earlier logs and accepted outputs are immutable.
                raw = self._unprojected_agent_input or []
                data["original_observation"] = copy.deepcopy(raw[-1]) if raw else None
                original_context = getattr(self, "_original_context_for_audit", None)
                if original_context is not None:
                    data["original_context"] = copy.deepcopy(original_context)
        return super()._write_log(path, data)

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            return super().build_payload(request, grid, mode)
        # The inherited request_agent adds v7 capabilities. Replace only that
        # advertised block, preserving its sensor validation and context allowlist.
        request = copy.deepcopy(request)
        fields = request.setdefault("fields", {})
        context = fields.get("context") if isinstance(fields, Mapping) else None
        if isinstance(context, dict):
            capabilities = context.setdefault("capabilities", {})
            if not isinstance(capabilities, dict):
                raise ValueError("v3 capabilities must be an object")
            capabilities["execute"] = v3_fragment_capabilities()
        core = getattr(self, "_current_decision_for_request", None)
        if core is None:
            core = current_decision_core(context or {})
        fields["current_decision"] = copy.deepcopy(core)
        payload = super().build_payload(request, grid, mode)
        payload["instructions"] = AGENT_INSTRUCTIONS
        payload["text"]["format"].update(name="agent_speedup_v3", schema=v3_agent_schema())
        self._unprojected_agent_input = copy.deepcopy(payload["input"])
        before = prompt_size_snapshot(payload)
        payload["input"] = project_agent_history(payload["input"])
        self._prompt_sizes = {"before_wire_projection": before, "wire": prompt_size_snapshot(payload)}
        if isinstance(self.last_request_meta, dict):
            self.last_request_meta.update(protocol=HISTORY_PROTOCOL, client_version=CLIENT_VERSION)
        return payload

    def request_key(self, payload, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            return super().request_key(payload, grid, mode)
        material = {"wire_sha256": hashlib.sha256(ordered_dumps(payload).encode()).hexdigest(),
                    "client_version": CLIENT_VERSION, "prompt_version": PROMPT_VERSION,
                    "history_protocol": HISTORY_PROTOCOL, "endpoint": self.config.endpoint}
        return hashlib.sha256(_canonical(material).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        if self._request_kind != AGENT_KIND:
            return super().validate_response(data, grid)
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        return validate_v3_agent_decision(data)
