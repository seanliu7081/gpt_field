"""Stateless subscription client for the agent_SpeedUp_v1 supervisor.

Two request kinds share one audited predict path, usage ledger and log set:
decision requests (upright external + wrist RGB and a compact state summary ->
one macro skill with discrete parameters) and pointing requests (one marked
wrist image -> one numbered cell or point). Every request is a single user
message; accepted outputs are never retained, so there is no transcript or
image history. The body is serialized in insertion order, so ``situation``
precedes the action fields on the wire. A confirmed provider overload is
retried with the bounded speed-first policy. Images are never flipped here;
the policy supplies upright frames. No simulator access.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

from .astra import ENDPOINT, APIError, SchemaError, _canonical
from .chatgpt_plan import UNSUPPORTED_FIELDS, _failure_metadata
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient
from .ordered_transport import ORDERED_TRANSPORT_VERSION, ordered_dumps, ordered_sse_transport
from .speed_first_hybrid import OVERLOAD_RETRY_POLICY
from .target_identity_guidance import normalize_target_description


AGENT_SPEEDUP_V1_VERSION = "agent-speedup-v1"
AGENT_PROMPT_VERSION = "agent-speedup-v1-prompt-5"
HISTORY_PROTOCOL = "agent-speedup-v1-stateless"
SKILLS = ("reach", "grasp_with", "pull", "transport", "place_release", "retreat")
STRATEGIES = ("rim_pinch", "frontal_bar_pinch")
SIDES = ("wrist_left", "wrist_right")
APPROACHES = ("world_-y", "world_+y", "world_-x", "world_+x")
TASK_TARGETS = {"bowl_plate": ("black_bowl", "plate"),
                # "bowl", not the asset name: the drawer scene's only bowl is white with a dark pattern and the
                # model began hunting a phantom "black" bowl after failures (drawer live 002)
                "drawer_bowl": ("top_drawer_handle", "bowl", "top_drawer_interior")}
MAX_WAYPOINT_M = 0.075
MAX_TRAVEL_RANGE_M = (0.02, 0.16)
SITUATION_MAX_CHARS = 280
MARK_COUNT = 64
PARAM_KEYS = ("target", "strategy", "side", "approach", "waypoint_offset_m", "more_waypoints_m", "max_travel_m")
MAX_EXTRA_WAYPOINTS = 2
REQUEST_KINDS = ("decision", "pointing_coarse", "pointing_fine")
POINTING_KEYS = {"coarse": "cell_id", "fine": "point_id"}
FORMAT_NAMES = {"decision": "agent_speedup_v1_decision",
                "pointing_coarse": "agent_speedup_v1_pointing_coarse",
                "pointing_fine": "agent_speedup_v1_pointing_fine"}
MAX_REQUEST_TEXT_CHARS = 60_000
MAX_TARGET_PHRASE_CHARS = 500
_KIND_BY_FORMAT = {name: kind for kind, name in FORMAT_NAMES.items()}
_DECISION_KEYS = ("situation", "skill", "params")
_PNG_PREFIX = "data:image/png;base64,"


# Schemas ------------------------------------------------------------------
def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _enum(values) -> dict:
    return {"type": "string", "enum": list(values)}


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def _task_targets(task_key) -> tuple[str, ...]:
    if not isinstance(task_key, str) or task_key not in TASK_TARGETS:
        raise ValueError(f"Unknown agent task; choose one of {', '.join(TASK_TARGETS)}")
    return TASK_TARGETS[task_key]


def _pointing_key(stage) -> str:
    if not isinstance(stage, str) or stage not in POINTING_KEYS:
        raise ValueError("pointing stage must be 'coarse' or 'fine'")
    return POINTING_KEYS[stage]


def decision_schema(task_key) -> dict:
    """Flat strict object; properties in wire order situation, skill, params."""
    low, high = MAX_TRAVEL_RANGE_M
    params = _object({
        "target": _nullable(_enum(_task_targets(task_key))),
        "strategy": _nullable(_enum(STRATEGIES)),
        "side": _nullable(_enum(SIDES)),
        "approach": _nullable(_enum(APPROACHES)),
        "waypoint_offset_m": _nullable({"type": "array", "minItems": 3, "maxItems": 3, "items": {
            "type": "number", "minimum": -MAX_WAYPOINT_M, "maximum": MAX_WAYPOINT_M}}),
        "more_waypoints_m": _nullable({"type": "array", "minItems": 1, "maxItems": MAX_EXTRA_WAYPOINTS, "items": {
            "type": "array", "minItems": 3, "maxItems": 3, "items": {
                "type": "number", "minimum": -MAX_WAYPOINT_M, "maximum": MAX_WAYPOINT_M}}}),
        "max_travel_m": _nullable({"type": "number", "minimum": low, "maximum": high}),
    })
    return _object({"situation": {"type": "string", "minLength": 1, "maxLength": SITUATION_MAX_CHARS},
                    "skill": _enum(SKILLS), "params": params})


def pointing_schema(stage) -> dict:
    # Plain bounded integer (a live-proven construct); 0 means "not visible" and is mapped to None.
    return _object({_pointing_key(stage): {"type": "integer", "minimum": 0, "maximum": MARK_COUNT}})


# Instructions -------------------------------------------------------------
def decision_instructions(task_key, target_description=None) -> str:
    targets = _task_targets(task_key)
    task = resolve_distance_task(task_key)
    description = normalize_target_description(target_description)
    text = (
        f"{AGENT_PROMPT_VERSION}. You supervise a robot arm with a parallel-jaw gripper. "
        f"Task: \"{task.prompt}\". Zero-shot: no demonstrations, object poses, depth or camera "
        "calibration are available.\n"
        "ROLE. Each request, choose ONE macro skill and its discrete parameters. A local executor runs "
        "it at 20 Hz with its own sensing: it aligns the gripper on a target with the wrist camera, "
        "finds depth by touch (motion stall or finger deflection), closes the gripper only after its "
        "own sensor checks pass, verifies a grasp with a short lift or pull, and retries a failed grasp "
        "locally. You never output poses, rotations, depths or gripper commands. You are asked again "
        "when the skill ends; state_summary then holds a ledger of finished skills and the last outcome "
        "(status, reason, evidence). The episode starts with an automatic gripper calibration.\n"
        "INPUTS. Both images are UPRIGHT (not mirrored). state_summary gives the measured end-effector "
        "position (the grip point midway between the fingertips; world frame, metres, +z up), the "
        "finger gap in mm, and tool_axes_world: world unit vectors for wrist-image right (the jaw "
        "axis, along which the fingers open), wrist-image up, and the approach axis (where the wrist "
        "camera looks and the fingers point). tool_axes_world apply ONLY to the WRIST image: something "
        "seen to the right in the wrist image lies along +wrist_image_right, something seen LOW in it lies "
        "along -wrist_image_up (the grip axis is below the image centre, see below). The EXTERNAL image comes "
        "from a different, fixed camera with its own axes; never convert external-image directions with "
        "tool_axes_world. external_image_axes_world (when present) gives the world horizontal direction "
        "for right and up in the EXTERNAL image, measured from the hand's own probe moves at the episode "
        "start (image_right: right in that view; image_up: farther back). "
        "Ledger lines' external_image_shift_px_right_down give each move's hand shift in that image "
        "(+ = right / down): check that moves went as intended. The "
        "external image's vertical direction mixes height and depth; judge heights from state_summary and "
        "the wrist view instead. The "
        "wrist camera sits 5 cm from the grip axis toward wrist-image up, so a point in line with the "
        "fingers appears below the wrist-image centre; the fingertips show at the bottom of the wrist "
        "image. world_up_in_wrist_image names the wrist-image direction of world up (e.g. right when the "
        "jaw is vertical); higher objects lie that way. approach_tilt_from_straight_down_deg is 0 for a "
        "top-down hand. Robot facts: maximum "
        "finger opening 79 mm; the jaw is horizontal in the wrist image, so a bar that can be pinched "
        "appears vertical between the fingers. The arm's reach is limited: near the edge of its "
        "workspace the hand cannot stay top-down and tilts; the executor then stops with reach_limit. "
        "robot_base_direction_world points from the grip toward the robot base (horizontal). With a turned "
        "hand the arm locks at full stretch when its wrist is too far from the shoulder "
        "(state_summary.frontal_arm_reach); such hovers and moves are refused: approach handles from the "
        "robot-side half, nearer the base. A turned (jaw-vertical) hand extends about "
        "10 cm above and below the grip point: keep about 11 cm between the grip point and the table or "
        "objects below it, also while it turns.\n"
        "SKILLS (set every parameter a skill does not use to null):\n"
        f"- reach(waypoint_offset_m): move the grip point by [dx, dy, dz] world metres relative to its "
        f"CURRENT position, each component within +/-{MAX_WAYPOINT_M:g}; chain calls for longer moves. "
        "Orientation and gripper are unchanged. It does not see obstacles: route around objects with "
        "intermediate waypoints (e.g. up first; move horizontally at a clear height, then descend): "
        "more_waypoints_m adds up to 2 further offsets run in order "
        "in the same call (a whole detour in one decision; it stops at the first waypoint that fails). Moves "
        "that would push further past a recorded reach_limit (state_summary.reach_limits) are rejected.\n"
        "- grasp_with(target, strategy, side or approach, max_travel_m for a handle): sensor-gated grasp of "
        "target with a strategy "
        "below. Meanwhile the executor asks you separate short pointing questions (numbered marks on "
        "the wrist image).\n"
        f"- pull(max_travel_m): with a grasped bar, pull back opposite to the approach axis until the "
        f"TOTAL travel since the grasp (including the 2 cm verification pull; see "
        f"state_summary.pulled_since_grasp_m) reaches max_travel_m ({MAX_TRAVEL_RANGE_M[0]:g}-"
        f"{MAX_TRAVEL_RANGE_M[1]:g} m); stops early at a stall or end-stop. Do not over-pull: an "
        "over-extended drawer can block later moves, and an open drawer front overhangs what lies in front of the "
        "cabinet.\n"
        "- transport(waypoint_offset_m, more_waypoints_m, target=destination): like reach while holding an "
        "object, keeping "
        "the grasp orientation; it stops at once if the hold is lost (lost_in_transport) or if the held "
        "object presses against something (held_object_contact; the hand then backs off about 1 cm: up, or "
        "against a rising move). While "
        "holding, the wrist view is mostly blocked by the object: judge the destination in the EXTERNAL "
        "image with external_image_axes_world. Carry a held object 3-6 cm above its pick height "
        "(state_summary.held_height_above_pick_m) when the path is clear, otherwise high enough to pass over "
        "what is in the way (e.g. above an open drawer's front edge); move it out from under anything "
        "overhanging it before lifting it high. A raised object looks farther "
        "back in the external image than its footprint. Within about 5 cm of the destination use moves "
        "of at most 3 cm.\n"
        "- place_release(waypoint_offset_m, target=destination): move horizontally (and up, never down) "
        "by the offset to bring the held object over the destination, then lower with touch sensing "
        "until the object meets a support (down to 3 cm below the pick height), open, back away 3 cm. "
        "Aim the held object's centre (not the grip point) over the destination centre. With [0, 0, 0] "
        "it lowers and releases at the current position; for a held handle it opens, backs away, lifts 5 cm "
        "and turns the hand top-down (the way to let go of a handle).\n"
        "- retreat(): back away 3 cm opposite to the approach axis, gripper unchanged; it keeps any "
        "grasp and never releases (clearance or a better view).\n"
        "STRATEGIES:\n"
        "- rim_pinch (bowls, open containers): the executor uses the top-down episode-start "
        "orientation; if the hand is turned (e.g. after a handle), grasp_with first restores it and "
        "returns reason orientation_restored so you can choose side in the new top-down wrist view. side (wrist_left | wrist_right) picks the rim point at the left or right of the "
        "container in the UPRIGHT wrist image. The executor puts the grip axis over that rim so the "
        "wall goes between the open fingers, lowers until a fingertip touches the container floor, "
        "then closes on the wall. Choose the side with free space next to the container (no "
        "neighbouring object within about 4 cm). Never centre the hand over a container wider than "
        "the 79 mm opening. The hand body is about 20 cm long along the jaw (wrist-image left-right), 6 cm "
        "across, its underside about 3 cm above the grip point: when the fingers reach the container floor, "
        "nothing taller than the container may lie within 10 cm of the grip along the jaw or 3 cm across (an "
        "open drawer counts). The executor aligns on the pointed rim from any hover where the whole "
        "container is visible, so after an aligned failure change the side or clear the obstacle rather "
        "than the hover. Before grasp_with, reach a hover 5-10 cm above the rim (or just above anything overhanging it, e.g. an "
        "open drawer front) with the whole "
        "container, including the chosen side, visible in the wrist image while the hand is top-down "
        "(a tilted camera looks forward and can show objects that are out of top-down reach). If "
        "grasp_with fails with orient_budget, orient_stall or pose_drift_reach_limit, the hand cannot "
        "become top-down there: move along robot_base_direction_world and up before trying again; an object "
        "beyond that point is out of reach for rim_pinch. A container held this way "
        "hangs off-centre: its centre is about one radius from the grip point along the jaw axis, "
        "toward wrist-image right for wrist_left and left for wrist_right.\n"
        "- frontal_bar_pinch (handles, horizontal bars): approach (world_-y | world_+y | world_-x | "
        "world_+x) is the horizontal world direction in which the hand moves toward the bar, from its "
        "free front side (e.g. a bar whose front faces world +x needs world_-x). The executor turns the "
        "hand to that horizontal approach with the jaw vertical, aligns the grip axis on the bar, "
        "advances until the fingertips touch the surface behind the bar, backs off slightly, closes, "
        "and verifies with a short pull. Before grasp_with, reach a hover on the approach side, about "
        "level with the bar and 6-10 cm in front of it, with free space between hand and bar. Give "
        "max_travel_m to grasp and then pull in the same call (or use pull afterwards).\n"
        "REASONS (outcome.reason): goal_reached = arrived; reach_limit = the hand could not keep its "
        "orientation moving that way (workspace edge or contact; do not push further that way, move the "
        "other way); finger_contact = an open finger touched something (go around it, up first); "
        "uncontrolled_motion = the arm jumped; left_planning_grid = the hand overshot the move; blocked = "
        "little progress (contact, or the arm cannot move that way in this orientation); slow_arrival = most "
        "of the move done, out of steps; orientation_drift = "
        "position reached, orientation drifted; no_progress = no motion; "
        "held_object_contact = the held object hit something; lost_in_transport = the object dropped; "
        "no_support_contact_within_bound = lowered to 3 cm below the pick height without meeting a "
        "support; orientation_restored = the hand was turned top-down, choose again in the new view; "
        "orient_budget / orient_stall / pose_drift_reach_limit = the hand could not be turned to the "
        "grasp orientation here; beyond_recorded_reach_limit = rejected (past a known limit); "
        "grasp_with failures (local retries already used; the detail says what to change): mark_not_visible / "
        "pointed_not_a_bar / textureless_mark = the target point was not found in the wrist view; "
        "hover_beyond_frontal_reach / bar_point_beyond_reach / beyond_frontal_reach / frontal_stretch_limit = too "
        "far from the robot base for a turned hand; probe_failed "
        "/ align_no_convergence / align_blocked / approach_drift / hover_off_bar_across / bar_point_far_along = "
        "the hand could not be aligned on it from this hover; no_contact_within_bound = nothing was touched; "
        "empty_close = the fingers closed on nothing; blocked_above_rim = the hand body landed on something before "
        "the fingers reached the rim (outward-shifted retries included); hold_lost / hold_jam = the grip did not hold; "
        "hover_<reason> = place_release's hover ended with <reason>.\n"
        f"RULES. Targets: {', '.join(targets)}. After a failure (local retries are already used), "
        "change something (hover position, side, strategy or target) instead of repeating the same "
        "choice; a failed close may have pushed the object, so re-check the images. A succeeded "
        "grasp_with means the executor verified the hold. If an object was dropped "
        "(state_summary.target_history.lost_at_step), it is still the task object: look for it near "
        "lost_eef_xyz, even if it no longer matches its described place. state_summary.loop_warning "
        "means recent choices keep failing: change the plan. You never declare completion; the episode "
        "is ended externally. If the task looks complete, choose retreat. Write situation first: one "
        "or two short sentences on what you see and where the target is relative to the fingers; while "
        "holding an object, say instead where the destination is relative to the held object in the "
        "EXTERNAL image and how high the object is above its pick height."
    )
    if description is not None:
        text += ("\nTarget appearance (user-provided from the initial RGB view; not a pose or position): "
                 + json.dumps(description, ensure_ascii=False) + ". Use it to identify the task objects.")
    return text


def pointing_instructions(stage) -> str:
    key = _pointing_key(stage)
    if stage == "coarse":
        body = (
            "The image is the robot's wrist-camera view (not mirrored) with an 8x8 grid of numbered cells, "
            "1-64 row-major from the top-left (1-8 top row, 57-64 bottom row). Choose the single cell "
            "that contains the target point; if the point lies on a border, choose the cell holding "
            "most of the target near the point.")
    else:
        body = (
            "The image is an enlarged crop of the robot's wrist-camera view (not mirrored) around the cell chosen before, with 64 "
            "numbered points in an 8x8 pattern, 1-64 row-major from the top-left. Choose the single point "
            "that lies on the target point (the nearest point if none lies exactly on it).")
    return (
        f"{AGENT_PROMPT_VERSION} pointing, {stage} stage. You help a robot locate one target point. "
        "target_phrase names the point; context adds details such as the strategy and side. " + body +
        " Orientation: if context.image_orientation is present, the view was turned as it states (world up "
        "at the top) and the robot's fingertips are at the edge it names; otherwise the view is upright "
        "and the fingertips are at the bottom. The robot's own fingertips are never the target. If "
        "context.previous_answer_rejected is present, your earlier answer failed a check; choose again. "
        f"Return {{\"{key}\": <id>}}, or {{\"{key}\": 0}} if the target point is not visible in this "
        "image. Answer only this question."
    )


# Validation ---------------------------------------------------------------
def _png_data_url(value, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(_PNG_PREFIX):
        raise ValueError(f"{name} must be an inline PNG data URL")
    try:
        decoded = base64.b64decode(value[len(_PNG_PREFIX):], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} has invalid base64") from exc
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{name} is not PNG data")
    return value


def _json_text(value, name: str) -> str:
    try:
        text = ordered_dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-safe (plain types, finite numbers)") from exc
    if len(text) > MAX_REQUEST_TEXT_CHARS:
        raise ValueError(f"{name} exceeds {MAX_REQUEST_TEXT_CHARS} characters")
    return text


def _exact_keys(request, keys, name: str) -> Mapping:
    if not isinstance(request, Mapping) or set(request) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(keys)}")
    return request


def _real(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise SchemaError(f"{name} is outside [{low:g}, {high:g}]")
    return float(value)


def _choice(value, name: str, allowed) -> str | None:
    if value is not None and (not isinstance(value, str) or value not in allowed):
        raise SchemaError(f"{name} must be one of {', '.join(allowed)} or null")
    return value


def _validate_decision(data, task_key) -> dict:
    if not isinstance(data, dict) or set(data) != set(_DECISION_KEYS):
        raise SchemaError("decision output must contain exactly situation, skill and params")
    situation = data["situation"]
    if not isinstance(situation, str) or len(situation) > SITUATION_MAX_CHARS:
        raise SchemaError(f"situation must be a string of at most {SITUATION_MAX_CHARS} characters")
    skill = data["skill"]
    if not isinstance(skill, str) or skill not in SKILLS:
        raise SchemaError(f"skill must be one of {', '.join(SKILLS)}")
    params = data["params"]
    if not isinstance(params, dict) or set(params) != set(PARAM_KEYS):
        raise SchemaError(f"params must contain exactly {', '.join(PARAM_KEYS)}")
    offset = params["waypoint_offset_m"]
    if offset is not None:
        if not isinstance(offset, list) or len(offset) != 3:
            raise SchemaError("waypoint_offset_m must be three numbers or null")
        offset = [_real(item, "waypoint_offset_m", -MAX_WAYPOINT_M, MAX_WAYPOINT_M) for item in offset]
    more = params["more_waypoints_m"]
    if more is not None:
        if not isinstance(more, list) or not 1 <= len(more) <= MAX_EXTRA_WAYPOINTS:
            raise SchemaError(f"more_waypoints_m must be 1-{MAX_EXTRA_WAYPOINTS} offsets or null")
        checked = []
        for item in more:
            if not isinstance(item, list) or len(item) != 3:
                raise SchemaError("each more_waypoints_m entry must be three numbers")
            checked.append([_real(v, "more_waypoints_m", -MAX_WAYPOINT_M, MAX_WAYPOINT_M) for v in item])
        more = checked
    travel = params["max_travel_m"]
    if travel is not None:
        travel = _real(travel, "max_travel_m", *MAX_TRAVEL_RANGE_M)
    return {"situation": situation, "skill": skill, "params": {
        "target": _choice(params["target"], "target", _task_targets(task_key)),
        "strategy": _choice(params["strategy"], "strategy", STRATEGIES),
        "side": _choice(params["side"], "side", SIDES),
        "approach": _choice(params["approach"], "approach", APPROACHES),
        "waypoint_offset_m": offset, "more_waypoints_m": more, "max_travel_m": travel}}


def _validate_pointing(data, stage) -> dict:
    key = _pointing_key(stage)
    if not isinstance(data, dict) or set(data) != {key}:
        raise SchemaError(f"{stage} pointing output must contain exactly {key}")
    value = data[key]
    if value is None:
        return {key: None}
    if type(value) is not int or not 0 <= value <= MARK_COUNT:
        raise SchemaError(f"{key} must be an integer in [0, {MARK_COUNT}]")
    return {key: value or None}


# Client -------------------------------------------------------------------
class AgentSpeedUpV1Client(ChatGPTPlanHybridClient):
    """Stateless decision/pointing requests over the audited subscription path."""

    history_protocol = HISTORY_PROTOCOL

    def __init__(self, config=None, *, token_provider, task_key="bowl_plate", target_description=None,
                 transport=None, log_dir=None, clock=time.monotonic):
        if config is not None and getattr(config, "model", None) != "gpt-6-astra":
            raise ValueError("agent_SpeedUp_v1 requires gpt-6-astra")
        _task_targets(task_key)
        # Everything reset_history/history_stats touch must exist before super():
        # DirectHistoryAstraClient.__init__ calls reset_history().
        self.task_key = task_key
        self.task_spec = resolve_distance_task(task_key)
        self.target_description = normalize_target_description(target_description)
        self._request_kind = None
        self.last_output_key_order = None
        self._instruction_text = {
            "decision": decision_instructions(task_key, self.target_description),
            "pointing_coarse": pointing_instructions("coarse"),
            "pointing_fine": pointing_instructions("fine")}
        self._schemas = {"decision": decision_schema(task_key),
                         "pointing_coarse": pointing_schema("coarse"),
                         "pointing_fine": pointing_schema("fine")}
        super().__init__(config, reasoning_effort="medium", token_provider=token_provider,
                         transport=transport if transport is not None else ordered_sse_transport,
                         log_dir=log_dir, clock=clock, task_prompt=self.task_spec.prompt, image_history=1)

    @property
    def prompt_version(self) -> str:
        return AGENT_PROMPT_VERSION

    @property
    def _offline_mock(self) -> bool:
        return getattr(self.transport, "offline_mock", False) is True

    def reset_history(self) -> None:
        with self._history_lock:
            super().reset_history()
            self._overload_attempt = 1
            self._overload_retry_requests = 0
            self._overload_retry_wait_s = 0.
            self._kind_calls = {kind: 0 for kind in REQUEST_KINDS}
            self._kind_accepted = {kind: 0 for kind in REQUEST_KINDS}

    def history_stats(self) -> dict:
        with self._history_lock:
            return {"protocol": HISTORY_PROTOCOL, "accepted_requests": self._observation_count,
                    "messages": len(self._history), "retained_image_pairs": 0,
                    "calls_by_kind": dict(self._kind_calls),
                    "accepted_by_kind": dict(self._kind_accepted),
                    "overload_retry_policy": OVERLOAD_RETRY_POLICY,
                    "overload_retry_requests": self._overload_retry_requests,
                    "overload_retry_wait_s": self._overload_retry_wait_s}

    def budget_snapshot(self) -> dict:
        snapshot = super().budget_snapshot()
        if self._offline_mock:
            snapshot = {**snapshot, "billing": "offline_mock", "offline_mock": True}
        return snapshot

    # payload --------------------------------------------------------------
    def _content(self, request, mode: str) -> list[dict]:
        if mode == "decision":
            request = _exact_keys(request, ("state_summary", "external_image", "wrist_image"),
                                  "decision request")
            if not isinstance(request["state_summary"], Mapping):
                raise ValueError("state_summary must be a JSON object")
            text = _json_text({"request": "decision", "task": self.task_prompt,
                               "state_summary": request["state_summary"]}, "state_summary")
            return [{"type": "input_text", "text": text},
                    {"type": "input_text", "text": "External camera (upright):"},
                    {"type": "input_image", "detail": "high",
                     "image_url": _png_data_url(request["external_image"], "external_image")},
                    {"type": "input_text", "text": "Wrist camera (upright):"},
                    {"type": "input_image", "detail": "high",
                     "image_url": _png_data_url(request["wrist_image"], "wrist_image")}]
        stage = mode.split("_", 1)[1]
        request = _exact_keys(request, ("stage", "image", "target_phrase", "context"), "pointing request")
        if request["stage"] != stage:
            raise ValueError("pointing request stage does not match the request mode")
        phrase = request["target_phrase"]
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > MAX_TARGET_PHRASE_CHARS:
            raise ValueError("target_phrase must be a nonempty string of at most "
                             f"{MAX_TARGET_PHRASE_CHARS} characters")
        context = request["context"] if request["context"] is not None else {}
        if not isinstance(context, Mapping):
            raise ValueError("pointing context must be a JSON object")
        text = _json_text({"request": "pointing", "stage": stage, "task": self.task_prompt,
                           "target_phrase": phrase.strip(), "context": context}, "pointing context")
        turned = "image_orientation" in context
        view = "turned so world up is at the top" if turned else "upright"
        label = (f"Wrist camera ({view}) with 64 numbered grid cells:" if stage == "coarse"
                 else f"Wrist-camera crop ({view}, enlarged) with 64 numbered points:")
        return [{"type": "input_text", "text": text}, {"type": "input_text", "text": label},
                {"type": "input_image", "detail": "high",
                 "image_url": _png_data_url(request["image"], "image")}]

    def build_payload(self, request, grid=None, mode="decision") -> dict[str, Any]:
        """Preview-safe: no authentication, no history, no network."""
        if mode not in REQUEST_KINDS:
            raise ValueError(f"mode must be one of {', '.join(REQUEST_KINDS)}")
        if grid is not None and grid != self.grid:
            raise ValueError("Requested grid must match the configured client grid")
        content = self._content(request, mode)
        self._request_kind = mode
        return {
            "model": self.config.model, "service_tier": "default", "store": False,
            "instructions": self._instruction_text[mode],
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": FORMAT_NAMES[mode], "strict": True,
                                "schema": copy.deepcopy(self._schemas[mode])}},
            "stream": True, "reasoning": {"effort": self.reasoning_effort},
            "include": ["reasoning.encrypted_content"],
        }

    def request_key(self, payload, grid, mode) -> str:
        material = {"wire_sha256": hashlib.sha256(ordered_dumps(payload).encode("utf-8")).hexdigest(),
                    "mode": mode, "client_version": AGENT_SPEEDUP_V1_VERSION,
                    "prompt_version": AGENT_PROMPT_VERSION, "history_protocol": HISTORY_PROTOCOL,
                    "transport_version": ORDERED_TRANSPORT_VERSION, "endpoint": self.config.endpoint}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def validate_response(self, data, grid) -> dict:
        """Structure only; skill/parameter semantics belong to the policy."""
        kind = self._request_kind or "decision"
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            if kind == "decision":
                return _validate_decision(data, self.task_key)
            return _validate_pointing(data, kind.split("_", 1)[1])
        except SchemaError:
            raise
        except Exception as exc:
            raise SchemaError(f"Malformed agent output ({type(exc).__name__})") from None

    def _history_input(self, payload) -> list[dict]:
        return []

    def _write_log(self, path, data: Any) -> None:
        request_record = isinstance(data, dict) and "payload" in data and "request_ordinal" in data
        if isinstance(data, dict):
            extra = {"wire_transport_version": ORDERED_TRANSPORT_VERSION,
                     "agent_speedup_v1_version": AGENT_SPEEDUP_V1_VERSION}
            if self._offline_mock:
                extra["offline_mock"] = True
            if request_record:
                data = {**data, "attempt": self._overload_attempt,
                        "overload_retry_policy": OVERLOAD_RETRY_POLICY, **extra}
            elif data.get("status") in ("valid", "error"):
                data = {**data, "request_kind": self._request_kind,
                        "output_key_order": self.last_output_key_order, **extra}
                if data["status"] == "error":
                    data.update(error="Subscription attempt failed; no API-key fallback",
                                overload_retry_policy=OVERLOAD_RETRY_POLICY)
            else:
                data = {**data, **extra}
        super()._write_log(path, data)
        if request_record:
            self._kind_calls[data["mode"]] += 1

    # requests -------------------------------------------------------------
    def predict(self, request, grid=None, mode="decision") -> dict[str, Any]:
        """One stateless request with the bounded same-request overload retry
        (copied from ChatGPTPlanSpeedFirstClient.predict). History is cleared
        after every call, accepted or failed."""
        if mode not in REQUEST_KINDS:
            raise ValueError(f"mode must be one of {', '.join(REQUEST_KINDS)}")
        with self._history_lock:
            same_request = copy.deepcopy(request)
            initial_observation_count = self._observation_count
            self._overload_attempt = 1
            try:
                while True:
                    self.last_output_key_order = None
                    try:
                        result = super().predict(same_request, grid, mode)
                    except APIError as exc:
                        if (self._overload_attempt >= 3
                                or getattr(exc, "error_code", None) != "server_is_overloaded"
                                or getattr(exc, "http_status", None) not in (None, 503)
                                or self.last_response is not None
                                or self._observation_count != initial_observation_count):
                            raise
                        delay = 2. ** self._overload_attempt + random.uniform(0., 1.)
                        delay = max(delay, _failure_metadata(exc).get("retry_after_s", 0.))
                        if delay > 60.:
                            raise
                        self._reserve()
                        if (self.config.max_wall_time_s is not None
                                and self.clock() - self._started + delay >= self.config.max_wall_time_s):
                            raise
                        record = {
                            "protocol": OVERLOAD_RETRY_POLICY, "status": "bounded_retry_scheduled",
                            **_failure_metadata(exc), "attempt": self._overload_attempt,
                            "next_attempt": self._overload_attempt + 1, "max_attempts": 3,
                            "wait_s": delay, "same_observation": True, "request_kind": mode,
                            "observation_index": initial_observation_count,
                            "failed_request_key": self.last_request_key,
                            "failed_request_ordinal": self._api_calls,
                            "actions_executed_from_failed_response": 0,
                            "budget": self.budget_snapshot(),
                        }
                        path = (Path(self.last_log_path).with_suffix(".overload_retry.json")
                                if self.last_log_path is not None else None)
                        self._write_log(path, record)
                        started = self.clock()
                        try:
                            time.sleep(delay)
                        finally:
                            self._overload_retry_wait_s += max(0., self.clock() - started)
                        self._overload_attempt += 1
                        self._overload_retry_requests += 1
                    else:
                        self._kind_accepted[mode] += 1
                        return result
            finally:
                self._overload_attempt = 1
                self._history = []

    def decide(self, request: dict) -> dict:
        return self.predict(request, mode="decision")

    def point(self, request: dict) -> dict:
        stage = request.get("stage") if isinstance(request, Mapping) else None
        _pointing_key(stage)
        return self.predict(request, mode=f"pointing_{stage}")


# Offline fixtures -----------------------------------------------------------
def _null_params() -> dict:
    return {key: None for key in PARAM_KEYS}


def _hold_decision() -> dict:
    return {"situation": "scripted smoke hold", "skill": "retreat", "params": _null_params()}


def default_smoke_script(task_key) -> list[dict]:
    """Plumbing-only decisions: a short open-gripper descent, then a retreat."""
    _task_targets(task_key)
    return [{"situation": "Scripted smoke: short descent with the gripper open.", "skill": "reach",
             "params": {**_null_params(), "waypoint_offset_m": [0.0, 0.0, -0.02]}},
            {"situation": "Scripted smoke: back away.", "skill": "retreat", "params": _null_params()}]


class ScriptedAgentTransport:
    """Offline transport for sim smoke tests: returns completed envelopes for
    scripted decisions and pointing answers; billing offline_mock.

    Decisions are consumed in order (then a scripted retreat repeats); pointing
    answers cycle per stage. Payload checks mirror the ordered transport."""

    offline_mock = True

    def __init__(self, decisions: list[dict], *, coarse=(36,), fine=(28,), model="gpt-6-astra",
                 effort="medium"):
        if (isinstance(decisions, (str, bytes, Mapping))
                or any(not isinstance(item, Mapping) for item in decisions)):
            raise ValueError("decisions must be a list of decision objects")
        self.decisions = [copy.deepcopy(dict(item)) for item in decisions]
        self.answers = {"pointing_coarse": tuple(coarse), "pointing_fine": tuple(fine)}
        for answers in self.answers.values():
            if any(item is not None and type(item) is not int for item in answers):
                raise ValueError("scripted pointing answers must be integers or None")
        self.model, self.effort = model, effort
        self.calls: list[dict] = []
        self.kinds: list[str] = []
        self._answer_index = {kind: 0 for kind in self.answers}

    def __call__(self, endpoint, payload, headers, timeout_s) -> dict:
        if endpoint != ENDPOINT:
            raise ValueError("Only the public OpenAI Responses endpoint is supported")
        if payload.get("store") is not False or payload.get("stream") is not True:
            raise ValueError("ChatGPT plan requests require store=false and stream=true")
        if UNSUPPORTED_FIELDS.intersection(payload):
            raise ValueError("Subscription request contains unsupported parameters")
        ordered_dumps(payload)
        kind = _KIND_BY_FORMAT.get(payload.get("text", {}).get("format", {}).get("name"))
        if kind is None:
            raise ValueError("Unknown agent request format")
        self.calls.append(copy.deepcopy(payload))
        self.kinds.append(kind)
        if kind == "decision":
            data = self.decisions.pop(0) if self.decisions else _hold_decision()
        else:
            answers, index = self.answers[kind], self._answer_index[kind]
            self._answer_index[kind] += 1
            data = {POINTING_KEYS[kind.split("_", 1)[1]]: answers[index % len(answers)] if answers else None}
        ident = len(self.calls)
        return {"id": f"offline-agent-{ident}", "status": "completed", "model": self.model,
                "reasoning": {"effort": self.effort}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "reasoning", "id": f"rs_offline_{ident}", "summary": []},
                           {"type": "message", "id": f"msg_offline_{ident}", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": ordered_dumps(data)}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
