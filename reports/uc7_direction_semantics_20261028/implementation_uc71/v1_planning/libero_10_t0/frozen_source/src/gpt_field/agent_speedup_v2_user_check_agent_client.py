"""Persistent task executive for the user-check variant.

The executive owns the complete task; monitored fragments and native v2
relations are optional execution tools.
The subscription transport, usage ledger and request/response logs are shared
with v2, but its stateless repair transcript and instructions are not. Native
pointing calls cannot erase the executive's visual/action history.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path

from .agent_speedup_v2_client import (
    AgentSpeedUpV2Client, OVERLOAD_BACKOFF_BASE_S, OVERLOAD_JITTER_S,
    OVERLOAD_MAX_WAIT_S, TRANSIENT_BACKOFF_BASE_S,
)
from .agent_speedup_v2_user_check_client import ReviewedV2Client
from .agent_speedup_v2_user_check_context import _bounded, causal_nodes, size
from .agent_speedup_v2_user_check_fragments import (
    execution_schema, validate_execution, fragment_capabilities,
)
from .agent_speedup_v2_user_check_motion import _conform_motion, local_motion_schema
from .astra import APIError, SchemaError, _canonical
from .robocasa_user_check_geometry import validate_geometry_metadata
from .astranomy_agentic_motion import measured_pose
from .chatgpt_plan import _failure_metadata
from .contact_skills import StepView
from .direct_history import _trim_images
from .hybrid_subscription import ChatGPTPlanHybridClient
from .mark_prompting import encode_png_data_url
from .ordered_transport import ordered_dumps


CLIENT_VERSION = "user-check-agent-first-client-2"
PROMPT_VERSION = "user-check-agent-first-prompt-3"
HISTORY_PROTOCOL = "persistent_task_executive_fragments_text_and_six_visual_pairs_v2"
IMAGE_HISTORY = 6
AGENT_KIND = "agent"
CONTEXT_KEYS = frozenset({
    "step", "approved_plan", "task_contract", "geometry", "capabilities",
    "measured_state", "model_belief", "last_feedback", "recent_feedback",
    "macro", "execution", "native_evidence", "task_success_source", "region_directions",
})

AGENT_INSTRUCTIONS = """
You are the continuous task executive for one Franka Panda with a parallel-jaw
Panda gripper. Complete the official task using synchronized RGB observations,
measured robot state, camera calibration and this episode's chronological
action/result history. No object poses, depth, segmentation, contact identities,
reward or simulator goal predicates are supplied. The two images are external
and wrist, upright and uncropped. Older images retain their own geometry and
timestamp; the six most recent observation pairs are retained. All chronological
text, accepted outputs, measured results and opaque reasoning are retained.

The reviewed JSON expresses the user's intended objects, task goals and
constraints. It is the initial proposed decomposition, NOT a mandatory execution
program or a requirement to satisfy every original intermediate waypoint.
Preserve the official task, reviewed final spatial goals, established hold_true
relations and applicable keep constraints. You may change grips, reorder work,
stage objects temporarily and plan new action sequences automatically. IDs keep
the same physical identities. A keep applies while pursuing its own intent;
future intent constraints do not block unrelated current work. Necessary grasp
and support contacts are distinct from unintended collisions.

YOU OWN EXECUTION. Monitored fragments or direct motion may perform an entire
subtask: approach, close, lift, carry, align, lower, open at the desired
destination and withdraw. You need
not resume a native macro or obtain a native verified holding flag. An unknown
holding estimate means uncertainty, not proof of an empty hand or a universal
motion prohibition. Use images across moves, measured fingers and motion error
to maintain your own explicitly uncertain belief. A failed or unavailable native
observer is evidence about that observer; repeating a move without a plausible
new information source does not improve its answer. Do not repeatedly lift only
to satisfy an unavailable co-motion detector. Choose short informative moves or
another view when evidence is insufficient. Explicit observed danger constrains
motion; uncertainty calls for shorter monitored steps, not automatic deadlock.

Each response has belief (current evidence and uncertainty), plan (unfinished
subgoals and the next approach), why (what this action changes or observes), type,
motion, macro and execution. All keys are required. These belief/plan fields are
YOUR estimates, never measured task-completion flags. Choose exactly one action
and set every unused action payload to null:

execute: motion=null, macro=null, execution={segments:[...]}. Prefer execute for
a sufficiently understood free-space approach, carry or retreat, or an explicit
short sequence whose intermediate stages do not need new visual reasoning.
Delegate the SAME selected pose target until measured arrival or a local event;
do not spend repeated model calls replaying it every five control steps. The
local controller recomputes error from robot measurements every step, completes
a stage on measured arrival or finger settling, and advances the selected
sequence unless observation_after requests a checkpoint. A stall, unexpected
measured response or exhausted motion budget returns control for reconsideration.
It does not certify semantic attachment, placement or task success. No native scene
registration or motion calibration is needed for these fragments.

execution.segments contains 1..6 objects, each with exactly skill, xyz_m,
quaternion_wxyz, gripper, profile, max_steps, observation_after. Coordinates are
in the fixed world frame, metres; orientation is absolute right_hand wxyz.
- move_to_pose: xyz_m is an absolute world grip-site target.
- move_relative: xyz_m is a world displacement from the measured pose at THIS
  stage's entry, not from the first stage or an earlier requested target.
- approach: xyz_m is likewise a world displacement at this stage's measured
  entry; use profile=contact for a slow monitored approach near contact.
- set_gripper: xyz_m=null, quaternion_wxyz=null, gripper=open or close. The arm
  holds its measured entry pose while finger-gap feedback is monitored.
- wait: xyz_m=null, quaternion_wxyz=null, gripper=keep; retain the measured entry
  pose and last gripper command for the requested bounded wait.
Arm-movement stages always use gripper=keep; put any finger change in a separate
set_gripper stage. For arm stages quaternion_wxyz=null retains measured entry
orientation. max_steps is 1..120 per stage, with sum(max_steps)<=240 across the
whole sequence. At 20 Hz, budgets are upper bounds, not arrival guarantees.
profile=transit allows at most 0.30 m / 1.2 rad from a stage's measured start;
profile=contact allows at most 0.08 m / 0.5 rad. The unchanged OSC feedback loop
caps each action at 0.02 m translation (0.005 m for contact) and 0.1 rad rotation.
These envelopes are not permission to cross uncertain space. Choose a clear
route and intermediate waypoints using RGB and camera calibration. The executor
does not discover obstacles or plan a collision-free path for you.

observation_after=true means STOP the sequence after this stage and return fresh
RGB to you. Remaining stages are reported as unexecuted and are NOT automatically
played or resumed after your next decision. Use this checkpoint near uncertain
contact or before an action whose justification depends on a preceding result.
For a grasp, use explicit set_gripper=close followed by an appropriately small
monitored test lift/checkpoint, then inspect new images; finger closure or stable
gap alone never proves attachment. For placement, inspect whole-object support
and pose after opening/settling. If a sequence yields on a stall, timeout or
unexpected measured response, reconsider its path from the new evidence rather
than blindly repeating it. Use motion below for uncertain fine adjustment or
when a very short observed action is preferable to committing a fragment.

motion: put a complete short target in motion; macro=null, execution=null. frame is world.
position_mode=relative means xyz_m is one world displacement from the latest
measured position; absolute means world XYZ in metres. World +z is up. XYZ
locates gripper0_grip_site between the fingers, not the hand origin. The optional
quaternion_wxyz is an absolute robot0_right_hand world orientation, or null to
retain the measured orientation. gripper=keep preserves the last commanded
open/close setting, not an attachment assertion; open and close directly command
the fingers at this pose. Direct opening does not return to an earlier support.
steps is 1..5 at 20 Hz. The target is at most 0.05 m translation and 0.35 rad
rotation from the latest measured pose. Each control step recomputes measured
pose error, capped at 0.02 m translation and 0.1 rad rotation. A segment may end
before arrival. Inspect actual feedback and fresh images; once the route/target
is understood, use execute for continuous measured execution toward it. Shorten
segments near uncertain contact. Simulation pauses
during inference. Closing is not proof of attachment; opening is not proof of
stable support. Check the actual object's final pose and support in new views.

macro: motion=null, execution=null, macro={step_index,budget_steps,continuation}. step_index is
the zero-based reviewed step whose native hold/turn/place_on/release controller
you choose as an optional accelerator. budget_steps is 1..60 control actions
before returning observation/control to you. Native prerequisites may be unknown
or unavailable, and setup itself consumes the budget. Initial native calibration
normally requires about 190 control steps, so a 60-step calibration budget pause
is NORMAL and may be continued; it is not evidence of failure. Inspect explicit
phase/progress and capabilities. Choose continuation=true only to continue that same paused session
without intervening direct motion. Other choices abandon its pending action
sequence. Macros may report blocked/failed/inconclusive; decide from fresh
evidence whether to continue or take over directly. Native release returns an
object to its original pick support; use direct motion with gripper=open when
you want to release at your present destination. Do not repeatedly select a
macro whose missing prerequisites are unchanged. Unknown native geometry or
holding NEVER prohibits your direct task execution.

observe: motion=null, macro=null, execution=null. Hold the measured pose briefly and receive
fresh observations, for example to inspect settling. This does not change the
camera viewpoint; move the wrist with motion when a new angle is needed.
finish: motion=null, macro=null, execution=null. This only requests a final fresh observation
when you believe the task is complete. It never ends the episode or sets success.
Only the independent evaluator terminates a successful task. If observation
continues, recheck unfinished spatial relations rather than repeatedly declaring
completion. There is no give_up action.

Direction words in the task and in reviewed regions are as seen in the external image (image 1):
right/left = toward its image-right/left; front/behind = nearer to/farther from the external camera
along its viewing direction. Never judge them in the wrist image. region_directions gives each
region's world-xy direction computed from calibration; use it over any direction wording in reviewed
text. If region_directions is unavailable, judge the words in image 1 as defined.
The direction frame is frozen from the external view at review; later camera motion does not change
these reviewed meanings. The full first-request contract is reviewed_goal_contract.task_contract;
context.region_directions repeats its compact direction information in every request, even when
there are no directional regions. Use current camera calibration to interpret the current RGB.

Camera position_world_m and rotation_world_from_camera map OpenGL camera axes to
world: p_world=origin+R*p_camera. Camera +X is image-right, +Y image-up, -Z forward.
Pixel u increases right and v down. For camera point [X,Y,Z], Z<0,
u=cx+fx*X/(-Z), v=cy-fy*Y/(-Z). Pixels specify rays, not known 3-D points; infer
depth from the views and observed motion. Do not use image axes as world axes or
assume image center is the grasp point. Use each image's synchronized camera
pose. The hand's local +Z is finger approach and local Y is jaw separation.
The grip site is 0.097 m along hand-local +Z from the hand origin; commanded XYZ
already locates it, so do not add this offset again. Symmetric finger pad centers
have midpoint [0,0,-0.0036] m relative to the grip site in hand coordinates; pad
size is [0.016,0.008,0.016] m, nominal open inner gap 0.079 m. Tool geometry does
not prove clearance or contact. Judge the whole hand, carried object and nearby
objects in both views. Keep task reasoning concise and output the JSON only.
""".strip()


def agent_schema() -> dict:
    motion = local_motion_schema()
    del motion["properties"]["recovery_goal"]
    motion["required"].remove("recovery_goal")
    macro = {"type": "object", "properties": {
        "step_index": {"type": "integer", "minimum": 0},
        "budget_steps": {"type": "integer", "minimum": 1, "maximum": 60},
        "continuation": {"type": "boolean"},
    }, "required": ["step_index", "budget_steps", "continuation"],
        "additionalProperties": False}
    properties = {
        "belief": {"type": "string", "minLength": 1, "maxLength": 1800},
        "plan": {"type": "string", "minLength": 1, "maxLength": 1800},
        "why": {"type": "string", "minLength": 1, "maxLength": 600},
        "type": {"type": "string", "enum": ["execute", "motion", "macro", "observe", "finish"]},
        "motion": {"anyOf": [motion, {"type": "null"}]},
        "macro": {"anyOf": [macro, {"type": "null"}]},
        "execution": {"anyOf": [execution_schema(), {"type": "null"}]},
    }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def validate_agent_decision(value) -> dict:
    """Reject malformed commands; position/rotation bounds use measured runtime pose."""
    try:
        schema = agent_schema()
        if not isinstance(value, Mapping) or set(value) != set(schema["required"]):
            raise ValueError("executive response must contain exactly belief, plan, why, type, motion, macro, execution")
        for key in ("belief", "plan", "why"):
            item = value[key]
            if (not isinstance(item, str) or not item.strip()
                    or len(item) > schema["properties"][key]["maxLength"]):
                raise ValueError(f"{key} must be a nonempty bounded string")
        action = value["type"]
        if action not in ("execute", "motion", "macro", "observe", "finish"):
            raise ValueError("unknown executive action")
        if action == "motion":
            if value["macro"] is not None:
                raise ValueError("motion cannot also request a macro")
            _conform_motion(value["motion"])
            if "recovery_goal" in value["motion"]:
                raise ValueError("direct executive motion has no mandatory recovery subgoal")
        elif value["motion"] is not None:
            raise ValueError("motion must be null for this action")
        if action == "macro":
            item = value["macro"]
            if not isinstance(item, Mapping) or set(item) != {"step_index", "budget_steps", "continuation"}:
                raise ValueError("macro requires exactly step_index, budget_steps, continuation")
            if type(item["step_index"]) is not int or item["step_index"] < 0:
                raise ValueError("macro step_index must be a nonnegative integer")
            if type(item["budget_steps"]) is not int or not 1 <= item["budget_steps"] <= 60:
                raise ValueError("macro budget_steps must be an integer in 1..60")
            if type(item["continuation"]) is not bool:
                raise ValueError("macro continuation must be boolean")
        elif value["macro"] is not None:
            raise ValueError("macro must be null for this action")
        if action == "execute":
            validate_execution(value["execution"])
        elif value["execution"] is not None:
            raise ValueError("execution must be null for this action")
        return copy.deepcopy(dict(value))
    except (ValueError, TypeError, KeyError) as exc:
        raise SchemaError(f"agent output: {exc}") from None


def _execution_evidence_view(value, budget=7000):
    """Keep stage outcomes and stall causes ahead of verbose controller detail."""
    if size(value) <= budget:
        return copy.deepcopy(value)
    stage_rows = []
    stage_keys = {"skill", "segment_index", "stage_index", "stage", "segment"}
    retained = ("skill", "segment_index", "stage_index", "stage", "segment", "status",
                "reason", "executed_steps", "requested_steps", "max_steps", "observation_after",
                "position_error_m", "orientation_error_rad", "finger_gap_mm", "gap_mm",
                "progress", "stall", "arrival", "measurements", "thresholds")

    def collect(item, path):
        if isinstance(item, dict):
            if stage_keys.intersection(item) and any(key in item for key in ("status", "reason", "progress")):
                stage_rows.append({"path": path, **{key: item[key] for key in retained if key in item}})
            for key, child in item.items():
                if isinstance(child, (dict, list, tuple)):
                    collect(child, path + "." + key)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                if isinstance(child, (dict, list, tuple)):
                    collect(child, path + f"[{index}]")

    collect(value, "execution")
    usable = max(1000, budget - 500)
    return {"summary": _bounded(value, int(usable * .5)),
            "stage_outcomes": _bounded(stage_rows[-12:], int(usable * .25)),
            "causes": _bounded(causal_nodes(value), int(usable * .25)),
            "full_detail": "execution_audit"}


class AgentFirstV2Client(ReviewedV2Client):
    """Task-wide decision history, independent of optional native pointing calls."""

    prompt_version = PROMPT_VERSION
    history_protocol = HISTORY_PROTOCOL
    reasoning_effort = "high"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_history = IMAGE_HISTORY
        self._instruction_text[AGENT_KIND] = AGENT_INSTRUCTIONS
        self._schemas[AGENT_KIND] = agent_schema()

    def reset_history(self):
        super().reset_history()
        self._agent_history = []
        self._agent_decisions = 0
        self._agent_latest_belief = None
        for name in ("_kind_calls", "_kind_accepted", "_kind_output_tokens",
                     "_kind_reasoning_tokens", "_kind_over_budget"):
            getattr(self, name)[AGENT_KIND] = 0

    def history_stats(self):
        result = super().history_stats()
        history = getattr(self, "_agent_history", [])
        result.update(
            protocol=HISTORY_PROTOCOL, messages=len(history),
            retained_image_pairs=sum(
                item.get("role") == "user" and isinstance(item.get("content"), list)
                and any(part.get("type") == "input_image" for part in item["content"])
                for item in history),
            image_history_limit=IMAGE_HISTORY,
            text_history="all_executive_observations_accepted_outputs_and_measured_feedback",
            reasoning_replay="opaque_encrypted_output_items",
            agent_decisions=getattr(self, "_agent_decisions", 0),
        )
        result["visible_tokens_by_kind"][AGENT_KIND] = (
            self._kind_output_tokens[AGENT_KIND] - self._kind_reasoning_tokens[AGENT_KIND])
        result["visible_token_budgets"][AGENT_KIND] = 1400
        return result

    def history_snapshot(self):
        with self._history_lock:
            return copy.deepcopy(self._agent_history)

    def _record_usage(self, response, reservation=None):
        super()._record_usage(response, reservation)
        if self._request_kind != AGENT_KIND:
            return
        usage = response.get("usage") if isinstance(response, Mapping) else None
        if not isinstance(usage, Mapping):
            return
        output = usage.get("output_tokens")
        details = usage.get("output_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
        if type(output) is int and output >= 0:
            spent = reasoning if type(reasoning) is int and 0 <= reasoning <= output else 0
            self._kind_output_tokens[AGENT_KIND] += output
            self._kind_reasoning_tokens[AGENT_KIND] += spent
            if output - spent > 1400:
                self._kind_over_budget[AGENT_KIND] += 1

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            return super().build_payload(request, grid, mode)
        if grid is not None and grid != self.grid:
            raise ValueError("Requested grid must match the configured client grid")
        # Skip the legacy framework compactor: executive requests already contain
        # one evidence snapshot and use a different, strictly allowed context.
        content = AgentSpeedUpV2Client._content(self, request, mode)
        self._request_kind = AGENT_KIND
        history = _trim_images(self._agent_history + [{"role": "user", "content": content}], IMAGE_HISTORY)
        return {
            "model": self.config.model, "service_tier": "default", "store": False,
            "instructions": AGENT_INSTRUCTIONS, "input": history,
            "text": {"format": {"type": "json_schema", "name": "user_check_agent_first",
                                  "strict": True, "schema": agent_schema()}},
            "stream": True, "reasoning": {"effort": self.reasoning_effort},
            "include": ["reasoning.encrypted_content"],
        }

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
        return validate_agent_decision(data)

    def _history_input(self, payload):
        if self._request_kind == AGENT_KIND:
            return payload["input"]
        return []

    def predict(self, request, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            # The base clears only its own stateless request history. The
            # executive's accepted transcript is separate and remains intact.
            return super().predict(request, grid, mode)
        with self._history_lock:
            initial_count = self._observation_count
            retries = {"overload": 0, "transient": 0}
            self._overload_attempt = 1
            try:
                while True:
                    self.last_output_key_order = None
                    self.last_reask_check = None
                    try:
                        result = ChatGPTPlanHybridClient.predict(self, request, grid, mode)
                    except APIError as exc:
                        cause = self._retry_cause(exc, initial_count, retries)
                        if cause is None:
                            raise
                        backoff = OVERLOAD_BACKOFF_BASE_S if cause == "overload" else TRANSIENT_BACKOFF_BASE_S
                        delay = max(backoff ** (retries[cause] + 1) + random.uniform(0, OVERLOAD_JITTER_S),
                                    _failure_metadata(exc).get("retry_after_s", 0.0))
                        if delay > OVERLOAD_MAX_WAIT_S:
                            raise
                        self._reserve()
                        if (self.config.max_wall_time_s is not None
                                and self.clock() - self._started + delay >= self.config.max_wall_time_s):
                            raise
                        path = (Path(self.last_log_path).with_suffix(f".{cause}_retry.json")
                                if self.last_log_path else None)
                        self._write_log(path, {"status": "bounded_retry_scheduled", "cause": cause,
                            **_failure_metadata(exc), "request_kind": AGENT_KIND,
                            "attempt": self._overload_attempt, "next_attempt": self._overload_attempt + 1,
                            "wait_s": delay, "same_observation": True,
                            "history_committed": False, "actions_executed_from_failed_response": 0})
                        started = self.clock()
                        time.sleep(delay)
                        waited = max(0, self.clock() - started)
                        retries[cause] += 1
                        self._overload_attempt += 1
                        if cause == "overload":
                            self._overload_retry_requests += 1
                            self._overload_retry_wait_s += waited
                        else:
                            self._transient_retry_requests += 1
                            self._transient_retry_wait_s += waited
                            code = getattr(exc, "error_code", None)
                            self._transient_retry_codes[code] = self._transient_retry_codes.get(code, 0) + 1
                    else:
                        self._agent_history = copy.deepcopy(self._history)
                        self._agent_decisions += 1
                        self._kind_accepted[AGENT_KIND] += 1
                        self._agent_latest_belief = {"belief": result["belief"], "plan": result["plan"],
                                                     "source": "model_estimate"}
                        return result
            finally:
                self._overload_attempt = 1

    def request_agent(self, view: StepView, context: dict) -> dict:
        """Request one decision from synchronized public sensors and runtime evidence."""
        if not isinstance(view, StepView):
            raise TypeError("request_agent requires a StepView")
        self.check_observation(view.observation)
        self.plan_lock.check_active()
        if not isinstance(context, Mapping) or set(context) - CONTEXT_KEYS:
            raise ValueError("executive context contains unexpected fields")
        current = copy.deepcopy(dict(context))
        capabilities = current.setdefault("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ValueError("executive capabilities must be an object")
        capabilities["execute"] = fragment_capabilities()
        geometry = current.pop("geometry", None)
        if geometry is None:
            geometry = getattr(view.observation, "geometry", None)
        geometry = validate_geometry_metadata(geometry)
        if geometry["cameras"]["wrist"]["pose_source"] != "observed_robot_pose_fixed_mount":
            raise ValueError("executive wrist calibration must be synchronized with robot/RGB observation")
        step = current.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("executive observation step must be a nonnegative integer")
        reviewed_plan = current.pop("approved_plan", self.plan_lock.plan())
        task_contract = current.pop("task_contract", None)
        from .agent_speedup_v2_user_check_spatial import region_directions
        # Recompute this small payload from the frozen approval on every call;
        # it cannot disappear with image-history trimming or stale caller context.
        directions = region_directions(self.plan_lock.view_frame, self.plan_lock.plan().get("regions", []))
        current["region_directions"] = directions
        if task_contract is None:
            task_contract = self.plan_lock.task_contract.to_dict()
        task_contract["region_directions"] = copy.deepcopy(directions)
        fields = {"step": step, "robot_pose_wxyz": measured_pose(view.state),
                  "finger_joints_m": [float(x) for x in view.state[6:8]],
                  "geometry": geometry, "context": current}
        if not self._agent_history:
            fields["reviewed_goal_contract"] = {"initial_plan": reviewed_plan,
                "task_contract": task_contract,
                "authority": "goals_and_applicable_constraints; execution_methods_may_change"}
        # Retain terminal causal details before bounding large native controller
        # internals. The original full feedback remains in runtime audit files.
        if size(fields) > 45000:
            budgets = {"execution": 7000, "native_evidence": 5000,
                       "last_feedback": 7000, "recent_feedback": 4500}
            for key, budget in budgets.items():
                if key in current:
                    current[key] = _execution_evidence_view(current[key], budget)
            fields["context_detail_compaction"] = (
                "large_execution_evidence_only; stage_outcomes_and_causes_prioritized; "
                "goals_and_sensor_geometry_preserved")
        json.dumps(fields, allow_nan=False)
        if size(fields) > 56000:
            raise ValueError("executive current observation exceeds bounded request text size")
        expected_shape = (geometry["image"]["height"], geometry["image"]["width"])
        images = []
        for label, rgb in (("External upright RGB", view.external_rgb_upright),
                           ("Wrist upright RGB", view.wrist_rgb_upright)):
            if tuple(rgb.shape[:2]) != expected_shape:
                raise ValueError("executive camera image does not match synchronized calibration")
            images.append((f"{label}, observation step {step}", encode_png_data_url(rgb)))
        self.last_request_meta = {"kind": AGENT_KIND, "step": step,
                                  "protocol": HISTORY_PROTOCOL, "image_history_limit": IMAGE_HISTORY}
        return self.predict({"fields": fields, "images": images}, mode=AGENT_KIND)
