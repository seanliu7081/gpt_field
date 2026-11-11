"""Model-proposed action queues over the existing subscription transport.

Astra proposes contact strategy and metric candidates. Geometry is public evidence,
not an independent second semantic planner or a hidden evaluator channel.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from time import perf_counter
from pathlib import Path

import numpy as np

from ..agent_speedup_v2_client import AgentSpeedUpV2Client
from ..action import USER_CHECK_TRANSIT_ACTION_M, TRANSLATION_SCALE_M
from ..astra import SchemaError
from ..astranomy_agentic_motion import measured_pose
from ..contact_skills import StepView
from ..hybrid_subscription import ChatGPTPlanHybridClient
from ..mark_prompting import encode_png_data_url
from ..ordered_transport import ordered_dumps
from ..robocasa_user_check_geometry import validate_geometry_metadata
from ..foundation_vision_harness_V1.client import (
    FoundationVisionHarnessClient as _SubscriptionClient,
    CONTEXT_KEYS as _V1_CONTEXT_KEYS, KIND, _public,
)

from .protocol import harness_schema, validate_contract, ProtocolRepairError

VERSION = "foundation_vision_harness_V2"
CONTEXT_KEYS = _V1_CONTEXT_KEYS | frozenset({
    "current_segment", "queue", "queue_remaining", "segment_history", "recovery",
    "scene_status", "held_review", "action_authority",
})

INSTRUCTIONS = """Control the robot from the task, current upright external/wrist RGB,
public calibration, robot state and measured outcomes. Choose the object,
contact material, approach direction, jaw orientation and a useful action queue.
Cover a coherent manipulation with local checkpoints; do not expand every future
task stage. Missing geometry is unknown. Metric poses are proposals, not truth.

Return status, short why, targets, segments, completion_conditions. plan has
1–24 segments; observe/finish/blocked have none. finish means visible task
relations achieved and required objects released. blocked means no useful
recovery remains. The executor does not complete or release objects for you.

Targets use stable id, short query, role, parent_id and region_pixels boxes
[external,wrist] on CURRENT original images, [x0,y0,x1,y1], right/bottom exclusive.
A new target needs a box in a visible view; null means not visible. A material
region identifies its parent. Existing targets need not be repeated. The harness
stamps the request frame; do not supply frame IDs or rename objects on failure.

Each segment has object, anchor, contact_mode, keep_upright, checkpoint,
grounding and 1–3 alternative candidates for the SAME local action. The harness
assigns segment IDs, primitive budgets and contact defaults. object names the
contacted/carried object; it can be null for free motion. anchor names related
material/support. contact_mode: free (no intended contact), grasp (finger/pad
contact with object), articulated (finger/pad mechanism contact, no payload),
articulated_tool (explicit tool contact), place (finger/pad and carried payload
contact with the observed support anchor). These permissions never exempt
unrelated objects or container walls. keep_upright preserves carried orientation.

Candidates: move_to_pose uses absolute WORLD grip_site xyz_m; move_relative and
approach use WORLD displacement resolved once at segment entry. Motion includes
quaternion_wxyz (absolute right_hand orientation, null retains entry orientation)
and profile=transit or contact; it always keeps the actual gripper command.
transit permits target changes <=.50 m / 1.2 rad; contact <=.08 m / .5 rad;
approach requires contact. open/close candidates need only skill. wait uses skill
and steps. Harness budgets: transit 120, contact 80, open 16, close 20 steps.
Alternatives preserve motion/gripper/wait semantics and cannot swap open/close.

For precise contact/approach, provide grounding rather than relying only on
metric guesses: {region, view, pixel:[u,v], offset_hand_m, standoff_m, reference}.
Select a visible MATERIAL pixel on the named region in the current external or
wrist image. reference=tool uses the hand-frame vector from grip_site to the
intended tool contact point; the pad midpoint offset is in public geometry.
standoff_m is 0–0.20 m away from material along minus hand approach axis.
reference=payload_support is for placement on a support pixel and uses observed
held geometry to position its bottom; offset_hand_m=[0,0,0]. Local measured depth
binds this pixel after the response, correcting initial position as well as
later motion while retaining your quaternion. With grounding, move_to_pose may
use xyz_m=null and let that material measurement supply the metric position.
A grounded approach uses move_to_pose with contact profile; every candidate in
that segment must be absolute. Relative lift/retreat keeps its displacement and
grounding=null. Use null grounding when no reference
is needed or visible; never point at background or use a whole-object centre as
an invented contact. Source pixels are not reused on unrelated frames.

checkpoint is null or {state:observe|held|released, object:id|null}. Its object
is the STATE-CHECK object, separate from the motion's contact object. null inherits
the unique preceding close/release (or current held/released state). Name it when
needed. A free retreat can therefore check the previously released object.
held follows your planned modest lift; released follows your planned opening
and retreat. Unknown evidence is not success or failure. Checkpoints continue
locally when supported; the executor invents no probes or extra semantic calls.

On failure use measured evidence to change a candidate or strategy. A geometry
conflict or search limit is not proof the task is impossible. Repeated identical
observe calls are not new evidence. If feedback reports protocol_repair, repair
the rejected proposal using its exact error; no prefix has executed.
"""
INSTRUCTIONS += (f"\nCurrent transit command cap is {USER_CHECK_TRANSIT_ACTION_M:g} m; "
                 f"normalized translation scale is {TRANSLATION_SCALE_M:g} m. "
                 "Local checks can shorten commands; limits are not clearance.\n")


def _visible_draft(response):
    """Only visible model text is eligible for protocol repair context."""
    if not isinstance(response, Mapping) or not isinstance(response.get("output"), list):
        return ""
    texts = []
    for item in response["output"]:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts)[:12000]


class FoundationVisionHarnessClient(_SubscriptionClient):
    prompt_version = VERSION
    history_protocol = "fvh_v2_compact_grounded_queue_r2"

    def __init__(self, *args, reasoning_effort="high", **kwargs):
        super().__init__(*args, reasoning_effort=reasoning_effort, **kwargs)
        self._instruction_text = {KIND: INSTRUCTIONS}
        self._schemas = {KIND: harness_schema()}

    def reset_history(self):
        super().reset_history()
        self._checkpoint_context = {"held_object_id": None, "released_object_id": None}
        self._protocol_repair = None
        self._schema_validation_seconds = 0.

    def history_stats(self):
        result = super().history_stats()
        result["text_history"] = "bounded accepted candidate queues and current public outcomes"
        return result

    def _write_log(self, path, data):
        request = isinstance(data, dict) and "payload" in data and "request_ordinal" in data
        if isinstance(data, dict):
            data = {**data, "harness_version": VERSION, "request_kind": KIND,
                    "automatic_request_retry": False, "api_key_fallback": False}
            if request:
                data["wire_payload_sha256"] = sha256(ordered_dumps(data["payload"]).encode()).hexdigest()
                data["request_meta"] = deepcopy(self.last_request_meta)
        ChatGPTPlanHybridClient._write_log(self, path, data)
        if request:
            self._kind_calls[KIND] += 1

    def build_payload(self, request, grid=None, mode=KIND):
        if mode != KIND or (grid is not None and grid != self.grid):
            raise ValueError("V2 accepts only its configured harness request")
        self._request_kind = KIND
        content = AgentSpeedUpV2Client._content(self, request, mode)
        return {"model": self.config.model, "service_tier": "default", "store": False,
                "instructions": INSTRUCTIONS,
                "input": deepcopy(list(self._contract_history))+[{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "foundation_vision_harness_V2_contract",
                                    "strict": True, "schema": harness_schema()}},
                "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}

    def request_key(self, payload, grid=None, mode=KIND):
        return sha256(ordered_dumps({"payload": payload, "version": VERSION,
                                    "endpoint": self.config.endpoint}).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        started = perf_counter()
        try:
            return validate_contract(data, known_ids=self._known_targets, frame_id=self._pending_frame_id,
                image_shape=self._pending_image_shape, **self._checkpoint_context)
        except ProtocolRepairError as exc:
            # Shared transport sanitizes/reconstructs SchemaError subclasses;
            # retain bounded public repair evidence across that reconstruction.
            self._protocol_repair = exc
            raise
        finally:
            self._schema_validation_seconds += perf_counter()-started

    def _request_timing(self, started, status, previous_log):
        path = self.last_log_path if self.last_log_path != previous_log else None
        stream = {}
        if path and path.endswith(".request.json"):
            try:
                stream = json.loads(Path(path[:-len(".request.json")]+".stream.json").read_text())
            except (OSError, ValueError):
                pass
        first, last = stream.get("first_visible_output_s"), stream.get("last_visible_output_s")
        return {"client_elapsed_s": perf_counter()-started, "status": status,
            "first_visible_output_s": first,
            "output_generation_s": None if first is None or last is None else max(0., last-first),
            "transport_elapsed_s": stream.get("elapsed_s"), "headers_received_s": stream.get("headers_received_s"),
            "completion_s": stream.get("completion_s"), "schema_validation_s": self._schema_validation_seconds,
            "stream_available": bool(stream),
            "scope": "payload_auth_transport_parse_log; excludes perception and robot execution",
            "request_key": self.last_request_key, "request_log": path}

    def predict(self, request, grid=None, mode=KIND):
        started = perf_counter()
        previous_log = self.last_log_path
        self._protocol_repair, self._schema_validation_seconds = None, 0.
        status = "request_error"
        try:
            with self._history_lock:
                result = ChatGPTPlanHybridClient.predict(self, request, grid, mode)
                self._kind_accepted[KIND] += 1
                self._known_targets.update({t["id"]: deepcopy(t) for t in result["targets"]})
                summary = {"historical_contract_frame": self._pending_frame_id,
                    "status": result["status"], "why": result["why"][:240],
                    # Current queue/outcomes contain executable coordinates. Old
                    # candidate clouds must not grow the conversation or appear
                    # to be fresh geometric observations.
                    "segments": [{**{key: segment[key] for key in
                        ("id", "object", "anchor", "contact_mode", "checkpoint")},
                        "skill": segment["candidates"][0]["skill"],
                        "gripper": segment["candidates"][0]["gripper"],
                        "candidate_count": len(segment["candidates"])}
                        for segment in result["segments"]],
                    "completion_conditions": result["completion_conditions"]}
                self._contract_history.append({"role": "assistant", "content": json.dumps(summary, ensure_ascii=False)})
                self._history = []
                status = "accepted"
                return result
        except SchemaError as exc:
            status = "protocol_repair"
            if not isinstance(exc, ProtocolRepairError):
                exc = ProtocolRepairError(str(exc), _visible_draft(self.last_response))
            if self._protocol_repair is not None:
                exc.evidence = deepcopy(self._protocol_repair.evidence)
            raise exc from None
        finally:
            self.last_harness_timing = self._request_timing(started, status, previous_log)

    def request_harness(self, view, context):
        if not isinstance(view, StepView):
            raise TypeError("request_harness requires public StepView")
        self.check_observation(view.observation)
        if not isinstance(context, Mapping) or set(context)-CONTEXT_KEYS:
            raise ValueError("unexpected V2 harness context fields")
        current = _public(context)
        if current.get("task_prompt", self.task_prompt) != self.task_prompt:
            raise ValueError("context changed the task sentence")
        step, frame_id = current.get("step"), current.get("frame_id")
        if type(step) is not int or step < 0 or type(frame_id) is not int or frame_id < 0:
            raise ValueError("step and frame_id must be nonnegative integers")
        payload = current.get("held") or {}
        held = payload.get("held") or {}
        released = payload.get("released") or {}
        feedback = payload.get("feedback") or {}
        released_id = released.get("object_id")
        if released_id is None and str(feedback.get("state", "")).startswith("release"):
            released_id = feedback.get("object_id")
        review = current.get("held_review") or {}
        self._checkpoint_context = {"held_object_id": review.get("held_object_id", held.get("object_id")),
                                    "released_object_id": review.get("released_object_id", released_id)}
        geometry = validate_geometry_metadata(getattr(view.observation, "geometry", None))
        if geometry["cameras"]["wrist"]["pose_source"] != "observed_robot_pose_fixed_mount":
            raise ValueError("wrist calibration must match the current observation")
        current.pop("geometry", None)
        pose = measured_pose(view.state)
        if self._pose_hemisphere is None:
            self._pose_hemisphere = np.asarray(pose[3:7])
        elif float(np.dot(self._pose_hemisphere, pose[3:7])) < 0:
            pose[3:7] = [-x for x in pose[3:7]]
        fields = _public({"step": step, "frame_id": frame_id, "robot_pose_wxyz": pose,
            "finger_joints_m": view.state[6:8],
            "joint_position_rad": getattr(view.observation, "joint_position", None),
            "joint_velocity_rad_s": getattr(view.observation, "joint_velocity", None),
            "geometry": geometry, "context": current})
        if len(json.dumps(fields, allow_nan=False)) > 32000:
            raise ValueError("current compact context exceeds 32000 characters")
        shape = (geometry["image"]["height"], geometry["image"]["width"])
        images = []
        for label, rgb in (("External upright RGB", view.external_rgb_upright),
                           ("Wrist upright RGB", view.wrist_rgb_upright)):
            if tuple(rgb.shape[:2]) != shape:
                raise ValueError("RGB raster differs from current calibration")
            images.append((f"{label}, frame {frame_id}, observation step {step}", encode_png_data_url(rgb)))
        self._pending_frame_id, self._pending_image_shape = frame_id, shape
        self.last_request_meta = {"kind": KIND, "step": step, "frame_id": frame_id,
                                 "protocol": self.history_protocol, "version": VERSION}
        return self.predict({"fields": fields, "images": images})
