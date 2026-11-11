"""Compact Astra batches over the existing subscription transport; no vision gate."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..action import USER_CHECK_TRANSIT_ACTION_M, TRANSLATION_SCALE_M
from ..agent_speedup_v2_client import AgentSpeedUpV2Client
from ..astra import SchemaError
from ..astranomy_agentic_motion import measured_pose
from ..contact_skills import StepView
from ..foundation_harness.types import jsonable
from ..hybrid_subscription import ChatGPTPlanHybridClient
from ..mark_prompting import encode_png_data_url
from ..ordered_transport import ordered_dumps
from ..robocasa_user_check_geometry import validate_geometry_metadata
from . import VERSION, REVISION
from .protocol import harness_schema, validate_contract, ProtocolRepairError

KIND = "harness"
HISTORY_CONTRACTS = 6
CONTEXT_KEYS = frozenset({"task_prompt", "step", "frame_id", "current_segment", "queue_remaining",
    "segment_history", "feedback", "gripper_command", "obstacle_map", "tool_results", "capabilities", "timing"})
_PRIVATE_KEYS = frozenset({"task_success", "libero_success", "libero_success_observed", "reward", "oracle",
    "evaluator", "physical_truth", "ground_truth", "answer_key", "bddl", "task_bddl", "simulator_contacts",
    "scene_object_poses", "object_truth", "success_predicates"})
_HEAVY_KEYS = frozenset({"scene_points", "scene_point_object_ids", "points_world", "point_cloud",
                       "mask", "masks", "arrays", "rgb", "images", "image_url", "backend_provenance"})

INSTRUCTIONS = """Control the robot from the task sentence, CURRENT upright external/wrist
RGB, public camera/robot calibration, proprioception and measured execution feedback.
You choose object identity, contact strategy, grip-site positions, hand orientations,
gripper commands and obstacle estimates. There is no second semantic planner and no
required vision-model geometry. Historical plans are intentions, not observations.

Return ALL fields: status, why, obstacle_map, segments, measurements. status=execute
has 1-24 segments and measurements=[]; status=measure has 1-8 measurements,
segments=[] and obstacle_map=null. finish/blocked have empty arrays and null map.
finish requires visible task relations and required release; no outside system
will place or release for you. blocked means no useful recovery remains. Explain
only the decisive observation in why. If protocol_repair is reported, correct the
specific rejected response; none of its proposed actions has executed.

Batch a useful manipulation into measured local actions. Each segment has object
(stable descriptive name or null), checkpoint and 1-3 alternatives for the SAME
local action. Do not call Astra for every waypoint or insert arbitrary waits.
Candidates: move_to_pose uses absolute WORLD grip_site xyz_m; move_relative and
approach use WORLD displacement from measured segment entry. xyz_m is always
required and cannot be null. quaternion_wxyz is absolute right_hand orientation;
null retains segment-entry orientation. profile is transit or contact; approach
requires contact. Transit target envelope is .50 m / 1.2 rad; contact .08 m / .5 rad.
Motion keeps the actual gripper state. open/close candidates contain only skill;
wait contains skill and steps (1-120). The harness supplies primitive budgets:
transit120, contact80, open16, close20. Alternatives cannot mix motion, gripper
or wait semantics or switch between open and close.

checkpoint is null or {state:observe|held|released,object:descriptive_name|null}.
Every explicit checkpoint pauses the queue for a CURRENT RGB review by you.
Remaining segments are supplied as context so you can continue or revise them.
Local arrival and stable finger gap mean only the requested motion completed;
they NEVER prove a grasp, supported placement, release or task success. Place
checkpoints where uncertain physical outcomes change what should happen next.
Choose any lift, opening and retreat explicitly in the batch; the executor does
not invent verification movements. Keep carried objects upright by choosing
appropriate orientations rather than assuming an implicit payload controller.

obstacle_map=null reuses your prior map. An object replaces it with avoid_boxes
(0-4 boxes, each lower_xyz_m/upper_xyz_m in WORLD metres) and clearance_m (0-.20).
Empty avoid_boxes explicitly clears it. On the first execute decision, supply
coarse relevant stationary obstacles from CURRENT RGB and calibration in the SAME
response as your actions. context.obstacle_map.source=uninitialized is not evidence
of free space. Use an empty set only when no relevant obstacles are visible for
the upcoming transits. Keep relevant boxes even when your selected segment appears
to clear them; the router decides whether it needs a detour. Reuse an unchanged
map with null; do not clear it merely to bypass a blocked route.

Before a grasp/carry sequence, replace the map to remove the object you will
manipulate from the avoid set, retaining other relevant obstacles. Account for
hand/carried-object extent and estimation error in clearance_m. After moving or
placing an object, switching targets, or observing changed/incorrect geometry,
reassess the map in your next normal execute response; include newly placed
objects when relevant. These are dated estimates, not measured occupancy or
automatically tracked geometry. The map is fixed for this entire batch: choose
a map valid for its actions. When later actions depend on observing a changed
scene, use a visual checkpoint at that boundary and update with the next batch.
No separate map-only model request or vision-tool call is required.

The existing transit router checks the grip-site route against these boxes;
contact motions do not use that router. Known-robot constraints are separate.
Neither a clear route nor this interface certifies whole-robot/environment
collision freedom. Plan clearance for the arm and any carried object yourself.

Use optional measurements only for an uncertainty that changes your next decision.
They add a tool/model roundtrip; ordinary RGB reasoning and motion need none.
Each measurement has id, kind:segment|point|surface, query, view:external|wrist,
region:null or [x0,y0,x1,y1] on the CURRENT original image (right/bottom exclusive),
and pixel:null or [u,v]. point requires a material pixel; segment requires a short
query and pixel=null. For point/surface, query=null requests DA3 depth only without
loading SAM: supply the selected pixel or image region. A surface requires at least
one of query, region or pixel. region can isolate the intended instance. The
harness binds the request frame.
SAM supplies masks and DA3 supplies predicted geometry when available; results
return to YOU and never silently change a goal or become an obstacle veto. Check
their source frame, age, ambiguity and validity against current RGB. A mask center
or partial surface center is not a grasp point; scores are not probabilities.
Unavailable, stale or contradictory measurements are not proof of free space or
task impossibility. Choose a supported RGB plan, another useful observation or
blocked; do not repeat the same uninformative query indefinitely.
"""
INSTRUCTIONS += (f"\nTransit command cap: {USER_CHECK_TRANSIT_ACTION_M:g} m; normalized OSC "
                 f"translation scale: {TRANSLATION_SCALE_M:g} m. These are command limits, not clearance.\n")


def _public(value, path="context"):
    if isinstance(value, Mapping):
        forbidden = set(value) & _PRIVATE_KEYS
        if forbidden:
            raise ValueError(f"private evaluator fields at {path}: {sorted(forbidden)}")
        result = {str(key): _public(item, f"{path}.{key}") for key, item in value.items()
                  if key not in _HEAVY_KEYS and key != "mask_candidates"}
        candidates = value.get("mask_candidates")
        if isinstance(candidates, (list, tuple)):
            result["mask_candidates"] = _public(candidates[:3], f"{path}.mask_candidates")
            result["mask_candidate_count"] = value.get("mask_candidate_count", len(candidates))
            result["mask_candidates_omitted"] = max(0, len(candidates)-3)
        return result
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return {"array_omitted": True, "shape": list(value.shape)}
        return _public(value.tolist(), path)
    if isinstance(value, (tuple, list)):
        return [_public(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return jsonable(value)


def _visible_draft(response):
    if not isinstance(response, Mapping) or not isinstance(response.get("output"), list):
        return ""
    return "".join(part.get("text", "") for item in response["output"]
        if isinstance(item, Mapping) and item.get("type") == "message"
        for part in item.get("content", []) if isinstance(part, Mapping)
        and part.get("type") == "output_text" and isinstance(part.get("text"), str))[:12000]


class FoundationVisionHarnessClient(AgentSpeedUpV2Client):
    prompt_version = VERSION
    history_protocol = "fvh_v3_astra_batch_optional_tools_r1"

    def __init__(self, *args, reasoning_effort="high", **kwargs):
        if reasoning_effort != "high":
            raise ValueError("V3 uses Astra high subscription reasoning")
        super().__init__(*args, **kwargs)
        self.reasoning_effort, self.image_history = "high", 1
        self._instruction_text, self._schemas = {KIND: INSTRUCTIONS}, {KIND: harness_schema()}

    def reset_history(self):
        super().reset_history()
        self._contract_history = deque(maxlen=HISTORY_CONTRACTS)
        self._pending_frame_id = self._pending_image_shape = self._pose_hemisphere = None
        self.last_harness_timing = self._protocol_repair = None
        self._schema_validation_seconds = 0.
        for name in ("_kind_calls", "_kind_accepted", "_kind_output_tokens", "_kind_reasoning_tokens", "_kind_over_budget"):
            getattr(self, name)[KIND] = 0

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=self.history_protocol, messages=len(self._contract_history),
            retained_image_pairs=0, history_contract_limit=HISTORY_CONTRACTS,
            text_history="bounded accepted batches/tool intents; current feedback supplied by executor",
            overload_retry_policy="none", transient_retry_policy="none")
        result["visible_tokens_by_kind"][KIND] = self._kind_output_tokens[KIND]-self._kind_reasoning_tokens[KIND]
        result["visible_token_budgets"][KIND] = None
        return result

    def history_snapshot(self):
        return deepcopy(list(self._contract_history))

    def _history_input(self, payload):
        return []

    def _write_log(self, path, data):
        request = isinstance(data, dict) and "payload" in data and "request_ordinal" in data
        if isinstance(data, dict):
            data = {**data, "harness_version": VERSION, "harness_revision": REVISION, "request_kind": KIND,
                    "automatic_request_retry": False, "api_key_fallback": False}
            if request:
                data["wire_payload_sha256"] = sha256(ordered_dumps(data["payload"]).encode()).hexdigest()
                data["request_meta"] = deepcopy(self.last_request_meta)
        ChatGPTPlanHybridClient._write_log(self, path, data)
        if request:
            self._kind_calls[KIND] += 1

    def _record_usage(self, response, reservation=None):
        super()._record_usage(response, reservation)
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        output = usage.get("output_tokens")
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        if type(output) is int and output >= 0:
            self._kind_output_tokens[KIND] += output
            self._kind_reasoning_tokens[KIND] += reasoning if type(reasoning) is int and 0 <= reasoning <= output else 0

    def build_payload(self, request, grid=None, mode=KIND):
        if mode != KIND or (grid is not None and grid != self.grid):
            raise ValueError("V3 accepts only its configured harness request")
        self._request_kind = KIND
        return {"model": self.config.model, "service_tier": "default", "store": False,
            "instructions": INSTRUCTIONS,
            "input": deepcopy(list(self._contract_history))+[{"role": "user", "content": self._content(request, mode)}],
            "text": {"format": {"type": "json_schema", "name": "foundation_vision_harness_V3_contract",
                                "strict": True, "schema": harness_schema()}},
            "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}

    def request_key(self, payload, grid=None, mode=KIND):
        return sha256(ordered_dumps({"payload": payload, "version": VERSION, "revision": REVISION,
                                    "endpoint": self.config.endpoint}).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        started = perf_counter()
        try:
            return validate_contract(data, frame_id=self._pending_frame_id, image_shape=self._pending_image_shape)
        except ProtocolRepairError as exc:
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
            "stream_available": bool(stream), "scope": "payload_auth_transport_parse_log; excludes tool/execution work",
            "request_key": self.last_request_key, "request_log": path}

    def predict(self, request, grid=None, mode=KIND):
        started, previous_log = perf_counter(), self.last_log_path
        self._protocol_repair, self._schema_validation_seconds = None, 0.
        status = "request_error"
        try:
            with self._history_lock:
                result = ChatGPTPlanHybridClient.predict(self, request, grid, mode)
                self._kind_accepted[KIND] += 1
                summary = {"historical_contract_frame": self._pending_frame_id, "status": result["status"],
                    "why": result["why"][:240],
                    "segments": [{"id": s["id"], "object": s["object"], "checkpoint": s["checkpoint"],
                                  "skill": s["candidates"][0]["skill"], "candidate_count": len(s["candidates"])}
                                 for s in result["segments"]],
                    "measurement_intents": [{k: m[k] for k in ("id", "kind", "query", "view", "frame_id")}
                                            for m in result["measurements"]]}
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
            raise ValueError("unexpected V3 harness context fields")
        current = _public(context)
        if current.get("task_prompt", self.task_prompt) != self.task_prompt:
            raise ValueError("context changed the task sentence")
        step, frame_id = current.get("step"), current.get("frame_id")
        if type(step) is not int or step < 0 or type(frame_id) is not int or frame_id < 0:
            raise ValueError("step and frame_id must be nonnegative integers")
        geometry = validate_geometry_metadata(getattr(view.observation, "geometry", None))
        if geometry["cameras"]["wrist"]["pose_source"] != "observed_robot_pose_fixed_mount":
            raise ValueError("wrist calibration must match the current observation")
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
            "protocol": self.history_protocol, "version": VERSION, "revision": REVISION}
        return self.predict({"fields": fields, "images": images})
