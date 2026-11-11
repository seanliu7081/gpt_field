"""Anchored operation contracts over the existing Astra subscription transport.

This module owns no robot actions. Pixel regions retain observation lineage;
metric contact poses are resolved locally from the selected regions. The model
sees current RGB and compact public feedback, not evaluator state or point clouds.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from time import perf_counter

import numpy as np

from ..agent_speedup_v2_client import AgentSpeedUpV2Client
from ..astra import SchemaError
from ..astranomy_agentic_motion import measured_pose
from ..contact_skills import StepView
from ..foundation_harness.types import jsonable
from ..hybrid_subscription import ChatGPTPlanHybridClient
from ..mark_prompting import encode_png_data_url
from ..ordered_transport import ordered_dumps
from ..robocasa_user_check_geometry import validate_geometry_metadata

VERSION = "foundation_vision_harness_V1"
KIND = "harness"
HISTORY_CONTRACTS = 6
CONTEXT_KEYS = frozenset({
    "task_prompt", "step", "frame_id", "source_frame_id", "request_frame_id",
    "scene_revision", "scene", "current_operation", "current_phase",
    "operation_history", "operations", "completion_conditions", "outcomes", "feedback", "last_feedback",
    "recent_feedback", "held", "holding", "possible_payload", "robot", "robot_joints",
    "measured_state", "capabilities", "geometry", "perception", "timing", "deadline",
    "event", "targets", "reason", "gripper_command", "motion_evidence",
})
_PRIVATE_KEYS = frozenset({
    "task_success", "libero_success", "libero_success_observed", "reward", "oracle",
    "evaluator", "physical_truth", "ground_truth", "answer_key", "bddl", "task_bddl",
    "simulator_contacts", "scene_object_poses", "object_truth", "success_predicates",
})
_HEAVY_KEYS = frozenset({
    "scene_points", "scene_point_object_ids", "scene_point_strong", "points_in_gripper",
    "points_world", "points", "normals", "mask", "masks", "arrays", "point_cloud",
    "depth_m", "rgb", "image_url", "images",
})

INSTRUCTIONS = """You are the semantic planner of foundation_vision_harness_V1.
Return the compact structured operation contract. Use the task sentence, the two
CURRENT upright RGB images, robot calibration, and current public feedback.
Predicted surfaces and uncertainty describe visible evidence, not exact geometry
or complete object shapes. Historical contracts are past intentions, not current
observations. The local executor computes metric contact poses and trajectories.

On the first request, provide an inventory of relevant visible objects, their
image regions, and a useful batch of operations IN THE SAME RESPONSE. There is
no separate initial-plan request. Include other visible obstacles; do not assume
an unnamed object is empty space. Use short category/part queries for perception.
Choose stable ids and reuse known ids. A target has role object, region, support,
obstacle, or robot. A region MUST name its parent object/support/obstacle id.
All target frame_id values equal context.frame_id. region_pixels contains a box
[x0,y0,x1,y1] in the original current image raster, right/bottom exclusive, or
null for an unobserved view. Select the SAME physical region across views; two
mask centres do not by themselves establish corresponding surface points.
A new object or region needs at least one visible box. Do not rename an instance
or silently bind a missing part to the whole object. Existing known targets can
be referenced without repeating them. Geometry resolves these regions after this
response and at stage entry; missing future placement geometry need not prevent
a useful approach. If the local region cannot support closure, approach and
re-observation precede a new contact attempt.

Operations are generic: pick_place, grasp, place, move, contact_move, release,
observe. pick_place and place include relation on, inside, or beside. Select a
destination region expressing the requested relation, including its side/interior
when necessary. For beside, target names the neighboring object, offset_m gives
an explicit lateral direction/preference, and the inventory includes the observed
support surface as a support-role target. This preference is not metric ground
truth: the local grounder computes a nonoverlapping placement from measured
object extents and supporting geometry. Grasp and pick_place need a specific grasp_region whose parent
is the manipulated object. A grasp keeps holding; pick_place and place include
release and retreat. Grasp of an articulated part must be followed by contact_move
in the SAME batch: this preserves engagement instead of lifting the whole fixture.
contact_move uses the selected contact region and explicit relative displacement
and/or rotation; it does not imply that the fixture is a carried object.

Directions, jaw_axis and rotation_axis are unit vectors in the public WORLD frame;
they are optional orientation hints for local geometry, not absolute pose targets.
displacement is relative to the measured operation-entry pose and is used only
for move/contact_move. rotation_angle_rad is paired with rotation_axis and used
only for contact_move. offset_m is an anchored geometric preference; use [0,0,0]
unless a measured anchor supports the offset. move uses a target reference or a
relative displacement. There are no absolute XYZ or quaternion goal fields.
keep_upright constrains the carried object. contact_parts explicitly lists which
parts may contact the operation's identified object: fingers, pads, or palm.
It never exempts neighbouring objects, robot links, or an entire container box.
Use palm only for an intended push/insert contact_move. stop_on=contact is for a
declared contact_move; ordinary transport uses goal. Unused fields must be null
or the empty contact_parts list, with offset_m=[0,0,0].

Batch understood operations so local pose tracking, alignment, gripper settling,
attachment checks, transport and release can continue without semantic requests.
Contact and grasp evidence comes from measured gripper behaviour and visible
material motion. Closure alone is not attachment; pose arrival is not placement.
The local executor preserves whole-object geometry while carrying. If it reports
unknown payload extent or an unresolved robot-surface/calibration fault, do not
try to remove an obstacle by renaming it or supplying a different coordinate.
Change object identity, contact region or strategy only when RGB/feedback supports
that change. For an unchanged local fault with no new relevant evidence use blocked.

status=plan requires operations; observe asks for new observation/grounding and
has no operations. status=finish ends the task based on observed achieved relations,
release and stable support, with the gripper clear where required. It is not a
command to have an outside system finish or release for you. status=blocked ends
the autonomous attempt with a concise reason. finish/blocked have no operations.
Give short visible completion_conditions; do not infer completion from continued
observations or from a planned trajectory. Explain only the decisive fact in why.
"""


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def _string(limit=64):
    return {"type": "string", "minLength": 1, "maxLength": limit}


def _vector(n):
    return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}


def harness_schema():
    target = _object({
        "id": _string(32), "query": _string(64),
        "role": {"type": "string", "enum": ["object", "region", "support", "obstacle", "robot"]},
        "parent_id": _nullable(_string(32)), "frame_id": {"type": "integer", "minimum": 0},
        "region_pixels": _object({v: _nullable(_vector(4)) for v in ("external", "wrist")}),
    })
    operation = _object({
        "kind": {"type": "string", "enum": ["pick_place", "grasp", "place", "move", "contact_move", "release", "observe"]},
        "object": _nullable(_string(32)), "target": _nullable(_string(32)),
        "grasp_region": _nullable(_string(32)),
        "relation": _nullable({"type": "string", "enum": ["on", "inside", "beside"]}),
        "keep_upright": {"type": "boolean"},
        "approach_direction": _nullable(_vector(3)), "jaw_axis": _nullable(_vector(3)),
        "displacement": _nullable(_vector(3)), "rotation_axis": _nullable(_vector(3)),
        "rotation_angle_rad": _nullable({"type": "number"}), "offset_m": _vector(3),
        "contact_parts": {"type": "array", "items": {"type": "string", "enum": ["fingers", "pads", "palm"]},
                          "maxItems": 3},
        "stop_on": {"type": "string", "enum": ["goal", "contact"]},
    })
    return _object({
        "status": {"type": "string", "enum": ["plan", "observe", "finish", "blocked"]},
        "why": _string(500), "targets": {"type": "array", "items": target, "maxItems": 24},
        "operations": {"type": "array", "items": operation, "maxItems": 12},
        "completion_conditions": {"type": "array", "items": _string(160), "maxItems": 12},
    })


def _conform(value, schema, path="answer"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _conform(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"invalid {path}")
    kind = schema["type"]
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str) and bool(value.strip()),
             "number": lambda: type(value) in (int, float) and math.isfinite(value),
             "integer": lambda: type(value) is int, "boolean": lambda: type(value) is bool,
             "null": lambda: value is None}[kind]()
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValueError(f"invalid {path}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"missing or unexpected fields in {path}")
        for key, item in value.items():
            _conform(item, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
            raise ValueError(f"invalid length of {path}")
        for i, item in enumerate(value):
            _conform(item, schema["items"], f"{path}[{i}]")
    elif kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
        raise ValueError(f"invalid length of {path}")
    elif kind in ("number", "integer") and value < schema.get("minimum", -math.inf):
        raise ValueError(f"invalid lower bound for {path}")


def validate_contract(data, *, known_ids=(), frame_id=None, image_shape=None):
    """Validate structure, observation lineage and contact-region identity.

    Missing metric geometry is handled by the executor; it is not a schema veto.
    """
    try:
        _conform(data, harness_schema())
        known = dict(known_ids) if isinstance(known_ids, Mapping) else {key: {} for key in known_ids}
        ids = [target["id"] for target in data["targets"]]
        if len(ids) != len(set(ids)):
            raise ValueError("target ids must be unique")
        records = {**known, **{target["id"]: target for target in data["targets"]}}
        for target in data["targets"]:
            if frame_id is not None and target["frame_id"] != frame_id:
                raise ValueError("target frame_id differs from the current request frame")
            parent = target["parent_id"]
            if target["role"] == "region":
                if parent not in records or parent == target["id"]:
                    raise ValueError("region requires a known distinct parent_id")
                if records[parent].get("role") in ("region", "robot"):
                    raise ValueError("region parent must be an object, support or obstacle")
            elif parent is not None:
                raise ValueError("only region targets have parent_id")
            previous = known.get(target["id"], {})
            if previous and (previous.get("role") != target["role"] or previous.get("parent_id") != parent):
                raise ValueError("target identity role/parent may not be rebound")
            boxes = target["region_pixels"].values()
            if target["id"] not in known and not any(box is not None for box in boxes):
                raise ValueError("new target requires a visible region in at least one view")
            for box in boxes:
                if box is not None:
                    if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                        raise ValueError("region_pixels must have positive area and nonnegative coordinates")
                    if image_shape is not None and (box[2] > image_shape[1] or box[3] > image_shape[0]):
                        raise ValueError("region_pixels extends beyond the current image")
        if (data["status"] == "plan") != bool(data["operations"]):
            raise ValueError("only status=plan has nonempty operations")
        for operation in data["operations"]:
            kind = operation["kind"]
            for key in ("object", "target", "grasp_region"):
                value = operation[key]
                if value is not None and value not in records:
                    raise ValueError(f"operation {key} must reference a known target id")
                if value is not None and records[value].get("role") == "robot":
                    raise ValueError("robot segmentation cannot be an operation target")
            if kind in ("pick_place", "grasp", "place", "contact_move") and operation["object"] is None:
                raise ValueError(f"{kind} requires object")
            if kind in ("pick_place", "grasp", "contact_move"):
                region = operation["grasp_region"]
                if region is None or records[region].get("role") != "region" or records[region].get("parent_id") != operation["object"]:
                    raise ValueError(f"{kind} requires the object's explicit grasp_region")
            if kind in ("pick_place", "place"):
                if operation["target"] is None or operation["relation"] is None:
                    raise ValueError(f"{kind} requires target and relation")
                if operation["object"] == operation["target"]:
                    raise ValueError("object and destination must differ")
            elif operation["relation"] is not None:
                raise ValueError("relation is used only for pick_place/place")
            for key in ("approach_direction", "jaw_axis", "rotation_axis"):
                value = operation[key]
                if value is not None and abs(math.hypot(*value)-1.) > .02:
                    raise ValueError(f"{key} must be a unit world vector")
            has_rotation = operation["rotation_axis"] is not None
            if has_rotation != (operation["rotation_angle_rad"] is not None):
                raise ValueError("rotation_axis and rotation_angle_rad must be paired")
            if has_rotation and (kind != "contact_move" or operation["rotation_angle_rad"] == 0):
                raise ValueError("nonzero rotation is used only for contact_move")
            if operation["displacement"] is not None and kind not in ("move", "contact_move"):
                raise ValueError("displacement is used only for move/contact_move")
            if kind == "move" and operation["target"] is None and operation["displacement"] is None:
                raise ValueError("move requires a measured target anchor or relative displacement")
            if kind == "contact_move" and operation["displacement"] is None and not has_rotation:
                raise ValueError("contact_move requires displacement and/or rotation")
            if len(set(operation["contact_parts"])) != len(operation["contact_parts"]):
                raise ValueError("contact_parts must be unique")
            if "palm" in operation["contact_parts"] and kind != "contact_move":
                raise ValueError("palm contact is scoped to contact_move")
            if operation["stop_on"] == "contact" and kind != "contact_move":
                raise ValueError("stop_on contact is scoped to contact_move")
            if any(operation["offset_m"]) and operation["target"] is None and operation["grasp_region"] is None:
                raise ValueError("offset_m requires a target or contact-region anchor")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SchemaError(f"anchored contract: {exc}") from None
    return deepcopy(data)


def _public(value, path="context"):
    """Remove bulky sensor arrays; fail if evaluator-only keys enter the boundary."""
    if isinstance(value, Mapping):
        forbidden = set(value) & _PRIVATE_KEYS
        if forbidden:
            raise ValueError(f"private evaluator fields at {path}: {sorted(forbidden)}")
        return {str(key): _public(item, f"{path}.{key}") for key, item in value.items() if key not in _HEAVY_KEYS}
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return {"array_omitted": True, "shape": list(value.shape)}
        return _public(value.tolist(), path)
    if isinstance(value, (tuple, list)):
        return [_public(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return jsonable(value)


class FoundationVisionHarnessClient(AgentSpeedUpV2Client):
    prompt_version = VERSION
    history_protocol = "fvh_v1_current_rgb_compact_contract_history"

    def __init__(self, *args, reasoning_effort="high", **kwargs):
        if reasoning_effort != "high":
            raise ValueError("foundation_vision_harness_V1 requires high subscription reasoning")
        super().__init__(*args, **kwargs)
        self.reasoning_effort = "high"
        self.image_history = 1
        self._instruction_text = {KIND: INSTRUCTIONS}
        self._schemas = {KIND: harness_schema()}

    def reset_history(self):
        super().reset_history()
        self._contract_history = deque(maxlen=HISTORY_CONTRACTS)
        self._known_targets = {}
        self._pending_frame_id = self._pending_image_shape = None
        self._pose_hemisphere = None
        self.last_harness_timing = None
        for name in ("_kind_calls", "_kind_accepted", "_kind_output_tokens", "_kind_reasoning_tokens", "_kind_over_budget"):
            getattr(self, name)[KIND] = 0

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=self.history_protocol, messages=len(self._contract_history),
                      retained_image_pairs=0, image_history_limit=1, history_contract_limit=HISTORY_CONTRACTS,
                      text_history="bounded accepted operation summaries; current public outcomes supplied by executor",
                      overload_retry_policy="none", transient_retry_policy="none")
        result["visible_tokens_by_kind"][KIND] = self._kind_output_tokens[KIND]-self._kind_reasoning_tokens[KIND]
        result["visible_token_budgets"][KIND] = None
        return result

    def history_snapshot(self):
        return deepcopy(list(self._contract_history))

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
            raise ValueError("V1 accepts only its configured harness request")
        self._request_kind = KIND
        content = AgentSpeedUpV2Client._content(self, request, mode)
        return {"model": self.config.model, "service_tier": "default", "store": False,
                "instructions": INSTRUCTIONS,
                "input": deepcopy(list(self._contract_history))+[{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "foundation_vision_harness_V1_contract",
                                    "strict": True, "schema": harness_schema()}},
                "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}

    def request_key(self, payload, grid=None, mode=KIND):
        return sha256(ordered_dumps({"payload": payload, "version": VERSION,
                                    "endpoint": self.config.endpoint}).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        return validate_contract(data, known_ids=self._known_targets, frame_id=self._pending_frame_id,
                                 image_shape=self._pending_image_shape)

    def _history_input(self, payload):
        # Transport responses are archived, but never retain prior images or
        # full scene snapshots in the next request.
        return []

    def predict(self, request, grid=None, mode=KIND):
        started = perf_counter()
        try:
            with self._history_lock:
                result = ChatGPTPlanHybridClient.predict(self, request, grid, mode)
                self._kind_accepted[KIND] += 1
                self._known_targets.update({target["id"]: deepcopy(target) for target in result["targets"]})
                summary = {"historical_contract_frame": self._pending_frame_id, "status": result["status"],
                           "why": result["why"][:240], "operations": result["operations"],
                           "completion_conditions": result["completion_conditions"]}
                self._contract_history.append({"role": "assistant", "content": json.dumps(summary, ensure_ascii=False)})
                self._history = []
                return result
        finally:
            self.last_harness_timing = {"client_elapsed_s": perf_counter()-started,
                "scope": "payload_auth_transport_parse_log; excludes perception and robot execution",
                "request_key": self.last_request_key, "request_log": self.last_log_path}

    def request_harness(self, view, context):
        if not isinstance(view, StepView):
            raise TypeError("request_harness requires public StepView")
        self.check_observation(view.observation)
        if not isinstance(context, Mapping) or set(context)-CONTEXT_KEYS:
            raise ValueError(f"unexpected harness context fields: {sorted(set(context)-CONTEXT_KEYS) if isinstance(context, Mapping) else 'not mapping'}")
        current = _public(context)
        if current.get("task_prompt", self.task_prompt) != self.task_prompt:
            raise ValueError("context changed the task sentence")
        step, frame_id = current.get("step"), current.get("frame_id")
        if type(step) is not int or step < 0 or type(frame_id) is not int or frame_id < 0:
            raise ValueError("step and frame_id must be nonnegative integers")
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
        for label, rgb in (("External upright RGB", view.external_rgb_upright), ("Wrist upright RGB", view.wrist_rgb_upright)):
            if tuple(rgb.shape[:2]) != shape:
                raise ValueError("RGB raster differs from current calibration")
            images.append((f"{label}, frame {frame_id}, observation step {step}", encode_png_data_url(rgb)))
        self._pending_frame_id, self._pending_image_shape = frame_id, shape
        self.last_request_meta = {"kind": KIND, "step": step, "frame_id": frame_id,
                                  "protocol": self.history_protocol, "version": VERSION}
        return self.predict({"fields": fields, "images": images})
