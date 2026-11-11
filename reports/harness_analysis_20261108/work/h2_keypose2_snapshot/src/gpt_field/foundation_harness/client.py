"""Task contracts and model-proposed keyposes over the subscription transport.

The model supplies task meaning, candidate identity and executable pose targets
from current RGB with predicted geometry as auxiliary evidence. The local
executor owns motion execution and feedback. A partial operation may end at an
observation boundary; no complete manipulation chain is required to propose a
useful approach. Construction and payload building perform no network requests.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from time import perf_counter

from ..agent_speedup_v2_client import AgentSpeedUpV2Client
from ..astra import SchemaError
from ..astranomy_agentic_motion import measured_pose
from ..contact_skills import StepView
from ..direct_history import _trim_images
from ..hybrid_subscription import ChatGPTPlanHybridClient
from ..mark_prompting import encode_png_data_url
from ..ordered_transport import ordered_dumps
from ..robocasa_user_check_geometry import validate_geometry_metadata


VERSION = "foundation-harness-keypose-2"
KIND = "harness"
IMAGE_HISTORY = 6
CONTEXT_KEYS = frozenset({"task_prompt", "step", "scene", "current_operation",
    "operation_history", "feedback", "held", "robot", "capabilities", "geometry",
    "perception", "measured_state", "deadline", "event", "last_feedback",
    "recent_feedback", "targets", "holding", "reason", "robot_joints",
    "possible_payload", "gripper_command", "keypose_index", "motion_evidence"})
OPERATION_KINDS = ("pick_place", "grasp", "place", "move", "contact_move", "release", "observe")
KEYPOSE_NAMES = ("pregrasp", "grasp", "close", "lift", "transport", "preplace", "place",
                "release", "retreat", "move")
INSTRUCTIONS = """You plan task operations and executable keyposes for a general robot harness.
Return one compact JSON object matching the schema. The official task sentence,
two current RGB views, calibrated robot observations, and context.scene are your
evidence. No simulator object poses, contacts, reward, or success predicates are
available. SAM masks and DA3 surfaces are predictions, not complete object shapes.
Scores are uncalibrated; absent/occluded/unseen geometry remains unknown. A visible
surface centroid is not the object centre, a grasp point, or proof of safe motion.
Prefer planning_surface_* over raw extrema when a coherent patch is available;
neither describes unseen object extent. Structured motion_evidence distinguishes
strong observed obstacles from uncertain single-view depth. Use both RGB views
to choose clearance and the next observation when depth is uncertain.
Use RGB and calibrated robot/camera geometry to propose useful pose targets even
when SAM has no mask or DA3 has no usable object geometry. Missing geometry does
not require a query-only response. Keep uncertainty explicit and choose a useful
approach/viewpoint checkpoint when closer visual feedback is needed. Do not wait
for a complete object reconstruction or a complete pick-and-place path before
proposing an approach. Prediction quality and robot feedback remain relevant to
your choice of motion; a supplied pose is not a clearance certificate.

Choose stable target ids and SHORT segmentation concepts such as an object noun
or visible part, not a full relational instruction. Disambiguate same-class
instances using current RGB and supplied candidate ids in selections.external and
selections.wrist; use null when no supported choice exists. Candidate ids belong
only to the current observation. Do not invent an id or associate different
objects across views. Do not rename a held object to escape uncertainty.
Targets may restrict a selected mask with region_pixels.external/wrist: a box
[x0,y0,x1,y1] in ORIGINAL current RGB pixels (exclusive right/bottom), or null.
Use this to choose a support subregion when the task names a spatial relation.
The box selects visible material; it does not establish whole-object fit. Keep
corresponding regions consistent across views. An existing candidate may be
assigned to a new target id only when its query and physical identity agree.

Supply ordered operations with their keyposes. pick_place means grasp object, transfer to
the visible support target, then release and retreat; grasp keeps holding; place
places the currently held object at target and releases; release opens at the
current pose; observe requests fresh evidence without a new motion goal. Explicit
placement_relation must be on or inside for pick_place/place, null otherwise.
Do not substitute on for inside. Use actual visible support/interior evidence and
preserve the requested final relation in completion_conditions.
grasp_region is the target ID of the intended visible material region (for
example a separately queried handle), or null to use the object surface.
Include that region in targets, even if its segmentation is unavailable.
keep_upright is a constraint, not a pose estimate.
contact_mode distinguishes free_object from articulated. An articulated grasp
closes at the selected region without an automatic lift; specify its subsequent
push/pull as contact_move with explicit displacement or keyposes. Motion of an articulated
part indicates possible engagement, not attachment of the whole fixture.

Every operation includes keyposes, an array of 0..12 ordered targets. Each has
name, pose={position:[x,y,z],quaternion_wxyz:[w,x,y,z]}, gripper=open|close|keep,
profile=transit|contact, and observe_after=true|false. Allowed names are pregrasp,
grasp, close, lift, transport, preplace, place, release, retreat and move. Names
describe intended stages, not measured success. You generate these metric poses
from RGB, current robot pose, calibration and available geometric evidence.
Positions are absolute WORLD grip-site coordinates in metres, not hand-origin or
object-centre coordinates; quaternions are normalized absolute right_hand wxyz.
Use profile=contact for near-contact approach, finger closure and placement;
transit is for travel. Retain the last finger command with gripper=keep. A close
stage uses gripper=close; a release stage uses gripper=open. Opening/closing should
usually be a separate pose at the intended arm position, not combined with a
distant arm move. No implicit closure or release is added merely from a name.

grasp, pick_place and place require at least one keypose, but may contain ONLY a
useful initial part, such as pregrasp with observe_after=true. A full grasp/lift/
carry/place/release chain is NOT required. Batch multiple understood keyposes per
call, up to 12 for each operation; there is no one-to-three-pose preference.
Set observe_after=true where a new viewpoint, grasp outcome or released placement
must be inspected before deciding the continuation. Remaining targets beyond
that boundary are proposals; do not assume they have executed. Inspect RGB and
feedback before continuing a carry after uncertain closure or before claiming a
released placement. For partial plans, end the supplied part at an observation
boundary. No attachment, stable support or operation completion follows merely
from reaching the final supplied pose.

move/contact_move use exactly one representation: nonempty keyposes, goal_pose,
or displacement. Do not combine them. goal_pose and every keypose use the absolute
world convention above; displacement is relative to measured operation-entry
position in world axes. For other kinds set goal_pose and displacement to null.
release and observe may use empty keyposes. approach_direction is an optional
world-unit vector. contact_move requires object and an explicit grasp_region
target ID identifying the intended contact region; it does not imply engagement
or attachment. Every optional field must be present with null when unused.

Use status=observe with targets and no operations when new queries or identity
selection are needed before deciding a manipulation. status=plan requires at least
one operation. Return a useful operation sequence rather than one correction per
call. Use a smaller or better-observed keypose when local feedback rejects a
motion; do not let an unavailable full-operation solution prevent a useful partial
approach. Replan from actual outcome evidence; do not repeat an unchanged failed
operation without addressing its reported cause. status=finish is your completion
claim only, never evaluator success. Gripper closure, pose arrival and planned
attachment are not evidence of a successful grasp or stable released placement.
Give short visible completion_conditions. Do not invent benchmark-specific skills.
"""


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def _string(limit=128):
    return {"type": "string", "minLength": 1, "maxLength": limit}


def _vector(length):
    return {"type": "array", "items": {"type": "number"},
            "minItems": length, "maxItems": length}


def harness_schema():
    target = _object({"id": _string(32), "query": _string(64),
        "selections": _object({view: _nullable(_string(128)) for view in ("external", "wrist")}),
        "region_pixels": _object({view: _nullable(_vector(4)) for view in ("external", "wrist")})})
    pose = _object({"position": _vector(3), "quaternion_wxyz": _vector(4)})
    keypose = _object({"name": {"type": "string", "enum": list(KEYPOSE_NAMES)},
        "pose": deepcopy(pose),
        "gripper": {"type": "string", "enum": ["open", "close", "keep"]},
        "profile": {"type": "string", "enum": ["transit", "contact"]},
        "observe_after": {"type": "boolean"}})
    operation = _object({"kind": {"type": "string", "enum": list(OPERATION_KINDS)},
        "object": _nullable(_string(32)), "target": _nullable(_string(32)),
        "contact_mode": {"type": "string", "enum": ["free_object", "articulated"]},
        "grasp_region": _nullable(_string(32)), "keep_upright": {"type": "boolean"},
        "placement_relation": _nullable({"type": "string", "enum": ["on", "inside"]}),
        "goal_pose": _nullable(deepcopy(pose)),
        "displacement": _nullable(_vector(3)), "approach_direction": _nullable(_vector(3)),
        "keyposes": {"type": "array", "items": keypose, "minItems": 0, "maxItems": 12}})
    return _object({"status": {"type": "string", "enum": ["plan", "observe", "finish"]},
        "why": _string(800),
        "targets": {"type": "array", "items": target, "maxItems": 12},
        "operations": {"type": "array", "items": operation, "maxItems": 12},
        "completion_conditions": {"type": "array", "items": _string(160), "maxItems": 12}})


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
    checks = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str) and bool(value.strip()),
        "number": lambda: type(value) in (int, float) and math.isfinite(value),
        "boolean": lambda: type(value) is bool, "null": lambda: value is None}
    if not checks[kind]() or ("enum" in schema and value not in schema["enum"]):
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


def validate_contract(data, *, known_ids=()):
    """Validate identity and executable targets without requiring a complete chain.

    Missing segmentation, unknown surfaces, and absent opposing contact pairs are
    not client admission conditions. They remain evidence for model and executor.
    """
    try:
        _conform(data, harness_schema())
        ids = [target["id"] for target in data["targets"]]
        if len(ids) != len(set(ids)):
            raise ValueError("target ids must be unique")
        for target in data["targets"]:
            for box in target["region_pixels"].values():
                if box is not None and not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                    raise ValueError("region_pixels must have positive area and nonnegative coordinates")
        available = set(ids) | set(known_ids)
        if (data["status"] == "plan") != bool(data["operations"]):
            raise ValueError("only status=plan has nonempty operations")
        for operation in data["operations"]:
            kind = operation["kind"]
            for key in ("object", "target", "grasp_region"):
                if operation[key] is not None and operation[key] not in available:
                    raise ValueError(f"operation {key} must reference a known target id")
            if kind in {"pick_place", "grasp", "place", "contact_move"} and operation["object"] is None:
                raise ValueError(f"{kind} requires object")
            if kind in {"pick_place", "place"} and operation["target"] is None:
                raise ValueError(f"{kind} requires target")
            if (kind in {"pick_place", "place"}) != (operation["placement_relation"] is not None):
                raise ValueError("placement_relation is required only for pick_place/place")
            if operation["object"] is not None and operation["object"] == operation["target"]:
                raise ValueError("manipulated object and placement target must differ")
            keyposes = operation["keyposes"]
            if kind in {"grasp", "pick_place", "place"} and not keyposes:
                raise ValueError(f"{kind} requires at least one model-proposed keypose")
            if kind in {"move", "contact_move"}:
                representations = (int(bool(keyposes)) + int(operation["goal_pose"] is not None)
                                   + int(operation["displacement"] is not None))
                if representations != 1:
                    raise ValueError(f"{kind} needs exactly one of keyposes, goal_pose or displacement")
            if kind == "contact_move" and operation["grasp_region"] is None:
                raise ValueError("contact_move needs a contact region target id")
            if kind not in {"move", "contact_move"} and operation["goal_pose"] is not None:
                raise ValueError("only move/contact_move accept goal_pose")
            if kind not in {"move", "contact_move"} and operation["displacement"] is not None:
                raise ValueError("only move/contact_move accept displacement")
            poses = [keypose["pose"] for keypose in keyposes]
            if operation["goal_pose"] is not None:
                poses.append(operation["goal_pose"])
            for pose in poses:
                if abs(sum(x*x for x in pose["quaternion_wxyz"])-1.) > .02:
                    raise ValueError("goal/keypose quaternion must be normalized")
            direction = operation["approach_direction"]
            if direction is not None and abs(sum(x*x for x in direction)-1.) > .02:
                raise ValueError("approach_direction must be normalized")
    except (ValueError, TypeError, KeyError) as exc:
        raise SchemaError(f"semantic contract: {exc}") from None
    return deepcopy(data)


def bootstrap_targets(prompt):
    """Task-text query seeds only; no benchmark vocabulary or generated motion."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("bootstrap requires the official task sentence")
    # Generic grammar split; the first visual planner can replace these weak seeds.
    chunks = re.split(r"\b(?:and|then|into|onto|inside|on|in|to|from|with|beside|behind|near)\b|[,.;]",
                      prompt.lower())
    queries = []
    for chunk in chunks:
        words = re.findall(r"[a-z][a-z0-9_-]*", chunk)
        while words and words[0] in {"the", "a", "an", "please", "put", "place", "move", "pick",
                "up", "take", "open", "close", "grasp", "lift", "push", "pull", "turn", "restock"}:
            words.pop(0)
        query = " ".join(words)
        if query and len(query) <= 64 and query not in queries:
            queries.append(query)
    if not queries:
        queries = ["object", "support surface"]
    return [{"id": f"seed_{index}", "query": query,
             "selections": {"external": None, "wrist": None},
             "region_pixels": {"external": None, "wrist": None}}
            for index, query in enumerate(queries[:12])]


class FoundationHarnessClient(AgentSpeedUpV2Client):
    prompt_version = VERSION
    history_protocol = "foundation_harness_persistent_text_six_rgb_pairs_v1"

    def __init__(self, *args, reasoning_effort="high", **kwargs):
        if reasoning_effort != "high":
            raise ValueError("foundation harness requires high subscription reasoning")
        super().__init__(*args, **kwargs)
        self.reasoning_effort = "high"
        self.image_history = IMAGE_HISTORY
        self._instruction_text = {KIND: INSTRUCTIONS}
        self._schemas = {KIND: harness_schema()}

    def reset_history(self):
        super().reset_history()
        self._harness_history = []
        self._known_targets = {}
        self.last_harness_timing = None
        for name in ("_kind_calls", "_kind_accepted", "_kind_output_tokens", "_kind_reasoning_tokens", "_kind_over_budget"):
            getattr(self, name)[KIND] = 0

    def history_stats(self):
        result = super().history_stats()
        history = getattr(self, "_harness_history", [])
        result.update(protocol=self.history_protocol, messages=len(history),
            retained_image_pairs=sum(item.get("role") == "user" and isinstance(item.get("content"), list)
                and any(p.get("type") == "input_image" for p in item["content"]) for item in history),
            image_history_limit=IMAGE_HISTORY, text_history="all_accepted_semantic_calls_and_public_feedback",
            overload_retry_policy="none", transient_retry_policy="none")
        result["visible_tokens_by_kind"][KIND] = self._kind_output_tokens[KIND]-self._kind_reasoning_tokens[KIND]
        result["visible_token_budgets"][KIND] = None
        return result

    def history_snapshot(self):
        return deepcopy(self._harness_history)

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
        if mode != KIND:
            raise ValueError("foundation harness only accepts semantic requests")
        if grid is not None and grid != self.grid:
            raise ValueError("grid differs from configured client")
        self._request_kind = KIND
        content = AgentSpeedUpV2Client._content(self, request, mode)
        return {"model": self.config.model, "service_tier": "default", "store": False,
            "instructions": INSTRUCTIONS,
            "input": _trim_images(self._harness_history+[{"role": "user", "content": content}], IMAGE_HISTORY),
            "text": {"format": {"type": "json_schema", "name": "foundation_harness_contract",
                                  "strict": True, "schema": harness_schema()}},
            "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}

    def request_key(self, payload, grid=None, mode=KIND):
        return sha256(ordered_dumps({"payload": payload, "version": VERSION,
                                    "endpoint": self.config.endpoint}).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        return validate_contract(data, known_ids=self._known_targets)

    def _history_input(self, payload):
        return payload["input"]

    def predict(self, request, grid=None, mode=KIND):
        started = perf_counter()
        try:
            with self._history_lock:
                result = ChatGPTPlanHybridClient.predict(self, request, grid, mode)
                self._harness_history = deepcopy(self._history)
                self._kind_accepted[KIND] += 1
                self._known_targets.update({target["id"]: deepcopy(target) for target in result["targets"]})
                return result
        finally:
            self.last_harness_timing = {"client_elapsed_s": perf_counter()-started,
                "scope": "payload_build_auth_transport_parse_log; excludes local perception",
                "request_key": self.last_request_key, "request_log": self.last_log_path}

    def request_harness(self, view, context):
        if not isinstance(view, StepView):
            raise TypeError("request_harness requires public StepView")
        self.check_observation(view.observation)
        if not isinstance(context, Mapping) or set(context)-CONTEXT_KEYS:
            raise ValueError("unexpected semantic context fields")
        current = deepcopy(dict(context))
        if current.get("task_prompt", self.task_prompt) != self.task_prompt:
            raise ValueError("context changed the official task")
        step = current.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        geometry = validate_geometry_metadata(getattr(view.observation, "geometry", None))
        if geometry["cameras"]["wrist"]["pose_source"] != "observed_robot_pose_fixed_mount":
            raise ValueError("wrist calibration must match the current robot observation")
        current.pop("geometry", None)
        fields = {"step": step, "robot_pose_wxyz": measured_pose(view.state),
            "finger_joints_m": [float(x) for x in view.state[6:8]],
            "joint_position_rad": getattr(view.observation, "joint_position", None),
            "joint_velocity_rad_s": getattr(view.observation, "joint_velocity", None),
            "geometry": geometry, "context": current}
        if len(json.dumps(fields, allow_nan=False)) > 120000:
            raise ValueError("semantic context exceeds bounded payload size")
        shape = (geometry["image"]["height"], geometry["image"]["width"])
        images = []
        for label, rgb in (("External upright RGB", view.external_rgb_upright), ("Wrist upright RGB", view.wrist_rgb_upright)):
            if tuple(rgb.shape[:2]) != shape:
                raise ValueError("RGB raster and current calibration differ")
            images.append((f"{label}, observation step {step}", encode_png_data_url(rgb)))
        self.last_request_meta = {"kind": KIND, "step": step, "protocol": self.history_protocol}
        return self.predict({"fields": fields, "images": images})
