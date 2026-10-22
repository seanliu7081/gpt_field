"""Opt-in native RGB / local alignment / lazily evaluated millimetre field.

Only RGB and measured robot feedback reach this module. Fine-grid planning does
not imply millimetre perception accuracy, and pixel agreement is never success.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .astra import AstraClient, SchemaError, _canonical, _numbers
from .attempt_memory import (ATTEMPT_MEMORY_VERSION, attempt_memory_instructions,
    attempt_review_schema, build_attempt_memory, validate_attempt_review)
from .action import _rotvec_quaternion
from .chatgpt_plan import UNSUPPORTED_FIELDS
from .compact_field import validate_compact_field
from .direct_history import _allowed, _trim_images
from .distance_subscription import compact_distance_feedback_history
from .fine_distance_planner import FineCompactField, FineDistancePlanner
from .hybrid_subscription import HYBRID_KEYS
from .policy import Decision, FieldPolicy
from .robot_geometry_guidance import ROBOT_GEOMETRY_GUIDANCE_VERSION, robot_geometry_instructions
from .schema import GridSpec
from .speed_first_history import bounded_history_input
from .speed_first_v2 import (ChatGPTPlanSpeedFirstV2Client, SpeedFirstV2Policy,
    QUATERNION_NORM_REPAIR_LIMIT, speed_first_v2_instructions, speed_first_v2_schema)
from .speed_first_visual import SpeedFirstVisualFeedback, visual_reference_schema, validate_visual_reference
from .speed_first_pregrasp import (PREGRASP_VERSION, PregraspVisualFeedback,
    pregrasp_instructions, pregrasp_reference_schema, validate_pregrasp_reference)
from .tracking import PathTracker
from .target_identity_guidance import (TARGET_IDENTITY_VERSION, normalize_target_description,
    target_identity_instructions)


SPEED_FIRST_V3_VERSION = "speed-first-distance-hybrid-v3"
FINE_GRID = GridSpec(size=161, spacing_m=.001)


def speed_first_v3_instructions(task_key="bowl_plate", image_history=2, *, robot_geometry_guidance=False,
                                target_description=None, attempt_memory=False, pregrasp_feedback=False):
    if type(robot_geometry_guidance) is not bool:
        raise ValueError("robot_geometry_guidance must be a bool")
    if type(attempt_memory) is not bool:
        raise ValueError("attempt_memory must be a bool")
    if type(pregrasp_feedback) is not bool:
        raise ValueError("pregrasp_feedback must be a bool")
    target_description = normalize_target_description(target_description)
    text = speed_first_v2_instructions(task_key, image_history).replace(
        "speed-first-distance-hybrid-v2", SPEED_FIRST_V3_VERSION, 1)
    text = text.replace("control_context contains phase and visual_reference=null;",
                        "control_context contains phase and optional visual_reference;")
    text = text.replace(
        "The 33 cubed grid has 0.005 m spacing. Local Dijkstra computes "
        "26-neighbor shortest path cost to the waypoint; cost is edge length times the average "
        "density, where density is 1 plus nonnegative Gaussian repulsors.",
        "The 161 cubed grid has 0.001 m spacing over the same 16 cm cube. Grid values are "
        "evaluated on demand. A clear uniform field uses an exact straight path; otherwise "
        "lazy 26-neighbor A* searches this 1 mm lattice. Cost is edge length times the average "
        "endpoint density, where density is 1 plus nonnegative Gaussian repulsors.")
    text = text.replace(
        "It does not require image-feature validation, change goals from pixel error, or automatically "
        "raise the arm for another view.",
        "Free-space Field does not require image-feature validation. Only explicitly referenced "
        "near-target planar alignment may refine its goal from local image feedback. There is "
        "no automatic arm raise or probing controller.")
    text += (
        " Both latest RGB cameras are natively rendered and supplied at 512 by 512 pixels. "
        "The 1 mm planning lattice is numerical resolution, not measured scene accuracy. "
        "Use coarse_field with phase=align for a short free-space planar correction of an "
        "already held object near its destination; use Direct for uncertain/contact motion. "
        "Optional control_context.visual_reference is ONLY for this local held-object alignment: "
        "{reference_id,moving_point_uv,target_point_uv,desired_offset_uv,planar_alignment:true," 
        "rigid_grasp:true}. All coordinates refer to the current FIXED EXTERNAL RGB, normalized "
        "[u,v], origin top-left. Select a textured moving patch rigidly attached to the held "
        "object and a distinct stationary textured patch on the destination. desired_offset_uv "
        "is where the moving feature should lie relative to the tracked destination feature, "
        "accounting for their different physical locations and the current object height. "
        "It is not automatically [0,0]; arbitrary bowl and plate patches should not coincide. "
        "Choose features away from fingers/occlusion. Use the wrist view to judge physical "
        "alignment and the grasp; wrist pixels are not valid reference coordinates. "
        "Set visual_reference=null if these assumptions cannot be supported. This never blocks "
        "normal Field or Direct. A supplied reference requires phase=align and a visually "
        "supported rigid grasp with the gripper closed. Hold current height and orientation "
        "through planar alignment; other motions invalidate the local map. Keep the SAME "
        "reference_id across refreshes only while tracking the same feature pair and desired "
        "relation; report their current coordinates. Changing the ID starts a new fit. "
        "The local program tracks both patches and learns how measured world XY motion changes "
        "their relative pixel error. It needs diverse XY motion, not just stationary or collinear "
        "samples. Natural alignment actions provide samples; if needed, a single purposeful "
        "Direct five-action alignment block with small noncollinear XY corrections can supply "
        "them. Do not repeatedly request measurement-only motions when feedback is unavailable. "
        "Once reliable, Field refines its XY goal by at most 1 cm per update, at most 2 cm from "
        "your nominal XY waypoint, replanning locally without another GPT call. Its 30-action "
        "refresh limit remains. Reliable alignment uses a 1 mm combined planner/tracker arrival "
        "tolerance and keeps refining intermediate waypoints while pixel error exceeds 2 pixels. "
        "Bad tracking, changed height/orientation/grip, or ill-conditioned motion disables new "
        "corrections; a previously accepted local waypoint remains until the usual model refresh. "
        "This is reported explicitly in local feedback. No pixel measurement "
        "proves grasp, 3D alignment, contact or success. Review local_visual_feedback and BOTH "
        "current views. Pixel convergence is a cue to inspect, never an instruction to release."
    )
    if robot_geometry_guidance:
        text += "\n\n" + robot_geometry_instructions()
    if target_description is not None:
        text += "\n\n" + target_identity_instructions(target_description)
    if attempt_memory:
        text += "\n\n" + attempt_memory_instructions()
    if pregrasp_feedback:
        text += "\n\n" + pregrasp_instructions()
    return text


def speed_first_v3_schema(task_key="bowl_plate", *, attempt_memory=False, pregrasp_feedback=False):
    if type(attempt_memory) is not bool:
        raise ValueError("attempt_memory must be a bool")
    if type(pregrasp_feedback) is not bool:
        raise ValueError("pregrasp_feedback must be a bool")
    schema = speed_first_v2_schema(task_key)
    schema["properties"]["control_context"]["properties"]["visual_reference"] = visual_reference_schema()
    if attempt_memory:
        schema["required"].append("interaction_review")
        schema["properties"]["interaction_review"] = attempt_review_schema()
    if pregrasp_feedback:
        context = schema["properties"]["control_context"]
        context["required"].append("pregrasp_reference")
        context["properties"]["pregrasp_reference"] = pregrasp_reference_schema()
    return schema


def _execution_envelope(value, regions):
    if not isinstance(value, dict) or set(value) != HYBRID_KEYS:
        raise SchemaError("V3 execution envelope has missing or unexpected fields")
    if value["controller"] not in ("coarse_field", "contact_direct"):
        raise SchemaError("Invalid controller")
    contact = value["controller"] == "contact_direct"
    if value["target_region"] not in regions or type(value["near_target"]) is not bool:
        raise SchemaError("Invalid routing region or near_target")
    if value["near_target"] != contact:
        raise SchemaError("near_target must match contact_direct")
    reason = value["routing_reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise SchemaError("routing_reason must be a short nonempty string")
    if contact:
        if value["field"] is not None:
            raise SchemaError("Inactive field must be null")
        direct = value["direct"]
        if not isinstance(direct, dict) or set(direct) != {"actions"}:
            raise SchemaError("Direct requires actions")
        if not isinstance(direct["actions"], list) or len(direct["actions"]) != 5:
            raise SchemaError("Direct requires exactly five actions")
        try:
            if any(abs(x) > 1 for a in direct["actions"] for x in _numbers(a, 7, "action")):
                raise ValueError("Direct actions must be normalized")
        except (ValueError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc
    else:
        if value["direct"] is not None:
            raise SchemaError("Inactive direct must be null")
        validate_compact_field(value["field"])
    return value


class ChatGPTPlanSpeedFirstV3Client(ChatGPTPlanSpeedFirstV2Client):
    speed_first_v3 = True

    def __init__(self, config=None, *, robot_geometry_guidance=False, target_description=None,
                 attempt_memory=False, pregrasp_feedback=False, **kwargs):
        if type(robot_geometry_guidance) is not bool:
            raise ValueError("robot_geometry_guidance must be a bool")
        if type(attempt_memory) is not bool:
            raise ValueError("attempt_memory must be a bool")
        if type(pregrasp_feedback) is not bool:
            raise ValueError("pregrasp_feedback must be a bool")
        self.pregrasp_feedback = pregrasp_feedback
        self.attempt_memory = attempt_memory
        self.robot_geometry_guidance = robot_geometry_guidance
        self.target_description = normalize_target_description(target_description)
        requested_grid = kwargs.pop("grid", FINE_GRID)
        if requested_grid != FINE_GRID:
            raise ValueError("V3 requires the 161 cubed / 1 mm grid")
        # Parent construction checks the unchanged legacy contract. All v3 wire,
        # validation and field construction paths below use the actual fine grid.
        super().__init__(config, **kwargs)
        self.grid = FINE_GRID
        self.field_representation = "lazy-1mm-weighted-distance-v1"
        self.latest_local_feedback = None

    @property
    def prompt_version(self):
        version = (SPEED_FIRST_V3_VERSION + "+" + ROBOT_GEOMETRY_GUIDANCE_VERSION
                   if self.robot_geometry_guidance else SPEED_FIRST_V3_VERSION)
        if self.target_description is not None:
            version += "+" + TARGET_IDENTITY_VERSION
        if self.attempt_memory:
            version += "+" + ATTEMPT_MEMORY_VERSION
        return version + "+" + PREGRASP_VERSION if self.pregrasp_feedback else version

    def _reset_prediction_diagnostics(self):
        super()._reset_prediction_diagnostics()
        self.last_interaction_review = None
        self.last_pregrasp_reference = None

    def reset_history(self):
        super().reset_history()
        self.last_attempt_memory = None
        self.last_interaction_review = None
        self.latest_pregrasp_feedback = None

    def history_stats(self):
        stats = super().history_stats()
        if self.attempt_memory:
            stats["attempt_memory"] = {
                "version": ATTEMPT_MEMORY_VERSION,
                "scope": "model_context_only",
                "source": "accepted_model_statements_and_measured_robot_feedback_only",
                "last_request_memory": copy.deepcopy(self.last_attempt_memory),
                "model_claims_independently_verified": False,
            }
        return stats

    @staticmethod
    def _instructions(grid, mode, *args):
        return "The versioned compact v3 instructions replace this construction placeholder."

    def build_payload(self, observation, grid=None, mode="hybrid"):
        if mode != "hybrid" or (grid is not None and grid != FINE_GRID):
            raise ValueError("V3 requires hybrid control on its fine grid")
        allowed = _allowed(observation, self.task_prompt)
        with self._history_lock:
            payload = AstraClient.build_payload(self, allowed.to_dict(), self.grid, "field")
            for name in UNSUPPORTED_FIELDS:
                payload.pop(name, None)
            payload.update(store=False, stream=True, reasoning={"effort": self.reasoning_effort},
                           include=["reasoning.encrypted_content"])
            payload["input"][0]["content"][0]["text"] = _canonical({
                "type": "observation", "observation_index": self._observation_count,
                "prompt": allowed.prompt, "observation/state": list(allowed.state)})
            if self.latest_local_feedback is not None:
                payload["input"][0]["content"].append({"type": "input_text", "text": _canonical({
                    "type": "local_visual_feedback", **self.latest_local_feedback})})
            complete = compact_distance_feedback_history(
                _trim_images(self._history + payload["input"], self.image_history))
            payload["input"], self.last_history_presentation = bounded_history_input(complete)
            # Keep the source transcript separate from this one current presentation.
            # The commit hook restores `complete`, so no memory snapshots recursively
            # accumulate in accepted history or leak from failed requests.
            self._bounded_history_pending = (payload, complete)
            if self.pregrasp_feedback and self.latest_pregrasp_feedback is not None:
                payload["input"].append({"role": "user", "content": [{"type": "input_text",
                    "text": _canonical({"type": "pregrasp_visual_feedback",
                        **self.latest_pregrasp_feedback})}]})
            if self.attempt_memory:
                memory = build_attempt_memory(self._history, self._observation_count, allowed.state)
                payload["input"].append({"role": "user", "content": [
                    {"type": "input_text", "text": _canonical(memory)}]})
                self.last_attempt_memory = copy.deepcopy(memory)
            payload["instructions"] = speed_first_v3_instructions(self.task_spec.key, self.image_history,
                robot_geometry_guidance=self.robot_geometry_guidance, target_description=self.target_description,
                attempt_memory=self.attempt_memory, pregrasp_feedback=self.pregrasp_feedback)
            payload["text"]["format"] = {"type": "json_schema", "strict": True,
                "name": "astra_speed_first_distance_hybrid_v3",
                "schema": speed_first_v3_schema(self.task_spec.key, attempt_memory=self.attempt_memory,
                    pregrasp_feedback=self.pregrasp_feedback)}
            return payload

    def validate_response(self, data, grid):
        # Preserve the historical disabled path. The opt-in review and action
        # envelope are validated together before committing prediction diagnostics.
        if not self.attempt_memory and not self.pregrasp_feedback:
            self._reset_prediction_diagnostics()
        required = HYBRID_KEYS | {"control_context", "gripper_operation"}
        if self.attempt_memory:
            required |= {"interaction_review"}
        if grid != FINE_GRID or not isinstance(data, dict) or set(data) != required:
            raise SchemaError("V3 wrapper or grid does not match its contract")
        raw = copy.deepcopy(data)
        adapted = copy.deepcopy(data)
        review = None
        if self.attempt_memory:
            try:
                review = validate_attempt_review(adapted.pop("interaction_review"))
            except ValueError as exc:
                raise SchemaError(str(exc)) from exc
        context = adapted.pop("control_context")
        context_keys = {"phase", "visual_reference"}
        if self.pregrasp_feedback:
            context_keys.add("pregrasp_reference")
        if (not isinstance(context, dict) or set(context) != context_keys
                or context["phase"] not in self.task_spec.control_phases):
            raise SchemaError("Invalid explicit control_context")
        pregrasp = None
        if self.pregrasp_feedback:
            try:
                pregrasp = validate_pregrasp_reference(context.pop("pregrasp_reference"))
            except ValueError as exc:
                raise SchemaError(str(exc)) from exc
            if pregrasp is not None and (context["phase"] != "align" or context["visual_reference"] is not None):
                raise SchemaError("Pregrasp reference requires phase=align without a held-object reference")
        reference = context["visual_reference"]
        if reference is not None:
            try:
                validate_visual_reference(reference)
            except ValueError as exc:
                raise SchemaError(str(exc)) from exc
            if context["phase"] != "align":
                raise SchemaError("Local visual reference requires phase=align")
        operation = adapted.pop("gripper_operation")
        if operation is not None:
            if (not isinstance(operation, dict) or set(operation) != {"command"}
                    or operation["command"] not in ("open", "close")
                    or adapted["controller"] != "contact_direct" or adapted["near_target"] is not True
                    or adapted["field"] is not None or adapted["direct"] is not None):
                raise SchemaError("Invalid explicit gripper operation")
            if reference is not None or pregrasp is not None:
                raise SchemaError("Opening/closing invalidates the visual alignment reference")
            adapted["direct"] = {"actions": [[0.] * 6 + [1. if operation["command"] == "close" else -1.]
                                             for _ in range(5)]}
        field = adapted["field"]
        normalization = None
        if isinstance(field, dict):
            if "goal_reference" not in field or field.pop("goal_reference") is not None:
                raise SchemaError("V3 field.goal_reference must be null; use explicit visual_reference")
            q = field.get("orientation_target_xyzw")
            if (isinstance(q, list) and len(q) == 4
                    and all(type(x) in (int, float) and math.isfinite(x) for x in q)):
                norm = math.sqrt(sum(x * x for x in q))
                if norm > 0 and abs(norm - 1.) <= QUATERNION_NORM_REPAIR_LIMIT:
                    field["orientation_target_xyzw"] = [x / norm for x in q]
                    if abs(norm - 1.) > 1e-12:
                        normalization = {"kind": "bounded_quaternion_normalization", "raw": q,
                            "raw_norm": norm, "normalized": field["orientation_target_xyzw"],
                            "max_norm_error": QUATERNION_NORM_REPAIR_LIMIT, "orientation_direction_changed": False}
        _execution_envelope(adapted, self.target_regions)
        if pregrasp is not None:
            if pregrasp["target_id"] != adapted["target_region"]:
                raise SchemaError("Pregrasp target_id must match the current target_region")
            if adapted["controller"] == "coarse_field":
                if field["gripper_command"] != "open":
                    raise SchemaError("Pregrasp Field requires an open gripper")
            elif any(action[6] >= 0 for action in adapted["direct"]["actions"]):
                raise SchemaError("Pregrasp Direct must retain open-gripper commands")
        if self.attempt_memory or self.pregrasp_feedback:
            self._reset_prediction_diagnostics()
            self.last_interaction_review = copy.deepcopy(review)
            self.last_pregrasp_reference = copy.deepcopy(pregrasp)
        self.last_compact_prediction = raw
        self.last_effective_compact_field = copy.deepcopy(field)
        self.last_control_context = copy.deepcopy(context)
        self.last_gripper_operation = copy.deepcopy(operation)
        self.last_quaternion_normalization = normalization
        return adapted


class _FineFieldPolicy(FieldPolicy):
    def _make_field(self, raw, center):
        return FineCompactField(self.grid, center, raw)


class _FinePlanner(FineDistancePlanner):
    def __init__(self, config, owner):
        super().__init__(config)
        self.owner = owner

    def plan_analytic(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        tolerance = .001 if (self.owner._local_alignment_active()
            or self.owner._pregrasp_alignment_active()) else goal_tolerance_m
        return super().plan_analytic(field, position, goal_xyz=goal_xyz,
            compact_prediction=compact_prediction, goal_tolerance_m=tolerance)


class SpeedFirstV3Policy(SpeedFirstV2Policy):
    def __init__(self, client):
        if not getattr(client, "speed_first_v3", False):
            raise ValueError("V3 policy requires the v3 client")
        super().__init__(client)
        config = self.planner.config
        self._field_policy = _FineFieldPolicy(self._field_cache, grid=self.grid,
            update_steps=30, planner_config=config, execution_steps=10,
            execution_mode="tracking", field_improvements=True)
        self._field_policy.planner = _FinePlanner(config, self)
        self._field_policy.tracker = PathTracker(endpoint_tolerance_m=.0025)
        self.method = SPEED_FIRST_V3_VERSION

    def reset(self):
        super().reset()
        self.visual = SpeedFirstVisualFeedback()
        self.local_feedback = {"status": "not_configured", "reliable": False,
                               "task_success_certified": False}
        self.local_goal_updates = self.local_revision = 0
        self._nominal_goal = None
        self._visual_seconds = self._update_seconds = 0.
        self._last_visual_update_step = -100
        self._local_updated = False
        self._local_goal_active = False
        self.client.latest_local_feedback = None
        self.pregrasp = PregraspVisualFeedback() if self.client.pregrasp_feedback else None
        self.pregrasp_local_feedback = {"status": "not_configured", "tracking_valid": False,
                                       "response_validated": False, "task_success_certified": False}
        self._pregrasp_counters = dict(references_requested=0, references_seeded=0,
            observed_transitions=0, tracked_transitions=0, applied_goals=0,
            nominal_goal_restorations=0, fallbacks=0, invalidations=0, errors=0, local_seconds=0.)
        self._pregrasp_seeded_ids = set()
        self._pregrasp_requested = self._pregrasp_goal_active = False
        self._last_pregrasp_update_step = -100
        self._pregrasp_last_position = None
        self._pregrasp_epoch_summary = None
        self.client.latest_pregrasp_feedback = None

    def pregrasp_stats(self):
        return {"version": PREGRASP_VERSION, **copy.deepcopy(self._pregrasp_counters),
                "last_feedback": copy.deepcopy(self.pregrasp_local_feedback)}

    def _remember_pregrasp_measurement(self, diagnostic, measurement_step=None):
        """Keep one dated measurement basis, never count lost-view error as progress."""
        identifier = diagnostic.get("reference_id")
        previous = self._pregrasp_epoch_summary
        step = self.step_index if measurement_step is None else measurement_step
        error = diagnostic.get("pixel_error_norm")
        measured = (diagnostic.get("tracking_valid", False)
            and isinstance(error, (int, float)) and math.isfinite(error))
        # A 2D wrist-only norm cannot be compared with the previous four-row
        # two-camera norm. Restart the error baseline when a view is removed or
        # restored, while retaining the count of corrections in the same epoch.
        active_views = diagnostic.get("active_views")
        basis = (sorted(view for view in active_views if view in ("external", "wrist"))
                 if isinstance(active_views, (list, tuple)) else None)
        same_epoch = previous is not None and previous["reference_id"] == identifier
        basis_changed = same_epoch and previous.get("measurement_views") != basis
        if measured and identifier and (not same_epoch or basis_changed):
            applied_count = previous["applied_goal_count"] if same_epoch else 0
            change_count = previous.get("measurement_basis_changes", 0) + 1 if basis_changed else 0
            previous = {"reference_id": identifier, "start_step": step,
                "last_measured_step": step, "initial_pixel_error_norm": float(error),
                "latest_pixel_error_norm": float(error), "best_pixel_error_norm": float(error),
                "applied_goal_count": applied_count, "image_error_reduction_px": 0.,
                "measurement_views": basis, "measurement_basis_changes": change_count,
                "stale": False, "invalidation_reason": None,
                "grasp_certified": False, "task_success_certified": False}
        elif measured and previous is not None and previous["reference_id"] == identifier:
            previous = copy.deepcopy(previous)
            previous.update(last_measured_step=step, latest_pixel_error_norm=float(error),
                best_pixel_error_norm=min(previous["best_pixel_error_norm"], float(error)),
                image_error_reduction_px=previous["initial_pixel_error_norm"] - float(error), stale=False)
        if previous is not None and not diagnostic.get("tracking_valid", False):
            previous = copy.deepcopy(previous)
            if not previous["stale"]:
                previous.update(stale=True, invalidated_step=step,
                    invalidation_reason=diagnostic.get("status", "tracking_unavailable"))
        self._pregrasp_epoch_summary = previous
        if previous is not None:
            diagnostic = {**diagnostic, "recent_epoch_summary": copy.deepcopy(previous)}
        return diagnostic

    def _pregrasp_call(self, operation, *, update=False, measurement_step=None):
        """Local vision failure disables this optional correction, never control."""
        began = time.monotonic()
        was_tracking = self.pregrasp_local_feedback.get("tracking_valid", False)
        try:
            value = operation()
            if update:
                goal, diagnostic = value
                value = (None if goal is None else np.asarray(goal, dtype=float), diagnostic)
            else:
                diagnostic = value
            if not isinstance(diagnostic, dict):
                raise TypeError("Local pregrasp diagnostic must be a dictionary")
        except Exception as exc:
            self._pregrasp_counters["errors"] += 1
            try:
                diagnostic = self.pregrasp.invalidate("local_feedback_error")
            except Exception:
                diagnostic = {}
            diagnostic = {**diagnostic, "status": "local_feedback_error",
                "error_type": type(exc).__name__, "tracking_valid": False,
                "response_validated": False, "proposal_available": False,
                "fallback": "continue_model_selected_field_or_direct", "task_success_certified": False}
            value = (None, diagnostic) if update else diagnostic
        diagnostic = self._remember_pregrasp_measurement(value[1] if update else value, measurement_step)
        value = (value[0], diagnostic) if update else diagnostic
        if was_tracking and not diagnostic.get("tracking_valid", False):
            self._pregrasp_counters["invalidations"] += 1
        self.pregrasp_local_feedback = copy.deepcopy(diagnostic)
        elapsed = time.monotonic() - began
        self._pregrasp_counters["local_seconds"] += elapsed
        if update:
            self._update_seconds += elapsed
            self.local_update_total_s += elapsed
        else:
            self._visual_seconds += elapsed
            self.local_visual_total_s += elapsed
        return value

    def _validate_envelope(self, value):
        value = _execution_envelope(value, self.client.target_regions)
        context = self.client.last_control_context
        reference = context["visual_reference"]
        closed = self.adapter._gripper > 0
        if value["field"] is not None:
            command = value["field"]["gripper_command"]
            closed = command == "close" or (command == "hold" and closed)
        began = time.monotonic()
        self.local_feedback = self.visual.configure(self._current_observation, reference,
                                                    gripper_closed=closed)
        elapsed = time.monotonic() - began
        self._visual_seconds += elapsed
        self.local_visual_total_s += elapsed
        self._nominal_goal = (None if value["field"] is None else
            np.asarray(self._current_observation.state[:3]) + np.asarray(value["field"]["goal_offset_m"]))
        self._local_goal_active = False
        if self.pregrasp is not None:
            pregrasp_reference = self.client.last_pregrasp_reference
            self._pregrasp_requested = pregrasp_reference is not None
            self._pregrasp_goal_active = False
            self._last_pregrasp_update_step = -100
            self._pregrasp_last_position = np.asarray(self._current_observation.state[:3])
            if self._pregrasp_requested:
                self._pregrasp_counters["references_requested"] += 1
            opened = (value["field"]["gripper_command"] == "open" if value["field"] is not None
                else self.client.last_gripper_operation is None
                and all(action[6] < 0 for action in value["direct"]["actions"]))
            diagnostic = self._pregrasp_call(lambda: self.pregrasp.configure(
                self._current_observation, pregrasp_reference, gripper_open=opened,
                target_id=value["target_region"]))
            if self._pregrasp_requested:
                identifier = pregrasp_reference["reference_id"]
                if diagnostic.get("tracking_valid") and identifier not in self._pregrasp_seeded_ids:
                    self._pregrasp_seeded_ids.add(identifier)
                    self._pregrasp_counters["references_seeded"] += 1
                elif not diagnostic.get("tracking_valid"):
                    self._pregrasp_counters["fallbacks"] += 1
        return value

    def _maybe_update_goal(self, observation):
        if self.pregrasp is not None and self._pregrasp_requested:
            self._maybe_update_pregrasp_goal(observation)
            return
        if (self.controller != "coarse_field" or self.field is None or self._nominal_goal is None
                or (self.control_context or {}).get("phase") != "align"
                or self._field_cache.effective_compact_field["motion_mode"] != "move"):
            return
        current_goal = np.asarray(self.field.center) + np.asarray(
            self._field_cache.effective_compact_field["goal_offset_m"])
        if (self.step_index - self._last_visual_update_step < 3
                and np.linalg.norm(np.asarray(observation.state[:3]) - current_goal) > .005):
            return
        began = time.monotonic()
        compact = self._field_cache.effective_compact_field
        current = _rotvec_quaternion(np.asarray(observation.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"])
        angle = 2 * math.acos(float(np.clip(abs(np.dot(current, target)), 0., 1.)))
        if angle > self.visual.tracker.maximum_orientation_change_rad:
            goal, diagnostic = None, {**self.visual.diagnostics(),
                "status": "nominal_goal_changes_orientation", "proposal_available": False}
        else:
            goal, diagnostic = self.visual.corrected_goal(observation, self._nominal_goal)
        self.local_feedback = diagnostic
        if goal is not None:
            goal = np.asarray(goal, dtype=float)
            displacement = goal[:2] - self._nominal_goal[:2]
            distance = np.linalg.norm(displacement)
            if distance > .02:
                self.local_feedback = {**diagnostic, "correction_applied": False,
                    "correction_rejected_reason": "outside_nominal_2cm_neighborhood"}
            elif (self.field.query(goal) is not None
                    and np.all(goal >= self.planner.config.workspace_min)
                    and np.all(goal <= self.planner.config.workspace_max)):
                compact = copy.deepcopy(self._field_cache.effective_compact_field)
                old = np.asarray(self.field.center) + np.asarray(compact["goal_offset_m"])
                if np.linalg.norm(goal - old) >= .0001:
                    compact["goal_offset_m"] = (goal - self.field.center).tolist()
                    self._field_cache.effective_compact_field = compact
                    self._field_policy.prediction = compact
                    self._field_policy.field = FineCompactField(self.grid, self.field.center, compact)
                    self._field_policy.clear_pending_plan()
                    self.planner.reset_cache()
                    self.local_goal_updates += 1
                    self.local_revision += 1
                    self._local_updated = True
                    self._local_goal_active = True
                    self._last_visual_update_step = self.step_index
                    self.local_feedback = {**diagnostic, "correction_applied": True,
                        "applied_goal_xyz": goal.tolist(), "local_goal_updates": self.local_goal_updates}
            else:
                self.local_feedback = {**diagnostic, "correction_applied": False,
                                      "correction_rejected_reason": "goal_invalid"}
        elapsed = time.monotonic() - began
        self._update_seconds += elapsed
        self.local_update_total_s += elapsed

    def _set_pregrasp_goal(self, goal, *, restoring=False):
        began = time.monotonic()
        compact = copy.deepcopy(self._field_cache.effective_compact_field)
        old = np.asarray(self.field.center) + np.asarray(compact["goal_offset_m"])
        if np.linalg.norm(goal - old) < .0001:
            return False
        compact["goal_offset_m"] = (goal - self.field.center).tolist()
        self._field_cache.effective_compact_field = compact
        self._field_policy.prediction = compact
        self._field_policy.field = FineCompactField(self.grid, self.field.center, compact)
        self._field_policy.clear_pending_plan()
        self.planner.reset_cache()
        self.local_revision += 1
        self._local_updated = True
        if restoring:
            self._pregrasp_counters["nominal_goal_restorations"] += 1
        else:
            self.local_goal_updates += 1
            self._pregrasp_counters["applied_goals"] += 1
            if self._pregrasp_epoch_summary is not None:
                self._pregrasp_epoch_summary["applied_goal_count"] += 1
        elapsed = time.monotonic() - began
        self._pregrasp_counters["local_seconds"] += elapsed
        self._update_seconds += elapsed
        self.local_update_total_s += elapsed
        return True

    def _restore_pregrasp_nominal(self):
        if self._pregrasp_goal_active and self.field is not None and self._nominal_goal is not None:
            self._set_pregrasp_goal(self._nominal_goal, restoring=True)
            self.pregrasp_local_feedback = {**self.pregrasp_local_feedback,
                "fallback": "resume_model_nominal_goal", "nominal_goal_restored": True,
                "correction_applied": False}
        self._pregrasp_goal_active = False

    def _maybe_update_pregrasp_goal(self, observation):
        if (self.controller != "coarse_field" or self.field is None or self._nominal_goal is None
                or (self.control_context or {}).get("phase") != "align"
                or self._field_cache.effective_compact_field["motion_mode"] != "move"):
            return
        if not self.pregrasp_local_feedback.get("tracking_valid", False):
            self._restore_pregrasp_nominal()
            return
        goal_now = np.asarray(self.field.center) + np.asarray(
            self._field_cache.effective_compact_field["goal_offset_m"])
        if (self.step_index - self._last_pregrasp_update_step < 3
                and np.linalg.norm(np.asarray(observation.state[:3]) - goal_now) > .005):
            return
        compact = self._field_cache.effective_compact_field
        current = _rotvec_quaternion(np.asarray(observation.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"], dtype=float)
        target /= np.linalg.norm(target)
        angle = 2 * math.acos(float(np.clip(abs(np.dot(current, target)), 0., 1.)))
        if angle > self.pregrasp.maximum_orientation_change_rad:
            self._pregrasp_call(lambda: self.pregrasp.invalidate("nominal_goal_changes_orientation"))
            self._pregrasp_counters["fallbacks"] += 1
            self._restore_pregrasp_nominal()
            return
        goal, diagnostic = self._pregrasp_call(
            lambda: self.pregrasp.corrected_goal(observation, self._nominal_goal), update=True)
        self._last_pregrasp_update_step = self.step_index
        if goal is None:
            self._pregrasp_counters["fallbacks"] += 1
            self._restore_pregrasp_nominal()
            return
        goal = np.asarray(goal, dtype=float)
        valid = (goal.shape == (3,) and np.isfinite(goal).all()
            and np.linalg.norm(goal - self._nominal_goal) <= .006 + 1e-12
            and np.all(goal >= self.planner.config.workspace_min)
            and np.all(goal <= self.planner.config.workspace_max)
            and self.field.query(goal) is not None)
        if not valid:
            self.pregrasp_local_feedback = {**diagnostic, "proposal_available": False,
                "correction_applied": False, "correction_rejected_reason": "goal_invalid_or_outside_nominal_6mm"}
            self._pregrasp_counters["fallbacks"] += 1
            self._restore_pregrasp_nominal()
            return
        changed = self._set_pregrasp_goal(goal)
        self._pregrasp_goal_active = bool(np.linalg.norm(goal - self._nominal_goal) >= .0001)
        self.pregrasp_local_feedback = {**diagnostic, "correction_applied": changed,
            "applied_goal_xyz": goal.tolist(), "local_goal_updates": self._pregrasp_counters["applied_goals"]}
        if self._pregrasp_epoch_summary is not None:
            self.pregrasp_local_feedback["recent_epoch_summary"] = copy.deepcopy(self._pregrasp_epoch_summary)

    def _pregrasp_alignment_active(self):
        return (self.pregrasp is not None and self._pregrasp_requested and self._pregrasp_goal_active
            and self.controller == "coarse_field" and (self.control_context or {}).get("phase") == "align"
            and self.pregrasp_local_feedback.get("response_validated", False)
            and self.pregrasp_local_feedback.get("proposal_available", False))

    def _local_alignment_active(self):
        return (self.controller == "coarse_field"
            and (self.control_context or {}).get("phase") == "align"
            and self.local_feedback.get("calibration_valid", False)
            and self.local_feedback.get("proposal_available", False)
            and not self.local_feedback.get("correction_rejected_reason"))

    def _event(self, reason):
        if (reason == "goal_reached" and self._pregrasp_alignment_active()
                and self._pregrasp_last_position is not None):
            goal = np.asarray(self.field.center) + np.asarray(
                self._field_cache.effective_compact_field["goal_offset_m"])
            # A newly corrected waypoint still needs actual motion. Do not hold
            # at an arrived goal waiting for unobservable image dimensions.
            if np.linalg.norm(self._pregrasp_last_position - goal) > .001:
                return
        if (reason == "goal_reached" and self._local_alignment_active()
                and self.local_feedback.get("pixel_error_norm", 0.) > self.visual.tracker.image_tolerance_px):
            # Keep the same finite 30-action interval for a valid intermediate
            # correction. Invalid motion, stall and budget events are untouched.
            return
        return super()._event(reason)

    def act(self, observation):
        self._current_observation = observation
        refreshed = (self.controller is None or self._field_refresh_event is not None
                     or self.step_index - self._chunk_start == self._chunk_length)
        if not refreshed:
            self._maybe_update_goal(observation)
        self._field_policy.tracker.endpoint_tolerance_m = (.0005 if (self._local_alignment_active()
            or self._pregrasp_alignment_active()) else .0025)
        self.client.latest_local_feedback = copy.deepcopy(self.local_feedback)
        if self.pregrasp is not None:
            self.client.latest_pregrasp_feedback = copy.deepcopy(self.pregrasp_local_feedback)
        decision = super().act(observation)
        details = dict(decision.diagnostics)
        details.update(speed_first_v3=True, local_visual_feedback=copy.deepcopy(self.local_feedback),
            local_visual_latency_s=self._visual_seconds, local_update_latency_s=self._update_seconds,
            local_goal_updates=self.local_goal_updates, local_field_revision=self.local_revision,
            local_field_updated=self._local_updated, map_grid=self.grid.to_dict(),
            map_materialization="lazy", rgb_input_size=512)
        if self.pregrasp is not None:
            details.update(pregrasp_local_feedback=copy.deepcopy(self.pregrasp_local_feedback),
                           pregrasp_diagnostics=self.pregrasp_stats())
        self._visual_seconds = self._update_seconds = 0.
        self._local_updated = False
        decision = Decision(decision.action, details)
        self._pending = (self._pending[0], decision)
        return decision

    def _check_field_progress(self, after, decision, feedback):
        # Local alignment can reuse its remaining action budget after reaching a
        # corrected intermediate waypoint. Every other v2 event is preserved.
        self._maybe_update_goal(after)
        super()._check_field_progress(after, decision, feedback)

    def observe_transition(self, before, decision, after):
        # Match the normal feedback boundary before mutating a visual epoch.
        before = _allowed(before.to_dict(), self.client.task_prompt)
        after = _allowed(after.to_dict(), self.client.task_prompt)
        if self._pending is None or before != self._pending[0] or decision is not self._pending[1]:
            raise ValueError("Feedback must correspond to the exact pending observation and decision")
        began = time.monotonic()
        self.local_feedback = (self.visual.invalidate("executed_open_command")
            if decision.action[6] < 0 else self.visual.observe(after))
        elapsed = time.monotonic() - began
        self._visual_seconds += elapsed
        self.local_visual_total_s += elapsed
        if self.pregrasp is not None and self._pregrasp_requested:
            self._pregrasp_counters["observed_transitions"] += 1
            self._pregrasp_last_position = np.asarray(after.state[:3])
            diagnostic = self._pregrasp_call(lambda: self.pregrasp.invalidate("executed_close_command")
                if decision.action[6] > 0 else self.pregrasp.observe(after),
                measurement_step=self.step_index)
            if diagnostic.get("tracking_valid"):
                self._pregrasp_counters["tracked_transitions"] += 1
            else:
                self._restore_pregrasp_nominal()
        feedback = super().observe_transition(before, decision, after)
        if self._local_goal_active and not self.local_feedback.get("calibration_valid"):
            self.local_feedback = {**self.local_feedback,
                "fallback": "follow_last_validated_local_waypoint_until_normal_refresh",
                "new_pixel_corrections_enabled": False}
        self.client.latest_local_feedback = copy.deepcopy(self.local_feedback)
        if self.pregrasp is not None:
            self.client.latest_pregrasp_feedback = copy.deepcopy(self.pregrasp_local_feedback)
        return feedback
