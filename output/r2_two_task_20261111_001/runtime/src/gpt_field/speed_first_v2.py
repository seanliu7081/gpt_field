"""Opt-in compact distance policy with explicit local gripper operations.

The model still selects every waypoint, grasp/release intent and recovery.
Only an explicitly requested open/close operation is executed locally; ordinary
five-action Direct outputs are never expanded or reinterpreted.
"""
from __future__ import annotations

import copy
import math

from .astra import SchemaError, _canonical
from .compact_subscription import compact_operational_guidance
from .direct_history import DirectHistoryPolicy
from .policy import Decision
from .speed_first_gripper import LocalGripperSettle
from .speed_first_history import SpeedFirstHistoryMixin, BOUNDED_HISTORY_INSTRUCTIONS
from .speed_first_hybrid import (
    ChatGPTPlanSpeedFirstClient, SpeedFirstHybridPolicy,
    speed_first_instructions, speed_first_output_schema,
)


SPEED_FIRST_V2_VERSION = "speed-first-distance-hybrid-v2"
QUATERNION_NORM_REPAIR_LIMIT = .01


def speed_first_v2_instructions(task_key="bowl_plate", image_history=2):
    text = speed_first_instructions(task_key, image_history).replace(
        "speed-first-distance-hybrid-v1", SPEED_FIRST_V2_VERSION, 1)
    text = text.replace(
        "transport. Use contact_direct for alignment near interaction, grasping, articulated "
        "interaction, insertion, release, and recovery after an unsuccessful attempt.",
        "transport, visible free-space retraction and repositioning after a failed attempt. "
        "Use contact_direct for alignment near interaction, grasping, articulated interaction, "
        "insertion, release and uncertain contact corrections. A recovery waypoint is an explicit "
        "coarse_field goal with phase=recover, selected from the CURRENT views; local code "
        "never infers a recovery direction or replays a fixed task pose.")
    text = text.replace(
        "For contact_direct output exactly five consecutive seven-number actions. They execute "
        "unchanged until an external episode stop; there is no local visual veto or replacement.",
        "For ordinary contact_direct output exactly five consecutive seven-number actions and "
        "set gripper_operation=null. These five actions execute unchanged. Alternatively, "
        "explicitly request a local gripper operation: controller=contact_direct, near_target=true, "
        "field=null, direct=null, gripper_operation={command:open} or {command:close}. Use this "
        "operation when aligned and ready to open/close while holding the CURRENT measured pose, "
        "instead of repeatedly requesting stationary Direct chunks. Local bounded pose feedback "
        "holds that pose and repeats the selected gripper intent. Close executes at least 15 "
        "actions, open at least 10, and either at most 30; measured finger motion followed by "
        "four stable feedback transitions (five measured finger states) may end it early. "
        "Use it for a new open/close intent or when settling is still needed; do not repeat "
        "it merely because fingers are already stationary at the intended state. "
        "An unmoved or unsettled gripper reaches "
        "the budget and returns for your review. Stability certifies neither grasp nor release, "
        "and there is no automatic lift, pull or task-completion claim. Check the resulting "
        "images and state before choosing the next motion. For coarse_field, gripper_operation "
        "must be null. Do not combine Direct action arrays with a gripper operation.")
    text = text.replace(
        f"The latest {image_history} live image pairs and block feedback summaries are supplied.",
        BOUNDED_HISTORY_INSTRUCTIONS)
    if task_key == "bowl_plate":
        text += " " + compact_operational_guidance(
            centered_placement_guidance=True, grasp_settle_guidance=True)
    else:
        text += (
            " Gripper closure may require multiple five-action chunks. Five actions advance only "
            "0.25 seconds; do not start moving away merely because a close command was issued. "
            "Hold the grasp pose until measured finger positions settle, then test engagement "
            "using movement appropriate to the object and mechanism. Correct observed alignment "
            "after a missed grasp. Before release compare the held object and destination in "
            "both views; after release check apparent support and separation.")
    return text


def speed_first_v2_schema(task_key="bowl_plate"):
    schema = speed_first_output_schema(task_key)
    schema["required"].append("gripper_operation")
    schema["properties"]["gripper_operation"] = {"anyOf": [
        {"type": "object", "additionalProperties": False, "required": ["command"],
         "properties": {"command": {"type": "string", "enum": ["open", "close"]}}},
        {"type": "null"}]}
    return schema


class ChatGPTPlanSpeedFirstV2Client(SpeedFirstHistoryMixin, ChatGPTPlanSpeedFirstClient):
    speed_first_v2 = True

    def __init__(self, config=None, **kwargs):
        if type(kwargs.get("image_history", 2)) is not int or kwargs.get("image_history", 2) != 2:
            raise ValueError("V2 requires exactly two live image pairs in request history")
        super().__init__(config, **kwargs)

    @property
    def prompt_version(self):
        return SPEED_FIRST_V2_VERSION

    def _reset_prediction_diagnostics(self):
        super()._reset_prediction_diagnostics()
        self.last_gripper_operation = None
        self.last_quaternion_normalization = None

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = speed_first_v2_instructions(self.task_spec.key, self.image_history)
        payload["text"]["format"] = {
            "type": "json_schema", "name": "astra_speed_first_distance_hybrid_v2", "strict": True,
            "schema": speed_first_v2_schema(self.task_spec.key)}
        return payload

    def validate_response(self, data, grid):
        self._reset_prediction_diagnostics()
        if not isinstance(data, dict) or "gripper_operation" not in data:
            raise SchemaError("V2 requires explicit gripper_operation (null for ordinary control)")
        raw = copy.deepcopy(data)
        adapted = copy.deepcopy(data)
        operation = adapted.pop("gripper_operation")
        if operation is not None:
            if (not isinstance(operation, dict) or set(operation) != {"command"}
                    or operation["command"] not in ("open", "close")):
                raise SchemaError("gripper_operation requires exactly command=open or close")
            if (adapted.get("controller") != "contact_direct" or adapted.get("near_target") is not True
                    or adapted.get("field") is not None or adapted.get("direct") is not None):
                raise SchemaError("Explicit gripper operation requires contact_direct and null field/direct")
            # Adapter envelope for the shared routing validator only. The v2
            # policy must replace these placeholders BEFORE any simulator step.
            adapted["direct"] = {"actions": [[0.] * 6 + [1. if operation["command"] == "close" else -1.]
                                            for _ in range(5)]}
        normalization = None
        field = adapted.get("field")
        if isinstance(field, dict):
            q = field.get("orientation_target_xyzw")
            if (isinstance(q, list) and len(q) == 4
                    and all(type(x) in (int, float) and math.isfinite(x) for x in q)):
                norm = math.sqrt(sum(x * x for x in q))
                if abs(norm - 1.) <= QUATERNION_NORM_REPAIR_LIMIT and norm > 0.:
                    normalized = [x / norm for x in q]
                    field["orientation_target_xyzw"] = normalized
                    if abs(norm - 1.) > 1e-12:
                        normalization = {"kind": "bounded_quaternion_normalization",
                            "raw": q, "raw_norm": norm, "normalized": normalized,
                            "max_norm_error": QUATERNION_NORM_REPAIR_LIMIT,
                            "orientation_direction_changed": False}
        result = super().validate_response(adapted, grid)
        self.last_gripper_operation = copy.deepcopy(operation)
        self.last_quaternion_normalization = normalization
        self.last_compact_prediction = raw
        return result

    def _feedback_action_limit(self, controller):
        if controller == "contact_direct" and self.last_gripper_operation is not None:
            return 30
        return super()._feedback_action_limit(controller)

    def record_gripper_result(self, result):
        with self._history_lock:
            self._history.append({"role": "user", "content": [
                {"type": "input_text", "text": _canonical(result)}]})


class SpeedFirstV2Policy(SpeedFirstHybridPolicy):
    def __init__(self, client):
        if not getattr(client, "speed_first_v2", False):
            raise ValueError("V2 policy requires its explicit operation contract")
        super().__init__(client)
        self.method = SPEED_FIRST_V2_VERSION

    def reset(self):
        super().reset()
        self.gripper_operation = None
        self.gripper_operations = self.gripper_operation_actions = 0

    def act(self, observation):
        # Keep shared routing and transition invariants. Placeholders exist only
        # to enter its contact branch; the following explicit operation always
        # supplies the actual action before it is returned to the runner.
        if self.gripper_operation is not None and not self.gripper_operation.done:
            self.chunk = [[0.] * 7]
        decision = super().act(observation)
        details = dict(decision.diagnostics)
        if details["refreshed"]:
            operation = self.client.last_gripper_operation
            self.gripper_operation = (LocalGripperSettle(operation["command"], observation.state)
                                      if operation is not None else None)
            if self.gripper_operation is not None:
                self.gripper_operations += 1
                self._chunk_length = 30
            details["quaternion_normalization"] = copy.deepcopy(self.client.last_quaternion_normalization)
        details["speed_first_v2"] = True
        details["execution_kind"] = ("local_gripper_settle" if self.gripper_operation is not None
                                     else "field_tracking" if self.controller == "coarse_field"
                                     else "direct_five_actions")
        action = decision.action
        if self.gripper_operation is not None:
            action = self.gripper_operation.next_action(observation.state)
            details["status"] = "local_gripper_settle"
            details["local_gripper"] = self.gripper_operation.diagnostics()
            details["update_steps"] = self._chunk_length
            # No user/model Direct array is claimed for an operation response.
            details.pop("prediction", None)
            details["hybrid_prediction"] = copy.deepcopy(self.client.last_compact_prediction)
        decision = Decision(tuple(action), details)
        self._pending = (self._pending[0], decision)
        return decision

    def observe_transition(self, before, decision, after):
        if self.gripper_operation is None:
            return super().observe_transition(before, decision, after)
        feedback = DirectHistoryPolicy.observe_transition(self, before, decision, after)
        self.adapter._gripper = 1. if decision.action[6] > 0 else -1.
        self.gripper_operation_actions += 1
        result = self.gripper_operation.observe_transition(before.state, decision.action, after.state)
        if self.gripper_operation.done:
            event = {"type": "distance_control_event", "step": self.step_index,
                     "chunk_index": self._chunk_index, "reason": "field_budget_exhausted",
                     "feedback": {"status": "gripper_" + self.gripper_operation.stop_reason,
                                  "reliable": False, "task_success_certified": False}}
            self.client.record_control_event(event)
            self._field_refresh_event = event
            self.client.record_gripper_result({"type": "local_gripper_result", "step": self.step_index,
                "chunk_index": self._chunk_index, **result})
        return feedback
