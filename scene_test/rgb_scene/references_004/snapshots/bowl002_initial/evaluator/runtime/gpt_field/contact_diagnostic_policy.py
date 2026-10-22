"""Opt-in paired contact-routing diagnostic over the unchanged v4 executor.

Both arms receive the same short objective, observation/history presentation and
operation wrapper. Only the controller allowed for declared contact alignment
differs. Scope is an auditable model claim, not an image-derived contact oracle.
This module never reads simulator state or chooses a movement/check direction.
"""
from __future__ import annotations

import copy
import json
import math

from .astra import SchemaError, _canonical
from .policy import Decision
from .speed_first_v4 import ChatGPTPlanSpeedFirstV4Client, SpeedFirstV4Policy, speed_first_v4_schema


CONTACT_DIAGNOSTIC_VERSION = "paired-contact-routing-diagnostic-v1"
CONTACT_ARMS = ("current", "direct5")
SCOPES = {
    "free_space_reposition": ("move", "recover"),
    "contact_alignment": ("pregrasp", "direct", "move", "recover"),
    "gripper_operation": ("grasp_close", "release"),
    "engagement_check": ("verify_grasp", "articulate"),
}
DIAGNOSTIC_GOAL = (
    "Establish controlled capture of the selected target, then request a short "
    "engagement check with a model-selected mechanism-compatible translation of "
    "at most 3 cm. Inspect the measured response; do not continue to transport, "
    "placement, or completing the source task."
)


def _arm(value):
    if value not in CONTACT_ARMS:
        raise ValueError("contact_arm must be current or direct5")
    return value


def _target(value):
    if value not in ("drawer_handle", "bowl", "black_bowl"):
        raise ValueError("diagnostic_target must be drawer_handle or bowl")
    return "black_bowl" if value == "bowl" else value


def contact_diagnostic_schema(task_key="bowl_plate", *, contact_arm="current",
                              diagnostic_target="bowl"):
    """Use a nested discriminated union; no root anyOf or implicit conversion.

Shared definitions avoid repeating the large visual/field schemas in each
branch. Both arms have the same four branches; only contact alignment changes.
    """
    arm, target = _arm(contact_arm), _target(diagnostic_target)
    base = speed_first_v4_schema(task_key)
    if target not in base["properties"]["target_region"]["enum"]:
        raise ValueError("Diagnostic target is not part of this source task")
    definitions = {"response_" + key: copy.deepcopy(value)
        for key, value in base["properties"].items() if key != "operation"}
    definitions.update({"operation_" + key: copy.deepcopy(value)
        for key, value in base["properties"]["operation"]["properties"].items()})
    definitions["direct_actions"] = next(copy.deepcopy(value) for value in
        base["properties"]["direct"]["anyOf"] if value.get("type") == "object")
    branches = []
    for scope, kinds in SCOPES.items():
        response = {"type": "object", "additionalProperties": False,
            "required": list(base["required"]), "properties": {
                key: {"$ref": "#/$defs/response_" + key}
                for key in base["properties"] if key != "operation"}}
        operation = copy.deepcopy(base["properties"]["operation"])
        operation["properties"] = {key: {"$ref": "#/$defs/operation_" + key}
                                   for key in operation["properties"]}
        operation["properties"]["kind"] = {"type": "string", "enum": list(kinds)}
        operation["properties"]["target_id"] = {"type": "string", "enum": [target]}
        response["properties"]["operation"] = operation
        response["properties"]["target_region"] = {"type": "string", "enum": [target]}
        if arm == "direct5" and scope == "contact_alignment":
            response["properties"].update(
                controller={"type": "string", "enum": ["contact_direct"]},
                near_target={"type": "boolean", "enum": [True]},
                field={"type": "null"},
                direct={"$ref": "#/$defs/direct_actions"},
                gripper_operation={"type": "null"})
            operation["properties"].update(
                local_action_budget={"type": "integer", "enum": [5]},
                allow_local_continuation={"type": "boolean", "enum": [False]})
        branches.append({"type": "object", "additionalProperties": False,
            "required": ["scope", "scope_reason", "response"], "properties": {
                "scope": {"type": "string", "enum": [scope]},
                "scope_reason": {"type": "string", "minLength": 1, "maxLength": 200},
                "response": response}})
    return {"type": "object", "additionalProperties": False, "required": ["decision"],
            "properties": {"decision": {"anyOf": branches}}, "$defs": definitions}


def _diagnostic_instructions(base, *, arm, target, source_task):
    # Change only the shared experimental scope and routing contract. Keep the
    # existing geometry, feedback, gripper, planning and memory instructions.
    text = base.replace("Task: " + source_task + ".", "Source task context: " + source_task + ".", 1)
    text = text.replace("Optimize time to completing the whole task, preserving conditions needed by later operations.",
        "Optimize time to the short diagnostic objective below.")
    text = text.replace("Only raw evaluator LIBERO success completes the episode;",
        "Only raw evaluator LIBERO success certifies the complete source task;")
    text = text.replace("Only LIBERO success ends the episode.",
        "This short diagnostic ends at external harness boundaries; it does not complete the source task.")
    text += (
        f"\n\n{CONTACT_DIAGNOSTIC_VERSION}. SHORT CONTACT DIAGNOSTIC ONLY. Selected target label: "
        f"{target}. Objective: {DIAGNOSTIC_GOAL} The source task text identifies the scene, "
        "not an instruction to complete its later stages. The external harness independently "
        "evaluates the interaction and may stop at its finite diagnostic budget; evaluator "
        "state/outcomes are not supplied to you. Do not invent success from robot motion. "
        "Return exactly {decision:{scope,scope_reason,response}}. response contains the "
        "unchanged v4 operation fields described above. scope_reason is a short current-view "
        "explanation, not measured contact truth. Choose scope=contact_alignment whenever "
        "correcting the open hand relative to the selected contact region, including approach "
        "depth, lateral alignment or grasp orientation corrections. This remains contact "
        "alignment if operation.kind is move, recover or direct; changing the kind name does "
        "not change its scope. Use free_space_reposition only for a purposeful motion away "
        "from contact, reorientation with clearance, or travel to a separate pre-contact "
        "viewpoint; state that purpose rather than relabeling a contact correction. Use "
        "gripper_operation for grasp_close/release, and engagement_check for a separate "
        "verify_grasp/articulate check. Open-hand contact alignment keeps the gripper open "
        "in both arms. Closing/opening and the explicit close-then-check executor are identical "
        "in both arms. Prefer an explicit grasp_close with your chosen verify_motion when "
        "ready to test capture; it performs settled closure and the bounded check without "
        "an intervening model request. A separate engagement_check also stays within 3 cm "
        "of commanded translation (Direct sum of segment lengths, or Field goal offset). "
        "No target pose, contact points, camera calibration, depth or demonstration is added. "
        "Null optional visual references remain valid; they do not delay ordinary execution. "
        "Do not issue transport/place operations or change the selected target.\n")
    if arm == "direct5":
        text += (
            "CONTACT ROUTING ARM direct5: Every contact_alignment response MUST choose "
            "contact_direct with exactly five model-authored actions, local_action_budget=5, "
            "allow_local_continuation=false and gripper_operation=null. This arm-specific "
            "contract overrides the earlier permission to use Field for short pregrasp "
            "corrections. Free-space reposition, gripper operation and engagement-check "
            "routing remain unchanged. No local program converts Field commands into Direct.")
    else:
        text += (
            "CONTACT ROUTING ARM current: Every contact_alignment response retains the "
            "current v4 choice between coarse_field and contact_direct, including Field for "
            "short pregrasp corrections. Free-space reposition, gripper operation and "
            "engagement-check routing remain unchanged. No local program converts commands.")
    return text


def _memory_transcript(history):
    """Presentation-only adapter; never rewrite the accepted wire transcript."""
    result = copy.deepcopy(history)
    for message in result:
        if message.get("role") != "assistant":
            continue
        for part in message.get("content", []):
            if part.get("type") != "output_text":
                continue
            try:
                value = json.loads(part["text"])
            except (ValueError, TypeError, KeyError):
                continue
            wrapper = value.get("decision") if isinstance(value, dict) else None
            if isinstance(wrapper, dict) and isinstance(wrapper.get("response"), dict):
                part["text"] = _canonical(wrapper["response"])
    return result


def _presentation_transcript(history):
    """Tag temporary unwrapped parts so every retained source part is restored.

The integer tags exist only during local compaction; they never reach a request
or accepted history. Older digests can inspect v4 fields, while recent raw model
output remains exactly the originally accepted wrapper, including whitespace.
    """
    result = _memory_transcript(history)
    originals = []
    for source, adapted in zip(history, result):
        for original_part, adapted_part in zip(source.get("content", []), adapted.get("content", [])):
            if original_part.get("text") != adapted_part.get("text"):
                adapted_part["_contact_wire_slot"] = len(originals)
                originals.append(original_part["text"])
    return result, originals


def _restore_wire_presentation(messages, originals):
    for message in messages:
        for part in message.get("content", []):
            slot = part.pop("_contact_wire_slot", None)
            if slot is not None:
                part["text"] = originals[slot]


class ChatGPTPlanContactDiagnosticClient(ChatGPTPlanSpeedFirstV4Client):
    contact_diagnostic = True

    def __init__(self, config=None, *, contact_arm="current", diagnostic_target="bowl", **kwargs):
        self.contact_arm = _arm(contact_arm)
        self.diagnostic_target = diagnostic_target
        self.diagnostic_target_region = _target(diagnostic_target)
        if kwargs.get("demonstration") is not None:
            raise ValueError("Contact diagnostic is zero-shot and does not accept demonstrations")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("Contact diagnostic requires gpt-6-astra")
        super().__init__(config, **kwargs)
        if self.diagnostic_target_region not in self.target_regions:
            raise ValueError("Diagnostic target is not part of this source task")

    @property
    def prompt_version(self):
        return CONTACT_DIAGNOSTIC_VERSION + "+" + self.contact_arm + "+" + self.planner_mode

    def _reset_prediction_diagnostics(self):
        super()._reset_prediction_diagnostics()
        self.last_contact_diagnostic = self.last_contact_wire = None

    def build_payload(self, observation, grid=None, mode="hybrid"):
        with self._history_lock:
            original_history = self._history
            adapted, original_texts = _presentation_transcript(original_history)
            self._history = adapted
            try:
                payload = super().build_payload(observation, grid, mode)
            finally:
                self._history = original_history
            _restore_wire_presentation(payload["input"], original_texts)
            pending_payload, complete = self._bounded_history_pending
            if pending_payload is not payload:
                raise RuntimeError("Diagnostic presentation lost the matching source transcript")
            _restore_wire_presentation(complete, original_texts)
            payload["instructions"] = _diagnostic_instructions(payload["instructions"],
                arm=self.contact_arm, target=self.diagnostic_target_region,
                source_task=self.task_spec.prompt)
            payload["text"]["format"] = {"type": "json_schema", "strict": True,
                "name": "astra_paired_contact_diagnostic_v1", "schema": contact_diagnostic_schema(
                    self.task_spec.key, contact_arm=self.contact_arm,
                    diagnostic_target=self.diagnostic_target)}
            return payload

    def validate_response(self, data, grid):
        if not isinstance(data, dict) or set(data) != {"decision"}:
            raise SchemaError("Contact diagnostic requires exactly the decision wrapper")
        wrapper = data["decision"]
        if not isinstance(wrapper, dict) or set(wrapper) != {"scope", "scope_reason", "response"}:
            raise SchemaError("Diagnostic decision requires scope, scope_reason and response")
        scope, reason, raw = wrapper["scope"], wrapper["scope_reason"], wrapper["response"]
        if not isinstance(scope, str) or scope not in SCOPES or not isinstance(reason, str) or not reason.strip() or len(reason) > 200:
            raise SchemaError("Invalid diagnostic scope or scope_reason")
        if not isinstance(raw, dict) or not isinstance(raw.get("operation"), dict):
            raise SchemaError("Diagnostic response requires the typed v4 operation")
        operation = raw["operation"]
        if operation.get("kind") not in SCOPES[scope]:
            raise SchemaError("Operation kind does not match its diagnostic scope")
        if (raw.get("target_region") != self.diagnostic_target_region
                or operation.get("target_id") != self.diagnostic_target_region):
            raise SchemaError("Diagnostic response must keep the selected target")
        if scope == "contact_alignment":
            if self.contact_arm == "direct5" and (raw.get("controller") != "contact_direct"
                    or operation.get("local_action_budget") != 5
                    or operation.get("allow_local_continuation") is not False
                    or raw.get("gripper_operation") is not None):
                raise SchemaError("direct5 contact alignment requires ordinary Direct five actions")
            if raw.get("controller") == "coarse_field":
                if not isinstance(raw.get("field"), dict) or raw["field"].get("gripper_command") != "open":
                    raise SchemaError("Diagnostic contact alignment requires an open gripper")
            elif raw.get("controller") == "contact_direct":
                direct = raw.get("direct")
                actions = direct.get("actions", []) if isinstance(direct, dict) else []
                if (not isinstance(actions, list) or len(actions) != 5
                        or any(not isinstance(a, list) or len(a) != 7
                               or type(a[6]) not in (int, float) or a[6] >= 0 for a in actions)):
                    raise SchemaError("Diagnostic contact alignment requires five open-gripper actions")
        if scope == "engagement_check":
            if raw.get("controller") == "coarse_field":
                field = raw.get("field")
                vector = field.get("goal_offset_m") if isinstance(field, dict) else None
                vectors = [vector]
                scale = 1.
            else:
                direct = raw.get("direct")
                actions = direct.get("actions") if isinstance(direct, dict) else None
                vectors = [a[:3] for a in actions] if isinstance(actions, list) and all(
                    isinstance(a, list) and len(a) == 7 for a in actions) else []
                scale = .05
            if (not vectors or any(not isinstance(v, list) or len(v) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                    for v in vectors)
                    or scale * sum(math.sqrt(sum(x*x for x in v)) for v in vectors) > .03 + 1e-12):
                raise SchemaError("Diagnostic engagement check exceeds 3 cm commanded translation")
        # All additional checks precede parent validation/diagnostic mutation.
        result = super().validate_response(copy.deepcopy(raw), grid)
        self.last_contact_wire = copy.deepcopy(data)
        self.last_contact_diagnostic = {"scope": scope, "scope_reason": reason,
            "scope_is_model_claim": True, "scope_independently_verified": False,
            "contact_arm": self.contact_arm, "diagnostic_target": self.diagnostic_target_region}
        return result


class ContactDiagnosticPolicy(SpeedFirstV4Policy):
    def __init__(self, client, planner_mode="field"):
        if not getattr(client, "contact_diagnostic", False):
            raise ValueError("Contact diagnostic policy requires its diagnostic client")
        self.contact_arm = client.contact_arm
        super().__init__(client, planner_mode=planner_mode)
        self.method = CONTACT_DIAGNOSTIC_VERSION + "+" + self.contact_arm

    def reset(self):
        super().reset()
        self._contact_records = []
        self._contact_seen = False
        self._closure_started = set()
        self._closure_completed = set()
        self._check_completed = set()
        self._attempt_completed = set()
        self._contact_counts = dict(accepted_responses=0, contact_alignment_responses=0,
            contact_alignment_field_responses=0, contact_alignment_direct_responses=0,
            contact_alignment_with_move_or_recover_kind=0,
            free_space_responses_after_contact_started=0, closure_requests=0,
            closure_operations_started=0, closure_operations_completed=0,
            completed_close_attempts=0, completed_close_check_attempts=0,
            ordinary_direct_actions=0, field_actions=0,
            gripper_actions=0, verification_actions=0)

    def needs_model_request(self):
        """Read-only next-act boundary check; an accepted close/check stays local."""
        if self._pending is not None:
            raise RuntimeError("needs_model_request requires a complete feedback boundary")
        if self._verify_pending:
            return False
        return bool(self.controller is None or self._field_refresh_event is not None
                    or self.step_index - self._chunk_start == self._chunk_length)

    def contact_diagnostic_stats(self):
        return {"version": CONTACT_DIAGNOSTIC_VERSION, "contact_arm": self.contact_arm,
            "diagnostic_target": self.client.diagnostic_target_region,
            **copy.deepcopy(self._contact_counts), "responses": copy.deepcopy(self._contact_records),
            "scope_independently_verified": False, "success_certified": False}

    def operation_stats(self):
        return {**super().operation_stats(), "contact_diagnostic": self.contact_diagnostic_stats()}

    def _validate_envelope(self, value):
        result = super()._validate_envelope(value)
        metadata = copy.deepcopy(self.client.last_contact_diagnostic)
        scope, kind = metadata["scope"], self.operation["kind"]
        counts = self._contact_counts
        counts["accepted_responses"] += 1
        counts["closure_requests"] += int(kind == "grasp_close")
        if scope == "contact_alignment":
            self._contact_seen = True
            counts["contact_alignment_responses"] += 1
            counts["contact_alignment_with_move_or_recover_kind"] += int(kind in ("move", "recover"))
            counter = "contact_alignment_field_responses" if value["controller"] == "coarse_field" else "contact_alignment_direct_responses"
            counts[counter] += 1
        elif scope == "free_space_reposition" and self._contact_seen:
            counts["free_space_responses_after_contact_started"] += 1
        self._contact_records.append({**metadata, "request_index": counts["accepted_responses"],
            "operation_id": self.operation["id"], "kind": kind, "controller": value["controller"],
            "start_step": self.step_index, "last_completed_step": None, "executed_actions": 0,
            "verify_motion": copy.deepcopy(self.operation["verify_motion"]),
            "closure_started": False, "closure_completed": False, "close_check_completed": False})
        return result

    def act(self, observation):
        decision = super().act(observation)
        details = {**decision.diagnostics, "contact_diagnostic": {
            **copy.deepcopy(self.client.last_contact_diagnostic),
            "request_index": self._contact_counts["accepted_responses"]}}
        revised = Decision(decision.action, details)
        self._pending = (self._pending[0], revised)
        return revised

    def observe_transition(self, before, decision, after):
        feedback = super().observe_transition(before, decision, after)
        record = self._contact_records[-1]
        index = record["request_index"]
        record["executed_actions"] += 1
        record["last_completed_step"] = self.step_index - 1
        execution = decision.diagnostics["execution_kind"]
        counter = {"field_tracking": "field_actions", "local_gripper_settle": "gripper_actions",
                   "local_close_verification": "verification_actions"}.get(execution, "ordinary_direct_actions")
        self._contact_counts[counter] += 1
        if record["kind"] == "grasp_close" and execution == "local_gripper_settle":
            if index not in self._closure_started:
                self._closure_started.add(index)
                self._contact_counts["closure_operations_started"] += 1
                record["closure_started"] = True
            if self.gripper_operation is not None and self.gripper_operation.done and index not in self._closure_completed:
                self._closure_completed.add(index)
                self._contact_counts["closure_operations_completed"] += 1
                record["closure_completed"] = True
        check_ended = (execution == "local_close_verification" and self.pending_field_event is not None)
        check_skipped = (record["kind"] == "grasp_close" and record["verify_motion"] is not None
                         and self._operation_last_event is not None
                         and self._operation_last_event.get("reason") == "verification_skipped_fingers_unsettled")
        if (check_ended or check_skipped) and index not in self._check_completed:
            self._check_completed.add(index)
            self._contact_counts["completed_close_check_attempts"] += 1
            record["close_check_completed"] = True
            record["close_check_end_reason"] = ("unsettled_close" if check_skipped else
                                                self.pending_field_event["reason"])
        attempt_ended = (record["kind"] == "grasp_close" and
                         (record["close_check_completed"] if record["verify_motion"] is not None
                          else record["closure_completed"]))
        if attempt_ended and index not in self._attempt_completed:
            self._attempt_completed.add(index)
            self._contact_counts["completed_close_attempts"] += 1
            record["close_attempt_completed"] = True
        return feedback
