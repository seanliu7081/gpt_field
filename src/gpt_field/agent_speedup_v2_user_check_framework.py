"""Task-independent contracts for the user-check executor.

Only standard-library and JSON values live here.  The task sentence remains the
authoritative task; a legacy v2 plan supplies *reviewed waypoint intents*, not a
claim that all its intermediate effects must hold simultaneously at the end.
Controllers provide observations.  Model statements and expected effects never
become observations merely because a new plan was accepted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping


FRAMEWORK_VERSION = "user-check-framework-2"
FEEDBACK_STATUSES = ("succeeded", "failed", "blocked", "pending_verification")


def _copy(value):
    """Detached finite JSON: interfaces must not hold mutable simulator objects."""
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _wire(value) -> dict:
    if isinstance(value, Mapping):
        return _copy(dict(value))
    for name in ("to_json", "to_dict"):
        method = getattr(value, name, None)
        if callable(method):
            return _copy(method())
    raise TypeError("Expected a JSON mapping or a to_json/to_dict record")


def _set_json(values) -> list:
    return [json.loads(item) for item in sorted({_canonical(v) for v in values})]


def intent_fields(step) -> dict:
    """Reviewed effect signature; grip choice and step ordering are methods."""
    step = _wire(step)
    return {"do": step["do"], "obj": step["obj"], "ref": step.get("ref"),
            "sense": step.get("sense", "any"),
            "until": _set_json(step.get("until", ())) }


def intent_key(step) -> str:
    return _canonical(intent_fields(step))


@dataclass(frozen=True)
class TaskContract:
    """Immutable reviewed intent, independent of the current execution plan.

    ``keep`` applies while its reviewed waypoint is being pursued, including its
    recovery, rather than to every later task step. ``hold_true`` claims are
    maintained once observed established; listing a claim is not evidence that
    it was true at reset. Releases may be freely inserted during recovery.
    """

    task_prompt: str
    _approved_json: str = field(repr=False)

    @classmethod
    def from_plan(cls, raw, task_prompt: str = "") -> "TaskContract":
        plan = _wire(raw)
        if not isinstance(plan.get("parts"), list) or not isinstance(plan.get("steps"), list):
            raise ValueError("Reviewed plan needs parts and steps lists")
        ids = [part["id"] for part in plan["parts"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Reviewed entity IDs must be unique")
        return cls(str(task_prompt), _canonical(plan))

    @property
    def approved_plan(self) -> dict:
        return json.loads(self._approved_json)

    @property
    def reviewed_intents(self) -> list[dict]:
        return [{"key": intent_key(step), "reviewed_step": i,
                 "intent": intent_fields(step), "keep": _set_json(step.get("keep", ())),
                 "scope": "while_pursuing_this_intent_including_recovery"}
                for i, step in enumerate(self.approved_plan["steps"])
                if step["do"] != "release"]

    @property
    def entity_bindings(self) -> dict:
        return {part["id"]: {"id": part["id"], "reviewed_description": part["text"],
                             "reviewed_move_prior": part.get("move", "unknown")}
                for part in self.approved_plan["parts"]}

    def to_dict(self) -> dict:
        return {"version": FRAMEWORK_VERSION, "task_prompt": self.task_prompt,
                "goal_source": "task_prompt_with_user_reviewed_waypoint_intents",
                "reviewed_plan_sha256": hashlib.sha256(self._approved_json.encode()).hexdigest(),
                "entity_bindings": self.entity_bindings,
                "identity_rule": "IDs persist; description edits are not identity changes or geometry evidence",
                "reviewed_intents": self.reviewed_intents,
                "hold_true": [{"claim": claim, "scope": "maintain_once_observed_established"}
                              for claim in self.approved_plan.get("hold_true", ())],
                "regions": self.approved_plan.get("regions", []),
                "placement_goals": self.approved_plan.get("placement_goals", []),
                "completion_rule": "The task evaluator remains authoritative; plan exhaustion is not proof"}

    def validate_replan(self, raw, completed_intents=()) -> list[dict]:
        """Preserve uncompleted reviewed intents, with freely chosen methods.

        Completion keys must come from verified execution, never model claims.
        This is a conservative legacy-schema adapter, not semantic equivalence
        reasoning or an automatic extraction of final predicates from language.
        """
        plan = _wire(raw)
        completed = Counter(completed_intents)
        candidates: dict[str, list[tuple[int, dict]]] = {}
        for i, step in enumerate(plan.get("steps", ())):
            candidates.setdefault(intent_key(step), []).append((i, step))
        issues = []
        for intent in self.reviewed_intents:
            key = intent["key"]
            if completed[key]:
                completed[key] -= 1
                continue
            options = candidates.get(key, [])
            expected_keeps = {_canonical(k) for k in intent["keep"]}
            matched = next((j for j, (_, step) in enumerate(options)
                            if expected_keeps <= {_canonical(k) for k in step.get("keep", ())}), None)
            if matched is not None:
                options.pop(matched)
                continue
            if options:
                issues.append({"code": "reviewed_constraint_missing", "step": options[0][0],
                               "pid": intent["intent"]["obj"],
                               "text": "Preserve the reviewed keep constraints while pursuing this intent",
                               "intent": intent["intent"], "required_keep": intent["keep"]})
            else:
                issues.append({"code": "reviewed_intent_missing", "step": None,
                               "pid": intent["intent"]["obj"],
                               "text": "Retain this uncompleted reviewed intent; methods and order may change",
                               "intent": intent["intent"]})
        claims = {_canonical(claim) for claim in plan.get("hold_true", ())}
        for claim in self.approved_plan.get("hold_true", ()):
            if _canonical(claim) not in claims:
                issues.append({"code": "reviewed_hold_true_missing", "step": None, "pid": claim["a"],
                               "text": "Retain this reviewed maintain-once-established constraint",
                               "claim": _copy(claim)})
        # Destination meaning is part of the reviewed goal, not a freely
        # changeable grasp/motion parameter. A replan may add recovery regions.
        current_regions = {r["id"]: r for r in plan.get("regions", ())}
        for region in self.approved_plan.get("regions", ()):
            if current_regions.get(region["id"]) != region:
                issues.append({"code": "reviewed_constraint_missing", "step": None,
                               "pid": region["id"], "text": "Preserve the reviewed spatial region binding"})
        current_goals = {_canonical(goal) for goal in plan.get("placement_goals", ())}
        for goal in self.approved_plan.get("placement_goals", ()):
            if _canonical(goal) not in current_goals:
                issues.append({"code": "reviewed_constraint_missing", "step": None,
                               "pid": goal["obj"], "text": "Preserve the reviewed whole-object placement goal"})
        return issues


def _value(record, name, default=None):
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _scalar(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        return {"value": float(value.value), "sigma": float(value.sigma), "source": value.source}
    if isinstance(value, Mapping):
        return _copy(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _count(value) -> int | None:
    return None if value is None else int(value) if isinstance(value, (int, float)) else len(value)


def _geometry_summary(record) -> dict:
    """Summaries of existing observations; no new geometry or object truth."""
    xyz = _value(record, "xyz")
    region = _value(record, "region_xy")
    widths = _value(record, "width_by_yaw")
    return {"position_m": None if xyz is None else [float(v) for v in xyz],
            "height_sigma_m": _value(record, "sigma_h_m"),
            "top_z": _scalar(_value(record, "top_z")),
            "support_z": _scalar(_value(record, "support_z")),
            "region_vertices": _count(region),
            "region_cells": _count(_value(record, "region_cells")),
            "width_measurements": _count(widths),
            "registered_empty_hand": bool(_value(record, "registered_empty_hand", False)),
            "provisional": bool(_value(record, "provisional", False)),
            "partial": bool(_value(record, "partial", False)),
            "top_bound": _value(record, "top_bound"),
            "axis_measured": _value(record, "elongation") is not None}


_INCIDENTAL_KEYS = frozenset({"step", "policy_step", "registration_step", "observed_at_step",
                             "text", "description", "reviewed_description", "timestamp",
                             "updated_at", "plan_revision", "attempt", "attempts", "roles", "move_prior",
                             "observer_check"})


def _semantic(value):
    if isinstance(value, Mapping):
        return {key: _semantic(item) for key, item in value.items() if key not in _INCIDENTAL_KEYS}
    if isinstance(value, (tuple, list)):
        return [_semantic(item) for item in value]
    return value


@dataclass(frozen=True)
class StateSnapshot:
    """An observation boundary, with stable entity IDs and explicit evidence age."""

    step: int
    entities: Mapping[str, dict] = field(default_factory=dict)
    holding: str | None = None
    facts: tuple[dict, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records, *, step: int, holding: str | None = None,
                     facts=(), evidence=None) -> "StateSnapshot":
        items = records.items() if isinstance(records, Mapping) else (
            (_value(record, "pid"), record) for record in records)
        entities = {}
        for pid, record in items:
            entities[str(pid)] = {"id": str(pid), "description": _value(record, "text", ""),
                                  "status": _value(record, "status", "unknown"),
                                  "reason": _value(record, "reason"),
                                  "observed_at_step": _value(record, "registration_step"),
                                  "evidence_source": "scene_registration",
                                  "roles": sorted(_value(record, "roles", ())),
                                  "geometry": _geometry_summary(record)}
        return cls(int(step), _copy(entities), holding, tuple(_copy(list(facts))), _copy(evidence or {}))

    def to_dict(self) -> dict:
        return _copy({"step": self.step, "entities": dict(self.entities), "holding": self.holding,
                      "facts": list(self.facts), "evidence": dict(self.evidence)})

    def fingerprint(self) -> str:
        """Ignore wording/time; only observed state can change this signature.

        Exact changed measurements count as new evidence, not physical progress.
        RecoveryMemory also tracks recurring unmet conditions independently.
        """
        return hashlib.sha256(_canonical(_semantic(self.to_dict())).encode()).hexdigest()


@dataclass(frozen=True)
class SkillContract:
    name: str
    parameters: tuple[str, ...]
    preconditions: tuple[str, ...]
    execution_constraints: tuple[str, ...]
    expected_effects: tuple[str, ...]
    verification: tuple[str, ...]
    plan_step: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "parameters": list(self.parameters),
                "preconditions": list(self.preconditions),
                "execution_constraints": list(self.execution_constraints),
                "expected_effects": list(self.expected_effects), "verification": list(self.verification),
                "plan_step": self.plan_step}


SKILL_CONTRACTS = {
    "hold": SkillContract("hold", ("obj", "grip", "keep"),
        ("hand_empty_or_holding_obj", "object_and_grip_geometry_usable", "future_destination_ready_before_grasp"),
        ("robot_limits", "scoped_keep_constraints"), ("holding_obj",), ("measured_hold_verification",)),
    "turn": SkillContract("turn", ("obj", "grip", "until", "keep"),
        ("hand_empty_or_holding_obj", "object_and_grip_geometry_usable", "effect_observable_if_requested"),
        ("robot_limits", "hold_continuity", "scoped_keep_constraints"),
        ("requested_turn_or_effect",), ("measured_motion_or_effect_observer",)),
    "place_on": SkillContract("place_on", ("obj", "grip", "ref", "keep"),
        ("hand_empty_or_holding_obj", "object_geometry_usable", "destination_geometry_usable_before_grasp"),
        ("robot_limits", "hold_continuity_until_support", "scoped_keep_constraints"),
        ("object_supported_on_ref", "hand_empty"), ("support_and_release_verification",)),
    "release": SkillContract("release", ("obj", "keep"),
        ("hand_empty_or_holding_obj",),
        ("robot_limits", "scoped_keep_constraints", "supported_before_opening_for_free_object"),
        ("hand_empty",), ("release_verification",)),
    "observe_region": SkillContract("observe_region", ("part", "purpose"),
        ("part_identified", "hand_empty_when_measurement_requires_it"),
        ("robot_limits", "no_release_without_supported_placement"), ("geometry_evidence_updated",),
        ("requested_geometry_fields_validated",), False),
    "point": SkillContract("point", ("part", "purpose"),
        ("part_identified", "camera_view_available"), (), ("image_seed_proposed",),
        ("seed_is_not_registered_geometry",), False),
    "verify": SkillContract("verify", ("relation", "entities"),
        ("relevant_observations_available",), ("unknown_is_not_false_or_true",),
        ("relation_evidence_updated",), ("observer_verdict_with_evidence",), False),
}


@dataclass(frozen=True)
class ExecutionFeedback:
    """Actual execution outcome. Expected/model-predicted effects stay separate."""

    status: str
    skill: str
    reason: str
    executed: bool
    unmet_conditions: tuple[dict, ...] = ()
    observed_effects: tuple[dict, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    policy_step: int | None = None

    def __post_init__(self):
        if self.status not in FEEDBACK_STATUSES:
            raise ValueError(f"Unknown feedback status: {self.status}")
        if self.status == "succeeded" and self.unmet_conditions:
            raise ValueError("A success cannot carry unmet preconditions")
        if self.status == "blocked" and self.executed:
            raise ValueError("A blocked skill was not executed; report actual failed execution separately")
        _copy(self.to_dict())

    @classmethod
    def from_outcome(cls, outcome, *, policy_step=None, unmet_conditions=(), observed_effects=(),
                     executed: bool | None = None) -> "ExecutionFeedback":
        raw_status = _value(outcome, "status", "unknown")
        evidence = _copy(_value(outcome, "evidence", {}) or {})
        adapter = evidence.get("execution_feedback", {})
        if not isinstance(adapter, Mapping):
            adapter = {}
        # A registration survey may move the wrist while the requested
        # manipulation remains blocked. Count execution of this skill, rather
        # than inferring it from all preceding policy actions.
        if not unmet_conditions:
            unmet_conditions = adapter.get("unmet_conditions", adapter.get("unmet", ()))
        unmet_conditions = tuple({"condition": item} if isinstance(item, str) else _copy(item)
                                 for item in unmet_conditions)
        if not observed_effects:
            observed_effects = adapter.get("observed_effects", ())
        steps = int(_value(outcome, "steps", 0))
        if executed is None:
            executed = bool(adapter["executed"]) if "executed" in adapter else steps > 0
        if adapter.get("status") in FEEDBACK_STATUSES:
            status = adapter["status"]
            if status == "succeeded" and raw_status != "succeeded":
                status = "failed" if raw_status == "failed" or executed else "blocked"
        elif unmet_conditions and not executed:
            status = "blocked"
        elif evidence.get("verification_status") in ("unknown", "pending", "pending_verification"):
            status = "pending_verification"
        elif raw_status == "succeeded":
            status = "succeeded"
        elif raw_status == "failed":
            # A passive observer can disprove a relation without taking a new
            # action. No action count alone does not mean a precondition block.
            status = "failed"
        elif raw_status == "rejected":
            status = "failed" if executed else "blocked"
        else:
            status = "pending_verification"
        if status == "succeeded" and evidence.get("verification_status") in (
                "unknown", "pending", "pending_verification"):
            status = "pending_verification"
        thresholds = []
        for threshold in _value(outcome, "thresholds", ()):
            thresholds.append(_wire(threshold) if hasattr(threshold, "to_dict")
                              or isinstance(threshold, Mapping) else _copy(threshold))
        evidence = {"controller_status": raw_status, "controller_steps": steps,
                    "pids": _copy(_value(outcome, "pids", {})),
                    "measurements": evidence, "thresholds": thresholds,
                    "text": _value(outcome, "evidence_text", "")}
        return cls(status, _value(outcome, "relation", _value(outcome, "skill", "unknown")),
                   _value(outcome, "reason", "unknown"), bool(executed), tuple(_copy(list(unmet_conditions))),
                   tuple(_copy(list(observed_effects))), evidence, policy_step)

    def to_dict(self) -> dict:
        return _copy({"status": self.status, "skill": self.skill, "reason": self.reason,
                      "executed": self.executed, "unmet_conditions": list(self.unmet_conditions),
                      "observed_effects": list(self.observed_effects), "evidence": dict(self.evidence),
                      "policy_step": self.policy_step})


@dataclass
class RecoveryMemory:
    """Episode memory independent of plan revision and step numbering.

    Recording an attempt does not impose an episode cap or claim a retry can
    succeed. It exposes unchanged failures and recurring missing conditions to
    the model. Full entries remain available; only model context is compacted.
    """

    entries: list[dict] = field(default_factory=list)
    _failures: Counter = field(default_factory=Counter, repr=False)
    _conditions: Counter = field(default_factory=Counter, repr=False)

    def record(self, feedback: ExecutionFeedback, state: StateSnapshot, context=None) -> dict:
        result = feedback.to_dict()
        condition_key = _canonical({"skill": feedback.skill, "reason": feedback.reason,
                                    "pids": feedback.evidence.get("pids", {}),
                                    "unmet_conditions": _semantic(result["unmet_conditions"])})
        signature = hashlib.sha256((condition_key + state.fingerprint()).encode()).hexdigest()
        failure = feedback.status != "succeeded"
        if failure:
            self._failures[signature] += 1
            self._conditions[condition_key] += 1
        previous = self.entries[-1] if self.entries else None
        consecutive = (previous["consecutive_unchanged"] + 1
                       if failure and previous and previous["failure_signature"] == signature
                       and previous["feedback"]["status"] != "succeeded" else int(failure))
        entry = {"n": len(self.entries), "feedback": result, "state_step": state.step,
                 "state_fingerprint": state.fingerprint(), "failure_signature": signature,
                 "same_state_failures": self._failures[signature] if failure else 0,
                 "recurring_condition_count": self._conditions[condition_key] if failure else 0,
                 "consecutive_unchanged": consecutive, "context": _copy(context or {})}
        self.entries.append(entry)
        return _copy(entry)

    def to_dict(self) -> dict:
        return {"version": FRAMEWORK_VERSION, "entries": _copy(self.entries)}

    def for_model(self, recent: int = 6) -> dict:
        if recent < 1:
            raise ValueError("recent must be positive")
        repeated = []
        for encoded, count in self._conditions.items():
            if count > 1:
                repeated.append({**json.loads(encoded), "count": count})
        return {"total_outcomes": len(self.entries), "recent": _copy(self.entries[-recent:]),
                "recurring_conditions": repeated,
                "recovery_rule": "Change an unmet condition or obtain relevant new evidence; plan text alone "
                                 "does not establish progress. Preserve reviewed goals and scoped constraints."}
