"""Initial user review followed by model-directed v2 repair and replanning.

Initial grounding still changes image cells only. Subsequent plans belong to the
model, are validated by the original v2 runner, and are audited separately from
the human-approved initial plan. The baseline v2 modules remain unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import agent_speedup_v2 as base
from .agent_speedup_v2 import AgentSpeedUpV2Policy, StepRunner
from .agent_speedup_v2_client import AgentSpeedUpV2Client
from .agent_speedup_v2_plan import PlanIssue
from .agent_speedup_v2_user_check_plan import parse_plan, native_plan, variant_plan_schema, with_extensions, validate_plan
from .agent_speedup_v2_user_check import canonical_sha256, plan_validation, semantic_plan
from .agent_speedup_v2_user_check_framework import FRAMEWORK_VERSION, SKILL_CONTRACTS, TaskContract, intent_key
from .agent_speedup_v2_user_check_runtime import FrameworkRunnerMixin, StableScene, compact_evidence


PROMPT_VERSION = "agent-speedup-v2-user-check-prompt-5"
USER_CHECK_REASONING_EFFORT = "high"
FORBIDDEN_REPAIRS = frozenset(("insert_step", "skip_step", "swap_next", "replan"))
GROUNDING_INSTRUCTIONS = """
USER CHECK EXECUTION. The user has approved the exact user_approved_plan.
Return that complete plan with every field unchanged except each part's w and x.
Those two cells came from the initial view and MUST be updated to the CURRENT
images, using 0 when invisible. Keep the exact scene, part ids, texts, move
values, hold_true, step order and all step fields. Do not add, remove, rewrite or
repair any semantic item. This request updates image grounding only.
"""
REPAIR_INSTRUCTIONS = """
USER CHECK EXECUTION. The exact user_approved_plan is locked for this episode.
Local retry and repoint are allowed. use_part may choose only from that step's
approved grip list (if empty, only its obj). If completing the task requires a
different object, constraint, criterion or step order, report the required
semantic repair honestly; the executor will stop for a new pre-test review.
There is no in-episode human editing.
"""
AUTO_REPAIR_INSTRUCTIONS = """
USER CHECK EXECUTION. The user reviewed the INITIAL plan before execution and
authorized automatic model repair and replanning during this episode. The
reviewed intents in execution_framework.task are the persistent goal contract.
Use the current images, holding state, plan status, last outcome and step history
to choose recovery. You may select another grip, insert or reorder steps, or
request a complete replan when needed. Preserve the official task goal; do not
claim success by omitting an unfinished requirement. Do not ask for a new human
review during execution. Later changes are model decisions, not user approvals.
Use only implemented relations: hold, turn, place_on, release; sense is any.
Only place_on takes a ref; release takes an empty grip list. Stop/angle criteria
take p=null, and only effect criteria take descriptive text.
While holding an object, preserve that object's and the held grip part's IDs and
descriptions until it is released. lateral_unreachable reports local lateral
correction stagnation, not proof that the target is outside the workspace; use
the current evidence to choose retry, repoint, another grip or a revised plan.
"""
AUTO_REPLAN_INSTRUCTIONS = """
USER CHECK EXECUTION. Return a revised complete plan for the official task from
the CURRENT images, holding state, status and step_history. The initial plan was
reviewed before execution; the user authorized you to revise it automatically.
The reviewed intent contract does not require repeating its steps or grip
choices. Account for completed work and any currently held object, and
retain every unfinished task goal. Do not request another human review. The
executor validates this model-generated revision before adopting it.
Use only implemented relations: hold, turn, place_on, release; sense is any.
Only place_on takes a ref; release takes an empty grip list. Stop/angle criteria
take p=null, and only effect criteria take descriptive text.
While holding an object, preserve that object's and the held grip part's IDs and
descriptions until it is released. lateral_unreachable reports local lateral
correction stagnation, not proof that the target is outside the workspace.
"""

FRAMEWORK_INSTRUCTIONS = """
EXECUTION CONTRACT. execution_framework is the current evidence, not a model
claim. The official task and reviewed waypoint intents/constraints persist
across plans. Preserve every uncompleted reviewed intent, including its until
and keep fields; you may change grips/order and insert recovery steps.
Entity IDs are persistent physical bindings. You may improve descriptions of
the SAME entity; never reuse an ID for another object. Rewording alone does not
remeasure or move its geometry. To change a region's physical grounding request
repoint. A repoint is observe_region: physical-object registration needs an empty
hand; typed regions can refresh from already observed support while holding.
If blocked, do not repeat it or rename the target; change the actual missing
resource. release returns a carried object to its original pick support before
opening. To complete the current destination use place_on or observed local
correction and explicit resume_macro; release is not a destination-placement skill.
Use status succeeded/failed/blocked/pending_verification distinctly. A request
or adopted plan is not an observed effect. Pending observation is not failure
or success of the task. For recovery, explain in why which unmet condition the
next action changes or which new evidence it obtains. A replan must start with
an executable step; future evidence may be obtained at skill boundaries.
Retain the reviewed goal even when temporarily staging an object elsewhere.
Do not skip unverified goals. Repeating identical plans/parameters without new
evidence does not reset failure memory. Use real measurements and their
uncertainty; do not turn a controller refusal into a claim of physical
impossibility. Existing motion limits remain enforced by the controllers.
Each keep belongs to its current intent and recovery; future keeps are inactive.
hold_true protects already established relations. Necessary grasp/support contact
is phase-specific; an explicitly contradictory active clear_of needs plan repair.
Preserve the reviewed regions and placement_goals when replanning. Observe and
verify capabilities do not add new macro do values to the plan schema.
"""

METRIC_INSTRUCTIONS = """
METRIC REASONING IS ALLOWED. You may reason about and describe directions,
distances, heights, angles and coordinates when useful. State the frame and
units. Image pixels/cells are not world coordinates. World +z is up; do not
assume image left/right is world -x/+x. Distinguish a visual estimate from a
measurement. Keep the existing symbolic plan schema: metric prose alone does
not command motion. During execution, repair/review can issue local_motion.
"""

LOCAL_CONTROL_INSTRUCTIONS = """
LOCAL CONTROL (in addition to the symbolic repair actions): choose local_motion
when a short physical move or visual correction is needed. Set part=null and
put the motion in step, with exactly these fields:
{"frame":"world","position_mode":"relative","xyz_m":[0,0,0.03],
 "quaternion_wxyz":null,"gripper":"keep","steps":5,"recovery_goal":null}.
This example raises the current grip site by 3 cm; it is not a prescribed path.
For relative position_mode, xyz_m is a displacement from the latest measured
pose. For absolute, xyz_m is a world position in metres (z is world height).
quaternion_wxyz is an absolute world right_hand orientation, or null to keep
the measured orientation. gripper is keep, open, or close. keep preserves the
last gripper command, not a claim that the hand holds an object.
execution_framework.local_control supplies measured pose and recent results.
Each target is at most 5 cm / 0.35 rad from the measured pose and executes
1--5 control steps at 20 Hz. A segment may end before reaching the target.
Inspect the returned measured errors and fresh external/wrist images before
the next segment. Use multiple observed segments for a longer move. You may
lift, retreat, change view, reapproach, adjust a held object, or correct a
placement, as the evidence requires. A controller refusal is not proof that
every route is impossible. Choose clearance based on visible geometry; the
local controller does not plan or certify a collision-free path for you.
Closing fingers does not establish a grasp; opening does not verify support.
Raw motion does not complete any reviewed intent. Use native hold/place_on/
release and their measured checks to reconcile physical state when resuming.
If grasp_state is unknown after direct finger contact, continue observed local
control and request verify_grasp with explicit object and grip identities before
resuming; a raw closing command alone cannot establish identity or a grasp.
After local motion, choose another local_motion, retry, repoint, replan or
another valid symbolic repair using current evidence. Local correction is
also available during visual review after a macro reports success: compare
the actual object, not just the gripper, against the reviewed task goal.
Preserve the reviewed goals/constraints. Do not change other objects just to
make a failure disappear. Native actions keep their original part/step fields;
the local_motion payload is not a do value in the initial user-reviewed plan.
"""


class ExecutionPlanIssue(PlanIssue):
    """Variant capability findings with the native issue interface and field checks."""

    def __post_init__(self):
        if self.code not in {"empty_steps", "relation_not_implemented", "ignored_selection",
                             "held_identity_changed", "entity_identity_changed", "skill_precondition",
                             "reviewed_intent_missing", "reviewed_constraint_missing",
                             "reviewed_hold_true_missing", "recovery_unchanged", "goal_not_verified",
                             "spatial_spec", "constraint_conflict"}:
            return super().__post_init__()
        # Validate shared field types/limits without modifying the native code list.
        PlanIssue("step_required", self.step, self.pid, self.text)


def _capability_issues(raw):
    codes = {"empty_steps", "relation_not_implemented", "ignored_selection", "spatial_spec", "constraint_conflict"}
    return [ExecutionPlanIssue(**issue) for issue in plan_validation(raw, require_native=True)["issues"] if issue["code"] in codes]


def _held_identity_issues(raw, bindings):
    parts = {part["id"]: part for part in raw["parts"]}
    issues = []
    for old in bindings:
        pid = old["id"]
        part = parts.get(pid, {})
        if (part.get("id") == pid and part.get("move") == old["move"]
                and base.kernel.same_object(base.kernel.object_words(part.get("text", "")),
                                            base.kernel.object_words(old["text"]))):
            continue
        # A deleted/rekeyed held ID can no longer name the affected step in the
        # rejected plan. Mark its replacement IDs by the frozen object name so
        # the targeted correction can restore that step, too.
        ids = {pid} | {part["id"] for part in raw["parts"]
                       if base.kernel.same_object(base.kernel.object_words(part["text"]),
                                                  base.kernel.object_words(old["text"]))}
        indices = [i for i, step in enumerate(raw["steps"])
                   if {step["obj"], step["ref"], *step["grip"]} & ids]
        if not indices and pid not in parts:
            # With the live ID entirely removed and no recognizable replacement,
            # a correction must be able to restore the missing held-object step.
            indices = list(range(len(raw["steps"])))
        for index in indices or [None]:
            issues.append(ExecutionPlanIssue(
                "held_identity_changed", index, pid,
                f"Held part {pid} must retain its identity and move prior until released"))
    return issues


class PlanApprovalRequired(KeyboardInterrupt):
    """Controlled episode stop, not a model/schema error.

    The unchanged base runner finalizes physical truth on its interruption path.
    The separate runner preserves that raw result and publishes ``plan_conflict``
    using ``PlanLock.conflict``; it must not present this as an operator SIGINT.
    """


@dataclass
class PlanLock:
    approved_plan: dict
    conflict: dict | None = None
    allow_auto_replan: bool = True
    task_prompt: str = ""
    _approved: dict = field(init=False, repr=False)
    _semantic_sha256: str = field(init=False, repr=False)
    _current_plan: dict = field(init=False, repr=False)
    _initial_plan_grounded: bool = field(default=False, init=False, repr=False)
    _revisions: list[dict] = field(default_factory=list, init=False, repr=False)
    _decisions: list[dict] = field(default_factory=list, init=False, repr=False)
    _held_parts: list[dict] = field(default_factory=list, init=False, repr=False)
    task_contract: TaskContract = field(init=False, repr=False)
    _world_state: dict = field(default_factory=dict, init=False, repr=False)
    _last_feedback: dict | None = field(default=None, init=False, repr=False)
    _completed_intents: list = field(default_factory=list, init=False, repr=False)
    _recovery_memory: object = field(default=None, init=False, repr=False)
    _last_plan_issues: list = field(default_factory=list, init=False, repr=False)
    _local_motion_context: dict = field(default_factory=dict, init=False, repr=False)
    _local_motion_feedback: dict | None = field(default=None, init=False, repr=False)
    _local_motion_history: list = field(default_factory=list, init=False, repr=False)
    _recovery_session: dict = field(default_factory=dict, init=False, repr=False)
    _checkpoint: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        raw = copy.deepcopy(self.approved_plan)
        if parse_plan(raw).to_json() != raw:
            raise ValueError("Approved plan must round-trip without changes")
        self.approved_plan = copy.deepcopy(raw)
        self._approved = raw
        self._current_plan = copy.deepcopy(raw)
        self._semantic_sha256 = canonical_sha256(semantic_plan(raw))
        self.task_contract = TaskContract.from_plan(raw, task_prompt=self.task_prompt)

    def framework_snapshot(self, *, for_model=False):
        recovery = {} if self._recovery_memory is None else (
            self._recovery_memory.for_model() if for_model else self._recovery_memory.to_dict())
        if for_model:
            for row in recovery.get("recent", []):
                row["feedback"] = {k: v for k, v in row["feedback"].items() if k != "evidence"}
        value = {"version": FRAMEWORK_VERSION, "task": self.task_contract.to_dict(),
                 "state": self._world_state, "last_feedback": self._last_feedback,
                 "verified_intents": self._completed_intents,
                 "skills": {key: value.to_dict() for key, value in SKILL_CONTRACTS.items()},
                 "recovery": recovery,
                 "local_control": {"context": self._local_motion_context,
                                   "last_feedback": self._local_motion_feedback,
                                   "recent_segments": self._local_motion_history[-8:],
                                   "recovery_session": self._recovery_session,
                                   "checkpoint": self._checkpoint}}
        if self._world_state.get("execution_owner") == "model":
            # The v6 executive publishes beliefs and optional skill evidence,
            # not the legacy macro prerequisites as a universal control gate.
            value.update(
                version="user-check-agent-first-framework-1",
                execution_owner="model",
                task=self._world_state.get("goal_contract", value["task"]),
                skills=self._local_motion_context.get("capabilities", {}),
                verification_role="evidence; unavailable_observers_do_not_disable_direct_control",
                task_success_source="independent_evaluator",
            )
        # Keep contracts complete. Only evidence/history is compacted in the
        # model view; artifacts retain every recorded structured result.
        if for_model:
            from .agent_speedup_v2_user_check_context import framework_view
            value = framework_view(value)
        return copy.deepcopy(value)

    def plan(self) -> dict:
        return copy.deepcopy(self._approved)

    @property
    def approval_scope(self):
        return "initial_plan_only" if self.allow_auto_replan else "episode_plan"

    def stop(self, reason: str, **details):
        if self.conflict is None:
            self.conflict = {"reason": reason, **details}
        raise PlanApprovalRequired(self.conflict["reason"])

    def check_active(self):
        if self.conflict is not None:
            raise PlanApprovalRequired(self.conflict["reason"])

    def check_plan(self, raw, *, source: str):
        self.check_active()
        if canonical_sha256(semantic_plan(raw)) != self._semantic_sha256:
            self.stop("The proposed plan changes user-approved semantics", source=source)

    def grips(self, index: int) -> tuple[str, ...]:
        step = self._approved["steps"][index]
        return tuple(step["grip"]) or (step["obj"],)

    def record_plan(self, raw: dict, *, source: str, step: int, initial=False, states=None):
        """Snapshot only an adopted plan; image grounding is revision zero."""
        self._current_plan = copy.deepcopy(raw)
        if initial:
            self._initial_plan_grounded = True
        row = {"revision": len(self._revisions), "source": source,
               "approval_scope": self.approval_scope, "user_approved": bool(initial),
               "step": int(step), "plan": copy.deepcopy(raw),
               "plan_sha256": canonical_sha256(raw), "states": copy.deepcopy(states)}
        self._revisions.append(row)
        return copy.deepcopy(row)

    def record_decision(self, **details):
        row = {"decision": len(self._decisions), "approval_scope": self.approval_scope,
               "user_approved": False, **copy.deepcopy(details)}
        self._decisions.append(row)
        return row

    def snapshot(self) -> dict:
        """Detached audit state for run artifacts; approval never transfers to revisions."""
        return copy.deepcopy({
            "allow_auto_replan": self.allow_auto_replan,
            "approval_scope": self.approval_scope,
            "initial_approved_plan": self._approved,
            "initial_approved_plan_sha256": canonical_sha256(self._approved),
            "initial_approved_semantic_sha256": self._semantic_sha256,
            "initial_plan_grounded": self._initial_plan_grounded,
            "current_plan": self._current_plan,
            "plan_revision_count": sum(not row["user_approved"] for row in self._revisions),
            "revisions": self._revisions, "model_decisions": self._decisions,
            "held_part_identities": self._held_parts,
            "conflict": self.conflict,
            "framework": self.framework_snapshot(),
        })


class UserCheckV2Client(AgentSpeedUpV2Client):
    """Variant-only high effort and metric language; the native plan stays intact."""

    prompt_version = PROMPT_VERSION
    reasoning_effort = USER_CHECK_REASONING_EFFORT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_effort = USER_CHECK_REASONING_EFFORT
        from .agent_speedup_v2_user_check_spatial import SPATIAL_INSTRUCTIONS
        for kind in ("plan", "replan"):
            self._schemas[kind] = variant_plan_schema()
            self._instruction_text[kind] += SPATIAL_INSTRUCTIONS
        forbidden = "Never give directions, distances, heights, angles or coordinates."
        for kind in ("plan", "replan", "repair", "review"):
            self._instruction_text[kind] = self._instruction_text[kind].replace(forbidden, "")
            self._instruction_text[kind] = self._instruction_text[kind].replace(
                "you name what matters and which physical relations to establish, never how to move.",
                "you name what matters and which physical relations to establish; "
                "during recovery you may also select short local motions.")
            self._instruction_text[kind] += METRIC_INSTRUCTIONS

    def validate_response(self, data, grid):
        if self._request_kind not in ("plan", "replan"):
            return super().validate_response(data, grid)
        from .astra import SchemaError
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            parse_plan(data, source=self._request_kind)
            if self._reask is not None:
                self._check_retention(self._request_kind, data)
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(f"{self._request_kind} output: {exc}") from None
        return copy.deepcopy(data)

    def request_plan(self, *args, **kwargs):
        raw = super().request_plan(*args, **kwargs)
        kind = "replan" if kwargs.get("replan") else "plan"
        self._previous_plan = {"kind": kind, "plan": parse_plan(raw, source=kind)}
        return raw

    def _check_retention(self, kind, data):
        return super()._check_retention(kind, native_plan(data))

    def _content(self, request, mode):
        from .agent_speedup_v2_user_check_context import bounded_fields
        request = {**request, "fields": bounded_fields(request["fields"])}
        return super()._content(request, mode)

    def request_key(self, payload, grid, mode):
        return canonical_sha256({"base_key": super().request_key(payload, grid, mode),
                                 "user_check_prompt_version": PROMPT_VERSION,
                                 "reasoning_effort": self.reasoning_effort})


class ReviewedV2Client(UserCheckV2Client):
    """Initial review, automatic semantic repair and observed local motion."""

    prompt_version = PROMPT_VERSION

    def __init__(self, *args, approved_plan: dict, lock: PlanLock, **kwargs):
        self.plan_lock = lock
        lock.check_plan(approved_plan, source="client_initialization")
        super().__init__(*args, **kwargs)
        self._instruction_text["plan"] += GROUNDING_INSTRUCTIONS
        for kind in ("repair", "review"):
            from .agent_speedup_v2_user_check_motion import reviewed_repair_schema
            self._schemas[kind] = reviewed_repair_schema()
            self._instruction_text[kind] = self._instruction_text[kind].replace(
                "and step only for insert_step", "and step only for insert_step or local_motion")
            self._instruction_text[kind] += (AUTO_REPAIR_INSTRUCTIONS if lock.allow_auto_replan
                                             else REPAIR_INSTRUCTIONS)
            self._instruction_text[kind] += LOCAL_CONTROL_INSTRUCTIONS
            from .agent_speedup_v2_user_check_motion import REVIEWED_RECOVERY_INSTRUCTIONS
            self._instruction_text[kind] += "\n" + REVIEWED_RECOVERY_INSTRUCTIONS
        if lock.allow_auto_replan:
            self._instruction_text["replan"] += AUTO_REPLAN_INSTRUCTIONS
            for kind in ("repair", "replan"):
                self._instruction_text[kind] = self._instruction_text[kind].replace(
                    "after repeated failures the robot replans or skips the step itself.",
                    "after repeated failures the robot asks you for a revised plan; "
                    "it does not automatically skip the step.")
            for kind in ("repair", "replan", "review"):
                self._instruction_text[kind] = self._instruction_text[kind].replace(
                    "preserve that object's and the held grip part's IDs and\ndescriptions until it is released.",
                    "preserve that object's and the held grip part's physical identities until released.")
                self._instruction_text[kind] += FRAMEWORK_INSTRUCTIONS

    def validate_response(self, data, grid):
        if self._request_kind not in ("repair", "review"):
            return super().validate_response(data, grid)
        from .agent_speedup_v2_user_check_motion import conform_reviewed_repair
        from .astra import SchemaError
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            conform_reviewed_repair(data)
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(f"{self._request_kind} output: {exc}") from None
        return copy.deepcopy(data)

    def _plan_request(self, *args, **kwargs):
        self.plan_lock.check_active()
        replan = bool(kwargs.get("replan"))
        if replan and not self.plan_lock.allow_auto_replan:
            self.plan_lock.stop("Replanning requires a new pre-test review", source="client_replan")
        kind, request, meta = super()._plan_request(*args, **kwargs)
        if not replan:
            request["fields"]["user_approved_plan"] = self.plan_lock.plan()
        request["fields"]["approval_scope"] = self.plan_lock.approval_scope
        request["fields"]["execution_framework"] = self.plan_lock.framework_snapshot(for_model=True)
        if replan:
            request["fields"]["held_part_identities"] = copy.deepcopy(self.plan_lock._held_parts)
        request["fields"]["permitted_changes"] = (
            "model-directed semantic revision preserving the official task goal" if replan else
            "parts.w and parts.x only; current-view image cells")
        meta["user_check_grounding_only"] = not replan
        meta["user_check_auto_replan"] = self.plan_lock.allow_auto_replan
        return kind, request, meta

    def request_plan(self, *args, **kwargs):
        raw = super().request_plan(*args, **kwargs)
        if not kwargs.get("replan"):
            self.plan_lock.check_plan(raw, source="grounding_response")
        return raw

    def _check_retention(self, kind, data):
        # The baseline retention rule must know which steps our extra execution
        # checks named, so correcting an unsupported relation is permitted.
        if self._reask is not None:
            self._reask["issues"] = self.plan_lock._last_plan_issues or (base.validate_plan(self._reask["plan"])
                                      + _capability_issues(self._reask["plan"].to_json())
                                      + _held_identity_issues(self._reask["plan"].to_json(),
                                                              self.plan_lock._held_parts))
        return super()._check_retention(kind, data)

    def _repair_request(self, *args, **kwargs):
        self.plan_lock.check_active()
        kind, request, meta = super()._repair_request(*args, **kwargs)
        request["fields"]["approval_scope"] = self.plan_lock.approval_scope
        request["fields"]["automatic_replanning_allowed"] = self.plan_lock.allow_auto_replan
        request["fields"]["execution_framework"] = self.plan_lock.framework_snapshot(for_model=True)
        meta["user_check_semantics_locked"] = not self.plan_lock.allow_auto_replan
        return kind, request, meta


from .agent_speedup_v2_user_check_local import LocalRecoveryRunnerMixin
from .agent_speedup_v2_user_check_constraints import ConstraintRunnerMixin


class LockedStepRunner(LocalRecoveryRunnerMixin, ConstraintRunnerMixin, FrameworkRunnerMixin, StepRunner):
    def __init__(self, ctx, *, lock: PlanLock, **kwargs):
        self.plan_lock = lock
        super().__init__(ctx, **kwargs)

    def obtain_plan(self, kind: str, *, history=None):
        self.plan_lock.check_active()
        self._sync_framework()
        if kind != "plan" and not self.plan_lock.allow_auto_replan:
            self.plan_lock.stop("Replanning requires a new pre-test review", source="obtain_plan")
        # Keep the native request/re-ask/validation flow, with the same additional
        # capability checks used by the initial review form.
        ctx = self.ctx
        self.plan_lock._held_parts = self._held_part_bindings() if kind == "replan" else []
        self._enter("plan" if kind == "plan" else "replan", None)
        reask, answer = None, None
        for attempt in range(1 + base.PLAN_REASKS):
            k = self.turns(ctx.view)
            block = self.status_block() if kind == "replan" and self.plan is not None else None
            need_type = base.HistoryPlanNeed if kind == "replan" and history is not None else base.PlanNeed
            extra = {"step_history": history} if need_type is base.HistoryPlanNeed else {}
            need = need_type(kind, k, base.FINGERTIP_EDGES[k], status_block=block,
                             reask_error=reask, holding=self.holding_pid(), **extra)
            answer = yield need
            if not isinstance(answer, base.PlanAnswer):
                raise TypeError("a PlanNeed is answered with a PlanAnswer")
            final = attempt == base.PLAN_REASKS
            if answer.plan is not None:
                issues = self._plan_issues(answer.plan)
                self.plan_lock._last_plan_issues = issues
                event = base.plan_validation_event(issues, reask=attempt > 0, final=final or not issues)
            else:
                issues = None
                code = base.SCHEMA_ISSUE_CODE if answer.schema_error else "parse_error"
                event = {"ok": False, "issues": [{"code": code, "step": None, "pid": None,
                         "text": answer.parse_error or base.SCHEMA_REASK_TEXT}],
                         "reask": attempt > 0, "final": final}
            self._emit("plan_validation", **event, kind=kind)
            if answer.plan is not None and not issues:
                self._accepted(kind)
                self._emit("relation_plan", **base.relation_plan_event(answer.plan))
                return answer.plan, answer
            if answer.plan is not None:
                self._rejected(kind)
            reask = (base.issues_text(issues) if issues else
                     (answer.parse_error or base.SCHEMA_REASK_TEXT))[:base.ISSUES_TEXT_MAX_CHARS]
        return None, answer

    def _held_part_bindings(self):
        """Names associated with live hand geometry, including a gripped fixture."""
        held = getattr(self.ctx, "holding", None)
        gripped = (getattr(self.ctx, "memory", None) or {}).get("gripped") or {}
        ids = {getattr(held, "pid", None), getattr(held, "grip_pid", None),
               gripped.get("obj"), gripped.get("pid")} - {None}
        if self.plan is None:
            return []
        return [{"id": part.pid, "text": part.text, "move": part.move}
                for part in self.plan.parts if part.pid in ids]

    def _plan_issues(self, plan):
        raw = plan.to_json()
        issues = (validate_plan(plan) + _capability_issues(raw)
                  + _held_identity_issues(raw, self.plan_lock._held_parts))
        if self.plan is None or not self.plan_lock.allow_auto_replan:
            return issues
        for issue in self.plan_lock.task_contract.validate_replan(raw, self.completed_intents()):
            issues.append(ExecutionPlanIssue(**{k: issue[k] for k in ("code", "step", "pid", "text")}))
        for part in plan.parts:
            old = self._entity_bindings.get(part.pid)
            if old is not None and not base.kernel.same_object(
                    base.kernel.object_words(old["reviewed_description"]), base.kernel.object_words(part.text)):
                issues.append(ExecutionPlanIssue("entity_identity_changed", None, part.pid,
                                                "A known ID cannot be reassigned to another physical object"))
        index, readiness = self.first_step_readiness(plan)
        if readiness is not None:
            missing = ", ".join(x["predicate"] for x in readiness["unmet"])
            issues.append(ExecutionPlanIssue("skill_precondition", index, plan.steps[index].obj,
                                            ("First step is blocked; change resources or obtain evidence: " + missing)[:120]))
        rows = self.recovery_memory.entries
        if rows and rows[-1]["same_state_failures"] >= 2 and self.plan is not None:
            old_steps = self._effective_plan()["steps"]
            if raw["steps"] == old_steps:
                issues.append(ExecutionPlanIssue("recovery_unchanged", 0, plan.steps[0].obj,
                                                "Same steps and parameters failed without new evidence; change the recovery"))
        return issues

    def repair_issues(self, reply, failed_i, *, review, failed=None, outcome=None):
        issues = super().repair_issues(reply, failed_i, review=review, failed=failed, outcome=outcome)
        if reply.answer is not None and reply.answer.action == "insert_step" and reply.answer.step is not None:
            raw = self.plan.to_json()
            raw["steps"] = [reply.answer.step.to_json()]
            at = failed_i if failed_i is not None else len(self.states)
            issues.extend(ExecutionPlanIssue(issue.code, at, issue.pid, issue.text)
                          for issue in _capability_issues(raw))
            if self._contract_runtime:
                from .agent_speedup_v2_user_check_skills import skill_readiness
                readiness = skill_readiness(self.ctx, reply.answer.step)
                if not readiness["ready"] and not readiness.get("can_observe"):
                    issues.append(ExecutionPlanIssue("skill_precondition", at, reply.answer.step.obj,
                                                    "Inserted recovery is blocked in the current resource state"))
        answer = reply.answer
        if answer is not None and answer.action == "skip_step" and failed is not None:
            if not self.stays_hold_now(failed):
                issues.append(ExecutionPlanIssue("goal_not_verified", failed_i, failed.step.obj,
                                                "An unfinished reviewed intent cannot be skipped without verified completion"))
        typed_regions = getattr(self.ctx.scene, "user_check_regions", {})
        if (answer is not None and answer.action == "repoint" and self.holding_pid() is not None
                and answer.part not in typed_regions):
            issues.append(ExecutionPlanIssue("skill_precondition", failed_i, answer.part,
                                            "observe_region needs an empty hand; choose an executable resource-changing step"))
        return issues

    def _record_invalid_plan(self, answer):
        issues = []
        if answer is not None and answer.plan is not None:
            issues = [issue.to_dict() for issue in self._plan_issues(answer.plan)]
        threshold = base.th_value("PLAN_reasks", base.PLAN_REASKS, "DD",
                                  "one model correction request, then invalid_plan; PLAN_REASKS")
        evidence = {"issues": issues, "schema_error": bool(getattr(answer, "schema_error", False)),
                    "parse_error": getattr(answer, "parse_error", None)}
        return self._outcome("plan", "rejected", "invalid_plan", {}, evidence, (threshold,), state=None)

    def adopt(self, plan, answer, *, replan: bool):
        if replan and not self.plan_lock.allow_auto_replan:
            self.plan_lock.stop("Plan replacement requires a new pre-test review", source="adopt")
        if not replan:
            self.plan_lock.check_plan(plan.to_json(), source="adopt")
        counters = copy.deepcopy(self._ladder) if replan else None
        result = yield from super().adopt(plan, answer, replan=replan)
        if counters is not None:
            for key, counter in counters.items():
                self._ladder[key] = counter
        for part in plan.parts:
            self._entity_bindings.setdefault(part.pid, {"id": part.pid, "reviewed_description": part.text,
                                                        "reviewed_move_prior": part.move})
        self._sync_framework()
        self.plan_lock.record_plan(
            self._effective_plan(), source="model_replan" if replan else "initial_grounding",
            step=getattr(self.ctx, "step", 0), initial=not replan, states=self._state_snapshot())
        return result

    def _effective_plan(self):
        self.plan = with_extensions(self.plan, self.plan_lock._current_plan)
        self.ctx.plan = self.plan
        raw = self.plan.to_json()
        for index, state in enumerate(self.states):
            if id(state) in self._grip:
                raw["steps"][index]["grip"] = list(self.grip_of(state))
        return raw

    def _refresh_roles(self):
        self.plan = with_extensions(self.plan, self.plan_lock._current_plan)
        return (yield from super()._refresh_roles())

    def _state_snapshot(self):
        return [{"step": state.i, "status": state.status, "attempts": state.attempts,
                 "reason": state.reason} for state in self.states]

    def apply(self, answer, failed, *, review: bool, outcome):
        self.plan_lock.check_active()
        if not self.plan_lock.allow_auto_replan and answer.action in FORBIDDEN_REPAIRS:
            self.plan_lock.stop(
                "Semantic repair requires a new pre-test review",
                source="review" if review else "repair", action=answer.action,
                proposed_answer=answer.to_json(),
            )
        if not self.plan_lock.allow_auto_replan and answer.action == "use_part":
            target = failed if failed is not None else self.named_state(answer.step)
            index = self.index_of(target) if target is not None else None
            if index is None or answer.part not in self.plan_lock.grips(index):
                self.plan_lock.stop(
                    "The requested grip was not approved for this step",
                    source="use_part", step=index, part=answer.part,
                )
        decision = self.plan_lock.record_decision(
            source="model_review" if review else "model_repair", status="applying",
            step=getattr(self.ctx, "step", 0), answer=answer.to_json(),
            failed_step=self.index_of(failed), holding=self.holding_pid())
        self._last_actual_effect = None
        if answer.action == "repoint" and self._contract_runtime:
            from .agent_speedup_v2_user_check_skills import observe_region
            observed = yield from observe_region(self.ctx, answer.part, relation_layer=self.relations)
            self.controller_returned()
            self.capture_feedback(observed, failed)
            detail = observed.evidence.get("execution_feedback", {})
            registered = (detail.get("status") == "succeeded"
                          and detail.get("geometry_ready", False))
            result = {"action": "repoint", "part": answer.part, "step_i": self.index_of(failed),
                      "registration": "registered" if registered else detail.get("status", "failed"),
                      "status": None if failed is None else "pending"}
            self._last_actual_effect = "reregistered" if registered else "no_change"
            if detail.get("geometry_invalidated", False):
                self._stale_keepouts.add(answer.part)
                for kind in ("clear_of", "hold_true"):
                    self.ctx.scene.keepouts.pop(f"{kind}:{answer.part}", None)
                self.ensure_keepouts()
            if registered:
                self._awaiting.discard(answer.part)
            if failed is not None:
                failed.status = "pending"
        else:
            result = yield from super().apply(answer, failed, review=review, outcome=outcome)
            if answer.action == "repoint":
                self._last_actual_effect = ("reregistered" if result.get("registration") == "registered"
                                            else "no_change")
            elif answer.action == "replan":
                self._last_actual_effect = "replanned" if result.get("replan") == "adopted" else "no_change"
        self._sync_framework()
        decision.update(status="invalid_plan" if result.get("replan") == "invalid_plan" else "applied",
                        result=copy.deepcopy(result))
        if answer.action in {"use_part", "insert_step", "skip_step", "swap_next"}:
            self.plan_lock.record_plan(
                self._effective_plan(), source=f"model_{answer.action}",
                step=getattr(self.ctx, "step", 0), states=self._state_snapshot())
            self._emit("relation_plan", plan=self._effective_plan(), source="model_revision")
            for state in self.states:
                self._emit("step_end", i=self.index_of(state), status=state.status, reason=state.reason)
        return result

    def climb(self, failed, outcome, *, why: str):
        if not self.plan_lock.allow_auto_replan:
            self.plan_lock.stop(
                "Automatic recovery requires replanning or skipping an approved step",
                source="recovery_ladder", step=self.index_of(failed), trigger=why,
            )
        # A ladder escalation must still expose the executable local motion
        # interface. A native repair can resume after a measured move; if the
        # repair is invalid/no-change twice, retain the ordinary replan fallback.
        if why != "no_change_refused":
            answer, info = yield from self.ask("repair", failed, outcome, history=self.step_history(failed))
            if answer is not None:
                predicted = self._effect_of(answer, failed)
                applied = yield from self.apply(answer, failed, review=False, outcome=outcome)
                effect = self._last_actual_effect or predicted
                self._add_row(failed, outcome, answer.action, answer.part, effect)
                applied.update(effect=effect, recovery_trigger=why)
                self._emit("repair", **info, applied=applied)
                return applied
        # The old second rung silently skipped an unfinished step. Here each
        # recovery escalation obtains a new plan from the model instead.
        key = self.step_key(failed.step)
        ladder = self._ladder_of(key)
        ladder["rung"] += 1
        applied = {"action": "replan", "part": None, "step_i": self.index_of(failed),
                   "forced": True, "why": why, "rung": ladder["rung"]}
        decision = self.plan_lock.record_decision(
            source="automatic_recovery", status="requesting_model_replan",
            step=getattr(self.ctx, "step", 0), trigger=why, failed_step=self.index_of(failed))
        self.capture_feedback(outcome, failed)
        plan, answer = yield from self.obtain_plan("replan", history=self.step_history(failed))
        if plan is None:
            self._record_invalid_plan(answer)
            self.halted = True
            applied.update(replan="invalid_plan", status=failed.status)
            decision.update(status="invalid_plan", result=copy.deepcopy(applied))
            return applied
        self._add_row(failed, outcome, "replan", None, "replanned")
        yield from self.adopt(plan, answer, replan=True)
        applied.update(replan="adopted", status=None)
        decision.update(status="adopted", result=copy.deepcopy(applied))
        return applied


class ReviewedV2Policy(AgentSpeedUpV2Policy):
    def __init__(self, client, *, lock: PlanLock, **kwargs):
        if "episode_factory" in kwargs:
            raise ValueError("The user-check policy owns its locked episode factory")
        self.plan_lock = lock

        def episode(ctx):
            from .agent_speedup_v2_user_check_control import build_relation_layer, refresh_relation_layer
            from .agent_speedup_v2_user_check_spatial import WholeObjectHeldFrameEstimator, install_spatial_layer
            if isinstance(ctx.scene, base.SceneModel):
                ctx.scene = StableScene(ctx.robot)
                ctx.held_estimator = WholeObjectHeldFrameEstimator(ctx.scene, ctx.robot)
            layer = build_relation_layer(self._layer)
            if hasattr(layer, "USER_CHECK_NAMESPACE"):
                install_spatial_layer(layer)
                refresh_relation_layer(layer)
            return LockedStepRunner(ctx, lock=lock, relations=layer,
                                    call_counts=self.fulfiller.counts)

        super().__init__(client, episode_factory=episode, **kwargs)

    def reset(self):
        super().reset()
        from .agent_speedup_v2_user_check_local import LocalRecoveryFulfiller
        previous = self.fulfiller
        fulfiller = LocalRecoveryFulfiller(self.client, self._events, clock=self._clock, ctx=self.ctx)
        # The runner already holds these dictionaries after the episode factory
        # was made, so retain their identity as well as the native counters.
        fulfiller.counts, fulfiller.errors = previous.counts, previous.errors
        fulfiller.guard = self._primitive_active
        fulfiller.plan_lock = self.plan_lock
        self.fulfiller = fulfiller
        if not hasattr(self.ctx, "memory"):
            self.ctx.memory = {}
        self.ctx.memory["user_check_gripper_command"] = "open"

    def observe_transition(self, before, decision, after):
        result = super().observe_transition(before, decision, after)
        grip = float(decision.action[6])
        if grip:
            self.ctx.memory["user_check_gripper_command"] = "close" if grip > 0 else "open"
        grasp = self.ctx.memory.get("user_check_grasp_state") or {}
        if self.ctx.holding is not None and grasp.get("status") == "empty":
            self.ctx.memory.pop("user_check_grasp_state", None)
        return result

    def act(self, observation):
        self.plan_lock.check_active()
        return super().act(observation)

    def agent_stats(self):
        result = super().agent_stats()
        result["user_check_framework"] = self.plan_lock.framework_snapshot(for_model=True)
        return result
