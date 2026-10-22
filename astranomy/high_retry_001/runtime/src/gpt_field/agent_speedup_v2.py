"""agent_SpeedUp_v2 policy and StepRunner (contract §C13, §C3; spec §4.2, §4.3, §4.11, §3.7).

``AgentSpeedUpV2Policy`` implements the runner contract (``reset``, ``act``, ``observe_transition``,
``agent_stats``, ``calibration``, ``outcomes``, ``ledger``), re-written from the v1 pattern because the v1 methods
are bound to a task-checking class. It is the ONLY place that calls the model client: a model call happens only to
fulfil a Need (plan, point, repair or review) that the episode generator yields at a phase boundary, at most B1
calls per act (one while the rejection storm damping is active), never while a primitive is active. Preemption by
the invariant bus follows the three delivery cases of §C3.3; outcomes recorded through ``ctx.record`` are drained
after every advance and numbered (§C3.2 [C-39]).

``StepRunner`` is the episode generator (§C3.6): Calibrate+ -> plan (one re-ask) -> part roles, plan merge and
region registration -> steps in model order with the implicit releases of §C3.6 and the StepRunner's own
preconditions (until.effect regions registered, keep.clear_of keep-outs present; registrations that wait for an
empty hand are made at the next one; an unmet precondition fails the step) -> stays checks at every relation
boundary (a lost step reopens; a lost already-true claim escalates at once; a claim without a check yet is re-tried
and blocks reviews) -> escalation by one repair request after the relation's own local retries (one request per
boundary for the same failure) -> reviews (at most B3, only when every step is finished and every stays check
holds) -> the hold loop (a preemption of the idle hold is recorded and escalated once). Repairs are unbounded; the
runner's stuck alert is the only cap.

The relation controllers live in ``agent_speedup_v2_relations`` (BR). The policy reads them through one layer
object (the module by default): ``RelationContext``, ``calibrate_plus``, ``register_regions``, ``CONTROLLERS``,
``release``, ``evidence_text``, ``CalibratePlus`` (for ``diagnostics["calibration_probe"]``) and the optional
``stays_check_for`` (skip_step of a failed step, §3.7). The policy takes no task key and never reads the task text:
the client holds the official sentence (spec §4.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time

import numpy as np

from . import agent_speedup_v2_geometry as geometry
from . import agent_speedup_v2_planner as planner
from . import agent_speedup_v2_robot as robot
from . import agent_speedup_v2_types as kernel
from .agent_speedup_v2_observers import OBSERVERS_VERSION, InvariantMonitor, RegionStaysCheck, StaysChecker
from .agent_speedup_v2_plan import (ISSUE_TEXT_MAX_CHARS, ISSUES_TEXT_MAX_CHARS, PLAN_MAX_GRIP, PLAN_SCHEMA_VERSION,
                                    PlanIssue, PlanParseError, issues_text, parse_plan, parse_repair, part_roles,
                                    plan_status_block, plan_validation_event, relation_plan_event, validate_plan,
                                    validate_repair, with_inserted)
from .agent_speedup_v2_scene import SCENE_VERSION, HeldFrameEstimator, SceneModel
from .agent_speedup_v2_types import (EVENTS, FINGERTIP_EDGES, NEED_KINDS, PREEMPTED_PREFIX, RESERVED_DO_KINDS,
                                     CallRecord, Plan, PlanAnswer, PlanNeed, PointAnswer, PointNeed, RelationOutcome,
                                     RepairAnswer, RepairNeed, RepairReply, RobotModel, StepContext, StepState,
                                     jsonable, th_value)
from .contact_skills import HoldPrimitive, Orient, StepView
from .policy import Decision
from .visual_field_v8 import measured_current_pose_hold_action

RELATIONS_MODULE = "agent_speedup_v2_relations"
try:                                       # BR (Wave 2) builds the relation layer; the policy imports before it lands
    from . import agent_speedup_v2_relations as relations
except ImportError as missing:             # only the absence of that module itself is tolerated
    if getattr(missing, "name_from", None) != RELATIONS_MODULE and not str(missing.name or "").endswith(
            RELATIONS_MODULE):
        raise
    relations = None

POLICY_VERSION = "agent-speedup-v2-policy-1"
CONTROLLER = "agent_relation"

LABELS = {
    "B1_MAX_CALLS_PER_ACT": ("DD", "spec §3.2 call rules (B1): at most 2 model calls per act (v1 "
                                   "MAX_DECISIONS_PER_ACT, agent_speedup_v1.py:37, restated)"),
    "B1_STORM_CALLS_PER_ACT": ("DD", "spec §3.4 / v1 rejection-storm damping: one model call per act after "
                                     "REJECTION_STORM_LIMIT consecutive rejections (agent_speedup_v1.py:38, :365)"),
    "REJECTION_STORM_LIMIT": ("DD", "v1 REJECTION_STORM_LIMIT, agent_speedup_v1.py:38, restated: 3 consecutive "
                                    "rejections switch the call limit to B1_STORM_CALLS_PER_ACT"),
    "MAX_TRANSITIONS_PER_ACT": ("DD", "v1 MAX_TRANSITIONS_PER_ACT, agent_speedup_v1.py:40, restated: generator "
                                      "advances per act before an idle hold action"),
    "B3_MAX_REVIEWS": ("DD", "spec §3.7 (B3): at most 2 reviews per episode"),
    "PLAN_REASKS": ("DD", "spec §3.4: an invalid plan gets one re-ask with the error text, then invalid_plan"),
    "REPAIR_REASKS": ("DD", "[C-9]: an invalid repair answer gets one re-ask, then hold_still"),
    "HOLD_RECHECK_STEPS": ("DD", "[C-17]: the stays checks re-run every 20 steps (1 s at 20 Hz) while holding "
                                 "still"),
    "LEDGER_LIMIT": ("DD", "v1 LEDGER_LIMIT, agent_speedup_v1.py:35, restated: compact outcome ledger length"),
    "OUTCOME_HISTORY_LIMIT": ("DD", "v1 OUTCOME_HISTORY_LIMIT, agent_speedup_v1.py:36, restated: outcome records "
                                    "kept by the policy"),
}

B1_MAX_CALLS_PER_ACT = 2
B1_STORM_CALLS_PER_ACT = 1
REJECTION_STORM_LIMIT = 3
MAX_TRANSITIONS_PER_ACT = 24
B3_MAX_REVIEWS = 2
PLAN_REASKS = 1
REPAIR_REASKS = 1
HOLD_RECHECK_STEPS = 20
LEDGER_LIMIT = 12
OUTCOME_HISTORY_LIMIT = 50

OVERRIDE_KEYS = ("first_sense", "depth_rule", "forced_yaw_rad")       # §C13.1: test and smoke branches only
RELATION_OF = {"hold": "held", "turn": "turned", "place_on": "supported_on", "release": "released",
               "slide": "slide", "place_in": "place_in", "push_to": "push_to"}
PHASE_MACROS = {"start": None, "calibrate": "calibrated", "plan": "plan", "register": "registered",
                "repair": "repair", "review": "review", "replan": "replan", "hold": "hold"}
REGION_ROLES = ("carried", "ref", "effect", "keepout", "hold_true")
FINISHED_STATUSES = ("done", "stays", "skipped")
STAYS_HOLDING_STATUSES = ("done", "stays")
# BC-R4 (Appendix L 14): without a failed step (a review, or a repair raised by a lost already-true claim) these
# actions must name the step they act on.
STEP_NAMING_ACTIONS = ("retry", "skip_step", "use_part", "repoint")
HOLD_PRIMITIVE_NAME = "hold_recheck"
SCHEMA_REASK_TEXT = "the previous answer did not match the required answer schema"
SCHEMA_ISSUE_CODE = "schema_error"
CALIBRATE_CLASS_NAME = "CalibratePlus"
SINK_KEYS = ("event", "policy_step")
_END = object()                            # sentinel: the episode generator returned


def _layer_or_default(layer):
    chosen = layer if layer is not None else relations
    if chosen is None:
        raise RuntimeError("the relation layer agent_speedup_v2_relations (Wave 2, BR) is not available")
    return chosen


def _calibrate_class(layer):
    """The CalibratePlus primitive class as the relation layer exposes it (None if it does not)."""
    found = getattr(layer, "CalibratePlus", None)
    if found is None:
        found = getattr(getattr(layer, "primitives", None), "CalibratePlus", None)
    return found if isinstance(found, type) else None


def _is_calibration(primitive, cls) -> bool:
    """diagnostics["calibration_probe"] (with the StepRunner's calibrate phase, §C18.2): isinstance(primitive,
    CalibratePlus); when the layer does not expose the class, the class name in the primitive's type hierarchy
    decides."""
    if primitive is None:
        return False
    if cls is not None:
        return isinstance(primitive, cls)
    return any(base.__name__ == CALIBRATE_CLASS_NAME for base in type(primitive).__mro__)


def _is_schema_error(exc: BaseException) -> bool:
    """A structural defect of model output: the client's SchemaError (a ValueError) or a PlanParseError."""
    return isinstance(exc, PlanParseError) or any(cls.__name__ == "SchemaError" for cls in type(exc).__mro__)


def _is_factory(item) -> bool:
    return callable(item) and not isinstance(item, (PlanNeed, PointNeed, RepairNeed))


def _failure_reason(code, default: str) -> str:
    """``code`` when it is a REASONS code a failed outcome may carry (no success reason), else ``default``."""
    if code in kernel.REASONS and (code in kernel.EITHER_STATUS_REASONS or not any(
            code in codes for codes in kernel.SUCCESS_REASONS.values())):
        return code
    return default


def _thresholds_of(items) -> list:
    """Threshold records back from their dicts (a delivered preemption report carries dicts); malformed entries
    are left out."""
    out = []
    for item in items or ():
        try:
            item = item.to_dict() if callable(getattr(item, "to_dict", None)) else item
            out.append(th_value(str(item["name"]), float(item["value"]), str(item["label"]),
                                str(item["derivation"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _issue_dicts(issues) -> list[dict]:
    out = []
    for issue in issues:
        out.append(issue.to_dict() if isinstance(issue, PlanIssue) else dict(issue))
    return out


# ------------------------------------------------------------------------------------------------- trace
class EventSink:
    """The policy's TraceSink (§C4.8): every ``emit`` becomes one JSON-safe row {"event", "policy_step", ...payload}
    that the next act drains into ``diagnostics["v2_events"]`` (a payload key that collides with those two is kept
    as ``payload_<key>``). Only EVENTS names are accepted: an unknown name raises."""

    def __init__(self, step_of: Callable[[], int]):
        self._step_of = step_of
        self._rows: list[dict] = []
        self.counts: dict[str, int] = {}

    def emit(self, event: str, /, **payload) -> None:
        self._append(self._step_of(), event, payload)

    def emit_at(self, step: int, event: str, /, **payload) -> None:
        """``emit`` stamped with an explicit policy step (observe_transition: the step of the executed action)."""
        self._append(step, event, payload)

    def _append(self, policy_step: int, event: str, payload: dict) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown trace event {event!r}")
        row = {"event": event, "policy_step": int(policy_step)}
        for key, value in jsonable(payload).items():
            row[f"payload_{key}" if key in SINK_KEYS else key] = value
        self._rows.append(row)
        self.counts[event] = self.counts.get(event, 0) + 1

    def drain(self) -> list[dict]:
        rows, self._rows = self._rows, []
        return rows


# ---------------------------------------------------------------------------------------------- fulfiller
class NeedFulfiller:
    """One client call per Need (§C13): plan / replan -> ``request_plan`` + ``parse_plan``; point -> ``request_point``;
    repair / review -> ``request_repair`` + ``parse_repair``. A schema or parse error becomes a ``schema_error`` /
    ``parse_error`` answer and one ``ctx.note_rejection``; API errors propagate (the runner maps them). Every call
    appends a CallRecord to ``calls``. ``guard`` (set by the policy) returns True while a primitive is active: a call
    then raises, which enforces "no model call inside a primitive" (§C3.2)."""

    def __init__(self, client, events, *, clock=time.monotonic, ctx=None):
        self.client = client
        self.events = events
        self.clock = clock
        self.ctx = ctx
        self.guard: Callable[[], bool] | None = None
        self.counts = {kind: 0 for kind in NEED_KINDS}
        self.errors = {"schema_error": 0, "parse_error": 0, "api_error": 0}

    def fulfil(self, need, view: StepView, calls: list):
        if self.guard is not None and self.guard():
            raise RuntimeError("no model call inside a primitive (§C3.2): a primitive is still active")
        if isinstance(need, PlanNeed):
            return self._plan(need, view, calls)
        if isinstance(need, PointNeed):
            return self._point(need, view, calls)
        if isinstance(need, RepairNeed):
            return self._repair(need, view, calls)
        raise TypeError(f"not a Need: {type(need).__name__}")

    # -------------------------------------------------------------------------------------- internals
    def _reject(self, kind: str) -> None:
        note = getattr(self.ctx, "note_rejection", None)
        if callable(note):
            note(kind)

    def _record(self, kind, started, calls, ok, error) -> CallRecord:
        log = getattr(self.client, "last_log_path", None)
        call = CallRecord(kind, max(0.0, float(self.clock() - started)), getattr(self.client, "last_request_key", None),
                          None if log is None else str(log), bool(ok), error)
        calls.append(call)
        return call

    def _call(self, kind, method, calls, *args, **kwargs):
        """(result or None, error or None, started); a schema error is caught, anything else propagates."""
        started = self.clock()
        self.counts[kind] += 1
        try:
            return method(*args, **kwargs), None, started
        except ValueError as exc:
            if not _is_schema_error(exc):
                self._record(kind, started, calls, False, f"{type(exc).__name__}")
                raise
            self.errors["schema_error"] += 1
            return None, SCHEMA_ISSUE_CODE, started
        except Exception as exc:
            self.errors["api_error"] += 1
            self._record(kind, started, calls, False, f"{type(exc).__name__}")
            raise

    def _plan(self, need: PlanNeed, view: StepView, calls) -> PlanAnswer:
        raw, error, started = self._call(need.kind, self.client.request_plan, calls, view.external_rgb_upright,
                                         view.wrist_rgb_upright, quarter_turns=need.quarter_turns,
                                         fingertip_edge=need.fingertip_edge, holding=need.holding,
                                         status_block=need.status_block, reask_error=need.reask_error,
                                         replan=need.kind == "replan")
        plan, parse_error = None, None
        if raw is not None:
            try:
                plan = parse_plan(raw, source=need.kind)
            except PlanParseError as exc:
                parse_error = str(exc)[:ISSUE_TEXT_MAX_CHARS] or "parse_error"
                self.errors["parse_error"] += 1
        if error is not None or parse_error is not None:
            self._reject(need.kind)
        call = self._record(need.kind, started, calls, error is None and parse_error is None, error or parse_error)
        return PlanAnswer(need, None if raw is None else dict(raw), plan, parse_error, error is not None, call, view)

    def _point(self, need: PointNeed, view: StepView, calls) -> PointAnswer:
        rgb = view.wrist_rgb_upright if need.camera == "wrist" else view.external_rgb_upright
        reply, error, started = self._call(need.kind, self.client.request_point, calls, rgb, stage=need.kind,
                                           camera=need.camera, text=need.text, pid=need.pid,
                                           center_uv=need.center_uv, quarter_turns=need.quarter_turns,
                                           previous_rejected=need.previous_rejected, marker_uv=need.marker_uv)
        if error is not None:
            self._reject(need.kind)
        call = self._record(need.kind, started, calls, error is None, error)
        return PointAnswer(need, reply, error is not None, call)

    def _repair(self, need: RepairNeed, view: StepView, calls) -> RepairReply:
        raw, error, started = self._call(need.kind, self.client.request_repair, calls, view.external_rgb_upright,
                                         view.wrist_rgb_upright, quarter_turns=need.quarter_turns,
                                         status_block=need.status_block, last_outcome=need.last_outcome,
                                         holding=need.holding, review=need.kind == "review")
        answer = None
        if raw is not None:
            try:
                answer = parse_repair(raw)
            except PlanParseError:
                error = "parse_error"
                self.errors["parse_error"] += 1
        if error is not None:
            self._reject(need.kind)
        call = self._record(need.kind, started, calls, error is None, error)
        return RepairReply(need, None if raw is None else dict(raw), answer, error is not None, call)


# ---------------------------------------------------------------------------------------------- StepRunner
class StepRunner:
    """The episode generator (§C3.6). ``run()`` never returns. ``relations`` = the relation layer (default: the
    module); ``call_counts`` = the fulfiller's calls by kind (per-step call accounting in step_end events).

    State: ``plan``, ``states`` (StepState, model order), ``reviews_used``, ``repairs_used``, ``ledger`` (one compact
    line per finished step), ``halted`` (a repair answered hold_still), ``phase`` ("start", "calibrate", "plan",
    "register", "step", "repair", "review", "replan", "hold"), ``active_state`` (the StepState being run or
    repaired), ``claims`` (unverified model claims)."""

    def __init__(self, ctx, *, relations=None, call_counts: Mapping | None = None):
        self.ctx = ctx
        self.relations = _layer_or_default(relations)
        self.call_counts = call_counts if call_counts is not None else {}
        self.plan: Plan | None = None
        self.states: list[StepState] = []
        self.roles: dict = {}
        self.reviews_used = 0
        self.repairs_used = 0
        self.ledger: list[dict] = []
        self.phase = "start"
        self.active_state: StepState | None = None
        self.calibration = None
        self.claims: list[dict] = []
        self.plan_view: StepView | None = None
        self.plan_turns = 0
        self.last_stays: dict[str, object] = {}
        self._grip: dict[int, tuple] = {}            # id(StepState) -> grip override (use_part)
        self._resume: dict[int, dict] = {}           # id(StepState) -> resume of the failed outcome (retry)
        self._owner: dict[str, StepState | None] = {}    # stays key -> the StepState it keeps (None: detached)
        self._completed_at: dict[int, int] = {}      # id(StepState) -> policy step of its last completion
        self._hold_true: dict[str, dict] = {}        # stays key -> {"claim", "escalated"}
        self._unchecked: dict[str, dict] = {}        # stays key -> {"claim", "reason"}: no check could be built yet
        self._hold_true_keepouts: set[str] = set()   # parts b of already-true claims that get a hold_true keep-out
        self._awaiting: set[str] = set()             # parts whose registration waits for an empty hand (repoint)
        self._stale_keepouts: set[str] = set()       # parts whose keep-outs are rebuilt once re-registered
        self._claim_baseline = None                  # external baseline of the claims (start, else the plan view)
        self._review_armed = True
        self._settled = False                        # a hold cycle re-checked the stays since the last change
        self._tags: dict[int, StepState | None] = {}  # id(recorded outcome) -> the StepState it belongs to
        self.halted = False                          # a repair answered hold_still: no step that never ran starts

    # ----------------------------------------------------------------------------------------- the episode
    def run(self):
        ctx = self.ctx
        self._enter("calibrate", "calibrated")
        self.calibration = yield from self.relations.calibrate_plus(ctx)
        self.controller_returned()
        if getattr(ctx, "cal", None) is None and self.calibration is not None:
            ctx.cal = self.calibration                   # §C10: calibrate_plus returns the V2Calibration
        self.tag_pending(None)
        plan, answer = yield from self.obtain_plan("plan")
        if plan is None:
            self._record_invalid_plan(answer)
            while True:                                          # nothing to check: hold forever (§C3.6 step 2)
                self._enter("hold", None)
                yield from self.hold_once()
        yield from self.adopt(plan, answer, replan=False)
        while True:
            state = self.next_pending()
            if state is not None:
                yield from self.run_step(state)
            elif self.review_due():
                yield from self.review()
            else:
                yield from self.hold_cycle()

    # -------------------------------------------------------------------------------------------- helpers
    def _enter(self, phase: str, relation_name):
        self.phase = phase
        if hasattr(self.ctx, "active_relation"):
            self.ctx.active_relation = relation_name

    def index_of(self, state: StepState | None) -> int | None:
        if state is None:
            return None
        for i, item in enumerate(self.states):
            if item is state:
                return i
        return None

    @property
    def active_i(self) -> int | None:
        return self.index_of(self.active_state)

    def next_pending(self) -> StepState | None:
        """The first pending step in model order. After a repair's hold_still only a step that already ran (a
        reopened one) is taken: the plan does not advance past the held failure."""
        for state in self.states:
            if state.status == "pending" and not (self.halted and state.attempts == 0):
                return state
        return None

    def holding_pid(self) -> str | None:
        return getattr(getattr(self.ctx, "holding", None), "pid", None)

    def turns(self, view: StepView) -> int:
        """§3.6 item 2: k quarter turns for the shown wrist image, computed at this view (never assumed)."""
        cal = getattr(self.ctx, "cal", None)
        if cal is None or view is None:
            return 0
        return int(geometry.frame_alignment_turns(view.R, getattr(cal, "external", None)))

    def grip_of(self, state: StepState) -> tuple:
        return tuple(self._grip.get(id(state), state.step.grip_parts))

    def tag_pending(self, state: StepState | None) -> None:
        """Bind the outcomes recorded since the last boundary (still in ctx.pending_outcomes) to ``state``, so the
        policy's drain numbers them with the plan step they belong to, not the step active when it drains."""
        for item in list(getattr(self.ctx, "pending_outcomes", None) or []):
            self._tags.setdefault(id(item), state)

    def plan_state_of(self, outcome) -> StepState | None:
        """The StepState an outcome belongs to (its tag; untagged: the active state). Consumes the tag."""
        return self._tags.pop(id(outcome), self.active_state)

    def _renumber(self) -> None:
        for i, state in enumerate(self.states):
            state.i = i

    def _calls_now(self) -> dict:
        return {kind: int(n) for kind, n in self.call_counts.items()}

    def _emit(self, event: str, **payload) -> None:
        self.ctx.events.emit(event, **payload)

    def _pids(self, step, grip=None) -> dict:
        return {"obj": step.obj, "grip": list(grip if grip is not None else step.grip_parts), "ref": step.ref}

    def _evidence_text(self, outcome: RelationOutcome) -> str:
        build = getattr(self.relations, "evidence_text", None)
        if callable(build):
            try:
                text = build(outcome)
            except (KeyError, ValueError, TypeError):
                text = None
            if isinstance(text, str):
                return text[:kernel.EVIDENCE_TEXT_MAX_CHARS]
        return outcome.reason[:kernel.EVIDENCE_TEXT_MAX_CHARS]

    def _outcome(self, relation_name, status, reason, pids, evidence, thresholds=(), *, state=None) -> RelationOutcome:
        """A StepRunner-owned outcome (invalid_plan, relation_not_implemented, kept_relation_lost, repair_invalid),
        recorded once through ctx.record [C-39] and bound to ``state`` (its plan step, or None)."""
        outcome = RelationOutcome(relation_name, status, reason, dict(pids), jsonable(evidence), list(thresholds),
                                  "", 0, 0)
        outcome.evidence_text = self._evidence_text(outcome)
        self.tag_pending(self.active_state)
        self.ctx.record(outcome)
        self._tags[id(outcome)] = state
        return outcome

    def _accepted(self, kind: str) -> None:
        note = getattr(self.ctx, "note_accepted", None)
        if callable(note):
            note(kind)

    def _rejected(self, kind: str) -> None:
        note = getattr(self.ctx, "note_rejection", None)
        if callable(note):
            note(kind)

    def controller_returned(self) -> None:
        """§C10.1: the bus scope "relation" is cleared on every controller return (calibrate_plus, register_regions,
        an implicit release and the step's own controller)."""
        self.ctx.monitor.clear("relation")

    def status_block(self) -> dict:
        """The plan status block (``_plan.plan_status_block``) with the grip list in force: a use_part override
        replaces the plan's grip of its step."""
        block = plan_status_block(self.plan, self.states, holding=self.holding_pid())
        if self.states:
            for row, state in zip(block["steps"], self.states):
                if id(state) in self._grip:
                    row["grip"] = list(self._grip[id(state)])
        return block

    def hold_factory(self):
        gripper = "close" if getattr(self.ctx, "holding", None) is not None else "open"

        def factory(view):
            return HoldPrimitive(view, steps=HOLD_RECHECK_STEPS, gripper=gripper, name=HOLD_PRIMITIVE_NAME)
        return factory

    def hold_once(self):
        """One hold primitive; a preemption delivered to it is absorbed (``absorb_preemption``). Returns True when
        the hold ran to its end."""
        reply = yield self.hold_factory()
        preempted = yield from self.absorb_preemption(reply)
        return not preempted

    def hold_cycle(self):
        """One hold primitive (HOLD_RECHECK_STEPS), then the stays checks (§C3.6 step 7)."""
        self._enter("hold", None)
        completed = yield from self.hold_once()
        yield from self.stays_boundary_check()
        if completed:
            self._settled = True

    def absorb_preemption(self, reply):
        """A bus preemption delivered to the idle hold (§C3.3; e.g. hold_lost while holding still after a repair):
        the invariant is removed from the bus (the hold must act while it stays violated, InvariantMonitor
        docstring), the loss is recorded once ("keeps", the report's reason) and, with a plan, escalated by one
        repair. Without this every hold primitive would be preempted after one action, with nothing recorded.
        Returns True when ``reply`` was a preemption."""
        result = reply[0] if isinstance(reply, tuple) and reply and isinstance(reply[0], Mapping) else None
        reason = str((result or {}).get("reason") or "")
        if not reason.startswith(PREEMPTED_PREFIX):
            return False
        ctx = self.ctx
        evidence = dict(result.get("evidence") or {})
        key = evidence.get("invariant")
        report = dict(evidence.get("report") or {})
        if isinstance(key, str):
            ctx.monitor.remove(key)
        code = _failure_reason(reason[len(PREEMPTED_PREFIX):], "kept_relation_lost")
        held = getattr(ctx, "holding", None)
        pids = {} if held is None else {"obj": getattr(held, "pid", None),
                                        "grip": [p for p in (getattr(held, "grip_pid", None),) if p is not None],
                                        "ref": None}
        lost = self._outcome("keeps", "failed", code, pids,
                             {"invariant": key, "report": report, "removed_from_bus": isinstance(key, str),
                              "phase": self.phase}, _thresholds_of(report.get("thresholds")), state=None)
        self._settled = False
        if self.plan is not None:
            yield from self.escalate(None, lost)
        return True

    # ----------------------------------------------------------------------------------------------- plan
    def obtain_plan(self, kind: str):
        """PlanNeed -> validate_plan; one re-ask carrying the issue text (spec §3.4). Returns (plan, answer) or
        (None, last answer) after the second invalid answer."""
        ctx = self.ctx
        self._enter("plan" if kind == "plan" else "replan", None)
        reask = None
        answer = None
        for attempt in range(1 + PLAN_REASKS):
            view = ctx.view
            k = self.turns(view)
            holding = self.holding_pid()
            block = self.status_block() if kind == "replan" and self.plan is not None else None
            need = PlanNeed(kind, k, FINGERTIP_EDGES[k], status_block=block, reask_error=reask, holding=holding)
            answer = yield need
            if not isinstance(answer, PlanAnswer):
                raise TypeError("a PlanNeed is answered with a PlanAnswer")
            final = attempt == PLAN_REASKS
            if answer.plan is not None:
                issues = validate_plan(answer.plan)
                event = plan_validation_event(issues, reask=attempt > 0, final=final or not issues)
            else:
                issues = None
                code = SCHEMA_ISSUE_CODE if answer.schema_error else "parse_error"
                event = {"ok": False, "issues": [{"code": code, "step": None, "pid": None,
                                                  "text": answer.parse_error or SCHEMA_REASK_TEXT}],
                         "reask": attempt > 0, "final": final}
            event["kind"] = kind
            self._emit("plan_validation", **event)
            if answer.plan is not None and not issues:
                self._accepted(kind)
                self._emit("relation_plan", **relation_plan_event(answer.plan))
                return answer.plan, answer
            if answer.plan is not None:
                self._rejected(kind)              # schema / parse errors were noted by the fulfiller
            # the error text of the re-ask: the validation issues, else the parse error, else the schema text
            reask = (issues_text(issues) if issues else (answer.parse_error or SCHEMA_REASK_TEXT))[
                :ISSUES_TEXT_MAX_CHARS]
        return None, answer

    def _record_invalid_plan(self, answer) -> RelationOutcome:
        issues = []
        if answer is not None and answer.plan is not None:
            issues = _issue_dicts(validate_plan(answer.plan))
        threshold = th_value("PLAN_reasks", PLAN_REASKS, "DD", "spec §3.4: one re-ask with the error text, then "
                                                               "invalid_plan; PLAN_REASKS")
        evidence = {"issues": issues, "schema_error": bool(getattr(answer, "schema_error", False)),
                    "parse_error": getattr(answer, "parse_error", None)}
        return self._outcome("plan", "rejected", "invalid_plan", {}, evidence, (threshold,), state=None)

    def adopt(self, plan: Plan, answer: PlanAnswer, *, replan: bool):
        """§C3.6 step 3: roles, merge_plan with THAT request's view and k, region registration (empty-hand region
        parts and provisional grip parts, by the relation layer), the step states (§C3.6 step 4), the already-true
        claims (stays entries against the start baseline + keep-outs of b), then the keep-outs of clear_of parts."""
        ctx = self.ctx
        old_states = list(self.states)
        self.plan = plan
        self.plan_view, self.plan_turns = answer.view, int(answer.need.quarter_turns)
        self.roles = part_roles(plan)
        ctx.plan, ctx.roles = plan, self.roles
        ctx.scene.merge_plan(plan, self.roles, answer.view, answer.need.quarter_turns)
        self._enter("register", "registered")
        if replan:                                  # parts persist by (id, text): register only what is missing
            yield from self.register_missing()
        else:
            yield from self.relations.register_regions(ctx, plan, self.roles)
            self.controller_returned()
            self.tag_pending(None)
        states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
        if replan:
            self._carry_done_steps(old_states, states)
        self.states = states
        self._grip, self._resume = {}, {}
        self.add_hold_true(plan, replan=replan)
        self.ensure_keepouts()
        self._review_armed = True
        self._settled = False

    def needs_registration(self, pid: str) -> bool:
        """A part with a role that is neither registered nor provisional (new, dropped or failed), or a provisional
        (grip-only) record that now has a region role (it then needs an empty-hand region registration)."""
        part = self.ctx.scene.parts.get(pid)
        roles = set(self.roles.get(pid, ()))
        if part is None or not roles or part.status == "registered":
            return False
        if getattr(part, "provisional", False):
            return bool(roles & set(REGION_ROLES))
        return True

    def register_missing(self, only=None):
        """Registration (relation layer) of the parts that need it (all, or those in ``only``), then the keep-outs.
        Registration needs an empty hand: with a full hand the parts wait in ``_awaiting`` and are registered at the
        next empty-hand boundary (the start of a step, after its implicit release; ``preconditions``). Returns the
        parts whose registration waits."""
        ctx = self.ctx
        fresh = [pid for pid in self.roles if self.needs_registration(pid) and (only is None or pid in only)]
        if fresh and getattr(ctx, "holding", None) is None:
            self._enter("register", "registered")
            yield from self.relations.register_regions(ctx, self.plan, self._roles_only(fresh))
            self.controller_returned()
            self.tag_pending(self.active_state)
            self._awaiting -= set(fresh)
        elif fresh:
            self._awaiting |= set(fresh)
        self._awaiting = {pid for pid in self._awaiting if self.needs_registration(pid)}
        self.ensure_keepouts()
        return sorted(self._awaiting & set(fresh))

    def _roles_only(self, pids) -> dict:
        """The plan's roles with every part outside ``pids`` emptied (registration of those parts only)."""
        keep = set(pids)
        return {pid: (roles if pid in keep else frozenset()) for pid, roles in self.roles.items()}

    def _carry_done_steps(self, old_states, new_states) -> None:
        """Replan (§C3.8): a step of the new plan that was already done keeps status stays (and its stays
        entries); stays entries of done steps the new plan drops become detached (a loss then escalates)."""
        done = [s for s in old_states if s.status in STAYS_HOLDING_STATUSES]
        used = set()
        for new in new_states:
            for old in done:
                if id(old) in used:
                    continue
                same = old.step == new.step or (old.step.do, old.step.obj, old.step.ref) == (new.step.do,
                                                                                             new.step.obj,
                                                                                             new.step.ref)
                if same:
                    used.add(id(old))
                    new.status, new.reason = "stays", old.reason
                    for key, owner in self._owner.items():
                        if owner is old:
                            self._owner[key] = new
                    if id(old) in self._completed_at:
                        self._completed_at[id(new)] = self._completed_at[id(old)]
                    break
        kept = {id(s) for s in new_states}
        for key, owner in list(self._owner.items()):
            if owner is not None and id(owner) not in kept:
                self._owner[key] = None

    def ensure_keepouts(self) -> None:
        """§C3.6 expansion: every keep.clear_of part has a KeepOut once it is registered, and so has the part b of
        every already-true claim that adds one (planner.keepout_for). An existing keep-out of the same part and
        kind is kept, unless the part was repointed: its keep-out is rebuilt from the new registration."""
        scene = self.ctx.scene
        for pid, roles in self.roles.items():
            if "keepout" in roles:
                self._add_keepout(pid, "clear_of", scene)
        for pid in sorted(self._hold_true_keepouts):
            self._add_keepout(pid, "hold_true", scene)
        self._stale_keepouts = {pid for pid in self._stale_keepouts
                                if getattr(scene.parts.get(pid), "status", None) != "registered"}

    def _add_keepout(self, pid: str, kind: str, scene) -> bool:
        part = scene.parts.get(pid)
        registered = part is not None and part.status == "registered"
        if f"{kind}:{pid}" in scene.keepouts and not (registered and pid in self._stale_keepouts):
            return True
        if not registered:
            return False
        try:
            keepout = planner.keepout_for(part, kind, sigma_loc_m=planner.horizontal_sigma_m(part),
                                          sigma_h_m=float(part.sigma_h_m or 0.0))
        except ValueError:
            return False
        scene.add_keepout(keepout)                  # replaces an older keep-out of the same part and kind
        return True

    @staticmethod
    def claim_key(claim) -> str:
        return f"hold_true:{claim.rel}:{claim.a}:{claim.b or '-'}"

    def unfinished_step_parts(self) -> set:
        """obj, ref and grip parts of every step not finished (pending, active or failed)."""
        out = set()
        for state in self.states:
            if state.status not in FINISHED_STATUSES:
                out |= {state.step.obj, *self.grip_of(state)} | ({state.step.ref} - {None})
        return out

    def add_hold_true(self, plan: Plan, *, replan: bool = False) -> None:
        """Spec §3.3 already-true claims: each is an unverified model claim (event model_claim), a stays entry
        against the start baseline (RegionStaysCheck, "unchanged") and adds clear_of(b): a hold_true keep-out of b,
        none when b is null, and none when b is the obj, ref or grip of a step still to run (the keep-out would
        fence off that step's own part; the skip is in the model_claim event). A claim whose check cannot be built
        yet stays unchecked: the check is re-tried at every relation boundary and the claim blocks reviews until it
        exists. On a replan, the claims the new plan no longer makes are no longer monitored."""
        ctx = self.ctx
        claims = {self.claim_key(claim): claim for claim in plan.hold_true}
        if replan:
            for key in [k for k in list(self._hold_true) + list(self._unchecked) if k not in claims]:
                ctx.stays.remove(key)
                self._hold_true.pop(key, None)
                self._unchecked.pop(key, None)
            self._hold_true_keepouts = set()
        if self._claim_baseline is None and ctx.view is not None:
            start = (ctx.scene.baselines.get("start") or {}).get("external")
            self._claim_baseline = np.asarray(start if start is not None else ctx.view.external_rgb_upright)
        fenced = self.unfinished_step_parts()
        for key, claim in claims.items():
            if claim.b is not None and claim.b not in fenced:
                self._hold_true_keepouts.add(claim.b)
            if key in self._hold_true or key in self._unchecked:
                continue
            check, reason = self.hold_true_check(claim)
            if check is not None:
                self._arm_hold_true(key, claim, check)
            else:
                self._unchecked[key] = {"claim": claim, "reason": reason}
            skipped = None
            if claim.b is None:
                skipped = "b_is_null"
            elif claim.b in fenced:
                skipped = "part_of_a_step_to_run"
            kept_out = skipped is None and self._add_keepout(claim.b, "hold_true", ctx.scene)
            detail = {"a": claim.a, "b": claim.b, "key": key, "checked": check is not None, "reason": reason,
                      "keepout": kept_out, "keepout_skipped": skipped}
            self.claims.append({"claim": f"hold_true:{claim.rel}", **detail})
            self._emit("model_claim", claim=f"hold_true:{claim.rel}", detail=detail, unverified=True)

    def _arm_hold_true(self, key: str, claim, check) -> None:
        self.ctx.stays.add(key, check, step_i=None, kind="hold_true")
        self._hold_true[key] = {"claim": claim, "escalated": False}
        self._unchecked.pop(key, None)

    def retry_unchecked_claims(self) -> None:
        """The checks of already-true claims that could not be built before, re-tried at a relation boundary."""
        for key, entry in list(self._unchecked.items()):
            check, reason = self.hold_true_check(entry["claim"])
            if check is not None:
                self._arm_hold_true(key, entry["claim"], check)
            else:
                entry["reason"] = reason

    def hold_true_check(self, claim):
        """(check, None) or (None, why): the region of part a in the external camera, unchanged against the start
        baseline (the hand masked by its projected footprint at the current pose) [C-48]."""
        ctx = self.ctx
        scene, cal, view = ctx.scene, getattr(ctx, "cal", None), ctx.view
        if cal is None:
            return None, "calibration_unavailable"
        try:
            region = scene.region_mask(claim.a, "external", view)
        except (ValueError, TypeError, KeyError, AttributeError):
            region = None
        if region is None or not np.asarray(region, dtype=bool).any():
            return None, "region_unavailable"
        region = np.asarray(region, dtype=bool)
        baseline = (scene.baselines.get("start") or {}).get("external")
        if baseline is None:                   # a check built later still compares with the adoption-time image
            baseline = self._claim_baseline if self._claim_baseline is not None else np.asarray(
                view.external_rgb_upright)
        sigma = float((getattr(cal, "sigma_px", None) or {}).get("external", 0.0) or 0.0)
        false_map = (getattr(cal, "false_change", None) or {}).get("external")
        false_rate = 0.0
        if false_map is not None and np.shape(false_map) == region.shape:
            false_rate = float(np.mean(np.asarray(false_map, dtype=float)[region]))

        def hand_mask(current):
            return scene.hand_swept_mask("external", current, (0.0, 0.0))
        try:
            check = RegionStaysCheck(region=region, baseline_rgb=baseline, projector_mask=hand_mask, sigma_px=sigma,
                                     false_rate=false_rate, kind="unchanged")
        except (ValueError, TypeError) as exc:
            return None, type(exc).__name__
        return check, None

    # ----------------------------------------------------------------------------------------------- steps
    def _needs_release(self, step, grip) -> bool:
        """§C3.6 expansion: turn needs a hand holding nothing but a grip part of the step; place_on nothing, or a
        grip part of the step; hold an empty hand. The release itself needs nothing."""
        held = getattr(self.ctx, "holding", None)
        if held is None or step.do == "release":
            return False
        if step.do == "hold":
            return True
        if step.do in ("turn", "place_on"):
            return getattr(held, "grip_pid", None) not in grip
        return False

    def run_step(self, state: StepState):
        ctx = self.ctx
        step = state.step
        state.status = "active"
        state.attempts += 1
        self.tag_pending(self.active_state)
        self.active_state = state
        grip = self.grip_of(state)
        i = self.index_of(state)
        context = StepContext.from_step(i, step, attempt=state.attempts, resume=self._resume.pop(id(state), None),
                                        grip=grip)
        self._enter("step", RELATION_OF[step.do])
        start_step, calls_before = int(ctx.step), self._calls_now()
        keys_before = set(ctx.stays.keys())
        self._emit("step_start", i=i, do=step.do, obj=step.obj, grip=list(grip), ref=step.ref, attempt=state.attempts)
        outcome = None
        if step.do in RESERVED_DO_KINDS:
            outcome = self._outcome(step.do, "failed", "relation_not_implemented", self._pids(step, grip),
                                    {"do": step.do, "reserved": True}, state=state)
        elif self._needs_release(step, grip):
            released = yield from self.relations.release(ctx, None)
            self.controller_returned()
            self._enter("step", RELATION_OF[step.do])
            if not isinstance(released, RelationOutcome):
                raise TypeError("release returns a RelationOutcome")
            if released.status != "succeeded":
                outcome = released
        if outcome is None:
            outcome = yield from self.preconditions(state, grip)
            self._enter("step", RELATION_OF[step.do])
        if outcome is None:
            controller = self.relations.CONTROLLERS[step.do]
            outcome = yield from controller(ctx, context)
        if not isinstance(outcome, RelationOutcome):
            raise TypeError(f"the {step.do} controller returns a RelationOutcome")
        self.tag_pending(state)
        self.controller_returned()
        steps_used = int(ctx.step) - start_step
        calls_after = self._calls_now()
        calls = {kind: calls_after.get(kind, 0) - calls_before.get(kind, 0) for kind in calls_after
                 if calls_after.get(kind, 0) != calls_before.get(kind, 0)}
        state.steps += steps_used
        for kind, n in calls.items():
            state.calls[kind] = state.calls.get(kind, 0) + n
        state.reason = outcome.reason
        state.evidence = {"relation": outcome.relation, "status": outcome.status,
                          "evidence_text": outcome.evidence_text, "local_retries": outcome.local_retries}
        succeeded = outcome.status == "succeeded"
        state.status = "done" if succeeded else "failed"
        for key in set(ctx.stays.keys()) - keys_before:
            self._owner[key] = state
        self._settled = False
        if succeeded:
            self._completed_at[id(state)] = int(ctx.step)
            self._review_armed = True
        line = {"i": self.index_of(state), "do": step.do, "obj": step.obj, "grip": list(grip), "ref": step.ref,
                "attempt": state.attempts, "status": state.status, "reason": outcome.reason, "steps": steps_used,
                "calls": calls, "local_retries": outcome.local_retries}
        self._emit("step_end", **line)
        self.ledger = (self.ledger + [jsonable(line)])[-LEDGER_LIMIT:]
        repairs_before = self.repairs_used
        yield from self.stays_boundary_check()
        if not succeeded and self.still_unanswered(state, repairs_before):
            yield from self.escalate(state, outcome)
        self.active_state = None

    def still_unanswered(self, state: StepState, repairs_before: int) -> bool:
        """The failed step still needs its own repair request. A repair asked at the same boundary (a lost
        already-true claim, whose status block shows this failure) answers it when the answer reset the step,
        replaced the plan, or held still; only an answer that acted on another step leaves it to its own request."""
        if self.repairs_used == repairs_before:
            return True
        return state.status == "failed" and self.index_of(state) is not None and not self.halted

    def preconditions(self, state: StepState, grip):
        """§C3.6 expansion, the StepRunner's own preconditions after the implicit release: registrations waiting for
        an empty hand (``_awaiting``, a repoint while holding) and the regions this step needs are made now when the
        hand is empty; then every until.effect part must be registered as a region and every keep.clear_of part
        must have a KeepOut. (The place_on destination precondition is place_on's own, §C10.4.) An unmet
        precondition is this step's failed outcome, recorded, and the step escalates: it is never skipped
        silently. Returns None or that outcome."""
        ctx, step = self.ctx, state.step
        if step.do == "release":
            return None
        regions = [until.p for until in step.until if until.o == "effect" and until.p is not None]
        clear = [keep.p for keep in step.keep if keep.c == "clear_of"]
        want = set(self._awaiting) | {pid for pid in regions + clear if self.needs_registration(pid)}
        if want and getattr(ctx, "holding", None) is None:
            yield from self.register_missing(only=want)
        else:
            self.ensure_keepouts()
        unmet = {pid: "region_registration" for pid in regions if self.needs_registration(pid)}
        for pid in clear:
            if f"clear_of:{pid}" not in ctx.scene.keepouts:
                unmet.setdefault(pid, "keepout")
        if not unmet:
            return None
        parts = ctx.scene.parts
        statuses = {pid: getattr(parts.get(pid), "status", None) for pid in unmet}
        failed = [pid for pid in sorted(unmet) if statuses[pid] == "failed"]
        holding = self.holding_pid()
        if failed:
            reason = _failure_reason(getattr(parts.get(failed[0]), "reason", None), "part_not_found")
        elif all(need == "keepout" and statuses[pid] == "registered" for pid, need in unmet.items()):
            reason = "geometry_unmeasured"
        else:
            reason = "precondition_unmet"
        evidence = {"unmet": unmet, "statuses": statuses, "holding": holding,
                    "awaiting_registration": sorted(self._awaiting)}
        return self._outcome(RELATION_OF[step.do], "failed", reason, self._pids(step, grip), evidence, state=state)

    # ----------------------------------------------------------------------------------------------- stays
    def stays_boundary_check(self):
        """§C3.7: every stays entry, one stays_check event each. A violated step entry records kept_relation_lost
        and reopens its step (pending; it runs after the current relation); a violated already-true claim (or an
        entry whose step left the plan) records the loss and escalates at once. Pending is unknown, never a loss. A
        step is not reopened at the policy step it completed (the loop must advance first)."""
        ctx = self.ctx
        if ctx.view is None:
            return
        self.retry_unchecked_claims()
        results = ctx.stays.check_all(ctx.view)
        self.last_stays = {key: report for key, report in results}
        for key, entry in self._unchecked.items():         # no check yet: unknown (pending), blocks the reviews
            self._emit("stays_check", key=key, verdict="pending", reason=entry["reason"], checked=False)
        escalate = []
        for key, report in results:
            self._emit("stays_check", key=key, verdict=report.verdict, reason=report.reason)
            if key in self._hold_true:
                entry = self._hold_true[key]
                if report.verdict == "met":
                    entry["escalated"] = False
                elif report.verdict == "violated" and not entry["escalated"]:
                    entry["escalated"] = True
                    claim = entry["claim"]
                    escalate.append(self._outcome("keeps", "failed", "kept_relation_lost",
                                                  {"obj": claim.a, "grip": [], "ref": claim.b},
                                                  {"key": key, "report": report.to_dict()}, report.thresholds,
                                                  state=None))
                continue
            if report.verdict != "violated":
                continue
            owner = self._owner.get(key, self._entry_state(key))
            if owner is not None and self._completed_at.get(id(owner)) == int(ctx.step):
                continue
            ctx.stays.remove(key)
            self._owner.pop(key, None)
            step = owner.step if owner is not None else None
            relation_name = RELATION_OF[step.do] if step is not None else "keeps"
            pids = self._pids(step, self.grip_of(owner)) if step is not None else {}
            lost = self._outcome(relation_name, "failed", "kept_relation_lost", pids,
                                 {"key": key, "report": report.to_dict(), "step_i": self.index_of(owner)},
                                 report.thresholds, state=owner)
            if owner is not None and self.index_of(owner) is not None and owner.status in STAYS_HOLDING_STATUSES:
                owner.status, owner.reason = "pending", "kept_relation_lost"
            elif owner is None:
                escalate.append(lost)
        for outcome in escalate:
            yield from self.escalate(None, outcome)

    def _entry_state(self, key: str) -> StepState | None:
        try:
            step_i = self.ctx.stays.entry(key).get("step_i")
        except KeyError:
            return None
        if step_i is None or not 0 <= int(step_i) < len(self.states):
            return None
        return self.states[int(step_i)]

    def _stays_hold_for(self, state: StepState) -> bool:
        """skip_step: the step's own stays entries hold at the last check (none -> False)."""
        keys = [key for key, owner in self._owner.items() if owner is state]
        verdicts = [getattr(self.last_stays.get(key), "verdict", None) for key in keys]
        return bool(verdicts) and all(v == "met" for v in verdicts)

    def stays_hold_now(self, state: StepState) -> bool:
        """skip_step (spec §3.7): the step becomes a stays relation if its observer currently holds. A step that
        owns stays entries is judged by them (``_stays_hold_for``); a failed step owns none (the relation layer
        registers stays entries on success only), so the layer's optional hook ``stays_check_for(ctx, step)`` ->
        (key, check) | None builds the relation's stays check, which is run on the current view; met -> the entry
        is registered for the step (checked at every boundary from now on). Without the hook: skipped."""
        if self._stays_hold_for(state):
            return True
        build = getattr(self.relations, "stays_check_for", None)
        i = self.index_of(state)
        if not callable(build) or i is None or self.ctx.view is None:
            return False
        made = build(self.ctx, StepContext.from_step(i, state.step, attempt=max(1, state.attempts),
                                                     grip=self.grip_of(state)))
        if not (isinstance(made, tuple) and len(made) == 2 and isinstance(made[0], str) and callable(made[1])):
            return False
        key, check = made
        report = check(self.ctx.view)
        self._emit("stays_check", key=key, verdict=report.verdict, reason=report.reason, skip_step=True)
        if report.verdict != "met":
            return False
        self.ctx.stays.add(key, check, step_i=i, kind="step")
        self._owner[key] = state
        self._completed_at[id(state)] = int(self.ctx.step)
        self.last_stays[key] = report
        return True

    # ------------------------------------------------------------------------------------- repair / review
    def named_state(self, step) -> StepState | None:
        """BC-R4: the plan step a step-naming answer acts on (equal step first, else the same do / obj / ref)."""
        if step is None:
            return None
        for state in self.states:
            if state.step == step:
                return state
        for state in self.states:
            if (state.step.do, state.step.obj, state.step.ref) == (step.do, step.obj, step.ref):
                return state
        return None

    def repair_issues(self, reply: RepairReply, failed_i: int | None, *, review: bool) -> list:
        if reply.schema_error or reply.answer is None:
            return [{"code": SCHEMA_ISSUE_CODE, "step": failed_i, "pid": None, "text": SCHEMA_REASK_TEXT}]
        answer = reply.answer
        issues = list(validate_repair(answer, self.plan, failed_i=failed_i, review=review))
        if failed_i is None and answer.action in STEP_NAMING_ACTIONS and not any(
                isinstance(issue, PlanIssue) and issue.code == "step_required" for issue in issues):
            if answer.step is None:
                issues.append(PlanIssue("step_required", None, None,
                                        f"{answer.action} must name the plan step it acts on (no failed step)"))
            elif self.named_state(answer.step) is None:
                issues.append(PlanIssue("step_required", None, None,
                                        f"{answer.action} names a step that is not in the plan"))
        return issues

    def ask(self, kind: str, failed: StepState | None, outcome: RelationOutcome | None):
        """One repair or review request with one re-ask on an invalid answer [C-9]; a second invalid answer records
        repair_invalid and becomes hold_still. Returns (answer, info) for the trace event."""
        ctx = self.ctx
        review = kind == "review"
        self._enter("review" if review else "repair", None)
        last = outcome.repair_payload() if outcome is not None else None
        failed_i = self.index_of(failed)
        if review:
            self.reviews_used += 1
        else:
            self.repairs_used += 1
        for attempt in range(1 + REPAIR_REASKS):
            holding = self.holding_pid()
            block = self.status_block()
            need = RepairNeed(kind, self.turns(ctx.view), block, holding=holding, last_outcome=last)
            reply = yield need
            if not isinstance(reply, RepairReply):
                raise TypeError("a RepairNeed is answered with a RepairReply")
            issues = self.repair_issues(reply, failed_i, review=review)
            info = {"request": {"status": block, "last_outcome": last}, "answer": reply.raw,
                    "valid": not issues, "issues": _issue_dicts(issues), "reask": attempt > 0,
                    "failed_i": failed_i}
            if not issues:
                self._accepted(kind)
                return reply.answer, info
            if not reply.schema_error:
                self._rejected(kind)
            if attempt < REPAIR_REASKS:
                # open (raised to B0/BC): RepairNeed and client.request_repair carry no re-ask error text yet
                # (PlanNeed has reask_error), so the re-ask repeats the request; the issues are in this event
                self._emit(kind, **info, applied=None)
        threshold = th_value("REPAIR_reasks", REPAIR_REASKS, "DD", "[C-9]: one re-ask of an invalid repair "
                                                                   "answer, then hold_still; REPAIR_REASKS")
        pids = self._pids(failed.step, self.grip_of(failed)) if failed is not None else {}
        self._outcome("plan", "rejected", "repair_invalid", pids, {"kind": kind, "issues": info["issues"],
                                                                    "failed_i": failed_i}, (threshold,), state=failed)
        info["fallback"] = "hold_still"
        return RepairAnswer("invalid twice", "hold_still", None, None), info

    def escalate(self, failed: StepState | None, outcome: RelationOutcome):
        """§C3.8: after a relation failed (after its own local retries), or a lost already-true claim: one repair
        request carrying the status block, the last outcome (reason, evidence text, thresholds) and the holding
        state; the answer's action is applied."""
        previous = self.active_state
        self.active_state = failed if failed is not None else previous
        answer, info = yield from self.ask("repair", failed, outcome)
        applied = yield from self.apply(answer, failed, review=False, outcome=outcome)
        self._emit("repair", **info, applied=applied)
        self.active_state = previous

    def review_due(self) -> bool:
        """Spec §3.7: every step finished, every stays check holding at the last boundary (an already-true claim
        without a check yet is unknown, so it blocks), at most B3 reviews,
        something finished since the last review (a review is not repeated over an unchanged state), and one hold
        cycle (HOLD_RECHECK_STEPS) since the last relation ended: the relations are re-checked after the scene
        settles, and the episode really continues (raw success often follows a release by a few steps)."""
        if self.plan is None or not self._review_armed or not self._settled or self.reviews_used >= B3_MAX_REVIEWS:
            return False
        if any(state.status not in FINISHED_STATUSES for state in self.states) or self._unchecked:
            return False
        return all(getattr(report, "verdict", None) == "met" for report in self.last_stays.values())

    def review(self):
        self._review_armed = False
        threshold = th_value("B3_reviews", B3_MAX_REVIEWS, "DD", "spec §3.7: at most B3_MAX_REVIEWS reviews per "
                                                                 "episode", used=self.reviews_used + 1)
        answer, info = yield from self.ask("review", None, None)
        applied = yield from self.apply(answer, None, review=True, outcome=None)
        self._emit("review", **info, applied=applied, threshold=threshold.to_dict())

    def apply(self, answer: RepairAnswer, failed: StepState | None, *, review: bool, outcome):
        """§C3.8 actions. Returns the applied-action record of the repair / review event."""
        ctx = self.ctx
        action = answer.action
        target = failed
        if target is None and action in STEP_NAMING_ACTIONS:
            target = self.named_state(answer.step)
        applied = {"action": action, "part": answer.part, "step_i": self.index_of(target)}
        if not review:
            self.halted = action == "hold_still"
        if action == "hold_still":
            if review:
                claim = {"claim": "hold_still", "detail": {"why": answer.why}, "unverified": True}
                self.claims.append(dict(claim))
                self._emit("model_claim", **claim)
            applied["status"] = None if target is None else target.status
            return applied
        if action == "retry":
            if failed is not None and outcome is not None and outcome.resume is not None:
                self._resume[id(target)] = dict(outcome.resume)
            target.status = "pending"
        elif action == "repoint":
            ctx.scene.drop(answer.part)
            self._stale_keepouts.add(answer.part)      # its keep-outs are rebuilt from the new registration
            target.status = "pending"
            waiting = yield from self.register_missing()
            applied["registration"] = ("waits_for_empty_hand" if answer.part in waiting
                                       else getattr(ctx.scene.parts.get(answer.part), "status", None))
        elif action == "use_part":
            rest = tuple(pid for pid in self.grip_of(target) if pid != answer.part)
            self._grip[id(target)] = ((answer.part,) + rest)[:PLAN_MAX_GRIP]
            target.status = "pending"
            yield from self._refresh_roles()
        elif action == "insert_step":
            at = self.index_of(failed) if failed is not None else len(self.states)
            self.plan = with_inserted(self.plan, at, answer.step)
            self.states.insert(at, StepState.pending(at, answer.step))
            self._renumber()
            if failed is not None:
                failed.status = "pending"
            yield from self._refresh_roles()
            applied["step_i"] = at
        elif action == "skip_step":
            target.status = "stays" if self.stays_hold_now(target) else "skipped"
        elif action == "swap_next":
            i = self.index_of(failed)
            self.states[i], self.states[i + 1] = self.states[i + 1], self.states[i]
            steps = list(self.plan.steps)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
            self.plan = Plan(self.plan.scene, self.plan.parts, self.plan.hold_true, tuple(steps), self.plan.source)
            ctx.plan = self.plan
            self._renumber()
            failed.status = "pending"
        elif action == "replan":
            plan, plan_answer = yield from self.obtain_plan("replan")
            if plan is None:                                    # spec §3.4: invalid_plan, the stuck alert, and
                self._record_invalid_plan(plan_answer)          # holding still: no step that never ran starts
                self.halted = True
                applied["replan"] = "invalid_plan"
            else:
                yield from self.adopt(plan, plan_answer, replan=True)
                applied["replan"] = "adopted"
        else:                                                   # unreachable: RepairAnswer validates the action
            raise ValueError(f"unknown repair action {action!r}")
        ctx.plan = self.plan
        applied["status"] = None if target is None else target.status
        applied["step_i"] = applied["step_i"] if action == "insert_step" else self.index_of(target)
        return applied

    def _refresh_roles(self):
        """Roles after an inserted step or a use_part grip (a use_part part gets the grip role); merge_plan refreshes
        the records' roles (registrations persist by id and text); parts that now need a registration get one."""
        ctx = self.ctx
        roles = part_roles(self.plan)
        for state in self.states:
            for pid in self._grip.get(id(state), ()):
                roles[pid] = frozenset(roles.get(pid, frozenset()) | {"grip"})
        self.roles = roles
        ctx.roles = roles
        ctx.scene.merge_plan(self.plan, roles, self.plan_view, self.plan_turns)
        yield from self.register_missing()


# --------------------------------------------------------------------------------------------- the policy
class AgentSpeedUpV2Policy:
    """Runner contract (§C13). ``episode_factory(ctx)`` returns the episode generator (or an object with ``run()``);
    None = ``StepRunner(ctx).run()``. ``overrides`` (only first_sense, depth_rule, forced_yaw_rad) and a
    ``relations`` stand-in layer are accepted only with an offline-mock client (tests and smokes; §C13.1)."""

    controller = CONTROLLER

    def __init__(self, client, *, episode_factory: Callable | None = None, overrides: Mapping | None = None,
                 clock=time.monotonic, relations=None):
        mock = getattr(client, "offline_mock", False) is True
        values = dict(overrides or {})
        unknown = sorted(set(values) - set(OVERRIDE_KEYS))
        if unknown:
            raise ValueError(f"unknown overrides {unknown}; allowed: {OVERRIDE_KEYS}")
        if values and not mock:
            raise ValueError("overrides are accepted only with an offline-mock client (tests and smokes)")
        if relations is not None and not mock:
            raise ValueError("a relation layer stand-in is accepted only with an offline-mock client")
        self.client = client
        self._layer = _layer_or_default(relations)
        self._episode_factory = episode_factory
        self._overrides = values
        self._clock = clock
        self.reset()

    # ------------------------------------------------------------------------------------------------ state
    def reset(self) -> None:
        layer = self._layer
        self.step_index = 0
        robot_model = RobotModel()
        scene = SceneModel(robot_model)
        self._events = EventSink(lambda: self.step_index)
        self.ctx = layer.RelationContext(robot=robot_model, scene=scene, monitor=InvariantMonitor(),
                                         stays=StaysChecker(), events=self._events,
                                         held_estimator=HeldFrameEstimator(scene, robot_model),
                                         overrides=dict(self._overrides))
        self.fulfiller = NeedFulfiller(self.client, self._events, clock=self._clock, ctx=self.ctx)
        self.fulfiller.guard = self._primitive_active
        if self._episode_factory is None:
            self._runner = StepRunner(self.ctx, relations=layer, call_counts=self.fulfiller.counts)
            self._gen = self._runner.run()
        else:
            made = self._episode_factory(self.ctx)
            is_runner = callable(getattr(made, "run", None)) and not callable(getattr(made, "send", None))
            self._runner = made if is_runner else None
            self._gen = made.run() if is_runner else made
        self._calibrate_class = _calibrate_class(layer)
        self._gen_started = False
        self._gen_done = False
        self.final_value = None
        self._reply = None
        self._pending_need = None
        self._primitive = None
        self._pending = None
        self._expected_observation = None
        self._cached_view: tuple | None = None
        self._client_checked = False
        self.outcomes: list[dict] = []
        self.ledger: list[dict] = []
        self._last_outcome: dict | None = None
        self._outcome_counter = 0
        self._macro: str | None = None
        self._macro_steps = 0
        self._counts = {"hold_actions": 0, "deferred_needs": 0, "preemptions": 0, "transition_limit": 0,
                        "acts_with_calls": 0}
        self._hold_causes: dict[str, int] = {}
        self._steps_by_phase: dict[str, int] = {}
        self._relations_by_status: dict[str, dict] = {}

    def _primitive_active(self) -> bool:
        return self._primitive is not None and not self._primitive.done

    # ------------------------------------------------------------------------------------- runner contract
    def act(self, observation) -> Decision:
        if self._pending is not None:
            raise RuntimeError("observe_transition must acknowledge the pending action before act")
        if self._expected_observation is not None and observation != self._expected_observation:
            raise ValueError("act requires the observation from the preceding transition")
        view = self._view(observation)
        if not self._client_checked:
            check = getattr(self.client, "check_observation", None)
            if callable(check):
                check(observation)           # the client's sentence must be this episode's (spec §4.2)
            self._client_checked = True
        ctx = self.ctx
        ctx.view, ctx.step = view, self.step_index
        parts_uv = self._parts_uv(view)
        calls: list[CallRecord] = []
        pointing: list[dict] = []
        outcomes: list[dict] = []
        routing = None
        action, acting = None, None
        hold_cause = "transition_limit"
        for _ in range(MAX_TRANSITIONS_PER_ACT):
            if self._gen_done and self._primitive is None:
                hold_cause = "episode_generator_ended"
                break
            pre = ctx.monitor.peek_preemption()
            primitive = self._primitive
            if pre is not None and primitive is not None:              # §C3.3 cases 1 and 2
                self._primitive = None
                item = self._advance(self._synthetic(pre, primitive), outcomes)
            elif primitive is not None and not primitive.done:
                action, acting = tuple(float(v) for v in primitive.next_action(view)), primitive
                break
            elif primitive is not None:
                self._primitive = None
                item = self._advance((primitive.outcome(), primitive), outcomes)
            elif self._pending_need is not None:
                item, self._pending_need = self._pending_need, None
            else:
                reply, self._reply = self._reply, None
                item = self._advance(reply, outcomes)
            status, detail = self._dispatch(item, view, calls, outcomes)
            if status == "need":
                routing = detail.get("routing") or routing
                if detail.get("pointing") is not None:
                    pointing.append(detail["pointing"])
            elif status == "deferred":
                hold_cause = "need_deferred"
                break
            elif status == "end":
                hold_cause = "episode_generator_ended"
                break
        if action is None:
            action = tuple(float(v) for v in measured_current_pose_hold_action())
            self._counts["hold_actions"] += 1
            self._hold_causes[hold_cause] = self._hold_causes.get(hold_cause, 0) + 1
            if hold_cause == "transition_limit":
                self._counts["transition_limit"] += 1
        if calls:
            self._counts["acts_with_calls"] += 1
        phase = self._phase()
        self._steps_by_phase[phase] = self._steps_by_phase.get(phase, 0) + 1
        macro = self._macro_label()
        self._macro_steps = self._macro_steps + 1 if macro == self._macro else 1
        self._macro = macro
        diagnostics = self._diagnostics(view, acting, calls, pointing, outcomes, routing, parts_uv)
        decision = Decision(action, diagnostics)
        self._pending = (observation, decision, view, acting)
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after) -> dict:
        if self._pending is None:
            raise RuntimeError("observe_transition requires a pending action")
        pending_observation, pending_decision, before_view, primitive = self._pending
        if before != pending_observation or decision is not pending_decision:
            raise ValueError("observe_transition requires the exact pending decision and observation")
        action = np.asarray(decision.action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-9):
            raise ValueError("executed action must contain seven finite values in [-1, 1]")
        after_view = self._view(after)
        feedback_of_primitive = None
        if primitive is not None:
            feedback_of_primitive = primitive.observe_transition(before_view, decision.action, after_view)
        ctx = self.ctx
        reorienting = isinstance(primitive, Orient) or bool(getattr(primitive, "commanded_reorientation", False))
        context = {"primitive_feedback": feedback_of_primitive,
                   "primitive": None if primitive is None else primitive.name,
                   "relation": getattr(ctx, "active_relation", None), "held": getattr(ctx, "holding", None),
                   "step": int(decision.diagnostics.get("step", self.step_index - 1)),
                   "commanded_reorientation": reorienting}
        ctx.monitor.observe(before_view, decision.action, after_view, context=context)
        changes = ctx.monitor.last_changes() if callable(getattr(ctx.monitor, "last_changes", None)) else []
        for change in changes:                   # stamped with the executed action's step, not the next act's
            self._events.emit_at(context["step"], "observer", **change)
        delta = after_view.position - before_view.position
        feedback = jsonable({
            "type": "agent_transition", "step": context["step"], "before_state": list(before_view.state),
            "after_state": list(after_view.state), "normalized_action": action.tolist(),
            "measured_translation_delta_m": delta.tolist(), "finger_gap_mm": after_view.gap_mm,
            "gripper_command": float(action[6]), "relation": context["relation"],
            "primitive": context["primitive"] or "hold", "primitive_feedback": feedback_of_primitive,
            "invariants": changes})
        self._expected_observation = after
        self._pending = None
        return feedback

    # -------------------------------------------------------------------------------------- generator steps
    def _advance(self, value, outcomes: list):
        """Send ``value`` into the episode generator (next() on its first advance); drain ctx.record outcomes."""
        self.ctx.step = self.step_index
        try:
            if not self._gen_started:
                self._gen_started = True
                item = next(self._gen)
            else:
                item = self._gen.send(value)
        except StopIteration as stop:
            self._gen_done = True
            self.final_value = stop.value
            item = _END
        outcomes.extend(self._drain())
        return item

    def _dispatch(self, item, view: StepView, calls: list, outcomes: list):
        """Factory -> construct the primitive (unless a preemption waits: §C3.3 case 3, the synthetic outcome with
        primitive None goes instead); Need -> one call within the act's limit, else hold it for the next act."""
        while True:
            if item is _END:
                return "end", None
            if _is_factory(item):
                pre = self.ctx.monitor.peek_preemption()
                if pre is not None:
                    item = self._advance(self._synthetic(pre, None), outcomes)
                    continue
                primitive = item(view)
                if getattr(primitive, "pointing_request", None) is not None:
                    raise RuntimeError(f"primitive {primitive.name} requests pointing; v2 primitives never do (§C1.6)")
                self._primitive = primitive
                return "primitive", None
            if isinstance(item, (PlanNeed, PointNeed, RepairNeed)):
                if len(calls) >= self.call_limit():
                    self._pending_need = item
                    self._counts["deferred_needs"] += 1
                    return "deferred", None
                answer = self.fulfiller.fulfil(item, view, calls)
                self._reply = answer
                return "need", {"routing": self._routing(item, answer), "pointing": self._pointing(item, answer)}
            raise TypeError(f"the episode generator yielded {type(item).__name__}, not a factory or a Need")

    def _synthetic(self, pre, primitive):
        """§C3.3 synthetic failed outcome; the delivered report is removed from the bus queue (take_preemption)."""
        key, report = pre
        actions = 0 if primitive is None else int(primitive.actions_executed)
        reason = PREEMPTED_PREFIX + str(report.reason or report.verdict)
        outcome = {"primitive": None if primitive is None else primitive.name, "status": "failed", "reason": reason,
                   "evidence": {"invariant": key, "report": report.to_dict(), "actions": actions},
                   "actions": actions}
        self.ctx.monitor.take_preemption()
        self._counts["preemptions"] += 1
        self._events.emit("invariant", key=key, report=report.to_dict(),
                          primitive=None if primitive is None else primitive.name)
        return jsonable(outcome), primitive

    def _drain(self) -> list[dict]:
        ctx = self.ctx
        pending = list(getattr(ctx, "pending_outcomes", []) or [])
        if not pending:
            return []
        ctx.pending_outcomes.clear()
        runner = self._runner
        records = []
        for outcome in pending:
            self._outcome_counter += 1
            state = runner.plan_state_of(outcome) if runner is not None else None
            plan_step = runner.index_of(state) if runner is not None else None
            record = outcome.record(self._outcome_counter, self.step_index, plan_step)
            records.append(record)
            if state is not None:
                state.outcomes.append(self._outcome_counter)
            bucket = self._relations_by_status.setdefault(record["relation"], {})
            bucket[record["status"]] = bucket.get(record["status"], 0) + 1
            self._last_outcome = record
            self.ledger = (self.ledger + [{key: record.get(key) for key in (
                "n", "step", "relation", "status", "reason", "plan_step", "pids", "evidence_text",
                "local_retries")}])[-LEDGER_LIMIT:]
        self.outcomes = (self.outcomes + records)[-OUTCOME_HISTORY_LIMIT:]
        return records

    def call_limit(self) -> int:
        """B1, or one call while the rejection storm damping is active (§C3.4)."""
        rejections = int(getattr(self.ctx, "consecutive_rejections", None) or 0)
        return B1_STORM_CALLS_PER_ACT if rejections >= REJECTION_STORM_LIMIT else B1_MAX_CALLS_PER_ACT

    # ------------------------------------------------------------------------------------------ diagnostics
    def _view(self, observation) -> StepView:
        if self._cached_view is not None and self._cached_view[0] is observation:
            return self._cached_view[1]
        view = StepView(observation)
        self._cached_view = (observation, view)
        return view

    def _parts_uv(self, view: StepView) -> dict:
        on_view = getattr(self.ctx.scene, "on_view", None)
        if not callable(on_view):
            return {}
        return on_view(view, self.step_index) or {}

    def _phase(self) -> str:
        runner = self._runner
        if runner is not None:
            return runner.phase
        return "episode_ended" if self._gen_done else "episode"

    def _macro_label(self):
        runner = self._runner
        if runner is None:
            return getattr(self.ctx, "active_relation", None)
        state = runner.active_state
        if runner.phase == "step" and state is not None:
            return f"{RELATION_OF[state.step.do]}({state.step.obj})"
        return PHASE_MACROS.get(runner.phase, runner.phase)

    def _relation(self) -> dict:
        runner = self._runner
        if runner is None:
            return {"step_i": None, "do": None, "pids": None, "phase": self._phase(),
                    "relation": getattr(self.ctx, "active_relation", None)}
        state = runner.active_state
        if state is None:
            return {"step_i": None, "do": None, "pids": None, "phase": runner.phase,
                    "relation": getattr(self.ctx, "active_relation", None)}
        step = state.step
        return {"step_i": runner.index_of(state), "do": step.do,
                "pids": {"obj": step.obj, "grip": list(runner.grip_of(state)), "ref": step.ref},
                "phase": runner.phase, "relation": getattr(self.ctx, "active_relation", None)}

    @staticmethod
    def _routing(need, answer) -> dict:
        if isinstance(need, PlanNeed):
            plan = getattr(answer, "plan", None)
            return {"kind": need.kind, "skill": need.kind,
                    "params": {"quarter_turns": need.quarter_turns, "fingertip_edge": need.fingertip_edge,
                               "holding": need.holding, "reask": need.reask_error is not None},
                    "situation": plan.scene if plan is not None else getattr(answer, "parse_error", None)}
        if isinstance(need, PointNeed):
            return {"kind": need.kind, "skill": need.kind,
                    "params": {"pid": need.pid, "camera": need.camera, "purpose": need.purpose,
                               "quarter_turns": need.quarter_turns, "reask": need.previous_rejected is not None},
                    "situation": None}
        repair = getattr(answer, "answer", None)
        return {"kind": need.kind, "skill": need.kind,
                "params": {"holding": need.holding,
                           "reason": None if need.last_outcome is None else need.last_outcome["reason"],
                           "action": None if repair is None else repair.action},
                "situation": None if repair is None else repair.why}

    @staticmethod
    def _pointing(need, answer) -> dict | None:
        if not isinstance(need, PointNeed):
            return None
        reply = answer.reply
        stage = "fine" if need.kind == "point_fine" else "coarse"
        answer_id = None if reply is None else reply.answer_id
        return jsonable({"kind": need.kind, "stage": stage, "pid": need.pid, "camera": need.camera,
                         "purpose": need.purpose, "point_id" if stage == "fine" else "cell_id": answer_id,
                         "answer_id": answer_id, "uv_upright": None if reply is None else reply.uv_upright,
                         "quantization_px": None if reply is None else reply.quantization_px,
                         "quarter_turns": need.quarter_turns, "schema_error": answer.schema_error,
                         "check": None, "round": None})

    def _diagnostics(self, view, acting, calls, pointing, outcomes, routing, parts_uv) -> dict:
        ctx = self.ctx
        name = acting.name if acting is not None else "hold"
        stats = getattr(self.client, "history_stats", None)
        latency = sum(call.latency_s for call in calls) if calls else None
        last_call = calls[-1] if calls else None
        limit = self.call_limit()
        held = getattr(ctx, "holding", None)
        snapshot = ctx.monitor.snapshot() if callable(getattr(ctx.monitor, "snapshot", None)) else {}
        # §C18.2: the CalibratePlus primitive "or one of its stages": any primitive the StepRunner's calibrate
        # phase runs is a Calibrate+ stage, so its closes stay out of the runner's first_close_event
        calibrating = acting is not None and (_is_calibration(acting, self._calibrate_class)
                                              or self._phase() == "calibrate")
        return jsonable({
            "step": self.step_index, "controller": CONTROLLER, "macro": self._macro, "primitive": name,
            "execution_kind": f"{CONTROLLER}:{name}", "macro_step": self._macro_steps,
            "refreshed": bool(calls), "routing": routing,
            "api_calls_in_act": [call.kind for call in calls], "api_latency_s": latency,
            "api_request_key": None if last_call is None else last_call.request_key,
            "api_request_log": None if last_call is None else last_call.request_log,
            "pointing": pointing, "outcomes": outcomes, "outcome": outcomes[-1] if outcomes else None,
            "history_stats": stats() if callable(stats) else None,
            "calibration_probe": bool(calibrating),
            "api_calls": [{"kind": call.kind, "latency_s": call.latency_s, "ok": call.ok} for call in calls],
            "v2_events": self._events.drain(), "relation": self._relation(), "observers": snapshot,
            "parts_uv": parts_uv, "holding": getattr(held, "pid", None), "finger_gap_mm": view.gap_mm,
            "pending_need": None if self._pending_need is None else self._pending_need.kind,
            "call_limit": th_value("B1_calls_per_act", limit, "DD", "B1_MAX_CALLS_PER_ACT, or B1_STORM_CALLS_PER_ACT "
                                   "after REJECTION_STORM_LIMIT consecutive rejections",
                                   rejections=int(getattr(ctx, "consecutive_rejections", None) or 0)).to_dict(),
            "update_steps": 1, "feedback": None})

    # ------------------------------------------------------------------------------------------- read-outs
    @property
    def calibration(self) -> dict | None:
        cal = getattr(self.ctx, "cal", None)
        summary = getattr(cal, "summary", None)
        return jsonable(summary()) if callable(summary) else None

    @property
    def relation_plan(self) -> dict | None:
        """The validated plan JSON (TRUTH-EXT parse_check), or None before a valid plan."""
        plan = self._runner.plan if self._runner is not None else getattr(self.ctx, "plan", None)
        return None if plan is None else plan.to_json()

    @property
    def holding(self) -> str | None:
        return getattr(getattr(self.ctx, "holding", None), "pid", None)

    def agent_stats(self) -> dict:
        ctx = self.ctx
        runner = self._runner
        cal = getattr(ctx, "cal", None)
        scene = getattr(ctx, "scene", None)
        stats = getattr(self.client, "history_stats", None)
        history = stats() if callable(stats) else {}
        parts = {}
        if scene is not None and callable(getattr(scene, "summary", None)):
            parts = (scene.summary() or {}).get("parts", {})
        local = sum(int(record.get("local_retries") or 0) for record in self.outcomes)
        versions = {"policy": POLICY_VERSION, "plan": PLAN_SCHEMA_VERSION, "scene": SCENE_VERSION,
                    "observers": OBSERVERS_VERSION, "planner": planner.PLANNER_VERSION,
                    "geometry": geometry.GEOMETRY_VERSION, "relations": getattr(self._layer, "RELATIONS_VERSION", None)}
        versions.update(kernel.module_versions())
        return jsonable({
            "policy_version": POLICY_VERSION, "step": self.step_index, "phase": self._phase(),
            "calibration": self.calibration,
            "counts": {"calls_by_kind": (history or {}).get("calls_by_kind", dict(self.fulfiller.counts)),
                       "calls_by_need": dict(self.fulfiller.counts),
                       "relations_by_status": self._relations_by_status, "local_retries": local,
                       "repairs": None if runner is None else runner.repairs_used,
                       "reviews": None if runner is None else runner.reviews_used,
                       "rejections": {"consecutive": int(getattr(ctx, "consecutive_rejections", None) or 0),
                                      **self.fulfiller.errors},
                       "preemptions": self._counts["preemptions"], "hold_actions": self._counts["hold_actions"],
                       "hold_actions_by_cause": dict(self._hold_causes),
                       "deferred_needs": self._counts["deferred_needs"],
                       "acts_with_calls": self._counts["acts_with_calls"], "events": dict(self._events.counts)},
            "step_budget": {"steps": self.step_index, "by_phase": dict(self._steps_by_phase),
                            "by_plan_step": [] if runner is None else [
                                {"i": s.i, "do": s.step.do, "obj": s.step.obj, "steps": s.steps, "attempts": s.attempts,
                                 "status": s.status} for s in runner.states]},
            "plan": self.relation_plan,
            "steps": [] if runner is None else [state.compact() for state in runner.states],
            "parts": parts, "holding": self.holding, "ledger": list(self.ledger),
            "step_ledger": [] if runner is None else list(runner.ledger),
            "claims": [] if runner is None else list(runner.claims),
            "last_outcome": self._last_outcome, "rulings": dict(robot.RULINGS),
            "mount": getattr(cal, "mount", None), "versions": versions})
