"""Optional, interruptible native skills for the model-owned user-check loop.

The native relations remain accelerators with their own applicability limits.
Their inability to measure a prerequisite never disables direct task control.
There is no native retry/replan ladder behind this adapter.
"""
from __future__ import annotations

from copy import deepcopy

from . import agent_speedup_v2 as base
from .agent_speedup_v2_primitives import CalibratePlus
from .agent_speedup_v2_user_check_constraints import ConstraintRunnerMixin, causal_feedback
from .agent_speedup_v2_user_check_plan import with_extensions
from .agent_speedup_v2_user_check_runtime import StableScene, compact_evidence
from .agent_speedup_v2_user_check_spatial import WholeObjectHeldFrameEstimator


class _OneSkillRunner(ConstraintRunnerMixin, base.StepRunner):
    """Run the chosen skill once; the executive owns every subsequent decision."""

    def escalate(self, *args, **kwargs):
        if False:
            yield

    def stays_boundary_check(self):
        # Preserve measurements, without invoking the native repair policy.
        self.last_stays = dict(self.ctx.stays.check_all(self.ctx.view))
        if False:
            yield


class AgentMacroSession:
    """Coroutine adapter; the host pauses it after the granted action budget.

    ``advance`` yields ordinary v2 primitive factories and observation Needs.
    The host retains an unfinished primitive when pausing, and only resumes it
    after a new explicit model decision. ``None`` means this attempt ended.
    """

    def __init__(self, ctx, lock, relations, call_counts=None):
        self.ctx, self.lock, self.relations = ctx, lock, relations
        self.call_counts = call_counts or {}
        self.runner = None
        self.generator = None
        self.started = False
        self.step_index = None
        self.result = None
        self._monitor_keys = set()
        self._geometry_stale = False
        self._start_step = int(getattr(ctx, "step", 0))
        self._end_step = self._start_step

    @property
    def active(self):
        return self.generator is not None

    def start(self, step_index):
        self.cancel("replaced_by_model")
        raw = self.lock.plan()
        if isinstance(step_index, bool) or not isinstance(step_index, int) or not 0 <= step_index < len(raw["steps"]):
            self.result = {"status": "unavailable", "reason": "unknown_reviewed_step", "executed": False}
            return False
        self.step_index, self.result = step_index, None
        self._start_step = int(self.ctx.step)
        self._end_step = None
        self._monitor_keys = set(self.ctx.monitor.keys())
        self.runner = None
        self.started = False
        self.generator = self._execute()
        return True

    def advance(self, reply=None):
        if self.generator is None:
            return None
        try:
            if not self.started:
                self.started = True
                return next(self.generator)
            return self.generator.send(reply)
        except StopIteration as stop:
            self.generator = None
            self._end_step = int(self.ctx.step)
            self.result = stop.value or {"status": "returned", "reason": "native_skill_returned"}
            return None

    def _execute(self):
        ctx, layer = self.ctx, self.relations
        selected = self.lock.plan()["steps"][self.step_index]
        if selected["do"] not in {"hold", "turn", "place_on", "release"}:
            return {"status": "unavailable", "reason": "native_skill_not_implemented", "executed": False}
        closed = ctx.memory.get("user_check_gripper_command") == "close"
        native_held = getattr(ctx, "holding", None)
        # Calibration and re-registration may open the hand. An unknown grasp
        # is not permission to do that. Direct control remains available.
        if closed and (native_held is None or self._geometry_stale or getattr(ctx, "cal", None) is None):
            return {"status": "unavailable", "reason": "native_geometry_unavailable_while_hand_closed",
                    "executed": False, "direct_control_available": True,
                    "remedy": "Continue observed direct placement, or explicitly open on support before a new native observation."}
        if self._geometry_stale and not closed:
            ctx.scene = StableScene(ctx.robot)
            ctx.held_estimator = WholeObjectHeldFrameEstimator(ctx.scene, ctx.robot)
            ctx.holding = None
            ctx.stays = base.StaysChecker()
            self._geometry_stale = False
        runner = self.runner = _OneSkillRunner(ctx, relations=layer, call_counts=self.call_counts)
        if getattr(ctx, "cal", None) is None:
            runner._enter("calibrate", "calibrated")
            runner.calibration = yield from layer.calibrate_plus(ctx)
            runner.controller_returned()
            if ctx.cal is None:
                ctx.cal = runner.calibration
            if ctx.cal is None:
                return {"status": "unavailable", "reason": "native_calibration_unavailable"}
        # No mandatory re-grounding call or calibration for the direct path.
        # Ground after any calibration movement, using current image cells.
        runner._enter("plan", None)
        k = runner.turns(ctx.view)
        answer = yield base.PlanNeed("plan", k, base.FINGERTIP_EDGES[k], holding=runner.holding_pid())
        if not isinstance(answer, base.PlanAnswer) or answer.plan is None:
            return {"status": "unavailable", "reason": "native_grounding_failed", "executed": False}
        plan = with_extensions(answer.plan, self.lock.plan())
        self.lock.check_plan(plan.to_json(), source="optional_macro_grounding")
        runner.plan = plan
        runner.plan_view, runner.plan_turns = answer.view, int(answer.need.quarter_turns)
        runner.states = [base.StepState.pending(i, step) for i, step in enumerate(plan.steps)]
        state = runner.states[self.step_index]
        runner.active_state = state
        # Only the selected step and its explicit constraints need native
        # geometry. Future steps do not introduce preconditions/keep-outs.
        roles = base.part_roles(plan)
        step = state.step
        wanted = {step.obj, step.ref, *step.grip}
        wanted.update(keep.p for keep in step.keep)
        wanted.update(until.p for until in step.until if until.p is not None)
        for claim in plan.hold_true:
            wanted.update((claim.a, claim.b))
        roles = {pid: role if pid in wanted else frozenset() for pid, role in roles.items()}
        runner.roles = roles
        ctx.plan, ctx.roles = plan, roles
        ctx.scene.merge_plan(plan, roles, answer.view, answer.need.quarter_turns)
        # The old checkpoint callback quarantined pending identity then forced
        # another lift. Native evidence is now returned to the executive as-is.
        ctx.memory.pop("user_check_checkpoint", None)
        if ctx.holding is None:
            runner._enter("register", "registered")
            yield from layer.register_regions(ctx, plan, roles)
            runner.controller_returned()
        runner.add_hold_true(plan)
        runner.ensure_keepouts()
        # Optional macro prerequisites may refuse locally. They never start a
        # hidden semantic repair or block the executive's direct interface.
        yield from runner.run_step(state)
        return {"status": "succeeded" if state.status == "done" else "failed",
                "reason": state.reason, "step_index": self.step_index,
                "native_evidence": deepcopy(state.evidence),
                "native_status_is_not_task_success": True,
                "direct_control_available": True}

    def _actions_consumed(self):
        end = int(self.ctx.step) if self._end_step is None else self._end_step
        return max(0, end - self._start_step)

    def setup_progress(self, primitive=None):
        """Describe native setup work, without treating a grant boundary as failure.

        The host owns the live primitive. Passing it supplies measured stage
        progress; omitting it deliberately leaves those details unknown. The
        nominal calibration cost documents the existing implementation and is
        neither a timeout nor permission to skip its required measurements.
        """
        phase = None if self.runner is None else self.runner.phase
        selected = None
        raw = self.lock.plan()
        if self.step_index is not None and 0 <= self.step_index < len(raw["steps"]):
            selected = raw["steps"][self.step_index]
        supported = selected is not None and selected["do"] in {"hold", "turn", "place_on", "release"}
        calibrated = getattr(self.ctx, "cal", None) is not None
        cal = primitive if isinstance(primitive, CalibratePlus) else None
        current = None if cal is None else cal.current
        result_status = (self.result or {}).get("status")
        primitive_failed = cal is not None and cal.status == "failed"
        if result_status in {"unavailable", "failed", "cancelled"}:
            status = result_status
        elif primitive_failed:
            status = "failed"
        elif self.active and phase in {None, "calibrate", "plan", "register"}:
            status = "normal_initialization"
        elif self.active:
            status = "executing_skill"
        elif result_status is not None:
            status = "returned"
        else:
            status = "not_requested"

        required_parts = set()
        if selected is not None:
            required_parts.update((selected["obj"], selected.get("ref"), *selected.get("grip", ())))
            required_parts.update(item["p"] for item in selected.get("keep", ()))
            required_parts.update(item.get("p") for item in selected.get("until", ()))
            for claim in raw.get("hold_true", ()):
                required_parts.update((claim["a"], claim.get("b")))
            required_parts.discard(None)

        # A completed/failed/cancelled session cannot be resumed. An active
        # generator may continue at a host-issued pause; it never restarts its
        # calibration merely because the previous action grant was exhausted.
        resumable = self.active and supported and not primitive_failed
        calibration_status = ("calibrated" if calibrated else "failed" if primitive_failed
                              else "pending" if supported else "not_requested")
        return base.jsonable({
            "status": status,
            "phase": phase,
            "observed_at_step": int(self.ctx.step),
            "actions_consumed": self._actions_consumed(),
            "calibrated": calibrated,
            "calibration": {
                "status": calibration_status,
                "stage_detail_available": cal is not None,
                "actions_consumed": None if cal is None else cal.actions_executed,
                "completed_stages": None if cal is None else deepcopy(cal.stage_outcomes),
                "current_stage": None if cal is None else cal.stage,
                "current_stage_actions": None if current is None else current.actions_executed,
                "pending_stages": None if cal is None else list(cal._queue),
                "stages_started": None if cal is None else list(cal.stages_run),
                "conditional_third_close": None if cal is None else cal.third_close,
                "estimated_total_steps": 190,
                "estimate_source": "CalibratePlus implementation docstring; nominal full cold setup",
                "estimate_is_hard_cap": False,
                "estimate_note": ("This is an approximate full-calibration cost, not remaining work or a "
                                  "failure threshold. Settling, probes and a conditional third close vary. "
                                  "A 60-action grant normally cannot finish cold calibration."),
                "interpretation": ("Closing, opening and yawing the empty hand are calibration progress. "
                                   "Unchanged task objects during these stages do not establish a stall."),
            },
            "required_capabilities": [] if not supported else [
                {"capability": "native_robot_image_calibration", "phase": "calibrate",
                 "available": calibrated,
                 "reason": "Native image-space control needs measured gripper, image-motion and camera relations."},
                {"capability": "current_view_plan_grounding", "phase": "plan",
                 "reason": "Refresh approved identities' image cells after calibration motion; preserve intent."},
                {"capability": "selected_step_part_registration", "phase": "register",
                 "part_ids": sorted(required_parts),
                 "reason": "Measure selected-step geometry and applicable constraints; future steps do not add roles."},
            ],
            "following_setup_cost": {
                "grounding": "A current-view plan model request follows calibration.",
                "registration": ("Required selected-step geometry may need pointing model requests and "
                                 "observed probe actions; their cost is additional to the calibration estimate."),
                "skill_execution": "Task-directed manipulation starts only after native prerequisites are available.",
            },
            "budget_pause": {
                "is_failure": False,
                "meaning": "A host action-grant boundary is a normal observation/decision boundary, not a native failure.",
                "resume_available_at_pause": resumable,
                "resume": ({"step_index": self.step_index, "continuation": True} if resumable else None),
                "instruction": ("If only the grant expired, continue the same paused macro to retain its "
                                "primitive and setup progress, or choose direct control. Diagnose an actual "
                                "failed outcome separately; this status does not certify future success."),
            },
            "scope": ("These prerequisites belong only to the selected native macro. They never gate "
                      "direct motion or generic fragments using available robot/camera facts. An unavailable "
                      "native image-motion estimate is not silently replaced with assumed calibration."),
        })

    def status(self, primitive=None):
        runner = self.runner
        pending = list(getattr(self.ctx, "pending_outcomes", ()) or ())
        last = pending[-1] if pending else None
        return base.jsonable({
            "active": self.active, "step_index": self.step_index,
            "phase": None if runner is None else runner.phase,
            "actions_since_start": self._actions_consumed(),
            "geometry_stale_after_direct": self._geometry_stale,
            "result": self.result,
            "setup_progress": self.setup_progress(primitive),
            "native_holding_estimate": getattr(getattr(self.ctx, "holding", None), "pid", None),
            "last_outcome": None if last is None else {
                "relation": last.relation, "status": last.status, "reason": last.reason,
                "evidence": compact_evidence(last.evidence), "causal": causal_feedback(last, self.ctx)},
            "unknown_observation_is_not_direct_motion_prohibition": True,
        })

    def cancel(self, reason="model_took_control"):
        if self.generator is not None:
            self.generator.close()
            self.generator = None
            self._end_step = int(self.ctx.step)
            self.result = {"status": "cancelled", "reason": reason,
                           "physical_effects_preserved": True}
        # Do not leave a paused controller's private observer preempting all
        # future direct movements. Explicit constraints remain in the contract.
        for key in set(self.ctx.monitor.keys()) - self._monitor_keys:
            self.ctx.monitor.remove(key)
        self.ctx.memory.pop("user_check_checkpoint", None)

    def invalidate_after_direct(self, *args, **kwargs):
        """Never reuse pre-direct object geometry as a fresh physical fact."""
        self._geometry_stale = True
        if self.ctx.memory.get("user_check_gripper_command") == "open":
            self.ctx.holding = None
            self.ctx.scene.held = None
            self.ctx.scene._held_local = {}
        self.ctx.memory["agent_first_native_geometry_stale"] = True
