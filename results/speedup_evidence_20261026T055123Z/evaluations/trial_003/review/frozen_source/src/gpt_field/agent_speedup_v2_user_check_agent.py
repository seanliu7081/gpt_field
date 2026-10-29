"""Model-owned manipulation with optional, interruptible v2 macro skills.

The reviewed plan describes intent. It does not own the control loop. The
executive can finish the entire task using short observed motions, choose a
native macro for a bounded stretch, or abandon that macro at its next observed
boundary. An unavailable geometric verifier cannot disable direct control.

Only RGB observations, robot proprioception, camera/robot calibration and
native controller measurements enter this policy. Task success belongs to the
external evaluator; neither a model claim nor a completed motion certifies it.
The baseline v2 policy, primitives and Astranomy executor remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from . import agent_speedup_v2 as base
from .agent_speedup_v2_types import PlanNeed, PointNeed, RepairNeed, jsonable
from .agent_speedup_v2_user_check_local import LocalRecoveryFulfiller
from .agent_speedup_v2_user_check_motion import LocalMotionPrimitive
from .agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive, fragment_capabilities
from .astranomy_agentic_motion import ShortTargetConfig, measured_pose
from .policy import Decision
from .astra import SchemaError


POLICY_VERSION = "agent-speedup-v2-user-check-agent-2"
CONTROLLER = base.CONTROLLER
MAX_MACRO_GRANT_STEPS = 60
MAX_NATIVE_CALLS_PER_ACT = 2


@dataclass(frozen=True)
class ExecutiveCall:
    """Variant call kind; native CallRecord deliberately restricts its kinds."""

    kind: str
    latency_s: float
    request_key: str | None
    request_log: str | None
    ok: bool = True
    error: str | None = None


def _unused_episode(_ctx):
    # The base initializes its bookkeeping without starting a macro-owned
    # generator, calibration, registration or a second initial plan request.
    if False:
        yield


class AgentFirstV2Policy(base.AgentSpeedUpV2Policy):
    """Keep the evaluator contract while giving the model every task boundary."""

    controller = CONTROLLER

    def __init__(self, client, *, lock, **kwargs):
        if "episode_factory" in kwargs:
            raise ValueError("The agent-owned user-check policy owns its execution loop")
        self.plan_lock = lock
        super().__init__(client, episode_factory=_unused_episode, **kwargs)

    def reset(self):
        super().reset()
        from .agent_speedup_v2_user_check_control import build_relation_layer, refresh_relation_layer
        from .agent_speedup_v2_user_check_runtime import StableScene
        from .agent_speedup_v2_user_check_spatial import WholeObjectHeldFrameEstimator, install_spatial_layer
        from .agent_speedup_v2_user_check_agent_macros import AgentMacroSession

        self.ctx.scene = StableScene(self.ctx.robot)
        self.ctx.held_estimator = WholeObjectHeldFrameEstimator(self.ctx.scene, self.ctx.robot)
        self._agent_layer = build_relation_layer(self._layer)
        install_spatial_layer(self._agent_layer)
        refresh_relation_layer(self._agent_layer)
        previous = self.fulfiller
        self.fulfiller = LocalRecoveryFulfiller(self.client, self._events, clock=self._clock, ctx=self.ctx)
        self.fulfiller.counts, self.fulfiller.errors = previous.counts, previous.errors
        self.fulfiller.guard = self._primitive_active
        self.fulfiller.plan_lock = self.plan_lock
        self._macro_session = AgentMacroSession(
            self.ctx, self.plan_lock, self._agent_layer, self.fulfiller.counts)
        self._runner = None
        self.ctx.memory["user_check_gripper_command"] = "open"
        self._mode = "agent"
        self._need_agent = True
        self._macro_grant = 0
        self._macro_used = 0
        self._macro_index = None
        self._macro_paused = False
        self._macro_item = None
        self._macro_reply = None
        self._macro_active = False
        self._agent_decision = None
        self._belief = {"source": "model_estimate", "belief": "", "plan": "", "step": None}
        self._feedback = None
        self._feedback_history = []
        self._segment_start = None
        self._executive_calls = 0
        self._direct_segments = 0
        self._fragment_grants = 0
        self._fragment_actions = 0
        self._direct_actions = 0
        self._native_actions = 0
        self._last_fragment_result = None
        self._macro_grants = 0
        self._model_completion_claims = []
        self._pending_agent_events = []
        self._last_macro_feedback = None
        self._macro_transition_feedback = None
        self._sync_framework()

    def _task_contract(self):
        contract = self.plan_lock.task_contract.to_dict()
        contract.update(
            goal_source="official_task_with_reviewed_goal_meanings_and_constraints",
            execution_authority="model_owned_continuous_control",
            method_rule=("Reviewed steps are initial methods and optional macro choices. The model may "
                         "change execution order, grips and intermediate actions while preserving task "
                         "goals, entity identity, region meanings and applicable reviewed constraints."),
            completion_rule=("Only the independent task evaluator can terminate with success. "
                             "Native skill outcomes are local evidence; model beliefs are estimates. "
                             "Direct control may accomplish any subtask without native certification."))
        return contract

    def _capabilities(self):
        return {
            "execution_owner": "model",
            "execute": fragment_capabilities(),
            "direct_motion": {**ShortTargetConfig().to_dict(),
                              "can_complete_subtasks": True,
                              "requires_registration": False,
                              "requires_native_grasp_certificate": False},
            "native_macros": {"implemented": ["hold", "turn", "place_on", "release"],
                              "selection": "approved_plan.steps[step_index]",
                              "optional": True, "budget_steps": [1, MAX_MACRO_GRANT_STEPS],
                              "setup": "lazy current-view grounding, calibration and necessary registration",
                              "yield": "budget, first failed primitive, unavailable precondition or completion",
                              "resume": "same step_index with continuation=true while macro is paused",
                              "release_semantics": "return carried object to its original pick support, then open",
                              "direct_open_semantics": "open at the current commanded pose",
                              "no_automatic_repair_or_replan": True},
            "observe": "one measured-pose step retaining the last gripper command, then fresh RGB",
            "finish": "model completion claim plus fresh observation; does not stop the evaluator",
            "constraints": ("Preserve applicable reviewed constraints. Direct targets use bounded OSC "
                            "commands, not a certified collision-free planner. Native observer evidence "
                            "is reported when available; unknown is not proof of a collision."),
        }

    def _native_evidence(self):
        held = getattr(self.ctx, "holding", None)
        parts = {}
        summary = getattr(self.ctx.scene, "summary", None)
        if callable(summary):
            parts = (summary() or {}).get("parts", {})
        return {"holding_estimate": None if held is None else {
                    "part": getattr(held, "pid", None),
                    "grip": getattr(held, "grip_pid", None),
                    "source": "native_controller_estimate", "task_success": "unknown"},
                "calibration": self.calibration,
                "parts": parts,
                "observers": self.ctx.monitor.snapshot(),
                "outcomes": deepcopy(self.outcomes[-8:]),
                "interpretation": ("Missing evidence means unknown. Native estimates are not model "
                                   "beliefs, simulator truth, or a gate on direct motion.")}

    def _macro_status(self):
        status = dict(self._macro_session.status())
        status.update(active=self._macro_active, paused=self._macro_paused,
                      step_index=self._macro_index,
                      budget_steps=self._macro_grant, executed_in_grant=self._macro_used,
                      primitive=None if self._primitive is None or self._mode != "macro"
                      else self._primitive.name,
                      last_feedback=deepcopy(self._last_macro_feedback))
        status["setup_progress"] = self._macro_session.setup_progress(
            self._primitive if self._mode == "macro" else None)
        return status

    def _execution_state(self, view=None):
        active = self._mode == "fragment" and isinstance(self._primitive, FragmentSequencePrimitive)
        measured = None if view is None else {
            "source": "robot_proprioception", "observed_at_step": self.step_index,
            "pose_wxyz": measured_pose(view.state), "finger_gap_mm": view.gap_mm,
            "joint_position_rad": getattr(view.observation, "joint_position", None),
            "joint_velocity_rad_s": getattr(view.observation, "joint_velocity", None)}
        native = self._native_evidence()["holding_estimate"]
        return {
            "active": bool(active and not self._primitive.done),
            "progress": self._primitive.progress() if active else None,
            "last_result": deepcopy(self._last_fragment_result),
            "shared_state": {
                "measured_robot": measured,
                "last_gripper_command": self.ctx.memory["user_check_gripper_command"],
                "model_belief": deepcopy(self._belief),
                "native_holding_estimate": native,
                "holding_status": "native_estimate" if native is not None else "unverified",
                "handoff_rule": "Start from current measured pose and commanded gripper; no automatic release, return-to-pick or native registration. Model belief is not measured attachment.",
            },
            "control_steps": {"fragment": self._fragment_actions, "direct": self._direct_actions,
                              "native_macro": self._native_actions},
        }

    def _context(self, view):
        return jsonable({
            "step": self.step_index,
            "approved_plan": self.plan_lock.plan(),
            "task_contract": self._task_contract(),
            "geometry": getattr(view.observation, "geometry", None),
            "capabilities": self._capabilities(),
            "measured_state": {"pose_wxyz": measured_pose(view.state),
                               "finger_gap_mm": view.gap_mm,
                               "joint_position_rad": getattr(view.observation, "joint_position", None),
                               "joint_velocity_rad_s": getattr(view.observation, "joint_velocity", None),
                               "last_gripper_command": self.ctx.memory["user_check_gripper_command"]},
            "model_belief": self._belief,
            "last_feedback": self._feedback,
            "recent_feedback": self._feedback_history[-8:],
            "macro": self._macro_status(),
            "execution": self._execution_state(view),
            "native_evidence": self._native_evidence(),
            "task_success_source": "independent evaluator; not provided to this policy",
        })

    def _sync_framework(self, view=None):
        measured = None if view is None else {
            "pose_wxyz": measured_pose(view.state), "finger_gap_mm": view.gap_mm,
            "joint_position_rad": getattr(view.observation, "joint_position", None),
            "joint_velocity_rad_s": getattr(view.observation, "joint_velocity", None),
            "observed_at_step": self.step_index}
        self.plan_lock._world_state = jsonable({
            "version": POLICY_VERSION, "execution_owner": "model", "step": self.step_index,
            "measured_robot": measured,
            "last_gripper_command": self.ctx.memory["user_check_gripper_command"],
            "native_estimate": self._native_evidence(),
            "model_estimate": self._belief,
            "macro": self._macro_status(),
            "execution": self._execution_state(view),
            "task_success": "unknown_to_policy",
            "goal_contract": self._task_contract(),
            "model_completion_claims": self._model_completion_claims,
        })
        self.plan_lock._last_feedback = deepcopy(self._feedback)
        self.plan_lock._local_motion_context = {
            "execution_owner": "model", "measured_robot": measured,
            "capabilities": self._capabilities(),
            "unknown_state_blocks_direct_control": False}
        self.plan_lock._local_motion_feedback = deepcopy(self._feedback)
        self.plan_lock._local_motion_history = deepcopy(self._feedback_history)
        self.plan_lock._checkpoint = {}
        self.plan_lock._recovery_session = {}

    def _record_feedback(self, value):
        self._feedback = jsonable({"observed_at_step": self.step_index, **value})
        self._feedback_history.append(deepcopy(self._feedback))

    def _ask_agent(self, view, calls):
        self._sync_framework(view)
        started = self._clock()
        self._executive_calls += 1
        try:
            answer = self.client.request_agent(view, self._context(view))
        except SchemaError as exc:
            calls.append(ExecutiveCall("agent", max(0., self._clock() - started),
                                       getattr(self.client, "last_request_key", None),
                                       str(getattr(self.client, "last_log_path", "")) or None,
                                       False, "SchemaError"))
            self._record_feedback({"action": "model_response", "status": "rejected",
                                   "reason": "invalid_action_arguments", "detail": str(exc)[:1800],
                                   "requested_action_executed": False,
                                   "continuation": "Correct the arguments from this same physical state; preserve the current gripper command."})
            self._need_agent = True
            return
        except Exception as exc:
            calls.append(ExecutiveCall("agent", max(0., self._clock() - started),
                                       getattr(self.client, "last_request_key", None),
                                       str(getattr(self.client, "last_log_path", "")) or None,
                                       False, type(exc).__name__))
            raise
        calls.append(ExecutiveCall("agent", max(0., self._clock() - started),
                                   getattr(self.client, "last_request_key", None),
                                   str(getattr(self.client, "last_log_path", "")) or None))
        self._agent_decision = deepcopy(answer)
        self._belief = {"source": "model_estimate", "belief": answer["belief"],
                        "plan": answer["plan"], "step": self.step_index}
        self.plan_lock.record_decision(step=self.step_index, source="agent_executive",
                                       action=deepcopy(answer), belief_is_measurement=False)
        self._need_agent = False
        self._apply_agent(answer, view)

    def _abandon_macro(self, reason):
        if self._macro_active:
            self._macro_session.cancel(reason)
        self._macro_active = False
        self._macro_paused = False
        self._macro_item = None
        self._macro_reply = None
        self._macro_grant = 0
        self._macro_used = 0
        self._primitive = None
        self.ctx.monitor.clear("episode")
        self.ctx.active_relation = None

    def _apply_agent(self, answer, view):
        kind = answer["type"]
        if kind == "macro":
            selected = answer["macro"]
            index, grant = selected["step_index"], selected["budget_steps"]
            if type(index) is not int or not 0 <= index < len(self.plan_lock.plan()["steps"]):
                self._record_feedback({"action": "macro", "status": "rejected",
                                       "reason": "step_index_out_of_range", "executed_steps": 0})
                self._need_agent = True
                return
            if type(grant) is not int or not 1 <= grant <= MAX_MACRO_GRANT_STEPS:
                raise ValueError("Macro grant must contain 1 through 60 control steps")
            resumable = self._macro_active and self._macro_paused and self._macro_index == index
            if selected["continuation"]:
                if not resumable:
                    self._record_feedback({"action": "macro", "status": "rejected",
                                           "reason": "no_paused_macro_to_continue", "executed_steps": 0})
                    self._need_agent = True
                    return
            else:
                self._abandon_macro("model selected a new macro")
                self._macro_index = index
                self._macro_active = bool(self._macro_session.start(index))
                if not self._macro_active:
                    self._record_feedback({"action": "macro", "status": "unavailable",
                                           "reason": "native_macro_not_started",
                                           "evidence": self._macro_session.status(), "executed_steps": 0})
                    self._need_agent = True
                    return
            self._mode = "macro"
            self._macro_grant, self._macro_used = grant, 0
            self._macro_transition_feedback = None
            self._macro_grants += 1
            self._macro_paused = False
            return

        self._abandon_macro("model took direct control")
        if kind == "execute":
            self._mode = "fragment"
            self._fragment_grants += 1
            self._segment_start = {"step": self.step_index, "pose_wxyz": measured_pose(view.state),
                                   "finger_gap_mm": view.gap_mm, "action_type": kind}
            self._primitive = FragmentSequencePrimitive(
                view, answer["execution"],
                keep_gripper=self.ctx.memory["user_check_gripper_command"])
            if self._primitive.done:
                self._complete_direct()
            return
        self._mode = "direct"
        if kind == "motion":
            motion = answer["motion"]
        elif kind in ("observe", "finish"):
            motion = {"frame": "world", "position_mode": "relative", "xyz_m": [0., 0., 0.],
                      "quaternion_wxyz": None, "gripper": "keep", "steps": 1}
            if kind == "finish":
                self._model_completion_claims.append({
                    "step": self.step_index, "claim": answer["belief"],
                    "source": "model_estimate", "evaluator_success": "not_observed"})
        else:
            raise ValueError(f"Unknown agent action type {kind!r}")
        self._segment_start = {"step": self.step_index, "pose_wxyz": measured_pose(view.state),
                               "finger_gap_mm": view.gap_mm, "action_type": kind}
        self._primitive = LocalMotionPrimitive(
            view, motion, keep_gripper=self.ctx.memory["user_check_gripper_command"])
        if self._primitive.done:
            self._complete_direct()

    def _complete_direct(self):
        primitive = self._primitive
        outcome = primitive.outcome()
        if self._mode == "fragment":
            evidence = outcome["evidence"]
            self._last_fragment_result = {
                "observed_at_step": self.step_index,
                "status": evidence.get("status", outcome["status"]),
                "reason": outcome["reason"], "executed_steps": outcome["actions"],
                "evidence": evidence,
            }
            self._record_feedback({
                "action": "execute", **self._last_fragment_result,
                "start": self._segment_start,
                "semantic_result": "Measured local predicates only; grasp, support and task success require observation.",
                "continuation": "Fresh RGB and shared measured state are available. Select a new fragment sequence, direct adjustment or optional native macro; pending segments are not automatically resumed.",
            })
            self._primitive = None
            self._need_agent = True
            return
        self._direct_segments += 1
        self._record_feedback({
            "action": "direct_motion", "status": "segment_complete" if outcome["status"] == "succeeded"
            else "rejected", "reason": outcome["reason"],
            "start": self._segment_start, "measurement": outcome["evidence"],
            "executed_steps": outcome["actions"],
            "semantic_result": "unknown; inspect current and preceding RGB observations",
            "continuation": "model may perform any next action; no macro handoff certificate required"})
        self._primitive = None
        self._need_agent = True

    def _macro_yield(self, reason, *, end=False, evidence=None):
        status = "returned" if end else "paused"
        self._last_macro_feedback = jsonable({
            "action": "native_macro", "status": status, "reason": reason,
            "step_index": self._macro_index, "executed_steps": self._macro_used,
            "evidence": evidence, "session": self._macro_session.status(self._primitive),
            "last_transition": deepcopy(self._macro_transition_feedback),
            "semantic_result": "native controller evidence only; independent evaluator decides task success",
            "continuation": ("choose direct control, a new macro, or observe" if end else
                             "continue this macro explicitly or take direct control")})
        self._record_feedback(self._last_macro_feedback)
        if end:
            self._abandon_macro(reason)
            self._mode = "agent"
        else:
            self._macro_paused = True
        self._need_agent = True

    def _drain_agent_outcomes(self):
        pending = list(self.ctx.pending_outcomes)
        self.ctx.pending_outcomes.clear()
        records = []
        for outcome in pending:
            self._outcome_counter += 1
            row = outcome.record(self._outcome_counter, self.step_index, self._macro_index)
            records.append(row)
            self.outcomes.append(row)
            self.ledger.append({key: row.get(key) for key in (
                "n", "step", "relation", "status", "reason", "plan_step", "pids", "evidence_text")})
            bucket = self._relations_by_status.setdefault(row["relation"], {})
            bucket[row["status"]] = bucket.get(row["status"], 0) + 1
            self._last_outcome = row
        return records

    def _next_macro(self, view, calls, outcomes, pointing):
        """Advance only the selected macro, never a repair/replan loop."""
        for _ in range(base.MAX_TRANSITIONS_PER_ACT):
            preemption = self.ctx.monitor.peek_preemption()
            if preemption is not None:
                key, report = self.ctx.monitor.take_preemption()
                self._counts["preemptions"] += 1
                self._macro_yield("native_observer_preemption", end=True,
                                  evidence={"key": key, "report": report.to_dict()})
                return None
            if self._primitive is not None:
                if not self._primitive.done:
                    return self._primitive
                primitive, self._primitive = self._primitive, None
                result = primitive.outcome()
                if result["status"] != "succeeded":
                    self._macro_yield("native_primitive_failed", end=True, evidence=result)
                    return None
                self._macro_reply = (result, primitive)
            if self._macro_item is None:
                reply, self._macro_reply = self._macro_reply, None
                self._macro_item = self._macro_session.advance(reply)
                outcomes.extend(self._drain_agent_outcomes())
            item = self._macro_item
            if item is None:
                self._macro_yield("native_macro_returned", end=True,
                                  evidence=self._macro_session.status())
                return None
            if callable(item) and not isinstance(item, (PlanNeed, PointNeed, RepairNeed)):
                self._macro_item = None
                self._primitive = item(view)
                continue
            if isinstance(item, RepairNeed):
                self._macro_yield("native_repair_requires_executive_decision", end=True,
                                  evidence={"kind": item.kind, "last_outcome": item.last_outcome})
                return None
            if isinstance(item, (PlanNeed, PointNeed)):
                if sum(call.kind != "agent" for call in calls) >= MAX_NATIVE_CALLS_PER_ACT:
                    return None  # preserve the pending Need across one observed pose hold
                self._macro_item = None
                self._macro_reply = self.fulfiller.fulfil(item, view, calls)
                if isinstance(item, PointNeed):
                    pointing.append(self._pointing(item, self._macro_reply))
                continue
            raise TypeError(f"Unsupported native macro yield {type(item).__name__}")
        return None

    def act(self, observation):
        self.plan_lock.check_active()
        if self._pending is not None:
            raise RuntimeError("observe_transition must acknowledge the pending action")
        if self._expected_observation is not None and observation != self._expected_observation:
            raise ValueError("act requires the observation from the preceding transition")
        view = self._view(observation)
        if not self._client_checked:
            check = getattr(self.client, "check_observation", None)
            if callable(check):
                check(observation)
            self._client_checked = True
        self.ctx.view, self.ctx.step = view, self.step_index
        calls, outcomes, pointing = [], self._drain_agent_outcomes(), []
        acting = None
        # At most one executive answer per act. A rejected answer gets an
        # explicit observation hold; the next call sees its measured feedback.
        # No episode, retry or repair budget is introduced here.
        for _ in range(base.MAX_TRANSITIONS_PER_ACT):
            if self._mode == "macro" and self._macro_active and not self._macro_paused \
                    and self._macro_used >= self._macro_grant:
                self._macro_yield("model_selected_action_budget")
            if self._need_agent:
                if any(call.kind == "agent" for call in calls):
                    break
                self._ask_agent(view, calls)
                if self._need_agent:
                    continue
            if self._mode == "macro" and self._macro_active:
                acting = self._next_macro(view, calls, outcomes, pointing)
                if self._need_agent:
                    continue
                break
            if self._primitive is not None:
                if self._primitive.done:
                    self._complete_direct()
                    continue
                acting = self._primitive
                break
            self._need_agent = True
        if acting is None:
            if not self._macro_active:
                self._mode = "direct"
            acting = LocalMotionPrimitive(view, {
                "frame": "world", "position_mode": "relative", "xyz_m": [0., 0., 0.],
                "quaternion_wxyz": None, "gripper": "keep", "steps": 1},
                keep_gripper=self.ctx.memory["user_check_gripper_command"])
            self._counts["hold_actions"] += 1
        action = tuple(float(v) for v in acting.next_action(view))
        if calls:
            self._counts["acts_with_calls"] += 1
        phase = self._phase()
        self._steps_by_phase[phase] = self._steps_by_phase.get(phase, 0) + 1
        macro = self._macro_label()
        self._macro_steps = self._macro_steps + 1 if macro == self._macro else 1
        self._macro = macro
        last_call = calls[-1] if calls else None
        diagnostics = jsonable({
            "step": self.step_index, "controller": CONTROLLER,
            "macro": macro, "primitive": acting.name,
            "execution_kind": ("framework_fragment:" if self._mode == "fragment" else
                               "native_macro:" if self._mode == "macro" else "agent_owned:") + acting.name,
            "macro_step": self._macro_steps, "refreshed": bool(calls),
            "routing": {"kind": "agent", "skill": None if self._agent_decision is None
                        else self._agent_decision["type"], "params": self._agent_decision,
                        "situation": self._belief["belief"]},
            "api_calls_in_act": [call.kind for call in calls],
            "api_latency_s": sum(call.latency_s for call in calls) if calls else None,
            "api_request_key": None if last_call is None else last_call.request_key,
            "api_request_log": None if last_call is None else last_call.request_log,
            "api_calls": [{"kind": call.kind, "latency_s": call.latency_s, "ok": call.ok}
                          for call in calls],
            "pointing": pointing, "outcomes": outcomes,
            "outcome": outcomes[-1] if outcomes else None,
            "v2_events": self._events.drain(), "relation": self._relation(),
            "observers": self.ctx.monitor.snapshot(), "parts_uv": {},
            "holding": self.holding, "finger_gap_mm": view.gap_mm,
            "calibration_probe": self._mode == "macro" and
                getattr(self._macro_session.runner, "phase", None) == "calibrate",
            "pending_need": None if self._macro_item is None else getattr(self._macro_item, "kind", None),
            "history_stats": self.client.history_stats(),
            "agent_belief": self._belief, "update_steps": 1, "feedback": self._feedback,
        })
        decision = Decision(action, diagnostics)
        self._pending = (observation, decision, view, acting)
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        # Retain exact native action acknowledgement and measured monitor
        # updates. Direct control has no stale native observers after handoff.
        result = super().observe_transition(before, decision, after)
        view = self._view(after)
        grip = float(decision.action[6])
        if grip:
            self.ctx.memory["user_check_gripper_command"] = "close" if grip > 0 else "open"
        if self._mode == "macro" and self._macro_active:
            self._native_actions += 1
            self._macro_used += 1
            self._last_macro_feedback = deepcopy(result)
            self._macro_transition_feedback = deepcopy(result)
        else:
            if self._mode == "fragment":
                self._fragment_actions += 1
            else:
                self._direct_actions += 1
            self._macro_session.invalidate_after_direct(view, gripper_command=
                self.ctx.memory["user_check_gripper_command"])
            if self._primitive is not None and self._primitive.done:
                self._complete_direct()
        self.ctx.view, self.ctx.step = view, self.step_index
        self._sync_framework(view)
        result["execution_owner"] = "model"
        result["model_belief_is_measurement"] = False
        result["task_success"] = "not_available_to_policy"
        return result

    def _phase(self):
        if getattr(self, "_mode", None) == "fragment":
            return "framework_fragment"
        if getattr(self, "_mode", None) == "macro" and getattr(self, "_macro_active", False):
            return "agent_macro_paused" if self._macro_paused else "agent_macro"
        return "agent_direct" if getattr(self, "_mode", None) == "direct" else "agent_decision"

    def _macro_label(self):
        if self._mode == "fragment":
            return "framework_fragment"
        if self._mode == "macro" and self._macro_index is not None:
            step = self.plan_lock.plan()["steps"][self._macro_index]
            return "optional:" + step["do"] + "(" + step["obj"] + ")"
        return "agent_direct"

    def _relation(self):
        step = None
        if self._mode == "macro" and self._macro_index is not None:
            step = self.plan_lock.plan()["steps"][self._macro_index]
        return {"step_i": self._macro_index if step else None,
                "do": None if step is None else step["do"],
                "pids": None if step is None else {"obj": step["obj"], "grip": step["grip"], "ref": step["ref"]},
                "phase": self._phase(), "relation": getattr(self.ctx, "active_relation", None)}

    @property
    def relation_plan(self):
        return self.plan_lock.plan()

    def agent_stats(self):
        result = super().agent_stats()
        result.update(policy_version=POLICY_VERSION, execution_owner="model",
                      phase=self._phase(), model_belief=deepcopy(self._belief),
                      last_feedback=deepcopy(self._feedback),
                      optional_macro=self._macro_status(),
                      user_check_framework=self.plan_lock.framework_snapshot())
        result["counts"].update(agent_decisions=self._executive_calls,
                               direct_segments=self._direct_segments,
                               fragment_grants=self._fragment_grants,
                               fragment_actions=self._fragment_actions,
                               direct_actions=self._direct_actions,
                               native_macro_actions=self._native_actions,
                               optional_macro_grants=self._macro_grants)
        result["counts"]["calls_by_need"]["agent"] = self._executive_calls
        result["versions"]["agent_owned"] = POLICY_VERSION
        return result
