"""Measured, task-independent local recovery for the reviewed v2 executor.

Motion is a repair action, not a new task relation. Each short segment uses the
normal primitive acknowledgement/invariant path, then returns control to the
model with fresh images. A completed segment never proves a task goal or grasp.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from . import agent_speedup_v2 as base
from .agent_speedup_v2_user_check_motion import (
    HandoffAnswer, LocalMotionAnswer, LocalMotionPrimitive, motion_context, parse_reviewed_repair,
)


def current_motion_context(ctx, view):
    value = motion_context(view)
    memory = getattr(ctx, "memory", {})
    value.update(observed_at_step=int(getattr(ctx, "step", 0)),
                 last_gripper_command=memory.get("user_check_gripper_command", "open"),
                 grasp_state=deepcopy(memory.get("user_check_grasp_state") or {
                     "status": "verified_held" if getattr(ctx, "holding", None) is not None else "empty",
                     "part": getattr(getattr(ctx, "holding", None), "pid", None)}))
    value["recovery"] = deepcopy(memory.get("user_check_recovery"))
    value["checkpoint"] = deepcopy(memory.get("user_check_phase_checkpoint"))
    return value


class LocalRecoveryFulfiller(base.NeedFulfiller):
    """Keep native accounting and response records, extending only repair parsing."""

    def _plan(self, need, view, calls):
        # Keep native transport/accounting while parsing the variant's reviewed
        # spatial contract instead of silently dropping it at the Need boundary.
        from types import FunctionType
        from .agent_speedup_v2_user_check_plan import parse_plan
        method = base.NeedFulfiller._plan
        namespace = dict(method.__globals__, parse_plan=parse_plan)
        return FunctionType(method.__code__, namespace, method.__name__, method.__defaults__,
                            method.__closure__)(self, need, view, calls)

    def _repair(self, need, view, calls):
        lock = self.plan_lock
        lock._local_motion_context = current_motion_context(self.ctx, view)
        extra = {}
        if isinstance(need, base.MarkedRepairNeed):
            extra = {"reask_error": need.reask_error, "marker_uv": need.marker_uv}
            if need.step_history is not None:
                extra["step_history"] = need.step_history
        raw, error, started = self._call(
            need.kind, self.client.request_repair, calls,
            view.external_rgb_upright, view.wrist_rgb_upright,
            quarter_turns=need.quarter_turns, status_block=need.status_block,
            last_outcome=need.last_outcome, holding=need.holding,
            review=need.kind == "review", **extra)
        answer = None
        if raw is not None:
            try:
                answer = parse_reviewed_repair(raw)
            except base.PlanParseError:
                error = "parse_error"
                self.errors["parse_error"] += 1
        if error is not None:
            self._reject(need.kind)
        call = self._record(need.kind, started, calls, error is None, error)
        return base.RepairReply(need, None if raw is None else dict(raw), answer, error is not None, call)


class LocalRecoveryRunnerMixin:
    """Extend the repair boundary with model-selected short motion segments."""

    def __init__(self, ctx, **kwargs):
        self._local_changed_since_failure = False
        self._local_recovery_verified = False
        self._checkpoint_active = False
        self._last_verified_goal = None
        self._local_grasp_candidate = None
        super().__init__(ctx, **kwargs)
        ctx.memory["user_check_checkpoint"] = self.motion_checkpoint

    def repair_issues(self, reply, failed_i, *, review, failed=None, outcome=None):
        if not reply.schema_error and isinstance(reply.answer, LocalMotionAnswer):
            return []  # numeric/frame validation already ran in the variant parser
        if not reply.schema_error and isinstance(reply.answer, HandoffAnswer):
            if reply.answer.action == "verify_grasp":
                return self._grasp_request_issues(reply.answer, failed_i, failed)
            return self._resume_issues(failed_i, failed)
        if self._unknown_local_grasp() and reply.answer is not None and reply.answer.action != "hold_still":
            return [{"code": "skill_precondition", "step": failed_i, "pid": None,
                     "text": "Local gripper contact is unverified. Continue local control; "
                             "use verify_grasp to measure identity/load, or establish measured opening."}]
        if reply.answer is not None and reply.answer.action == "retry":
            blocked = self._resume_issues(failed_i, failed)
            if blocked:
                return blocked
        issues = super().repair_issues(reply, failed_i, review=review, failed=failed, outcome=outcome)
        if self._local_recovery_verified:
            # A measured subgoal, rather than any tiny motion, permits another
            # macro attempt. Its readiness is checked again on entry.
            issues = [issue for issue in issues if getattr(issue, "code", None) != "retry_unchanged"]
        return issues

    def _grasp_request_issues(self, answer, failed_i, failed):
        state = failed or self.active_state
        step = getattr(state, "step", None)
        parts = {part.pid: part for part in self.plan.parts} if self.plan is not None else {}
        reason = None
        if answer.part not in parts or answer.step["grip"] not in parts:
            reason = "verify_grasp must name reviewed object and grip IDs."
        elif step is None or answer.part != step.obj:
            reason = "verify_grasp must name the current macro's object."
        elif step.do == "turn" or getattr(parts[answer.part], "move", None) in ("fixed", "hinged", "sliding"):
            reason = ("A fixture grip requires its native motion/contact verifier, not a free-object lift. "
                      "Measure an opening, then resume the fixture macro to re-establish its grasp.")
        elif answer.step["grip"] not in tuple(self.grip_of(state) or (step.obj,)):
            reason = "verify_grasp grip must be an active grip candidate; use the plan repair to change candidates."
        elif self.ctx.memory.get("user_check_gripper_command") != "close":
            reason = "verify_grasp requires an actually executed close command."
        elif getattr(self.ctx, "holding", None) is not None and self.ctx.holding.pid != answer.part:
            reason = "The measured held object has another identity."
        return [] if reason is None else [{"code": "skill_precondition", "step": failed_i,
                                           "pid": answer.part, "text": reason}]

    def _resume_issues(self, failed_i, failed):
        ctx = self.ctx
        issues = []
        if failed is None and not self._checkpoint_active:
            issues.append({"code": "no_active_macro", "step": None, "pid": None,
                           "text": "No macro is paused. Finish the goal review with hold_still or request a plan repair."})
        recovery = ctx.memory.get("user_check_recovery") or {}
        if recovery and recovery.get("status") != "verified":
            issues.append({"code": "recovery_goal_unmet", "step": failed_i, "pid": None,
                           "text": "The measured local recovery subgoal is not met; inspect its residual and "
                                   "continue local motion, observe, or choose another recovery."})
        if (recovery.get("goal", {}).get("kind") == "observe" and not self._checkpoint_active
                and not recovery.get("measurement", {}).get("prerequisite_changed")):
            issues.append({"code": "observation_only", "step": failed_i, "pid": None,
                           "text": "A new view alone does not resolve the previous blocker. Request fresh geometry "
                                   "or a measured pose/height recovery before retrying the macro."})
        if self._unknown_local_grasp():
            issues.append({"code": "skill_precondition", "step": failed_i, "pid": None,
                           "text": "Grasp identity/load is unverified; verify_grasp or measure a full opening."})
        snapshot = ctx.monitor.snapshot()
        violations = {key: value for key, value in snapshot.items()
                      if value.get("preempt") and value.get("verdict") == "violated"}
        if violations:
            issues.append({"code": "active_constraint_violation", "step": failed_i, "pid": None,
                           "text": "Measured invariants still block macro handoff: " + ", ".join(violations)[:160]})
        # The controller performs its complete skill preconditions on resume;
        # no model-provided success flag is accepted here.
        return issues

    def _unknown_local_grasp(self):
        return (getattr(self.ctx, "memory", {}).get("user_check_grasp_state") or {}).get("status") == "unknown"

    def hold_factory(self):
        if not self._unknown_local_grasp():
            return super().hold_factory()
        gripper = self.ctx.memory.get("user_check_gripper_command", "close")
        return lambda view: base.HoldPrimitive(view, steps=base.HOLD_RECHECK_STEPS,
                                               gripper=gripper, name=base.HOLD_PRIMITIVE_NAME)

    def ask(self, kind, failed, outcome, **kwargs):
        self._local_changed_since_failure = False
        self._local_recovery_verified = False
        if not self._checkpoint_active:
            pending = self.ctx.memory.pop("user_check_checkpoint_pending", None)
            if pending is not None:
                return pending
        continuing = False
        while True:
            if self.ctx.view is not None:
                self.plan_lock._local_motion_context = current_motion_context(self.ctx, self.ctx.view)
            used = self.reviews_used
            answer, info = yield from super().ask(kind, failed, outcome, **kwargs)
            if continuing and kind == "review":
                # These are closed-loop segments within the same review, not
                # additional goal reviews. Provider calls still count normally.
                self.reviews_used = used
            if isinstance(answer, HandoffAnswer):
                if answer.action == "verify_grasp":
                    applied = yield from self.verify_local_grasp(answer, failed)
                    self._emit(kind, **info, applied=applied)
                    kwargs["marker_uv"] = None
                    if failed is not None:
                        kwargs["history"] = self.step_history(failed)
                    continuing = True
                    continue
                self.ctx.memory.pop("user_check_recovery", None)
                self.plan_lock._local_motion_context = current_motion_context(self.ctx, self.ctx.view)
                self._sync_framework()
                if self._checkpoint_active:
                    return answer, info
                # Native retry reuses a verified held frame and re-enters the
                # macro's remaining measured prerequisites.
                return base.RepairAnswer(answer.why, "retry", None, None), info
            if not isinstance(answer, LocalMotionAnswer):
                if answer is not None and answer.action not in ("hold_still", "retry"):
                    self.ctx.memory.pop("user_check_recovery", None)
                return answer, info
            applied = yield from self.execute_local_motion(answer, failed, kind=kind)
            self._emit(kind, **info, applied=applied)
            # The marked pixel belongs to the old camera pose. The new request
            # uses current images and measured segment feedback instead.
            kwargs["marker_uv"] = None
            if failed is not None:
                kwargs["history"] = self.step_history(failed)
            continuing = True

    def motion_checkpoint(self, phase, step, evidence=None):
        """Suspend a macro at a physical phase, then explicitly hand control back.

        Clean grasp checkpoints are recorded without a model call. Alignment
        and uncertain placement boundaries use fresh images and the same short
        motion protocol as failures. The paused macro owns recomputation.
        """
        ctx = self.ctx
        evidence = base.jsonable(deepcopy(evidence or {}))
        checkpoint = {"phase": phase, "observed_at_step": int(ctx.step),
                      "step_i": self.index_of(self.active_state), "obj": getattr(step, "obj", None),
                      "ref": getattr(step, "ref", None), "evidence": evidence}
        ctx.memory["user_check_phase_checkpoint"] = checkpoint
        self.plan_lock._local_motion_context = current_motion_context(ctx, ctx.view)
        grasp_flags = evidence.get("verification_flags") or evidence.get("flags") or []
        if phase == "grasp_confirmed" and ctx.holding is not None and not any(
                flag in grasp_flags for flag in ("comotion_unverified", "rotation_unmeasured")):
            ctx.memory["user_check_grasp_state"] = {
                "status": "verified_held", "part": ctx.holding.pid, "source": "macro_load_test"}
            ctx.memory.pop("user_check_phase_checkpoint", None)
            self.plan_lock._local_motion_context = current_motion_context(ctx, ctx.view)
            return {"resume": True, "changed": False, "action": "measured_grasp",
                    "reason": "native_grasp_verified", "evidence": evidence}
        if phase == "grasp_confirmed" and "comotion_unverified" in grasp_flags and ctx.holding is not None:
            self._quarantine_grasp(ctx.holding, "object_identity_unverified")
        from .agent_speedup_v2_user_check_framework import ExecutionFeedback
        self._last_feedback = ExecutionFeedback(
            status="pending_verification", skill=getattr(step, "do", "verify"),
            reason="skill_checkpoint", executed=False,
            unmet_conditions=({"condition": "checkpoint_handoff", "phase": phase},),
            evidence={"checkpoint": checkpoint, "grasp_state": deepcopy(ctx.memory.get("user_check_grasp_state"))},
            policy_step=int(ctx.step)).to_dict()
        self._sync_framework()
        previous_phase = self.phase
        started = int(ctx.step)
        self._checkpoint_active = True
        try:
            answer, info = yield from self.ask("repair", self.active_state, None)
            resume = isinstance(answer, HandoffAnswer) and answer.action == "resume_macro"
            if not resume:
                # A plan edit is applied by the ordinary runner after this
                # controller yields a blocked outcome, never inside its stack.
                self.ctx.memory["user_check_checkpoint_pending"] = (answer, info)
            result = {"resume": resume, "changed": int(ctx.step) != started,
                      "action": None if answer is None else answer.action,
                      "reason": "measured_handoff" if resume else "checkpoint_repair_requested",
                      "evidence": {"phase": phase, "started_at_step": started,
                                   "resumed_at_step": int(ctx.step), "holding": self.holding_pid()}}
            self._emit("repair", **info, applied={"action": "checkpoint", "result": result})
            return result
        finally:
            self._checkpoint_active = False
            self.phase = previous_phase
            ctx.memory.pop("user_check_phase_checkpoint", None)

    def verify_local_grasp(self, answer, failed):
        """Adopt a local grasp only after object-specific measured co-motion."""
        ctx = self.ctx
        obj, grip = answer.part, answer.step["grip"]
        if (ctx.holding is not None and ctx.holding.pid == obj and ctx.holding.grip_pid == grip
                and not self._unknown_local_grasp()):
            result = {"ok": True, "reason": "already_verified", "evidence": {"comotion_decided": True},
                      "held": ctx.holding}
        elif self._local_grasp_candidate is not None and (
                self._local_grasp_candidate["held"].pid,
                self._local_grasp_candidate["held"].grip_pid) == (obj, grip):
            result = yield from self._verify_grasp_candidate()
        elif not callable(getattr(self.relations, "_load_test", None)):
            result = {"ok": False, "reason": "measured_grasp_verifier_unavailable", "evidence": {}}
        else:
            # The regular load test uses robot calibration, current images,
            # the object's registration and a short measured lift. Its native
            # permissive pending-comotion result is NOT sufficient for taking
            # identity from a raw model close.
            result = yield from self.relations._load_test(ctx, obj=obj, grip=grip, attempt=1)
        evidence = deepcopy(result.get("evidence") or {})
        held = result.get("held")
        verified = bool(result.get("ok") and evidence.get("comotion_decided") is True and held is not None
                        and getattr(held, "pid", None) == obj and getattr(held, "grip_pid", None) == grip)
        if verified:
            ctx.holding = held
            ctx.scene.held = held
            ctx.grasp_quaternion = tuple(float(v) for v in ctx.view.quaternion_xyzw)
            ctx.hold_gap_ref_mm = float(ctx.view.gap_mm)
            step = getattr(failed or self.active_state, "step", None)
            self.relations._moved_by_contact(ctx, obj, grip, "close")
            evidence["bus"] = self.relations._register_held_bus(
                ctx, obj=obj, grip=grip, purpose="carry", keep=getattr(step, "keep", ()), held=held)
            ctx.memory["user_check_grasp_state"] = {
                "status": "verified_held", "part": obj, "grip": grip,
                "source": "measured_load_and_object_comotion", "observed_at_step": int(ctx.step)}
            from .agent_speedup_v2_user_check_spatial import refresh_whole_object_frame
            evidence["whole_object_frame"] = refresh_whole_object_frame(ctx, held)
            self._local_grasp_candidate = None
            # The measured lift consumes the approach/contact subgoal. Its
            # verified held frame becomes the next macro's current prerequisite.
            evidence["consumed_recovery"] = deepcopy(ctx.memory.pop("user_check_recovery", None))
            self._local_recovery_verified = True
        else:
            self._quarantine_grasp(held, result.get("reason") if not result.get("ok") else "object_identity_unverified",
                                   obj=obj, grip=grip)
        feedback = {"action": "verify_grasp", "status": "verified" if verified else "unverified",
                    "reason": result.get("reason") if verified or not result.get("ok") else "object_identity_unverified",
                    "observed_at_step": int(ctx.step), "grasp_effect": deepcopy(ctx.memory["user_check_grasp_state"]),
                    "task_effect": "unverified", "evidence": evidence}
        self.plan_lock._local_motion_feedback = feedback
        self.plan_lock._local_motion_history.append(deepcopy(feedback))
        self.plan_lock._local_motion_context = current_motion_context(ctx, ctx.view)
        self._sync_framework()
        return {"action": "verify_grasp", "part": obj, "step_i": self.index_of(failed),
                "status": feedback["status"], "result": feedback}

    def _quarantine_grasp(self, held, reason, *, obj=None, grip=None):
        """Retain a candidate for measurement without publishing a held fact."""
        ctx = self.ctx
        if held is not None:
            local = getattr(ctx.scene, "_held_local", {})
            if not local and self._local_grasp_candidate is not None and self._local_grasp_candidate["held"] is held:
                local = self._local_grasp_candidate["local"]
            self._local_grasp_candidate = {"held": held, "local": deepcopy(local)}
        elif self._local_grasp_candidate is not None and self._local_grasp_candidate["held"].pid != obj:
            self._local_grasp_candidate = None
        ctx.holding = None
        ctx.scene.held = None
        ctx.scene._held_local = {}
        ctx.monitor.clear("held")
        ctx.memory.pop("gripped", None)
        ctx.memory["user_check_grasp_state"] = {
            "status": "unknown", "part": obj or getattr(held, "pid", None),
            "grip": grip or getattr(held, "grip_pid", None), "reason": reason}

    def _verify_grasp_candidate(self):
        """Measure co-motion again without inventing another close-time frame."""
        ctx = self.ctx
        candidate = self._local_grasp_candidate
        held = candidate["held"]
        # These transforms are visible only while the measurement is active;
        # ctx.holding remains unknown until the complete result is accepted.
        ctx.scene.held = held
        ctx.scene._held_local = deepcopy(candidate["local"])
        key = "user_check_local_grasp_identity"
        evidence = {}
        try:
            try:
                observer = self.relations._comotion(ctx, held, held.pid, rotation_verdict=True)
            except (ValueError, TypeError) as exc:
                return {"ok": False, "reason": "object_identity_unmeasured", "held": held,
                        "evidence": {"comotion_error": str(exc)}}
            self.relations._bus_add(ctx, key, observer, preempt=False, scope="relation")
            threshold = self.relations._g6(ctx)
            margin = float(threshold.value) - float(ctx.cal.gripper.empty_floor_mm)
            result, _ = yield from self.relations._run(lambda view: self.relations.VerifyHold(
                view, axis=self.relations.WORLD_UP, travel_m=self.relations.LIFT_TEST_M,
                calibration=ctx.cal.gripper, hold_margin_mm=margin, name="local_grasp_identity"))
            report = ctx.monitor.snapshot().get(key) or {}
            evidence = {"verify_hold": result, "comotion": report,
                        "comotion_decided": report.get("verdict") == "met"}
            ok = result.get("reason") == "hold_pass" and evidence["comotion_decided"]
            return {"ok": ok, "reason": "hold_pass" if ok else (report.get("reason") or "object_identity_unverified"),
                    "held": held, "evidence": evidence}
        finally:
            ctx.monitor.remove(key)
            if not evidence.get("comotion_decided"):
                ctx.scene.held = None
                ctx.scene._held_local = {}

    def _refresh_local_stays(self):
        ctx = self.ctx
        self._stays_observed_at = int(ctx.step)
        self.last_stays = dict(ctx.stays.check_all(ctx.view))
        for key, report in self.last_stays.items():
            self._emit("stays_check", key=key, verdict=report.verdict, reason=report.reason,
                       source="local_motion_boundary")
            owner = self._owner.get(key)
            if owner is not None and owner.status in base.STAYS_HOLDING_STATUSES and report.verdict != "met":
                # Unknown is not completion, including after a model-directed
                # correction. The macro must establish its effect again.
                owner.status, owner.reason = "pending", "kept_relation_lost"

    def _begin_recovery_goal(self, motion, before, failed):
        ctx = self.ctx
        old = ctx.memory.get("user_check_recovery")
        proposed = motion.get("recovery_goal")
        if old and proposed is None:
            return
        goal = deepcopy(proposed)
        if goal is None:
            target = np.asarray(motion["xyz_m"], dtype=float)
            if motion["position_mode"] == "relative":
                target = np.asarray(before.position, dtype=float) + target
            goal = {"kind": "pose", "xyz_m": target.tolist(), "height_m": None, "tolerance_m": .005}
        # Repeated segments for the same absolute subgoal preserve the measured
        # starting residual and cannot manufacture a new progress event.
        if old and old.get("goal") == goal:
            return
        session = {"goal": goal, "status": "pending", "started_at_step": int(ctx.step),
                   "step_i": self.index_of(failed), "segments": 0,
                   "orientation_wxyz": deepcopy(motion.get("quaternion_wxyz")),
                   "observation_baseline": {"xyz_m": np.asarray(before.position).tolist(),
                       "rotation": np.asarray(before.R).tolist(), "geometry": self._geometry_evidence(),
                       "registration_steps": {pid: getattr(record, "registration_step", None)
                           for pid, record in self.ctx.scene.parts.items()}}}
        ctx.memory["user_check_recovery"] = session
        initial = self._measure_recovery_goal(before, session)
        session["initial_residual_m"] = initial.get("residual_m")
        session["initially_met"] = bool(initial["met"])

    def _measure_recovery_goal(self, view, session):
        goal = session["goal"]
        tolerance = float(goal.get("tolerance_m") or .005)
        kind = goal["kind"]
        measured = {"kind": kind, "tolerance_m": tolerance, "observed_at_step": int(self.ctx.step)}
        if kind == "pose":
            residual = float(np.linalg.norm(np.asarray(view.position) - np.asarray(goal["xyz_m"])))
            met = residual <= tolerance
            orientation = session.get("orientation_wxyz")
            if orientation is not None:
                from .astranomy_agentic_motion import rotation_error_rad
                angle = float(np.linalg.norm(rotation_error_rad(orientation, view.state[3:6])))
                measured["orientation_residual_rad"] = angle
                met = met and angle <= .05
            measured.update(residual_m=residual, met=bool(met))
        elif kind == "height":
            residual = max(0., float(goal["height_m"]) - float(view.position[2]))
            measured.update(residual_m=residual, met=residual <= tolerance, measured_height_m=float(view.position[2]))
        elif kind == "open_hand":
            calibration = getattr(getattr(self.ctx, "cal", None), "gripper", None)
            open_gap = getattr(calibration, "g_open_mm", getattr(calibration, "open_gap_mm", None))
            measured["met"] = bool(self.ctx.memory.get("user_check_gripper_command") == "open"
                                   and open_gap is not None
                                   and view.gap_mm >= float(open_gap) - getattr(self.relations, "OPEN_TOLERANCE_MM", 1.))
            measured["gap_mm"] = float(view.gap_mm)
        else:
            baseline = session["observation_baseline"]
            translation = float(np.linalg.norm(np.asarray(view.position) - np.asarray(baseline["xyz_m"])))
            rotation = float(np.linalg.norm(np.asarray(view.R) - np.asarray(baseline["rotation"])))
            registered = any(getattr(record, "registration_step", None) != baseline["registration_steps"].get(pid)
                             for pid, record in self.ctx.scene.parts.items())
            geometry_changed = registered and self._geometry_evidence() != baseline["geometry"]
            view_changed = translation > .001 or rotation > .01
            acquired = registered or (int(self.ctx.step) > session["started_at_step"] and view_changed)
            measured.update(met=bool(acquired), observation_acquired=bool(acquired),
                            view_translation_m=translation, view_rotation_matrix_delta=rotation,
                            new_registration=bool(registered), prerequisite_changed=bool(geometry_changed),
                            blocker_resolved=False)
        return measured

    def _geometry_evidence(self):
        # Measurement values, not timestamps or prose, establish whether a
        # registration changed a prerequisite for a failed macro.
        geometry = base.jsonable({pid: {"status": getattr(record, "status", None),
            "xyz": getattr(record, "xyz", None), "cov": getattr(record, "cov", None),
            "region_xy": getattr(record, "region_xy", None),
            "registered_empty_hand": getattr(record, "registered_empty_hand", None)}
            for pid, record in self.ctx.scene.parts.items()})
        return hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _record_recovery_goal(self, failed, result):
        session = self.ctx.memory["user_check_recovery"]
        session["segments"] += 1
        measured = self._measure_recovery_goal(self.ctx.view, session)
        session["measurement"] = measured
        session["status"] = "verified" if measured["met"] else "pending"
        session["observed_at_step"] = int(self.ctx.step)
        physical_prerequisite = session["goal"]["kind"] != "observe" or measured.get("prerequisite_changed")
        if measured["met"] and result.get("status") == "succeeded" and physical_prerequisite:
            self._local_recovery_verified = True
            signature = json.dumps({"goal": session["goal"], "step_i": session["step_i"]}, sort_keys=True)
            if failed is not None and not session["initially_met"] and signature != self._last_verified_goal:
                self._restart_counts(self._ladder_of(self.step_key(failed.step)), "local_recovery_subgoal_verified")
                self._last_verified_goal = signature
        return deepcopy(session)

    def _reconcile_local_contact(self, motion, result, primitive, failed, *, previous_command=None):
        ctx = self.ctx
        actions = int(result.get("actions") or 0)
        if not actions:
            return
        memory = ctx.memory
        command = getattr(primitive, "gripper_command", memory.get("user_check_gripper_command", "open"))
        memory["user_check_gripper_command"] = command
        already_unknown = self._unknown_local_grasp()
        contact_change = (motion["gripper"] == "close" and ctx.holding is None) or (
            motion["gripper"] == "open" and (
                ctx.holding is not None or getattr(ctx.scene, "held", None) is not None
                or already_unknown or previous_command != "open"))
        if contact_change:
            # An intentional opening invalidates the old grasp before any
            # claim of an empty hand. Closing without a known held frame never
            # creates a synthetic held object from the model's target.
            memory["user_check_grasp_state"] = {
                "status": "unknown", "part": getattr(ctx.holding, "pid", None),
                "reason": "direct_gripper_command_needs_measured_reconciliation"}
            ctx.holding = None
            memory.pop("gripped", None)
        unknown = self._unknown_local_grasp()
        if unknown:
            # A full measured opening establishes an empty hand. The calibrated
            # endpoint is a robot property, independent of task/object names.
            gripper = getattr(getattr(ctx, "cal", None), "gripper", None)
            g_open = getattr(gripper, "g_open_mm", None)
            if g_open is None:
                g_open = getattr(gripper, "open_gap_mm", None)
            tolerance = getattr(self.relations, "OPEN_TOLERANCE_MM", 1.0)
            empty = command == "open" and g_open is not None and ctx.view.gap_mm >= float(g_open) - tolerance
            if empty:
                self._local_grasp_candidate = None
                if getattr(ctx.scene, "held", None) is not None:
                    ctx.scene.release_held(ctx.view)
                memory["user_check_grasp_state"] = {"status": "empty", "part": None,
                                                     "source": "measured_full_open"}
                # Raw local contact may concern any visible movable object;
                # it cannot refresh a particular object's registration by fiat.
                for pid, record in list(ctx.scene.parts.items()):
                    if getattr(record, "move_prior", None) == "free":
                        ctx.scene.drop(pid)
                        self._awaiting.add(pid)
                memory.pop("plans", None)
            ctx.monitor.clear("held")
        if contact_change or unknown:
            ctx.scene.invalidate_views()

    def execute_local_motion(self, answer, failed, *, kind):
        ctx = self.ctx
        before = ctx.view
        command = ctx.memory.get("user_check_gripper_command", "close" if ctx.holding is not None else "open")
        self._begin_recovery_goal(answer.step, before, failed)
        decision = self.plan_lock.record_decision(
            source="model_local_motion", status="applying", step=int(ctx.step), answer=answer.to_json(),
            failed_step=self.index_of(failed), holding=self.holding_pid())
        def factory(view):
            primitive = LocalMotionPrimitive(view, answer.step, keep_gripper=command)
            if not primitive.done and answer.step["gripper"] == "open":
                # Deliberate opening must not be preempted as accidental hold
                # loss. An invalid target must not alter observer state.
                ctx.monitor.clear("held")
            return primitive
        previous_phase = ctx.memory.get("user_check_contact_phase")
        ctx.memory["user_check_contact_phase"] = "local_recovery"
        try:
            reply = yield factory
        finally:
            if previous_phase is None:
                ctx.memory.pop("user_check_contact_phase", None)
            else:
                ctx.memory["user_check_contact_phase"] = previous_phase
        result, primitive = reply
        self._reconcile_local_contact(answer.step, result, primitive, failed, previous_command=command)
        reason = str(result.get("reason") or "")
        if reason.startswith(base.PREEMPTED_PREFIX):
            evidence = result.get("evidence") or {}
            # A violated held observer means its identity is no longer trusted.
            if "hold" in reason or "grip" in reason:
                ctx.memory["user_check_grasp_state"] = {"status": "unknown", "part": self.holding_pid(),
                                                         "reason": reason}
                ctx.holding = None
                ctx.memory.pop("gripped", None)
        delta = np.asarray(ctx.view.position) - np.asarray(before.position)
        changed = bool(np.linalg.norm(delta) > 1e-5 or
                       np.linalg.norm(ctx.view.R - before.R) > 1e-4 or
                       abs(ctx.view.gap_mm - before.gap_mm) > .01)
        self._local_changed_since_failure |= changed
        subgoal = self._record_recovery_goal(failed, result)
        feedback = {"action": "local_motion", "status": "executed" if result.get("status") == "succeeded" else "blocked",
                    "reason": reason, "observed_at_step": int(ctx.step),
                    "executed_steps": int(result.get("actions") or 0),
                    "measured_translation_m": delta.tolist(), "state_changed": changed,
                    "before": motion_context(before), "after": current_motion_context(ctx, ctx.view),
                    "evidence": deepcopy(result.get("evidence") or {}),
                    "subgoal": subgoal, "handoff": {"ready": not self._resume_issues(self.index_of(failed), failed)},
                    "task_effect": "unverified", "grasp_effect": deepcopy(ctx.memory.get("user_check_grasp_state"))}
        self.plan_lock._local_motion_feedback = feedback
        self.plan_lock._local_motion_history.append(deepcopy(feedback))
        self.plan_lock._local_motion_context = current_motion_context(ctx, ctx.view)
        self._settled = False
        self._refresh_local_stays()
        self._sync_framework()
        applied = {"action": "local_motion", "part": None, "step_i": self.index_of(failed),
                   "status": feedback["status"], "result": deepcopy(feedback)}
        decision.update(status=feedback["status"], result=deepcopy(applied))
        self._emit("transit", source="model_local_motion", **feedback)
        return applied

