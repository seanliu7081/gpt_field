"""Temporal constraint scope and contact semantics for reviewed execution.

Plan roles still describe what needs observing. They do not decide when a
constraint is active. In particular a later step's obstacle cannot fence off
the destination of an earlier step. No scene names or task IDs occur here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from . import agent_speedup_v2 as base
from . import agent_speedup_v2_observers as observers


CONSTRAINT_VERSION = "user-check-scoped-constraints-1"


def _wire_step(step):
    if isinstance(step, Mapping):
        return dict(step)
    return step.to_json() if hasattr(step, "to_json") else {
        "do": step.do, "obj": step.obj, "ref": step.ref,
        "grip": list(step.grip), "keep": [{"c": k.c, "p": k.p} for k in step.keep]}


def constraint_conflicts(step):
    """An explicit prohibition and a necessary contact need a plan repair."""
    step = _wire_step(step)
    required = {step.get("obj"), *step.get("grip", ())} - {None}
    if step.get("do") == "place_on" and step.get("ref") is not None:
        required.add(step["ref"])
    return [{"predicate": "contact_constraint_consistent", "part": k["p"],
             "required": "grasp_or_support_contact", "observed": "explicit_clear_of",
             "source": "active_step.keep", "scope": "current_intent",
             "remedy": "revise the contradictory constraint or manipulation intent"}
            for k in step.get("keep", ()) if k.get("c") == "clear_of" and k.get("p") in required]


class ConstraintRunnerMixin:
    """The currently pursued intent owns keep constraints, including recovery."""

    def ensure_keepouts(self):
        ctx, scene = self.ctx, self.ctx.scene
        state = getattr(self, "active_state", None)
        step = getattr(state, "step", None)
        raw = None if step is None else _wire_step(step)
        previous = ctx.memory.get("user_check_constraint_scope", {})
        holding = getattr(getattr(ctx, "holding", None), "pid", None)
        # An inserted release/recovery keeps the carried intent's prohibitions
        # until the held resource is released. A future manipulation gets its
        # own scope; idle/review does not activate all pending steps.
        if raw is None or (raw.get("do") == "release" and holding == previous.get("obj")):
            raw = previous.get("step")
        if raw is not None and state is None and holding is None:
            raw = None
        local = [] if raw is None else [dict(k) for k in raw.get("keep", ())]
        active = {k["p"] for k in local if k.get("c") == "clear_of"}
        established = set(ctx.memory.get("user_check_established_claims", ()))
        for key, report in getattr(self, "last_stays", {}).items():
            if key.startswith("hold_true:") and getattr(report, "verdict", None) == "met":
                established.add(key)
        claims = tuple(getattr(getattr(self, "plan", None), "hold_true", ()))
        claim_keys = {self.claim_key(c) for c in claims}
        established &= claim_keys
        ctx.memory["user_check_established_claims"] = sorted(established)
        global_pids = {c.b for c in claims if self.claim_key(c) in established and c.b is not None}
        global_pids &= set(getattr(self, "_hold_true_keepouts", ()))
        origins = {f"clear_of:{pid}": {"scope": "current_intent", "source": "step.keep",
                    "step": self.index_of(state) if state is not None else previous.get("step_index"),
                    "constraint": {"c": "clear_of", "p": pid}} for pid in active}
        for pid in global_pids:
            claims = [c.to_json() if hasattr(c, "to_json") else {"rel": c.rel, "a": c.a, "b": c.b}
                      for c in getattr(getattr(self, "plan", None), "hold_true", ()) if c.b == pid]
            origins[f"hold_true:{pid}"] = {"scope": "global_claim", "source": "hold_true", "claims": claims}
        for key, keepout in list(scene.keepouts.items()):
            if (keepout.kind == "clear_of" and keepout.pid not in active) or (
                    keepout.kind == "hold_true" and keepout.pid not in global_pids):
                del scene.keepouts[key]
        for pid in active:
            self._add_keepout(pid, "clear_of", scene)
        for pid in global_pids:
            self._add_keepout(pid, "hold_true", scene)
        self._stale_keepouts = {pid for pid in self._stale_keepouts
                               if getattr(scene.parts.get(pid), "status", None) != "registered"}
        scope = {"version": CONSTRAINT_VERSION, "step": raw,
                 "step_index": self.index_of(state) if state is not None else None,
                 "obj": None if raw is None else raw.get("obj"),
                 "origins": origins, "active_constraints": local,
                 "completed_goal_protection": "measured stays observers"}
        ctx.memory["user_check_constraint_scope"] = scope
        if origins != previous.get("origins", {}):
            monitor = getattr(ctx, "monitor", None)
            if monitor is not None and "keepout" in monitor.keys():
                # Old queued violations belong to the old scope. The new
                # controller installs a fresh observer before yielding motion.
                monitor.remove("keepout")

    def _enter(self, phase, relation_name):
        super()._enter(phase, relation_name)
        if hasattr(self, "_stale_keepouts") and hasattr(self.ctx.scene, "keepouts"):
            self.ensure_keepouts()

    def preconditions(self, state, grip):
        self.ensure_keepouts()
        conflicts = constraint_conflicts(state.step)
        if conflicts:
            return self._outcome(base.RELATION_OF[state.step.do], "failed", "precondition_unmet",
                self._pids(state.step, grip), {
                    "execution_feedback": {"status": "blocked", "unmet_conditions": conflicts},
                    "cause": {"code": "constraint_conflict", "source": "step.keep",
                              "phase": "before_execution", "constraints": conflicts},
                    "constraint_scope": self.ctx.memory["user_check_constraint_scope"]}, state=state,
                text="The active clear_of constraint forbids contact required by this manipulation.")
        return (yield from super().preconditions(state, grip))


class ScopedKeepOutObserver(observers.KeepOutObserver):
    """Live scope plus explicit pair/phase classification of overlap evidence.

An overlap with a destination never exempts the hand. Only the held object can
make intended support contact during lowering/release, and an explicit active
clear_of/global constraint always wins (normally caught before execution).
"""

    def __init__(self, *, ctx, keepouts, robot, held):
        self.ctx = ctx
        super().__init__(keepouts=keepouts, robot=robot, held=held)

    def update(self, before, action, after, context=None):
        ctx = self.ctx
        held = getattr(ctx, "holding", None)
        self.held = held
        own = {getattr(held, "pid", None), getattr(held, "grip_pid", None)} - {None}
        self.keepouts = tuple(k for k in ctx.scene.keepouts.values() if k.pid not in own)
        report = super().update(before, action, after, context)
        evidence = dict(report.evidence)
        scope = ctx.memory.get("user_check_constraint_scope", {})
        step = scope.get("step") or {}
        phase = ctx.memory.get("user_check_contact_phase", "unknown")
        origins = scope.get("origins", {})
        forbidden, expected, separating = [], [], []
        for hit in evidence.get("hits", ()):
            hit = dict(hit)
            key = f"{hit.get('kind')}:{hit.get('pid')}"
            hit["origin"] = origins.get(key, {"scope": "physical_geometry", "source": hit.get("kind")})
            support_pair = (step.get("do") == "place_on" and hit.get("pid") == step.get("ref")
                            and hit.get("by") == "held" and phase in {"lower", "support", "release"}
                            and hit.get("kind") not in {"clear_of", "hold_true"}
                            and getattr(held, "pid", None) == step.get("obj"))
            hit["classification"] = "intended_support_contact" if support_pair else "forbidden_contact_candidate"
            if (not support_pair and phase == "local_recovery" and
                    self._measured_separation(hit, before, after)):
                hit["classification"] = "measured_separating_contact"
                separating.append(hit)
            else:
                (expected if support_pair else forbidden).append(hit)
        evidence.update(hits=forbidden, expected_contacts=expected, separating_contacts=separating, phase=phase,
                        scope_step=scope.get("step_index"),
                        evidence_kind="stall_with_geometric_overlap",
                        confidence="candidate_contact; overlap is not a measured collision pair")
        return replace(report, evidence=evidence, verdict="violated" if forbidden else "met",
                       reason="keepout_contact" if forbidden else None)

    def _measured_separation(self, hit, before, after):
        """Only allow a bounded recovery that measurably reduces this overlap.

        This is a geometric progress check, not a claim that contact ended. It
        never exempts another hit, normal transport, or zero-realized motion.
        """
        if np.linalg.norm(np.asarray(after.position) - np.asarray(before.position)) <= 1e-5:
            return False
        obstacle = next((k for k in self.keepouts if k.pid == hit["pid"] and k.kind == hit["kind"]), None)
        if obstacle is None:
            return False
        def overlap(view):
            poly, _, z_range, pose = self._hand_samples(view)
            if hit["by"] == "held":
                if self.held is None:
                    return None
                poly, _, z_range = self._held_samples(pose)
            poly = np.asarray(poly, dtype=float)
            obstacle_poly = np.asarray(obstacle.polygon_xy, dtype=float)
            lo = np.maximum(np.r_[poly.min(axis=0), z_range[0]],
                            np.r_[obstacle_poly.min(axis=0), obstacle.z_low])
            hi = np.minimum(np.r_[poly.max(axis=0), z_range[1]],
                            np.r_[obstacle_poly.max(axis=0), obstacle.z_high])
            return float(np.prod(np.maximum(hi - lo, 0.0)))
        old, new = overlap(before), overlap(after)
        return old is not None and old > 0.0 and new < old - max(1e-12, old * 1e-3)


def preserved_preemption_info(result):
    """Keep the actual observer report; callers may compact it for the model."""
    result = result or {}
    evidence = result.get("evidence") or {}
    report = evidence.get("report") or {}
    detail = report.get("evidence") or {}
    return {"invariant": evidence.get("invariant"),
            "cause": detail.get("cause") or report.get("reason") or result.get("reason"),
            "reason": report.get("reason"), "phase": detail.get("phase"),
            "tilt_change_deg": detail.get("tilt_change_deg"),
            "hits": detail.get("hits", []), "report": report,
            "primitive_reason": result.get("reason")}


def causal_feedback(outcome, ctx=None):
    """Extract a small causal spine without dropping nested native failures.

Full evidence remains on the outcome. Paths distinguish the failed planner,
observer or prerequisite from successful checks elsewhere in the same attempt.
"""
    evidence = getattr(outcome, "evidence", {}) or {}
    rows = []
    successful_reasons = {reason for values in base.kernel.SUCCESS_REASONS.values() for reason in values}
    def visit(value, path):
        if isinstance(value, Mapping):
            reason = value.get("reason") or value.get("cause")
            informative = (reason is not None and (value.get("ok") is False
                           or value.get("status") in {"failed", "blocked", "rejected"}
                           or value.get("verdict") == "violated" or value.get("feasible") is False
                           or ("preemption" in path and isinstance(reason, str) and reason not in successful_reasons)
                           or reason == getattr(outcome, "reason", None)))
            if informative or value.get("hits") or value.get("unmet_conditions"):
                keys = ("reason", "cause", "phase", "source", "ok", "status", "verdict", "invariant", "hits",
                        "unmet_conditions", "values", "measured", "required", "expected",
                        "failed_in", "blocking_condition", "event", "confidence", "evidence_kind")
                rows.append({"path": path, **{k: value[k] for k in keys if k in value}})
            for key, child in value.items():
                if key not in {"thresholds", "text_items", "text_flags", "registration", "scene"}:
                    visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            # The latest attempts explain the current blocker. Older attempts
            # belong in persistent failure memory, not this causal summary.
            for index, child in list(enumerate(value))[-3:]:
                visit(child, f"{path}[{index}]")
    visit(evidence, "evidence")
    scope = (getattr(ctx, "memory", {}) or {}).get("user_check_constraint_scope", {})
    unmet = list((evidence.get("execution_feedback") or {}).get("unmet_conditions") or [])
    for pid, need in (evidence.get("unmet") or {}).items() if isinstance(evidence.get("unmet"), Mapping) else ():
        unmet.append({"predicate": need, "part": pid, "source": "controller_precondition",
                      "observed": (evidence.get("statuses") or {}).get(pid),
                      "holding": evidence.get("holding")})
    if not unmet and getattr(outcome, "status", None) != "succeeded":
        # A native refusal is an unmet executable condition even when its
        # compact wrapper did not explicitly enumerate prerequisites. Keep
        # uncertainty separate from observed failure, and never invent a
        # collision or threshold that the controller did not measure.
        candidates = list(reversed(rows)) + [{"reason": getattr(outcome, "reason", None), "path": "outcome"}]
        seen = set()
        for row in candidates:
            reason = row.get("reason") or row.get("cause")
            if not isinstance(reason, str) or not reason:
                continue
            reason = reason.removeprefix("preempted:")
            if reason in seen or reason in successful_reasons:
                continue
            seen.add(reason)
            unavailable = any(word in reason for word in (
                "unmeasured", "unavailable", "unregistered", "not_registered", "unverifiable", "unverified"))
            predicate = {
                "geometry_unmeasured": "geometry_observed",
                "destination_not_registered": "destination_geometry_ready",
                "footprint_blocked": "grasp_footprint_clear",
                "column_blocked": "approach_column_clear",
                "pointed_not_grasp_point": "usable_grasp_point",
                "keepout_contact": "contact_constraints_satisfied",
                "hold_lost": "held_object_retained",
                "lateral_unreachable": "reachable_alignment",
                "lateral_drift": "alignment_within_bound",
            }.get(reason, "controller_accepts:" + reason)
            assessment = "unavailable" if unavailable else (
                "observed_false" if row.get("verdict") == "violated" and not row.get("hits")
                else "controller_refusal")
            measured = {key: row[key] for key in ("values", "measured", "required", "expected", "hits", "event")
                        if key in row}
            unmet.append({"predicate": predicate, "value": None if unavailable else False,
                          "assessment": assessment, "reason": reason,
                          "source": row.get("path", "outcome"),
                          "stage": row.get("phase") or evidence.get("failed_in") or evidence.get("phase"),
                          "entities": dict(getattr(outcome, "pids", {}) or {}), "evidence": measured})
            if len(unmet) >= 3:
                break
    return {"code": getattr(outcome, "reason", None), "unmet_conditions": unmet,
            "phase": evidence.get("failed_in") or evidence.get("phase"),
            "entities": dict(getattr(outcome, "pids", {}) or {}),
            "causes": rows[-8:], "constraint_origins": scope.get("origins", {})}
