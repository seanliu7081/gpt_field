"""Evidence and recovery orchestration scoped to the reviewed v2 executor.

No task names, simulator truth, or process-wide patches are used here. The
legacy motion controllers remain responsible for their measured robot limits.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from . import agent_speedup_v2 as base
from .agent_speedup_v2_scene import SceneModel
from .agent_speedup_v2_types import ObsReport
from .agent_speedup_v2_user_check_framework import (
    ExecutionFeedback, RecoveryMemory, StateSnapshot, intent_key,
)
from .agent_speedup_v2_user_check_skills import ensure_ready, skill_readiness


def compact_evidence(value, *, depth=0):
    """Bound model context; the full structured evidence stays in run artifacts."""
    if depth > 7:
        return {"omitted": "nested detail; available in execution log"}
    if isinstance(value, dict):
        result = {k: compact_evidence(v, depth=depth + 1) for k, v in list(value.items())[:32]}
        if len(value) > 32:
            result["omitted_fields"] = len(value) - 32
        return result
    if isinstance(value, (list, tuple)):
        result = [compact_evidence(v, depth=depth + 1) for v in value[:12]]
        if len(value) > 12:
            result.append({"omitted_items": len(value) - 12})
        return result
    if isinstance(value, str) and len(value) > 1200:
        return value[:1200] + " [truncated; full value in execution log]"
    return value


class StableScene(SceneModel):
    """Descriptions are labels; only observations invalidate live geometry.

    A known ID keeps its current record across rewording. Explicit observation,
    contact/motion and release still use the ordinary invalidation paths. New
    IDs are new entities; the reviewed executor rejects repurposing a known ID.
    """

    def merge_plan(self, plan, roles, view, quarter_turns):
        from .agent_speedup_v2_user_check_spatial import install_spatial_spec
        install_spatial_spec(self, plan.to_json())
        # Dependencies must also remain in the runner's registration inventory,
        # not just in the first survey call. Their ordinary object observations
        # are reused when a relative destination is refreshed later.
        for region in self.user_check_regions.values():
            for pid in (region["support"], region["anchor"]):
                if pid is not None and pid != "observed_plane" and pid in roles:
                    roles[pid] = frozenset(set(roles[pid]) | {"ref"})
        for part in plan.parts:
            old = self.parts.get(part.pid)
            if old is not None and old.text != part.text:
                self.parts[part.pid] = replace(old, text=part.text)
        return super().merge_plan(plan, roles, view, quarter_turns)


class FrameworkRunnerMixin:
    def __init__(self, ctx, **kwargs):
        super().__init__(ctx, **kwargs)
        self.recovery_memory = RecoveryMemory()
        self.plan_lock._recovery_memory = self.recovery_memory
        self._last_feedback = None
        self._framework_seen = []
        self._last_actual_effect = None
        self._intent_sources = {}
        reviewed_steps = self.plan_lock.task_contract.approved_plan["steps"]
        # An explicit grasp before a reviewed manipulation is a waypoint. Its
        # held resource is consumed by that later operation; maintaining it
        # forever would contradict legitimate placement and release.
        self._transient_hold_intents = {
            intent_key(step) for i, step in enumerate(reviewed_steps)
            if step["do"] == "hold" and any(
                later["obj"] == step["obj"] and later["do"] in {"turn", "place_on"}
                for later in reviewed_steps[i + 1:])}
        self._historical_intents = set()
        self._stays_observed_at = None
        self._entity_bindings = dict(self.plan_lock.task_contract.entity_bindings)
        self._contract_runtime = hasattr(self.relations, "USER_CHECK_CONTROL_VERSION")
        self._sync_framework()

    def _sync_framework(self):
        ctx = self.ctx
        scene = getattr(ctx, "scene", None)
        facts = [{"relation": key, "value": {"met": True, "violated": False}.get(report.verdict),
                  "verdict": report.verdict, "source": "relation_observer",
                  "observed_at_step": self._stays_observed_at,
                  "observer_check": report.step}
                 for key, report in self.last_stays.items()]
        state = StateSnapshot.from_records(
            getattr(scene, "parts", {}), step=int(getattr(ctx, "step", 0)),
            holding=self.holding_pid(), facts=facts,
            evidence={"awaiting_registration": sorted(self._awaiting)})
        value = state.to_dict()
        regions = getattr(scene, "user_check_regions", {})
        for pid, item in value["entities"].items():
            item["kind"] = "region" if pid in regions else "object_or_part"
            if pid in regions:
                item["region_spec"] = copy.deepcopy(regions[pid])
                item["region_evidence"] = copy.deepcopy(getattr(scene,"user_check_region_evidence",{}).get(pid))
        memory = getattr(ctx, "memory", {})
        value["evidence"]["constraint_scope"] = base.jsonable(memory.get("user_check_constraint_scope"))
        value["evidence"]["placement_validation"] = base.jsonable(memory.get("user_check_placement_validation"))
        self.plan_lock._world_state = value
        self.plan_lock._recovery_session = base.jsonable(memory.get("user_check_recovery") or {})
        self.plan_lock._checkpoint = base.jsonable(memory.get("user_check_phase_checkpoint") or {})
        self.plan_lock._last_feedback = self._last_feedback
        self.plan_lock._completed_intents = self.completed_intents()
        return state

    def completed_intents(self):
        # A model skip or plan rewrite is never completion evidence. Even a
        # transient waypoint first needs a verified successful execution.
        for state in self.states:
            if state.status in {"done", "stays"} and self._stays_hold_for(state):
                keys = [key for key, owner in self._owner.items() if owner is state]
                intent = intent_key(state.step)
                self._intent_sources[intent] = keys
                if intent in self._transient_hold_intents:
                    self._historical_intents.add(intent)
        # Keep links to live evidence even if a new plan omits an already met
        # goal. Historical intermediate holds remain completed after their
        # resource is consumed; final holds, placements and effects stay live.
        return [intent for intent, keys in self._intent_sources.items()
                if intent in self._historical_intents or (
                    keys and all(getattr(self.last_stays.get(k), "verdict", None) == "met" for k in keys))]

    def capture_feedback(self, outcome, state=None):
        if outcome is None:
            return
        detail = (outcome.evidence or {}).get("execution_feedback", {})
        unmet = detail.get("unmet_conditions", detail.get("unmet", ()))
        feedback = ExecutionFeedback.from_outcome(
            outcome, policy_step=int(getattr(self.ctx, "step", 0)),
            unmet_conditions=unmet,
            executed=False if detail.get("status") == "blocked" else None)
        from .agent_speedup_v2_user_check_constraints import causal_feedback
        causal = causal_feedback(outcome, self.ctx)
        if causal:
            evidence = {**feedback.evidence, "causal": causal}
            unmet = feedback.unmet_conditions
            if not unmet and feedback.status != "succeeded":
                unmet = tuple(causal.get("unmet_conditions", ()))
            feedback = replace(feedback, evidence=evidence, unmet_conditions=unmet)
        if feedback.status == "succeeded":
            feedback = replace(feedback, observed_effects=({
                "relation": outcome.relation, "pids": outcome.pids,
                "reason": outcome.reason, "source": "controller_verification"},))
            # The production hold controller verifies its result through the
            # held-monitor bus; it does not install a persistent stays entry.
            # Its measured success is sufficient evidence for a transient
            # waypoint, while final goals still need current evidence.
            if state is not None and state.step.do == "hold" and outcome.relation == "held":
                intent = intent_key(state.step)
                if intent in self._transient_hold_intents:
                    self._intent_sources.setdefault(intent, [])
                    self._historical_intents.add(intent)
        self._last_feedback = feedback.to_dict()
        snapshot = self._sync_framework()
        if not any(old is outcome for old in self._framework_seen):
            self._framework_seen = (self._framework_seen + [outcome])[-256:]
            self.recovery_memory.record(feedback, snapshot, context={
                "intent": None if state is None else intent_key(state.step)})
        return feedback

    def observe_outcome(self, outcome, state):
        super().observe_outcome(outcome, state)
        self.capture_feedback(outcome, state)

    def ask(self, kind, failed, outcome, **kwargs):
        self.capture_feedback(outcome, failed)
        self._sync_framework()
        return (yield from super().ask(kind, failed, outcome, **kwargs))

    def step_history(self, state):
        # The legacy transport has a strict history schema. Richer state and
        # persistent memory travel separately in execution_framework.
        return super().step_history(state)

    def _carry_done_steps(self, old_states, new_states):
        # Changing a stopping condition creates a different intent, even when
        # do/obj/ref match. It cannot inherit success from the previous step.
        done = [s for s in old_states if s.status in base.STAYS_HOLDING_STATUSES]
        used = set()
        for new in new_states:
            for old in done:
                if id(old) in used or intent_key(old.step) != intent_key(new.step):
                    continue
                if old.step.keep != new.step.keep:
                    continue
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

    def _unregistered(self, pids=None):
        result = super()._unregistered(pids)
        for pid, part in getattr(getattr(self.ctx, "scene", None), "parts", {}).items():
            if pids is not None and pid not in pids:
                continue
            if pid in self._awaiting or not (getattr(part, "status", None) == "registered"
                                             or getattr(part, "provisional", False)):
                result.setdefault(pid, "precondition_unmet")
        return result

    def preconditions(self, state, grip):
        blocked = yield from super().preconditions(state, grip)
        if blocked is not None or not self._contract_runtime:
            return blocked
        step = replace(state.step, grip=tuple(grip))
        # Reserve necessary evidence before occupying the hand. Only a known
        # next destination for this object is considered, not all future work.
        if step.do == "hold" and self.holding_pid() is None:
            index = self.index_of(state)
            for later in self.states[(index or 0) + 1:]:
                if later.status in {"done", "stays"}:
                    continue
                if later.step.obj == step.obj and later.step.do == "release":
                    break
                if later.step.obj == step.obj and later.step.do == "place_on":
                    upcoming = replace(later.step, grip=tuple(self.grip_of(later)))
                    blocked = yield from ensure_ready(self.ctx, upcoming, relation_layer=self.relations)
                    if blocked is not None:
                        return blocked
                    break
        return (yield from ensure_ready(self.ctx, step, relation_layer=self.relations))

    def first_step_readiness(self, plan):
        if not self._contract_runtime:
            return None, None
        completed = set(self.completed_intents())
        for i, step in enumerate(plan.steps):
            if intent_key(step) in completed:
                continue
            ready = skill_readiness(self.ctx, step)
            # Missing observations can be acquired by the skill itself while
            # resources permit; a held-hand observation deadlock cannot.
            if not ready["ready"] and not ready.get("can_observe", False):
                known_ids = {p.pid for p in plan.parts}
                observation_only = all(
                    u["predicate"].startswith("destination_") or
                    (u["predicate"] in {"known_part", "known_destination"}
                     and (u.get("part") or step.ref) in known_ids)
                    for u in ready["unmet"])
                if self.holding_pid() is None and observation_only:
                    break  # adoption registers new named parts before execution
                return i, ready
            break
        return None, None

    def _add_row(self, state, outcome, action, part, effect):
        # Legacy computes an expected effect before apply(). Record its actual
        # disposition after execution instead, in both model history and logs.
        if self._last_actual_effect is not None:
            effect = self._last_actual_effect
            self._last_actual_effect = None
        super()._add_row(state, outcome, action, part, effect)
        if self.plan_lock._decisions:
            result = self.plan_lock._decisions[-1].get("result")
            if result is not None:
                result["effect"] = effect

    def escalate(self, failed, outcome):
        """Same model recovery boundary, with effects evaluated after apply."""
        if self.exhausted:
            return
        previous = self.active_state
        self.active_state = failed if failed is not None else previous
        if failed is not None:
            for item in list(getattr(self.ctx, "pending_outcomes", ()) or ()):
                self.observe_outcome(item, self._tags.get(id(item), self.active_state))
            self._name_failing_grip(failed, outcome)
        marker = self.add_runner_facts(failed, outcome)
        self.capture_feedback(outcome, failed)
        trigger = self._ladder_trigger(failed, outcome) if failed is not None else None
        if trigger is not None:
            applied = yield from self.climb(failed, outcome, why=trigger)
            self._emit("repair", request=None, answer=None, valid=None, issues=[], reask=False,
                       failed_i=applied["step_i"], applied=applied)
        else:
            history = self.step_history(failed) if failed is not None else None
            answer, info = yield from self.ask("repair", failed, outcome, marker_uv=marker, history=history)
            if answer is None:
                applied = yield from self.climb(failed, outcome, why="no_change_refused")
            else:
                predicted = self._effect_of(answer, failed) if failed is not None else None
                applied = yield from self.apply(answer, failed, review=False, outcome=outcome)
                if failed is not None:
                    effect = self._last_actual_effect or predicted
                    self._add_row(failed, outcome, answer.action, answer.part, effect)
                    applied["effect"] = effect
            self._emit("repair", **info, applied=applied)
        self.active_state = previous

    def hold_true_check(self, claim):
        if not self._contract_runtime:
            return super().hold_true_check(claim)
        # A claim is not an observation. Reuse a live verified relation when
        # there is one; otherwise keep it unknown, never compare to a cold
        # episode-start image or assume the model's assertion is already true.
        for key, owner in self._owner.items():
            if owner is None:
                continue
            step = owner.step
            same = (claim.rel == "on" and step.do == "place_on" and
                    (step.obj, step.ref) == (claim.a, claim.b))
            same |= (claim.rel == "effect" and any(u.o == "effect" and u.p == claim.a for u in step.until))
            if not same or key not in self.ctx.stays.keys():
                continue

            def report(view, source=key):
                try:
                    data = self.ctx.stays.entry(source)["last"]
                except KeyError:
                    data = None
                current = int(getattr(self.ctx, "step", 0))
                if not data:
                    return ObsReport("reviewed_claim", "pending", None, "source_not_current",
                                     {"source": source}, (), current)
                return ObsReport("reviewed_claim", data["verdict"], data.get("progress"), data.get("reason"),
                                 {"source": source, "observation": data["evidence"]}, (), current)
            return report, None
        return None, "claim_not_verified"

    def stays_boundary_check(self):
        self._stays_observed_at = int(getattr(self.ctx, "step", 0))
        # Relayed claims must run after their source checks on this boundary.
        # Report.step is an observer counter, not the simulator's policy step.
        entries = getattr(self.ctx.stays, "_entries", {})
        for key in self._hold_true:
            if key in entries:
                entries[key] = entries.pop(key)
        yield from super().stays_boundary_check()
        self._sync_framework()
