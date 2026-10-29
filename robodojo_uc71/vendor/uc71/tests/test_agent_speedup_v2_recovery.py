"""Gap round, repair-loop lane (judge decision §2.7-§2.10; rulings L2 / L3 / L4): the recovery information contract
(RL-C1 step_history, the failing grip named, runner facts of the current attempt, unregistered parts in the status
block), the answer-effect translation (RL-C2 answer_effect, regrasp, retry_unchanged), the escalation ladder
(RL-C3 K=3 S=4 R=2 H=5: forced replan, forced skip_step, terminal recovery_exhausted, motionless suppression), the
client's step_history field and prompt-4 paragraph, and the runner's recovery_exhausted automatic cap.

No network, no simulator, no model: the policy runs on the policy tests' StandIn relation layer and fake world; the
runner on the chain tests' fake world sim. One positive and one negative per rule.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from agent_speedup_v2_fakes import FakeV2WorldSim
from test_agent_speedup_v2 import (HOLD_D, PARTS, PLACE, TASK2_PLAN, TURN, HeldStub, StandIn, answers, build, drive,
                                   events, outcome_seen, plan_json, repair, repair_requests, request_text)
from gpt_field import agent_speedup_v2 as policy_module
from gpt_field import agent_speedup_v2_client as C
from gpt_field import agent_speedup_v2_plan as P
from gpt_field import agent_speedup_v2_relations as relations
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field.agent_speedup_v2_types import PlanStep, RepairAnswer, StepState

ROOT = Path(__file__).resolve().parents[1]
RUNNER_FILE = ROOT / "scripts/evaluate_agent_speedup_v2.py"
_spec = importlib.util.spec_from_file_location("evaluate_agent_speedup_v2_recovery", RUNNER_FILE)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


@pytest.fixture(scope="module")
def scene():
    return fakes.task2_like_scene()


def answer(action, part=None):
    return RepairAnswer("scripted", action, part, None)


def plan():
    return P.parse_plan(TASK2_PLAN)


# ------------------------------------------------------------------------------------- constants (§6)
def test_the_ladder_constants_are_registered_with_their_derivations():
    assert (P.REPAIR_IDENT_K, P.REPAIR_STALL_S, P.REPAIR_NOCHANGE_R, P.REPAIR_HISTORY_H) == (3, 4, 2, 5)
    assert P.REPAIR_HISTORY_H == P.REPAIR_STALL_S + 1
    for name in ("REPAIR_IDENT_K", "REPAIR_STALL_S", "REPAIR_NOCHANGE_R", "REPAIR_HISTORY_H"):
        label, derivation = P.LABELS[name]
        assert label == "DD" and derivation
    assert "recovery_exhausted" in kernel.EVENTS and set(kernel.REPAIR_EFFECTS) >= {"no_change", "regrasp"}


# ------------------------------------------------------------------------------------- RL-C2 (plan module)
def test_answer_effect_classifies_each_answer():
    assert P.answer_effect(answer("retry"), grip=("C",)) == "no_change"
    assert P.answer_effect(answer("use_part", "C"), grip=("C", "D")) == "no_change"         # already first
    assert P.answer_effect(answer("use_part", "D"), grip=("C", "D")) == "grip_order_changed"
    assert P.answer_effect(answer("use_part", "D"), grip=("C",), held_grip="C") == "regrasp"  # holding by C
    assert P.answer_effect(answer("use_part", "C"), grip=("D", "C"), held_grip="C") == "no_change"  # held grip
    assert P.answer_effect(answer("repoint", "C"), grip=("C",)) == "reregistered"
    assert [P.answer_effect(answer(a)) for a in ("replan", "swap_next", "skip_step", "hold_still")] == [
        "replanned", "replanned", "skipped", "no_change"]
    assert all(P.answer_effect(answer(a, "C"), grip=("C",)) in kernel.REPAIR_EFFECTS for a in kernel.REPAIR_ACTIONS
               if a != "insert_step")


def test_validate_repair_refuses_only_the_third_consecutive_no_change_answer():
    recovery = {"grip": ("C",), "held_grip": None, "no_change": 2, "failed": ["lost_in_transport", "hold_lost"]}
    issues = P.validate_repair(answer("retry"), plan(), failed_i=1, review=False, recovery=recovery)
    assert [(i.code, i.step) for i in issues] == [("retry_unchanged", 1)]
    assert "retry changes nothing (2 times already; last failed hold_lost)" in issues[0].text
    assert len(issues[0].text) <= P.ISSUE_TEXT_MAX_CHARS
    same = P.validate_repair(answer("use_part", "C"), plan(), failed_i=1, review=False, recovery=recovery)
    assert [i.code for i in same] == ["retry_unchanged"] and same[0].text.startswith("use_part C changes nothing")
    # negatives: the first two no-change answers, a real reorder, a regrasp, other actions, no recovery state
    assert P.validate_repair(answer("retry"), plan(), failed_i=1, review=False,
                             recovery={**recovery, "no_change": 1}) == []
    assert P.validate_repair(answer("use_part", "D"), plan(), failed_i=1, review=False, recovery=recovery) == []
    assert P.validate_repair(answer("use_part", "D"), plan(), failed_i=1, review=False,
                             recovery={**recovery, "held_grip": "C"}) == []
    assert P.validate_repair(answer("repoint", "C"), plan(), failed_i=1, review=False, recovery=recovery) == []
    assert P.validate_repair(answer("retry"), plan(), failed_i=1, review=False) == []


# ------------------------------------------------------------------------------- RL-C1 status block (plan)
def test_a_pending_step_shows_the_reason_its_part_failed_to_register():
    p = plan()
    states = [StepState(0, p.steps[0], "failed", "part_not_found", {}, {}, 0, 1, []),
              StepState(1, p.steps[1], "pending", None, {}, {}, 0, 0, [])]
    block = P.plan_status_block(p, states, holding=None, unregistered={"B": "part_not_found"})
    assert [(r["status"], r["reason"]) for r in block["steps"]] == [("failed", "part_not_found"),
                                                                    ("pending", "part_not_found")]
    # negatives: once B registers the reason is null again; a step's own reason is never replaced
    block = P.plan_status_block(p, states, holding=None, unregistered={})
    assert block["steps"][1]["reason"] is None
    block = P.plan_status_block(p, states, holding=None, unregistered={"A": "part_not_found"})
    assert block["steps"][0]["reason"] == "part_not_found" and block["steps"][1]["reason"] is None
    with pytest.raises(ValueError):
        P.plan_status_block(p, states, holding=None, unregistered={"B": "not a code"})


# ---------------------------------------------------------------------------------------- RL-C1 (client)
def history(**changes):
    row = {"attempt": 1, "reason": "lost_in_transport", "grips": [["C", "hold_lost"]], "then": ["retry", None],
           "effect": "no_change"}
    out = {"step": 1, "attempts": 2, "earlier": [row], "this_attempt": [["C", "hold_lost"]], "held_by": "C",
           "unregistered": [["E", "part_not_found"]]}
    out.update(changes)
    return out


def repair_fields(client, **kwargs):
    grey = np.full((512, 512, 3), 128, dtype=np.uint8)
    block = P.plan_status_block(plan(), [], holding=None)
    outcome = {"reason": "hold_lost", "evidence_text": "held failed for part D via part C: hold_lost",
               "thresholds": []}
    _, request, _ = client._repair_request(grey, grey, quarter_turns=0, status_block=block, last_outcome=outcome,
                                           holding="D", **kwargs)
    return request["fields"]


def test_repair_and_replan_requests_carry_step_history_in_order():
    client = C.AgentSpeedUpV2Client(task_prompt="put the pot on the stove")
    fields = repair_fields(client, step_history=history())
    assert list(fields) == ["plan", "last_outcome", "step_history", "holding", "previous_answer_rejected"]
    assert fields["step_history"] == history()
    assert repair_fields(client)["step_history"] is None                       # no failed step: null
    assert "step_history" not in repair_fields(client, review=True)            # a review has none
    grey = np.full((512, 512, 3), 128, dtype=np.uint8)
    block = P.plan_status_block(plan(), [], holding=None)
    _, request, _ = client._plan_request(grey, grey, quarter_turns=0, fingertip_edge="bottom", status_block=block,
                                         replan=True, step_history=history())
    assert list(request["fields"]) == ["holding", "images", "status", "step_history", "previous_plan",
                                       "previous_answer_rejected"]
    _, first, _ = client._plan_request(grey, grey, quarter_turns=0, fingertip_edge="bottom")
    assert "step_history" not in first["fields"]
    with pytest.raises(ValueError):
        client._plan_request(grey, grey, quarter_turns=0, fingertip_edge="bottom", step_history=history())


@pytest.mark.parametrize("bad", [
    {"earlier": [history()["earlier"][0]] * (P.REPAIR_HISTORY_H + 1)},           # more than H rows
    {"earlier": [{**history()["earlier"][0], "effect": "teleported"}]},           # not an effect code
    {"earlier": [{**history()["earlier"][0], "why": "free text"}]},              # an extra key
    {"this_attempt": [["C", "the handle slipped"]]},                             # not a reason code
    {"held_by": "Z"}, {"attempts": -1}, {"note": "x"}])
def test_the_step_history_validator_accepts_codes_ids_and_counts_only(bad):
    client = C.AgentSpeedUpV2Client(task_prompt="put the pot on the stove")
    with pytest.raises(ValueError):
        repair_fields(client, step_history=history(**bad))


def test_prompt_4_adds_one_history_paragraph_to_repair_and_replan_only():
    from test_agent_speedup_v2_client import vocabulary_hits
    assert C.PROMPT_VERSION == "agent-speedup-v2-prompt-4"
    assert not vocabulary_hits(C._HISTORY) and C._HISTORY.isascii()
    for kind in ("repair", "replan"):
        text = C.instructions(kind)
        assert text.startswith(C.PROMPT_VERSION) and text.count(C._HISTORY) == 1
    for kind in ("plan", "review", "point_fine", "point_coarse"):
        assert C._HISTORY not in C.instructions(kind)
    assert set(P.repair_schema()["properties"]) == {"why", "action", "part", "step"}   # AC-5: schema unchanged


# ------------------------------------------------------------------------------------------ the policy
class GraspLayer(StandIn):
    """StandIn whose place_on / hold attempts record the grasp outcomes of their spec's ``grasps`` ([pid, status,
    reason]) before the step outcome; ``holding`` = (obj, grip) sets ctx.holding whatever the status. Evidence texts
    are the real relation layer's (``relations.evidence_text``)."""

    def _controller(self, do):
        base = super()._controller(do)

        def controller(ctx, step):
            spec = (self.script.get(do) or [{}])[0]
            for pid, status, reason in spec.get("grasps", ()):
                held = kernel.RelationOutcome("held", status, reason, {"obj": step.obj, "grip": [pid], "ref": None},
                                              {}, [], "", 0, 0)
                held.evidence_text = relations.evidence_text(held)
                ctx.record(held)
            result = yield from base(ctx, step)
            result.evidence_text = relations.evidence_text(result)
            if spec.get("holding"):
                ctx.holding = HeldStub(*spec["holding"])
            return result
        return controller


def last_outcomes(transport):
    return [text["last_outcome"] for text, meta in repair_requests(transport) if meta["kind"] == "repair"]


def histories(transport):
    return [text["step_history"] for text, meta in repair_requests(transport) if meta["kind"] == "repair"]


def test_the_evidence_names_the_grip_that_failed_last_including_the_object_itself(scene):
    fail = {"status": "failed", "reason": "no_feasible_candidate"}
    layer = GraspLayer(script={"place_on": [{**fail, "grasps": [["C", "failed", "footprint_blocked"],
                                                                ["D", "failed", "no_feasible_candidate"]]},
                                            {**fail, "grasps": [["C", "failed", "footprint_blocked"]]}]})
    scripted = answers(repairs=(repair("use_part", part="D"), repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 600, stop=lambda p, d: len(repair_requests(transport)) >= 2)
    first, second = last_outcomes(transport)[:2]
    assert first["evidence_text"].startswith("supported_on failed for part D gripping part D itself onto part B")
    assert second["evidence_text"].startswith("supported_on failed for part D via part C onto part B")  # negative
    hist = histories(transport)
    assert hist[0]["this_attempt"] == [["C", "footprint_blocked"], ["D", "no_feasible_candidate"]]
    assert hist[1]["earlier"] == [{"attempt": 1, "reason": "no_feasible_candidate",
                                   "grips": [["C", "footprint_blocked"], ["D", "no_feasible_candidate"]],
                                   "then": ["use_part", "D"], "effect": "grip_order_changed"}]
    assert hist[1]["attempts"] == 2 and hist[1]["this_attempt"] == [["C", "footprint_blocked"]]


def test_history_survives_a_replan_and_restarts_when_the_step_succeeds(scene):
    fail = {"status": "failed", "reason": "turn_slip"}
    layer = StandIn(script={"turn": [fail, fail]})
    scripted = answers(plans=(TASK2_PLAN, TASK2_PLAN), repairs=(repair("replan"), repair("retry")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 900, stop=outcome_seen("supported_on"))
    hist = histories(transport)
    assert [h["attempts"] for h in hist] == [1, 2]
    assert hist[1]["earlier"] == [{"attempt": 1, "reason": "turn_slip", "grips": [], "then": ["replan", None],
                                   "effect": "replanned"}]                  # kept across the replan (same key)
    assert transport.kinds.count("replan") == 1
    replan_text = [request_text(c) for c, k in zip(transport.calls, transport.kinds) if k == "replan"][0]
    assert replan_text["step_history"] == hist[0]                           # the model's replan carries it too
    runner_ = policy._runner
    assert ("turn", "A", None) not in runner_._history                      # the turn succeeded at attempt 3
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["done", "done"]


def test_runner_facts_count_only_the_current_attempt(scene):
    """The t8 stale-B pattern: rejected pointing rounds of an earlier attempt are not reported again."""
    fail = {"status": "failed", "reason": "pointed_not_grasp_point"}
    layer = StandIn(script={"turn": [{**fail, "checks": [("A", (100.0, 100.0), False)]}, fail]})
    scripted = answers(repairs=(repair("retry"), repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 600, stop=outcome_seen("supported_on"))
    texts = [o["evidence_text"] for o in last_outcomes(transport)]
    assert "part A: 1 pointing rejected" in texts[0]
    assert "pointing rejected" not in texts[1]


def test_use_part_of_another_part_while_holding_releases_on_support_first(scene):
    fail = {"status": "failed", "reason": "lost_in_transport", "holding": ("D", "C")}
    layer = GraspLayer(script={"place_on": [fail, fail]})
    scripted = answers(repairs=(repair("use_part", part="D"), repair("use_part", part="C"), repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 900, stop=outcome_seen("supported_on", "placed"))
    applied = [r["applied"] for r in events(decisions, "repair")]
    assert applied[0]["effect"] == "regrasp" and applied[0]["regrasp"] is True
    log = [row[:2] if row[0] == "controller" else row for row in layer.log]
    i = log.index(("controller", "place_on"))
    assert log[i + 1:i + 3] == [("release", None), ("controller", "place_on")]   # set down, then grasp D
    # negative: use_part of the held grip changes nothing and does not release
    assert applied[1]["effect"] == "no_change" and "regrasp" not in applied[1]
    j = len(log) - 1 - log[::-1].index(("controller", "place_on"))
    assert log[j - 1] != ("release", None)


def turn_failures(n, **extra):
    return {"turn": [{"status": "failed", "reason": "turn_slip", **extra}] * n}


def test_the_ladder_replans_then_skips_then_ends_with_recovery_exhausted(scene):
    """Retry answers forever: the 3rd no-change answer is refused, the re-ask too -> forced replan; the same again
    on the replanned step -> forced skip_step (skipped, never stays); the place_on runs; nothing can run, so
    recovery_exhausted (once), no review, and no further call."""
    layer = StandIn(script=turn_failures(20))
    scripted = answers(plans=(TASK2_PLAN, TASK2_PLAN), repairs=(repair("retry"),))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 2500)
    kinds = [k for k in transport.kinds if k != "point_fine"]
    assert kinds == ["plan", "repair", "repair", "repair", "repair", "replan",
                     "repair", "repair", "repair", "repair"]
    reps = events(decisions, "repair")
    forced = [r["applied"] for r in reps if (r["applied"] or {}).get("forced")]
    assert [(a["action"], a["rung"], a["why"]) for a in forced] == [("replan", 1, "no_change_refused"),
                                                                   ("skip_step", 2, "no_change_refused")]
    refused = [r for r in reps if r["valid"] is False]
    assert refused and all(i["code"] == "retry_unchanged" for r in refused for i in r["issues"])
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["skipped", "done"]
    exhausted = events(decisions, "recovery_exhausted")
    assert len(exhausted) == 1 and exhausted[0]["forced_skips"] == [0]
    after = next(k for k, d in enumerate(decisions) if any(r["event"] == "recovery_exhausted"
                                                          for r in d.diagnostics["v2_events"]))
    assert all(not d.diagnostics["api_calls"] for d in decisions[after:])     # zero calls from then on
    assert "review" not in transport.kinds


def test_four_changing_failures_then_success_are_not_escalated(scene):
    """Negative: answers that change the state (repoints of two parts) for 4 failures, then success: no rung."""
    fail = {"status": "failed", "reason": "no_feasible_candidate"}
    layer = StandIn(script={"place_on": [fail] * 4})
    scripted = answers(repairs=(repair("repoint", part="C"), repair("repoint", part="D"), repair("repoint", part="C"),
                                repair("repoint", part="D")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 1500, stop=outcome_seen("supported_on", "placed"))
    assert [r["applied"]["action"] for r in events(decisions, "repair")] == ["repoint"] * 4
    assert not any(r["applied"].get("forced") for r in events(decisions, "repair"))
    assert "replan" not in transport.kinds and not events(decisions, "recovery_exhausted")
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["done", "done"]


@pytest.mark.parametrize("parts,why", [(("C", "D", "C", "D"), "identical_attempts"),
                                       (("C", "D", "uD", "uC"), "stalled_attempts")])
def test_identical_or_stalled_failures_climb_without_a_call(scene, parts, why):
    """The fifth failure: ident > K (the same reason, grips and answer effect: repoints) or stall > S (changing
    effects: repoints then reorders) -> a forced replan with the history and no repair call."""
    fail = {"status": "failed", "reason": "no_feasible_candidate"}
    layer = StandIn(script={"place_on": [fail] * 5})
    reps = [repair("use_part", part=p[1]) if p.startswith("u") else repair("repoint", part=p) for p in parts]
    scripted = answers(plans=(TASK2_PLAN, TASK2_PLAN), repairs=(*reps, repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 2000, stop=lambda p, d: "replan" in transport.kinds)
    forced = [r["applied"] for r in events(decisions, "repair") if r["applied"].get("forced")]
    assert [(a["action"], a["why"]) for a in forced] == [("replan", why)]
    assert transport.kinds.count("repair") == 4                                # the 5th failure made no call
    replan_text = [request_text(c) for c, k in zip(transport.calls, transport.kinds) if k == "replan"][0]
    assert replan_text["step_history"]["attempts"] == 5 and len(replan_text["step_history"]["earlier"]) == 4


def test_a_motionless_repeat_after_a_no_change_answer_makes_no_call(scene):
    first = {"status": "failed", "reason": "reach_limit", "evidence": {"motionless": True}}
    repeat = {"status": "failed", "reason": "reach_limit", "evidence": {"motionless": True, "motionless_repeat": 1}}
    layer = StandIn(script={"turn": [first, repeat]})
    scripted = answers(plans=(TASK2_PLAN, TASK2_PLAN), repairs=(repair("retry"),))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 900, stop=lambda p, d: "replan" in transport.kinds)
    assert [k for k in transport.kinds if k != "point_fine"] == ["plan", "repair", "replan"]
    forced = [r["applied"] for r in events(decisions, "repair") if r["applied"].get("forced")]
    assert [(a["action"], a["why"]) for a in forced] == [("replan", "motionless_repeat_after_no_change")]
    # negative: the same repeat after an answer that changed something (a repoint) is asked as before
    layer = StandIn(script={"turn": [first, repeat]})
    scripted = answers(repairs=(repair("repoint", part="A"), repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 900, stop=outcome_seen("supported_on"))
    assert [k for k in transport.kinds if k != "point_fine"][:3] == ["plan", "repair", "repair"]
    assert "replan" not in transport.kinds


class FailRef(StandIn):
    """StandIn whose registration of part B fails (part_not_found)."""

    def register_regions(self, ctx, plan, roles):
        out = yield from super().register_regions(ctx, plan, roles)
        part = ctx.scene.parts.get("B")
        if part is not None:
            ctx.scene.parts["B"] = dataclasses.replace(part, status="failed", reason="part_not_found")
        return out


def test_the_status_block_names_the_unregistered_ref_of_a_pending_step(scene):
    layer = FailRef()
    policy, transport, world = build(scene, layer, scripted=answers(repairs=(repair("hold_still"),)))
    drive(policy, world, 400, stop=lambda p, d: bool(repair_requests(transport)))
    text, _ = repair_requests(transport)[0]
    assert [(r["status"], r["reason"]) for r in text["plan"]["steps"]] == [("failed", "part_not_found"),
                                                                           ("pending", "part_not_found")]
    assert text["step_history"]["unregistered"] == [["B", "part_not_found"]]


# ----------------------------------------------------------------------------- the runner (ruling L3)
class ExhaustingPolicy(runner.PlumbingHoldPolicy):
    """A runner-contract policy that emits the v2 event recovery_exhausted at act ``at``."""

    at = None

    def act(self, observation):
        decision = super().act(observation)
        rows = [{"event": "recovery_exhausted", "policy_step": self.index, "forced_skips": [0]}] \
            if self.index == self.at else []
        decision.diagnostics["v2_events"] = rows
        return decision


def smoke(tmp_path, scene, name, at, steps=20):
    policy = type("P", (ExhaustingPolicy,), {"at": at})
    return runner.execute(tmp_path / name, config=runner.configuration(),
                          sim_config=runner.build_sim_config("libero_10_t2")[0], task_key="libero_10_t2",
                          episode_index=0, sim_factory=lambda cfg: FakeV2WorldSim(fakes.FakeTwoViewSim(
                              scene, images=False)), video=False, smoke=True, smoke_steps=steps,
                          policy_factory=policy)


def test_recovery_exhausted_ends_the_episode_with_an_automatic_cap(tmp_path, scene, capsys):
    summary = smoke(tmp_path, scene, "exhausted", at=5)
    assert summary["status"] == "recovery_exhausted" and summary["steps"] == 6 and not summary["v2_success"]
    alert = summary["stuck_alerts"][-1]
    assert alert["automatic_cap"] is True and alert["reason"].startswith("recovery_exhausted")
    trace = [json.loads(line) for line in (tmp_path / "exhausted" / "trace.jsonl").read_text().splitlines()]
    assert [row["event"] for row in trace].count("recovery_stop") == 1
    assert "automatic cap" in capsys.readouterr().err
    # negative: no such event -> the smoke runs to its step budget, no automatic cap
    summary = smoke(tmp_path, scene, "plain", at=None)
    assert summary["status"] == "diagnostic_complete" and summary["steps"] == 20
    assert all(a["automatic_cap"] is False for a in summary["stuck_alerts"])


def test_progress_of_the_object_restarts_the_counters_and_rows_cap_at_h(scene):
    """A succeeded held outcome of the step's object restarts ident / stall (not the rung); another object's success
    does not (negative); the history keeps at most REPAIR_HISTORY_H rows."""
    policy, _, world = build(scene, StandIn())
    drive(policy, world, 400, stop=outcome_seen("supported_on"))
    runner_ = policy._runner
    state = runner_.states[1]                                     # place_on D (grip C) onto B
    failure = kernel.RelationOutcome("supported_on", "failed", "lost_in_transport",
                                     {"obj": "D", "grip": ["C"], "ref": "B"}, {}, [], "", 0, 0)
    key = runner_.step_key(state.step)
    for _ in range(3):                                           # the same signature (no answer in between)
        assert runner_._ladder_trigger(state, failure) is None
    other = kernel.RelationOutcome("held", "succeeded", "hold_pass", {"obj": "A", "grip": ["A"]}, {}, [], "", 0, 0)
    runner_.observe_outcome(other, state)
    assert runner_._ladder_trigger(state, failure) == "identical_attempts"      # 4th: no restart by A's success
    runner_._ladder[key]["rung"] = 1
    held = kernel.RelationOutcome("held", "succeeded", "hold_pass", {"obj": "D", "grip": ["C"]}, {}, [], "", 0, 0)
    runner_.observe_outcome(held, state)
    assert (runner_._ladder[key]["stall"], runner_._ladder[key]["ident"], runner_._ladder[key]["rung"]) == (0, 0, 1)
    assert runner_._ladder_trigger(state, failure) is None
    for k in range(P.REPAIR_HISTORY_H + 3):
        runner_._add_row(state, failure, "retry", None, "no_change")
    rows = runner_.step_history(state)["earlier"]
    assert len(rows) == P.REPAIR_HISTORY_H
    C._step_history(runner_.step_history(state))                              # the client accepts it as built
