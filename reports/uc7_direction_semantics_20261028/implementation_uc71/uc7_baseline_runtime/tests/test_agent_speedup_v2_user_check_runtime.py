"""Focused boundary scenarios using real orchestration and inert controller fakes."""

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2 import EventSink
from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_observers import InvariantMonitor, StaysChecker
from gpt_field.agent_speedup_v2_plan import parse_plan, parse_repair, part_roles
from gpt_field.agent_speedup_v2_types import HeightField, HoldTrue, Meas, ObsReport, RobotModel, Seed, StepState, Until
from gpt_field.agent_speedup_v2_user_check_client import LockedStepRunner, PlanLock, ReviewedV2Client
from gpt_field.agent_speedup_v2_user_check_framework import intent_key
from gpt_field.agent_speedup_v2_user_check_runtime import StableScene
from gpt_field.contact_skills import StepView
from gpt_field.observation import PolicyObservation

import agent_speedup_v2_fakes as fakes
from test_agent_speedup_v2 import StandInContext, outcome


PLACE = {"do": "place_on", "obj": "A", "grip": ["B"], "ref": "C", "sense": "any",
         "keep": [{"c": "upright", "p": "A"}], "until": []}
RELEASE = {"do": "release", "obj": "A", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}
HOLD = {"do": "hold", "obj": "D", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}


def raw_plan(*, extra=False):
    parts = [{"id": "A", "text": "cup: body", "move": "free", "w": 1, "x": 1},
             {"id": "B", "text": "cup: handle", "move": "free", "w": 2, "x": 2},
             {"id": "C", "text": "tray: top surface", "move": "fixed", "w": 3, "x": 3}]
    if extra:
        parts.append({"id": "D", "text": "box: body", "move": "free", "w": 4, "x": 4})
    return {"scene": "An object on a table beside a support.", "parts": parts, "hold_true": [],
            "steps": deepcopy([PLACE, HOLD] if extra else [PLACE])}


def view():
    blank = fakes._blank_url()
    return StepView(PolicyObservation(blank, blank, (0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.02, -0.02),
                                     "Put the cup on the tray."))


def runner_fixture(*, extra=False, holding=False, raw=None):
    raw = raw_plan(extra=extra) if raw is None else deepcopy(raw)
    plan = parse_plan(raw)
    robot = RobotModel()
    scene = StableScene(robot)
    scene.hf = HeightField.empty((0.0, 0.0), half_extent_m=0.05)
    current = view()
    scene.merge_plan(plan, part_roles(plan), current, 0)
    for pid, record in list(scene.parts.items()):
        scene.parts[pid] = replace(
            record, xyz=np.array([0.0, 0.0, 0.8]), sigma_h_m=0.001, registration_step=10,
            top_z=Meas(0.8, 0.001, "dense"), support_z=Meas(0.7, 0.001, "plane"),
            status="registered", registered_empty_hand=True,
            region_cells=np.empty((0, 2), dtype=int) if pid == "C" else np.array([[1, 1], [1, 2]]))
    ctx = StandInContext(robot, scene, InvariantMonitor(), StaysChecker(), None, view=current, step=20,
                         holding=SimpleNamespace(pid="A", grip_pid="B") if holding else None)
    ctx.events = EventSink(lambda: ctx.step)
    layer = SimpleNamespace(USER_CHECK_CONTROL_VERSION="offline-contract-controller", CONTROLLERS={})
    lock = PlanLock(raw, task_prompt="Put the cup on the tray.")
    runner = LockedStepRunner(ctx, lock=lock, relations=layer)
    runner.plan = ctx.plan = plan
    runner.roles = ctx.roles = part_roles(plan)
    runner.states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
    runner._sync_framework()
    return runner, lock


def complete(generator):
    try:
        request = next(generator)
    except StopIteration as finished:
        return finished.value
    pytest.fail(f"Unexpected robot action or model request: {request!r}")


def blocked_outcome():
    result = outcome("supported_on", "failed", "precondition_unmet", {"obj": "A", "grip": ["B"], "ref": "C"})
    result.evidence["execution_feedback"] = {
        "status": "blocked", "executed": False,
        "unmet_conditions": [{"predicate": "destination_cells_nonempty", "part": "C", "value": False},
                             {"predicate": "empty_hand_for_observation", "part": "C", "value": False}],
        "evidence": {"region_cells": 0, "holding": "A"},
    }
    return result


def test_stable_scene_preserves_measured_geometry_across_description_edit():
    runner, _ = runner_fixture()
    scene = runner.ctx.scene
    original = scene.parts["C"]
    changed = runner.plan.to_json()
    changed["parts"][2]["text"] = "tray: visible upper support surface"
    plan = parse_plan(changed, source="replan")
    scene.merge_plan(plan, part_roles(plan), runner.ctx.view, 0)
    revised = scene.parts["C"]
    assert revised.text == changed["parts"][2]["text"]
    assert revised.xyz is original.xyz and revised.region_cells is original.region_cells
    assert revised.top_z == original.top_z and revised.registration_step == original.registration_step
    assert revised.status == "registered" and "C" not in scene.retired


def test_held_unusable_destination_requires_executable_resource_changing_replan():
    runner, _ = runner_fixture(holding=True)
    same = parse_plan(raw_plan(), source="replan")
    issues = runner._plan_issues(same)
    assert "skill_precondition" in {issue.code for issue in issues}
    recovery = raw_plan()
    recovery["steps"].insert(0, deepcopy(RELEASE))
    assert runner._plan_issues(parse_plan(recovery, source="replan")) == []
    # Readiness and plan checking must not silently release a held object.
    assert runner.ctx.holding.pid == "A"
    assert runner.ctx.pending_outcomes == []


def test_pending_registration_never_becomes_reregistered_in_apply_or_history(monkeypatch):
    runner, lock = runner_fixture()
    state = runner.states[0]
    state.status, state.attempts = "failed", 1
    runner._awaiting.add("C")
    runner.ctx.memory = {}
    previous = runner.ctx.scene.parts["C"]
    seed = Seed("wrist", (250.0, 250.0), runner.ctx.step, "coarse", 32.0, tuple(runner.ctx.view.state))

    def point(ctx, pid, **kwargs):
        assert pid == "C"
        return seed
        yield

    def register(pid, view, **kwargs):
        assert kwargs["seed"] is seed
        # Fake perception supplies a new observation but still no usable cells.
        runner.ctx.scene.parts[pid] = replace(previous, seeds=[seed])
        return runner.ctx.scene.parts[pid]

    runner.relations._coarse_seed = point
    runner.relations._seed_lift = lambda *args: None
    monkeypatch.setattr(runner.ctx.scene, "register", register)

    def survey(ctx, pid, *, purpose, need):
        assert pid == "C" and not need(ctx.scene.parts[pid])
        ctx.step += 3
        # A controller's successful survey still left no usable support region.
        result = outcome("registered", "succeeded", "registered", {"obj": pid}, steps=3)
        ctx.record(result)
        return result
        yield

    runner.relations.survey_hover = survey
    failed = blocked_outcome()
    answer = parse_repair({"why": "Obtain usable region evidence", "action": "repoint", "part": "C", "step": None})
    result = complete(runner.apply(answer, state, review=False, outcome=failed))
    assert result["registration"] == "pending_verification"
    assert "C" in runner._awaiting
    assert lock._last_feedback["status"] == "pending_verification"
    assert lock._last_feedback["observed_effects"] == []
    runner._add_row(state, failed, "repoint", "C", "reregistered")
    assert runner.step_history(state)["earlier"][-1]["effect"] == "no_change"
    assert lock._decisions[-1]["result"]["effect"] == "no_change"


def test_replan_adoption_preserves_counters_memory_and_live_completion_evidence():
    runner, lock = runner_fixture(extra=True)
    finished, pending = runner.states
    finished.status = "done"
    source = "stays:supported_on:A"
    report = ObsReport("offline_relation", "met", None, None, {}, (), runner.ctx.step)
    runner.ctx.stays.add(source, lambda _: report, step_i=0, kind="step")
    runner._owner[source] = finished
    runner.last_stays[source] = report
    key = intent_key(finished.step)
    step_key = runner.step_key(pending.step)
    ladder = runner._ladder_of(step_key)
    ladder.update(ident=3, stall=7, nochange=2)
    runner.capture_feedback(outcome("held", "failed", "no_feasible_candidate", {"obj": "D"}), pending)
    new_raw = raw_plan(extra=True)
    new_raw["steps"] = [deepcopy(HOLD)]
    # Native schema rejects unused objects; omit their plan entries while the
    # runner retains their already verified relation observers.
    new_raw["parts"] = [part for part in new_raw["parts"] if part["id"] == "D"]
    replanned = parse_plan(new_raw, source="replan")
    assert runner._plan_issues(replanned) == []
    answer = SimpleNamespace(view=runner.ctx.view, need=SimpleNamespace(quarter_turns=0))
    complete(runner.adopt(replanned, answer, replan=True))
    assert key in runner.completed_intents()
    assert key in lock.framework_snapshot()["verified_intents"]
    assert runner._ladder[step_key]["ident"] == 3
    assert runner._ladder[step_key]["stall"] == 7
    assert runner._ladder[step_key]["nochange"] == 2
    assert len(runner.recovery_memory.entries) == 1
    runner.last_stays[source] = replace(report, verdict="pending")
    assert key not in runner.completed_intents()


def test_actual_repair_and_replan_payloads_keep_structured_unmet_evidence():
    runner, lock = runner_fixture(holding=True)
    state = runner.states[0]
    state.status = "failed"
    failed = blocked_outcome()
    runner.capture_feedback(failed, state)
    transport = ScriptedV2Transport(FixedAnswers())
    client = ReviewedV2Client(task_prompt=lock.task_prompt, approved_plan=lock.plan(), lock=lock, transport=transport)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    _, repair_request, _ = client._repair_request(
        rgb, rgb, quarter_turns=0, status_block=runner.status_block(),
        last_outcome=failed.repair_payload(), holding="A", step_history=runner.step_history(state))
    _, plan_request, _ = client._plan_request(
        rgb, rgb, quarter_turns=0, fingertip_edge="bottom", holding="A", replan=True,
        status_block=runner.status_block(), step_history=runner.step_history(state))
    for request in (repair_request, plan_request):
        framework = request["fields"]["execution_framework"]
        assert framework["last_feedback"]["status"] == "blocked"
        assert {item["predicate"] for item in framework["last_feedback"]["unmet_conditions"]} == {
            "destination_cells_nonempty", "empty_hand_for_observation"}
        assert framework["last_feedback"]["evidence"]["measurements"]["execution_feedback"]["evidence"]["region_cells"] == 0
        assert framework["state"]["holding"] == "A"
        assert framework["recovery"]["total_outcomes"] == 1
    assert transport.calls == []


def test_completed_status_does_not_transfer_to_changed_until_or_sense():
    runner, _ = runner_fixture(extra=True)
    turning = replace(runner.states[1].step, do="turn", sense="any",
                      until=(Until("quarter_turn", None, None),))
    old = StepState.pending(0, turning)
    old.status = "done"
    changed_until = StepState.pending(0, replace(turning, until=(Until("half_turn", None, None),)))
    changed_sense = StepState.pending(1, replace(turning, sense="open"))
    same_intent = StepState.pending(2, turning)
    runner._owner["turn-observer"] = old
    runner._completed_at[id(old)] = runner.ctx.step
    runner._carry_done_steps([old], [changed_until, changed_sense, same_intent])
    assert changed_until.status == changed_sense.status == "pending"
    assert same_intent.status == "stays"
    assert runner._owner["turn-observer"] is same_intent


def test_hold_true_relay_uses_current_source_despite_its_local_check_counter():
    runner, _ = runner_fixture()
    state = runner.states[0]
    state.status = "done"
    source = "stays:supported_on:A"
    counts = {"n": 0, "verdict": "met"}

    def source_check(view):
        counts["n"] += 1
        return ObsReport("support_observer", counts["verdict"], None, None,
                         {"observed_support": counts["verdict"]}, (), counts["n"])

    runner.ctx.stays.add(source, source_check, step_i=0, kind="step")
    runner._owner[source] = state
    claim = HoldTrue("on", "A", "C")
    relay, reason = runner.hold_true_check(claim)
    assert reason is None
    runner.ctx.stays.add("reviewed-claim", relay, step_i=None, kind="hold_true")
    runner._hold_true["reviewed-claim"] = {"claim": claim, "escalated": False}
    # A replaced source moves behind the relay; boundary ordering must restore
    # source-before-dependent evaluation rather than using last boundary data.
    runner.ctx.stays.add(source, source_check, step_i=0, kind="step")
    complete(runner.stays_boundary_check())
    assert counts["n"] == 1 and runner.ctx.step == 20
    assert runner.last_stays["reviewed-claim"].verdict == "met"
    assert runner.last_stays["reviewed-claim"].step == 20
    counts["verdict"] = "pending"
    runner.ctx.step = 30
    complete(runner.stays_boundary_check())
    assert counts["n"] == 2
    assert runner.last_stays["reviewed-claim"].verdict == "pending"


def test_verified_intermediate_hold_remains_completed_after_resource_is_consumed():
    approved = raw_plan(extra=True)
    intermediate = deepcopy(PLACE)
    intermediate.update(do="hold", ref=None)
    approved["steps"].insert(0, intermediate)
    runner, lock = runner_fixture(raw=approved)
    preparatory_hold, placement, final_hold = runner.states
    assert runner.completed_intents() == []  # A reviewed intention is not evidence.
    # Production hold reports verified success without a stays observer.
    runner.capture_feedback(outcome("held", "succeeded", "hold_pass", {"obj": "A", "grip": ["B"]}),
                            preparatory_hold)
    assert runner.completed_intents() == [intent_key(preparatory_hold.step)]
    met = ObsReport("offline_relation", "met", None, None, {}, (), runner.ctx.step)
    for key, state in (("preparatory-hold", preparatory_hold), ("placement", placement), ("final-hold", final_hold)):
        state.status = "done"
        runner._owner[key] = state
        runner.last_stays[key] = met
    completed = runner.completed_intents()
    assert len(completed) == 3
    transient_key = intent_key(preparatory_hold.step)
    # Successful placement consumes the earlier holding relation. The next
    # plan can omit this achieved waypoint without grasping the placed object.
    runner._owner.pop("preparatory-hold")
    runner.last_stays.pop("preparatory-hold")
    runner.states = runner.states[1:]
    assert transient_key in runner.completed_intents()
    # Final predicates still require current supporting evidence.
    runner.last_stays["placement"] = replace(met, verdict="pending")
    runner.last_stays["final-hold"] = replace(met, verdict="violated")
    assert runner.completed_intents() == [transient_key]
    revised = deepcopy(approved)
    revised["steps"] = revised["steps"][1:]
    assert lock.task_contract.validate_replan(revised, runner.completed_intents()) == []
