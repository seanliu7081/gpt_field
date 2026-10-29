"""Offline integration checks for the separate, user-approved v2 variant.

The original controller, runner, and existing fixtures stay untouched.  These tests
exercise the boundaries where ordinary v2 is allowed to change a semantic plan.
No real model, credentials, or robotics simulator is used.
"""

from __future__ import annotations

import copy
from functools import partial

from gpt_field.agent_speedup_v2_client import ScriptedV2Transport as _ScriptedV2Transport

ScriptedV2Transport = partial(_ScriptedV2Transport, effort="high")
from types import SimpleNamespace

import pytest

from gpt_field.agent_speedup_v2 import EventSink
from gpt_field.agent_speedup_v2_plan import parse_plan, parse_repair
from gpt_field.agent_speedup_v2_types import StepState
from gpt_field.agent_speedup_v2_user_check_client import (
    LockedStepRunner,
    PlanApprovalRequired,
    PlanLock,
    ReviewedV2Policy,
)
from test_agent_speedup_v2 import (
    HOLD_D,
    PLACE,
    TASK2,
    TASK2_PLAN,
    StandIn,
    answers,
    make_client,
    make_world,
    outcome,
    plan_json,
    repair,
)
import agent_speedup_v2_fakes as fakes


def hold_only_plan():
    return plan_json(steps=[HOLD_D], parts=[p for p in TASK2_PLAN["parts"] if p["id"] in ("C", "D")])


def locked_runner(raw=None, *, allow_auto_replan=False):
    approved = copy.deepcopy(raw or TASK2_PLAN)
    lock = PlanLock(approved_plan=approved, allow_auto_replan=allow_auto_replan)
    ctx = SimpleNamespace(events=EventSink(lambda: 0), plan=None, holding=None, view=None,
                          scene=SimpleNamespace(parts={}), step=0)
    runner = LockedStepRunner(ctx, lock=lock, relations=SimpleNamespace())
    runner.plan = parse_plan(approved)
    ctx.plan = runner.plan
    runner.states = [StepState.pending(i, step) for i, step in enumerate(runner.plan.steps)]
    failed = runner.states[0]
    failed.status = "failed"
    failed.attempts = 1
    return runner, lock, failed


def exhaust(generator):
    """Run a zero-action runner operation and return its generator result."""
    try:
        yielded = next(generator)
    except StopIteration as exc:
        return exc.value
    pytest.fail(f"Operation unexpectedly requested an action or a model call: {yielded!r}")


@pytest.mark.parametrize("action", ["insert_step", "skip_step", "swap_next", "replan"])
@pytest.mark.parametrize("review", [False, True])
def test_semantic_repairs_are_refused_before_plan_or_status_mutation(action, review):
    runner, lock, failed = locked_runner()
    before = runner.plan.to_json()
    statuses = [state.status for state in runner.states]
    response = parse_repair(repair(action, step=HOLD_D if action == "insert_step" else None))

    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.apply(response, failed, review=review, outcome=None))

    assert runner.plan.to_json() == before
    assert [state.status for state in runner.states] == statuses
    assert runner._grip == {}
    assert lock.conflict is not None


@pytest.mark.parametrize("rung", [0, 1])
def test_automatic_recovery_ladder_cannot_replan_or_skip_without_user(rung):
    runner, lock, failed = locked_runner()
    runner._ladder_of(runner.step_key(failed.step))["rung"] = rung
    before = runner.plan.to_json()
    failure = outcome("turned", "failed", "no_feasible_candidate", {"obj": "A", "grip": ["A"]})

    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.climb(failed, failure, why="identical_attempts"))

    assert runner.plan.to_json() == before
    assert failed.status == "failed"
    assert runner._forced == set()
    assert lock.conflict is not None


def test_use_part_cannot_select_an_object_outside_this_steps_approved_grips():
    runner, lock, failed = locked_runner(plan_json(steps=[PLACE]))
    # D is a known scene object (the pot body), but only C (its handle) was approved.
    response = parse_repair(repair("use_part", part="D"))
    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.apply(response, failed, review=False, outcome=None))
    assert runner.grip_of(failed) == ("C",)
    assert failed.status == "failed"
    assert lock.conflict is not None


def test_use_part_can_select_a_preapproved_alternative_without_changing_plan(monkeypatch):
    place = copy.deepcopy(PLACE)
    place["grip"] = ["C", "D"]
    runner, lock, failed = locked_runner(plan_json(steps=[place]))
    original = runner.plan.to_json()
    monkeypatch.setattr(runner, "_refresh_roles", lambda: iter(()))
    monkeypatch.setattr(runner, "_held_grip_for", lambda state: None)

    exhaust(runner.apply(parse_repair(repair("use_part", part="D")), failed,
                         review=False, outcome=None))

    assert runner.grip_of(failed) == ("D", "C")
    assert runner.plan.to_json() == original
    assert failed.status == "pending"
    assert lock.conflict is None


def test_empty_grip_means_only_the_approved_object_not_every_scene_part(monkeypatch):
    hold = copy.deepcopy(HOLD_D)
    hold["grip"] = []
    runner, lock, failed = locked_runner(plan_json(steps=[hold]))
    monkeypatch.setattr(runner, "_refresh_roles", lambda: iter(()))
    monkeypatch.setattr(runner, "_held_grip_for", lambda state: None)
    exhaust(runner.apply(parse_repair(repair("use_part", part="D")), failed,
                         review=False, outcome=None))
    assert runner.grip_of(failed) == ("D",)
    assert lock.conflict is None
    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.apply(parse_repair(repair("use_part", part="C")), failed,
                             review=False, outcome=None))


def test_retry_and_repoint_preserve_approved_semantics(monkeypatch):
    runner, lock, failed = locked_runner()
    before = runner.plan.to_json()
    exhaust(runner.apply(parse_repair(repair("retry")), failed, review=False, outcome=None))
    assert failed.status == "pending"
    dropped = []
    runner.ctx.scene = SimpleNamespace(parts={}, drop=dropped.append)

    def register_missing():
        return []
        yield

    monkeypatch.setattr(runner, "register_missing", register_missing)
    exhaust(runner.apply(parse_repair(repair("repoint", part="A")), failed,
                         review=False, outcome=None))
    assert dropped == ["A"]
    assert runner.plan.to_json() == before
    assert lock.conflict is None


def test_direct_replan_request_is_blocked_before_it_can_yield_a_model_need():
    runner, lock, _ = locked_runner()
    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.obtain_plan("replan"))
    assert lock.conflict is not None


def test_replan_adoption_is_blocked_before_scene_registration():
    runner, lock, _ = locked_runner()
    before = runner.plan.to_json()
    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.adopt(parse_plan(plan_json(steps=[HOLD_D]), source="replan"), None, replan=True))
    assert runner.plan.to_json() == before
    assert lock.conflict is not None



@pytest.mark.parametrize("change", ["object", "step_order", "part_description", "constraint"])
def test_initial_adoption_cannot_sneak_in_semantic_changes(change):
    runner, lock, _ = locked_runner()
    changed = copy.deepcopy(TASK2_PLAN)
    if change == "object":
        changed["steps"][1]["obj"] = "E"
    elif change == "step_order":
        changed["steps"].reverse()
    elif change == "part_description":
        changed["parts"][2]["text"] = "frying pan: long handle"
    else:
        changed["steps"][1]["keep"] = []
    before = runner.plan.to_json()
    with pytest.raises(PlanApprovalRequired):
        exhaust(runner.adopt(parse_plan(changed), None, replan=False))
    assert runner.plan.to_json() == before
    assert lock.conflict is not None


def test_fake_policy_stops_at_semantic_conflict_before_next_action():
    approved = hold_only_plan()
    lock = PlanLock(approved_plan=approved, allow_auto_replan=False)
    client, transport = make_client(answers(plans=[approved], repairs=[repair("replan")]))
    layer = StandIn(script={"hold": [{"status": "failed", "reason": "no_feasible_candidate"}]})
    policy = ReviewedV2Policy(client, lock=lock, relations=layer)
    world = make_world(fakes.task2_like_scene(), TASK2)
    observation = world.observation()
    completed = 0

    with pytest.raises(PlanApprovalRequired):
        for _ in range(50):
            decision = policy.act(observation)
            after = world.step(decision.action)
            policy.observe_transition(observation, decision, after)
            observation = after
            completed += 1

    assert completed == world.steps
    assert completed > 0
    assert lock.conflict is not None
    assert "replan" not in transport.kinds
    assert policy._runner.plan.to_json() == approved
    assert [row[1] for row in layer.log if row[0] == "controller"] == ["hold"]
    with pytest.raises(PlanApprovalRequired):
        policy.act(observation)
    assert world.steps == completed


def load_review_runner():
    import importlib.util
    from pathlib import Path
    import sys

    name = "evaluate_agent_speedup_v2_user_check_integration"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate_agent_speedup_v2_user_check.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def approved_fake_run(tmp_path, *, conflict=False):
    from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
    from gpt_field.agent_speedup_v2_user_check import ReviewStore
    from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client
    from test_agent_speedup_v2_chain import runner as original_runner

    module = load_review_runner()
    config = original_runner.configuration()
    sim_config, monitor = original_runner.build_v2_sim_config("libero_10_t2")
    task = resolve_v2_task("libero_10_t2")
    sim = fakes.FakeV2WorldSim(fakes.FakeTwoViewSim(fakes.task2_like_scene(), images=False))
    initial = sim.reset(task.task_id, 0).to_dict()
    approved = hold_only_plan() if conflict else copy.deepcopy(TASK2_PLAN)
    inventory = module.runtime_inventory(config, sim_config, monitor_options=monitor,
                                         video=False, stop_after_steps=40)
    review = ReviewStore.create(tmp_path / "review", task_key=task.key, episode_index=0,
                                task_prompt=task.prompt, initial_observation=initial,
                                plan=approved, runtime_sha256=inventory)
    review.approve(expected_revision=0, reviewed_steps=list(range(len(approved["steps"]))))
    scripted = answers(plans=[approved], repairs=[repair("replan")] if conflict else [repair("hold_still")])

    def make_reviewed(config, *, approved_plan, lock, **kwargs):
        # Retain a regression for the optional strict adapter, while production
        # execution now defaults to automatic model recovery.
        if conflict:
            lock.allow_auto_replan = False
        kwargs["transport"] = ScriptedV2Transport(scripted)
        return ReviewedV2Client(config, approved_plan=approved_plan, lock=lock, **kwargs)

    def policy_factory(client, *, lock):
        layer = StandIn(script={"hold": [{"status": "failed", "reason": "no_feasible_candidate"}]}) \
            if conflict else StandIn()
        return ReviewedV2Policy(client, lock=lock, relations=layer)

    kwargs = dict(review_dir=review.path, config=config, sim_config=sim_config,
                  task_key=task.key, episode_index=0, monitor_options=monitor,
                  token_provider=lambda: "offline-test-token", video=False, stop_after_steps=40,
                  client_factory=make_reviewed, policy_factory=policy_factory,
                  sim_factory=lambda config: sim)
    return module, review, sim, kwargs


def test_full_fake_runner_consumes_review_and_refuses_reuse(tmp_path):
    from gpt_field.agent_speedup_v2_user_check import ReviewError

    module, review, sim, kwargs = approved_fake_run(tmp_path)
    summary = module.execute_reviewed(tmp_path / "first", **kwargs)
    assert summary["status"] == "user_stop"
    assert summary["steps"] == 40
    assert len(sim.actions) == 40
    assert review.load()["status"] == "consumed"
    before = list(sim.actions)
    with pytest.raises(ReviewError):
        module.execute_reviewed(tmp_path / "second", **kwargs)
    assert sim.actions == before
    assert not (tmp_path / "second").exists()


def test_full_fake_runner_marks_plan_conflict_consistently(tmp_path):
    import json

    module, review, sim, kwargs = approved_fake_run(tmp_path, conflict=True)
    output = tmp_path / "conflict"
    summary = module.execute_reviewed(output, **kwargs)
    assert summary["status"] == "plan_conflict", summary.get("error")
    assert summary["task_success"] is False
    assert summary["v2_success"] is False
    assert 0 < summary["steps"] < 40
    assert summary["pending_action_index"] is None
    assert summary["steps"] == len(sim.actions)
    assert review.load()["status"] == "consumed"
    for filename in ("summary.json", "runner_summary.json", "task_evaluation.json", "run_state.json"):
        artifact = json.loads((output / filename).read_text())
        assert artifact["status"] == "plan_conflict", filename
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert episodes[-1]["status"] == "plan_conflict"


def test_full_fake_runner_rejects_changed_reset_before_any_action(tmp_path):
    module, review, sim, kwargs = approved_fake_run(tmp_path)
    reset = sim.reset

    def changed_reset(task_id, episode_index):
        reset(task_id, episode_index)
        sim.world.grip[0] += 0.01
        sim.world._obs_cache = None
        return sim.world.observation()

    sim.reset = changed_reset
    summary = module.execute_reviewed(tmp_path / "wrong-reset", **kwargs)
    assert summary["task_success"] is False
    assert summary["steps"] == 0
    assert sim.actions == []
    assert review.load()["status"] == "consumed"


def make_grounding_client(reply, approved=None, *, allow_auto_replan=False):
    from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client

    approved = copy.deepcopy(approved or TASK2_PLAN)
    lock = PlanLock(approved_plan=approved, allow_auto_replan=allow_auto_replan)
    transport = ScriptedV2Transport(answers(plans=[reply]))
    client = ReviewedV2Client(task_prompt=TASK2, approved_plan=approved, lock=lock, transport=transport)
    return client, transport, lock


def call_grounding(client, **kwargs):
    from test_agent_speedup_v2_client import image

    return client.request_plan(image(1), image(2), quarter_turns=0,
                               fingertip_edge="bottom", **kwargs)


def test_grounding_refreshes_only_cells_and_sends_entire_approved_plan():
    from test_agent_speedup_v2_client import request_text

    grounded = copy.deepcopy(TASK2_PLAN)
    for part in grounded["parts"]:
        part["w"], part["x"] = 0, 64
    client, transport, lock = make_grounding_client(grounded)
    result = call_grounding(client)
    assert result == grounded
    assert lock.conflict is None
    assert lock.plan() == TASK2_PLAN
    sent = request_text(transport.calls[0])
    assert sent["user_approved_plan"] == TASK2_PLAN
    assert transport.metas[0]["user_check_grounding_only"] is True
    assert "USER CHECK EXECUTION" in transport.calls[0]["instructions"]


@pytest.mark.parametrize("change", ["scene", "object", "description", "step_order", "constraint"])
def test_grounding_refuses_model_rewriting_user_choices(change):
    changed = copy.deepcopy(TASK2_PLAN)
    if change == "scene":
        changed["scene"] = "A different scene proposed by the model"
    elif change == "object":
        changed["steps"][1]["obj"] = "E"
    elif change == "description":
        changed["parts"][2]["text"] = "frying pan: long handle"
    elif change == "step_order":
        changed["steps"].reverse()
    else:
        changed["steps"][1]["keep"] = []
    client, transport, lock = make_grounding_client(changed)
    with pytest.raises(PlanApprovalRequired):
        call_grounding(client)
    assert lock.conflict is not None
    assert len(transport.calls) == 1
    assert lock.plan() == TASK2_PLAN
    # Once a conflict is latched, a retry cannot silently resume or call the model.
    with pytest.raises(PlanApprovalRequired):
        call_grounding(client)
    assert len(transport.calls) == 1


def test_client_replan_is_refused_without_any_model_call():
    client, transport, lock = make_grounding_client(TASK2_PLAN)
    with pytest.raises(PlanApprovalRequired):
        call_grounding(client, replan=True)
    assert transport.calls == []
    assert lock.conflict is not None


def test_plan_lock_is_a_snapshot_not_a_reference_to_mutable_input():
    approved = copy.deepcopy(TASK2_PLAN)
    lock = PlanLock(approved_plan=approved)
    approved["steps"].reverse()
    lock.approved_plan["steps"].clear()
    exposed = lock.plan()
    exposed["parts"][0]["text"] = "Changed by a caller"
    assert lock.plan() == TASK2_PLAN
    lock.check_plan(TASK2_PLAN, source="snapshot_test")
    assert lock.conflict is None


def test_runtime_repair_request_carries_approved_plan_and_lock_instructions():
    from gpt_field.agent_speedup_v2_plan import plan_status_block
    from test_agent_speedup_v2_client import image, request_text

    client, transport, lock = make_grounding_client(TASK2_PLAN)
    plan = parse_plan(TASK2_PLAN)
    states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
    client.request_repair(image(1), image(2), quarter_turns=0,
                          status_block=plan_status_block(plan, states, holding=None),
                          last_outcome=None, holding=None, review=True)
    assert request_text(transport.calls[0])["user_approved_plan"] == TASK2_PLAN
    assert transport.metas[0]["user_check_semantics_locked"] is True
    assert "no in-episode human editing" in transport.calls[0]["instructions"]
    assert lock.conflict is None


def test_model_draft_preparation_has_no_robot_actions_and_requires_user_approval(tmp_path):
    from test_agent_speedup_v2_chain import runner as original_runner

    module = load_review_runner()
    config = original_runner.configuration()
    sim_config, monitor = original_runner.build_v2_sim_config("libero_10_t2")
    sim = fakes.FakeV2WorldSim(fakes.FakeTwoViewSim(fakes.task2_like_scene(), images=False))
    transport = ScriptedV2Transport(answers())
    store = module.prepare_review(
        tmp_path / "prepared", config=config, sim_config=sim_config,
        task_key="libero_10_t2", episode_index=0, monitor_options=monitor,
        token_provider=lambda: "offline-test-token", sim_factory=lambda config: sim,
        transport=transport, video=False, stop_after_steps=40)

    review = store.load()
    assert sim.actions == []
    assert sim.closed
    assert transport.kinds == ["plan"]
    assert review["status"] == "draft"
    assert review["approved"] is None
    assert review["reviewed_steps"] == []
    assert review["preparation"]["policy_actions"] == 0
    assert review["plan"] == TASK2_PLAN
    assert len(review["preparation"]["review_images"]) == 4


def test_unapproved_review_cannot_start_the_simulator_or_model(tmp_path):
    from gpt_field.agent_speedup_v2_user_check import ReviewError

    module, review, sim, kwargs = approved_fake_run(tmp_path)
    state = review.load()
    review.update(state["plan"], expected_revision=state["revision"], reviewed_steps=[])
    attempted = []

    def forbidden_factory(*args, **kwargs):
        attempted.append(True)
        pytest.fail("Unapproved review started execution")

    kwargs.update(sim_factory=forbidden_factory, client_factory=forbidden_factory)
    with pytest.raises(ReviewError):
        module.execute_reviewed(tmp_path / "unapproved", **kwargs)
    assert attempted == []
    assert sim.actions == []
    assert not (tmp_path / "unapproved").exists()
    assert review.load()["status"] == "draft"


def test_prepare_wrong_initial_state_makes_no_model_call_or_robot_action(tmp_path):
    from gpt_field.agent_speedup_v2_user_check import ReviewError
    from test_agent_speedup_v2_chain import runner as original_runner

    module = load_review_runner()
    config = original_runner.configuration()
    sim_config, monitor = original_runner.build_v2_sim_config("libero_10_t2")
    sim = fakes.FakeV2WorldSim(fakes.FakeTwoViewSim(fakes.task2_like_scene(), images=False))
    transport = ScriptedV2Transport(answers())
    with pytest.raises(ReviewError):
        module.prepare_review(
            tmp_path / "wrong-prepare", config=config, sim_config=sim_config,
            task_key="libero_10_t2", episode_index=0, monitor_options=monitor,
            token_provider=lambda: "offline-test-token", sim_factory=lambda config: sim,
            transport=transport, expected_initial_observation_sha256="0" * 64,
            video=False, stop_after_steps=40)
    assert transport.calls == []
    assert sim.actions == []
    assert sim.closed


def test_auto_replan_is_default_and_audit_snapshots_do_not_mutate_approval():
    lock = PlanLock(approved_plan=TASK2_PLAN)
    assert lock.allow_auto_replan is True
    assert lock.snapshot()["approval_scope"] == "initial_plan_only"
    snapshot = lock.snapshot()
    snapshot["initial_approved_plan"]["steps"].clear()
    snapshot["current_plan"]["steps"].clear()
    assert lock.plan() == TASK2_PLAN
    assert lock.snapshot()["current_plan"] == TASK2_PLAN


def test_automatic_execution_still_locks_initial_grounding():
    changed = copy.deepcopy(TASK2_PLAN)
    changed["steps"][1]["grip"] = ["D"]
    client, transport, lock = make_grounding_client(changed, allow_auto_replan=True)
    with pytest.raises(PlanApprovalRequired):
        call_grounding(client)
    assert lock.conflict is not None
    assert lock.snapshot()["plan_revision_count"] == 0
    assert len(transport.calls) == 1


def test_automatic_replan_prompt_carries_task_current_state_history_and_initial_plan():
    from gpt_field.agent_speedup_v2_plan import plan_status_block
    from test_agent_speedup_v2_client import request_text

    revised = copy.deepcopy(TASK2_PLAN)
    revised["steps"][1]["grip"] = ["D"]
    client, transport, lock = make_grounding_client(revised, allow_auto_replan=True)
    plan = parse_plan(TASK2_PLAN)
    states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
    states[0].status = "succeeded"
    states[1].status = "failed"
    history = {"step": 1, "attempts": 1, "earlier": [],
               "this_attempt": [["C", "lateral_unreachable"]], "held_by": "C", "unregistered": []}
    status = plan_status_block(plan, states, holding="D")
    result = call_grounding(client, replan=True, holding="D", status_block=status, step_history=history)

    assert result == revised
    assert lock.conflict is None
    assert transport.kinds == ["replan"]
    sent = request_text(transport.calls[0])
    assert sent["task"] == TASK2
    assert sent["holding"] == "D"
    assert sent["status"] == status
    assert sent["step_history"] == history
    assert sent["initial_user_approved_plan"] == TASK2_PLAN
    assert sent["approval_scope"] == "initial_plan_only"
    assert transport.metas[0]["user_check_grounding_only"] is False
    assert "Do not request another human review" in transport.calls[0]["instructions"]
    assert "replans or skips the step itself" not in transport.calls[0]["instructions"]


def test_automatic_repair_prompt_does_not_claim_later_plans_are_user_approved():
    from gpt_field.agent_speedup_v2_plan import plan_status_block
    from test_agent_speedup_v2_client import image, request_text

    client, transport, lock = make_grounding_client(TASK2_PLAN, allow_auto_replan=True)
    plan = parse_plan(TASK2_PLAN)
    states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
    client.request_repair(image(1), image(2), quarter_turns=0,
                          status_block=plan_status_block(plan, states, holding=None),
                          last_outcome=None, holding=None)
    sent = request_text(transport.calls[0])
    assert sent["automatic_replanning_allowed"] is True
    assert sent["approval_scope"] == "initial_plan_only"
    assert transport.metas[0]["user_check_semantics_locked"] is False
    assert "Later changes are model decisions, not user approvals" in transport.calls[0]["instructions"]
    assert "replans or skips the step itself" not in transport.calls[0]["instructions"]


def test_automatic_use_part_can_change_grip_and_records_effective_revision(monkeypatch):
    runner, lock, failed = locked_runner(plan_json(steps=[PLACE]), allow_auto_replan=True)
    original = lock.plan()
    monkeypatch.setattr(runner, "_refresh_roles", lambda: iter(()))
    monkeypatch.setattr(runner, "_held_grip_for", lambda state: None)
    exhaust(runner.apply(parse_repair(repair("use_part", part="D")), failed,
                         review=False, outcome=None))
    assert runner.grip_of(failed) == ("D", "C")
    assert failed.status == "pending"
    audit = lock.snapshot()
    assert audit["initial_approved_plan"] == original
    assert audit["current_plan"]["steps"][0]["grip"] == ["D", "C"]
    assert audit["plan_revision_count"] == 1
    assert audit["revisions"][0]["source"] == "model_use_part"
    assert audit["revisions"][0]["user_approved"] is False
    assert audit["model_decisions"][0]["status"] == "applied"
    assert lock.conflict is None


def test_fake_policy_automatically_replans_and_keeps_completed_work():
    from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client
    from test_agent_speedup_v2_client import request_text

    approved = copy.deepcopy(TASK2_PLAN)
    revised = copy.deepcopy(approved)
    revised["steps"][1]["grip"] = ["D"]
    lock = PlanLock(approved_plan=approved)
    transport = ScriptedV2Transport(answers(plans=[approved, revised],
                                           repairs=[repair("replan"), repair("hold_still")]))
    client = ReviewedV2Client(task_prompt=TASK2, approved_plan=approved, lock=lock, transport=transport)
    layer = StandIn(script={"place_on": [{"status": "failed", "reason": "lateral_unreachable",
                                         "hold": True}]})
    policy = ReviewedV2Policy(client, lock=lock, relations=layer)
    world = make_world(fakes.task2_like_scene(), TASK2)
    observation = world.observation()
    for _ in range(70):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    assert transport.kinds.count("replan") == 1
    assert len(layer.controller_calls("turn")) == 1
    assert len(layer.controller_calls("place_on")) == 2
    assert layer.controller_calls("place_on")[-1][4] == ("D",)
    assert lock.conflict is None
    audit = lock.snapshot()
    assert audit["initial_plan_grounded"] is True
    assert audit["plan_revision_count"] == 1
    assert audit["current_plan"] == revised
    assert audit["initial_approved_plan"] == approved
    assert [row["user_approved"] for row in audit["revisions"]] == [True, False]
    replan_request = next(request_text(call) for call in transport.calls
                          if request_text(call)["request"] == "replan")
    assert replan_request["status"]["steps"][0]["status"] == "done"
    assert replan_request["step_history"]["attempts"] == 1


@pytest.mark.parametrize("rung", [0, 1, 4])
def test_automatic_recovery_requests_model_replan_instead_of_forced_skip(monkeypatch, rung):
    runner, lock, failed = locked_runner(allow_auto_replan=True)
    runner._ladder_of(runner.step_key(failed.step))["rung"] = rung
    calls = []

    def ask(kind, state, failure, *, history):
        calls.append((kind, history))
        return None, {"fallback": "ladder"}
        yield

    def obtain(kind, *, history):
        calls.append((kind, history))
        return runner.plan, None
        yield

    monkeypatch.setattr(runner, "obtain_plan", obtain)
    monkeypatch.setattr(runner, "ask", ask)
    monkeypatch.setattr(runner, "adopt", lambda *args, **kwargs: iter(()))
    failure = outcome("turned", "failed", "no_feasible_candidate", {"obj": "A", "grip": ["A"]})
    result = exhaust(runner.climb(failed, failure, why="identical_attempts"))
    assert [call[0] for call in calls] == ["repair", "replan"]
    assert calls[0][1]["step"] == 0
    assert result["action"] == "replan"
    assert result["replan"] == "adopted"
    assert failed.status != "skipped"
    assert runner._forced == set()
    assert lock.conflict is None


def test_invalid_automatic_replan_does_not_skip_an_unfinished_task(monkeypatch):
    runner, lock, failed = locked_runner(allow_auto_replan=True)
    recovery = []

    def ask(kind, state, failure, *, history):
        recovery.append((kind, history["step"]))
        return None, {"fallback": "ladder"}
        yield

    def obtain(*args, **kwargs):
        return None, None
        yield

    monkeypatch.setattr(runner, "obtain_plan", obtain)
    monkeypatch.setattr(runner, "ask", ask)
    recorded = []
    monkeypatch.setattr(runner, "_record_invalid_plan", recorded.append)
    failure = outcome("turned", "failed", "no_feasible_candidate", {"obj": "A", "grip": ["A"]})
    result = exhaust(runner.climb(failed, failure, why="identical_attempts"))
    assert result["replan"] == "invalid_plan"
    assert recovery == [("repair", 0)]
    assert recorded == [None]
    assert failed.status == "failed"
    assert runner.halted is True
    assert not runner._forced
    assert lock.conflict is None


@pytest.mark.parametrize("change", ["reserved_relation", "ignored_sense"])
def test_automatic_replan_capability_errors_are_reasked_without_executing_them(change):
    from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client
    from test_agent_speedup_v2_client import request_text

    approved = hold_only_plan()
    bad = copy.deepcopy(approved)
    if change == "reserved_relation":
        bad["steps"][0].update(do="place_in", ref="C", grip=["D"])
    else:
        bad["steps"][0]["sense"] = "open"
    revised = copy.deepcopy(approved)
    revised["steps"][0]["grip"] = ["D"]
    lock = PlanLock(approved_plan=approved)
    transport = ScriptedV2Transport(answers(plans=[approved, bad, revised],
                                           repairs=[repair("replan"), repair("hold_still")]))
    client = ReviewedV2Client(task_prompt=TASK2, approved_plan=approved, lock=lock, transport=transport)
    layer = StandIn(script={"hold": [{"status": "failed", "reason": "no_feasible_candidate"}]})
    policy = ReviewedV2Policy(client, lock=lock, relations=layer)
    world = make_world(fakes.task2_like_scene(), TASK2)
    observation = world.observation()
    for _ in range(50):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    assert transport.kinds.count("replan") == 2
    assert [row[1] for row in layer.log if row[0] == "controller"] == ["hold", "hold"]
    assert lock.snapshot()["current_plan"] == revised
    assert lock.snapshot()["plan_revision_count"] == 1
    requests = [request_text(call) for call in transport.calls if request_text(call)["request"] == "replan"]
    expected = "not implemented" if change == "reserved_relation" else "sense must be any"
    assert expected in requests[1]["previous_answer_rejected"]
    assert lock.conflict is None


@pytest.mark.parametrize("change", ["reserved_relation", "ignored_sense"])
def test_automatic_insert_step_rejects_unsupported_execution_fields(change):
    runner, lock, failed = locked_runner(allow_auto_replan=True)
    inserted = copy.deepcopy(HOLD_D)
    if change == "reserved_relation":
        inserted.update(do="place_in", ref="B")
    else:
        inserted["sense"] = "open"
    reply = SimpleNamespace(schema_error=False, answer=parse_repair(repair("insert_step", step=inserted)))
    before = runner.plan.to_json()
    issues = runner.repair_issues(reply, 0, review=False, failed=failed)
    expected = "relation_not_implemented" if change == "reserved_relation" else "ignored_selection"
    assert expected in [issue.code for issue in issues]
    assert runner.plan.to_json() == before
    assert lock.conflict is None


@pytest.mark.parametrize("change", ["delete_grip", "rename_object", "swap_texts", "change_move"])
def test_held_object_identity_survives_model_replan_and_targeted_correction(change):
    from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client
    from test_agent_speedup_v2 import HeldStub
    from test_agent_speedup_v2_client import request_text

    approved = copy.deepcopy(TASK2_PLAN)
    bad = copy.deepcopy(approved)
    bad["steps"][1]["grip"] = ["D"]
    if change == "delete_grip":
        bad["parts"] = [part for part in bad["parts"] if part["id"] != "C"]
    elif change == "rename_object":
        next(part for part in bad["parts"] if part["id"] == "D")["id"] = "F"
        bad["steps"][1].update(obj="F", grip=["C"], keep=[{"c": "upright", "p": "F"},
                                                          {"c": "clear_of", "p": "E"}])
    elif change == "swap_texts":
        c = next(part for part in bad["parts"] if part["id"] == "C")
        d = next(part for part in bad["parts"] if part["id"] == "D")
        c["text"], d["text"] = d["text"], c["text"]
    else:
        next(part for part in bad["parts"] if part["id"] == "C")["move"] = "unknown"
    revised = copy.deepcopy(approved)
    revised["steps"][1]["grip"] = ["D"]

    class HeldFailureLayer(StandIn):
        def _controller(self, do):
            native = super()._controller(do)

            def controller(ctx, step):
                result = yield from native(ctx, step)
                if do == "place_on" and result.status == "failed":
                    ctx.holding = HeldStub("D", "C")
                return result

            return controller

    lock = PlanLock(approved_plan=approved)
    transport = ScriptedV2Transport(answers(plans=[approved, bad, revised],
                                           repairs=[repair("replan"), repair("hold_still")]))
    client = ReviewedV2Client(task_prompt=TASK2, approved_plan=approved, lock=lock, transport=transport)
    layer = HeldFailureLayer(script={"place_on": [{"status": "failed", "reason": "lateral_unreachable"}]})
    policy = ReviewedV2Policy(client, lock=lock, relations=layer)
    world = make_world(fakes.task2_like_scene(), TASK2)
    observation = world.observation()
    for _ in range(70):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    requests = [request_text(call) for call in transport.calls if request_text(call)["request"] == "replan"]
    assert len(requests) == 2
    assert requests[0]["holding"] == "D"
    assert {part["id"] for part in requests[0]["held_part_identities"]} == {"C", "D"}
    assert "held_identity_changed" in requests[1]["previous_answer_rejected"]
    assert transport.kinds.count("replan") == 2
    assert lock.conflict is None
    assert lock.snapshot()["current_plan"] == revised
    assert lock.snapshot()["plan_revision_count"] == 1
    assert len(layer.controller_calls("turn")) == 1
    assert len(layer.controller_calls("place_on")) == 2
