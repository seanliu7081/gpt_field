"""A small set of contract checks; no models, simulations or archived runs."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_relations as legacy
from gpt_field.agent_speedup_v2_types import HeightField, Meas, PartRecord, PlanStep, RelationOutcome, Seed
from gpt_field.agent_speedup_v2_user_check_control import build_relation_layer
from gpt_field.agent_speedup_v2_user_check_skills import destination_readiness, ensure_ready, observe_region


def _fixture(*, holding=None):
    destination = PartRecord(
        "B", "support region", "fixed", frozenset({"ref"}),
        xyz=np.array([0.0, 0.0, 0.1]), sigma_h_m=0.001, registration_step=1,
        top_z=Meas(0.1, 0.001, "dense"), support_z=Meas(0.0, 0.001, "plane"),
        region_cells=np.empty((0, 2), dtype=int), registered_empty_hand=True, status="registered")
    obj = PartRecord("A", "object", "free", frozenset({"carried", "grip"}))
    recorded = []
    ctx = SimpleNamespace(step=5, holding=holding, record=recorded.append,
                          scene=SimpleNamespace(parts={"A": obj, "B": destination},
                                                hf=HeightField.empty([0, 0]), planes=[]))
    step = PlanStep("place_on", "A", ("A",), "B", "any", (), ())
    return ctx, step, destination, recorded


def _complete(generator):
    with pytest.raises(StopIteration) as done:
        next(generator)
    return done.value.value


def test_preflight_and_controller_require_same_region_geometry_without_mutating_v2():
    ctx, _, record, _ = _fixture()
    old_check = legacy._destination_check
    assert old_check(ctx, record)["ok"]  # A top can satisfy the old height check alone.
    layer = build_relation_layer()
    assert not destination_readiness(ctx, record)["ready"]
    assert not layer._destination_check(ctx, record)["ok"]
    assert not layer._destination_ok(ctx)(record)
    assert layer.place_on.__globals__["_destination_check"] is layer._destination_check
    # The planner can rasterize a measured polygon even when explicit cells are empty.
    record.region_xy = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]])
    assert destination_readiness(ctx, record)["ready"]
    assert layer._destination_check(ctx, record)["ok"]
    assert layer._destination_ok(ctx)(record)
    assert legacy._destination_check is old_check


@pytest.mark.parametrize("repaired", [False, True])
def test_preflight_observes_once_and_checks_actual_geometry_after_reported_success(repaired):
    ctx, step, record, recorded = _fixture()
    calls = []

    def survey(ctx, pid, *, purpose, need):
        calls.append((pid, purpose))
        assert not need(record)
        if repaired:
            record.region_cells = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        if False:
            yield
        return RelationOutcome("registered", "succeeded", "registered", {}, {}, [], "registered", 0, 0)

    out = _complete(ensure_ready(ctx, step, relation_layer=SimpleNamespace(survey_hover=survey)))
    assert calls == [("B", "skill_precondition")]
    if repaired:
        assert out is None and not recorded
    else:
        assert out is recorded[0]
        feedback = out.evidence["execution_feedback"]
        assert feedback["status"] == "blocked" and not feedback["executed"]
        assert "destination_cells_nonempty" in {c["predicate"] for c in feedback["unmet_conditions"]}


def test_observation_while_holding_is_blocked_and_preserves_existing_geometry():
    ctx, _, record, recorded = _fixture(holding=SimpleNamespace(pid="A", grip_pid="A"))
    original_cells = record.region_cells

    def survey(*args, **kwargs):
        pytest.fail("blocked observation must not invoke the survey")

    out = _complete(observe_region(ctx, "B", relation_layer=SimpleNamespace(survey_hover=survey)))
    assert out is recorded[0]
    assert out.evidence["execution_feedback"]["status"] == "blocked"
    assert out.evidence["execution_feedback"]["unmet_conditions"][0]["predicate"] == "empty_hand_for_observation"
    assert ctx.scene.parts["B"] is record and record.region_cells is original_cells
    assert record.status == "registered" and ctx.holding.pid == "A"


def test_repoint_localizes_again_and_survey_uses_new_seed_without_old_hover_cache():
    ctx, _, old, _ = _fixture()
    pose = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04, -0.04)
    stale = Seed("wrist", (10.0, 10.0), 1, "plan_cell", 32.0, pose)
    fresh = Seed("wrist", (250.0, 250.0), 5, "coarse", 32.0, pose)
    old.seeds = [stale]
    ctx.view = SimpleNamespace(state=pose)
    ctx.memory = {"seed_lift": {"B": "stale"}, "start_records": {"B": old}}
    ctx.scene._start_record = {"B": old}
    calls = []

    def point(ctx, pid, **kwargs):
        calls.append("point")
        assert ctx.scene.parts[pid] is old  # Prior evidence survives until localization starts.
        if False:
            yield
        return fresh

    def drop(pid):
        calls.append("drop")
        ctx.scene.parts[pid] = PartRecord(pid, old.text, old.move_prior, old.roles,
                                         seeds=[stale], status="dropped")

    def lift(ctx, pid, view):
        calls.append("lift")
        assert ctx.scene.parts[pid].seeds == [fresh]
        assert pid not in ctx.memory["seed_lift"] and pid not in ctx.scene._start_record
        ctx.memory["seed_lift"][pid] = "fresh"

    def register(pid, view, *, seed, empty_hand, allow_partial):
        calls.append("register")
        assert seed is fresh and empty_hand
        ctx.scene.parts[pid] = replace(old, seeds=[fresh], xyz=np.array([0.03, 0.04, 0.1]),
                                       region_cells=np.array([[1, 1]]))
        return ctx.scene.parts[pid]

    def survey(ctx, pid, *, purpose, need):
        calls.append("survey")
        assert ctx.memory["seed_lift"][pid] == "fresh" and need(ctx.scene.parts[pid])
        assert ctx.scene.parts[pid].seeds == [fresh]
        if False:
            yield
        return RelationOutcome("registered", "succeeded", "registered", {}, {}, [], "registered", 0, 0)

    ctx.scene.drop, ctx.scene.register = drop, register
    layer = SimpleNamespace(_coarse_seed=point, _seed_lift=lift, survey_hover=survey)
    out = _complete(observe_region(ctx, "B", relation_layer=layer))
    assert calls == ["point", "drop", "lift", "register", "survey"]
    assert out.evidence["execution_feedback"]["status"] == "succeeded"
