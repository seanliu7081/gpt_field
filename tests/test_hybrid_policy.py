"""Controller handoffs preserve execution semantics and measured history."""

import copy

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.hybrid_subscription import HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID
from gpt_field.policy import Decision, FieldPolicy
from gpt_field.schema import GridSpec
from test_field_history import observation, prediction


def field(gripper="open", axis=2, grid=HYBRID_GRID):
    values = prediction(gripper=gripper, axis=axis)
    values["values"] = grid.coordinates([0., 0., 0.])[:, axis].tolist()
    return dict(controller="coarse_field", target_region="black_bowl", near_target=False,
                routing_reason="Object is ahead of the gripper.",
                field=values, direct=None)


def direct(gripper=1, chunk_size=HYBRID_CHUNK_SIZE):
    return dict(controller="contact_direct", target_region="black_bowl", near_target=True,
                routing_reason="Gripper is next to the bowl rim.", field=None,
                direct={"actions": [[.1 * (i % 5), -.05, .1, 0, .02, 0, gripper if i == 0 else 0]
                                    for i in range(chunk_size)]})


class Client:
    last_request_key = "offline-key"
    last_log_path = "offline.json"

    def __init__(self, *predictions, planner_horizon=3, field_execution_steps=1):
        self.planner_horizon = planner_horizon
        self.field_execution_steps = field_execution_steps
        self.predictions = copy.deepcopy(list(predictions))
        self.calls, self.feedback, self.resets = [], [], 0

    def reset_history(self):
        self.resets += 1
        self.feedback.clear()

    def history_stats(self):
        return {"feedback_records": len(self.feedback)}

    def record_transition(self, feedback):
        self.feedback.append(copy.deepcopy(feedback))

    def predict(self, obs, grid, mode):
        self.calls.append((copy.deepcopy(obs), grid, mode))
        return self.predictions.pop(0)


def execute(policy, obs, step):
    decision = policy.act(obs)
    state = list(obs.state)
    for i in range(3):
        state[i] += decision.action[i] * .025
    state[6] -= .001 * decision.action[6]
    state[7] += .001 * decision.action[6]
    after = observation(step + 1, state)
    feedback = policy.observe_transition(obs, decision, after)
    return after, decision, feedback


def test_coarse_actions_and_plans_equal_original_policy_with_explicit_three_step_horizon():
    data = field()
    client = Client(data)
    policy = HybridFieldDirectPolicy(client)
    original = FieldPolicy(Client(data["field"]), grid=HYBRID_GRID,
                           update_steps=HYBRID_FIELD_UPDATE_STEPS)
    obs = observation()
    initial_center = list(obs.state[:3])
    for i in range(HYBRID_FIELD_UPDATE_STEPS):
        measured_position = obs.state[:3]
        reference = original.act(obs)
        obs, actual, _ = execute(policy, obs, i)
        assert actual.action == reference.action
        assert actual.diagnostics["plan"] == reference.diagnostics["plan"]
        assert actual.diagnostics["field_center"] == reference.diagnostics["field_center"] == initial_center
        assert actual.diagnostics["plan"]["candidates"][0]["positions"][0] == list(measured_position)
        assert actual.diagnostics["update_steps"] == HYBRID_FIELD_UPDATE_STEPS
    assert policy.grid == GridSpec(size=33, spacing_m=.005)
    assert policy.planner.config.horizon == 3
    assert policy.planner.config.step_m == .01
    assert len(client.calls) == 1


@pytest.mark.parametrize("intent", [-1, 1])
def test_field_direct_field_handoffs_preserve_history_grip_and_fresh_center(intent):
    data = direct(intent)
    client = Client(field(), data, field("hold", axis=0))
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    decisions, centers = [], []
    field_steps, direct_steps = HYBRID_FIELD_UPDATE_STEPS, HYBRID_CHUNK_SIZE
    return_step = field_steps + direct_steps
    total_steps = 2 * field_steps + direct_steps
    for step in range(total_steps):
        centers.append(list(obs.state[:3]))
        obs, decision, feedback = execute(policy, obs, step)
        decisions.append(decision)
        if step < field_steps:
            chunk_index, chunk_step, chunk_start, length = 0, step, 0, field_steps
        elif step < return_step:
            chunk_index, chunk_step, chunk_start, length = 1, step - field_steps, field_steps, direct_steps
        else:
            chunk_index, chunk_step, chunk_start, length = 2, step - return_step, return_step, field_steps
        assert feedback["step"] == step
        assert feedback["chunk_index"] == decision.diagnostics["chunk_index"] == chunk_index
        assert feedback["chunk_step"] == decision.diagnostics["chunk_step"] == chunk_step
        assert decision.diagnostics["routing_decision_step"] == chunk_start
        assert decision.diagnostics["update_steps"] == length
        assert decision.diagnostics["refreshed"] == (chunk_step == 0)
        assert (policy.field is None) == (field_steps <= step < return_step)
        if field_steps <= step < return_step:
            assert decision.action == tuple(data["direct"]["actions"][step - field_steps])
            assert "plan" not in decision.diagnostics
        if step >= return_step:
            assert decision.action[6] == intent
            assert decision.diagnostics["field_center"] == centers[return_step]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["mode_changed"]] == [30, 35]
    assert [c[0]["observation/state"][:3] for c in client.calls] == [centers[i] for i in (0, 30, 35)]
    assert decisions[35].diagnostics["field_center"] == centers[35]
    assert centers[0] != centers[35]
    assert decisions[35].diagnostics["field_refreshed"]
    assert [x["step"] for x in client.feedback] == list(range(65))
    assert client.resets == 1
    assert len(client.calls) == 3
    assert policy.step_index == 65
    assert policy.update_steps == policy.field_update_steps == 30
    assert policy.direct_action_steps == 5


def test_initial_direct_and_zero_do_not_open_remembered_close():
    client = Client(direct(), field("hold"))
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    for step in range(HYBRID_CHUNK_SIZE + 1):
        obs, decision, _ = execute(policy, obs, step)
        if step == 0:
            assert not decision.diagnostics["mode_changed"]
        if 1 <= step < HYBRID_CHUNK_SIZE:
            assert decision.action[6] == 0
        else:
            assert decision.action[6] == 1


def test_feedback_required_exactly_once_with_matching_decision_and_observations():
    policy = HybridFieldDirectPolicy(Client(direct()))
    before = observation()
    decision = policy.act(before)
    after = observation(1)
    with pytest.raises(RuntimeError):
        policy.act(before)
    with pytest.raises(ValueError):
        policy.observe_transition(before, Decision(decision.action, decision.diagnostics), after)
    with pytest.raises(ValueError):
        policy.observe_transition(observation(2), decision, after)
    policy.observe_transition(before, decision, after)
    with pytest.raises(RuntimeError):
        policy.observe_transition(before, decision, after)
    with pytest.raises(ValueError):
        policy.act(before)
    policy.act(after)


def test_invalid_envelope_issues_no_action_or_step():
    data = direct()
    data["field"] = field()["field"]
    policy = HybridFieldDirectPolicy(Client(data))
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert policy.step_index == 0
    assert policy._pending is None
    assert policy.controller is None


def test_reset_clears_episode_state_and_restores_initial_open_intent():
    client = Client(direct(), field("hold"))
    policy = HybridFieldDirectPolicy(client)
    execute(policy, observation(), 0)
    assert policy.adapter._gripper == 1
    policy.reset()
    assert client.resets == 2
    assert client.feedback == []
    assert policy.step_index == 0
    assert policy._chunk_index == -1
    assert policy._chunk_start == policy._chunk_length == 0
    assert policy.controller is None
    assert policy.field is None
    assert policy.latest_feedback is None
    assert policy._progress == []
    assert policy.chunk == []
    assert policy._pending is None
    assert policy._expected_observation is None
    assert policy.adapter._gripper == -1
    assert policy.act(observation()).action[6] == -1


def test_legacy_client_grid_and_chunk_are_explicitly_supported():
    client = Client(field(grid=GridSpec()), direct(chunk_size=5))
    client.grid, client.chunk_size = GridSpec(), 5
    del client.planner_horizon
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    for step in range(6):
        obs, decision, feedback = execute(policy, obs, step)
        assert decision.diagnostics["chunk_step"] == feedback["chunk_step"] == step % 5
    assert policy.grid == GridSpec()
    assert policy.update_steps == 5
    assert policy.planner.config.horizon == 3
    assert decision.diagnostics["controller"] == "contact_direct"
    assert len(client.calls) == 2


def test_fixed_field_boundary_does_not_refresh_early_or_extend_planner_horizon():
    client = Client(field(axis=0), field(axis=0))
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    center = list(obs.state[:3])
    lower = center[0] - .08
    valid_counts, statuses = [], []
    # Realize the requested 1 cm primitive exactly to reach the fixed boundary.
    for step in range(HYBRID_FIELD_UPDATE_STEPS):
        decision = policy.act(obs)
        plan = decision.diagnostics["plan"]
        valid_counts.append(plan["valid_count"])
        statuses.append(plan["status"])
        assert decision.diagnostics["field_center"] == center
        assert decision.diagnostics["refreshed"] == (step == 0)
        assert policy.planner.config.horizon == 3
        if plan["selected_index"] is not None:
            chosen = plan["candidates"][plan["selected_index"]]
            assert len(chosen["primitives"]) == 3
            assert all(point[0] >= lower - 1e-12 for point in chosen["positions"])
        state = list(obs.state)
        for axis in range(3):
            state[axis] += .05 * decision.action[axis]
        # Tracking disturbances can take the measured pose outside this stale field.
        # The final two actions must hold translation until the scheduled refresh.
        if step == HYBRID_FIELD_UPDATE_STEPS - 3:
            state[0] = lower - .01
        after = observation(step + 1, state)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert len(client.calls) == 1
    assert valid_counts[0] == 150
    assert any(0 < count < 150 for count in valid_counts)
    assert "plateau_move" in statuses
    assert statuses[-2:] == ["no_valid_candidates", "no_valid_candidates"]
    assert decision.action[:3] == (0., 0., 0.)
    refreshed = policy.act(obs)
    assert len(client.calls) == 2
    assert refreshed.diagnostics["refreshed"]
    assert refreshed.diagnostics["field_center"] == list(obs.state[:3])
    assert refreshed.diagnostics["plan"]["valid_count"] == 150


def test_repeated_direct_blocks_reconsider_every_five_actions_before_field_return():
    first, second, last = direct(1), direct(-1), direct(1)
    client = Client(first, second, field("hold"), last)
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    decisions = []
    for step in range(41):
        obs, decision, feedback = execute(policy, obs, step)
        decisions.append(decision)
        if step < 10:
            index, offset, start = step // 5, step % 5, (step // 5) * 5
        elif step < 40:
            index, offset, start = 2, step - 10, 10
            assert decision.action[6] == -1
            assert decision.diagnostics["field_center"] == list(client.calls[2][0]["observation/state"][:3])
        else:
            index, offset, start = 3, 0, 40
        assert feedback["chunk_index"] == index
        assert feedback["chunk_step"] == offset
        assert decision.diagnostics["routing_decision_step"] == start
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["refreshed"]] == [0, 5, 10, 40]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["mode_changed"]] == [10, 40]
    assert decisions[10].diagnostics["field_refreshed"]
    assert decisions[5].diagnostics["direct_chunk_refreshed"]
    assert len(client.calls) == 4


def test_current_default_executes_all_ten_primitives_before_replanning():
    client = Client(field())
    del client.planner_horizon  # Exercise the current hybrid default without client override.
    del client.field_execution_steps
    policy = HybridFieldDirectPolicy(client)
    assert policy.planner.config.horizon == 10
    assert policy.planner.config.step_m == .01
    assert policy.field_execution_steps == 10
    obs = observation()
    original_plan, expected_deltas = None, None
    for step in range(10):
        decision = policy.act(obs)
        plan = decision.diagnostics["plan"]
        if step == 0:
            original_plan = plan
            chosen = plan["candidates"][plan["selected_index"]]
            positions = np.asarray(chosen["positions"])
            assert len(chosen["primitives"]) == 10 and positions.shape == (11, 3)
            expected_deltas = np.diff(positions, axis=0)
            assert np.linalg.norm(expected_deltas, axis=1) == pytest.approx([.01] * 10)
        assert plan == original_plan
        assert decision.diagnostics["plan_refreshed"] is (step == 0)
        assert decision.diagnostics["plan_start_step"] == 0
        assert decision.diagnostics["plan_step"] == step
        assert decision.diagnostics["plan_execution_steps"] == 10
        assert ("plan_latency_s" in decision.diagnostics) == (step == 0)
        assert np.asarray(decision.action[:3]) * .05 == pytest.approx(expected_deltas[step])
        # Deliberate tracking lag must not alter the committed primitive sequence.
        state = list(obs.state)
        for axis in range(3):
            state[axis] += decision.action[axis] * .01
        after = observation(step + 1, state)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert len({tuple(np.round(delta, 6)) for delta in expected_deltas}) > 1
    next_decision = policy.act(obs)
    assert next_decision.diagnostics["plan_refreshed"]
    assert next_decision.diagnostics["plan_start_step"] == 10
    assert next_decision.diagnostics["plan_step"] == 0
    next_plan = next_decision.diagnostics["plan"]
    assert next_plan["candidates"][next_plan["selected_index"]]["positions"][0] == pytest.approx(obs.state[:3])
    assert len(client.calls) == 1
    assert policy.field_update_steps == 30 and policy.direct_action_steps == 5


def test_committed_plans_use_global_starts_across_field_direct_field_blocks():
    from test_field_execution import ScriptedPlanner

    client = Client(field(), direct(), field("hold"), planner_horizon=10, field_execution_steps=10)
    policy = HybridFieldDirectPolicy(client)
    planner = ScriptedPlanner()
    policy._field_policy.planner = planner
    obs = observation()
    decisions, measured = [], []
    for step in range(65):
        measured.append(obs.state[:3])
        obs, decision, feedback = execute(policy, obs, step)
        decisions.append(decision)
        assert feedback["step"] == step
        if decision.diagnostics["controller"] == "coarse_field":
            base = 0 if step < 30 else 35
            assert decision.diagnostics["plan_start_step"] == base + ((step - base) // 10) * 10
            assert decision.diagnostics["plan_step"] == (step - base) % 10
        else:
            assert "plan_refreshed" not in decision.diagnostics
            assert policy._field_policy._plan_deltas == []
    starts = [0, 10, 20, 35, 45, 55]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics.get("plan_refreshed")] == starts
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["refreshed"]] == [0, 30, 35]
    assert [call[1] for call in planner.calls] == [measured[step] for step in starts]
    assert len(client.feedback) == 65 and len(client.calls) == 3


def test_reset_discards_unexecuted_field_plan_and_pending_transition():
    from test_field_execution import ScriptedPlanner

    client = Client(field(), field(), planner_horizon=10, field_execution_steps=10)
    policy = HybridFieldDirectPolicy(client)
    planner = ScriptedPlanner()
    policy._field_policy.planner = planner
    policy.act(observation())
    assert len(policy._field_policy._plan_deltas) == 9
    assert policy._pending is not None
    policy.reset()
    assert policy._field_policy._plan_deltas == []
    assert policy._field_policy._plan_result is None
    assert policy._pending is None
    decision = policy.act(observation())
    assert decision.diagnostics["plan_refreshed"]
    assert decision.diagnostics["plan_step"] == decision.diagnostics["plan_start_step"] == 0
    assert len(planner.calls) == 2
