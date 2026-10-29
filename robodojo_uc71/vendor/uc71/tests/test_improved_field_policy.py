"""Observable, offline integration of grounded goals and early field refreshes."""

from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.observation import image_to_data_url
from test_compact_subscription import client, hybrid
from test_field_history import FakeTransport, envelope, observation
from test_block_feedback_history import text_records
from test_hybrid_subscription_runner import runner, sim_config


def grounded(goal=(.025, 0., 0.), *, reference=None, hold=False):
    raw = hybrid()
    raw["field"].update(goal_offset_m=list(goal), goal_reference=reference,
                        motion_mode="hold" if hold else "move")
    return raw


def improved(*responses):
    transport = FakeTransport(*(envelope(raw, id=f"improved-{i}", reasoning={"effort": "medium"})
                                for i, raw in enumerate(responses)))
    instance = client(transport, field_improvements=True, execution_mode="tracking",
                      reasoning_effort="medium", feedback_history="block_summary")
    return HybridFieldDirectPolicy(instance), transport


def transition(policy, obs, *, fraction=1.):
    decision = policy.act(obs)
    state = list(obs.state)
    state[:3] = np.asarray(state[:3]) + np.asarray(decision.action[:3]) * .05 * fraction
    after = replace(obs, state=tuple(state))
    policy.observe_transition(obs, decision, after)
    return after, decision


def test_short_goal_refreshes_after_actual_arrival_then_direct_executes_all_five():
    policy, transport = improved(grounded(), hybrid(True), grounded(hold=True))
    obs, first = transition(policy, observation())
    path = first.diagnostics["plan"]["candidates"][0]
    assert path["path_length_m"] == pytest.approx(.025)
    assert len(path["primitives"]) == 3
    assert policy.pending_field_event["reason"] == "goal_reached"
    for index in range(5):
        obs, direct = transition(policy, obs)
        assert direct.diagnostics["controller"] == "contact_direct"
        assert direct.diagnostics["refreshed"] is (index == 0)
        assert direct.action == tuple(hybrid(True)["direct"]["actions"][index])
    obs, last = transition(policy, obs)
    assert last.diagnostics["step"] == 6 and last.diagnostics["refreshed"]
    assert len(transport.calls) == 3
    records = text_records(transport.calls[1]["input"])
    summary = next(r for r in records if r.get("type") == "execution_block_summary")
    assert summary["executed_actions"] == 1
    assert next(r for r in records if r.get("type") == "field_control_event")["step"] == 1


def test_planned_short_endpoint_does_not_mean_measured_arrival():
    policy, transport = improved(grounded())
    obs, first = transition(policy, observation(), fraction=.2)
    assert first.diagnostics["plan"]["search"]["endpoint_is_goal"]
    assert policy.pending_field_event is None
    obs, second = transition(policy, obs, fraction=.2)
    assert not second.diagnostics["refreshed"] and not second.diagnostics["plan_refreshed"]
    assert len(transport.calls) == 1


def test_boundary_requests_new_observation_and_recenters_only_on_fresh_response():
    policy, transport = improved(grounded((.3, 0., 0.)), grounded((.3, 0., 0.)))
    obs = observation()
    initial = obs.state[:3]
    for _ in range(3):
        obs, decision = transition(policy, obs)
    assert policy.pending_field_event["reason"] == "field_boundary_reached"
    assert decision.diagnostics["field_center"] == list(initial)
    obs, fresh = transition(policy, obs)
    assert fresh.diagnostics["refreshed"] and len(transport.calls) == 2
    assert fresh.diagnostics["field_center"][0] == pytest.approx(initial[0] + .08)


def test_no_progress_requests_fresh_model_but_never_marks_task_complete():
    policy, transport = improved(grounded(), hybrid(True))
    obs = observation()
    for _ in range(5):
        obs, _ = transition(policy, obs, fraction=0.)
    assert policy.pending_field_event["reason"] == "no_progress"
    obs, direct = transition(policy, obs)
    assert direct.diagnostics["refreshed"] and direct.diagnostics["controller"] == "contact_direct"
    assert len(transport.calls) == 2 and policy._completion is None


def test_explicit_hold_keeps_thirty_action_interval_without_spurious_events():
    policy, transport = improved(grounded(hold=True), hybrid(True))
    obs = observation()
    for _ in range(30):
        obs, decision = transition(policy, obs)
        assert decision.action[:3] == (0., 0., 0.)
        assert policy.pending_field_event is None
    obs, direct = transition(policy, obs)
    assert direct.diagnostics["step"] == 30 and len(transport.calls) == 2


def test_held_goal_waits_for_measured_orientation_without_per_tick_model_calls():
    reference = {"kind": "held_object", "object_center_offset_m": [.04, 0., 0.],
                 "desired_object_center_offset_m": [0., .04, 0.], "visual_anchor": None}
    raw = grounded(reference=reference)
    raw["field"]["orientation_target_xyzw"] = [0., 0., 2**-.5, 2**-.5]
    policy, transport = improved(raw, hybrid(True))
    obs = observation()
    for angle in (.1, .3, .8, np.pi/2):
        decision = policy.act(obs)
        assert decision.diagnostics["plan"]["status"] == "goal_reached"
        assert decision.action[:3] == (0., 0., 0.)
        assert decision.action[5] > 0
        state = list(obs.state)
        state[5] = angle
        after = replace(obs, state=tuple(state))
        policy.observe_transition(obs, decision, after)
        obs = after
        assert (policy.pending_field_event is not None) is (angle == np.pi/2)
    assert len(transport.calls) == 1
    assert policy.pending_field_event["reason"] == "goal_reached"


def test_measured_grid_overshoot_requests_refresh_without_waiting_for_replan():
    policy, _ = improved(grounded((.3, 0., 0.)))
    obs = observation()
    decision = policy.act(obs)
    after = replace(obs, state=(.181, *obs.state[1:]))
    policy.observe_transition(obs, decision, after)
    assert policy.pending_field_event["reason"] == "invalid_direction"


def test_held_object_offset_affects_both_grid_and_analytic_goal_without_changing_raw():
    reference = {"kind": "held_object", "object_center_offset_m": [.03, 0., 0.],
                 "desired_object_center_offset_m": [.05, 0., 0.], "visual_anchor": None}
    raw = grounded((.05, 0., 0.), reference=reference)
    policy, _ = improved(raw)
    decision = policy.act(observation())
    diag = decision.diagnostics
    assert diag["compact_prediction"] == raw
    assert diag["effective_compact_field"]["goal_offset_m"] == pytest.approx([.02, 0., 0.])
    assert diag["plan"]["search"]["goal_xyz"] == pytest.approx([.12, .2, 1.])
    assert diag["plan"]["candidates"][0]["path_length_m"] == pytest.approx(.02)


def test_real_image_checker_interrupts_wrong_direction_before_thirty_actions():
    rng = np.random.default_rng(8)
    patch = rng.integers(0, 256, (11, 11, 3), dtype=np.uint8)
    def image(x):
        pixels = np.full((96, 96, 3), 20, dtype=np.uint8)
        pixels[35:46, x-5:x+6] = patch
        return image_to_data_url(pixels, size=96, rotate=False)
    reference = {"kind": "end_effector", "object_center_offset_m": None,
                 "desired_object_center_offset_m": None, "visual_anchor": {
                     "moving_point_uv": [60/95, 40/95], "target_point_uv": [80/95, 40/95]}}
    policy, transport = improved(grounded((.1, 0., 0.), reference=reference), hybrid(True))
    obs = replace(observation(), image=image(60))
    for x in (55, 50):
        decision = policy.act(obs)
        state = list(obs.state)
        state[0] += .003
        after = replace(obs, image=image(x), state=tuple(state))
        policy.observe_transition(obs, decision, after)
        obs = after
    event = policy.pending_field_event
    assert event["reason"] == "visual_divergence" and event["step"] == 2
    assert event["visual_check"]["current_distance_px"] > event["visual_check"]["baseline_distance_px"]
    direct = policy.act(obs)
    assert direct.diagnostics["refreshed"] and len(transport.calls) == 2
    assert event in text_records(transport.calls[1]["input"])


def test_improved_preflight_locks_required_settings_without_credentials_or_actions():
    result = runner.preflight(runner.configuration(), sim_config(), field_improvements=True,
                              execution_mode="tracking", reasoning_effort="medium")
    assert result["policy"]["field_improvements"]
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["policy"]["direct_action_steps"] == 5
    assert result["max_wall_time_s"] is None and result["max_policy_steps"] is None
    assert not result["credentials_checked"] and result["api_calls"] == 0
    assert result["stopping_rule"][0] == "LIBERO success"


@pytest.mark.parametrize("change", [{"reasoning_effort": "high"}, {"execution_mode": "fixed"},
    {"controller_mode": "direct"}, {"stop_on_model_complete": True}, {"field_representation": "dense"}])
def test_improved_runner_rejects_incompatible_controls(change):
    settings = {"field_improvements": True, "execution_mode": "tracking", "reasoning_effort": "medium"}
    settings.update(change)
    with pytest.raises(ValueError):
        runner.preflight(runner.configuration(), sim_config(), **settings)


def test_improved_cli_sets_medium_tracking_and_summaries(capsys):
    assert runner.main(["--field-improvements", "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["policy"]["execution_mode"] == "tracking"
    assert result["policy"]["feedback_history"] == "block_summary"
    assert result["policy"]["reasoning_effort"] == "medium"
    assert result["policy"]["centered_placement_guidance"] and result["policy"]["grasp_settle_guidance"]
