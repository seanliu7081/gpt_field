"""Offline integration: distance planning, measured feedback and recovery."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from gpt_field.distance_policy import DistanceHybridPolicy
from gpt_field.distance_subscription import ChatGPTPlanDistanceClient
from test_compact_subscription import hybrid
from test_field_history import FakeTransport, envelope, observation


def intent(*, goal=(.03, 0., 0.), phase="approach", hold=False, direct=False):
    raw = hybrid(direct)
    raw["control_context"] = {"phase": phase, "visual_reference": None}
    if phase in ("align", "release", "verify", "recover"):
        raw["target_region"] = "plate"
    if not direct:
        raw["field"].update(goal_offset_m=list(goal), goal_reference=None,
                             motion_mode="hold" if hold else "move")
    return raw


def controller(*responses):
    transport = FakeTransport(*(envelope(x, id=f"distance-{i}", reasoning={"effort": "medium"})
                                for i, x in enumerate(responses)))
    client = ChatGPTPlanDistanceClient(token_provider=lambda: "offline-oauth", transport=transport)
    return DistanceHybridPolicy(client), transport


def tick(policy, obs, fraction=1., actual_position=None):
    decision = policy.act(obs)
    state = list(obs.state)
    state[:3] = (np.asarray(state[:3]) + .05 * np.asarray(decision.action[:3]) * fraction
                 if actual_position is None else actual_position)
    after = replace(obs, state=tuple(state))
    policy.observe_transition(obs, decision, after)
    return after, decision


def test_distance_arrival_then_direct_five_unchanged_and_no_success_claim():
    raw_direct = intent(direct=True, phase="grasp")
    policy, transport = controller(intent(), raw_direct, intent(hold=True))
    obs, d = tick(policy, observation())
    assert d.diagnostics["plan"]["search"]["method"] == "dijkstra_distance_field"
    assert d.diagnostics["plan"]["candidate_count"] == 1
    assert policy.pending_field_event["reason"] == "goal_reached"
    for i in range(5):
        obs, d = tick(policy, obs)
        assert d.action == tuple(raw_direct["direct"]["actions"][i])
        assert d.diagnostics["refreshed"] is (i == 0)
        assert not d.diagnostics["task_success_certified"]
    assert len(transport.calls) == 2
    tick(policy, obs)
    assert len(transport.calls) == 3


def test_goal_region_and_tracker_tolerances_do_not_accumulate_into_deadlock():
    policy, _ = controller(intent(goal=(.06, 0., 0.)))
    initial = observation()
    obs, _ = tick(policy, initial, actual_position=np.asarray(initial.state[:3]) + [.051, 0., 0.])
    obs, d = tick(policy, obs)
    assert np.linalg.norm(d.action[:3]) > 0
    assert np.linalg.norm(np.asarray(obs.state[:3]) - (np.asarray(initial.state[:3]) + [.06, 0, 0])) <= .005
    assert policy.pending_field_event["reason"] == "goal_reached"


def test_outside_known_cube_requests_observation_without_fictitious_goal():
    policy, _ = controller(intent(goal=(.3, 0., 0.)))
    _, d = tick(policy, observation())
    assert d.action[:3] == (0., 0., 0.)
    assert d.diagnostics["plan"]["status"] == "needs_observation"
    assert policy.pending_field_event["reason"] == "field_boundary_reached"
    assert "distance_field_display" not in d.diagnostics


def test_verify_missing_anchors_requests_one_bounded_view_then_recovery():
    policy, transport = controller(intent(phase="verify", hold=True), intent(phase="verify", hold=True))
    obs = observation()
    initial_z = obs.state[2]
    for _ in range(5):
        obs, d = tick(policy, obs)
        assert d.action[:3] == (0., 0., 0.)
    assert policy._recovery_active and policy.pending_field_event is None
    obs, d = tick(policy, obs)
    assert d.diagnostics["local_recovery_active"]
    assert 0 < obs.state[2] - initial_z <= .02 + 1e-12
    assert d.action[6] == -1.
    assert policy.pending_field_event["reason"] == "placement_unverified"
    z = obs.state[2]
    for _ in range(5):
        obs, d = tick(policy, obs)
        assert obs.state[2] == z
    assert policy.pending_field_event["reason"] == "placement_unverified"
    assert not policy._recovery_active and len(transport.calls) == 2
    assert policy._completion is None


def test_direct_verification_never_overrides_or_truncates_actions():
    direct = intent(phase="verify", direct=True)
    policy, _ = controller(direct)
    obs = observation()
    for i in range(5):
        obs, d = tick(policy, obs)
        assert d.action == tuple(direct["direct"]["actions"][i])
        assert (policy.pending_field_event is None) is (i < 4)
    assert policy.pending_field_event["reason"] == "placement_unverified"


class VisualFixture:
    def __init__(self, **overrides):
        self.result = {"status": "tracked", "reliable": True, "calibration_valid": False,
                       "pixel_error": [20., 0.], "goal_correction_xy_m": None,
                       "task_success_certified": False, **overrides}

    def observe(self, observation):
        return copy.deepcopy(self.result)

    def validate_metric_goal(self, goal_xyz):
        return True

    def invalidate(self, reason):
        self.result.update(status=reason, reliable=False, calibration_valid=False,
                           goal_correction_xy_m=None)
        return copy.deepcopy(self.result)


def test_outside_calibration_proposal_refreshes_without_adopting_goal():
    policy, _ = controller(intent(phase="align", goal=(.04, 0., 0.)))
    obs, _ = tick(policy, observation(), fraction=.1)
    original = copy.deepcopy(policy._field_cache.effective_compact_field)
    visual = VisualFixture(status="local_guidance_ready", calibration_valid=True,
                           goal_correction_xy_m=[0., .01])
    visual.validate_metric_goal = lambda goal_xyz: False
    policy.local_visual = visual
    tick(policy, obs, fraction=.1)
    assert policy._field_cache.effective_compact_field == original
    assert policy.local_goal_updates == 0
    assert policy.pending_field_event["reason"] == "visual_unavailable"
    assert policy.local_feedback["status"] == "proposed_goal_outside_calibration"


def test_lost_previously_valid_tracking_triggers_early_refresh():
    policy, _ = controller(intent(goal=(.07, 0., 0.)))
    obs, _ = tick(policy, observation(), fraction=.1)
    policy.local_feedback = VisualFixture().result
    policy.local_visual = VisualFixture(status="moving_match_lost", reliable=False)
    tick(policy, obs, fraction=.1)
    assert policy.pending_field_event["reason"] == "visual_unavailable"


def test_calibrated_goal_updates_keep_world_origin_and_obstacle_map():
    raw = intent(phase="align", goal=(.04, 0., 0.))
    raw["field"]["blocked_boxes"] = [{"lower_offset_m": [-.06, -.06, -.06],
                                       "upper_offset_m": [-.05, -.05, -.05]}]
    policy, transport = controller(raw)
    obs, _ = tick(policy, observation(), fraction=.1)
    center = policy.field.center
    masks = policy.field.prediction.blocked_indices
    policy.local_visual = VisualFixture(status="local_guidance_ready", calibration_valid=True,
                                        goal_correction_xy_m=[0., .01])
    obs, _ = tick(policy, obs, fraction=.1)
    assert policy.local_goal_updates == 1 and len(transport.calls) == 1
    assert policy.field.center == center and policy.field.prediction.blocked_indices == masks
    goal = np.asarray(center) + policy._field_cache.effective_compact_field["goal_offset_m"]
    assert goal[:2] == pytest.approx(np.asarray(obs.state[:2]) + [0., .01])
    assert raw["field"]["goal_offset_m"] == [.04, 0., 0.]


def test_tracking_confidence_without_calibration_never_changes_goal():
    policy, _ = controller(intent(phase="align", goal=(.07, 0., 0.)))
    obs, _ = tick(policy, observation(), fraction=.1)
    original = copy.deepcopy(policy._field_cache.effective_compact_field)
    policy.local_visual = VisualFixture(goal_correction_xy_m=[.01, 0.])
    tick(policy, obs, fraction=.1)
    assert policy._field_cache.effective_compact_field == original


def test_sustained_progress_can_reuse_field_past_thirty_without_reconsuming_cache():
    policy, transport = controller(intent(goal=(.075, .075, 0.)))
    obs, _ = tick(policy, observation(), fraction=.02)
    policy.local_visual = VisualFixture()
    for _ in range(32):
        obs, d = tick(policy, obs, fraction=.02)
    assert len(transport.calls) == 1
    assert d.diagnostics["chunk_step"] == 32
    assert d.diagnostics["update_steps"] == 60
    assert d.diagnostics["field_reuse_extensions"] == 1
    assert not d.diagnostics["refreshed"]


def test_thin_obstacle_cannot_be_crossed_by_tracker_lookahead():
    raw = intent(goal=(.06, 0., 0.))
    raw["field"]["blocked_boxes"] = [{"lower_offset_m": [.011, -.007, -.007],
                                       "upper_offset_m": [.012, .007, .007]}]
    policy, _ = controller(raw)
    obs = observation()
    for _ in range(3):
        before = np.asarray(obs.state[:3])
        obs, d = tick(policy, obs)
        assert policy._segment_valid(before, obs.state[:3])
        if policy.pending_field_event:
            break


def test_placement_consistency_requires_consecutive_stable_observations():
    policy, _ = controller(intent(phase="verify", hold=True))
    obs, _ = tick(policy, observation())
    policy.local_visual = VisualFixture(pixel_error=[1., 0.], moving_point_px=[30., 40.],
                                        target_point_px=[31., 40.])
    for _ in range(4):
        obs, _ = tick(policy, obs)
    assert policy.pending_field_event is None
    assert policy._recovery_active  # one final good frame cannot prove stability


def test_new_close_intent_does_not_enable_held_object_calibration():
    raw = intent(phase="align")
    raw["field"]["gripper_command"] = "close"
    raw["control_context"]["visual_reference"] = {
        "kind": "held_object", "moving_point_uv": [.2, .5], "target_point_uv": [.8, .5],
        "alignment_assumption": "same_rigid_point_desired_projection"}
    policy, _ = controller(raw)
    captured = []
    configure = policy.local_visual.configure
    def record(obs, ref):
        captured.append(copy.deepcopy(ref))
        return configure(obs, ref)
    policy.local_visual.configure = record
    policy.act(observation())
    assert "local_alignment" not in captured[0]
