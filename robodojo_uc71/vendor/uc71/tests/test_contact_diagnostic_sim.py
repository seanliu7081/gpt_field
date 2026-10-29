"""Evaluator contact/motion qualification and observation isolation tests."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.contact_diagnostic_sim import (
    ContactDiagnosticSimProcess, EngagementEvidence, EngagementThresholds,
    _ContactDiagnosticWorker, _ContactInspector, observation_fingerprint,
)
from gpt_field.libero import LiberoConfig, SimulatorError, _LiberoWorker
from gpt_field.observation import OBSERVATION_KEYS, PolicyObservation, image_to_data_url


def observation(x=0.):
    image = image_to_data_url(np.zeros((16, 16, 3), dtype=np.uint8), 16, rotate=False)
    return PolicyObservation(image, image, (x, 0., .3, 0., 0., 0., .03, -.03), "test task")


def measurement(target="bowl", z=0., q=0., contact=True, **changes):
    return {"target_xyz_m": [0., q, z], "tool_xyz_m": [0., q, z+.05],
        "articulation_q_m": q if target == "drawer_handle" else None,
        "left_contact": contact, "right_contact": contact,
        "left_pad_contact": contact, "right_pad_contact": contact,
        "other_support_contact": False, "nonfinger_robot_contact": False, **changes}


CLOSE = [0., 0., 0., 0., 0., 0., 1.]
OPEN = [0., 0., 0., 0., 0., 0., -1.]


def tracker(target="bowl", **kwargs):
    result = EngagementEvidence(target, **kwargs)
    result.begin(measurement(target, contact=False))
    return result


def test_fingerprint_is_exact_canonical_observable_identity():
    obs = observation()
    assert observation_fingerprint(obs) == observation_fingerprint(dict(reversed(list(obs.to_dict().items()))))
    changed = observation(1e-10)
    assert observation_fingerprint(changed)["state8_sha256"] != observation_fingerprint(obs)["state8_sha256"]
    assert observation_fingerprint(changed)["external_image_sha256"] == observation_fingerprint(obs)["external_image_sha256"]
    hidden = obs.to_dict(); hidden["object_pose"] = [0.]*7
    with pytest.raises(ValueError):
        observation_fingerprint(hidden)


@pytest.mark.parametrize("kwargs", [
    {"minimum_bilateral_frames": True}, {"minimum_bilateral_frames": 2},
    {"minimum_motion_m": 0.}, {"minimum_motion_m": float("nan")},
    {"minimum_tool_motion_m": True}, {"maximum_relative_drift_m": .2},
])
def test_thresholds_are_explicit_and_finite(kwargs):
    with pytest.raises(ValueError):
        EngagementThresholds(**kwargs)


def test_bilateral_contact_alone_is_not_grasp_success():
    evidence = tracker()
    for step in range(8):
        result = evidence.observe(measurement(), CLOSE, step)
    assert result["bilateral_streak"] == 8
    assert result["evidence_status"] == "bilateral_contact_without_validated_motion"
    assert result["engagement_success"] is False
    assert result["task_success_certified"] is False


def test_bowl_requires_sustained_capture_then_lift_with_tool():
    evidence = tracker()
    for step, height in enumerate([0., .005, .011]):
        result = evidence.observe(measurement(z=height), CLOSE, step)
    assert result["engagement_success"] and result["engagement_success_ever"]
    assert result["first_engagement_step"] == 2
    assert result["deltas"]["criterion_motion_m"] == pytest.approx(.011)
    assert not result["task_success_certified"]
    assert not result["libero_success_replaced"]


def test_handle_requires_articulation_motion_not_only_ee_motion():
    evidence = tracker("drawer_handle")
    for step in range(3):
        result = evidence.observe(measurement("drawer_handle", tool_xyz_m=[0., -.006*step, .05]), CLOSE, step)
    assert not result["engagement_success"]
    for step, q in enumerate([-.004, -.008, -.012], 3):
        result = evidence.observe(measurement("drawer_handle", q=q), CLOSE, step)
    assert result["engagement_success"]
    assert result["deltas"]["opening_motion_from_checkpoint_m"] == pytest.approx(.012)


def test_articulation_absolute_displacement_and_direction_reported_separately():
    evidence = tracker("drawer_handle")
    for step, q in enumerate([0., .006, .012]):
        result = evidence.observe(measurement("drawer_handle", q=q), CLOSE, step)
    assert result["engagement_success"]
    assert result["deltas"]["opening_motion_from_checkpoint_m"] == pytest.approx(-.012)


def test_one_sided_push_cannot_pass_even_when_target_moves():
    evidence = tracker("drawer_handle")
    for step in range(6):
        result = evidence.observe(measurement("drawer_handle", q=-step*.01, right_contact=False), CLOSE, step)
    assert result["evidence_status"] == "one_sided_contact_or_push"
    assert not result["engagement_success_ever"]


def test_closed_command_without_contact_and_open_command_contact_do_not_pass():
    evidence = tracker()
    for step in range(3):
        result = evidence.observe(measurement(z=step*.01, contact=False), CLOSE, step)
    assert result["evidence_status"] == "close_command_without_target_contact"
    result = evidence.observe(measurement(z=.03), OPEN, 3)
    assert result["evidence_status"] == "bilateral_contact_without_close_intent"
    assert not result["engagement_success_ever"]


def test_previous_push_lift_cannot_count_as_capture_lift():
    evidence = tracker()
    evidence.observe(measurement(z=.02, right_contact=False), CLOSE, 0)
    for step in range(1, 5):
        result = evidence.observe(measurement(z=.02), CLOSE, step)
    assert result["deltas"]["target_from_checkpoint_xyz_m"][2] == .02
    assert result["deltas"]["criterion_motion_m"] == 0.
    assert not result["engagement_success"]


@pytest.mark.parametrize("extra", [
    {"other_support_contact": True}, {"nonfinger_robot_contact": True},
    {"tool_xyz_m": [0., 0., .05]}, {"tool_xyz_m": [.08, 0., .07]},
])
def test_bowl_supported_elsewhere_or_not_following_tool_is_not_qualified(extra):
    evidence = tracker()
    evidence.observe(measurement(), CLOSE, 0)
    evidence.observe(measurement(z=.01), CLOSE, 1)
    result = evidence.observe(measurement(z=.02, **extra), CLOSE, 2)
    assert not result["engagement_success"]


def test_contact_loss_resets_capture_and_does_not_latch_current_success():
    evidence = tracker()
    for step, z in enumerate([0., .006, .012]):
        result = evidence.observe(measurement(z=z), CLOSE, step)
    assert result["engagement_success"]
    result = evidence.observe(measurement(z=.012, contact=False), CLOSE, 3)
    assert not result["engagement_success"] and result["engagement_success_ever"]
    assert result["bilateral_streak"] == 0
    for step in range(4, 7):
        result = evidence.observe(measurement(z=.012), CLOSE, step)
    assert not result["engagement_success"]
    assert result["first_engagement_step"] == 2


def test_supported_seating_does_not_poison_later_unsupported_bilateral_comotion():
    evidence = tracker()
    evidence.observe(measurement(other_support_contact=True), CLOSE, 0)
    evidence.observe(measurement(target_xyz_m=[.03, 0., 0.],
                                 other_support_contact=True), CLOSE, 1)
    for step, z in enumerate([.001, .007, .013], 2):
        result = evidence.observe(measurement(target_xyz_m=[.03, 0., z],
                                             tool_xyz_m=[0., 0., z+.05]), CLOSE, step)
    assert result["engagement_success"]
    assert result["deltas"]["qualifying_interval_start_step"] == 2
    assert result["deltas"]["relative_translation_drift_m"] == pytest.approx(.03)


def test_interval_cannot_hide_a_large_middle_slip_by_returning_to_origin():
    evidence = tracker()
    evidence.observe(measurement(), CLOSE, 0)
    evidence.observe(measurement(z=.006, target_xyz_m=[.04, 0., .006]), CLOSE, 1)
    result = evidence.observe(measurement(z=.012), CLOSE, 2)
    assert not result["engagement_success"]


def test_invalid_feedback_is_atomic_and_missing_step_is_rejected():
    evidence = tracker()
    evidence.observe(measurement(), CLOSE, 10)
    before = evidence.snapshot()
    with pytest.raises(ValueError):
        evidence.observe(measurement(z=float("nan")), CLOSE, 11)
    with pytest.raises(ValueError):
        evidence.observe(measurement(), CLOSE, 12)
    assert evidence.snapshot() == before


def test_finite_short_diagnostic_limit_and_single_arm_per_reset():
    evidence = tracker(diagnostic_action_limit=2)
    evidence.observe(measurement(), CLOSE, 0)
    assert evidence.observe(measurement(z=.02), CLOSE, 1)["diagnostic_limit_reached"]
    with pytest.raises(RuntimeError):
        evidence.observe(measurement(), CLOSE, 2)
    with pytest.raises(RuntimeError):
        evidence.begin(measurement())


class ReplayFixture(ContactDiagnosticSimProcess):
    def __init__(self, observations):
        super().__init__(LiberoConfig(suite="libero_goal", stop_on_collision=False), target="drawer_handle")
        self.observations = observations
        self.reset_calls = self.step_calls = 0

    def reset(self, task_id, episode_index):
        self.reset_calls += 1
        self.step_calls = 0
        return self._observation(self.observations[0].to_dict())

    def step(self, action):
        self.step_calls += 1
        return self._observation(self.observations[self.step_calls].to_dict()), False


def test_replay_validates_prefix_before_any_reset_and_matches_exact_checkpoint():
    initial, final = observation(), observation(.01)
    simulator = ReplayFixture([initial, final])
    with pytest.raises(ValueError):
        simulator.replay_checkpoint(3, 0, [CLOSE, [float("nan")]*7], observation_fingerprint(final))
    assert simulator.reset_calls == simulator.step_calls == 0
    result = simulator.replay_checkpoint(3, 0, [CLOSE], observation_fingerprint(final),
                                         initial_fingerprint=observation_fingerprint(initial))
    assert result == final and simulator.checkpoint_check["matched"]
    assert simulator.checkpoint_check["initial_matched"]
    assert simulator.step_calls == 1


def test_wrong_initial_fingerprint_prevents_any_prefix_action():
    simulator = ReplayFixture([observation(), observation(.01)])
    with pytest.raises(SimulatorError, match="Initial reset"):
        simulator.replay_checkpoint(3, 0, [CLOSE], observation_fingerprint(observation(.01)),
                                   initial_fingerprint=observation_fingerprint(observation(.02)))
    assert simulator.step_calls == 0


def test_wrong_final_fingerprint_cannot_begin_short_diagnostic():
    simulator = ReplayFixture([observation(), observation(.01)])
    with pytest.raises(SimulatorError, match="Replayed checkpoint"):
        simulator.replay_checkpoint(3, 0, [CLOSE], observation_fingerprint(observation(.02)))
    assert simulator.checkpoint_check["matched"] is False
    with pytest.raises(SimulatorError, match="exact observable checkpoint"):
        simulator.begin_diagnostic(observation_fingerprint(observation(.02)))


def test_client_configuration_and_observation_boundary():
    config = LiberoConfig(suite="libero_goal", stop_on_collision=False)
    simulator = ContactDiagnosticSimProcess(config, target="drawer_handle", diagnostic_action_limit=100)
    assert simulator.config == config
    assert "gpt_field.contact_diagnostic_sim" in simulator._command
    assert "gpt_field.libero" not in simulator._command
    assert set(simulator._observation(observation().to_dict()).to_dict()) == OBSERVATION_KEYS
    hidden = observation().to_dict(); hidden["evaluator_metrics"] = {"target_xyz_m": [1., 2., 3.]}
    with pytest.raises(ValueError):
        simulator._observation(hidden)
    with pytest.raises(ValueError):
        ContactDiagnosticSimProcess(LiberoConfig(stop_on_collision=True), target="bowl")
    with pytest.raises(ValueError):
        ContactDiagnosticSimProcess(config, target="bowl", diagnostic_action_limit=None)


def test_worker_keeps_privileged_metrics_outside_observation_and_enforces_limit(monkeypatch):
    worker = _ContactDiagnosticWorker.__new__(_ContactDiagnosticWorker)
    worker.diagnostic_action_limit = 2
    worker.inspector = SimpleNamespace(sample=lambda: measurement())
    worker.evidence = tracker(diagnostic_action_limit=2)
    worker.status = {"steps": 10, "terminated": False, "success": False}
    worker.latest_observation = observation().to_dict()
    worker.checkpoint_physics_fingerprint = {"combined_sha256": "fixture"}
    def base_step(self, action):
        self.status["steps"] += 1
        return {"status": self.status.copy(), "observation": observation().to_dict()}
    monkeypatch.setattr(_LiberoWorker, "step", base_step)
    first = worker.step(CLOSE)
    assert set(first["observation"]) == OBSERVATION_KEYS
    assert first["evaluator_metrics"]["measurements"]["target_xyz_m"] == [0., 0., 0.]
    second = worker.step(CLOSE)
    assert second["status"]["terminated"] and second["status"]["diagnostic_limit_reached"]
    assert second["status"]["success"] is False
    with pytest.raises(RuntimeError, match="Short diagnostic"):
        worker.step(CLOSE)
    assert worker.status["steps"] == 12


def inspector_fixture(contacts):
    inspector = _ContactInspector.__new__(_ContactInspector)
    inspector.target = "bowl"
    inspector.groups = {"left": {0, 1}, "right": {2, 3}, "left_pad": {1}, "right_pad": {3}}
    inspector.target_geoms = {4}
    inspector.geometry = {0: {"role": "robot_finger"}, 1: {"role": "robot_finger"},
        2: {"role": "robot_finger"}, 3: {"role": "robot_finger"},
        4: {"role": "target_bowl"}, 5: {"role": "table"}, 6: {"role": "robot"}}
    inspector.core = SimpleNamespace(_check_grasp=lambda *args: True)
    inspector.gripper = object()
    inspector.body_name, inspector.body_id, inspector.joint_name, inspector.tool_site = "bowl", 0, None, "grip"
    inspector.sim = SimpleNamespace(model=SimpleNamespace(geom_id2name=lambda i:f"geom{i}"),
        data=SimpleNamespace(ncon=len(contacts), contact=contacts, body_xpos=[[0., 0., .1]],
            get_site_xpos=lambda name:[0., 0., .15]))
    return inspector


def contact(a, b, distance=0.):
    return SimpleNamespace(geom1=a, geom2=b, dist=distance)


def test_contact_sampling_requires_two_distinct_sides_and_ignores_margin_contacts():
    inspector = inspector_fixture([contact(0, 4), contact(1, 4), contact(3, 4, .0001)])
    data = inspector.sample()
    assert data["left_contact"] and data["left_pad_contact"]
    assert not data["right_contact"]
    assert data["native_pad_grasp_check"] is True  # Not promoted into our criterion.
    assert len(data["contacts"]) == 2
    inspector = inspector_fixture([contact(4, 1), contact(3, 4), contact(4, 5), contact(6, 4)])
    data = inspector.sample()
    assert data["left_contact"] and data["right_contact"]
    assert data["other_support_contact"] and data["nonfinger_robot_contact"]


def test_controller_fingerprint_serializes_hidden_goals_and_rejects_unsupported_state():
    from gpt_field.contact_diagnostic_sim import _state_value
    import numpy as np
    controller = SimpleNamespace(sim=object(), goal_pos=np.array([0., .1, .2]),
        interpolation=SimpleNamespace(step=3, goal=np.array([1., 2., 3.])))
    first = _state_value(controller)
    assert "sim" not in first["members"]
    controller.interpolation.step = 4
    assert _state_value(controller) != first
    controller.goal_pos[0] = .001
    assert _state_value(controller)["members"]["goal_pos"] != first["members"]["goal_pos"]
    controller.unsupported = object()
    with pytest.raises(ValueError, match="Unsupported controller"):
        _state_value(controller)
