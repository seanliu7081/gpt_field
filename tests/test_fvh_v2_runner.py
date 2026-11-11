"""New entrypoint lifecycle using public fake observations only."""
import importlib.util
from dataclasses import replace
import json
from pathlib import Path

import pytest
import numpy as np

from gpt_field.agent_speedup_v2_sim import initial_observation_sha256
from gpt_field.foundation_vision_harness_V2.runner import execute_episode, OPERATOR_STOP_FILE
from gpt_field.policy import Decision
from test_fvh_v1_client import view
from test_fvh_v1_runner import TASK, FakeSim, FakeClient, FakePolicy


def launch(tmp_path, policy):
    obs = view().observation
    sim, client = FakeSim(obs), FakeClient()
    result = execute_episode(tmp_path / "trial", task=TASK, episode_index=0,
        sim_factory=lambda: sim, client_factory=lambda **kwargs: client,
        policy_factory=lambda supplied: policy,
        expected_initial_observation_sha256=initial_observation_sha256(obs.to_dict()))
    return result, sim, client


def test_v2_autonomous_finish_keeps_raw_success_outside_policy(tmp_path):
    policy = FakePolicy()
    result, sim, client = launch(tmp_path, policy)
    assert result["version"] == "foundation_vision_harness_V2"
    assert result["status"] == "finish" and result["steps"] == 3
    assert result["first_raw_success"]["step"] == 1
    assert policy.calls == 4 and policy.transitions == 3
    assert len(sim.actions) == 13 and all(action[-1] == 1 for action in sim.actions)
    assert all(action[:3] == (0., 0., 0.) for action in sim.actions[-10:])
    assert result["strict_success"] and result["completion_time_eligible"]
    assert sim.closed and policy.closed and client.closed


def test_v2_operator_stop_during_decision_withholds_action(tmp_path):
    policy = FakePolicy(stop_path=tmp_path / "trial" / OPERATOR_STOP_FILE)
    result, sim, client = launch(tmp_path, policy)
    assert result["status"] == "user_stop" and result["withheld_decisions_after_stop"] == 1
    assert sim.actions == [] and result["same_gripper_hold_steps"] == 0
    assert sim.closed and policy.closed and client.closed


def test_full_task_entrypoint_requires_approval_and_offers_no_caps(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_foundation_vision_harness_V2.py"
    spec = importlib.util.spec_from_file_location("fvh_v2_script_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["run", "--task", "fixture", "--sam-checkpoint", "unused",
                                            "--da3-model", "unused", "--output", str(tmp_path / "never")])
    with pytest.raises(ValueError, match="explicit user approval"):
        module.run(args, None, None, None, None, None, None, {})
    assert not (tmp_path / "never").exists()
    assert not hasattr(args, "max_steps") and not hasattr(args, "retry")


class Clock:
    now = 0.
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def test_progress_distinguishes_measured_motion_from_local_goal_advancement(tmp_path):
    clock = Clock()
    observed = view().observation
    goal = np.asarray(observed.state[:3])+[.03, 0., 0.]
    class MovingSim(FakeSim):
        def step(self, action):
            clock.advance(.2)
            state = np.asarray(self.observation.state).copy()
            state[:3] += [.01, .02, 0.]
            self.observation = replace(self.observation, state=tuple(state))
            return super().step(action)
    class TimedPolicy(FakePolicy):
        def act(self, obs):
            clock.advance(.5)
            answer = super().act(obs)
            return Decision(answer.action, {**answer.diagnostics,
                "primitive": "measured_v2_fragment", "goal_xyz_m": goal.tolist(),
                "motion_evidence": {"variant": "local_tangent", "scale": .25},
                "control_timing": {"filter_calls": 2, "filter_wall_s": .15}})
        def observe_transition(self, *args):
            clock.advance(.1)
            super().observe_transition(*args)
    class Recorder:
        def __init__(self, *args, **kwargs): pass
        def submit(self, *args, **kwargs): clock.advance(.05)
        def close(self, **kwargs): return {}
    policy, sim = TimedPolicy(finish_after=1), MovingSim(observed)
    result = execute_episode(tmp_path/'trial', task=TASK, episode_index=0,
        sim_factory=lambda: sim, client_factory=lambda **kwargs: FakeClient(),
        policy_factory=lambda client: policy, recorder_factory=Recorder, clock=clock,
        expected_initial_observation_sha256=initial_observation_sha256(observed.to_dict()))
    rows = [json.loads(line) for line in (tmp_path/'trial/trace.jsonl').read_text().splitlines()]
    action = next(row for row in rows if row['event'] == 'action')
    assert action['progress']['translation_m'] == pytest.approx(np.sqrt(.01**2+.02**2))
    assert action['progress']['goal_distance_reduction_m'] == pytest.approx(.03-np.sqrt(.02**2+.02**2))
    assert action['progress']['command_variant'] == 'local_tangent'
    assert action['control_timing']['filter_calls'] == 2
    assert action['feedback_wall_s'] == pytest.approx(.1)
    assert action['recording_submit_wall_s'] == pytest.approx(.1)
    times = result['performance']['timings']
    assert times['decision']['total_s'] == pytest.approx(1.)
    assert times['simulator_step']['total_s'] == pytest.approx(.2)  # Evaluator holds excluded.
    assert times['recording_submit']['total_s'] == pytest.approx(.15)
    assert result['performance']['progress']['by_primitive']['measured_v2_fragment']['steps'] == 1


def test_zero_action_decision_error_preserves_elapsed_time_and_policy_stats(tmp_path):
    clock = Clock()
    observed = view().observation
    class FailedRequest(FakePolicy):
        def act(self, obs):
            clock.advance(7.)
            raise ValueError('fixture protocol rejection')
        def agent_stats(self): return {'timing': {'model_s': 7.}}
    policy, sim = FailedRequest(), FakeSim(observed)
    result = execute_episode(tmp_path/'trial', task=TASK, episode_index=0,
        sim_factory=lambda: sim, client_factory=lambda **kwargs: FakeClient(),
        policy_factory=lambda client: policy, clock=clock,
        expected_initial_observation_sha256=initial_observation_sha256(observed.to_dict()))
    assert result['status'] == 'error' and result['steps'] == 0 and not sim.actions
    assert result['online_wall_s'] == 7.
    assert result['performance']['timings']['decision']['total_s'] == 7.
    assert result['performance']['timings']['failed_decision']['count'] == 1
    assert result['agent_stats']['timing']['model_s'] == 7.
