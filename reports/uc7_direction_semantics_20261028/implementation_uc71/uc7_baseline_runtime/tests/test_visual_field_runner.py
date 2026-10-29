"""No-contact precedence, exact Direct actions and credential/input isolation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from test_field_history import FakeTransport, envelope

SPEC = importlib.util.spec_from_file_location("visual_runner", Path(__file__).parents[1] / "scripts/evaluate_visual_field_subscription.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def raw_prediction():
    return {"controller": "direct", "phase": "approach", "target_id": "object_1", "hypothesis_id": "approach_1",
        "reason": "Move toward the visible object", "visual_references": {"external": None, "wrist": None},
        "field_intent": None, "direct_actions": [[.1, 0, 0, 0, 0, 0, -1]] * 5}


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "initial_collision", "interrupt"])
def test_terminal_outcomes_and_privacy(tmp_path, outcome):
    pixels = np.random.default_rng(7).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    image = image_to_data_url(pixels, 48, False)
    obs = PolicyObservation(image, image, (.0, .0, 1., 0., 0., 0., .03, -.03),
        runner.resolve_distance_task("bowl_plate").prompt,
        joint_position=(0., -.4, 0., -2., 0., 1.6, .7), joint_velocity=(0.,)*7)
    instances = []

    class Sim:
        def __init__(self, config):
            self.actions = []
            self.metadata = {"fixture": True}
            self.status = {}
            instances.append(self)
            assert config.include_arm_joints and config.max_steps is None
        def start(self):
            pass
        def reset(self, task, episode):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": outcome == "initial_collision",
                              "private": "HIDDEN_CONTACT_ORACLE"}}
            return obs
        def step(self, action):
            if outcome == "interrupt":
                raise KeyboardInterrupt
            self.actions.append(action)
            raw = outcome in ("success", "simultaneous")
            collision = outcome in ("collision", "simultaneous")
            self.status.update(success=raw and not collision, libero_success=raw, terminated=True,
                collision={"mode": "task_manipulation_v1", "detected": collision,
                           "private": "HIDDEN_CONTACT_ORACLE"})
            return obs, self.status["success"]
        def close(self):
            self.status = {}

    class Recorder:
        def __init__(self, output):
            pass
        def frame(self, *args):
            pass
        def close(self):
            pass

    args = SimpleNamespace(output=tmp_path / "trial", task="bowl_plate", controller="direct",
        demo=None, initial_state=None, bddl=None, seed=7, episode_index=0,
        sim_python="python", libero_root="/fixture")
    transport = FakeTransport(envelope(raw_prediction(), model="gpt-6-astra", reasoning={"effort": "medium"}))
    result = runner.execute(args, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert "HIDDEN_CONTACT_ORACLE" not in json.dumps(transport.calls)
    assert result["budget"]["api_key_calls"] == 0
    if outcome == "initial_collision":
        assert result["status"] == "infrastructure_error" and not transport.calls
    elif outcome == "interrupt":
        assert result["status"] == "user_stop" and not result["scored"]
    else:
        assert instances[0].actions == [tuple(raw_prediction()["direct_actions"][0])]
        assert result["status"] == ("success" if outcome == "success" else "collision_failure")
        assert result["libero_success"] == (outcome != "collision")
        assert result["completion_time_eligible"] == (outcome == "success")
        assert result["steps"] == 1
    assert result["control_without_video_s"] == result["control_elapsed_s"] - result["timing_components"]["video_render_s"]


def test_stall_episode_is_not_recounted_at_each_direct_request():
    from test_visual_field_protocol import client, observation, prediction
    from gpt_field.visual_field_policy import VisualFieldPolicy
    raw = prediction(actions=[[.1, 0, 0, 0, 0, 0, 1.]] * 5)
    instance, transport = client(raw, raw, raw, raw, controller="direct")
    policy = VisualFieldPolicy(instance)
    obs = observation()
    for _ in range(20):
        decision = policy.act(obs)
        policy.observe_transition(obs, decision, obs)
    assert len(transport.calls) == 4
    assert len(policy.events) == policy.recovery_episodes == 1
