"""V2 execution requires evaluator success, preserves privacy, and never starts Direct baselines."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, prediction


SPEC = importlib.util.spec_from_file_location(
    "visual_v2_runner", Path(__file__).parents[1] / "scripts/evaluate_visual_field_subscription.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def args(tmp_path, **changes):
    values = dict(output=tmp_path / "trial", task="bowl_plate", controller="hybrid",
        method_version="v2", demo=None, initial_state=None, bddl=None, seed=7,
        episode_index=0, sim_python="python", libero_root="/fixture")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("changes", [{"controller": "direct"}, {"demo": "missing-demo.json"}])
def test_v2_scope_rejected_before_creating_trial_or_auth(tmp_path, changes):
    config = args(tmp_path, **changes)
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(config, token_provider=lambda: pytest.fail("No auth allowed"))
    assert not config.output.exists()


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupted", "wrong_initial"])
def test_v2_real_route_uses_evaluator_not_model_verify(tmp_path, outcome):
    obs = observation()
    instances = []

    class Sim:
        def __init__(self, config):
            self.metadata = {"fixture": True}
            self.status = {}
            self.actions = []
            self.closed = False
            assert config.include_arm_joints
            assert config.max_steps is None
            assert config.collision_check_mode == "task_manipulation_v1"
            instances.append(self)

        def start(self):
            pass

        def reset(self, task, episode):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False,
                              "private": "EVALUATOR_ONLY_CONTACT"}}
            return obs

        def step(self, action):
            if outcome == "interrupted" and len(self.actions) == 12:
                raise KeyboardInterrupt
            self.actions.append(tuple(action))
            if len(self.actions) == 13:
                raw = outcome in ("success", "simultaneous")
                collision = outcome in ("collision", "simultaneous")
                self.status.update(success=raw and not collision, libero_success=raw,
                    terminated=True, collision={"mode": "task_manipulation_v1", "detected": collision,
                                               "private": "EVALUATOR_ONLY_CONTACT"})
            return obs, self.status["success"]

        def close(self):
            self.closed = True

    class Recorder:
        def __init__(self, output):
            pass

        def frame(self, *args):
            pass

        def close(self):
            pass

    config = args(tmp_path)
    if outcome == "wrong_initial":
        config.expected_initial = tmp_path / "wrong.json"
        config.expected_initial.write_text("{}")
    raw = prediction(phase="verify", actions=[[0.] * 7 for _ in range(5)])
    raw["reason"] = "The task appears complete; hold and verify."
    transport = FakeTransport(*(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"})
                                for _ in range(16)))
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert instances[0].closed
    assert "EVALUATOR_ONLY_CONTACT" not in json.dumps(transport.calls)
    assert result["budget"]["api_key_calls"] == 0
    assert result["method_version"] == "v2"
    assert result["control_without_video_s"] == result["control_elapsed_s"] - result["timing_components"]["video_render_s"]
    if outcome == "wrong_initial":
        assert result["status"] == "infrastructure_error"
        assert result["steps"] == 0 and not transport.calls
    elif outcome == "interrupted":
        assert result["status"] == "user_stop" and not result["scored"]
        assert result["steps"] == 12
    else:
        assert result["steps"] == 13
        assert result["status"] == ("success" if outcome == "success" else "collision_failure")
        assert result["completion_time_eligible"] == (outcome == "success")
    assert not any(any(action) for action in instances[0].actions)
    for request in transport.calls:
        assert request["model"] == "gpt-6-astra"
        assert request["reasoning"] == {"effort": "medium"}
