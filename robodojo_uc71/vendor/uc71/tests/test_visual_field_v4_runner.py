"""V4 evaluator routing/timing/privacy without a live model or simulator."""
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v4_protocol import prediction


SPEC = importlib.util.spec_from_file_location(
    "visual_v4_runner", Path(__file__).parents[1] / "scripts/evaluate_visual_field_subscription.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def args(tmp_path, **changes):
    values = dict(output=tmp_path / "trial", task="bowl_plate", controller="hybrid",
        method_version="v4", demo=None, initial_state=None, bddl=None, seed=7,
        episode_index=0, sim_python="python", libero_root="/fixture")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("changes", [{"controller": "direct"}, {"demo": "missing-demo.json"}])
def test_scope_rejected_before_auth_or_creating_output(tmp_path, changes):
    config = args(tmp_path, **changes)
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(config, token_provider=lambda: pytest.fail("No auth allowed"))
    assert not config.output.exists()


@pytest.mark.parametrize("outcome", ["success", "field_success", "collision", "simultaneous", "interrupted", "wrong_initial"])
def test_v4_route_uses_evaluator_and_preserves_zero_shot_protocol(tmp_path, outcome):
    obs = observation()
    instances = []

    class Sim:
        def __init__(self, config):
            self.metadata, self.status, self.actions, self.closed = {"fixture": True}, {}, [], False
            assert config.include_arm_joints and config.max_steps is None
            assert config.collision_check_mode == "task_manipulation_v1"
            instances.append(self)

        def start(self):
            pass

        def reset(self, task, episode):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False,
                              "private": "EVALUATOR_ONLY_CONTACT"}}
            self.obs = obs
            return self.obs

        def step(self, action):
            if outcome == "interrupted" and len(self.actions) == 12:
                raise KeyboardInterrupt
            self.actions.append(tuple(action))
            if outcome == "field_success":
                # Acknowledged measured movement, without claiming this toy transition
                # or its explicit fixture success is a robot-task performance result.
                self.obs = replace(self.obs, state=(self.obs.state[0]+.0005, *self.obs.state[1:]))
            if len(self.actions) == (2 if outcome == "field_success" else 13):
                raw = outcome in ("success", "field_success", "simultaneous")
                collision = outcome in ("collision", "simultaneous")
                self.status.update(success=raw and not collision, libero_success=raw, terminated=True,
                    collision={"mode": "task_manipulation_v1", "detected": collision,
                               "private": "EVALUATOR_ONLY_CONTACT"})
            return self.obs, self.status["success"]

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
    value = prediction(phase="verify", actions=[[0.] * 7 for _ in range(5)])
    value["reason"] = "The task appears complete; hold and verify."
    if outcome == "field_success":
        value = prediction(field=True)
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort": "medium"})
                                for _ in range(16)))
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert instances[0].closed
    assert "EVALUATOR_ONLY_CONTACT" not in json.dumps(transport.calls)
    assert result["budget"]["api_key_calls"] == 0 and result["method_version"] == "v4"
    assert result["shots"] == 0
    assert result["control_without_video_s"] == result["control_elapsed_s"] - result["timing_components"]["video_render_s"]
    if outcome == "field_success":
        assert result["field_actions"] == result["local_servo_actions"] == 2
        assert result["field_selection_status_counts"] == {"metric_field_tracking": 2}
        assert result["policy_diagnostic_counters"]["field_effective_actions"] == 2
        assert len(transport.calls) == 1
    else:
        assert result["field_actions"] == result["local_servo_actions"] == 0
    if outcome == "wrong_initial":
        assert result["status"] == "infrastructure_error" and result["steps"] == 0
        assert not transport.calls
    elif outcome == "interrupted":
        assert result["status"] == "user_stop" and not result["scored"] and result["steps"] == 12
    else:
        assert result["steps"] == (2 if outcome == "field_success" else 13)
        successful = outcome in ("success", "field_success")
        assert result["status"] == ("success" if successful else "collision_failure")
        assert result["completion_time_eligible"] == successful
    if outcome != "field_success":
        assert not any(any(action) for action in instances[0].actions)
    manifest = json.loads((config.output / "manifest.json").read_text())
    assert manifest["field_grid"] == {"shape": [33, 33, 33], "spacing_m": .005, "coverage_m": [.16, .16, .16]}
    assert "hypothesized" in manifest["field_units"] and "not a measured SDF" in manifest["field_units"]
    assert "optional target bearing" in manifest["camera_roles"]["wrist"]
    assert manifest["api_key_fallback"] is False
    for request in transport.calls:
        assert request["model"] == "gpt-6-astra" and request["reasoning"] == {"effort": "medium"}
        assert request["text"]["format"]["name"] == "visual_field_rgb_joints_v4"
