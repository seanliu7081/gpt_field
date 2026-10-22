"""Real policy/transport integration with an independent offline evaluator."""
from dataclasses import replace
import json

import pytest

from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v5_runner import runner, args as args_v5
from test_visual_field_v6_protocol import prediction


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupt"])
def test_v6_gripper_execution_keeps_evaluator_privacy_and_terminal_precedence(tmp_path, outcome):
    instances = []
    class Sim:
        def __init__(self, config):
            assert config.max_steps is None and config.include_arm_joints
            self.metadata = {}; self.closed = False; self.actions = []
            instances.append(self)
        def start(self): pass
        def reset(self, *args):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False}}
            return self.obs()
        def obs(self):
            obs = observation()
            gap = .04-.002*len(self.actions)
            return replace(obs, state=(*obs.state[:6], gap, -gap))
        def step(self, action):
            if outcome == "interrupt" and len(self.actions) == 7:
                raise KeyboardInterrupt
            self.actions.append(tuple(action))
            if len(self.actions) == 8:
                collision = outcome in ("collision", "simultaneous")
                success = outcome in ("success", "simultaneous")
                self.status.update(success=success and not collision, libero_success=success,
                    collision={"mode": "task_manipulation_v1", "detected": collision,
                               "private": "V6_EVALUATOR_ONLY_MARKER"})
            return self.obs(), self.status["success"]
        def close(self): self.closed = True
    class Recorder:
        def __init__(self, *args): pass
        def frame(self, *args): pass
        def close(self): pass
    value = prediction(controller="gripper", phase="grasp")
    value["visual_references"] = {"external": None, "wrist": None}
    transport = FakeTransport(envelope(value, model="gpt-6-astra", reasoning={"effort":"medium"}))
    config = args_v5(tmp_path, method_version="v6")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    count = 7 if outcome == "interrupt" else 8
    assert result["steps"] == result["gripper_actions"] == count
    assert result["field_actions"] == result["direct_actions"] == 0
    assert instances[0].actions == [(0., 0., 0., 0., 0., 0., 1.)]*count
    assert len(transport.calls) == 1 and result["budget"]["api_key_calls"] == 0
    assert result["task_success"] is (outcome == "success")
    assert result["completion_time_eligible"] is (outcome == "success")
    if outcome in ("collision", "simultaneous"):
        assert result["status"] == "collision_failure"
    assert "V6_EVALUATOR_ONLY_MARKER" not in json.dumps(transport.calls)
    assert result["control_without_video_s"] == result["control_elapsed_s"]-result["timing_components"]["video_render_s"]
    assert instances[0].closed and result["cleanup_errors"] == []
    manifest = json.loads((config.output/"manifest.json").read_text())
    assert manifest["max_policy_steps"] is None and "gripper_settle" in manifest


@pytest.mark.parametrize("kwargs", [{"controller":"direct"}, {"demo":"not-a-demo.json"}])
def test_v6_unrequested_baseline_or_demo_rejected_before_auth(tmp_path, kwargs):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(args_v5(tmp_path, method_version="v6", **kwargs),
                       token_provider=lambda: pytest.fail("Must fail before auth"))
    assert not (tmp_path/"trial").exists()
