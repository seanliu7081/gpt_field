"""The new policy cannot replace the evaluator or change the observation boundary."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v5_protocol import prediction


spec = importlib.util.spec_from_file_location("v5_runner", Path(__file__).parents[1] /
    "scripts/evaluate_visual_field_subscription.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def args(tmp_path, **changes):
    values = dict(output=tmp_path/"trial", task="bowl_plate", controller="hybrid", method_version="v5",
        demo=None, initial_state=None, bddl=None, seed=7, episode_index=0,
        sim_python="python", libero_root="/fixture")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupted"])
def test_v5_evaluation_keeps_independent_completion_and_collision(outcome, tmp_path):
    sims = []
    class Sim:
        def __init__(self, config):
            assert config.include_arm_joints and config.max_steps is None
            assert config.collision_check_mode == "task_manipulation_v1"
            self.metadata = {}; self.status = {}; self.actions = []; self.closed = False
            sims.append(self)
        def start(self): pass
        def reset(self, *args):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False}}
            return observation()
        def step(self, action):
            if outcome == "interrupted" and len(self.actions) == 7:
                raise KeyboardInterrupt
            self.actions.append(action)
            if len(self.actions) == 8:
                collision = outcome in ("collision", "simultaneous")
                raw = outcome in ("success", "simultaneous")
                self.status.update(success=raw and not collision, libero_success=raw,
                    collision={"mode": "task_manipulation_v1", "detected": collision,
                               "private": "EVALUATOR_ONLY_MARKER"})
            return observation(), self.status["success"]
        def close(self): self.closed = True
    class Recorder:
        def __init__(self, *args): pass
        def frame(self, *args): pass
        def close(self): pass
    value = prediction(phase="verify", actions=[[0.]*7]*5)
    value["reason"] = "Model thinks finished; this is not an evaluation signal."
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort":"medium"}) for _ in range(3)))
    config = args(tmp_path)
    s = runner.execute(config, token_provider=lambda:"offline-oauth", transport=transport,
        sim_factory=Sim, recorder_factory=Recorder)
    assert sims[0].closed
    assert s["steps"] == (7 if outcome == "interrupted" else 8)
    assert s["task_success"] is (outcome == "success")
    assert s["completion_time_eligible"] is (outcome == "success")
    assert s["budget"]["api_key_calls"] == 0
    assert "EVALUATOR_ONLY_MARKER" not in json.dumps(transport.calls)
    assert s["control_without_video_s"] == s["control_elapsed_s"]-s["timing_components"]["video_render_s"]
    assert s["method_version"] == "v5"
    assert all(c["text"]["format"]["name"] == "visual_field_rgb_joints_v5" for c in transport.calls)
    manifest = json.loads((config.output/"manifest.json").read_text())
    assert manifest["max_policy_steps"] is None
    assert "measured" in manifest["orientation_hold"]
    assert "120" in manifest["field_refresh"]


@pytest.mark.parametrize("changes", [{"controller":"direct"}, {"demo":"not-a-demo.json"}])
def test_v5_rejects_unrequested_baseline_or_demo_before_auth(changes, tmp_path):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(args(tmp_path, **changes), token_provider=lambda: pytest.fail("Auth should not run"))
    assert not (tmp_path/"trial").exists()


def test_terminal_schema_repair_cost_is_counted_without_executing_actions(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    instances = []
    class Sim:
        def __init__(self, config):
            self.metadata = {}; self.closed = False; self.actions = []
            instances.append(self)
        def start(self): pass
        def reset(self, *args):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False}}
            return observation()
        def step(self, action):
            self.actions.append(action)
            pytest.fail("Invalid model output must never execute")
        def close(self): self.closed = True
    class Recorder:
        def __init__(self, *args): pass
        def frame(self, *args): pass
        def close(self): pass
    value = prediction(phase="grasp")
    value["visual_references"]["external"]["moving_id"] = value["target_id"]
    class SlowInvalidTransport(FakeTransport):
        def __call__(self, *args, **kwargs):
            clock[0] += 7.
            return super().__call__(*args, **kwargs)
    transport = SlowInvalidTransport(*(envelope(value, model="gpt-6-astra",
        reasoning={"effort":"medium"}) for _ in range(2)))
    result = runner.execute(args(tmp_path), token_provider=lambda: "offline-oauth",
        transport=transport, sim_factory=Sim, recorder_factory=Recorder)
    assert result["status"] == "subscription_error"
    assert result["error"]["category"] == "response_schema_validation"
    assert result["budget"]["api_calls"] == 2
    assert result["steps"] == 0 and not instances[0].actions and instances[0].closed
    assert result["timing_components"]["failed_decision_s"] == 14.
    assert result["timing_components"]["decision_s"] == 14.
    assert result["control_without_video_s"] == 14.
    assert not result["task_success"] and not result["completion_time_eligible"]
