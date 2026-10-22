"""Lifecycle regressions: failure tools must not become success or infrastructure failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.robodojo_direct_sim import PandaKinematics


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robodojo_direct.py"
    spec = importlib.util.spec_from_file_location("_test_robodojo_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "freeze_sources", lambda output: None)
    monkeypatch.setattr(module, "source_manifest", lambda: {})
    monkeypatch.setattr(module.life, "DEFAULT_EXPECTED_SHA256", {})
    monkeypatch.setattr(PandaKinematics, "from_xml", lambda *a, **kw: SimpleNamespace(metadata={}))
    return module


class FakeSim:
    def __init__(self, config):
        self.metadata = {}
        self.status = {}
        self.closed = False

    def start(self):
        return self

    def reset(self, task_id, episode_index):
        url = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), size=512, rotate=False)
        self.status = {"success": False, "libero_success": False, "terminated": False, "steps": 0}
        return PolicyObservation(url, url, (0., 0., 1., np.pi, 0., 0., .04, -.04),
                                 resolve_distance_task("bowl_plate").prompt,
                                 tuple([0.] * 7), tuple([0.] * 7))

    def step(self, action):
        raise AssertionError("No physical action authorized by a give_up tool")

    def close(self):
        self.closed = True


class GiveUpClient:
    def __init__(self, *a, **kw):
        self.calls = 0

    def budget_snapshot(self):
        return {"api_calls": self.calls, "api_key_calls": 0}

    def predict(self, request):
        assert set(request) == {"external_image", "wrist_image", "robot_state", "joint_positions", "step"}
        assert not {"success", "reward", "collision"}.intersection(request)
        self.calls += 1
        return {"name": "give_up", "arguments": {"reason": "Cannot continue"}}

    def record_result(self, value):
        raise AssertionError("give_up has no pending motion tool result")


def test_give_up_is_unsuccessful_terminal_without_motion(runner, tmp_path):
    result = runner.execute(tmp_path / "attempt", task_key="bowl_plate", token_provider=lambda: "unused",
        video=False, client_factory=GiveUpClient, sim_factory=FakeSim)
    assert result["status"] == "model_give_up"
    assert result["steps"] == 0 and result["api_calls"] == 1
    assert result["libero_success"] is False
    assert result["completion_time_eligible"] is False
    assert result["success_rate"] == 0.0
    assert result["error"] is None


def test_changed_initial_state_stops_before_inference(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.life, "DEFAULT_EXPECTED_SHA256", {("bowl_plate", 0): "wrong"})
    result = runner.execute(tmp_path / "attempt", task_key="bowl_plate", token_provider=lambda: "unused",
        video=False, client_factory=GiveUpClient, sim_factory=FakeSim)
    assert result["status"] == "infrastructure_error"
    assert result["api_calls"] == 0 and result["steps"] == 0


def test_fresh_directory_required(runner, tmp_path):
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, task_key="bowl_plate", token_provider=lambda: "unused",
                       video=False, client_factory=GiveUpClient, sim_factory=FakeSim)


@pytest.mark.parametrize("failure,status", [(RuntimeError("Subscription unavailable"), "infrastructure_error"),
                                           (KeyboardInterrupt(), "user_stop")])
def test_failed_request_wait_is_counted_without_counting_a_model_response(runner, tmp_path, monkeypatch,
                                                                       failure, status):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.now)

    class FailingClient(GiveUpClient):
        def predict(self, request):
            self.calls += 1
            clock.now += 8.0
            raise failure

    output = tmp_path / "attempt"
    result = runner.execute(output, task_key="bowl_plate", token_provider=lambda: "unused",
        video=False, client_factory=FailingClient, sim_factory=FakeSim)
    assert result["status"] == status
    assert result["api_calls"] == 1 and result["model_responses"] == 0 and result["steps"] == 0
    assert result["timing_components"]["model_pipeline_s"] == 8.0
    assert result["control_without_video_s"] == 8.0
    progress = json.loads((output / "run_state.json").read_text())
    assert progress["api_calls"] == 1 and progress["model_responses"] == 0
