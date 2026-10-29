import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field import experiment
from gpt_field.libero import LiberoConfig
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.policy import Decision
from gpt_field.astra import APIError


class FakeSim:
    initial_success = False
    def __init__(self,config):
        self.status = {}
        self.metadata = {"source":"fake_for_tests"}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def list_tasks(self): return [{"id":0}]
    def reset(self,*args):
        self.status = {"success":self.initial_success,"terminated":self.initial_success}
        url = image_to_data_url(np.zeros((2,2,3),dtype=np.uint8))
        return PolicyObservation(url,url,(0,0,0.5,0,0,0,0.03,-0.03),"reach")
    def step(self,action):
        obs = self.reset()
        self.status = {"success":True,"terminated":True}
        return obs,True


class FakePolicy:
    method = "test_policy"
    update_steps = 5
    def __init__(self): self.calls = 0
    def reset(self): pass
    def act(self,obs):
        self.calls += 1
        return Decision((0,0,0,0,0,0,-1),{"status":"hold","refreshed":True})


@pytest.mark.parametrize("already_successful",[False,True])
def test_success_lifecycle_and_resume(tmp_path,monkeypatch,already_successful):
    monkeypatch.setattr(experiment,"SimProcess",FakeSim)
    monkeypatch.setattr(FakeSim,"initial_success",already_successful)
    policy = FakePolicy()
    output = tmp_path/"run"
    config = LiberoConfig(max_steps=2)
    experiment.run_libero(output,policy,config,task_ids=[0],episode_indices=[0],save_video=False)
    rows = [json.loads(x) for x in (output/"episodes.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "success"
    assert rows[0]["steps"] == (0 if already_successful else 1)
    calls = policy.calls
    experiment.run_libero(output,policy,config,task_ids=[0],episode_indices=[0],save_video=False,resume=True)
    assert policy.calls == calls
    assert len((output/"episodes.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError,match="resume rejected"):
        experiment.run_libero(output,policy,LiberoConfig(max_steps=3),task_ids=[0],episode_indices=[0],save_video=False,resume=True)


def test_recover_only_final_interrupted_append(tmp_path):
    path = tmp_path/"episodes.jsonl"
    path.write_bytes(b'{"status":"success"}\n{"status":')
    assert experiment._read_episode_records(path,True) == [{"status":"success"}]
    assert list(tmp_path.glob("*.interrupted-*"))
    assert path.read_text() == '{"status":"success"}\n'
    path.write_text('not json\n{"status":"success"}\n')
    with pytest.raises(ValueError,match="corruption"):
        experiment._read_episode_records(path,True)


@pytest.mark.parametrize("api_failure", [False, True])
def test_video_events_follow_actual_control_loop_even_on_api_failure(tmp_path, monkeypatch, api_failure):
    from gpt_field import video
    events = []

    class Recorder:
        def __init__(self, path, **kwargs):
            self.path = path
        def observe(self, obs, **kwargs):
            events.append("observe")
        def decision(self, obs, decision, **kwargs):
            events.append("plan")
        def executed(self, obs, **kwargs):
            assert kwargs["success"] is True
            events.append("executed")
        def close(self, **kwargs):
            events.append(kwargs["status"])
            self.path.write_bytes(b"test-video-placeholder")

    class Policy(FakePolicy):
        def act(self, obs):
            if api_failure:
                raise APIError("injected API failure")
            return super().act(obs)

    monkeypatch.setattr(experiment, "SimProcess", FakeSim)
    monkeypatch.setattr(video, "EpisodeVideoRecorder", Recorder)
    output = tmp_path / "run"
    experiment.run_libero(output, Policy(), LiberoConfig(max_steps=2),
                          task_ids=[0], episode_indices=[0], save_video=True)
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["status"] == ("api_failure" if api_failure else "success")
    assert events == (["observe", "api_failure"] if api_failure
                      else ["observe", "plan", "executed", "success"])
    assert (output / row["video"]).is_file()
    with gzip.open(output / row["log"], "rt") as stream:
        entries = [json.loads(line) for line in stream]
    assert entries[0]["event"] == "reset"
    if not api_failure:
        assert entries[-1]["event"] == "executed"
        assert entries[-1]["success"] is True
        assert set(entries[-1]["observation"]) == {
            "observation/image", "observation/wrist_image", "observation/state", "prompt"}


class LongSim(FakeSim):
    def __init__(self, config):
        super().__init__(config)
        self.steps = 0
    def step(self, action):
        self.steps += 1
        obs = FakeSim.reset(self)
        success = self.steps >= 225
        self.status = {"success": success, "terminated": success, "horizon_reached": False}
        return obs, success


def test_unbounded_evaluation_passes_old_limit_and_stops_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "SimProcess", LongSim)
    output = tmp_path / "unbounded"
    result = experiment.run_libero(output, FakePolicy(), LiberoConfig(max_steps=None),
                                  task_ids=[0], episode_indices=[0], save_video=False)
    row = json.loads((output / "episodes.jsonl").read_text())
    assert result["status"] == "complete"
    assert row["status"] == "success" and row["steps"] == 225
    assert json.loads((output / "manifest.json").read_text())["configuration"]["simulator"]["max_steps"] is None


def test_finite_evaluator_still_enforces_its_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "SimProcess", LongSim)
    output = tmp_path / "finite"
    experiment.run_libero(output, FakePolicy(), LiberoConfig(max_steps=3),
                          task_ids=[0], episode_indices=[0], save_video=False)
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["status"] == "timeout" and row["steps"] == 3


def test_interrupt_persists_incomplete_episode_without_timeout(tmp_path, monkeypatch):
    class InterruptingPolicy(FakePolicy):
        def act(self, obs):
            if self.calls == 2:
                raise KeyboardInterrupt()
            return super().act(obs)
    monkeypatch.setattr(experiment, "SimProcess", LongSim)
    output = tmp_path / "interrupted"
    result = experiment.run_libero(output, InterruptingPolicy(), LiberoConfig(max_steps=None),
                                  task_ids=[0], episode_indices=[0], save_video=False)
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["steps"] == 2 and row["status"] == "interrupted"
    assert result["status"] == "interrupted"
    assert json.loads((output / "run_state.json").read_text())["status"] == "interrupted"
