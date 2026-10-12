from dataclasses import asdict
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.libero import LiberoConfig, SimProcess, SimulatorError, _LiberoWorker
from gpt_field.observation import OBSERVATION_KEYS, PolicyObservation, image_to_data_url


def fake_command(*, leak=False):
    image = image_to_data_url(np.zeros((4, 4, 3), dtype=np.uint8), size=4)
    observation = PolicyObservation(image, image, (0, 0, 1, 0, 0, 0, 0.04, -0.04), "pick up the mug").to_dict()
    if leak:
        observation["object_pose"] = [1, 2, 3]
    program = '''
import sys,json
observation=json.loads(sys.argv[1])
steps=0
for line in sys.stdin:
    request=json.loads(line)
    if request['op']=='metadata': result={'protocol':'test','source_sha':'fixture'}
    elif request['op']=='tasks': result={'tasks':[{'task_id':0,'prompt':'pick up the mug'}]}
    else:
        steps=0 if request['op']=='reset' else steps+1
        result={'observation':observation,'status':{'success':steps==2,'steps':steps,'terminated':steps==2,'evaluator_secret':'never goes to policy'}}
    print(json.dumps({'ok':True,'result':result}),flush=True)
'''
    return [sys.executable, "-u", "-c", program, json.dumps(observation)]


def test_process_round_trip_and_evaluator_policy_separation():
    with SimProcess(LiberoConfig(), command=fake_command()) as sim:
        assert sim.metadata["source_sha"] == "fixture"
        assert len(sim.list_tasks()) == 1
        observation = sim.reset(0, 49)
        assert set(observation.to_dict()) == OBSERVATION_KEYS
        assert "evaluator_secret" in sim.status
        observation, success = sim.step([0, 0, 0, 0, 0, 0, -1])
        assert not success
        _, success = sim.step([0, 0, 0, 0, 0, 0, -1])
        assert success
        with pytest.raises(SimulatorError, match="reset"):
            sim.step([0, 0, 0, 0, 0, 0, -1])
    assert sim._process is None


def test_extra_observation_keys_rejected_at_process_boundary():
    with SimProcess(LiberoConfig(), command=fake_command(leak=True)) as sim:
        with pytest.raises(ValueError, match="four allowlisted"):
            sim.reset(0, 0)


@pytest.mark.parametrize("task,episode", [(0, 50), (10, 0), (-1, 0), (0, -1), (0, True)])
def test_invalid_official_indices_never_spawn(task, episode):
    sim = SimProcess(LiberoConfig())
    with pytest.raises(ValueError):
        sim.reset(task, episode)
    assert sim._process is None


def test_development_suite_task_bounds():
    with SimProcess(LiberoConfig(suite="libero_90"), command=fake_command()) as sim:
        assert sim.reset(89, 49).prompt == "pick up the mug"
        with pytest.raises(ValueError, match="task_id"):
            sim.reset(90, 0)
    for suite in ("libero_spatial", "libero_object", "libero_goal"):
        assert LiberoConfig(suite=suite).suite == suite
    with pytest.raises(ValueError, match="suite"):
        LiberoConfig(suite="libero_plus")


def test_bad_actions_rejected_before_transport():
    with SimProcess(LiberoConfig(), command=fake_command()) as sim:
        sim.reset(0, 0)
        for action in ([0] * 6, [0] * 6 + [1.1], [float("nan")] + [0] * 6):
            with pytest.raises(ValueError, match="normalized"):
                sim.step(action)
        assert sim.status["steps"] == 0


def test_worker_error_is_reported_and_cleaned_up():
    command = [sys.executable, "-u", "-c", 'import sys; sys.stdin.readline(); print(\'{"ok":false,"error":"fixture failure"}\',flush=True)']
    sim = SimProcess(LiberoConfig(), command=command)
    with pytest.raises(SimulatorError, match="fixture failure"):
        sim.start()
    assert sim._process is None


def test_timeout_terminates_transport():
    command = [sys.executable, "-u", "-c", "import sys,time; sys.stdin.readline(); time.sleep(30)"]
    sim = SimProcess(LiberoConfig(response_timeout_s=0.1), command=command)
    with pytest.raises(SimulatorError, match="timed out"):
        sim.start()
    assert sim._process is None


def test_policy_side_does_not_import_simulator_dependencies():
    script = "import gpt_field.libero,sys; assert not any(k in sys.modules for k in ['libero','robosuite','mujoco','torch'])"
    import subprocess
    import os
    from pathlib import Path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


@pytest.mark.parametrize("max_steps", [0, -1, True, False, 2.5, math.inf, "220"])
def test_horizon_requires_positive_integer_or_none(max_steps):
    with pytest.raises(ValueError, match="max_steps"):
        LiberoConfig(max_steps=max_steps)


def test_unbounded_config_round_trips_as_json_null():
    serialized = json.dumps(asdict(LiberoConfig(max_steps=None)), allow_nan=False)
    assert json.loads(serialized)["max_steps"] is None
    assert LiberoConfig(**json.loads(serialized)).max_steps is None


@pytest.fixture
def worker_with_fake_environment(monkeypatch, tmp_path):
    """Exercise the actual worker lifecycle without loading a physics engine."""
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels,
                   "robot0_eef_pos": [0, 0, 0.5], "robot0_eef_quat": [0, 0, 0, 1],
                   "robot0_gripper_qpos": [0.03, -0.03]}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.success = False
            self.timestep = 0

        def seed(self, seed):
            pass

        def reset(self):
            return observation

        def set_init_state(self, state):
            return observation

        def step(self, action):
            self.timestep += 1
            done = self.timestep >= self.options["horizon"] and not self.options["ignore_done"]
            return observation, 0, done, {}

        def check_success(self):
            return self.success

        def close(self):
            pass

    initial_path = tmp_path / "init_files" / "libero_10" / "task.init"
    initial_path.parent.mkdir(parents=True)
    initial_path.write_bytes(b"fixture initial states")
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text("fixture task")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, load=lambda *args, **kwargs: [None] * 50))
    task = SimpleNamespace(language="pick up the mug", problem_folder="libero_10", init_states_file="task.init")

    def create(max_steps, *, stop_on_success=True):
        worker = _LiberoWorker.__new__(_LiberoWorker)
        worker.config = LiberoConfig(max_steps=max_steps, image_size=4, settle_steps=2,
                                     stop_on_success=stop_on_success)
        worker.env = None
        worker.benchmark_root = tmp_path
        worker.benchmark = SimpleNamespace(get_num_tasks=lambda: 10, get_task=lambda task_id: task,
                                           get_task_bddl_file_path=lambda task_id: str(bddl_path))
        worker._env_class = FakeEnvironment
        worker.reset(0, 0)
        return worker

    return create


def test_unbounded_worker_passes_220_actions_and_still_stops_on_success(worker_with_fake_environment):
    worker = worker_with_fake_environment(None)
    assert worker.env.options["horizon"] == math.inf
    assert worker.env.options["ignore_done"] is True
    action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(221):
        response = worker.step(action)
        assert response["status"]["horizon_reached"] is False
        assert response["status"]["terminated"] is False
    assert response["status"]["steps"] == 221
    json.dumps(response, allow_nan=False)
    worker.env.success = True
    response = worker.step(action)
    assert response["status"]["steps"] == 222
    assert response["status"]["success"] is True
    assert response["status"]["terminated"] is True
    assert response["status"]["horizon_reached"] is False
    with pytest.raises(RuntimeError, match="reset"):
        worker.step(action)


def test_finite_worker_stops_at_policy_horizon_excluding_settle_steps(worker_with_fake_environment):
    worker = worker_with_fake_environment(3)
    assert worker.env.options["horizon"] == 6
    assert worker.env.options["ignore_done"] is True
    assert worker.env.timestep == 2
    action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(2):
        assert worker.step(action)["status"]["terminated"] is False
    response = worker.step(action)
    assert response["status"]["steps"] == 3
    assert response["status"]["horizon_reached"] is True
    assert response["status"]["terminated"] is True
    assert response["status"]["success"] is False
    with pytest.raises(RuntimeError, match="reset"):
        worker.step(action)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_stop_on_success_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="stop_on_success"):
        LiberoConfig(stop_on_success=value)


def test_model_completion_mode_keeps_success_feedback_but_allows_further_actions(worker_with_fake_environment):
    worker = worker_with_fake_environment(None, stop_on_success=False)
    worker.env.success = True
    action = [0, 0, 0, 0, 0, 0, -1]
    for step in range(1, 4):
        response = worker.step(action)
        assert response["status"]["success"] is True
        assert response["status"]["terminated"] is False
        assert response["status"]["steps"] == step
    worker.env.success = False
    assert worker.step(action)["status"]["success"] is False


def test_model_completion_mode_still_honors_finite_horizon(worker_with_fake_environment):
    worker = worker_with_fake_environment(2, stop_on_success=False)
    worker.env.success = True
    assert worker.step([0, 0, 0, 0, 0, 0, -1])["status"]["terminated"] is False
    status = worker.step([0, 0, 0, 0, 0, 0, -1])["status"]
    assert status["terminated"] is status["horizon_reached"] is status["success"] is True


def test_success_at_reset_does_not_terminate_model_completion_mode(worker_with_fake_environment, monkeypatch):
    worker = worker_with_fake_environment(None, stop_on_success=False)
    monkeypatch.setattr(worker._env_class, "check_success", lambda self: True)
    status = worker.reset(0, 0)["status"]
    assert status["success"] is True and status["terminated"] is False
    assert worker.step([0, 0, 0, 0, 0, 0, -1])["status"]["steps"] == 1
