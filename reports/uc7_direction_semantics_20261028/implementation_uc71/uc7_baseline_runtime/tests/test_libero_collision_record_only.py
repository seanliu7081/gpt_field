"""Record-only collision evaluation retains contacts without altering execution."""

from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.libero import (
    LiberoConfig, _BowlObstacleCollisionMonitor, _LiberoWorker,
    _TaskManipulationCollisionMonitor,
)
from gpt_field.observation import OBSERVATION_KEYS
from test_libero_collision import contact, environment
from test_libero_custom_scene import ORIGINAL, official_parser


def make_worker(env, *, mode="bowl_obstacles_v1", stop_on_collision=False,
                stop_on_success=True, max_steps=None):
    worker = _LiberoWorker.__new__(_LiberoWorker)
    worker.config = LiberoConfig(
        suite="libero_spatial", image_size=4, max_steps=max_steps,
        initial_state_override="state.npy", task_bddl_override="scene.bddl",
        collision_check_mode=mode, stop_on_collision=stop_on_collision,
        stop_on_success=stop_on_success,
    )
    worker.env, worker.prompt = env, "pick up the bowl and place it on the plate"
    worker.status = {"steps": 0, "success": False, "terminated": False, "horizon_reached": False}
    if mode == "task_manipulation_v1":
        worker._collision_monitor = _TaskManipulationCollisionMonitor(
            env, task_key="bowl_plate", stop_on_collision=stop_on_collision)
    elif mode == "bowl_obstacles_v1":
        worker._collision_monitor = _BowlObstacleCollisionMonitor(env, stop_on_collision=stop_on_collision)
    else:
        worker._collision_monitor = None
    if worker._collision_monitor is not None:
        worker._collision_monitor.inspect(phase="initial_state")
        worker._collision_monitor.attach()
    return worker


@pytest.mark.parametrize("mode", ["bowl_obstacles_v1", "task_manipulation_v1"])
def test_collision_then_continuation_then_raw_libero_success(mode):
    # A transient contact in each action must neither truncate its four physics
    # substeps nor prevent later actions. The first event remains immutable.
    frames = [[], [contact("finger", "ketchup")], [], [],
              [], [contact("arm", "cabinet")], [], [], [], [], [], []]
    env = environment(frames)()
    worker = make_worker(env, mode=mode)
    first = worker.step([0, 0, 0, 0, 0, 0, -1])
    assert env.sim.calls == 4 and env.regenerations == 0
    assert first["status"]["collision"]["detected"]
    assert first["status"]["collision"]["counts"]["checks"] == 6
    assert not first["status"]["terminated"] and not first["status"]["success"]
    assert first["status"]["collision"]["collision_policy"] == "record_only"
    assert first["status"]["collision"]["stop_on_collision"] is False
    assert first["status"]["collision"]["success_qualification"] == "raw_libero_goal"
    second = worker.step([0] * 7)
    assert env.sim.calls == 8 and env.regenerations == 0
    assert second["status"]["collision"]["counts"]["forbidden_contact_points"] == 2
    assert second["status"]["collision"]["counts"]["checks"] == 11
    assert second["status"]["collision"]["first_event"] == first["status"]["collision"]["first_event"]
    assert first["status"]["collision"]["counts"]["forbidden_contact_points"] == 1
    assert not second["status"]["terminated"]
    env.success = True
    final = worker.step([0] * 7)
    assert env.sim.calls == 12 and env.regenerations == 0
    assert final["status"]["steps"] == 3
    assert final["status"]["success"] and final["status"]["libero_success"] and final["status"]["terminated"]
    assert not final["status"]["collision_free_success"]
    assert final["status"]["collision"]["counts"]["physics_substeps"] == 12
    assert set(final["observation"]) == OBSERVATION_KEYS
    for forbidden in ("collision", "libero_success", "ketchup", "cabinet", "record_only"):
        assert forbidden not in json.dumps(final["observation"])
    with pytest.raises(RuntimeError, match="reset"):
        worker.step([0] * 7)


@pytest.mark.parametrize("mode", ["bowl_obstacles_v1", "task_manipulation_v1"])
def test_strict_default_still_stops_first_contact_and_disqualifies_raw_success(mode):
    env = environment([[], [contact("finger", "ketchup")], [], []], raw_success=True)()
    worker = make_worker(env, mode=mode, stop_on_collision=LiberoConfig().stop_on_collision)
    result = worker.step([0] * 7)["status"]
    assert env.sim.calls == 2 and env.regenerations == 1
    assert result["libero_success"] and not result["success"] and result["terminated"]
    assert not result["collision_free_success"]
    assert result["collision"]["collision_policy"] == "stop_and_disqualify"
    assert result["collision"]["success_qualification"] == "libero_goal_and_no_forbidden_contact"
    with pytest.raises(RuntimeError, match="reset"):
        worker.step([0] * 7)


def test_endpoint_contacts_are_still_sampled_after_an_earlier_contact():
    env = environment([[contact("finger", "ketchup")], [], [], []],
                      endpoint=[contact("bowl", "milk")])()
    worker = make_worker(env)
    status = worker.step([0] * 7)["status"]
    assert status["collision"]["counts"]["forbidden_contact_points"] == 2
    assert status["collision"]["counts"]["physics_substeps"] == 4
    assert status["collision"]["counts"]["checks"] == 6
    assert status["collision"]["first_event"]["physics_substep"] == 0
    assert not status["terminated"]


def test_record_only_respects_horizon_and_opt_out_of_success_stopping():
    env = environment([[contact("arm", "table")], [], [], [], [], [], [], []], raw_success=True)()
    worker = make_worker(env, stop_on_success=False, max_steps=2)
    first = worker.step([0] * 7)["status"]
    assert first["success"] and not first["terminated"] and not first["horizon_reached"]
    second = worker.step([0] * 7)["status"]
    assert second["success"] and second["terminated"] and second["horizon_reached"]
    assert env.sim.calls == 8


@pytest.mark.parametrize("mode", ["bowl_obstacles_v1", "task_manipulation_v1"])
@pytest.mark.parametrize("raw_success", [False, True])
def test_initial_collision_records_without_disqualifying_or_blocking_reset(
        tmp_path, monkeypatch, mode, raw_success):
    initial_path = tmp_path / "init_files" / "libero_spatial" / "task.init"
    initial_path.parent.mkdir(parents=True)
    initial_path.write_bytes(b"fixture")
    official = tmp_path / "official.bddl"
    official.write_text(ORIGINAL)
    custom = tmp_path / "scene.bddl"
    custom.write_text(ORIGINAL)
    state = tmp_path / "state.npy"
    np.save(state, np.zeros(12, dtype=float))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda _: None, load=lambda *a, **k: [np.zeros(12)] * 50))
    official_parser(monkeypatch)
    worker = _LiberoWorker.__new__(_LiberoWorker)
    worker.config = LiberoConfig(
        suite="libero_spatial", image_size=4, max_steps=None, settle_steps=2,
        initial_state_override=str(state), task_bddl_override=str(custom),
        collision_check_mode=mode, stop_on_collision=False,
    )
    worker.env, worker.metadata, worker.benchmark_root = None, {}, tmp_path
    worker._env_class = environment(initial=[contact("finger", "ketchup")], raw_success=raw_success)
    task = SimpleNamespace(language="original bowl task", problem_folder="libero_spatial", init_states_file="task.init")
    worker.benchmark = SimpleNamespace(get_num_tasks=lambda: 10, get_task=lambda _: task,
        get_task_bddl_file_path=lambda _: str(official))
    first = worker.reset(0, 0)["status"]
    assert first["collision"]["first_event"]["phase"] == "initial_state"
    assert first["success"] is raw_success and first["libero_success"] is raw_success
    assert first["terminated"] is raw_success and first["collision_free_success"] is False
    assert first["collision"]["counts"]["physics_substeps"] == 0  # settling remains excluded
    if not raw_success:
        next_status = worker.step([0] * 7)["status"]
        assert next_status["steps"] == 1 and not next_status["terminated"]
        assert next_status["collision"]["counts"]["physics_substeps"] == 4
        assert worker.env.regenerations == 0


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_stop_on_collision_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match="stop_on_collision"):
        LiberoConfig(stop_on_collision=value)


def test_disabled_collision_monitor_makes_flag_inert_and_config_round_trips():
    config = LiberoConfig(stop_on_collision=False)
    assert config.collision_check_mode == "none"
    assert LiberoConfig(**json.loads(json.dumps(asdict(config)))).stop_on_collision is False
    env = environment([[contact("finger", "ketchup")]], raw_success=True)()
    status = make_worker(env, mode="none").step([0] * 7)["status"]
    assert status["success"] and status["terminated"]
    assert "collision" not in status and "collision_free_success" not in status
