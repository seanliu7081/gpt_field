"""Evaluator-only collision latching at actual physics substeps."""

from dataclasses import replace
import copy
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from gpt_field.libero import (
    LiberoConfig, _BowlObstacleCollisionMonitor, _ForbiddenCollisionStop, _LiberoWorker,
)
from gpt_field.observation import OBSERVATION_KEYS
from test_libero_custom_scene import ORIGINAL, official_parser


GEOMS = {name: i for i, name in enumerate(("finger", "palm", "arm", "bowl", "plate", "table",
                                         "ketchup", "milk", "ramekin", "cabinet", "visual", "other_arm"))}


def contact(first, second, distance=-1e-6):
    return SimpleNamespace(geom1=GEOMS[first], geom2=GEOMS[second], dist=distance)


def environment(contacts_by_substep=None, *, initial=(), endpoint=None, raw_success=False):
    body_names = ["world", "robot_base", "gripper_base", "finger_body", "arm_body", "bowl_body", "plate_body",
                  "table", "ketchup_body", "milk_body", "ramekin_body", "cabinet_body", "visual_body"]
    names = ["finger_collision", "hand_collision", "link_collision", "bowl_collision", "plate_collision",
             "table_collision", "ketchup_collision", "milk_collision", "ramekin_collision", "cabinet_collision",
             "visual_only", "other_link_collision"]
    bodies = [3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 4]
    model = SimpleNamespace(ngeom=len(names), npair=0, geom_contype=np.array([1]*10+[0, 1]),
        geom_conaffinity=np.array([1]*10+[0, 1]), geom_bodyid=np.array(bodies),
        body_parentid=np.array([0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        body_name2id=body_names.index, body_id2name=lambda i: body_names[i], geom_id2name=lambda i: names[i])
    frames = contacts_by_substep or []

    class Sim:
        def __init__(self):
            self.model = model
            self.calls = 0
            self.data = SimpleNamespace(contact=list(initial), ncon=len(initial))

        def set_contacts(self, contacts):
            self.data.contact, self.data.ncon = list(contacts), len(contacts)

        def step(self, *args, **kwargs):
            if self.calls < len(frames):
                self.set_contacts(frames[self.calls])
            self.calls += 1

        def forward(self):
            if endpoint is not None and self.calls >= 4:
                self.set_contacts(endpoint)

        def get_state(self):
            return SimpleNamespace(flatten=lambda: np.zeros(12, dtype=float))

    class Environment:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.sim = Sim()
            self.env = self
            self.objects_dict = {name: SimpleNamespace(root_body=body) for name, body in (
                ("akita_black_bowl_1", "bowl_body"), ("plate_1", "plate_body"), ("ketchup_1", "ketchup_body"),
                ("milk_1", "milk_body"), ("glazed_rim_porcelain_ramekin_1", "ramekin_body"))}
            self.fixtures_dict = {"wooden_cabinet_1": SimpleNamespace(root_body="cabinet_body")}
            self.robots = [SimpleNamespace(robot_model=SimpleNamespace(root_body="robot_base"),
                gripper=SimpleNamespace(root_body="gripper_base", important_geoms={"left_finger": ["finger_collision"]}))]
            self.regenerations = 0
            self.closed = False
            self.success = raw_success

        def observation(self):
            pixels = np.zeros((4, 4, 3), dtype=np.uint8)
            return {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels,
                    "robot0_eef_pos": [0, 0, .5+self.sim.calls*.001], "robot0_eef_quat": [0, 0, 0, 1],
                    "robot0_gripper_qpos": [.03, -.03]}

        def seed(self, seed):
            pass

        def reset(self):
            return self.observation()

        def set_init_state(self, state):
            return self.observation()

        def regenerate_obs_from_state(self, state):
            self.regenerations += 1
            return self.observation()

        def step(self, action):
            for _ in range(4):
                self.sim.forward()
                self.sim.step()
            return self.observation(), 0., self.success, {}

        def check_success(self):
            return self.success

        def close(self):
            self.closed = True

    return Environment


def monitored_worker(env):
    worker = _LiberoWorker.__new__(_LiberoWorker)
    worker.config = LiberoConfig(suite="libero_spatial", image_size=4, max_steps=None,
        initial_state_override="state.npy", task_bddl_override="scene.bddl", collision_check_mode="bowl_obstacles_v1")
    worker.env, worker.prompt = env, "pick up the bowl and place it on the plate"
    worker.status = {"steps": 0, "success": False, "terminated": False, "horizon_reached": False}
    worker._collision_monitor = _BowlObstacleCollisionMonitor(env)
    worker._collision_monitor.inspect(phase="initial_state")
    worker._collision_monitor.attach()
    return worker


@pytest.mark.parametrize("first,second", [
    ("finger", "ketchup"), ("finger", "milk"), ("finger", "ramekin"), ("arm", "cabinet"),
    ("palm", "table"), ("finger", "plate"), ("bowl", "ketchup"), ("bowl", "milk"),
    ("bowl", "ramekin"), ("bowl", "cabinet"), ("palm", "bowl"), ("arm", "bowl"),
])
def test_forbidden_robot_or_target_contacts_include_light_zero_force_contacts(first, second):
    env = environment(initial=[contact(first, second, 0.)])()
    monitor = _BowlObstacleCollisionMonitor(env)
    monitor.inspect(phase="initial_state")
    result = monitor.snapshot()
    assert result["detected"] and result["counts"]["forbidden_contact_points"] == 1
    assert result["first_event"]["contact_distance_m"] == 0.
    assert result["first_event"]["phase"] == "initial_state"
    assert result["first_event"]["action_index"] is result["first_event"]["physics_substep"] is None
    assert result["counts"]["physics_substeps"] == 0


@pytest.mark.parametrize("first,second", [
    ("finger", "bowl"), ("bowl", "table"), ("bowl", "plate"), ("milk", "table"),
    ("plate", "table"), ("cabinet", "table"), ("arm", "other_arm"), ("finger", "palm"),
])
def test_intentional_grasp_supports_and_excluded_internal_or_scene_contacts_are_allowed(first, second):
    env = environment(initial=[contact(first, second)])()
    monitor = _BowlObstacleCollisionMonitor(env)
    monitor.inspect(phase="initial_state")
    assert not monitor.snapshot()["detected"]
    assert monitor.snapshot()["counts"]["allowed_contact_points"] == 1


def test_positive_distance_margin_and_noncollision_visual_geometry_do_not_fail():
    env = environment(initial=[contact("finger", "ketchup", 1e-12), contact("bowl", "visual", -1.)])()
    monitor = _BowlObstacleCollisionMonitor(env)
    monitor.inspect(phase="initial_state")
    assert not monitor.snapshot()["detected"]
    assert monitor.snapshot()["counts"]["positive_distance_contacts_ignored"] == 1


def test_transient_substep_contact_latches_and_stops_remaining_physics_steps():
    env = environment([[], [contact("finger", "ketchup")], [], []])()
    worker = monitored_worker(env)
    result = worker.step([0, 0, 0, 0, 0, 0, -1])
    status = result["status"]
    assert env.sim.calls == 2 and env.regenerations == 1
    assert status["steps"] == 1 and status["terminated"] and not status["success"]
    assert status["libero_success"] is False
    event = status["collision"]["first_event"]
    assert (event["action_index"], event["physics_substep"]) == (0, 1)
    assert status["collision"]["counts"]["physics_substeps"] == 2
    assert set(result["observation"]) == OBSERVATION_KEYS
    assert "collision" not in json.dumps(result["observation"])
    with pytest.raises(RuntimeError, match="reset"):
        worker.step([0]*7)
    assert env.sim.calls == 2
    snapshot = copy.deepcopy(status["collision"])
    worker._collision_monitor.sim.set_contacts([contact("arm", "table", -.2)])
    worker._collision_monitor.inspect(phase="action", action_index=8, physics_substep=9)
    assert worker._collision_monitor.snapshot()["first_event"] == snapshot["first_event"]
    assert status["collision"] == snapshot


def test_final_integrated_endpoint_is_checked_in_addition_to_all_physics_substeps():
    env = environment([[], [], [], []], endpoint=[contact("bowl", "milk")])()
    status = monitored_worker(env).step([0]*7)["status"]
    assert status["collision"]["detected"]
    assert status["collision"]["counts"]["physics_substeps"] == 4
    assert status["collision"]["counts"]["checks"] == 6  # initial + four steps + endpoint
    assert status["collision"]["first_event"]["physics_substep"] == 3


@pytest.mark.parametrize("collision", [False, True])
def test_raw_goal_success_is_retained_but_only_contact_free_success_is_qualified(collision):
    env = environment([[contact("arm", "ketchup")] if collision else []], raw_success=True)()
    status = monitored_worker(env).step([0]*7)["status"]
    assert status["libero_success"] is True
    assert status["success"] is not collision and status["terminated"]
    assert status["collision"]["detected"] is collision


def test_action_and_substep_indices_are_zero_based_while_counts_are_cumulative():
    env = environment([[], [], [], [], [], [contact("finger", "milk")], [], []])()
    worker = monitored_worker(env)
    first = worker.step([0]*7)["status"]
    second = worker.step([0]*7)["status"]
    assert first["collision"]["counts"]["physics_substeps"] == 4
    assert second["collision"]["counts"]["physics_substeps"] == 6
    assert (second["collision"]["first_event"]["action_index"],
            second["collision"]["first_event"]["physics_substep"]) == (1, 1)


@pytest.mark.parametrize("mode", [None, "collision", True, []])
def test_invalid_monitor_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="collision_check_mode"):
        LiberoConfig(collision_check_mode=mode)


def test_monitor_requires_custom_bowl_scene_and_state():
    for kwargs in ({}, {"suite": "libero_spatial"},
                   {"initial_state_override": "state.npy", "task_bddl_override": "scene.bddl"}):
        with pytest.raises(ValueError, match="bowl_obstacles_v1"):
            LiberoConfig(collision_check_mode="bowl_obstacles_v1", **kwargs)
    assert LiberoConfig().collision_check_mode == "none"


@pytest.mark.parametrize("initial_collision", [False, True])
def test_reset_settling_is_unscored_but_initial_forbidden_contacts_fail_before_any_action(
        tmp_path, monkeypatch, initial_collision):
    initial_file = tmp_path / "init_files" / "libero_spatial" / "task.init"
    initial_file.parent.mkdir(parents=True)
    initial_file.write_bytes(b"official fixture")
    official = tmp_path / "official.bddl"
    official.write_text(ORIGINAL)
    custom = tmp_path / "scene.bddl"
    custom.write_text(ORIGINAL)
    state = tmp_path / "state.npy"
    np.save(state, np.zeros(12, dtype=float))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None,
        load=lambda *a, **k: [np.zeros(12)]*50))
    official_parser(monkeypatch)
    worker = _LiberoWorker.__new__(_LiberoWorker)
    worker.config = LiberoConfig(suite="libero_spatial", image_size=4, max_steps=None, settle_steps=2,
        initial_state_override=str(state), task_bddl_override=str(custom), collision_check_mode="bowl_obstacles_v1")
    worker.env, worker.metadata, worker.benchmark_root = None, {}, tmp_path
    worker._env_class = environment(initial=[contact("finger", "ketchup")] if initial_collision else [])
    task = SimpleNamespace(language="original bowl task", problem_folder="libero_spatial", init_states_file="task.init")
    worker.benchmark = SimpleNamespace(get_num_tasks=lambda:10, get_task=lambda _:task,
        get_task_bddl_file_path=lambda _:str(official))
    result = worker.reset(0, 0)
    status = result["status"]
    assert worker.env.sim.calls == 8  # two settling commands, not scored substeps
    assert status["collision"]["counts"]["physics_substeps"] == 0
    assert status["collision"]["counts"]["checks"] == 1
    assert status["collision"]["detected"] is initial_collision
    assert status["terminated"] is initial_collision and status["steps"] == 0
    assert set(result["observation"]) == OBSERVATION_KEYS
    if initial_collision:
        assert status["collision"]["first_event"]["phase"] == "initial_state"
        with pytest.raises(RuntimeError, match="reset"):
            worker.step([0]*7)


def test_default_mode_retains_existing_success_semantics_and_no_collision_status():
    env = environment([[contact("arm", "table")]], raw_success=True)()
    worker = monitored_worker(env)
    worker.config = replace(worker.config, collision_check_mode="none")
    env.sim.step = worker._collision_monitor._original_step
    worker._collision_monitor = None
    status = worker.step([0]*7)["status"]
    assert status["success"] and status["terminated"]
    assert "collision" not in status and "libero_success" not in status
