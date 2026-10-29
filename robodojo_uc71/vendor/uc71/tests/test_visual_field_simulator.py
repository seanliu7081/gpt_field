"""Opt-in arm proprioception and task-aware evaluator contact rules."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gpt_field.libero import (LiberoConfig, SimProcess, SimulatorError, _LiberoWorker,
                             _TaskManipulationCollisionMonitor, _collision_task_key)
from gpt_field.observation import (JOINT_OBSERVATION_KEYS, OBSERVATION_KEYS,
                                   PolicyObservation)
from test_libero_collision import environment
from test_observation import example


def observed():
    raw = example()
    raw.update(robot0_joint_pos=np.linspace(-.4, .8, 7), robot0_joint_vel=np.linspace(-.1, .2, 7),
               camera_intrinsics="FORBIDDEN", camera_extrinsics="FORBIDDEN", depth="FORBIDDEN",
               object_positions="FORBIDDEN", target_region="FORBIDDEN", success=True)
    return raw


def test_joint_extension_is_opt_in_paired_immutable_and_round_trips():
    raw = observed()
    legacy = PolicyObservation.from_libero(raw, "move", image_size=4)
    assert set(legacy.to_dict()) == OBSERVATION_KEYS
    assert legacy.joint_position is legacy.joint_velocity is None
    extended = PolicyObservation.from_libero(raw, "move", image_size=4, include_arm_joints=True)
    assert set(extended.to_dict()) == JOINT_OBSERVATION_KEYS
    assert extended.state == legacy.state
    assert extended.joint_position == tuple(raw["robot0_joint_pos"])
    assert extended.joint_velocity == tuple(raw["robot0_joint_vel"])
    assert PolicyObservation.from_dict(extended.to_dict()) == extended
    raw["robot0_joint_pos"][0] = 999
    assert extended.joint_position[0] == -.4
    assert "FORBIDDEN" not in json.dumps(extended.to_dict())
    assert all(key not in extended.to_dict() for key in ("success", "depth", "object_positions"))


@pytest.mark.parametrize("name", ["joint_position", "joint_velocity"])
@pytest.mark.parametrize("bad", [[0]*6, [0]*8, [False]*7, [np.bool_(True)]*7,
                                 [float("nan")]*7, [float("inf")]*7,
                                 [1+0j]*7, ["0"]*7, [[0]]*7, 2.0])
def test_joint_values_are_seven_finite_real_non_boolean_numbers(name, bad):
    obs = PolicyObservation.from_libero(observed(), "move", image_size=4, include_arm_joints=True)
    with pytest.raises(ValueError, match="7 finite real"):
        replace(obs, **{name: bad})


@pytest.mark.parametrize("missing", ["observation/joint_position", "observation/joint_velocity"])
def test_partial_or_null_joint_transport_is_rejected(missing):
    data = PolicyObservation.from_libero(observed(), "move", image_size=4, include_arm_joints=True).to_dict()
    null = dict(data, **{missing: None})
    with pytest.raises(ValueError, match="not null"):
        PolicyObservation.from_dict(null)
    del data[missing]
    with pytest.raises(ValueError, match="both arm joint"):
        PolicyObservation.from_dict(data)
    with pytest.raises(ValueError, match="together"):
        replace(PolicyObservation.from_libero(observed(), "move", image_size=4), joint_position=(0.,)*7)


def test_unknown_keys_and_missing_measured_joints_never_fall_back_to_hidden_state():
    raw = observed()
    data = PolicyObservation.from_libero(raw, "move", image_size=4, include_arm_joints=True).to_dict()
    data["camera_intrinsics"] = [[1,0,0],[0,1,0],[0,0,1]]
    with pytest.raises(ValueError):
        PolicyObservation.from_dict(data)
    del raw["robot0_joint_pos"]
    raw["robot0_proprio-state"] = [42]*30
    with pytest.raises(KeyError, match="robot0_joint_pos"):
        PolicyObservation.from_libero(raw, "move", image_size=4, include_arm_joints=True)


@pytest.mark.parametrize("value", [True, False])
def test_joint_config_requires_actual_bool(value):
    assert LiberoConfig(include_arm_joints=value).include_arm_joints is value


@pytest.mark.parametrize("value", [1, 0, None, "true", [], np.bool_(True)])
def test_joint_config_and_factory_reject_truthy_non_bools(value):
    with pytest.raises(ValueError, match="boolean"):
        LiberoConfig(include_arm_joints=value)
    with pytest.raises(ValueError, match="boolean"):
        PolicyObservation.from_libero(observed(), "move", image_size=4, include_arm_joints=value)


def transport_command(include_joints):
    observation = PolicyObservation.from_libero(observed(), "move", image_size=4,
                                               include_arm_joints=include_joints).to_dict()
    program = """
import json,sys
observation=json.loads(sys.argv[1])
for line in sys.stdin:
    op=json.loads(line)['op']
    result={'fixture':True} if op=='metadata' else {'observation':observation,'status':{'success':False,'terminated':False,'contact_secret':'hidden'}}
    print(json.dumps({'ok':True,'result':result}),flush=True)
"""
    return [sys.executable, "-u", "-c", program, json.dumps(observation)]


def test_process_carries_joint_extension_without_evaluator_fields():
    with SimProcess(LiberoConfig(include_arm_joints=True), command=transport_command(True)) as sim:
        obs = sim.reset(0, 0)
        assert set(obs.to_dict()) == JOINT_OBSERVATION_KEYS
        after, success = sim.step([0]*7)
        assert after == obs and not success
        assert sim.status["contact_secret"] == "hidden"
        assert "contact_secret" not in json.dumps(obs.to_dict())


@pytest.mark.parametrize("requested,returned", [(True, False), (False, True)])
def test_process_enforces_joint_opt_in_at_both_ends(requested, returned):
    with SimProcess(LiberoConfig(include_arm_joints=requested), command=transport_command(returned)) as sim:
        with pytest.raises(SimulatorError, match="include_arm_joints"):
            sim.reset(0, 0)


def drawer_environment():
    """Collision groups mirror the inspected asset; no renderer or physics import."""
    env = environment()()
    model = env.sim.model
    old_ngeom = model.ngeom
    names = [model.geom_id2name(i) for i in range(old_ngeom)]
    bodies = [model.body_id2name(i) for i in range(len(model.body_parentid))]
    parent = list(model.body_parentid)
    top = len(bodies)
    bodies += ["wooden_cabinet_1_cabinet_top", "wooden_cabinet_1_cabinet_middle"]
    parent += [bodies.index("cabinet_body"), bodies.index("cabinet_body")]
    shell = [f"top_shell_{i}" for i in range(6)]
    handle = [f"top_handle_{i}" for i in range(3)]
    names += shell + handle + ["middle_handle", "middle_shell"]
    geom_bodyid = list(model.geom_bodyid) + [top]*9 + [top+1]*2
    model.ngeom = len(names)
    model.geom_bodyid = np.asarray(geom_bodyid)
    model.body_parentid = np.asarray(parent)
    model.geom_contype = np.r_[model.geom_contype, np.ones(11,dtype=int)]
    model.geom_conaffinity = np.r_[model.geom_conaffinity, np.ones(11,dtype=int)]
    model.geom_id2name = names.__getitem__
    model.geom_name2id = names.index
    model.body_id2name = bodies.__getitem__
    model.body_name2id = bodies.index
    model.site_name2id = lambda name: 0 if name == "wooden_cabinet_1_top_region" else -1
    model.site_bodyid = np.asarray([top])
    model.site_size = np.asarray([[.10,.08,.03]])
    env.sim.data.site_xpos = np.asarray([[.3,.2,1.]])
    # Non-identity orientation makes the containment test exercise world->local.
    env.sim.data.site_xmat = np.asarray([[0,-1,0, 1,0,0, 0,0,1]])
    root = ET.Element("body", name="cabinet_body")
    drawer = ET.SubElement(root,"body",name=bodies[top])
    ET.SubElement(drawer,"joint",name="wooden_cabinet_1_top_level",type="slide")
    ET.SubElement(drawer,"site",name="wooden_cabinet_1_top_region",type="box")
    ET.SubElement(drawer,"geom",name="drawer_visual",type="mesh",mesh="wooden_cabinet_1_wooden_cabinet_top_vis")
    for name in shell:ET.SubElement(drawer,"geom",name=name,type="box")
    ET.SubElement(drawer,"geom",name="handle_visual",type="mesh",mesh="wooden_cabinet_1_wooden_cabinet_top_handle_vis")
    for name in handle:ET.SubElement(drawer,"geom",name=name,type="box")
    env.fixtures_dict["wooden_cabinet_1"].get_obj = lambda:root
    original_observation = env.observation
    def observation():
        obs = original_observation()
        obs.update(robot0_joint_pos=np.arange(7)*.1+env.sim.calls*.001,
                   robot0_joint_vel=np.arange(7)*.02,
                   object_pose="FORBIDDEN", camera_matrix="FORBIDDEN", depth="FORBIDDEN")
        return obs
    env.observation = observation
    return env


def collision(env, first, second, *, pos=None, distance=0.):
    return SimpleNamespace(geom1=env.sim.model.geom_name2id(first),
        geom2=env.sim.model.geom_name2id(second), dist=distance,
        pos=np.asarray([.3,.2,1.] if pos is None else pos, dtype=float))


@pytest.mark.parametrize("first,second", [
    ("finger_collision","top_handle_0"), ("top_handle_2","finger_collision"),
    ("finger_collision","bowl_collision"), ("bowl_collision","table_collision"),
    ("top_shell_0","bowl_collision"), ("bowl_collision","top_shell_5"),
    ("middle_handle","table_collision"), ("link_collision","other_link_collision"),
])
def test_drawer_explicit_intended_contact_pairs_and_scene_exclusions(first, second):
    env = drawer_environment()
    monitor = _TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")
    env.sim.set_contacts([collision(env,first,second)])
    monitor.inspect(phase="initial_state")
    assert not monitor.snapshot()["detected"]


@pytest.mark.parametrize("first,second", [
    ("hand_collision","top_handle_0"), ("link_collision","top_handle_0"),
    ("finger_collision","top_shell_0"), ("finger_collision","middle_handle"),
    ("finger_collision","cabinet_collision"), ("finger_collision","plate_collision"),
    ("finger_collision","table_collision"), ("bowl_collision","plate_collision"),
    ("bowl_collision","top_handle_0"), ("bowl_collision","middle_shell"),
    ("bowl_collision","cabinet_collision"), ("bowl_collision","milk_collision"),
    ("hand_collision","bowl_collision"),
])
def test_drawer_never_exempts_whole_cabinet_other_handles_or_distractors(first, second):
    env = drawer_environment()
    monitor = _TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")
    env.sim.set_contacts([collision(env,first,second)])
    monitor.inspect(phase="initial_state")
    assert monitor.snapshot()["detected"]
    assert monitor.snapshot()["first_event"]["contact_distance_m"] == 0.


@pytest.mark.parametrize("local,allowed", [([.1019,0,0],True),([.1021,0,0],False),
    ([0,.0819,0],True),([0,.0821,0],False),([0,0,.0319],True),([0,0,.0321],False)])
def test_drawer_bowl_support_requires_rotated_containment_region(local, allowed):
    env = drawer_environment()
    monitor = _TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")
    pos = env.sim.data.site_xpos[0]+env.sim.data.site_xmat[0].reshape(3,3)@np.asarray(local)
    env.sim.set_contacts([collision(env,"bowl_collision","top_shell_0",pos=pos)])
    monitor.inspect(phase="action",action_index=5,physics_substep=2)
    snap=monitor.snapshot()
    assert snap["detected"] is not allowed
    if not allowed:assert snap["first_event"]["reason"] == "target_bowl_drawer_exterior_contact"


def test_original_bowl_scene_does_not_require_added_obstacles_and_plate_is_task_specific():
    env = drawer_environment()
    for name in ("ketchup_1","milk_1","glazed_rim_porcelain_ramekin_1"):
        env.objects_dict.pop(name)
    bowl = _TaskManipulationCollisionMonitor(env,task_key="bowl_plate")
    env.sim.set_contacts([collision(env,"bowl_collision","plate_collision")])
    bowl.inspect(phase="initial_state")
    assert not bowl.snapshot()["detected"]
    drawer = _TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")
    drawer.inspect(phase="initial_state")
    assert drawer.snapshot()["detected"]


@pytest.mark.parametrize("change", ["missing_marker","missing_geom","wrong_body","wrong_site_body","wrong_site_size"])
def test_drawer_asset_group_mismatch_fails_closed(change):
    env=drawer_environment();top=env.fixtures_dict["wooden_cabinet_1"].get_obj().find("body")
    if change=="missing_marker":top.remove(top.find("geom[@name='handle_visual']"))
    elif change=="missing_geom":top.remove(top.find("geom[@name='top_handle_2']"))
    elif change=="wrong_body":env.sim.model.geom_bodyid[env.sim.model.geom_name2id("top_handle_0")]=0
    elif change=="wrong_site_body":env.sim.model.site_bodyid[0]=0
    else:env.sim.model.site_size[0,0]=float("nan")
    with pytest.raises(ValueError):_TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")


def test_task_registry_rejects_other_tasks_and_accepts_original_layouts():
    assert LiberoConfig(suite="libero_spatial",collision_check_mode="task_manipulation_v1")
    assert LiberoConfig(suite="libero_goal",collision_check_mode="task_manipulation_v1")
    assert _collision_task_key("libero_spatial",0)=="bowl_plate"
    assert _collision_task_key("libero_goal",3)=="drawer_bowl"
    for suite,task in [("libero_spatial",1),("libero_goal",0),("libero_10",0)]:
        with pytest.raises(ValueError):_collision_task_key(suite,task)
    with pytest.raises(ValueError):LiberoConfig(collision_check_mode="task_manipulation_v1")


def worker_for(env, *, include_joints=True):
    worker=_LiberoWorker.__new__(_LiberoWorker)
    worker.config=LiberoConfig(suite="libero_goal",image_size=4,max_steps=None,
        collision_check_mode="task_manipulation_v1",include_arm_joints=include_joints)
    worker.env=env;worker.prompt="open the top drawer and put the bowl inside"
    worker.status={"steps":0,"success":False,"terminated":False,"horizon_reached":False}
    worker._collision_monitor=_TaskManipulationCollisionMonitor(env,task_key="drawer_bowl")
    worker._collision_monitor.inspect(phase="initial_state");worker._collision_monitor.attach()
    return worker


def test_generic_mode_latches_substep_contact_and_retains_joint_observables_and_raw_goal():
    env=drawer_environment()
    frames=[[],[collision(env,"finger_collision","middle_handle")],[],[]]
    def step():
        env.sim.set_contacts(frames[env.sim.calls]);env.sim.calls+=1
    env.sim.step=step;env.success=True
    worker=worker_for(env)
    result=worker.step([0,0,0,0,0,0,-1])
    assert env.sim.calls==2 and env.regenerations==1
    assert set(result["observation"])==JOINT_OBSERVATION_KEYS
    assert result["observation"]["observation/joint_position"][0]==.002
    assert result["status"]["libero_success"] is True
    assert result["status"]["success"] is False and result["status"]["terminated"]
    event=result["status"]["collision"]["first_event"]
    assert (event["action_index"],event["physics_substep"])==(0,1)
    assert "FORBIDDEN" not in json.dumps(result["observation"])
    assert "middle_handle" not in json.dumps(result["observation"])
    with pytest.raises(RuntimeError):worker.step([0]*7)


def test_generic_worker_reset_checks_initial_contact_after_unscored_settling(monkeypatch,tmp_path):
    env=drawer_environment()
    forbidden=collision(env,"finger_collision","middle_handle")
    env.sim.set_contacts([forbidden])
    worker=_LiberoWorker.__new__(_LiberoWorker)
    worker.config=LiberoConfig(suite="libero_goal",image_size=4,max_steps=None,settle_steps=2,
        include_arm_joints=True,collision_check_mode="task_manipulation_v1")
    worker.env=None;worker._env_class=lambda **kwargs:env;worker.benchmark_root=tmp_path
    initial=tmp_path/'init_files/libero_goal/task.init';initial.parent.mkdir(parents=True);initial.write_bytes(b'fixture')
    bddl=tmp_path/'task.bddl';bddl.write_bytes(b'fixture')
    task=SimpleNamespace(language="open the top drawer and put the bowl inside",problem_folder="libero_goal",init_states_file="task.init")
    worker.benchmark=SimpleNamespace(get_num_tasks=lambda:10,get_task=lambda i:task,get_task_bddl_file_path=lambda i:str(bddl))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(manual_seed=lambda n:None,load=lambda *a,**k:[np.zeros(12)]*50))
    result=worker.reset(3,0)
    assert env.sim.calls==8  # Two unscored four-substep settling controls.
    assert result['status']['terminated'] and not result['status']['success']
    assert result['status']['collision']['first_event']['phase']=='initial_state'
    assert result['status']['collision']['first_event']['action_index'] is None
    assert result['status']['collision']['counts']['physics_substeps']==0
    assert set(result['observation'])==JOINT_OBSERVATION_KEYS


def test_generic_worker_can_keep_legacy_four_field_observation_independently():
    result=worker_for(drawer_environment(),include_joints=False).step([0]*7)
    assert set(result['observation'])==OBSERVATION_KEYS
    assert result['status']['collision']['mode']=='task_manipulation_v1'
