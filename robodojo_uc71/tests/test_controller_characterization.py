"""Pure robot-only proposal/trace analysis tests; no native dispatch."""
import json
from pathlib import Path

import numpy as np
import pytest

from robodojo_uc71.controller import ArmState
from robodojo_uc71 import controller_characterization as module
from robodojo_uc71.native_characterization import build_probe_plan
from robodojo_uc71.native_action_observer import clean_joint_action
from test_native_characterization import setup


def proposal(tmp_path, monkeypatch):
    _, fixed, _, _ = setup(tmp_path, monkeypatch)
    inventory_path, geometry_path = Path(fixed["initial_inventory_path"]), Path(fixed["geometry_candidate_path"])
    inventory, geometry = json.loads(inventory_path.read_text()), json.loads(geometry_path.read_text())
    for index, robot in enumerate(inventory["robots"]):
        robot["body_names"] = ["base_link", "link6"]
        robot["root_link_pose_w"] = {"values": [[index*.6-.3, -.45, .8, 1., 0., 0., 0.]]}
        robot["body_link_pose_w"] = {"values": [[[0., 0., 0., 1., 0., 0., 0.],
            [index*.6-.3, -.2, .92, 1., 0., 0., 0.]]]}
    geometry["link6_from_tool"] = [[1., 0., 0., .149], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    inventory_path.write_text(json.dumps(inventory)); geometry_path.write_text(json.dumps(geometry))
    fixed = build_probe_plan(inventory_path, geometry_path, fixed["preparation_path"])
    path = tmp_path / "fixed_plan.json"; path.write_text(json.dumps(fixed))
    return module.build_controller_probe_proposal(path)


def test_fixed_bounded_scope_and_fresh_clearance_before_close(tmp_path, monkeypatch):
    plan = proposal(tmp_path, monkeypatch)
    module.validate_controller_probe_proposal(plan)
    assert sum(row["native_actions"] for row in plan["stages"]) == 960 < 1400
    assert len(plan["stages"]) == 39
    assert plan["maximum_simulated_time_s"] == 38.4
    assert plan["candidate_controller"]["angular_speed_rad_s"] == .5
    assert all(row["controller_config"]["angular_speed_rad_s"] == .5 for row in plan["stages"])
    assert [(row["label"], row["native_actions"]) for row in plan["stages"] if row["kind"] == "orientation"] == [
        ("left_worldZ_rotation_0.5_pulse", 20), ("left_worldZ_rotation_0.5_return", 20),
        ("right_worldZ_rotation_0.5_pulse", 20), ("right_worldZ_rotation_0.5_return", 20)]
    assert plan["execution_authorized"] is False and plan["benchmark_attempt_started"] is False
    for index, stage in enumerate(plan["stages"]):
        if stage["kind"] == "gripper_close":
            previous = plan["stages"][index-1]
            assert previous["kind"] == "visual_checkpoint" and previous["native_actions"] == 0
            assert stage["operator_clearance_required_before"] is True
            assert stage["active_gripper_command"] == 0.
        if "return_from_retreat" in stage["label"]:
            assert stage["active_gripper_command"] == 1.
    assert plan["native_gripper_contract"]["native_close_target_m"] == -.01


def test_gripper_retreat_adds_base_derived_outward_clearance(tmp_path, monkeypatch):
    plan = proposal(tmp_path, monkeypatch)
    for arm in module.ARMS:
        row = next(s for s in plan["stages"] if s["label"] == arm+"_generic_negative_toolX_retreat")
        home = np.array(plan["home"][arm]["tool_pose_wxyz"][:3])
        outward_x = -.015 if arm == "left" else .015
        assert np.asarray(row["target_tool_pose_wxyz"][:3])-home == pytest.approx([-.025+outward_x, 0., .03])
        held = [s for s in plan["stages"] if s["active_arm"] == arm
                and s["kind"] in ("visual_checkpoint", "gripper_close", "gripper_open")]
        assert len(held) == 3
        assert all(s["target_tool_pose_wxyz"] == row["target_tool_pose_wxyz"] for s in held)
        assert plan["candidate_abort_envelopes"]["joint_distance_from_home_rad"] == .35
        assert plan["candidate_abort_envelopes"]["tool_distance_from_home_m"] == .065


def test_outward_direction_uses_horizontal_base_geometry_and_rejects_coincidence():
    inventory = {"robots": [
        {"arm_name": "left_arm", "root_link_pose_w": {"values": [[-1., -1., 3., 1., 0., 0., 0.]]}},
        {"arm_name": "right_arm", "root_link_pose_w": {"values": [[1., 1., -5., 1., 0., 0., 0.]]}}]}
    directions = module._base_outward_directions(inventory)
    assert directions["right"] == pytest.approx([2**-.5, 2**-.5, 0.])
    assert directions["left"] == pytest.approx(-directions["right"])
    inventory["robots"][1]["root_link_pose_w"]["values"][0][:2] = [-1., -1.]
    with pytest.raises(ValueError, match="Distinct horizontal"):
        module._base_outward_directions(inventory)


def test_worldZ_rotation_is_premultiplied_not_local():
    q = [2**-.5, 2**-.5, 0., 0.]
    angle = .08
    rotated = module._rotation(module._world_z_rotation(q, angle))
    rz = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    assert rotated == pytest.approx(rz@module._rotation(q))


def test_trace_analysis_reports_actual_sampled_speed_and_no_calibration():
    stage = {"label": "synthetic_z", "active_arm": "left", "native_actions": 10,
             "target_tool_pose_wxyz": [0., 0., .01, 1., 0., 0., 0.]}
    samples = []
    for index in range(11):
        z = min(index*.002, .01)
        arms = {arm: {"tool_pose_wxyz": [0., 0., z if arm == "left" else 0., 1., 0., 0., 0.],
            "joint_positions": [0.] * 6, "finger_positions": [.044, .044]} for arm in module.ARMS}
        samples.append({"physics_time_s": index*.04, "arms": arms})
    report = module.summarize_stage(stage, samples)
    assert report["arms"]["left"]["sampled_translation_speed_peak_m_s"] == pytest.approx(.05)
    assert report["arms"]["right"]["sampled_translation_speed_peak_m_s"] == 0.
    assert report["calibration_passed"] is False and report["automatic_configuration_selection"] is False
    settings = next(s for s in report["arms"]["left"]["settling_threshold_sweep"]
                    if s["position_tolerance_m"] == .001 and s["orientation_tolerance_rad"] == .01
                    and s["minimum_stable_samples"] == 5)
    assert settings["first_stable_native_action"] == 5
    samples[-1]["physics_time_s"] += .001
    with pytest.raises(ValueError, match="cadence"): module.summarize_stage(stage, samples)


def test_checkpoint_cannot_produce_a_cartesian_command():
    with pytest.raises(ValueError, match="does not use"):
        module.stage_controller({"kind": "visual_checkpoint"}, {})


def test_every_cartesian_stage_preserves_real_controller_values_at_native_boundary(tmp_path, monkeypatch):
    plan = proposal(tmp_path, monkeypatch)
    states = {}
    for arm in module.ARMS:
        home = plan["home"][arm]
        pose = home["tool_pose_wxyz"]
        offset = np.asarray(plan["tool_offset_link6_m"])
        states[arm] = ArmState(np.asarray(home["joint_positions"], dtype=np.float64), np.zeros(6),
            np.asarray(pose[:3])-module._rotation(pose[3:])@offset, pose[3:], np.eye(6),
            -np.ones(6)*10, np.ones(6)*10, np.ones(6)*5, offset)
    tested = []
    for stage in plan["stages"]:
        if stage["kind"] in ("joint_hold", "visual_checkpoint"):
            continue
        target = stage["target_tool_pose_wxyz"]
        reference = module.stage_controller(stage, states).command(states, target[:3], target[3:],
            profile=stage["profile"], gripper_commands={stage["active_arm"]:stage["active_gripper_command"]})
        action, _ = module.stage_joint_command(stage, module.stage_controller(stage, states), states)
        assert clean_joint_action(action) == action
        for arm in module.ARMS:
            values = action[f"{arm}_arm_joint_state"]
            assert all(type(value) is float for value in values)
            assert np.array_equal(np.asarray(values), reference.joint_positions[arm])
            assert action[f"{arm}_ee_joint_state"] == [reference.gripper_commands[arm]]
        tested.append(stage["label"])
    assert len(tested) == 36
    assert any(name.startswith("left_") for name in tested)
    assert any(name.startswith("right_") for name in tested)
