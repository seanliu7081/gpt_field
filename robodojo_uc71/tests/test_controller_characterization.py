"""Pure robot-only proposal/trace analysis tests; no native dispatch."""
import json
from pathlib import Path

import numpy as np
import pytest

from robodojo_uc71.controller import ArmState
from robodojo_uc71 import controller_characterization as module
from robodojo_uc71.native_characterization import build_probe_plan
from test_native_characterization import setup


def proposal(tmp_path, monkeypatch):
    _, fixed, _, _ = setup(tmp_path, monkeypatch)
    inventory_path, geometry_path = Path(fixed["initial_inventory_path"]), Path(fixed["geometry_candidate_path"])
    inventory, geometry = json.loads(inventory_path.read_text()), json.loads(geometry_path.read_text())
    for index, robot in enumerate(inventory["robots"]):
        robot["body_names"] = ["base_link", "link6"]
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
    assert sum(row["native_actions"] for row in plan["stages"]) == 1040 < 1400
    assert plan["maximum_simulated_time_s"] == 41.6
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


def test_gripper_retreat_is_negative_initial_toolX_plus_worldZ(tmp_path, monkeypatch):
    plan = proposal(tmp_path, monkeypatch)
    for arm in module.ARMS:
        row = next(s for s in plan["stages"] if s["label"] == arm+"_generic_negative_toolX_retreat")
        home = np.array(plan["home"][arm]["tool_pose_wxyz"][:3])
        assert np.asarray(row["target_tool_pose_wxyz"][:3])-home == pytest.approx([-.04, 0., .03])


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
