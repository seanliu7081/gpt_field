"""Offline matrix contracts; no simulator, model, network or real data needed."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prepare_visual_field_matrix", Path(__file__).parents[1] / "scripts/prepare_visual_field_matrix.py")
matrix_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix_module)


def fixture_inputs(tmp_path):
    scenes = [
        {"id": name, "task": task, "actual_bddl_sha256": "a" * 64,
         "initial_state": str(tmp_path / name / "state.npy") if "original" not in name else None,
         "initial_state_sha256": "b" * 64 if "original" not in name else None,
         "bddl": str(tmp_path / name / "task.bddl") if name == "bowl_obstacles" else None}
        for name, task, _, _ in matrix_module.SCENE_SOURCES
    ]
    demos = {task: {"path": str(tmp_path / f"{task}.json"), "sha256": task * 3}
             for task in ("bowl_plate", "drawer_bowl")}
    return scenes, demos


def test_sixteen_trials_eight_pairs_match_shared_inputs(tmp_path):
    scenes, demos = fixture_inputs(tmp_path)
    result = matrix_module.build_matrix(scenes, demos, tmp_path)
    assert result["trial_count"] == len(result["trials"]) == 16
    assert result["pair_count"] == len(result["pairs"]) == 8
    assert len({t["id"] for t in result["trials"]}) == 16
    for pair in result["pairs"]:
        members = [t for t in result["trials"] if t["pair_id"] == pair["id"]]
        assert {t["controller"] for t in members} == {"hybrid", "direct"}
        for key in ("task", "shot", "demo", "demo_sha256", "initial_state", "initial_state_sha256",
                    "bddl", "bddl_sha256", "seed", "episode_index"):
            assert members[0][key] == members[1][key]
        for t in members:
            assert (t["demo"] is None) == (t["shot"] == "zero")
            assert ("--demo" in t["runner_arguments"]) == (t["shot"] == "one")
    assert [p["execution_order"][0].endswith("hybrid") for p in result["pairs"]] == [True, False, False, True] * 2


def test_matrix_is_deterministic_and_requires_all_four_scenes(tmp_path):
    scenes, demos = fixture_inputs(tmp_path)
    assert matrix_module.build_matrix(scenes, demos, tmp_path) == matrix_module.build_matrix(scenes, demos, tmp_path)
    with pytest.raises(ValueError, match="four declared scenes"):
        matrix_module.build_matrix(scenes[:-1], demos, tmp_path)


def test_baseline_scope_and_no_caps_are_explicit(tmp_path):
    result = matrix_module.build_matrix(*fixture_inputs(tmp_path), tmp_path)
    settings = result["settings"]
    assert settings["model"] == "gpt-6-astra"
    assert settings["reasoning_effort"] == "medium"
    assert settings["billing"] == "chatgpt_subscription"
    assert settings["api_key_calls_allowed"] == 0
    assert settings["direct_action_steps"] == 5
    assert settings["include_arm_joints"] is True
    assert all(settings[k] is None for k in ("max_steps", "max_wall_time_s", "max_model_calls"))
    assert not settings["model_completion_stopping"]
    assert not result["comparison"]["pure_direct_baseline_included"]
    assert not result["scope"]["target_object_OOD"]
    assert result["replicates_per_condition"] == 1
    assert settings["local_vision_inference_included_in_time"]
    assert "object_poses" in settings["forbidden_policy_inputs"]
    assert settings["primary_time_metric"] == "control_elapsed_s - timing_components.video_render_s"
    assert result["live_model_calls"] == result["simulator_rollouts"] == 0


def test_freeze_checks_hash_and_never_overwrites(tmp_path):
    source, target = tmp_path / "source", tmp_path / "snapshot"
    source.write_bytes(b"trusted bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hash mismatch"):
        matrix_module.freeze_file(source, target, "0" * 64)
    assert not target.exists()
    result = matrix_module.freeze_file(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    assert result["sha256"] == digest
    with pytest.raises(FileExistsError):
        matrix_module.freeze_file(source, target, digest)


def simple_layout():
    return matrix_module.joint_layout('<mujoco><worldbody><body><joint name="arm"/>'
        '<body><joint name="bowl" type="free"/></body></body></worldbody></mujoco>')


def test_position_difference_excludes_time_robot_and_settling_height():
    layout = simple_layout()
    state = np.zeros(1 + layout[1] + layout[2])
    altered = state.copy()
    altered[0] = 200  # timestamp
    altered[1] = .7  # robot angle
    altered[4] = .08  # bowl z, not xy
    with pytest.raises(ValueError, match="distinct target positions"):
        matrix_module.compare_target_positions(state, altered, layout, ["bowl"])
    altered[2] = .013
    comparison = matrix_module.compare_target_positions(state, altered, layout, ["bowl"])
    assert comparison["target_positions_differ"]
    assert comparison["targets"]["bowl"]["xy_difference_m"] == pytest.approx(.013)
    assert set(comparison["targets"]["bowl"]) == {"xy_difference_m", "xyz_difference_m", "different_xy"}


def test_changed_model_uses_explicit_target_joint_mapping():
    layout = simple_layout()
    state = np.zeros(1 + layout[1] + layout[2])
    altered = np.zeros(len(state) + 13)
    altered[10:13] = [.02, .03, .04]
    with pytest.raises(ValueError, match="explicit verified joint mapping"):
        matrix_module.compare_target_positions(state, altered, layout, ["bowl"])
    result = matrix_module.compare_target_positions(state, altered, layout, ["bowl"], evaluation_addresses={"bowl": 9})
    assert result["targets"]["bowl"]["xy_difference_m"] == pytest.approx(np.hypot(.02, .03))


@pytest.mark.parametrize("bad", [np.zeros(2), np.full(16, np.nan)])
def test_unsupported_demo_state_rejected(bad):
    with pytest.raises(ValueError, match="demo flat-state"):
        matrix_module.compare_target_positions(bad, bad, simple_layout(), ["bowl"])


@pytest.mark.parametrize("xml", [
    '<mujoco><worldbody><joint name="x"/><joint name="x"/></worldbody></mujoco>',
    '<mujoco><worldbody><joint/></worldbody></mujoco>',
    '<mujoco><include file="other.xml"/><worldbody/></mujoco>',
])
def test_ambiguous_joint_mapping_rejected(xml):
    with pytest.raises(ValueError):
        matrix_module.joint_layout(xml)
