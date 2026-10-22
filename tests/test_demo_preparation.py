"""Observable demo preparation: timing, coordinate conventions, and read allowlist."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.demonstration import select_keyframe_indices, validate_demonstration
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import decode_image


spec = importlib.util.spec_from_file_location("_demo_preparation", Path(__file__).resolve().parents[1] / "scripts/prepare_libero_demonstration.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
TASK = resolve_distance_task("drawer_bowl")


class Forbidden:
    def __getitem__(self, key):
        raise AssertionError("Hidden state, reward, done, or object metadata was read")


class Group(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


@pytest.fixture
def data_group():
    count = 28
    actions = np.zeros((count, 7))
    actions[:, 0] = np.arange(count) / 100
    actions[:9, 6] = -1
    actions[9:20, 6] = 1
    actions[20:, 6] = -1
    pixels = np.zeros((count, 2, 2, 3), dtype=np.uint8)
    pixels[:, 0, 0] = [255, 0, 0]
    pixels[:, 0, 1] = [0, 255, 0]
    pixels[:, 1, 0] = [0, 0, 255]
    pixels[:, 1, 1] = [255, 255, 0]
    positions = np.zeros((count, 3))
    positions[:, 0] = .001 + np.arange(count) / 1000
    rotations = np.zeros((count, 3))
    rotations[0, 2] = 3.2
    obs = Group(ee_pos=positions, ee_ori=rotations,
                gripper_states=np.tile([.04, -.04], (count, 1)),
                agentview_rgb=pixels, eye_in_hand_rgb=pixels.copy(),
                joint_states=Forbidden(), object_states=Forbidden())
    demo = Group(actions=actions, obs=obs, states=Forbidden(), robot_states=Forbidden(),
                 rewards=Forbidden(), dones=Forbidden(), attrs={"init_state": Forbidden(), "model_file": Forbidden()})
    attrs = {
        "problem_info": json.dumps({"language_instruction": TASK.prompt}),
        "bddl_file_name": f"libero/libero/bddl_files/libero_goal/{TASK.prompt.replace(' ', '_')}.bddl",
        "macros_image_convention": "opengl",
        "env_args": json.dumps({"env_kwargs": {"controller_configs": {
            "type": "OSC_POSE", "control_delta": True, "input_min": -1, "input_max": 1,
            "output_min": [-.05, -.05, -.05, -.5, -.5, -.5],
            "output_max": [.05, .05, .05, .5, .5, .5]}}}),
    }
    return Group(demo_0=demo, attrs=attrs)


def build(data_group):
    return prepare.build_observable_demo(data_group,
        source_path=TASK.prompt.replace(' ', '_') + '_demo.hdf5', file_sha256='a' * 64)


def test_builder_preserves_all_post_action_pairs_and_reads_only_observables(data_group):
    result = build(data_group)
    metadata = validate_demonstration(result, task_key="drawer_bowl")
    assert metadata
    assert len(result["steps"]) == result["source"]["episode_step_count"] == 28
    assert result["preprocess"]["observation_timing"] == "after_action"
    assert [s["action_index"] for s in result["steps"]] == list(range(28))
    for i, step in enumerate(result["steps"]):
        assert step["normalized_action"][0] == round(i / 100, 6)
        assert step["after_state"][0] == round(.001 + i / 1000, 6)
        assert len(step["after_state"]) == 8
    assert tuple(f["after_action_index"] for f in result["keyframes"]) == select_keyframe_indices(
        [s["normalized_action"] for s in result["steps"]], max_frames=24)
    text = json.dumps(result)
    for forbidden in ('model_file', 'init_state', 'object_states', 'joint_states', 'robot_states', 'rewards', 'dones'):
        assert forbidden not in text


def test_images_use_both_axis_rotation_and_axisangle_matches_live_canonical_form(data_group):
    result = build(data_group)
    first = result["keyframes"][0]
    for key in ("external_image", "wrist_image"):
        image = decode_image(first[key])
        assert image.size == (224, 224)
        assert image.getpixel((0, 0)) == (255, 255, 0)
        assert image.getpixel((223, 223)) == (255, 0, 0)
    assert result["steps"][0]["after_state"][3:6] == [0., 0., round(3.2 - 2 * np.pi, 6)]
    assert result["preprocess"]["orientation_conversion"] == "axisangle_shortest_equivalent"


@pytest.mark.parametrize("attribute,value,error", [
    ("problem_info", json.dumps({"language_instruction": "put bowl on plate"}), "task language"),
    ("bddl_file_name", "libero_spatial/open_the_top_drawer_and_put_the_bowl_inside.bddl", "BDDL identity"),
    ("macros_image_convention", "opencv", "opengl RGB"),
])
def test_source_metadata_must_match_selected_task_and_image_protocol(data_group, attribute, value, error):
    data_group.attrs[attribute] = value
    with pytest.raises(ValueError, match=error):
        build(data_group)


@pytest.mark.parametrize("change,error", [
    ({"control_delta": False}, "delta OSC_POSE"),
    ({"output_max": [.1, .1, .1, .5, .5, .5]}, "action scale"),
    ({"input_max": 2}, "normalized"),
])
def test_incompatible_controller_protocol_rejected(data_group, change, error):
    env = json.loads(data_group.attrs["env_args"])
    env["env_kwargs"]["controller_configs"].update(change)
    data_group.attrs["env_args"] = json.dumps(env)
    with pytest.raises(ValueError, match=error):
        build(data_group)


@pytest.mark.parametrize("kind", ["nonfinite", "out_of_range", "unaligned_proprio", "unaligned_rgb"])
def test_invalid_or_temporally_unaligned_arrays_rejected(data_group, kind):
    demo = data_group["demo_0"]
    if kind == "nonfinite": demo["actions"][0, 0] = np.nan
    elif kind == "out_of_range": demo["actions"][0, 0] = 1.01
    elif kind == "unaligned_proprio": demo["obs"]["ee_pos"] = demo["obs"]["ee_pos"][:-1]
    else: demo["obs"]["eye_in_hand_rgb"] = demo["obs"]["eye_in_hand_rgb"][:-1]
    with pytest.raises(ValueError):
        build(data_group)


def test_cli_refuses_to_overwrite_existing_reference_before_reading_source(tmp_path):
    output = tmp_path / "demo.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare.main(["--source", str(tmp_path / "absent.hdf5"), "--output", str(output)])
    assert output.read_text() == "preserve"
