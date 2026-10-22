"""Observable-demo packaging and isolation from live execution records."""

import copy
from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pytest

from gpt_field.demonstration import (
    DEMONSTRATION_VERSION, DemoReference, load_demonstration, select_keyframe_indices,
    validate_demonstration,
)
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import image_to_data_url


def make_bundle(n=4):
    """Small full-episode fixture shared by client/runner/audit integration tests."""
    task = resolve_distance_task("drawer_bowl")
    steps = [{"action_index": i, "normalized_action": [.01, 0., 0., 0., 0., 0., -1. if i < n//2 else 1.],
              "after_state": [round(.1+.0005*i, 6), .2, 1., 0., 0., 0., .04, -.04]}
             for i in range(n)]
    image = image_to_data_url(np.full((8, 8, 3), 73, dtype=np.uint8), size=224, rotate=True)
    wrist = image_to_data_url(np.full((8, 8, 3), 119, dtype=np.uint8), size=224, rotate=True)
    return {
        "schema_version": DEMONSTRATION_VERSION, "task_key": task.key, "task_prompt": task.prompt,
        "source": {"format": "libero_hdf5", "file_name": "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
                   "file_sha256": "a"*64, "demo_key": "demo_0", "suite": task.suite,
                   "task_id": task.task_id, "episode_step_count": n},
        "preprocess": {"observation_timing": "after_action", "state_layout": "eef_xyz_axisangle_gripper_qpos",
            "orientation_conversion": "axisangle_shortest_equivalent",
            "action_layout": "normalized_world_osc_pose_7", "image_orientation": "rotate_180",
            "image_size": 224, "numeric_decimals": 6,
            "keyframe_selection": "uniform_plus_gripper_transitions_v1"},
        "steps": steps,
        "keyframes": [{"after_action_index": i, "external_image": image, "wrist_image": wrist}
                      for i in select_keyframe_indices([step["normalized_action"] for step in steps])],
    }


def write_bundle(tmp_path, bundle=None):
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(make_bundle() if bundle is None else bundle, indent=2))
    return path


def test_full_demo_is_one_static_reference_message_with_explicit_after_action_convention(tmp_path):
    bundle = make_bundle(60)
    path = write_bundle(tmp_path, bundle)
    reference = load_demonstration(path)
    message = reference.message()
    assert message["role"] == "user"
    header = json.loads(message["content"][0]["text"])
    assert header["type"] == "external_demonstration" and header["not_current_episode"] is True
    assert header["trajectory"] == bundle["steps"] and len(header["trajectory"]) == 60
    assert header["all_actions_and_after_states_included"] is True
    assert header["images_are_sampled_keyframes"] is True
    assert "AFTER action[t]" in header["trajectory_convention"]
    images = [part for part in message["content"] if part["type"] == "input_image"]
    labels = [json.loads(part["text"]) for part in message["content"][1:] if part["type"] == "input_text"]
    assert len(images) == 48 and len(labels) == 48
    assert [label["camera"] for label in labels] == ["external", "wrist"]*24
    assert all(label["type"] == "demonstration_keyframe" and label["not_current_episode"] for label in labels)
    assert all(image["detail"] == "high" for image in images)
    assert all("before_state" not in step for step in header["trajectory"])


def test_metadata_includes_source_and_full_content_identity_but_no_episode_outcome(tmp_path):
    bundle = make_bundle()
    path = write_bundle(tmp_path, bundle)
    meta = load_demonstration(path).metadata()
    assert meta["bundle_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["source_file_sha256"] == bundle["source"]["file_sha256"]
    assert meta["source_episode_step_count"] == meta["action_count"] == meta["state_count"] == 4
    assert meta["demo_key"] == "demo_0" and meta["all_steps_included"] is True
    assert meta["image_sampling"] == "keyframes_only"
    assert meta["task_key"] == "drawer_bowl"
    assert meta["preprocess"] == bundle["preprocess"]
    assert len(meta["full_step_sha256"]) == 64
    assert len(meta["frame_hashes"]) == 4
    assert meta["frame_hashes"][0]["external"]["pixel_sha256"] != meta["frame_hashes"][0]["wrist"]["pixel_sha256"]
    assert not {"success", "reward", "dones", "object_pose", "contact"}.intersection(meta)


def test_reference_is_immutable_and_every_returned_structure_is_independent(tmp_path):
    path = write_bundle(tmp_path)
    reference = load_demonstration(path)
    expected_message, expected_metadata = reference.message(), reference.metadata()
    altered_message, altered_meta = reference.message(), reference.metadata()
    altered_message["content"].clear()
    altered_meta["frame_hashes"][0]["external"]["png_sha256"] = "forged"
    assert reference.message() == expected_message and reference.metadata() == expected_metadata
    with pytest.raises(FrozenInstanceError):
        reference._bundle_bytes = b"forged"
    with pytest.raises(TypeError, match="load_demonstration"):
        DemoReference()
    snapshot = tmp_path / "snapshot.json"
    reference.write(snapshot)
    assert snapshot.read_bytes() == path.read_bytes()
    assert load_demonstration(snapshot).metadata() == expected_metadata


def test_export_validation_returns_independent_metadata_without_claiming_serialized_digest():
    bundle = make_bundle()
    before = copy.deepcopy(bundle)
    meta = validate_demonstration(bundle)
    assert bundle == before
    assert "bundle_sha256" not in meta
    meta["preprocess"]["observation_timing"] = "bad"
    assert bundle == before


@pytest.mark.parametrize("n", [1, 2, 4, 23, 24, 25, 60, 501])
def test_selection_is_chronological_deterministic_and_spans_entire_episode(n):
    actions = np.zeros((n, 7))
    actions[:n//2, 6] = -1.
    actions[n//2:, 6] = 1.
    selected = select_keyframe_indices(actions)
    assert selected == select_keyframe_indices(actions.tolist())
    assert selected == tuple(sorted(set(selected)))
    assert len(selected) == min(n, 24)
    assert selected[0] == 0 and selected[-1] == n-1
    if n >= 2:
        assert n//2 in selected and n//2-1 in selected


def test_gripper_transitions_use_last_nonzero_intent_and_are_bounded_if_many():
    actions = np.zeros((100, 7))
    actions[0, 6], actions[20, 6], actions[75, 6] = -1., 1., -1.
    selected = select_keyframe_indices(actions)
    assert {0, 19, 20, 74, 75, 99} <= set(selected)
    actions[:, 6] = [1 if i%2 else -1 for i in range(100)]
    selected = select_keyframe_indices(actions)
    assert len(selected) == 24 and selected[0] == 0 and selected[-1] == 99


@pytest.mark.parametrize("where,key", [
    ("root", "success"), ("source", "object_pose"), ("preprocess", "contact"),
    ("step", "reward"), ("step", "before_state"), ("frame", "libero_success"),
])
def test_hidden_or_ambiguous_extra_fields_are_rejected_not_silently_filtered(tmp_path, where, key):
    bundle = make_bundle()
    target = {"root": bundle, "source": bundle["source"], "preprocess": bundle["preprocess"],
              "step": bundle["steps"][0], "frame": bundle["keyframes"][0]}[where]
    target[key] = 1
    with pytest.raises(ValueError, match="non-allowlisted"):
        load_demonstration(write_bundle(tmp_path, bundle))


@pytest.mark.parametrize("change", [
    lambda b: b["steps"].pop(),
    lambda b: b["steps"][1].update(action_index=0),
    lambda b: b["steps"][0].update(action_index=True),
    lambda b: b["source"].update(episode_step_count=True),
    lambda b: b["steps"][0].update(after_state=[0.]*7),
    lambda b: b["steps"][0]["after_state"].__setitem__(0, True),
    lambda b: b["steps"][0]["normalized_action"].__setitem__(0, 1.01),
    lambda b: b["steps"][0]["normalized_action"].__setitem__(0, .1234567),
    lambda b: b["steps"][0]["after_state"].__setitem__(0, float("nan")),
    lambda b: b["steps"][0]["after_state"].__setitem__(0, 10**500),
])
def test_partial_nonchronological_or_invalid_numeric_sequences_fail(tmp_path, change):
    bundle = make_bundle()
    change(bundle)
    with pytest.raises(ValueError):
        load_demonstration(write_bundle(tmp_path, bundle))


@pytest.mark.parametrize("updates", [
    {"format": "simulator_replay"}, {"file_name": "/secret/demo.hdf5"},
    {"file_name": "demo.json"}, {"file_sha256": "not-a-hash"},
    {"demo_key": "demo_0+demo_1"}, {"demo_key": "demo_" + "9"*70},
    {"suite": "libero_spatial"}, {"task_id": True},
])
def test_invalid_or_wrong_task_source_provenance_is_rejected(tmp_path, updates):
    bundle = make_bundle()
    bundle["source"].update(updates)
    with pytest.raises(ValueError):
        load_demonstration(write_bundle(tmp_path, bundle))


@pytest.mark.parametrize("change", [
    lambda b: b.update(task_key="bowl_plate"),
    lambda b: b.update(task_prompt="close the drawer"),
    lambda b: b["preprocess"].update(observation_timing="before_action"),
    lambda b: b["preprocess"].update(image_orientation="unrotated"),
    lambda b: b["preprocess"].update(orientation_conversion="none"),
    lambda b: b["preprocess"].update(numeric_decimals=True),
])
def test_task_or_preprocessing_mismatch_cannot_enter_model_input(tmp_path, change):
    bundle = make_bundle()
    change(bundle)
    with pytest.raises(ValueError):
        load_demonstration(write_bundle(tmp_path, bundle))


@pytest.mark.parametrize("change", [
    lambda b: b["keyframes"].pop(),
    lambda b: b["keyframes"].reverse(),
    lambda b: b["keyframes"][0].update(after_action_index=True),
    lambda b: b["keyframes"][0].update(external_image="https://example.org/image.png"),
    lambda b: b["keyframes"][0].update(wrist_image="data:image/png;base64,invalid"),
    lambda b: b["keyframes"][0].update(external_image=image_to_data_url(np.zeros((8, 8, 3), dtype=np.uint8), size=8)),
])
def test_images_and_keyframe_indexing_are_strict(tmp_path, change):
    bundle = make_bundle()
    change(bundle)
    with pytest.raises(ValueError):
        load_demonstration(write_bundle(tmp_path, bundle))


def test_more_than_twenty_four_images_pairs_are_rejected(tmp_path):
    bundle = make_bundle(30)
    bundle["keyframes"].append(copy.deepcopy(bundle["keyframes"][-1]))
    with pytest.raises(ValueError, match="at most 24"):
        load_demonstration(write_bundle(tmp_path, bundle))


@pytest.mark.parametrize("raw", [b"", b"{not-json", b"\xff", b'{"schema_version":"a","schema_version":"b"}'])
def test_invalid_json_or_duplicate_keys_are_rejected(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_demonstration(path)


@pytest.mark.parametrize("actions,frames", [([], 24), ([[0.]*6], 24), ([[float("nan")]*7], 24),
                                          ([[2.]*7], 24), ([[0.]*7], 1), ([[0.]*7], 25),
                                          ([[0.]*7], True)])
def test_selector_rejects_invalid_inputs(actions, frames):
    with pytest.raises(ValueError):
        select_keyframe_indices(actions, frames)
