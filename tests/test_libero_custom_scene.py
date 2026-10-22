"""Custom BDDL changes scene objects without changing the task or policy input."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.libero import (
    LiberoConfig, _validate_official_bddl_contract, validate_task_bddl_contract,
    validate_task_bddl_override,
)
from gpt_field.observation import OBSERVATION_KEYS
from test_libero import worker_with_fake_environment


ORIGINAL = """(define (problem LIBERO_Tabletop_Manipulation)
 (:domain robosuite)
 (:language Pick the black bowl and place it on the plate)
 (:fixtures main_table - table)
 (:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate)
 (:obj_of_interest akita_black_bowl_1 plate_1)
 (:init (On akita_black_bowl_1 main_table) (On plate_1 main_table))
 (:goal (And (On akita_black_bowl_1 plate_1))))
"""
CUSTOM = ORIGINAL.replace("plate_1 - plate)", "plate_1 - plate distractor_1 - red_coffee_mug)")
CONTRACT = {"language_preserved": True, "goal_preserved": True, "target_assets_preserved": True,
            "target_assets": {"akita_black_bowl_1": "akita_black_bowl", "plate_1": "plate"}}


def files(tmp_path, custom=CUSTOM):
    original_path, custom_path = tmp_path / "original.bddl", tmp_path / "custom.bddl"
    original_path.write_text(ORIGINAL)
    custom_path.write_text(custom)
    return original_path, custom_path


def parsed_description(path):
    return {"problem_name": "libero_tabletop_manipulation",
        "language_instruction": ["pick", "the", "black", "bowl", "and", "place", "it", "on", "the", "plate"],
        "goal_state": [["on", "akita_black_bowl_1", "plate_1"]],
        "obj_of_interest": ["akita_black_bowl_1", "plate_1"],
        "objects": {"akita_black_bowl": ["akita_black_bowl_1"], "plate": ["plate_1"],
                    **({"red_coffee_mug": ["distractor_1"]} if "distractor_1" in Path(path).read_text() else {})},
        "fixtures": {"table": ["main_table"]}}


def official_parser(monkeypatch, parser=parsed_description):
    monkeypatch.setitem(sys.modules, "libero.libero.envs.bddl_utils",
                        SimpleNamespace(robosuite_parse_problem=parser))


def test_file_identity_and_task_contract_allow_new_distractors_but_preserve_targets(tmp_path):
    original, custom = files(tmp_path)
    metadata = validate_task_bddl_override(custom)
    assert metadata == {"path": str(custom.resolve()), "sha256": hashlib.sha256(custom.read_bytes()).hexdigest(),
                        "byte_count": len(custom.read_bytes())}
    assert validate_task_bddl_contract(original, custom) == CONTRACT
    # Whitespace, comments and token capitalization do not change BDDL meaning.
    custom.write_text("; evaluator-only scene comment\n" + CUSTOM.upper())
    assert validate_task_bddl_contract(original, custom) == CONTRACT


@pytest.mark.parametrize("custom", [
    CUSTOM.replace("place it on the plate", "drop it on the floor"),
    CUSTOM.replace("(:goal (And (On akita_black_bowl_1 plate_1)))", "(:goal (And (On plate_1 main_table)))"),
    CUSTOM.replace("akita_black_bowl_1 - akita_black_bowl", "akita_black_bowl_1 - white_bowl"),
    CUSTOM.replace("plate_1 - plate", "plate_1 - red_coffee_mug"),
    CUSTOM.replace("akita_black_bowl_1", "akita_black_bowl_2"),
    CUSTOM.replace("LIBERO_Tabletop_Manipulation", "LIBERO_Living_Room_Tabletop_Manipulation"),
    CUSTOM.replace("(:obj_of_interest akita_black_bowl_1 plate_1)", "(:obj_of_interest plate_1)"),
    CUSTOM.replace("plate_1 - plate", "plate_1 plate_1 - plate"),
])
def test_changed_task_goal_or_target_assets_are_rejected(tmp_path, custom):
    original, changed = files(tmp_path, custom)
    with pytest.raises(ValueError):
        validate_task_bddl_contract(original, changed)


@pytest.mark.parametrize("data", [
    b"", b"\xff", b"not a BDDL definition", ORIGINAL.encode()[:-2],
    (ORIGINAL + ORIGINAL).encode(),
    ORIGINAL.replace("(:domain robosuite)", "(:domain robosuite) (:domain robosuite)").encode(),
    ORIGINAL.replace("(:domain robosuite)", "(:domain arbitrary)").encode(),
    ORIGINAL.replace("(:language Pick the black bowl and place it on the plate)", "").encode(),
    ORIGINAL.replace("(:init", "(:regions (r (:yaw_rotation ((__import__('os') 0))))) (:init").encode(),
    b"x" * (1024*1024 + 1),
])
def test_malformed_oversized_or_executable_bddl_is_rejected_without_simulator(tmp_path, data):
    path = tmp_path / "bad.bddl"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="task_bddl_override"):
        validate_task_bddl_override(path)


@pytest.mark.parametrize("value", ["", " ", 1, True])
def test_custom_bddl_config_requires_path_and_matching_state(value):
    with pytest.raises(ValueError, match="task_bddl_override"):
        LiberoConfig(task_bddl_override=value, initial_state_override="state.npy")


def test_new_scene_cannot_use_the_old_official_state_implicitly():
    with pytest.raises(ValueError, match="requires initial_state_override"):
        LiberoConfig(task_bddl_override="scene.bddl")
    config = LiberoConfig(task_bddl_override="scene.bddl", initial_state_override="state.npy")
    assert config.task_bddl_override == "scene.bddl"
    assert LiberoConfig().task_bddl_override is None


def test_pure_file_and_contract_validation_does_not_import_simulator_dependencies(tmp_path):
    original, custom = files(tmp_path)
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = """import sys
sys.path.insert(0, sys.argv[1])
from gpt_field.libero import validate_task_bddl_override, validate_task_bddl_contract
validate_task_bddl_override(sys.argv[3])
validate_task_bddl_contract(sys.argv[2], sys.argv[3])
assert not any(x in sys.modules for x in ['libero','robosuite','mujoco','torch'])
"""
    subprocess.run([sys.executable, "-c", script, source, str(original), str(custom)], check=True)


@pytest.mark.parametrize("changed_key", ["language_instruction", "goal_state", "obj_of_interest", "problem_name", "objects"])
def test_official_parser_disagreement_cannot_pass_even_if_text_contract_passes(tmp_path, monkeypatch, changed_key):
    original, custom = files(tmp_path)

    def inconsistent_parser(path):
        result = parsed_description(path)
        if Path(path) == custom:
            result[changed_key] = ({"white_bowl": ["akita_black_bowl_1"], "plate": ["plate_1"]}
                                   if changed_key == "objects" else ["changed"])
        return result

    official_parser(monkeypatch, inconsistent_parser)
    with pytest.raises(ValueError, match="Official parser"):
        _validate_official_bddl_contract(original, custom, CONTRACT)


def prepare_worker(worker_with_fake_environment, tmp_path, monkeypatch, *, state_size=25, model_size=25):
    worker = worker_with_fake_environment(None)
    original = Path(worker.benchmark.get_task_bddl_file_path(0))
    original.write_text(ORIGINAL)
    custom = tmp_path / "custom-scene.bddl"
    custom.write_text(CUSTOM)
    state_path = tmp_path / "scene-state.npy"
    state = np.arange(state_size, dtype=np.float64)/100
    np.save(state_path, state)
    # Official model had fewer objects and a shorter state. Only the custom
    # model's actual state dimension can validate the newly mapped state.
    monkeypatch.setattr(sys.modules["torch"], "load", lambda *a, **k: [np.zeros(12)]*50)
    original_environment = worker._env_class
    applied = []

    class CustomEnvironment(original_environment):
        def __init__(self, **kwargs):
            assert kwargs["bddl_file_name"] == str(custom.resolve())
            super().__init__(**kwargs)
            self.sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: np.zeros(model_size)))

        def set_init_state(self, selected):
            applied.append(selected.copy())
            return super().set_init_state(selected)

    worker.config = replace(worker.config, initial_state_override=str(state_path), task_bddl_override=str(custom))
    worker._env_class = CustomEnvironment
    worker.metadata = {}
    official_parser(monkeypatch)
    return worker, original, custom, state_path, state, applied


def test_new_object_state_matches_custom_model_and_all_scene_details_stay_evaluator_side(
        worker_with_fake_environment, tmp_path, monkeypatch):
    worker, original, custom, state_path, state, applied = prepare_worker(
        worker_with_fake_environment, tmp_path, monkeypatch)
    response = worker.reset(0, 7)
    np.testing.assert_array_equal(applied[0], state)
    assert applied[0].shape != (12,)
    status = response["status"]
    assert status["episode_index"] == 7 and status["initial_state_source"] == "custom_override"
    assert status["task_bddl_override_file"] == str(custom.resolve())
    assert status["task_bddl_override_sha256"] == status["bddl_sha256"] == hashlib.sha256(custom.read_bytes()).hexdigest()
    assert status["original_bddl_file"] == str(original.resolve())
    assert status["original_bddl_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert status["custom_task_contract"] == CONTRACT
    assert status["initial_state_override_sha256"] == hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert worker.metadata["custom_task_contract"] == CONTRACT
    assert response["observation"]["prompt"] == worker.benchmark.get_task(0).language
    assert set(response["observation"]) == OBSERVATION_KEYS
    assert not any(token in json.dumps(response["observation"]) for token in (
        "custom_task_contract", "task_bddl_override", "distractor_1", status["bddl_sha256"]))
    stepped = worker.step([0, 0, 0, 0, 0, 0, -1])
    assert stepped["status"]["bddl_sha256"] == status["bddl_sha256"]
    assert stepped["status"]["custom_task_contract"] == CONTRACT


@pytest.mark.parametrize("state_size,model_size", [(12, 25), (24, 25), (26, 25)])
def test_custom_state_must_match_built_model_before_state_application(
        worker_with_fake_environment, tmp_path, monkeypatch, state_size, model_size):
    worker, _, _, _, _, applied = prepare_worker(worker_with_fake_environment, tmp_path, monkeypatch,
                                                state_size=state_size, model_size=model_size)
    with pytest.raises(ValueError, match="custom BDDL simulator model"):
        worker.reset(0, 0)
    assert applied == []


def test_changed_goal_fails_before_environment_construction_or_state_application(
        worker_with_fake_environment, tmp_path, monkeypatch):
    worker, _, custom, _, _, applied = prepare_worker(worker_with_fake_environment, tmp_path, monkeypatch)
    custom.write_text(CUSTOM.replace("(:goal (And (On akita_black_bowl_1 plate_1)))",
                                    "(:goal (And (On plate_1 main_table)))"))
    worker._env_class = lambda **kwargs: pytest.fail("Changed task constructed an environment")
    with pytest.raises(ValueError, match=":goal"):
        worker.reset(0, 0)
    assert applied == []
