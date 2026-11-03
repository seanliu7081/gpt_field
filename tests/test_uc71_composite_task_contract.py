"""Exact composite-task artifacts without relaxing official task contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_tasks import (
    COMPOSITE_TASK_KEY, CUSTOM_TASK_KEYS, V2_TASK_KEYS, build_v2_sim_config,
    custom_v2_task_contract, resolve_custom_v2_task, resolve_v2_task,
)
from gpt_field.libero import (
    LiberoConfig, _load_task_bddl, _validate_official_bddl_contract,
    validate_registered_custom_task_artifacts, validate_task_bddl_contract,
)
from test_uc71_custom_task import fake_parser, parsed
from test_libero import worker_with_fake_environment

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "output/layouts/drawer_bowl_right_moka_stove_left_no_obstacles_20261103_002"
OFFICIAL = Path("/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files/libero_10/"
                "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl")
TASK = resolve_v2_task(COMPOSITE_TASK_KEY)


def test_registration_preserves_official_and_previous_custom_profiles():
    config, monitor = build_v2_sim_config(TASK.key)
    assert TASK.key in CUSTOM_TASK_KEYS and TASK.key not in V2_TASK_KEYS
    assert (TASK.suite, TASK.task_id) == ("libero_10", 2)
    assert TASK.scene == "CUSTOM_DRAWER_BOWL_MOKA_STOVE"
    assert TASK.frozen_reset_sha256[0] == (LAYOUT / "initial_observation.sha256").read_text().strip()
    assert Path(config.task_bddl_override) == LAYOUT / "custom_task.bddl"
    assert Path(config.initial_state_override) == LAYOUT / "initial_state.npy"
    assert resolve_custom_v2_task(config.suite, config.custom_task_id, config.custom_task_prompt) == TASK
    assert LiberoConfig(**json.loads(json.dumps(asdict(config)))) == config
    assert monitor == {"goal_aware": True, "truth_window": True}
    for key in ("libero_10_t2", CUSTOM_TASK_KEYS[0]):
        assert custom_v2_task_contract(key) is None
        assert validate_registered_custom_task_artifacts(build_v2_sim_config(key)[0]) is None
    official = resolve_v2_task("libero_10_t2")
    assert official.prompt == "turn on the stove and put the moka pot on it"
    assert official.scene == "KITCHEN_SCENE3"


def test_exact_contract_is_explicit_and_accepts_frozen_copies(tmp_path, monkeypatch):
    config, _ = build_v2_sim_config(TASK.key)
    registered = custom_v2_task_contract(TASK.key)
    assert registered["external_camera_fovy_deg"] == 70.0
    assert registered["ordered_success_kind"] == "drawer_bowl_moka_v1"
    with pytest.raises(TypeError):
        registered["external_camera_fovy_deg"] = 45.0
    bddl, state = tmp_path / "frozen.bddl", tmp_path / "frozen.npy"
    bddl.write_bytes((LAYOUT / "custom_task.bddl").read_bytes())
    state.write_bytes((LAYOUT / "initial_state.npy").read_bytes())
    copied = replace(config, task_bddl_override=str(bddl), initial_state_override=str(state))
    artifact = validate_registered_custom_task_artifacts(copied)
    assert artifact["bddl_sha256"] == registered["bddl_sha256"]
    assert artifact["initial_state_sha256"] == registered["initial_state_sha256"]
    contract = validate_task_bddl_contract(OFFICIAL, bddl, expected_language=TASK.prompt)
    assert contract["contract_kind"] == "registered_composite"
    assert not contract["goal_preserved"] and not contract["target_assets_preserved"]
    assert contract["target_assets"] == {
        "akita_black_bowl_1": "akita_black_bowl", "wooden_cabinet_1": "wooden_cabinet",
        "flat_stove_1": "flat_stove", "moka_pot_1": "moka_pot"}
    fake_parser(monkeypatch)
    _validate_official_bddl_contract(OFFICIAL, bddl, contract)
    with pytest.raises(ValueError, match="preserve original"):
        validate_task_bddl_contract(OFFICIAL, bddl)


@pytest.mark.parametrize("replace_text", [
    ("(Turnon flat_stove_1)", "(Turnoff flat_stove_1)"),
    ("moka_pot_1 - moka_pot", "moka_pot_1 - white_bowl"),
    ("-0.230001", "-0.330001"),
    ("(Close wooden_cabinet_1_top_region)", "(Open wooden_cabinet_1_top_region)"),
    ("(:domain robosuite)", "(:domain robosuite)\n"),
])
def test_composite_cannot_change_goal_asset_layout_init_or_frozen_bytes(tmp_path, replace_text):
    config, _ = build_v2_sim_config(TASK.key)
    original = (LAYOUT / "custom_task.bddl").read_text()
    assert replace_text[0] in original
    altered = tmp_path / "altered.bddl"
    altered.write_text(original.replace(*replace_text))
    with pytest.raises(ValueError, match="exact frozen BDDL"):
        validate_task_bddl_contract(OFFICIAL, altered, expected_language=TASK.prompt)
    with pytest.raises(ValueError, match="exact frozen BDDL"):
        validate_registered_custom_task_artifacts(replace(config, task_bddl_override=str(altered)))


def test_composite_rejects_different_initial_state_before_reset(tmp_path):
    config, _ = build_v2_sim_config(TASK.key)
    state = np.load(config.initial_state_override, allow_pickle=False)
    state[1] += .001
    altered = tmp_path / "altered.npy"
    np.save(altered, state, allow_pickle=False)
    with pytest.raises(ValueError, match="exact frozen initial state"):
        validate_registered_custom_task_artifacts(replace(config, initial_state_override=str(altered)))


def test_official_parser_comparison_includes_changed_goal_and_assets(tmp_path, monkeypatch):
    copy = tmp_path / "copy.bddl"
    copy.write_bytes((LAYOUT / "custom_task.bddl").read_bytes())
    contract = validate_task_bddl_contract(OFFICIAL, copy, expected_language=TASK.prompt)

    def divergent(path):
        value = parsed(path)
        if Path(path) == copy:
            value["goal_state"] = [["turnon", "flat_stove_1"]]
        return value

    fake_parser(monkeypatch, divergent)
    with pytest.raises(ValueError, match="unexpected registered composite"):
        _validate_official_bddl_contract(OFFICIAL, copy, contract)


def test_worker_applies_registered_camera_before_state_and_settle(
        worker_with_fake_environment, monkeypatch):
    worker = worker_with_fake_environment(None)
    original = Path(worker.benchmark.get_task_bddl_file_path(2))
    original.write_bytes(OFFICIAL.read_bytes())
    worker.benchmark.get_task(2).name = "official_single_moka_stove"
    config, _ = build_v2_sim_config(TASK.key)
    worker.config = replace(config, image_size=4, render_size=4, settle_steps=2)
    worker.metadata = {}
    state = np.load(config.initial_state_override, allow_pickle=False)
    old_env = worker._env_class
    calls = []

    class Environment(old_env):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.sim = SimpleNamespace(
                get_state=lambda: SimpleNamespace(flatten=lambda: np.zeros_like(state)),
                model=SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=np.array([45.])),
                forward=lambda: calls.append("forward"))

        def set_init_state(self, values):
            calls.append(("state", float(self.sim.model.cam_fovy[0])))
            return super().set_init_state(values)

        def step(self, action):
            calls.append(("step", float(self.sim.model.cam_fovy[0])))
            return super().step(action)

    worker._env_class = Environment
    monkeypatch.setattr(sys.modules["torch"], "load", lambda *a, **k: [np.zeros(12)] * 50)
    fake_parser(monkeypatch)
    response = worker.reset(2, 0)
    assert calls == ["forward", ("state", 70.0), ("step", 70.0), ("step", 70.0)]
    assert response["observation"]["prompt"] == TASK.prompt
    assert response["status"]["registered_composite_task"]["task_key"] == TASK.key
    assert "registered_composite_task" not in response["observation"]
    assert worker.tasks()[2]["name"] == TASK.key
    for task_id, episode in ((8, 0), (2, 1)):
        with pytest.raises(ValueError, match="Registered custom task"):
            worker.reset(task_id, episode)
