"""Explicit custom-task routing without policy, credentials, or simulator calls."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_tasks import (
    CUSTOM_TASK_KEYS, V2_TASK_KEYS, build_v2_sim_config, list_v2_tasks, resolve_v2_task,
)
from gpt_field.libero import (
    LiberoConfig, _bddl_assets, _load_task_bddl, _validate_official_bddl_contract,
    validate_task_bddl_contract,
)
from test_libero import worker_with_fake_environment

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "output/layouts/libero_10_t8_stove_off_bottle_obstacle_20261103_001"
SOURCE = LAYOUT.parent / "libero_10_t8_bottle_obstacle_20261103_001"
TASK = resolve_v2_task(CUSTOM_TASK_KEYS[0])


def parsed(path):
    sections, _ = _load_task_bddl(path)
    assets = _bddl_assets(sections)
    by_class = {}
    for name, kind in assets.items():
        by_class.setdefault(kind, []).append(name)
    return {"problem_name": sections["problem"],
            "language_instruction": sections[":language"],
            "goal_state": sections[":goal"],
            "obj_of_interest": sections[":obj_of_interest"],
            "objects": by_class, "fixtures": {}}


def fake_parser(monkeypatch, parser=parsed):
    monkeypatch.setitem(sys.modules, "libero.libero.envs.bddl_utils",
                        SimpleNamespace(robosuite_parse_problem=parser))


def test_custom_identity_is_explicit_and_official_iteration_unchanged():
    official = resolve_v2_task("libero_10_t8")
    assert official.prompt == "put both moka pots on the stove"
    assert official.frozen_reset_sha256[0] == "d07133dc3b5ced57c573b9148abfc1416f66e38b91618f8b0e83b09962cbc4bb"
    assert TASK.key not in V2_TASK_KEYS
    assert TASK not in list_v2_tasks()
    assert (TASK.suite, TASK.task_id, TASK.prompt) == (
        "libero_10", 8, "turn on the stove and put both moka pots on it")
    config, monitor = build_v2_sim_config(TASK.key)
    assert (config.custom_task_prompt, config.custom_task_id) == (TASK.prompt, TASK.task_id)
    assert Path(config.task_bddl_override) == LAYOUT / "custom_task.bddl"
    assert Path(config.initial_state_override) == LAYOUT / "initial_state.npy"
    assert monitor == {"goal_aware": True, "truth_window": True}
    assert LiberoConfig(**json.loads(json.dumps(asdict(config)))) == config
    normal, _ = build_v2_sim_config(official.key)
    assert normal.custom_task_prompt is normal.custom_task_id is None
    assert normal.task_bddl_override is normal.initial_state_override is None


@pytest.mark.parametrize("changes", [
    {"custom_task_prompt": None}, {"custom_task_id": None},
    {"custom_task_prompt": "turn on something else"},
    {"custom_task_id": 7}, {"custom_task_id": True},
    {"suite": "libero_goal"}, {"task_bddl_override": None},
    {"initial_state_override": None},
])
def test_custom_config_cannot_be_partial_or_undeclared(changes):
    config, _ = build_v2_sim_config(TASK.key)
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_language_change_requires_declared_task_and_both_parsers_agree(monkeypatch):
    original, custom = SOURCE / "custom_task.bddl", LAYOUT / "custom_task.bddl"
    with pytest.raises(ValueError, match=":language"):
        validate_task_bddl_contract(original, custom)
    contract = validate_task_bddl_contract(original, custom, expected_language=TASK.prompt)
    assert contract["language_preserved"] is False
    assert contract["goal_preserved"] and contract["target_assets_preserved"]
    assert contract["custom_task_language"] == TASK.prompt
    fake_parser(monkeypatch)
    _validate_official_bddl_contract(original, custom, contract)


@pytest.mark.parametrize("replacement", [
    ("(:language turn on the stove", "(:language turn off the stove"),
    ("(Turnon flat_stove_1))", "(Turnoff flat_stove_1))"),
    ("moka_pot_1 moka_pot_2 - moka_pot", "moka_pot_1 moka_pot_2 - white_bowl"),
])
def test_declared_custom_task_cannot_change_language_goal_or_assets(tmp_path, replacement):
    original_text = (LAYOUT / "custom_task.bddl").read_text()
    assert replacement[0] in original_text
    custom = tmp_path / "bad.bddl"
    custom.write_text(original_text.replace(*replacement))
    with pytest.raises(ValueError):
        validate_task_bddl_contract(SOURCE / "custom_task.bddl", custom, expected_language=TASK.prompt)


def test_official_parser_must_match_declared_new_language(monkeypatch):
    original, custom = SOURCE / "custom_task.bddl", LAYOUT / "custom_task.bddl"
    contract = validate_task_bddl_contract(original, custom, expected_language=TASK.prompt)

    def wrong_language(path):
        result = parsed(path)
        if Path(path) == custom:
            result["language_instruction"] = ["wrong"]
        return result

    fake_parser(monkeypatch, wrong_language)
    with pytest.raises(ValueError, match="Official parser"):
        _validate_official_bddl_contract(original, custom, contract)


def test_worker_lists_and_resets_new_instruction_without_changing_other_tasks(
        worker_with_fake_environment, tmp_path, monkeypatch):
    worker = worker_with_fake_environment(None)
    original = Path(worker.benchmark.get_task_bddl_file_path(8))
    original.write_text((SOURCE / "custom_task.bddl").read_text())
    official_task = worker.benchmark.get_task(8)
    official_task.name = "official_t8"
    official_task.language = "put both moka pots on the stove"
    config, _ = build_v2_sim_config(TASK.key)
    worker.config = replace(config, image_size=4, render_size=4, settle_steps=2)
    worker.metadata = {}
    old_env_class = worker._env_class
    state = np.load(config.initial_state_override, allow_pickle=False)
    monkeypatch.setattr(sys.modules["torch"], "load", lambda *a, **k: [np.zeros(12)] * 50)

    class CustomEnvironment(old_env_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: np.zeros_like(state)))

    worker._env_class = CustomEnvironment
    fake_parser(monkeypatch)
    listed = worker.tasks()
    assert listed[8]["prompt"] == TASK.prompt and listed[8]["name"] == TASK.key
    assert listed[7]["prompt"] == official_task.language
    response = worker.reset(8, 0)
    assert response["observation"]["prompt"] == TASK.prompt
    assert response["status"]["custom_task_contract"]["custom_task_language"] == TASK.prompt
    np.testing.assert_array_equal(worker.env.initial_state, state)
    assert worker.step([0, 0, 0, 0, 0, 0, -1])["observation"]["prompt"] == TASK.prompt
    for task_id, episode in ((7, 0), (8, 1)):
        with pytest.raises(ValueError, match="Registered custom task"):
            worker.reset(task_id, episode)


def test_runner_binds_custom_identity_and_keeps_official_profiles_strict():
    from test_agent_speedup_v2_runner import runner
    config = runner.configuration()
    custom, _ = build_v2_sim_config(TASK.key)
    allow = {"allow_initial_state_override": True, "allow_task_bddl_override": True}
    runner.validate_configuration(config, custom, TASK, **allow)
    with pytest.raises(ValueError, match="Official task profiles"):
        runner.validate_configuration(config, custom, resolve_v2_task("libero_10_t8"), **allow)
    plain, _ = build_v2_sim_config("libero_10_t8")
    with pytest.raises(ValueError, match="registered custom task"):
        runner.validate_configuration(config, plain, TASK, **allow)


def test_custom_task_cli_preflight_accepts_transit100_without_simulator_or_model():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                       GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M="0.10")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_agent_speedup_v2_user_check.py"),
         "preflight", "--task", TASK.key, "--episode-index", "0",
         "--expected-initial-observation-sha256", TASK.frozen_reset_sha256[0],
         "--reasoning-effort", "high", "--planning-video"],
        cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["lint_check"]["passed"]
    assert TASK.key in report["live_command"]
    assert TASK.prompt in result.stdout
    capabilities = report["framework"]["execution_fragments"]
    assert capabilities["profiles"]["transit"]["target_translation_m"] == .5
    assert capabilities["profiles"]["transit"]["action_translation_m"] == .1
    assert capabilities["routing"]["max_action_translation_m"] == .1
