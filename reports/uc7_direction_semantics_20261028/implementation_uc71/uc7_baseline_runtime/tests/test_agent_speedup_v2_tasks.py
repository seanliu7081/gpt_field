"""agent_SpeedUp_v2 task registry: official prompts, frozen reset hashes, no task facts."""

import dataclasses
import json
from pathlib import Path
import re

import pytest

from gpt_field import agent_speedup_v2_tasks as tasks

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = {
    "bowl_plate": ("libero_spatial", 0,
                   "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    "drawer_bowl": ("libero_goal", 3, "open the top drawer and put the bowl inside"),
    "libero_10_t0": ("libero_10", 0, "put both the alphabet soup and the tomato sauce in the basket"),
    "libero_10_t1": ("libero_10", 1, "put both the cream cheese box and the butter in the basket"),
    "libero_10_t2": ("libero_10", 2, "turn on the stove and put the moka pot on it"),
    "libero_10_t3": ("libero_10", 3, "put the black bowl in the bottom drawer of the cabinet and close it"),
    "libero_10_t4": ("libero_10", 4, "put the white mug on the left plate and put the yellow and white mug on the "
                                     "right plate"),
    "libero_10_t5": ("libero_10", 5, "pick up the book and place it in the back compartment of the caddy"),
    "libero_10_t6": ("libero_10", 6, "put the white mug on the plate and put the chocolate pudding to the right of "
                                     "the plate"),
    "libero_10_t7": ("libero_10", 7, "put both the alphabet soup and the cream cheese box in the basket"),
    "libero_10_t8": ("libero_10", 8, "put both moka pots on the stove"),
    "libero_10_t9": ("libero_10", 9, "put the yellow and white mug in the microwave and close it"),
    "libero_goal_g0": ("libero_goal", 0, "open the middle drawer of the cabinet"),
    "libero_goal_g1": ("libero_goal", 1, "put the bowl on the stove"),
    "libero_goal_g2": ("libero_goal", 2, "put the wine bottle on top of the cabinet"),
    "libero_goal_g4": ("libero_goal", 4, "put the bowl on top of the cabinet"),
    "libero_goal_g5": ("libero_goal", 5, "push the plate to the front of the stove"),
    "libero_goal_g6": ("libero_goal", 6, "put the cream cheese in the bowl"),
    "libero_goal_g7": ("libero_goal", 7, "turn on the stove"),
    "libero_goal_g8": ("libero_goal", 8, "put the bowl on the plate"),
    "libero_goal_g9": ("libero_goal", 9, "put the wine bottle on the rack"),
}
# spec §6.3 (bowl/drawer: v1 frozen; t2, t7: Phase A measured)
SPEC_HASHES = {
    "bowl_plate": "8626d3e51bf8618a2c2793dbac9121dc8c87106197b95a7a6e7f483ffed65835",
    "drawer_bowl": "c6853a2d01f24753fce9fe3b06dd05f78d5d667c258fd048144c908f29321eb8",
    "libero_10_t2": "7b854d9bd19a0d91a843980142f15b6586416ff51790e447b044d6c89d1eceef",
    "libero_10_t7": "5ed858a8309893828ce7646d9187890c96345bfd9f5354e0f2815fe6ff565c2c",
}


def test_prompt_table_is_the_official_language():
    assert set(tasks.V2_TASK_KEYS) == set(OFFICIAL)
    for key, (suite, task_id, prompt) in OFFICIAL.items():
        spec = tasks.resolve_v2_task(key)
        assert (spec.suite, spec.task_id, spec.prompt) == (suite, task_id, prompt)


def test_frozen_hashes_episode_zero():
    for key in tasks.V2_TASK_KEYS:
        sha = tasks.frozen_reset_sha256(key, 0)
        assert re.fullmatch(r"[0-9a-f]{64}", sha), key
    for key, sha in SPEC_HASHES.items():
        assert tasks.frozen_reset_sha256(key, 0) == sha
    assert len({tasks.frozen_reset_sha256(k, 0) for k in tasks.V2_TASK_KEYS}) == len(tasks.V2_TASK_KEYS)
    assert tasks.frozen_reset_sha256("libero_10_t2", 1) is None    # unmeasured episodes stay fail-closed


def test_v1_runner_frozen_values_agree():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v1r", ROOT / "scripts/evaluate_agent_speedup_v1.py")
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    for (key, episode), sha in v1.DEFAULT_EXPECTED_SHA256.items():
        assert tasks.frozen_reset_sha256(key, episode) == sha
    from gpt_field.distance_tasks import resolve_distance_task
    for key in ("bowl_plate", "drawer_bowl"):
        old = resolve_distance_task(key)
        new = tasks.resolve_v2_task(key)
        assert (old.suite, old.task_id, old.prompt) == (new.suite, new.task_id, new.prompt)


def test_registry_holds_no_task_facts():
    fields = {f.name for f in dataclasses.fields(tasks.V2TaskSpec)}
    assert fields == {"key", "suite", "task_id", "prompt", "scene", "frozen_reset_sha256"}
    spec = tasks.resolve_v2_task("libero_10_t2")
    with pytest.raises(TypeError):
        spec.frozen_reset_sha256[0] = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.prompt = "x"


def test_alias_and_unknown_keys():
    assert tasks.resolve_v2_task("libero_goal_g3") is tasks.resolve_v2_task("drawer_bowl")
    for bad in ("libero_10_t10", "bowl", None, 3):
        with pytest.raises(ValueError):
            tasks.resolve_v2_task(bad)
    assert tasks.PRIMARY_TASK == "libero_10_t2" and tasks.REGRESSION_TASKS == ("bowl_plate", "drawer_bowl")


def test_sim_config_is_v1_protocol():
    for key in tasks.V2_TASK_KEYS:
        config, monitor = tasks.build_v2_sim_config(key)
        assert (config.seed, config.settle_steps, config.max_steps) == (7, 10, None)
        assert config.render_size == config.image_size == 512 and config.include_arm_joints is False
        assert config.stop_on_success is True and config.stop_on_collision is False
        assert config.collision_check_mode == ("task_manipulation_v1" if key in tasks.REGRESSION_TASKS else "none")
        assert monitor == {"goal_aware": True, "truth_window": True}


def test_measured_record_if_present():
    """The M0 re-measurement (scripts/agent_speedup_v2_measure_resets.py), when its record exists."""
    record = ROOT / "runs/agent_speedup_v2_sim_smoke_m0_resets/all_keys_ep0_pass3_registry.json"
    if not record.is_file():
        pytest.skip("M0 reset measurement record not present")
    data = json.loads(record.read_text())
    assert data["all_frozen_match"] and data["all_prompts_match"]
    measured = {row["key"]: row["sha256"] for row in data["rows"]}
    assert measured == {k: tasks.frozen_reset_sha256(k, 0) for k in tasks.V2_TASK_KEYS}
