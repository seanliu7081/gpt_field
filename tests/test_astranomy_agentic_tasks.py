"""Task identity compatibility; no simulator, credentials, or model calls."""
import pytest

from gpt_field.agent_speedup_v2_tasks import V2_TASK_KEYS, resolve_v2_task
from gpt_field.astranomy_agentic_tasks import ASTRANOMY_TASK_KEYS, resolve_astranomy_task
from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task


@pytest.mark.parametrize("key", DISTANCE_TASK_KEYS)
def test_existing_distance_task_contracts_are_preserved(key):
    assert resolve_astranomy_task(key) is resolve_distance_task(key)


@pytest.mark.parametrize("key", V2_TASK_KEYS)
def test_additional_libero_identities_match_the_existing_registry(key):
    actual, official = resolve_astranomy_task(key), resolve_v2_task(key)
    assert (actual.key, actual.suite, actual.task_id, actual.prompt) == (
        official.key, official.suite, official.task_id, official.prompt)


def test_t08_and_drawer_alias_are_available():
    assert "libero_10_t8" in ASTRANOMY_TASK_KEYS
    assert "libero_goal_g3" in ASTRANOMY_TASK_KEYS
    task = resolve_astranomy_task("libero_10_t8")
    assert (task.suite, task.task_id, task.prompt) == (
        "libero_10", 8, "put both moka pots on the stove")
    assert resolve_astranomy_task("libero_goal_g3") is resolve_distance_task("drawer_bowl")
    assert len(ASTRANOMY_TASK_KEYS) == len(set(ASTRANOMY_TASK_KEYS))


@pytest.mark.parametrize("key", [None, [], True, "libero_10_t10", "unknown"])
def test_unknown_tasks_are_rejected(key):
    with pytest.raises(ValueError, match="Unknown Astranomy task"):
        resolve_astranomy_task(key)
