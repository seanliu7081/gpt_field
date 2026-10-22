"""User-check approval lifecycle: no models, simulators, or robot actions."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from gpt_field.agent_speedup_v2_user_check import (
    ReviewError, ReviewStore, canonical_sha256, initial_observation_sha256,
    semantic_plan, validate_review_plan, plan_validation,
)


@pytest.fixture
def plan():
    return {
        "scene": "Cup beside a tray.",
        "parts": [
            {"id": "A", "text": "cup: whole cup", "move": "free", "w": 20, "x": 30},
            {"id": "B", "text": "cup: curved handle", "move": "free", "w": 21, "x": 31},
            {"id": "C", "text": "tray: flat surface", "move": "fixed", "w": 40, "x": 50},
        ],
        "hold_true": [],
        "steps": [{
            "do": "place_on", "obj": "A", "grip": ["B"], "ref": "C", "sense": "any",
            "keep": [{"c": "upright", "p": "A"}], "until": [],
        }],
    }


@pytest.fixture
def observation():
    return {"step": 0, "pose": [0, 1, 2], "image": [[1, 2], [3, 4]]}


@pytest.fixture
def runtime():
    return {"agent_speedup_v2.py": "a" * 64, "user_check_runner.py": "b" * 64}


@pytest.fixture
def store(tmp_path, plan, observation, runtime):
    return ReviewStore.create(
        tmp_path / "review", task_key="fixture", episode_index=2,
        task_prompt="Place the cup on the tray.", initial_observation=observation,
        plan=plan, runtime_sha256=runtime,
    )


def claim(store, observation, runtime, **changes):
    args = dict(output=store.path.parent / "run", task_key="fixture", episode_index=2,
                initial_observation_sha256=initial_observation_sha256(observation),
                runtime_sha256=runtime)
    args.update(changes)
    return store.claim(**args)


def write_review(store, data, *, reseal_state=False):
    if reseal_state:
        data["state_sha256"] = canonical_sha256(
            {k: v for k, v in data.items() if k != "state_sha256"})
    store.review_path.write_text(json.dumps(data), encoding="utf-8")


def disk_review(store):
    return json.loads(store.review_path.read_text(encoding="utf-8"))


def test_native_plan_validation_and_semantics_are_lossless(plan):
    assert validate_review_plan(plan) == plan
    original = copy.deepcopy(plan)
    stripped = semantic_plan(plan)
    assert plan == original
    assert set(stripped["parts"][0]) == {"id", "text", "move"}
    changed_cells = copy.deepcopy(plan)
    changed_cells["parts"][0]["w"] = 0
    assert semantic_plan(changed_cells) == stripped
    changed_cells["steps"][0]["grip"] = []
    assert semantic_plan(changed_cells) != stripped


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(extra="not supported"),
    lambda p: p["steps"][0].update(strategy="frontal_bar_pinch"),
    lambda p: p["parts"][0].update(w=True),
    lambda p: p["parts"][0].update(x=65),
    lambda p: p["steps"][0].update(obj="H"),
    lambda p: p["steps"][0].update(ref="A"),
    lambda p: p.update(steps=[]),
    lambda p: p.update(scene="Caf\u00e9 cup beside a tray."),
])
def test_unexecutable_or_silently_rewritten_plan_rejected(plan, mutation):
    mutation(plan)
    with pytest.raises(ReviewError):
        validate_review_plan(plan)


def test_observation_hash_matches_frozen_v2_recipe(observation):
    import hashlib
    expected = hashlib.sha256(
        json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert initial_observation_sha256(observation) == expected


def test_direct_intent_can_be_approved_without_native_macro(tmp_path, plan, observation, runtime):
    plan["steps"][0]["do"] = "place_in"
    store = ReviewStore.create(
        tmp_path / "review", task_key="fixture", episode_index=2, task_prompt="Task.",
        initial_observation=observation, plan=plan, runtime_sha256=runtime)
    assert store.load()["validation"]["valid"]
    assert "relation_not_implemented" in {
        issue["code"] for issue in plan_validation(plan, require_native=True)["issues"]}
    assert store.approve(expected_revision=0, reviewed_steps=[0])["status"] == "approved"
    assert claim(store, observation, runtime)["plan"] == plan


def test_semantically_invalid_selector_state_can_be_saved(store, plan):
    plan["steps"][0]["ref"] = None
    saved = store.update(plan, expected_revision=0, reviewed_steps=[])
    assert saved["revision"] == 1
    assert not saved["validation"]["valid"]
    with pytest.raises(ReviewError):
        store.approve(expected_revision=1, reviewed_steps=[0])


def test_each_test_requires_explicit_approval(store, observation, runtime):
    with pytest.raises(ReviewError, match="user-approved"):
        claim(store, observation, runtime)
    with pytest.raises(ReviewError, match="Every step"):
        store.approve(expected_revision=0, reviewed_steps=[])
    approved = store.approve(expected_revision=0, reviewed_steps=[0])
    assert approved["approved"]["plan_sha256"] == canonical_sha256(approved["plan"])
    consumed = claim(store, observation, runtime)
    assert consumed["status"] == "consumed"
    assert consumed["claim"]["output"].endswith("/run")
    with pytest.raises(ReviewError):
        claim(store, observation, runtime)
    with pytest.raises(ReviewError, match="consumed"):
        store.approve(expected_revision=0, reviewed_steps=[0])


def test_approval_requires_all_steps(store, plan):
    plan["steps"].append({
        "do": "release", "obj": "A", "grip": [], "ref": None, "sense": "any",
        "keep": [], "until": [],
    })
    store.update(plan, expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError, match="Every step"):
        store.approve(expected_revision=1, reviewed_steps=[0])
    assert store.approve(expected_revision=1, reviewed_steps=[1, 0])["reviewed_steps"] == [0, 1]


@pytest.mark.parametrize("indices", [[True], [-1], [1], [0, 0], "0"])
def test_review_checks_are_typed_unique_indices(store, indices):
    with pytest.raises(ReviewError):
        store.approve(expected_revision=0, reviewed_steps=indices)


def test_edit_invalidates_approval_and_stale_browser_cannot_approve(store, plan, observation, runtime):
    store.approve(expected_revision=0, reviewed_steps=[0])
    plan["steps"][0]["grip"] = []
    state = store.update(plan, expected_revision=0, reviewed_steps=[])
    assert state["revision"] == 1
    assert state["approved"] is None
    assert state["status"] == "draft"
    with pytest.raises(ReviewError, match="revision"):
        store.approve(expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError, match="user-approved"):
        claim(store, observation, runtime)
    store.approve(expected_revision=1, reviewed_steps=[0])
    assert claim(store, observation, runtime)["plan"] == plan


def test_plan_edit_clears_step_checks_even_when_stale_browser_sends_them(store, plan):
    store.update(plan, expected_revision=0, reviewed_steps=[0])
    plan["steps"][0]["grip"] = []
    state = store.update(plan, expected_revision=1, reviewed_steps=[0])
    assert state["reviewed_steps"] == []


def test_supported_relation_rejects_options_its_controller_ignores(plan):
    plan["steps"][0]["sense"] = "open"
    with pytest.raises(ReviewError, match="sense must be any"):
        validate_review_plan(plan)
    plan["steps"][0].update(do="hold", sense="any")
    with pytest.raises(ReviewError) as error:
        validate_review_plan(plan)
    assert any(item["code"] == "ignored_selection" for item in error.value.issues)


@pytest.mark.parametrize("change", [
    {"task_key": "different"},
    {"episode_index": 3},
    {"episode_index": True},
    {"initial_observation_sha256": "c" * 64},
    {"runtime_sha256": {"agent_speedup_v2.py": "c" * 64}},
])
def test_approval_is_bound_to_task_reset_and_code(store, observation, runtime, change):
    store.approve(expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError):
        claim(store, observation, runtime, **change)
    assert store.load()["status"] == "approved"
    assert claim(store, observation, runtime)["status"] == "consumed"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(scene="User invented scene."),
    lambda p: p["parts"][0].update(text="different cup: whole cup"),
    lambda p: p["parts"][0].update(w=42),
    lambda p: p["steps"][0].update(until=[{"o": "effect", "p": "C", "txt": "invented phrase"}]),
])
def test_selector_updates_cannot_invent_model_provenance(store, plan, mutation):
    mutation(plan)
    with pytest.raises(ReviewError):
        store.update(plan, expected_revision=0, reviewed_steps=[])


def test_raw_file_edit_does_not_bypass_approval(store):
    store.approve(expected_revision=0, reviewed_steps=[0])
    data = disk_review(store)
    data["plan"]["steps"][0]["grip"] = []
    write_review(store, data)
    with pytest.raises(ReviewError, match="integrity"):
        store.load()


def test_approval_binding_survives_recomputed_mutable_state_hash(store):
    store.approve(expected_revision=0, reviewed_steps=[0])
    data = disk_review(store)
    data["plan"]["steps"][0]["grip"] = []
    write_review(store, data, reseal_state=True)
    with pytest.raises(ReviewError, match="Approved plan"):
        store.load()


def test_immutable_context_has_independent_seal(store):
    data = disk_review(store)
    data["task_key"] = "different"
    data["context_sha256"] = canonical_sha256(ReviewStore._context(data))
    write_review(store, data, reseal_state=True)
    with pytest.raises(ReviewError, match="Immutable"):
        store.load()


def test_changed_initial_image_requires_new_review(store, observation):
    observation["image"][0][0] = 99
    (store.path / "initial_observation.json").write_text(json.dumps(observation))
    with pytest.raises(ReviewError, match="Initial observation"):
        store.approve(expected_revision=0, reviewed_steps=[0])


@pytest.mark.parametrize("key,value", [("version", "future"), ("schema_version", 99),
                                     ("unexpected", True)])
def test_unknown_artifact_schema_and_version_rejected(store, key, value):
    data = disk_review(store)
    data[key] = value
    write_review(store, data, reseal_state=True)
    with pytest.raises(ReviewError, match="schema"):
        store.load()


def test_restoring_old_approved_file_cannot_replay_a_test(store, observation, runtime):
    store.approve(expected_revision=0, reviewed_steps=[0])
    old = disk_review(store)
    claim(store, observation, runtime)
    write_review(store, old)
    with pytest.raises(ReviewError, match="consumed"):
        claim(store, observation, runtime)


def test_concurrent_runners_can_consume_approval_only_once(store, observation, runtime):
    store.approve(expected_revision=0, reviewed_steps=[0])

    def attempt(_):
        try:
            return claim(ReviewStore(store.path), observation, runtime)["status"]
        except ReviewError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert results.count("consumed") == 1
    assert results.count("rejected") == 3


def test_creating_over_existing_review_never_erases_approval(store, plan, observation, runtime):
    original = store.load()["id"]
    with pytest.raises(ReviewError, match="already exists"):
        ReviewStore.create(
            store.path, task_key="fixture", episode_index=2, task_prompt="Task.",
            initial_observation=observation, plan=plan, runtime_sha256=runtime)
    assert store.load()["id"] == original
