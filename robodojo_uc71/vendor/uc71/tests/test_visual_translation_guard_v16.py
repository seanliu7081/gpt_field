"""Distinct projected scene identities remain distinct throughout admission."""
import copy

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v16 import ProjectedTranslationGuardV16
from test_visual_translation_guard_v11 import plant as old_plant, pixels, context
from test_visual_translation_guard_v14 import assess, observe


def scene(*, mode="transit", ids=("target", "neighbor")):
    ctx = context(mode=mode, tool=(43, 53, 125, 77))
    ctx["nearby_obstacles"] = [
        {"id": identity, "external_roi_uv": (np.array([98+i, 53, 122+i, 77])/223).tolist()}
        for i, identity in enumerate(ids)]
    if not ids:
        ctx["obstacle_assessment"] = "none_relevant"
    return ctx


def plant(ctx=None):
    ctx = scene() if ctx is None else ctx
    _, monitor, state = old_plant()
    guard = ProjectedTranslationGuardV16()
    refresh(guard, monitor, state, ctx)
    assert guard.context_invalid is None
    return guard, monitor, state


def refresh(guard, monitor, state, ctx):
    image = pixels()
    return guard.configure({"external": image, "wrist": image}, ctx, state,
                           target_id="target", monitor=monitor)


def test_overlapping_distinct_objects_are_not_merged_or_aliased():
    ctx = scene()
    original = copy.deepcopy(ctx)
    guard, _, _ = plant(ctx)
    assert ctx == original and guard.context == ctx
    assert [row["id"] for row in guard.neighbors] == ["target", "neighbor"]
    assert [row["aliases"] for row in guard.neighbors] == [{"target"}, {"neighbor"}]
    assert set(guard._previous_snapshot["neighbors"]) == {"target", "neighbor"}
    assert guard.summary()["conflicting_obstacle_ids"] == ["target", "neighbor"]
    assert not guard.summary()["neighbor_identity_contract"]["projected_overlap_merges_objects"]


def test_omitted_objects_and_real_snapshot_survive_an_empty_refresh():
    guard, monitor, state = plant()
    records = list(guard.neighbors)
    refresh(guard, monitor, state, scene(ids=()))
    assert all(a is b for a, b in zip(records, guard.neighbors))
    assert set(guard._previous_snapshot["neighbors"]) == {"target", "neighbor"}
    assert guard.summary()["conflicting_obstacle_ids"] == ["target", "neighbor"]
    assert guard.context["nearby_obstacles"] == []


def test_same_identity_updates_support_without_replacing_other_overlapping_object():
    guard, monitor, state = plant()
    target, other = guard.neighbors
    track = target["track"]
    ctx = scene(ids=("target",))
    ctx["nearby_obstacles"][0]["external_roi_uv"] = (np.array([96, 52, 124, 78])/223).tolist()
    refresh(guard, monitor, state, ctx)
    assert guard.neighbors == [target, other]
    assert target["track"] is track
    assert np.allclose(guard._transport(target), [96, 52, 124, 78])
    assert target["aliases"] == {"target"}
    assert other["aliases"] == {"neighbor"}


def test_same_identity_lost_support_reseed_preserves_prior_extent():
    guard, monitor, state = plant()
    old = guard.neighbors[0]
    old["track"].reliable = False
    ctx = scene(ids=("target",))
    ctx["nearby_obstacles"][0]["external_roi_uv"] = (np.array([100, 55, 120, 75])/223).tolist()
    refresh(guard, monitor, state, ctx)
    new = guard.neighbors[0]
    assert new is not old and new["track"] is not old["track"]
    assert new["aliases"] == {"target"}
    assert np.allclose(new["box"], [98, 53, 122, 77])
    assert [row["id"] for row in guard.neighbors] == ["target", "neighbor"]


def test_contact_permission_exempts_exact_target_only_despite_overlap():
    guard, monitor, _ = plant(scene(mode="interaction"))
    risks = {row["id"]: row for row in guard.summary()["neighbors"]}
    assert risks["target"]["intended_contact"]
    assert not risks["neighbor"]["intended_contact"]
    assert guard.summary()["conflicting_obstacle_ids"] == ["neighbor"]
    # Direct observation without a validated metric path cannot use the target
    # exemption to approach the distinct, overlapping neighbor.
    result = guard.assess([.002, 0, 0], monitor=monitor, direct_observation=True)
    assert not result["accepted"]


def test_refresh_and_rename_preserve_acknowledged_uncertain_debt_and_direction():
    guard, monitor, state = plant(scene(ids=("neighbor",)))
    assert assess(guard, monitor, [.002, 0, 0])["accepted"]
    state = observe(guard, monitor, state, [.002, 0, 0], [.001, 0, 0])
    assert guard.observation_travel_m == pytest.approx(.001)
    ledger = copy.deepcopy(guard.experiments)
    for ids in (("neighbor",), (), ("renamed_neighbor",)):
        refresh(guard, monitor, state, scene(ids=ids))
        assert guard.observation_travel_m == pytest.approx(.001)
        assert guard.experiments == ledger
    assert {row["id"] for row in guard.neighbors} == {"neighbor", "renamed_neighbor"}
    assert all(len(row["aliases"]) == 1 for row in guard.neighbors)
    # Residual motion under a zero command remains charged after refresh.
    state = observe(guard, monitor, state, [0, 0, 0], [.0002, 0, 0])
    assert guard.observation_travel_m == pytest.approx(.0012)
    assert guard.experiments[0]["measured_travel_m"] == pytest.approx(.0012)


@pytest.mark.parametrize("bad,reason", [
    ("unobservable", "translation_neighborhood_unobservable_or_inconsistent"),
    ("inconsistent", "translation_neighborhood_unobservable_or_inconsistent"),
    ("bad_box", "translation_neighbor_invalid"),
    ("duplicate", "translation_neighbor_identity_duplicate"),
    ("contact", "translation_contact_role_inconsistent"),
    ("tool", "translation_tool_envelope_unavailable"),
])
def test_full_invalid_context_refuses_without_any_neighbor_mutation(bad, reason):
    guard, monitor, state = plant()
    rows = list(guard.neighbors)
    before = [guard._transport(row).copy() for row in rows]
    ctx = scene(ids=("new_object",))
    if bad == "unobservable": ctx["obstacle_assessment"] = "unobservable"
    elif bad == "inconsistent": ctx["obstacle_assessment"] = "none_relevant"
    elif bad == "bad_box": ctx["nearby_obstacles"].append({"id": "bad", "external_roi_uv": None})
    elif bad == "duplicate": ctx["nearby_obstacles"].append(copy.deepcopy(ctx["nearby_obstacles"][0]))
    elif bad == "contact": ctx["intended_contact_id"] = "target"
    elif bad == "tool": ctx["tool_external_roi_uv"] = None
    refresh(guard, monitor, state, ctx)
    assert guard.context == ctx and guard.context_invalid == reason
    assert all(a is b for a, b in zip(rows, guard.neighbors)) and len(guard.neighbors) == len(rows)
    assert all(np.array_equal(box, guard._transport(row)) for box, row in zip(before, rows))
    result = assess(guard, monitor, [.002, 0, 0])
    assert not result["accepted"] and result["reason"] == reason
