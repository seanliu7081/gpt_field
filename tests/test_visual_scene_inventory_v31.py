"""Revision declarations are auditable hypotheses, never permission or truth."""
import copy
import pickle

import numpy as np
import pytest

from gpt_field.visual_scene_inventory_v16 import VisualSceneInventoryV16
from gpt_field.visual_scene_inventory_v31 import VisualSceneInventoryV31
from test_visual_scene_inventory_v16 import entry, observation


def revision(identity="object_a", kind="current_two_view", reason="Current image relationship conflicts with previous estimate"):
    return {"id": identity, "reason": reason, "evidence_kind": kind}


def plant():
    context = []
    inventory = VisualSceneInventoryV31(lambda: context)
    obs = observation()
    inventory.configure(obs, [entry()])
    return inventory, obs, context


def fingerprint(inventory):
    return pickle.dumps({key: value for key, value in inventory.__dict__.items()
                         if key != "revision_context_provider"})


def changed_entry(key="center"):
    result = entry()
    if key == "center": result["geometry"]["center_offset_m"][0] += .02
    elif key == "extent": result["geometry"]["half_extents_m"][1] += .01
    elif key == "uncertainty": result["geometry"]["uncertainty_m"] += .01
    return result


def test_initial_metric_declaration_needs_no_revision_and_has_current_binding():
    inventory, obs, _ = plant()
    report = inventory.summary(); row = report["entries"][0]; log = report["revision_log"][0]
    assert row["world_geometry"]["center_xyz_m"] == pytest.approx([.2, -.05, .9])
    assert row["estimated_at_observation"] == row["metric_age_observations"] == 0
    assert row["metric_declared_at_update"] == 1
    assert log["kind"] == "initial_metric_declaration" and log["revision"] is None
    assert log["observation_binding"]["state8"] == obs["observation/state"]
    assert set(log["observation_binding"]["rgb_sha256"]) == {"external", "wrist"}
    assert not log["geometry_measured_or_certified"] and not report["clearance_certified"]


@pytest.mark.parametrize("key", ["center", "extent", "uncertainty"])
def test_changed_geometry_without_record_rejected_before_any_mutation(key):
    inventory, obs, _ = plant(); before = fingerprint(inventory)
    with pytest.raises(ValueError, match="Explicit scene revision required"):
        inventory.configure(obs, [entry("new_object"), changed_entry(key)])
    assert fingerprint(inventory) == before and "new_object" not in inventory.objects


@pytest.mark.parametrize("kind", sorted(VisualSceneInventoryV31.revision_evidence_kinds))
def test_valid_explicit_revision_records_previous_world_current_view_and_claim(kind):
    inventory, obs, context = plant(); old = copy.deepcopy(inventory.objects["object_a"]["world_geometry"])
    context.append(revision(kind=kind)); current = changed_entry()
    inventory.configure(obs, [current]); report = inventory.summary(); log = report["revision_log"][-1]
    assert log["previous_world_geometry"] == old
    assert log["new_world_geometry"]["center_xyz_m"] == pytest.approx([.22, -.05, .9])
    assert log["revision"] == context[0]
    assert log["evidence_status"] == "model_reported_hypothesis_not_independently_verified"
    assert not log["motion_allowances_or_evidence_changed"]
    assert report["entries"][0]["last_explicit_revision"] == log
    current["geometry"]["half_extents_m"][0] = .4; context[0]["reason"] = "changed outside"
    assert inventory.summary()["revision_log"][-1] == log


def test_two_view_claim_requires_both_current_views_even_if_previous_wrist_tracked():
    inventory, obs, context = plant(); context.append(revision())
    current = changed_entry(); current["wrist_roi_uv"] = None; before = fingerprint(inventory)
    with pytest.raises(ValueError, match="both current declared RGB views"):
        inventory.configure(obs, [current])
    assert fingerprint(inventory) == before
    context[0]["evidence_kind"] = "measured_motion_conflict"
    inventory.configure(obs, [current])
    assert inventory.summary()["entries"][0]["views"]["wrist"]["current_roi_uv"] is None


@pytest.mark.parametrize("entries,records", [
    ([], [revision()]),
    ([entry(geometry=False)], [revision()]),
    ([entry()], [revision()]),
    ([entry("new_object")], [revision("new_object")]),
    ([changed_entry()], [revision("unknown")]),
    ([changed_entry()], [revision(), revision()]),
    ([changed_entry()], [{"id":"object_a", "reason":" ", "evidence_kind":"current_two_view"}]),
    ([changed_entry()], [{"id":"object_a", "reason":"x", "evidence_kind":"certified_truth"}]),
    ([changed_entry()], [{"id":"object_a", "reason":"x", "evidence_kind":[]}]),
    ([changed_entry()], [{"id":"object_a", "reason":"x", "evidence_kind":"current_two_view", "extra":1}]),
])
def test_orphan_stale_duplicate_or_invalid_record_is_transactional(entries, records):
    inventory, obs, context = plant(); context.extend(records); before = fingerprint(inventory)
    # Even discontinuous RGB/robot input cannot change continuity bookkeeping
    # until the entire revision batch has passed validation.
    different = observation((.11, -.1, 1.), blank=True)
    # Stale/unchanged cases need current pose so geometry really is unchanged.
    if entries and entries[0].get("geometry") == entry()["geometry"]: different = obs
    with pytest.raises(ValueError): inventory.configure(different, entries)
    assert fingerprint(inventory) == before


def test_bad_later_revision_does_not_apply_earlier_valid_revision():
    inventory, obs, context = plant()
    inventory.configure(obs, [entry("object_b")])
    context.extend([revision(), revision("object_b", reason="")]); before = fingerprint(inventory)
    second = changed_entry(); second["id"] = "object_b"
    with pytest.raises(ValueError): inventory.configure(obs, [changed_entry(), second])
    assert fingerprint(inventory) == before


def test_world_rebase_is_not_revision_and_subtolerance_changes_do_not_accumulate():
    inventory, before, context = plant(); after = observation((.12, -.08, .99))
    inventory.observe(before, after)
    old = copy.deepcopy(inventory.objects["object_a"]["world_geometry"])
    declaration = entry(center=(.08, .03, -.09))
    inventory.configure(after, [declaration])
    assert inventory.objects["object_a"]["world_geometry"] == old
    for _ in range(20):
        small = copy.deepcopy(declaration); small["geometry"]["center_offset_m"][0] += .9e-9
        inventory.configure(after, [small])
    assert inventory.objects["object_a"]["world_geometry"] == old
    assert inventory.objects["object_a"]["geometry_revision_count"] == 1
    assert len(inventory.revision_log) == 1
    small["geometry"]["center_offset_m"][0] += .2e-9
    with pytest.raises(ValueError, match="Explicit scene revision required"):
        inventory.configure(after, [small])


def test_metric_age_distinct_from_seen_age_and_null_or_omission_never_refreshes_it():
    inventory, obs, context = plant()
    for _ in range(3): inventory.observe(obs, obs)
    inventory.configure(obs, [entry(geometry=False)])
    row = inventory.summary()["entries"][0]
    assert row["estimated_at_observation"] == 0 and row["metric_age_observations"] == 3
    assert row["last_seen_observation"] == 3 and row["metric_age_updates"] == 1
    inventory.configure(obs, []); row = inventory.summary()["entries"][0]
    assert row["metric_age_updates"] == 2 and row["metric_age_observations"] == 3
    inventory.configure(obs, [entry()]); row = inventory.summary()["entries"][0]
    assert row["estimated_at_observation"] == 3 and row["metric_age_observations"] == 0
    assert row["metric_changed_at_observation"] == 0 and len(inventory.revision_log) == 1


def test_unknown_geometry_has_no_fabricated_age_and_later_initial_estimate_needs_no_revision():
    inventory = VisualSceneInventoryV31(); obs = observation()
    inventory.configure(obs, [entry(geometry=False)])
    assert inventory.summary()["entries"][0]["metric_age_observations"] is None
    inventory.observe(obs, obs); inventory.configure(obs, [entry()])
    assert inventory.summary()["entries"][0]["estimated_at_observation"] == 1
    assert inventory.revision_log[0]["kind"] == "initial_metric_declaration"


def test_complete_log_survives_bounded_legacy_history_and_materialization_is_isolated():
    inventory, obs, context = plant(); context.append(revision())
    for step in range(7):
        declaration = entry(center=(.11+.005*step, .05, -.1))
        inventory.configure(obs, [declaration])
    assert len(inventory.revision_log) == 8
    assert len(inventory.objects["object_a"]["geometry_revisions"]) == 4
    material = inventory.materialize(obs["observation/state"], "object_a")
    material["objects"][0]["last_explicit_revision"]["revision"]["reason"] = "external edit"
    assert inventory.revision_log[-1]["revision"]["reason"] != "external edit"


def test_old_inventory_default_behavior_is_unchanged():
    legacy = VisualSceneInventoryV16(); obs = observation()
    legacy.configure(obs, [entry()]); legacy.configure(obs, [changed_entry()])
    assert legacy.objects["object_a"]["world_geometry"]["center_xyz_m"] == pytest.approx([.22, -.05, .9])
    assert "revision_log" not in legacy.summary()


def test_revision_provider_is_called_once_and_throwing_provider_changes_nothing():
    inventory, obs, context = plant(); count = []
    def provider(): count.append(1); return context
    inventory.revision_context_provider = provider
    inventory.configure(obs, []); assert count == [1]
    before = fingerprint(inventory)
    def failed(): raise RuntimeError("no current response")
    inventory.revision_context_provider = failed
    with pytest.raises(RuntimeError): inventory.configure(obs, [])
    assert fingerprint(inventory) == before


def test_pure_prevalidator_uses_explicit_records_never_provider_or_image_decode():
    inventory, obs, context = plant()
    inventory.revision_context_provider = lambda: pytest.fail("previous-response provider must not be consulted")
    state_only = {"observation/state": obs["observation/state"]}
    before = fingerprint(inventory)
    assert inventory.validate_revision_update(state_only, [changed_entry()], [revision()]) is True
    assert inventory.validate_revision_update(state_only, [entry()], []) is True
    assert inventory.validate_revision_update(state_only, [entry(geometry=False)], []) is True
    with pytest.raises(ValueError, match="Explicit scene revision required"):
        inventory.validate_revision_update(state_only, [changed_entry()], [])
    with pytest.raises(ValueError, match="orphaned, stale, unchanged"):
        inventory.validate_revision_update(state_only, [entry()], [revision()])
    assert fingerprint(inventory) == before


def test_pure_prevalidation_rebases_old_world_with_current_measured_state():
    inventory, obs, context = plant(); before = fingerprint(inventory)
    moved = {"observation/state": [.12, -.08, .99, 0., 0., 0., .04, -.04]}
    assert inventory.validate_revision_update(moved, [entry(center=(.08,.03,-.09))], [])
    with pytest.raises(ValueError, match="Explicit scene revision required"):
        inventory.validate_revision_update(moved, [entry()], [])
    assert fingerprint(inventory) == before
