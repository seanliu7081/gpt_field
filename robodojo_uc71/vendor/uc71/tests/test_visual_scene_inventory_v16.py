"""Persistent scene memory from model declarations, RGB and measured robot state."""
import copy

import numpy as np
import pytest

from gpt_field.observation import image_to_data_url
from gpt_field.visual_scene_inventory_v16 import VisualSceneInventoryV16


def observation(xyz=(.1, -.1, 1.), blank=False):
    rgb = (np.zeros((224, 224, 3), np.uint8) if blank else
           np.random.default_rng(5).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    url = image_to_data_url(rgb, rotate=False)
    return {"observation/image": url, "observation/wrist_image": url,
            "observation/state": [*xyz, 0., 0., 0., .04, -.04]}


def entry(identity="object_a", center=(.1, .05, -.1), *, geometry=True, external=True):
    return {"id": identity, "external_roi_uv": [.25, .25, .5, .5] if external else None,
        "wrist_roi_uv": [.3, .3, .55, .55], "geometry": ({"center_offset_m": list(center),
        "half_extents_m": [.03, .04, .05], "uncertainty_m": .025} if geometry else None)}


def test_static_world_hypothesis_rebases_only_against_actual_measured_eef():
    inventory = VisualSceneInventoryV16()
    before, after = observation(), observation((.12, -.08, .99))
    inventory.configure(before, [entry()])
    inventory.observe(before, after)
    report = inventory.materialize(after["observation/state"], "other")
    world, relative = report["objects"][0]["world_geometry"], report["objects"][0]["current_geometry"]
    assert world["center_xyz_m"] == pytest.approx([.2, -.05, .9])
    assert relative["center_offset_m"] == pytest.approx([.08, .03, -.09])
    assert report["obstacles"][0]["lower_offset_m"] == pytest.approx([.05, -.01, -.14])
    assert inventory.summary()["entries"][0]["geometry_revision_count"] == 1


def test_omission_and_new_goal_never_delete_geometry_or_current_tracked_roi():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry("first"), entry("second", (.2, 0., -.1))])
    inventory.observe(obs, obs)
    inventory.configure(obs, [])
    result = inventory.materialize(obs["observation/state"], "new_target")
    assert {r["id"] for r in result["obstacles"]} == {"first", "second"}
    assert {r["id"] for r in result["nearby_obstacles"]} == {"first", "second"}
    assert all(r["omitted_from_latest_inventory"] for r in result["objects"])
    assert all(r["views"]["external"]["current_roi_uv"] is not None for r in result["objects"])
    assert not result["objects_deleted"] and not result["relevance_filter_applied"]


def test_explicit_current_visual_revision_changes_world_geometry_and_records_prior():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry()])
    revised = entry(center=(.12, .04, -.09))
    revised["geometry"]["half_extents_m"] = [.025, .035, .045]
    inventory.configure(obs, [revised])
    row = inventory.summary()["entries"][0]
    assert row["world_geometry"]["center_xyz_m"] == pytest.approx([.22, -.06, .91])
    assert row["geometry_revision_count"] == 2
    history = row["geometry_revisions"][-1]
    assert history["previous_world_geometry"]["center_xyz_m"] == pytest.approx([.2, -.05, .9])
    assert history["anchored_eef_xyz_m"] == obs["observation/state"][:3]
    assert "unverified" in history["evidence"]


def test_geometry_null_retains_known_world_hypothesis_but_new_object_is_image_only():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry()])
    old = inventory.summary()["entries"][0]["world_geometry"]
    inventory.configure(obs, [entry(geometry=False), entry("unlocalized", geometry=False)])
    result = inventory.materialize(obs["observation/state"], "object_a")
    assert result["target_geometry"]["center_offset_m"] == pytest.approx([.1, .05, -.1])
    assert result["objects"][0]["world_geometry"] == old
    assert result["image_only_ids"] == ["unlocalized"]
    assert result["missing_geometry_ids"] == ["unlocalized"]
    assert "unlocalized" in {r["id"] for r in result["nearby_obstacles"]}
    assert not any(r["id"] == "unlocalized" for r in result["obstacles"])


def test_actual_lost_rgb_tracking_preserves_last_rectangle_and_world_risk():
    inventory = VisualSceneInventoryV16(); obs = observation(); after = observation((.101, -.1, 1.), blank=True)
    inventory.configure(obs, [entry()])
    inventory.observe(obs, after)
    result = inventory.materialize(after["observation/state"], "different")
    record = result["objects"][0]["views"]["external"]
    assert record["current_roi_uv"] is None and not record["reliable"]
    assert record["last_known_roi_uv"] == [.25, .25, .5, .5]
    assert result["stale_external_ids"] == ["object_a"]
    assert result["lost_external_ids"] == ["object_a"]
    assert len(result["obstacles"]) == len(result["nearby_obstacles"]) == 1
    assert record["status"] == "visual_correspondence_lost"


def test_explicit_current_external_occlusion_retains_last_risk_and_wrist_evidence():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry()])
    inventory.configure(obs, [entry(geometry=False, external=False)])
    result = inventory.materialize(obs["observation/state"], "object_a")
    row = result["objects"][0]
    assert row["views"]["external"]["current_roi_uv"] is None
    assert row["views"]["wrist"]["current_roi_uv"] is not None
    assert result["nearby_obstacles"][0]["external_roi_uv"] == [.25, .25, .5, .5]
    assert result["stale_external_ids"] == ["object_a"]


def test_switching_primary_target_only_changes_materialization_role():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry("source"), entry("destination", (.2, .1, -.1))])
    first = inventory.materialize(obs["observation/state"], "source", allowed_contact="fingers")
    second = inventory.materialize(obs["observation/state"], "destination", allowed_contact="none")
    assert first["target_geometry"]["id"] == "source"
    assert [r["id"] for r in first["obstacles"]] == ["destination"]
    assert second["target_geometry"]["id"] == "destination"
    assert [r["id"] for r in second["obstacles"]] == ["source"]
    assert first["objects"] == second["objects"]
    assert len(second["nearby_obstacles"]) == 2


def test_explicit_controlled_payload_exemption_never_removes_inventory_identity():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry("payload"), entry("goal", (.2, 0, -.1))])
    result = inventory.materialize(obs["observation/state"], "goal", controlled_payload_id="payload")
    assert len(result["objects"]) == 2 and not result["obstacles"]
    assert result["controlled_payload_exclusion_requires_caller_observed_evidence"]
    assert inventory.materialize(obs["observation/state"], "goal")["obstacles"][0]["id"] == "payload"


def test_far_offset_or_more_than_legacy_eight_objects_are_not_silently_clipped_or_dropped():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry(f"visible_{i}", (.4, 0., -.1), external=False) for i in range(10)])
    moved = observation((-.7, -.1, 1.))
    inventory.observe(obs, moved)
    result = inventory.materialize(moved["observation/state"], "missing")
    assert len(result["objects"]) == len(result["obstacles"]) == 10
    assert len(result["outside_legacy_center_domain_ids"]) == 10
    assert len(result["outside_legacy_bounds_domain_ids"]) == 10
    assert result["obstacles"][0]["lower_offset_m"][0] == pytest.approx(1.17)
    assert result["obstacles"][0]["upper_offset_m"][0] == pytest.approx(1.23)


def test_materialized_feedback_and_raw_declarations_are_independent_copies():
    inventory = VisualSceneInventoryV16(); obs = observation(); declarations = [entry()]
    original = copy.deepcopy(declarations)
    inventory.configure(obs, declarations)
    baseline = inventory.summary()
    result = inventory.materialize(obs["observation/state"], "other")
    result["objects"][0]["world_geometry"]["center_xyz_m"][0] = 999
    result["obstacles"][0]["lower_offset_m"][0] = 999
    result["nearby_obstacles"][0]["external_roi_uv"][0] = 0.
    assert declarations == original and inventory.summary() == baseline
    assert not baseline["automatic_identity_merge"] and not baseline["automatic_deletion"]
    assert not baseline["clearance_certified"] and baseline["environment_inputs"] == []


def test_invalid_unseen_geometry_revision_cannot_partially_mutate_inventory():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry()]); original = inventory.summary()
    invalid = entry("new"); invalid["external_roi_uv"] = invalid["wrist_roi_uv"] = None
    with pytest.raises(ValueError):
        inventory.configure(obs, [entry(center=(.2, 0, 0)), invalid])
    assert inventory.summary() == original


def test_new_name_is_not_automatically_merged_or_used_to_erase_previous_object():
    inventory = VisualSceneInventoryV16(); obs = observation()
    inventory.configure(obs, [entry("original")])
    inventory.configure(obs, [entry("renamed")])
    result = inventory.materialize(obs["observation/state"], "other")
    assert result["all_remembered_ids"] == ["original", "renamed"]
    assert len(result["obstacles"]) == 2


def test_transition_gap_does_not_fabricate_current_tracking_or_world_update():
    inventory = VisualSceneInventoryV16(); before = observation(); after = observation((.11, -.1, 1.))
    inventory.configure(before, [entry()]); inventory.observe(before, after)
    report = inventory.observe(before, after)
    assert report["status"] == "unacknowledged_transition_gap"
    assert report["observation_index"] == 1
    assert report["entries"][0]["world_geometry"]["center_xyz_m"] == pytest.approx([.2, -.05, .9])


@pytest.mark.parametrize("edge,excess_px,available", [
    ("right", .02798404235793, True),
    ("right", .249, True),
    ("right", .251, False),
    ("left", .249, True),
    ("left", .251, False),
    ("top", .249, True),
    ("top", .251, False),
    ("bottom", .249, True),
    ("bottom", .251, False),
])
def test_successfully_tracked_boundary_has_only_quarter_pixel_coordinate_tolerance(
        monkeypatch, edge, excess_px, available):
    inventory = VisualSceneInventoryV16(); obs = observation()
    declaration = entry()
    declaration["external_roi_uv"] = [.25, .25, .75, .75]
    inventory.configure(obs, [declaration])
    old_world = copy.deepcopy(inventory.objects['object_a']['world_geometry'])
    record = inventory.objects['object_a']['views']['external']
    track = record['track']
    # Isolate the ROI-boundary decision from optical-flow numerical variation.
    # Real recorded RGB correspondence is exercised by the archive test below.
    monkeypatch.setattr(track, 'track', lambda image: True)
    track.matrix = np.eye(2); track.translation = np.zeros(2)
    box = np.array([.25, .25, .75, .75])*223
    component = {"left": 0, "top": 1, "right": 2, "bottom": 3}[edge]
    box[component] = -excess_px if component < 2 else 223+excess_px
    record['source'] = inventory._corners(box)
    inventory.observe(obs, obs)
    view = inventory.summary()['entries'][0]['views']['external']
    assert (view['current_roi_uv'] is not None) == available
    assert view['reliable'] == available
    assert np.allclose(view['projected_roi_uv'], box/223)
    assert inventory.objects['object_a']['world_geometry'] == old_world
    material = inventory.materialize(obs['observation/state'], 'different')
    assert material['obstacles'][0]['id'] == 'object_a'
    if available:
        evidence = view['boundary_quantization']
        assert evidence['maximum_excess_px'] == pytest.approx(excess_px)
        assert evidence['full_projected_extent_preserved']
        assert not evidence['world_geometry_changed'] and not evidence['clearance_certified']
        assert view['current_roi_uv'][component] == (0. if component < 2 else 1.)
    else:
        assert view['status'] == 'whole_rectangle_not_fully_in_view'
        assert view['boundary_quantization'] is None
        assert view['last_known_roi_uv'] == declaration['external_roi_uv']


def test_real_v16_first_transition_preserves_cabinet_support_and_world_geometry():
    import json
    from pathlib import Path
    archive = Path(__file__).parents[1]/'runs/visual_field_v16_zero_shot_001/trials/bowl_original_zero_hybrid_v16/trace.jsonl'
    if not archive.exists():
        pytest.skip('local read-only archive unavailable')
    reset = decision = executed = original_feedback = None
    for line in archive.open():
        row = json.loads(line)
        if row['event'] == 'reset': reset = row['observation']
        elif row.get('step') == 0 and row['event'] == 'action': decision = row['diagnostics']
        elif row.get('step') == 0 and row['event'] == 'executed': executed = row['observation']
        elif row.get('step') == 0 and row['event'] == 'feedback':
            original_feedback = row['feedback']; break
    assert reset is not None and executed is not None and original_feedback is not None
    inventory = VisualSceneInventoryV16()
    inventory.configure(reset, decision['prediction']['scene_inventory'])
    before = inventory.materialize(reset['observation/state'], 'black_bowl')
    inventory.observe(reset, executed)
    after = inventory.materialize(executed['observation/state'], 'black_bowl')
    old = next(row for row in original_feedback['scene_inventory']['entries'] if row['id'] == 'cabinet')
    current = next(row for row in after['objects'] if row['id'] == 'cabinet')
    view = current['views']['external']
    assert old['views']['external']['current_roi_uv'] is None
    assert view['current_roi_uv'] is not None and view['reliable']
    assert view['boundary_quantization']['maximum_excess_px'] == pytest.approx(.02798404235793, abs=1e-7)
    assert view['projected_roi_uv'] == pytest.approx(old['views']['external']['projected_out_of_view_roi_uv'])
    assert view['projected_roi_uv'][2] > 1. and view['current_roi_uv'][2] == 1.
    assert [r['id'] for r in before['objects']] == [r['id'] for r in after['objects']]
    assert [r['world_geometry'] for r in before['objects']] == [r['world_geometry'] for r in after['objects']]
    # The larger table overrun is not treated as quantization or current support.
    table = next(row for row in after['objects'] if row['id'] == 'table')['views']['external']
    assert table['current_roi_uv'] is None and table['boundary_quantization'] is None
    # Re-evaluate ONLY the pure coverage gate with the exact saved geometry,
    # robot state, projected guard and original 2mm downward proposal. This is
    # not a simulator rollout or a new execution/clearance certification.
    from types import SimpleNamespace
    from gpt_field.visual_scene_guard_v16 import SceneCoverageGuardV16
    saved = original_feedback['field']
    guard_saved = original_feedback['translation_guard']
    projected = SimpleNamespace(neighbors=guard_saved['neighbors'], scale=np.array([223, 223]),
                                tool_box=np.array(guard_saved['tool_box_px']))
    field = SimpleNamespace(origin=np.array(saved['origin_xyz']), grid=SimpleNamespace(half_width_m=.08),
                            region_geometry=saved['inflated_regions'], translation_guard=projected,
                            target_configuration=saved['target_contact_geometry'])
    after.update(applied_metric_ids=[r['id'] for r in saved['inflated_regions']],
                 applied_image_ids=[r['id'] for r in guard_saved['neighbors']], geometrically_disjoint_ids=[])
    old_material = copy.deepcopy(after)
    old_material['objects'] = original_feedback['scene_inventory']['entries']
    checker = SceneCoverageGuardV16()
    options = dict(field=field, prediction=decision['prediction'])
    command = np.array(original_feedback['action'][:3])*.05
    old_result = checker.assess(executed['observation/state'], command,
                               materialization=old_material, **options)
    assert old_result['reason'] == 'scene_local_metric_region_lacks_current_image_support'
    assert old_result['diagnostics']['coverage_issues'] == [
        {'id': 'cabinet', 'reason': 'scene_local_metric_region_lacks_current_image_support'}]
    repaired = checker.assess(executed['observation/state'], command, materialization=after, **options)
    assert repaired['accepted'] and not repaired['diagnostics']['clearance_certified']


def test_boundary_tolerance_is_absolute_and_does_not_accumulate_per_frame(monkeypatch):
    inventory = VisualSceneInventoryV16(); obs = observation()
    declaration = entry(); declaration['external_roi_uv'] = [.25, .25, 1., .75]
    inventory.configure(obs, [declaration])
    record = inventory.objects['object_a']['views']['external']; track = record['track']
    monkeypatch.setattr(track, 'track', lambda image: True)
    source = record['source'].copy()
    track.translation = np.array([.2, 0.])
    inventory.observe(obs, obs)
    assert record['current_roi_uv'] is not None
    track.translation = np.array([.4, 0.])
    inventory.observe(obs, obs)
    assert record['current_roi_uv'] is None
    assert record['projected_roi_uv'][2] == pytest.approx(1.+.4/223)
    assert np.array_equal(record['source'], source)
