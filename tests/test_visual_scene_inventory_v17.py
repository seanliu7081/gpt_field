"""Partial current support preserves the full scene and requires live texture."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_scene_inventory_v17 import VisualSceneInventoryV17
from test_visual_scene_inventory_v16 import observation, entry


def plant():
    inventory = VisualSceneInventoryV17(); obs = observation()
    inventory.configure(obs, [entry()])
    return inventory, obs


def project(monkeypatch, inventory, full_roi):
    record = inventory.objects['object_a']['views']['external']; track = record['track']
    monkeypatch.setattr(track, 'track', lambda image: True)
    track.matrix = np.eye(2); track.translation = np.zeros(2)
    record['source'] = inventory._corners(np.array(full_roi)*223)
    return record, track


def test_full_view_support_is_separate_from_complete_extent_or_clearance_claim():
    inventory, obs = plant(); inventory.observe(obs, obs)
    row = inventory.summary()['entries'][0]['views']['external']
    assert row['current_visible_roi_uv'] == pytest.approx(row['current_roi_uv'])
    assert not row['partial_current_support'] and row['support_inlier_count'] >= 3
    assert row['support_observed_at'] == 1
    assert not row['support_is_complete_extent_observation']
    assert not row['support_is_clearance_certificate']


def test_partial_visible_support_keeps_full_projection_world_and_identity(monkeypatch):
    inventory, obs = plant(); old_world = copy.deepcopy(inventory.objects['object_a']['world_geometry'])
    record, track = project(monkeypatch, inventory, [.2, .2, 1.2, .6])
    inventory.observe(obs, obs)
    assert record['current_roi_uv'] is None and not record['reliable']  # V16 semantics.
    assert record['partial_current_support']
    assert record['current_visible_roi_uv'] == pytest.approx([.2,.2,1.,.6])
    assert record['projected_roi_uv'] == pytest.approx([.2,.2,1.2,.6])
    assert record['track'] is track and track.reliable
    material = inventory.materialize(obs['observation/state'], 'different')
    assert material['objects'][0]['world_geometry'] == old_world
    assert material['objects'][0]['id'] == 'object_a'
    assert material['nearby_obstacles'][0]['external_roi_uv'] == record['current_visible_roi_uv']
    assert material['nearby_obstacles'][0]['external_full_projected_roi_uv'] == record['projected_roi_uv']
    assert material['partial_external_support_ids'] == ['object_a']
    assert material['stale_external_ids'] == ['object_a']  # Explicit full-box status retained.
    assert not material['visible_support_changes_world_geometry']


@pytest.mark.parametrize('full_roi', [[1.1,.2,1.4,.6], [-.5,.2,-.1,.6], [.2,1.1,.6,1.4], [.99,.2,1.3,.6]])
def test_fully_outside_or_unusable_intersection_never_claims_support(monkeypatch, full_roi):
    inventory, obs = plant(); record, _ = project(monkeypatch, inventory, full_roi)
    world = copy.deepcopy(inventory.objects['object_a']['world_geometry'])
    inventory.observe(obs, obs)
    assert record['current_roi_uv'] is None and record['current_visible_roi_uv'] is None
    assert not record['partial_current_support']
    assert record['support_status'] == 'visible_intersection_unusable'
    assert inventory.objects['object_a']['world_geometry'] == world
    assert inventory.materialize(obs['observation/state'], 'other')['obstacles'][0]['id'] == 'object_a'


def test_real_texture_loss_cannot_keep_partial_support_or_automatically_reseed(monkeypatch):
    inventory, obs = plant(); record, track = project(monkeypatch, inventory, [.2,.2,1.2,.6])
    inventory.observe(obs, obs)
    monkeypatch.undo()
    blank = observation(blank=True)
    inventory.observe(obs, blank)
    assert record['current_visible_roi_uv'] is None and not record['partial_current_support']
    assert record['status'] == 'visual_correspondence_lost'
    assert record['track'] is track and not track.reliable
    assert record['projected_roi_uv'][2] == pytest.approx(1.2)
    # Restoring appearance cannot autonomously replace a lost correspondence.
    inventory.observe(blank, obs)
    assert record['current_visible_roi_uv'] is None and record['track'] is track


def test_fewer_than_three_current_in_frame_descriptors_rejects_partial_support(monkeypatch):
    inventory, obs = plant(); record, track = project(monkeypatch, inventory, [.2,.2,1.2,.6])
    track.points = np.array([[70.,70.],[75.,75.],[230.,75.]])
    track.last_inliers = 3
    inventory.observe(obs, obs)
    assert record['support_inlier_count'] == 2
    assert record['current_visible_roi_uv'] is None
    assert not record['partial_current_support']


def test_omission_target_switch_and_materialization_do_not_reseed_or_shrink(monkeypatch):
    inventory, obs = plant(); record, track = project(monkeypatch, inventory, [-.2,.2,.6,.6])
    inventory.observe(obs, obs)
    before = copy.deepcopy(inventory.summary()['entries'][0])
    inventory.configure(obs, [])
    assert record['track'] is track and record['partial_current_support']
    target = inventory.materialize(obs['observation/state'], 'object_a', allowed_contact='fingers')
    other = inventory.materialize(obs['observation/state'], 'other')
    assert target['objects'] == other['objects']
    assert other['objects'][0]['world_geometry'] == before['world_geometry']
    assert other['objects'][0]['views']['external']['projected_roi_uv'] == before['views']['external']['projected_roi_uv']
    target['objects'][0]['views']['external']['current_visible_roi_uv'][0] = .8
    assert record['current_visible_roi_uv'][0] == 0.


def test_explicit_null_view_and_transition_gap_never_promote_old_support(monkeypatch):
    inventory, obs = plant(); record, _ = project(monkeypatch, inventory, [.2,.2,1.2,.6])
    inventory.observe(obs, obs)
    inventory.configure(obs, [entry(external=False, geometry=False)])
    assert record['current_visible_roi_uv'] is None and not record['partial_current_support']
    wrong = observation((.2,-.1,1.))
    inventory.observe(wrong, wrong)
    wrist = inventory.objects['object_a']['views']['wrist']
    assert wrist['current_visible_roi_uv'] is None
    assert wrist['support_status'] == 'unacknowledged_transition_gap'


def test_archived_v16_002_cabinet_drift_has_real_current_partial_support():
    archive = Path(__file__).parents[1]/'runs/visual_field_v16_zero_shot_002/trials/bowl_original_zero_hybrid_v16/trace.jsonl'
    if not archive.exists():pytest.skip('optional saved RGB archive unavailable')
    inventory = VisualSceneInventoryV17(); obs = None; world_at_seed = None; original = None
    for line in archive.open():
        row = json.loads(line)
        if row['event'] == 'reset': obs = row['observation']
        elif row['event'] == 'action':
            pred = row['diagnostics'].get('prediction')
            if pred:
                inventory.configure(obs, pred['scene_inventory'])
                if row['step'] == 1:
                    world_at_seed = {r['id']:r['world_geometry'] for r in inventory.summary()['entries']}
        elif row['event'] == 'executed':
            inventory.observe(obs, row['observation']); obs = row['observation']
        elif row['event'] == 'feedback' and row['step'] == 5:
            original = row['feedback'];break
    assert original is not None and world_at_seed is not None
    material = inventory.materialize(obs['observation/state'], 'target_black_bowl')
    current = next(r for r in material['objects'] if r['id']=='drawer_cabinet')
    view = current['views']['external']
    old = next(r for r in original['scene_inventory']['entries'] if r['id']=='drawer_cabinet')['views']['external']
    assert view['current_roi_uv'] is None and old['current_roi_uv'] is None
    assert view['partial_current_support'] and view['current_visible_roi_uv'] is not None
    assert (view['projected_roi_uv'][2]-1)*223 == pytest.approx(.4098556206519, abs=2e-6)
    assert view['projected_roi_uv'] == pytest.approx(old['projected_roi_uv'])
    assert view['support_inlier_count'] >= 3
    assert view['support_observed_at'] == inventory.observation_index
    points = np.asarray(view['support_points_px'])
    assert np.all(points-3 >= 0.) and np.all(points+3 <= 223.)
    assert {r['id']:r['world_geometry'] for r in material['objects']} == world_at_seed
    assert len(material['objects']) == 8 and len(material['nearby_obstacles']) == 8
    assert next(r for r in material['nearby_obstacles'] if r['id']=='drawer_cabinet')['external_roi_uv'][2] == 1.
    assert not material['clearance_certified']


@pytest.mark.parametrize('boundary_box', [[0.,.25,.5,.5],[.25,0.,.5,.5],[.25,.25,1.,.5],[.25,.25,.5,1.]])
def test_fresh_boundary_declaration_is_partial_despite_legacy_current_roi(boundary_box):
    inventory=VisualSceneInventoryV17();obs=observation();declaration=entry()
    declaration['external_roi_uv']=boundary_box
    inventory.configure(obs,[declaration])
    view=inventory.summary()['entries'][0]['views']['external']
    assert view['current_roi_uv'] == boundary_box
    assert view['extent_visibility'] == 'boundary_limited'
    assert view['extent_touches_image_boundary']
    assert view['partial_current_support'] and view['current_visible_roi_uv'] is not None


def test_textureless_fresh_boundary_is_incomplete_extent_without_fake_support():
    inventory=VisualSceneInventoryV17();obs=observation(blank=True);declaration=entry()
    declaration['external_roi_uv']=[.25,.25,1.,.5]
    inventory.configure(obs,[declaration])
    view=inventory.summary()['entries'][0]['views']['external']
    assert view['current_roi_uv'] is not None
    assert view['extent_visibility']=='boundary_limited'
    assert not view['partial_current_support'] and view['current_visible_roi_uv'] is None
    assert view['support_inlier_count']==0


def test_quantized_edge_remains_partial_and_tracking_does_not_certify_missing_extent(monkeypatch):
    inventory=VisualSceneInventoryV17();obs=observation();declaration=entry()
    declaration['external_roi_uv']=[.25,.25,1.,.5]
    inventory.configure(obs,[declaration])
    record,track=project(monkeypatch,inventory,[.25,.25,1.,.5])
    track.translation=np.array([.2,0.])
    inventory.observe(obs,obs)
    assert record['current_roi_uv'] is not None and record['boundary_quantization']['applied']
    assert record['partial_current_support'] and record['extent_visibility']=='boundary_limited'
    # A later fit shifts the projected edge inward. The originally image-cut
    # declaration still contains no observation of the missing object extent.
    track.translation=np.array([-1.,0.])
    inventory.observe(obs,obs)
    assert record['current_roi_uv'] is not None and not record['extent_touches_image_boundary']
    assert record['partial_current_support'] and record['extent_visibility']=='boundary_limited'
    assert record['declared_extent_touches_image_boundary']
    revised=entry(geometry=False);revised['external_roi_uv']=[.2,.2,.99,.55]
    world=copy.deepcopy(inventory.objects['object_a']['world_geometry'])
    inventory.configure(obs,[revised])
    assert record['extent_visibility']=='within_image' and not record['partial_current_support']
    assert record['current_visible_roi_uv'] is not None
    assert inventory.objects['object_a']['world_geometry']==world
