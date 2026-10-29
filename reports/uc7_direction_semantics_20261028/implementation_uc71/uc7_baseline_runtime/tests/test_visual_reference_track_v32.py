"""Exact part registration while preserving bounded identity/appearance checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle, _sample
from gpt_field.visual_reference_track_v32 import ExactDescriptorPartTrackV32
from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21
from test_visual_monitor_v5 import textured, transform


@pytest.mark.parametrize("point", [[55., 55.], [55.2, 55.3], [55.5, 55.5], [55.75, 54.8]])
def test_exact_integer_and_fractional_descriptor_has_no_parabolic_motion(point):
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    point = np.array(point)
    legacy = LocalFeatureBundle._match(track, pixels, pixels.copy(), point)
    assert legacy is not None
    assert np.linalg.norm(legacy[0]-point) > 1e-5
    result = track._match(pixels, pixels.copy(), point)
    assert np.array_equal(result[0], point) and result[1] == 1.


def test_repeated_exact_image_keeps_geometry_and_initial_identity_bank():
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    initial = track.initial_points.copy()
    templates = [x.copy() for x in track.initial_templates]
    for _ in range(30):
        assert track.track(pixels.copy())
        assert np.array_equal(track.position, track.seed)
        assert np.array_equal(track.matrix, np.eye(2))
        assert np.array_equal(track.translation, np.zeros(2))
    assert np.array_equal(track.initial_points, initial)
    assert all(np.array_equal(a, b) for a, b in zip(templates, track.initial_templates))
    assert len(track.active) == len(initial)
    assert not track.summary()["exact_descriptor_matching"]["whole_object_stationarity_certified"]


@pytest.mark.parametrize("shift", [.2, .5, 1., 1.4, 2.5])
def test_actual_changed_descriptor_translation_uses_unchanged_legacy_match(shift):
    pixels = textured()
    moved = transform(pixels, (shift, .2))
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    point = np.array([55., 55.])
    assert not np.array_equal(_sample(pixels, point+track.offsets), _sample(moved, point+track.offsets))
    legacy = LocalFeatureBundle._match(track, pixels, moved, point)
    actual = track._match(pixels, moved, point)
    assert legacy is not None and actual is not None
    assert np.array_equal(actual[0], legacy[0]) and actual[1] == legacy[1]
    assert np.linalg.norm(actual[0]-point) > .05


def test_small_nonzero_photometric_change_is_not_rounded_into_exact_equality():
    pixels = textured()
    changed = pixels+1e-10
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    point = np.array([55.2, 55.3])
    legacy = LocalFeatureBundle._match(track, pixels, changed, point)
    actual = track._match(pixels, changed, point)
    assert not np.array_equal(_sample(pixels, point+track.offsets), _sample(changed, point+track.offsets))
    assert np.array_equal(actual[0], legacy[0]) and actual[1] == legacy[1]


@pytest.mark.parametrize("case", ["low_texture", "repeated_texture", "occlusion"])
def test_exactness_never_rescues_inherited_texture_ambiguity_or_occlusion_failure(case):
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    if case == "low_texture":
        first = second = np.full_like(pixels, 100.)
    elif case == "repeated_texture":
        first = second = np.tile(pixels[:4, :4], (28, 28))
    else:
        first, second = pixels, np.full_like(pixels, 100.)
    assert LocalFeatureBundle._match(track, first, second, np.array([55., 55.])) is None
    assert track._match(first, second, np.array([55., 55.])) is None


def test_success_outside_rounding_plus_parabola_bound_is_never_overridden(monkeypatch):
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    point = np.array([55., 55.])
    distant = (point+np.array([1.0001, 0.]), .93)
    monkeypatch.setattr(LocalFeatureBundle, "_match", lambda *a, **kw: distant)
    assert track._match(pixels, pixels.copy(), point) is distant


def test_failed_identity_cannot_be_resurrected_by_restoring_exact_pixels():
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [.5, .5], "neighbor")
    assert not track.track(np.full_like(pixels, 100.))
    assert not track.track(pixels)
    assert not track.reliable




def test_moving_part_retains_seed_verified_appearance_and_never_creates_identities():
    pixels = textured()
    track = ExactDescriptorPartTrackV32(pixels, [55/111, 55/111], "tool")
    points = track.initial_points.copy()
    templates = [x.copy() for x in track.seed_templates]
    for frame in range(1, 13):
        current = transform(pixels, (.4*frame, -.2*frame), .004*frame, 1+.002*frame)
        assert track.track(current), track.failure_reason
        assert np.linalg.norm(track.position-[55+.4*frame, 55-.2*frame]) < .7
        assert set(track.active).issubset(range(len(points)))
        assert len(track.appearance_keyframes) <= track.maximum_keyframes
    assert track.keyframe_updates >= 1
    assert np.array_equal(track.initial_points, points)
    assert all(np.array_equal(a,b) for a,b in zip(templates,track.seed_templates))
    assert not track.summary()['exact_descriptor_matching']['independent_robot_motion_or_semantic_evidence']
    assert not track.summary()['exact_descriptor_matching']['physical_budgets_modified']


def archive():
    path = Path(__file__).parents[1]/'runs/visual_field_v31_zero_shot_001/trials/bowl_original_zero_hybrid_v31/trace.jsonl'
    if not path.exists(): pytest.skip('Actual V31 saved RGB unavailable')
    observations, predictions = {}, {}
    for line in path.open():
        row=json.loads(line)
        if row['event']=='reset': observations[-1]=row['observation']
        if row['event']=='executed': observations[row['step']]=row['observation']
        if row['event']=='action' and row['diagnostics'].get('prediction') is not None:
            predictions[row['step']]=row['diagnostics']['prediction']
    return observations,predictions


@pytest.mark.parametrize('first,last', [(0,29),(31,42)])
def test_actual_stationary_target_descriptor_stays_fixed_during_robot_motion(first,last):
    observations,predictions=archive()
    seed=predictions[first]['visual_references']['external']['seed']
    base=_gray(decode_image(observations[first-1]['observation/image']))
    args=(base,seed['target_anchor_uv'],'actual_target')
    kwargs=dict(roi_uv=seed['target_roi_uv'],support_radius_px=16)
    legacy=PartFeatureTrackV21(*args,**kwargs)
    fixed=ExactDescriptorPartTrackV32(*args,**kwargs)
    original=fixed.initial_points.copy()
    for step in range(first,last+1):
        current=_gray(decode_image(observations[step]['observation/image']))
        # This is observed exact sample equality, not object-class stationarity.
        if first == 31:
            assert all(np.array_equal(_sample(base,p+fixed.offsets),_sample(current,p+fixed.offsets)) for p in original)
        assert legacy.track(current) and fixed.track(current)
        if first == 31:
            assert np.array_equal(fixed.position,fixed.seed)
            assert np.array_equal(fixed.matrix,np.eye(2))
            assert np.array_equal(fixed.translation,np.zeros(2))
        else:
            # First block has a few tiny real raster changes; preserve them.
            assert np.linalg.norm(fixed.position-fixed.seed) < .02
        assert fixed.last_inliers==len(original)
    assert np.linalg.norm(legacy.position-legacy.seed)>.15
    assert np.array_equal(fixed.initial_points,original)


def test_actual_tool_does_not_get_frozen_by_stationary_target_correction():
    observations,predictions=archive()
    seed=predictions[0]['visual_references']['external']['seed']
    base=_gray(decode_image(observations[-1]['observation/image']))
    track=ExactDescriptorPartTrackV32(base,seed['moving_point_uv'],'actual_tool',
        roi_uv=seed['moving_roi_uv'],support_radius_px=16)
    for step in range(30):
        assert track.track(_gray(decode_image(observations[step]['observation/image'])))
    assert np.linalg.norm(track.position-track.seed)>.5
    assert track.accepted_image_transitions==30
    assert set(track.active).issubset(range(len(track.initial_points)))


def test_actual_first30_response_improves_without_claiming_enough_observable_energy():
    from gpt_field.visual_field import SupportedLinearModel
    observations,predictions=archive()
    seed=predictions[0]['visual_references']['external']['seed']
    base=_gray(decode_image(observations[-1]['observation/image']))
    counts=[]
    endpoints=[-1,5,8,11,14,17,21,25,28]
    for kind in (PartFeatureTrackV21,ExactDescriptorPartTrackV32):
        tool=kind(base,seed['moving_point_uv'],'tool',roi_uv=seed['moving_roi_uv'],support_radius_px=16)
        target=kind(base,seed['target_anchor_uv'],'target',roi_uv=seed['target_roi_uv'],support_radius_px=16)
        values={-1:tool.position-target.position}
        for step in range(30):
            gray=_gray(decode_image(observations[step]['observation/image']))
            assert tool.track(gray) and target.track(gray)
            values[step]=tool.position-target.position
        model=SupportedLinearModel(residual_limit=2.5,min_samples=3,min_rank=1,max_samples=24,minimum_excitation=.0015)
        error=zero=0.;heldout=0
        for a,b in zip(endpoints,endpoints[1:]):
            delta=np.asarray(observations[b]['observation/state'][:3])-observations[a]['observation/state'][:3]
            flow=values[b]-values[a]
            prediction=model.predict(delta)
            if prediction is not None:
                error+=float(np.sum((flow-prediction)**2));zero+=float(flow@flow);heldout+=1
            model.add(delta,flow)
        counts.append((error,zero,heldout))
    old,new=counts
    assert old[2]==new[2]==5
    assert old[0]>old[1]
    assert new[0]<.75*new[1] and new[0]<old[0]/5
    # The unchanged V31 energy condition would still refuse qualification.
    assert new[1]<.5
