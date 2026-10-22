"""Causal bounded appearance support, with real archived RGB regressions."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_monitor_v5 import LocalFeatureBundle, _ncc, _sample
from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21
from test_visual_monitor_v5 import textured, transform


def test_affine_motion_preserves_fixed_geometry_and_immutable_seed_bank():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [55/111, 55/111], "tool")
    points = tracker.initial_points.copy()
    seed_bank = [x.copy() for x in tracker.seed_templates]
    for frame in range(1, 13):
        current = transform(initial, (.4*frame, -.2*frame), .004*frame, 1+.002*frame)
        assert tracker.track(current), tracker.failure_reason
        assert np.linalg.norm(tracker.position-[55+.4*frame, 55-.2*frame]) < .7
        assert set(tracker.active).issubset(range(len(points)))
        assert len(tracker.appearance_keyframes) <= 2
        for anchor in tracker.appearance_keyframes:
            for index, template in anchor["templates"].items():
                assert _ncc(tracker.seed_templates[index], template) >= .6
    assert tracker.keyframe_updates >= 1
    assert np.array_equal(tracker.initial_points, points)
    assert all(np.array_equal(a, b) for a, b in zip(seed_bank, tracker.seed_templates))
    with pytest.raises(ValueError):
        tracker.seed_templates[0][0, 0] = 0


def test_identical_retain_observation_cannot_age_or_renew_appearance_support():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "tool")
    for _ in range(80):
        assert tracker.track(initial.copy())
    assert tracker.accepted_image_transitions == tracker.keyframe_updates == 0
    assert tracker.appearance_keyframes == []


def test_stationary_brightness_change_does_not_create_motion_or_new_anchors():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "claimed_tool")
    for amount in range(1, 10):
        assert tracker.track(initial+amount)
    assert np.linalg.norm(tracker.position-tracker.seed) < .05
    assert tracker.keyframe_updates == 0
    assert tracker.summary()["appearance_maintenance"]["independent_robot_motion_or_semantic_evidence"] is False


@pytest.mark.parametrize("replacement", [lambda image: textured(41),
                                         lambda image: np.full_like(image, 120),
                                         lambda image: transform(image, (15, 0)),
                                         lambda image: transform(image, (0, 0), .3)])
def test_occlusion_distractor_jump_or_rotation_cannot_be_repaired_by_template_refresh(replacement):
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "tool")
    for frame in range(1, 6):
        assert tracker.track(transform(initial, (.4*frame, 0)))
    old_updates = tracker.keyframe_updates
    assert not tracker.track(replacement(initial))
    assert tracker.keyframe_updates == old_updates
    assert not tracker.track(initial)


def test_keyframes_do_not_renew_transitively_or_recover_inactive_points():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [55/111, 55/111], "tool")
    for frame in range(1, 9):
        assert tracker.track(transform(initial, (.4*frame, 0)))
    current = transform(initial, (3.6, 0))
    # A changed original identity must not borrow an appearance anchor. This
    # deliberately corrupts only immutable-test fixtures to isolate the branch:
    # real tracking never changes these descriptors after initialization.
    inactive = int(tracker.active[0])
    tracker.active = tracker.active[1:]
    tracker.points = tracker.points[1:]
    seed = list(tracker.seed_templates)
    seed[inactive] = -seed[inactive]
    tracker.seed_templates = tuple(seed)
    for anchor in tracker.appearance_keyframes:
        anchor["templates"][inactive] = _sample(
            current, (tracker.initial_points[inactive]+tracker.offsets)
            @ tracker.matrix+tracker.translation)
    assert tracker.track(current)
    assert inactive not in tracker.active
    assert inactive not in tracker.last_good_templates


def test_finite_anchor_age_and_displacement_cannot_support_a_changed_seed_forever():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "tool")
    for frame in range(1, 9):
        assert tracker.track(transform(initial, (.4*frame, 0)))
    assert tracker.appearance_keyframes
    # Anchor expiry is independent of visual similarity to a last-good frame.
    # Remove seed agreement in the fixture; an expired adaptive bank alone must
    # never maintain correspondence or create a new keyframe.
    tracker.seed_templates = tuple(-x for x in tracker.seed_templates)
    for anchor in tracker.appearance_keyframes:
        anchor["transition"] = -tracker.maximum_anchor_age
    updates = tracker.keyframe_updates
    assert not tracker.track(transform(initial, (3.6, 0)))
    assert tracker.failure_reason == "bounded_appearance_support_lost"
    assert tracker.keyframe_updates == updates


def test_keyframe_promotion_is_bounded_and_requires_direct_seed_appearance():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "tool")
    tracker.maximum_keyframe_updates = 2
    for frame in range(1, 18):
        assert tracker.track(transform(initial, (.3*frame, 0))), tracker.failure_reason
    assert tracker.keyframe_updates == 2
    assert len(tracker.appearance_keyframes) <= 2


def test_gradual_texture_identity_drift_eventually_loses_bounded_support():
    original, distractor = textured(12), textured(73)
    tracker = PartFeatureTrackV21(original, [.5, .5], "tool")
    # Frame-to-frame changes are small, so simply replacing a descriptor each
    # step could follow this appearance morph forever. Immutable-linked anchors
    # must run out even though adjacent frames remain similar.
    for frame in range(1, 121):
        angle = min(np.pi/2, .024*frame)
        mixed = 120+np.cos(angle)*(original-120)+np.sin(angle)*(distractor-120)
        if not tracker.track(transform(mixed, (.15*frame, 0))):
            break
    assert frame < 120
    assert tracker.failure_reason == "bounded_appearance_support_lost"
    assert tracker.keyframe_updates < tracker.maximum_keyframe_updates
    assert not tracker.track(transform(original, (.15*frame, 0)))


def test_distant_keyframe_cannot_bridge_absent_seed_appearance():
    initial = textured()
    tracker = PartFeatureTrackV21(initial, [.5, .5], "tool")
    for frame in range(1, 9):
        assert tracker.track(transform(initial, (.4*frame, 0)))
    tracker.seed_templates = tuple(-x for x in tracker.seed_templates)
    for anchor in tracker.appearance_keyframes:
        anchor["position"] += [30, 0]
    assert not tracker.track(transform(initial, (3.6, 0)))
    assert tracker.failure_reason == "bounded_appearance_support_lost"


@pytest.fixture(scope="module")
def archive():
    path = Path(__file__).parents[1]/"runs/visual_field_v20_zero_shot_001/trials/bowl_original_zero_hybrid_v20/trace.jsonl"
    if not path.exists():
        pytest.skip("recorded V20 RGB fixture not installed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    observed = {row["step"]: row["observation"] for row in rows if row["event"] == "executed"}
    actions = {row["step"]: row for row in rows if row["event"] == "action"}
    images = {}
    def image(step):
        if step not in images:
            images[step] = np.asarray(decode_image(observed[step]["observation/image"]), float).mean(2)
        return images[step]
    return image, actions


def run_archive(archive, cls, start, end):
    image, actions = archive
    seed = actions[start]["diagnostics"]["prediction"]["visual_references"]["external"]["seed"]
    track = cls(image(start-1), seed["moving_point_uv"], "gripper",
                roi_uv=seed["moving_roi_uv"], support_radius_px=16)
    reports = []
    for step in range(start, end+1):
        if step != start and actions[step]["diagnostics"].get("refreshed"):
            assert track.track(image(step-1))
        accepted = track.track(image(step))
        reports.append((step, accepted, track.summary()))
        if not accepted:
            break
    return track, reports


@pytest.mark.parametrize("start,old_failure,through", [(8, 19, 24), (22, 37, 44), (65, 123, 136)])
def test_causal_recorded_appearance_failures_continue_with_finite_seed_verified_anchors(
        archive, start, old_failure, through):
    legacy, old = run_archive(archive, LocalFeatureBundle, start, through)
    candidate, new = run_archive(archive, PartFeatureTrackV21, start, through)
    assert old[-1][0] == old_failure
    assert legacy.failure_reason == "original_appearance_changed"
    assert new[-1][0] == through and candidate.reliable
    assert candidate.keyframe_updates <= candidate.maximum_keyframe_updates
    assert candidate.last_inliers >= 3
    assert candidate.summary()["semantic_correspondence_certified"] is False
    assert candidate.summary()["appearance_maintenance"]["physical_budgets_modified"] is False


def test_recorded_geometric_inconsistency_at59_remains_rejected(archive):
    for cls in (LocalFeatureBundle, PartFeatureTrackV21):
        track, reports = run_archive(archive, cls, 38, 65)
        assert reports[-1][0] == 59
        assert track.failure_reason == "inconsistent_feature_motion"
        assert not track.track(archive[0](58))
